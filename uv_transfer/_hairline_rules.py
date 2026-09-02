"""HAIRLINE by the user's RULES (clean, geometric, character-agnostic) -- replaces path3's crease-graph.

Mesh axes (path3 / _gb_mesh space): X=lateral, Y=depth (face=-Y so FRONT=low y), Z=height.

  R1  candidate points = ALL concave points (crease verts) -- catch everything.
  R2  drop FACE points (face-landmark region; geometric fallback = front-central just under the hairline).
  R3  drop NECK + points clearly IN FRONT and BELOW the face (throat/chest).
  R7  drop EARS (lateral protrusion at head height) -- forbidden.
  R4  drop points BEHIND (in +Y) a kept point at the same (x,z), with margin -> only the FRONT surface
      survives -> no back-of-hair / interior seams (R6).
  R5  draw the lines (connect kept points) -- done by the caller / mark step.

All thresholds are HEAD-relative (head = verts above the neck floor), no per-character constants.
Run standalone (__main__) to render the stages on _gb_mesh.npz for no-bake verification."""
import numpy as np, os
from scipy.spatial import cKDTree

UV = os.path.dirname(os.path.abspath(__file__))


def _face_region_geom(co, head, xc, hw, neck):
    """Geometric stand-in for the face landmarks: front-central band just under the hairline."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    zn = (z - z.min()) / (z.max() - z.min())
    ymid = 0.5 * (y[head].min() + y[head].max())
    return (y < ymid) & (np.abs(x - xc) < 0.55 * hw) & (zn >= neck) & (zn < neck + 0.08)


def hairline_points(co, crease_edges, neck_floor, face_mask=None,
                    ear_frac=0.85, behind_radius=0.06, behind_margin=0.08, verbose=True):
    """Return the kept hairline vertex indices after rules 1-4,7. face_mask: optional per-vertex bool
    (landmark face region, R2); if None, a geometric face region is used."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    zn = (z - z.min()) / (z.max() - z.min())
    head = zn >= neck_floor
    xc = float(x[head].mean()) if head.any() else float(x.mean())
    hw = 0.5 * float(x[head].max() - x[head].min()) if head.any() else 0.1
    ymin_h, ymax_h = float(y[head].min()), float(y[head].max())
    ymid_h = 0.5 * (ymin_h + ymax_h)
    ydepth = max(ymax_h - ymin_h, 1e-6)

    # R1 -- all concave points
    cand = np.unique(crease_edges.ravel()).astype(np.int64)
    keep = np.ones(len(cand), bool)
    cx, cy, cz, czn = x[cand], y[cand], z[cand], zn[cand]
    n1 = len(cand)

    # R2 -- drop FACE (landmark mask if given, else geometric)
    if face_mask is not None:
        face = face_mask[cand]
    else:
        fm = _face_region_geom(co, head, xc, hw, neck_floor)
        face = fm[cand]
    keep &= ~face
    n2 = int(keep.sum())

    # R3 -- drop NECK + everything below the head. The hairline lives on the HEAD (above the neck
    # floor); the front chest/throat below it is never hairline, and any back-below is dropped by R4
    # (behind) anyway. (Tested "front" against the head mid-depth before, which missed the chest because
    # the chest sits behind the face's forward jut -> caught nothing.)
    neck = (czn < neck_floor)
    keep &= ~neck
    n3 = int(keep.sum())

    # R7 -- EARS = FORBIDDEN. The full-head hw is INFLATED by the elf ear-tips, so the old |x|>0.85*hw
    # test left the ears in. Measure the skull/temple width from the FRONT-UPPER forehead (ears sit
    # behind & lateral, never there) and drop ANYTHING lateral to it -> ears gone.
    fhmask = head & (y < ymin_h + 0.40 * ydepth)   # FRONT of the head (face/temple) -- ears sit behind it
    skull_hw = float(np.percentile(np.abs(x[fhmask] - xc), 97)) if fhmask.any() else hw
    ear = np.abs(cx - xc) > skull_hw
    keep &= ~ear
    n7 = int(keep.sum())
    if verbose:
        print('[rules] R7 ears: skull_hw=%.4f (forehead) vs inflated full hw=%.4f; max kept |x|=%.4f'
              % (skull_hw, hw, float(np.abs(cx[keep] - xc).max()) if keep.any() else 0.0))

    # R4 -- drop points BEHIND a kept point at the same (x,z) (front-most survives -> no back/interior)
    xz = np.column_stack([cx, cz])
    T = cKDTree(xz)
    rad = behind_radius * 2.0 * hw
    marg = behind_margin * ydepth
    ki = np.where(keep)[0]
    drop = np.zeros(len(cand), bool)
    for i in ki:
        for j in T.query_ball_point(xz[i], rad):
            if keep[j] and cy[j] < cy[i] - marg:
                drop[i] = True
                break
    keep &= ~drop
    n4 = int(keep.sum())

    if verbose:
        print('[rules] R1 all=%d  R2 -face=%d  R3 -neck=%d  R7 -ears=%d  R4 -behind=%d (final)'
              % (n1, n2, n3, n7, n4))
    return cand[keep]


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    m = np.load(os.path.join(UV, '_gb_mesh.npz')); co = m['co'].astype(float)
    cre = np.load(os.path.join(UV, '_gb_crease.npz'))['edges']
    nf = float(np.load(os.path.join(UV, '_gb_loop.npz'))['neck_floor'])
    hl = hairline_points(co, cre, nf)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]; zn = (z - z.min()) / (z.max() - z.min())
    head = zn >= nf; xc = float(x[head].mean()); ymin_h = float(y[head].min()); ydepth = float(y[head].max() - y[head].min())
    fhmask = head & (y < ymin_h + 0.40 * ydepth)
    skull_hw = float(np.percentile(np.abs(x[fhmask] - xc), 97)) if fhmask.any() else 0.1
    earzone = np.where(head & (np.abs(x - xc) > skull_hw))[0]
    bad = [int(v) for v in hl if abs(x[v] - xc) > skull_hw]
    print('VERIFY EARS: kept points in ear zone = %d (MUST be 0)  | ear-zone verts=%d' % (len(bad), len(earzone)))
    fig, axs = plt.subplots(1, 2, figsize=(15, 9))
    for ax, (t, p) in zip(axs, [('FRONT (x,z)', 0), ('SIDE (y,z) front=left', 1)]):
        s = zn > 0.5
        ax.scatter(co[s, p], co[s, 2], s=0.3, c='#eee', lw=0)
        ax.scatter(co[earzone, p], co[earzone, 2], s=5, c='orange', lw=0, label='EAR zone (forbidden)')
        ax.scatter(co[hl, p], co[hl, 2], s=12, c='red', lw=0, label='kept hairline')
        ax.set_aspect('equal'); ax.set_title(t); ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(UV, '_rules_pts.png'), dpi=95)
    print('saved _rules_pts.png  (%d kept)' % len(hl))
