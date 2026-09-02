"""The user's canonical 7-rule hairline, implemented cleanly and standalone (NO path3 graph tangle).
One step per rule, on the frozen mesh. Inputs already on disk: _frozen/{gb_mesh,crease,facevert,ear}.
Out: _frozen/loop.npz (loop=seam edges) + _frozen/hairline.npz (kept verts)."""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "_frozen", "gb_mesh.npz"))
co = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); nv = len(co); nf = len(fv)
x, y, z = co[:, 0], co[:, 1], co[:, 2]
znf = (z - z.min()) / (z.max() - z.min())                    # per-vertex normalized height (0=feet,1=crown)
E = np.unique(np.sort(np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], 0), 1), axis=0)
el = float(np.median(np.linalg.norm(co[E[:, 0]] - co[E[:, 1]], axis=1)))   # mesh resolution, self-calibrating

# RULE 1 — ALL concave points (low threshold, miss nothing)
ed = np.load(os.path.join(HERE, "_frozen", "crease.npz"))["edges"]
kept = np.zeros(nv, bool); kept[np.unique(ed.reshape(-1))] = True
r1 = int(kept.sum())

# RULE 2 — remove FACE points (MediaPipe region)
face = np.load(os.path.join(HERE, "_frozen", "facevert.npy")).astype(bool)
kept &= ~face; r2 = int(kept.sum())

# RULE 7 — EARS forbidden
e = np.load(os.path.join(HERE, "_frozen", "ear.npz")); earf = e["ear_faces"].astype(bool); neck_floor = float(e["neck_floor"])
earv = np.zeros(nv, bool); earv[fv[earf].reshape(-1)] = True
kept &= ~earv; r7 = int(kept.sum())

# RULE 3 — remove NECK/throat/chest: (a) the front-CENTRAL neck band between the face-bottom and the neck
# floor (the throat); (b) everything below the neck floor (body). Scalable: chin, neck floor, head width
# and the front/back divider are all data-driven; the only ratio is the central column width (head-relative).
chin = float(znf[face].min()) if face.any() else neck_floor          # bottom of the face region
hw = float(np.ptp(x[znf >= neck_floor])); xc = float(np.median(x[znf >= neck_floor]))
ymid = float(np.median(y[znf >= neck_floor]))                         # head front/back divider (front = low y)
neckfrac = float(os.environ.get('HL_NECKFRAC', '0.22'))              # central column width, fraction of head width
throat = (znf >= neck_floor) & (znf < chin) & (np.abs(x - xc) < neckfrac * hw) & (y < ymid)
kept &= ~throat                                                      # 3a: front-central throat
kept &= (znf >= neck_floor); r3 = int(kept.sum())                   # 3b: body below the head
np.save(os.path.join(HERE, "_frozen", "filtered.npy"), face | throat | earv)   # blue viz = all filter regions

# RULE 4 — remove points BEHIND kept ones, per lateral column, with a margin -> front-most surface only
margin = float(os.environ.get('HL_MARGIN', '1e9')) * el       # 1e9 = OFF (see raw head set first), then tune
_hw = float(np.ptp(x[znf >= neck_floor]))                     # head width (data-driven)
_colw = (float(os.environ.get('HL_COLFRAC', '0')) * _hw) or el   # column width: frac of head width, else mesh res
gx = np.floor(x / _colw).astype(np.int64)
colmin = {}
for v in np.where(kept)[0]:
    k = int(gx[v])
    if y[v] < colmin.get(k, 1e18): colmin[k] = float(y[v])
behind = np.array([kept[v] and (y[v] - colmin[int(gx[v])] > margin) for v in range(nv)])
kept &= ~behind; r4 = int(kept.sum())

print('[hairline] rule1 all=%d | -face=%d | -ears=%d | -body=%d | -behind=%d  (margin=%.4f, el=%.4f)' %
      (r1, r2, r7, r3, r4, margin, el))

# RULE 5 — connect kept points into the seam: mesh edges with BOTH endpoints kept
seam = E[kept[E[:, 0]] & kept[E[:, 1]]]
print('[hairline] seam edges: %d' % len(seam))
np.savez(os.path.join(HERE, "_frozen", "loop.npz"), loop=seam, anchors=np.where(kept)[0].astype(np.int64), neck_floor=np.float64(neck_floor))
np.save(os.path.join(HERE, "_frozen", "kept.npy"), kept)
