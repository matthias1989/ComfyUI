"""Skeleton-free hand replacement on raw vertex/face arrays. BLAS-FREE numpy only (no dot/matmul/linalg)
so it runs both in the ComfyUI main env (the node) and the bpy worker (test harness).
Public: replace_hands(V, F, donorR, donorL) -> (V2, F2, info)."""
import numpy as np


def _norm(v): return float(np.sqrt((np.asarray(v, float) ** 2).sum()))
def _unit(v):
    v = np.asarray(v, float); n = _norm(v); return v / n if n > 1e-12 else v
def _dot(a, b): return float((np.asarray(a, float) * np.asarray(b, float)).sum())
def _perp(a):
    a = _unit(a); e = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(np.cross(a, e))


def rodrigues(a, b):
    """3x3 rotation aligning unit a -> unit b (no matmul; closed-form with outer + skew)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b); s = _norm(v); c = _dot(a, b)
    if s < 1e-8:
        return np.eye(3) if c > 0 else (2 * np.outer(_perp(a), _perp(a)) - np.eye(3))
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return c * np.eye(3) + (1 - c) * np.outer(k, k) + s * K       # outer/elementwise only


def _apply_R(R, P):
    """Apply 3x3 R to Nx3 P without matmul."""
    return np.stack([R[i, 0] * P[:, 0] + R[i, 1] * P[:, 1] + R[i, 2] * P[:, 2] for i in range(3)], axis=1)


def boundary_loops(F):
    """Ordered vertex-index loops for every open boundary (edges used by exactly one face). BLAS-free."""
    from collections import defaultdict
    F = np.asarray(F, np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    uniq, counts = np.unique(np.sort(E, axis=1), axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return []
    adj = defaultdict(list)
    for a, b in bnd.tolist():
        adj[a].append(b); adj[b].append(a)
    seen, loops = set(), []
    for start in list(adj):
        if start in seen:
            continue
        loop, cur, prev = [], start, None
        while cur is not None and cur not in seen:
            loop.append(cur); seen.add(cur)
            nxts = [n for n in adj[cur] if n != prev]
            prev, cur = cur, (nxts[0] if nxts else None)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_holes(V, F):
    """Fan-fill every open boundary loop with a centroid vertex -> closed solid."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    loops = boundary_loops(F)
    if not loops:
        return V, F
    Vn, Fn, off = [V], [F], len(V)
    for loop in loops:
        Vn.append(V[loop].mean(0)[None, :]); ci = off; off += 1
        Fn.append(np.array([[ci, loop[i], loop[(i + 1) % len(loop)]] for i in range(len(loop))], np.int64))
    return np.concatenate(Vn, 0), np.concatenate(Fn, 0)


def bridge_rings(V, loopA, loopB, axis, center):
    """Watertight triangle strip bridging ring loopA to ring loopB. Both are ordered by angle around
    `axis` about `center`, then stitched with a two-pointer angular merge that advances whichever ring's
    next vertex has the smaller angle. This covers EVERY edge of BOTH rings exactly once, so it stays
    watertight even when the rings have very different vertex counts (e.g. 17-vert stump vs 44-vert cuff)."""
    import math
    ax = _unit(axis)
    ref = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(ref - _dot(ref, ax) * ax); w = np.cross(ax, u)
    def ang(i):
        p = V[i] - center
        return math.atan2(_dot(p, w), _dot(p, u)) % (2 * math.pi)
    A = sorted(loopA, key=ang); B = sorted(loopB, key=ang)
    na, nb = len(A), len(B)
    if na < 3 or nb < 3:
        return np.zeros((0, 3), np.int64)
    aA = [ang(x) for x in A]; aB = [ang(x) for x in B]
    faces, i, j = [], 0, 0
    while i < na or j < nb:
        if j >= nb or (i < na and aA[i] <= aB[j]):      # advance A: consume edge A[i]->A[i+1]
            faces.append([A[i % na], A[(i + 1) % na], B[j % nb]]); i += 1
        else:                                            # advance B: consume edge B[j]->B[j+1]
            faces.append([A[i % na], B[(j + 1) % nb], B[j % nb]]); j += 1
    return np.array(faces, np.int64) if faces else np.zeros((0, 3), np.int64)


def _cap_loop(V, loop, base):
    """Fan-fill one boundary loop with a centroid vertex. Returns (centroid (1,3), faces using index `base`)."""
    c = np.asarray(V)[loop].mean(0)[None, :]
    f = np.array([[base, loop[k], loop[(k + 1) % len(loop)]] for k in range(len(loop))], np.int64)
    return c, f


def find_arm_axis(V):
    """Return (arm, up, depth) axis indices. depth = thinnest overall; arm = the axis whose far
    extremes are the THINNEST cross-section (hands), distinguishing arms from head/feet (up)."""
    span = V.max(0) - V.min(0)
    depth = int(np.argmin(span))
    others = [i for i in range(3) if i != depth]

    def ext_thick(ax):
        o2 = [i for i in range(3) if i != ax]
        lo, hi = V[:, ax].min(), V[:, ax].max(); rng = hi - lo
        vals = []
        for at_lo in (True, False):
            m = (V[:, ax] < lo + 0.08 * rng) if at_lo else (V[:, ax] > hi - 0.08 * rng)
            if int(m.sum()) < 5: vals.append(1e9); continue
            sub = V[m][:, o2]; cen = sub.mean(0)
            vals.append(float(np.sqrt(((sub - cen) ** 2).sum(1)).mean()))
        return min(vals)
    arm = min(others, key=ext_thick)
    up = [i for i in others if i != arm][0]
    return arm, up, depth


def find_wrist(V, arm, sign):
    """sign=-1 (min side) / +1 (max side). Returns (wrist_coord, wrist_centre(3,), radius).
    Resolution-robust (works on the ~25k raw mesh AND the ~150k remeshed one): coarse slices, low
    min-verts, median radius, dynamic torso cutoff. Wrist = deepest dip after the hand peak, before the body."""
    A = V[:, arm]; o2 = [i for i in range(3) if i != arm]
    tip = A.min() if sign < 0 else A.max(); full = A.max() - A.min()
    NSL = 30
    edges = np.linspace(tip, tip - sign * full * 0.5, NSL + 1)
    xs, rs = [], []
    for k in range(NSL):
        a, b = sorted((edges[k], edges[k + 1])); m = (A >= a) & (A < b)
        if int(m.sum()) < 3: continue                                    # was 8 -> dropped sparse hand slices on coarse meshes
        sub = V[m][:, o2]; cen = sub.mean(0)
        rs.append(float(np.median(np.sqrt(((sub - cen) ** 2).sum(1)))))  # median = robust on few-vert slices
        xs.append(0.5 * (a + b))
    if len(rs) < 4:                                                       # too sparse -> default to 25% from the tip
        wx = tip - sign * full * 0.25
        m = (A >= wx - full * 0.03) & (A < wx + full * 0.03)
        O = V[m].mean(0) if int(m.sum()) else V.mean(0)
        return float(wx), O, full * 0.03
    sm = [(rs[max(0, i - 1)] + rs[i] + rs[min(len(rs) - 1, i + 1)]) / 3 for i in range(len(rs))]
    # Palm bulge = the THICKEST slice inside the hand, which occupies roughly the first
    # quarter of this half-span scan. Measured absolutely, per side.
    # The old test compared each slice to 1.25 * sm[0] -- i.e. to the FINGERTIP slice, whose
    # thickness varies between a character's own two hands by chance. On a real bake the
    # right hand's tip came in at 0.016 vs the left's 0.013, lifting the bar to 0.0200 so the
    # palm at 0.019 missed it by 0.001; the scan then ran past the arm and latched a "peak"
    # in the torso, putting the wrist at -0.09 instead of -0.33 and amputating the arm.
    peak = max(range(max(3, int(0.25 * len(sm)))), key=lambda i: sm[i])
    torso = next((i for i in range(peak + 1, len(sm)) if sm[i] > 3.0 * sm[peak]), len(sm))   # body onset (radius explodes)
    regain = next((i for i in range(peak + 1, torso) if sm[i] >= sm[peak]), torso)           # forearm regains hand thickness
    end = max(peak + 2, min(regain, torso))
    wx = xs[min(range(peak, end), key=lambda i: sm[i])]
    m = (A >= wx - full * 0.025) & (A < wx + full * 0.025)
    sub = V[m]; O = sub.mean(0); cen = O[o2]
    r = float(np.median(np.sqrt(((sub[:, o2] - cen) ** 2).sum(1))))
    return float(wx), O, r


def replace_hands(V, F, donorR, donorL, seam=0.5, oversize=1.0):
    """Remove webbed hands at both wrists, place the clean donor hands (open cuff) just distal of the
    forearm stumps, and BRIDGE each donor cuff ring to its forearm stump ring -> ONE connected manifold
    (the downstream remesh then just densifies it; no reliance on volumetric fusion).
    donorR/donorL = npz dicts with verts, faces (OPEN cuff), cuff_O, axis, radius.
    seam = gap between stump and donor cuff in body wrist-radii. Returns (V2, F2, info)."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    arm, up, depth = find_arm_axis(V)
    A = V[:, arm]
    wxR, OR, rR = find_wrist(V, arm, -1)
    wxL, OL, rL = find_wrist(V, arm, +1)

    # ROBUSTNESS for x-offset / asymmetric bodies: find_wrist slices a fixed depth from each
    # fingertip, so on the SHORTER arm it overshoots into the torso and reports a wrist that's
    # too far inboard (the donor then grafts in the wrong place, inflating the bbox + shifting
    # the textures). A mis-detect ALWAYS overshoots, and the two wrists should sit ~the same
    # distance from their tips — so clamp an overshooting wrist to the reliable (closer) one.
    full = float(A.max() - A.min()); tipR = float(A.min()); tipL = float(A.max())
    dR = abs(wxR - tipR); dL = abs(wxL - tipL); dmin = min(dR, dL)
    o2 = [i for i in range(3) if i != arm]
    def _wrist_at(wx):
        mm = (A >= wx - full * 0.025) & (A < wx + full * 0.025)
        sub = V[mm]; O = sub.mean(0) if int(mm.sum()) else V.mean(0)
        rr = float(np.median(np.sqrt(((sub[:, o2] - O[o2]) ** 2).sum(1)))) if int(mm.sum()) else full * 0.03
        return O, rr
    if dR > 1.6 * dmin:
        wxR = tipR + dmin; OR, rR = _wrist_at(wxR)
        print(f"[HandGraft] R-wrist overshoot clamped -> {wxR:.3f} (sym with L, d={dmin:.3f})")
    if dL > 1.6 * dmin:
        wxL = tipL - dmin; OL, rL = _wrist_at(wxL)
        print(f"[HandGraft] L-wrist overshoot clamped -> {wxL:.3f} (sym with R, d={dmin:.3f})")

    keep = (A >= wxR) & (A <= wxL)                       # body = everything between the wrists (open stumps)
    newidx = -np.ones(len(V), np.int64); newidx[keep] = np.arange(int(keep.sum()))
    Vb = V[keep]; Fb = newidx[F[keep[F].all(1)]]
    stump_loops = boundary_loops(Fb)                     # all wrist stump rings (Vb indices); a thin-walled
    o2 = [i for i in range(3) if i != arm]               # forearm cut yields concentric rings + tiny holes
    span = A.max() - A.min()

    def _rad(lp):                                        # median cross-section radius of a loop
        sub = Vb[lp][:, o2]; return float(np.median(np.sqrt(((sub - sub.mean(0)) ** 2).sum(1))))

    parts, faces, off = [Vb], [Fb], len(Vb)
    bridges, cap_loops = [], []
    for dz, sign, Ob, rb, wx in ((donorR, -1, OR, rR, wxR), (donorL, +1, OL, rL, wxL)):
        here = [lp for lp in stump_loops if abs(Vb[lp].mean(0)[arm] - wx) < 0.06 * span]
        if not here:
            continue
        outer = max(here, key=_rad)                       # OUTER rim = bridge target; rest are inner/holes
        rim_c = Vb[outer].mean(0)
        dv = np.asarray(dz['verts'], np.float64); dfc = np.asarray(dz['faces'], np.int64)
        Od = np.asarray(dz['cuff_O'], np.float64); ad = np.asarray(dz['axis'], np.float64); rd = float(dz['radius'])
        a_out = np.zeros(3); a_out[arm] = sign                          # outward along the arm
        s = (rb / rd) * oversize
        target = rim_c + a_out * (seam * rb)                            # cuff ring just distal of the real rim
        P = _apply_R(rodrigues(ad, a_out), dv - Od) * s + target
        cuff = boundary_loops(dfc)
        cuff_g = [off + i for i in max(cuff, key=len)] if cuff else []  # donor cuff ring -> global indices
        bridges.append((list(outer), cuff_g, a_out.copy(), rim_c.copy()))
        cap_loops += [list(lp) for lp in here if lp is not outer]       # seal inner ring + tiny cut-holes
        parts.append(P); faces.append(dfc + off); off += len(P)

    V2 = np.concatenate(parts, 0); F2 = np.concatenate(faces, 0)
    extra = []
    for lp in cap_loops:                                                # cap leftover loops (centroid fan)
        c, cf = _cap_loop(V2, lp, len(V2)); V2 = np.concatenate([V2, c], 0); extra.append(cf)
    extra += [bridge_rings(V2, st, cu, ax, ce) for st, cu, ax, ce in bridges if len(st) >= 3 and len(cu) >= 3]
    if extra:
        F2 = np.concatenate([F2] + extra, 0)
    return V2.astype(np.float32), F2.astype(np.int64), dict(arm=arm, up=up, wxR=wxR, wxL=wxL)
