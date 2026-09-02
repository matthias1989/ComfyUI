"""Seam = the HAIR-MASK BOUNDARY (where hair faces meet skin faces) above the neck floor = the actual
hairline, a clean line by definition (no scattered creases). For long hair the back of the head is
COVERED by hair (no hair<->skin edge there), so the above-floor boundary is naturally the FRONT hairline
+ sides + under-ear. Env: HB_MESH (npz), HB_OUT (loop npz)."""
import numpy as np, os
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _ccomp

m = np.load(os.environ['HB_MESH'], allow_pickle=True)
co = m['co']; fv = m['fv']; nv = int(m['nv']); hair = m['hair'].astype(bool)
znv = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)

def neck_floor():
    nb = 60
    w = np.array([float(np.ptp(co[(znv >= i / nb) & (znv < (i + 1) / nb), 0]))
                  if ((znv >= i / nb) & (znv < (i + 1) / nb)).sum() > 20 else 0.0 for i in range(nb)])
    ws = w.copy()
    for i in range(1, nb - 1):
        ws[i] = (w[i - 1] + w[i] + w[i + 1]) / 3.0
    arm = int(np.argmax(ws)); thr = 0.3 * ws[arm]; i = arm
    while i < nb - 1 and ws[i + 1] >= thr:
        i += 1
    return (i + 1 + 0.5) / nb
floor = neck_floor()

e2f = defaultdict(list)
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)

# region filter: the largest non-hair component = the BODY/skin. Keep boundary edges whose non-hair face
# borders the body, NOT internal HOLES in the hair mask (which make messy blobs, e.g. charA's crown).
_rr = []; _cc = []
for (a, b), fs in e2f.items():
    if len(fs) == 2 and (not hair[fs[0]]) and (not hair[fs[1]]):
        _rr.append(fs[0]); _cc.append(fs[1])
_G = sp.csr_matrix((np.ones(len(_rr)), (_rr, _cc)), shape=(len(fv), len(fv))); _G = _G + _G.T
_nc, _lab = _ccomp(_G, directed=False)
_bodyc = int(np.bincount(_lab[~hair]).argmax())

# Full hair-mask boundary that borders the body/skin (NO floor cut, so the front-side jaw/sideburn line
# below the neck floor IS included). Internal holes are excluded by the body-component test.
bnd = []
for (u, w), fs in e2f.items():
    if len(fs) == 2 and (bool(hair[fs[0]]) != bool(hair[fs[1]])):
        _nh = fs[0] if not hair[fs[0]] else fs[1]
        if _lab[_nh] == _bodyc:
            bnd.append((u, w))
print('neck floor=%.3f  hair-mask boundary edges (body-bordering): %d' % (floor, len(bnd)))

# remove everything BEHIND the front line: drop a boundary vert if a vert in the SAME TIGHT X/Z cell is
# more forward (lower Y) by more than a small margin. Tight cell so the front line is never self-suppressed.
if int(os.environ.get('HB_RULE', '1')) and bnd:
    head_w = float(np.ptp(co[znv >= floor, 0])); head_d = float(np.ptp(co[znv >= floor, 1]))
    cell = float(os.environ.get('HB_CELL', '0.05')) * head_w
    margin = float(os.environ.get('HB_MARGIN', '0.10')) * head_d
    bverts = np.array(sorted(set([u for u, w in bnd] + [w for u, w in bnd])), dtype=np.int64)
    gx = np.floor(co[bverts, 0] / cell).astype(np.int64); gz = np.floor(co[bverts, 2] / cell).astype(np.int64)
    _y = co[bverts, 1]
    cmin = {}
    for i in range(len(bverts)):
        k = (int(gx[i]), int(gz[i]))
        if _y[i] < cmin.get(k, 1e9):
            cmin[k] = float(_y[i])
    keep = np.array([_y[i] <= cmin[(int(gx[i]), int(gz[i]))] + margin for i in range(len(bverts))])
    # USER: do NOT filter around the MIDDLE in Y -- the side-of-neck hairline lives there and the front-most
    # test wrongly drops it as "behind" (the bare front is more forward). Keep the central-Y band regardless.
    _hy0 = float(co[znv >= floor, 1].min()); _hy1 = float(co[znv >= floor, 1].max())
    _ylo = _hy0 + float(os.environ.get('HB_MIDY_LO', '0.25')) * (_hy1 - _hy0)
    _yhi = _hy0 + float(os.environ.get('HB_MIDY_HI', '0.75')) * (_hy1 - _hy0)
    _midband = (_y > _ylo) & (_y < _yhi)
    keep = keep | _midband
    print('  mid-Y exempt [%.3f,%.3f]: +%d verts kept there' % (_ylo, _yhi, int(_midband.sum())))
    survive = set(int(v) for v in bverts[keep].tolist())
    bnd = [(u, w) for u, w in bnd if u in survive and w in survive]
    print('  remove-behind (cell=%.4f margin=%.4f): kept %d/%d -> %d edges' % (cell, margin, int(keep.sum()), len(bverts), len(bnd)))

# ADD the hair-neck GROOVE (the red line) in the mid-Y region: creases that run along the hair edge (within
# a few rings of the boundary) AND inside the mid-Y band. This brings that red line into the seam, only in
# this region -- no strand interior (it must hug the boundary), front untouched.
if os.environ.get('HB_CRE') and bnd:
    cre = np.load(os.environ['HB_CRE'])['edges']
    if len(cre):
        vadj = defaultdict(set)
        for f in fv:
            vs = [int(a) for a in f]; k = len(vs)
            for a in range(k):
                vadj[vs[a]].add(vs[(a + 1) % k]); vadj[vs[(a + 1) % k]].add(vs[a])
        near = set(int(v) for v in (set([u for u, w in bnd] + [w for u, w in bnd])))
        for _ in range(int(os.environ.get('HB_GROOVE_RINGS', '3'))):
            ad = set()
            for v in near:
                ad |= vadj[v]
            near |= ad
        _hy0 = float(co[znv >= floor, 1].min()); _hy1 = float(co[znv >= floor, 1].max())
        _ylo = _hy0 + float(os.environ.get('HB_MIDY_LO', '0.25')) * (_hy1 - _hy0)
        _yhi = _hy0 + float(os.environ.get('HB_MIDY_HI', '0.75')) * (_hy1 - _hy0)
        extra = [(int(u), int(w)) for (u, w) in cre
                 if int(u) in near and int(w) in near
                 and _ylo < co[int(u), 1] < _yhi and _ylo < co[int(w), 1] < _yhi]
        bnd += extra
        print('  +groove creases (mid-Y, near-boundary): %d' % len(extra))

np.savez(os.environ['HB_OUT'], loop=np.array(bnd, dtype=np.int64), anchors=np.zeros((0,), dtype=np.int64))
