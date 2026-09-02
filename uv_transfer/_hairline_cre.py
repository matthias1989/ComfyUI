"""Seam = the CREASE line itself (the RED), so the green sits ON the red. Steps:
  (1) keep crease verts NEAR hair (drop bare-skin creases: throat, eye) -- self-calibrating via the mask,
  (2) YOUR RULE: drop a crease vert if a vert in the SAME TIGHT X/Z cell is more forward (lower Y) by more
      than a small margin -> removes everything BEHIND the front line, keeps the line (tight cell so the
      front never suppresses its own width),
  (3) output the surviving CREASE EDGES as the seam (green overlaps red).
Env: HC_MESH (npz), HC_CRE (crease npz, the RED), HC_OUT (loop npz)."""
import numpy as np, os
from collections import defaultdict

m = np.load(os.environ['HC_MESH'], allow_pickle=True)
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

cre = np.load(os.environ['HC_CRE'])['edges']
crev = np.unique(cre)

# vert adjacency + hair-vertex set
vadj = defaultdict(set)
for f in fv:
    vs = [int(a) for a in f]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        vadj[u].add(w); vadj[w].add(u)
hairv = np.zeros(nv, bool)
for fi in np.where(hair)[0]:
    for v in fv[fi]:
        hairv[int(v)] = True

# (1) keep crease verts NEAR hair (drop bare-skin: throat, eye)
near = set(int(v) for v in np.where(hairv)[0].tolist())
for _ in range(int(os.environ.get('HC_RINGS', '3'))):
    add = set()
    for v in near:
        add |= vadj[v]
    near |= add
keepv = set(int(v) for v in crev.tolist() if v in near)
print('floor=%.3f  crease verts: %d, near-hair: %d' % (floor, len(crev), len(keepv)))

# (2) THE RULE: remove everything BEHIND the front line (front-most per tight X/Z cell + margin).
# OFF by default (HC_RULE=0) so ALL near-hair creases stay blue ("all the red where hair is -> blue").
if int(os.environ.get('HC_RULE', '0')) and keepv:
    head_w = float(np.ptp(co[znv >= floor, 0])); head_d = float(np.ptp(co[znv >= floor, 1]))
    cell = float(os.environ.get('HC_CELL', '0.05')) * head_w
    margin = float(os.environ.get('HC_MARGIN', '0.10')) * head_d
    kv = np.array(sorted(keepv), dtype=np.int64)
    gx = np.floor(co[kv, 0] / cell).astype(np.int64); gz = np.floor(co[kv, 2] / cell).astype(np.int64)
    _y = co[kv, 1]
    cmin = {}
    for i in range(len(kv)):
        kk = (int(gx[i]), int(gz[i]))
        if _y[i] < cmin.get(kk, 1e9):
            cmin[kk] = float(_y[i])
    keep = np.array([_y[i] <= cmin[(int(gx[i]), int(gz[i]))] + margin for i in range(len(kv))])
    keepv = set(int(v) for v in kv[keep].tolist())
    print('  remove-behind (cell=%.4f margin=%.4f): kept %d/%d verts' % (cell, margin, int(keep.sum()), len(kv)))

# (3) output the crease EDGES among survivors
bnd = [(int(u), int(w)) for (u, w) in cre if int(u) in keepv and int(w) in keepv]
np.savez(os.environ['HC_OUT'], loop=np.array(bnd, dtype=np.int64), anchors=np.zeros((0,), dtype=np.int64))
print('  crease-line edges: %d' % len(bnd))
