"""Extend the hair MASK down the drape (root fix for the seam dead-ending at the shoulder).
The hair is a thin shell: where it drapes OVER the body there is body right behind it (behind-dist
~0.008), whereas solid body has its far side far away (neck ~0.03, torso ~0.07). So flag below-floor
faces that have body close behind them (behind-dist < D) as hair, then keep only the connected
component touching the existing hair (drops stray hits). D is set between the hair and body scales.
Writes _Amesh_ext.npz. Env: EH_D (thickness, default 0.015), EH_OUT."""
import numpy as np, os
from collections import defaultdict, deque
import pyvista as pv

m = np.load('_Amesh.npz', allow_pickle=True)
co = m['co']; fv = m['fv']; nv = int(m['nv'])
hair = m['hair'].astype(bool).copy()
zmin, zmax = float(co[:, 2].min()), float(co[:, 2].max())
xc = 0.5 * (float(co[:, 0].min()) + float(co[:, 0].max()))
znv = (co[:, 2] - zmin) / (zmax - zmin + 1e-9)

# neck floor (mirror path3: widest band = shoulders/arms, climb to neck base)
def neck_floor():
    nb = 60
    cnt = np.array([int(((znv >= i / nb) & (znv < (i + 1) / nb)).sum()) for i in range(nb)])
    w = np.array([float(np.ptp(co[(znv >= i / nb) & (znv < (i + 1) / nb), 0])) if cnt[i] > 20 else 0.0 for i in range(nb)])
    ws = w.copy()
    for i in range(1, nb - 1):
        ws[i] = (w[i - 1] + w[i] + w[i + 1]) / 3.0
    arm = int(np.argmax(ws)); thr = 0.3 * ws[arm]; i = arm
    while i < nb - 1 and ws[i + 1] >= thr:
        i += 1
    return (i + 1 + 0.5) / nb
FLOOR = neck_floor()

faces_flat = []; fn = np.zeros((len(fv), 3)); fc = np.zeros((len(fv), 3))
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]; faces_flat.append(len(vs)); faces_flat.extend(vs)
    pts = co[vs]; fc[fi] = pts.mean(0)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0]); ln = float(np.linalg.norm(n))
    fn[fi] = n / ln if ln > 0 else n
mesh = pv.PolyData(co, np.array(faces_flat)).triangulate()
fznv = (fc[:, 2] - zmin) / (zmax - zmin + 1e-9)
diag = float(np.linalg.norm(co.max(0) - co.min(0)))
eps = 0.0008 * diag
D = float(os.environ.get('EH_D', '0.015'))

# face adjacency
e2f = defaultdict(list)
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]; e2f[(u, w) if u < w else (w, u)].append(fi)
adj = defaultdict(list)
for (u, w), fs in e2f.items():
    if len(fs) == 2:
        adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])

# candidates: below floor, above the waist, not already hair
cand = np.where((~hair) & (fznv < FLOOR) & (fznv > 0.40))[0]
print('floor znv=%.3f  candidates (below floor, znv>0.40, non-hair): %d  D=%.3f' % (FLOOR, len(cand), D))
shell = np.zeros(len(fv), bool)
for c in cand:
    o = fc[c] - eps * fn[c]
    pts, _ = mesh.ray_trace(o, o - fn[c] * (0.2 * diag), first_point=True)
    if pts.size and np.linalg.norm(pts - o) < D:
        shell[c] = True
print('shell-flagged candidates: %d' % int(shell.sum()))

# union, then connected component containing the original hair
flag = hair | shell
seen = np.zeros(len(fv), bool); dq = deque([fi for fi in range(len(fv)) if hair[fi]])
for fi in dq:
    seen[fi] = True
while dq:
    f = dq.popleft()
    for g in adj[f]:
        if flag[g] and not seen[g]:
            seen[g] = True; dq.append(g)
new = hair | (shell & seen)        # keep only shell faces connected to the existing hair
am = new & ~hair
print('ADDED %d faces (connected to hair).' % int(am.sum()))
if am.any():
    print('  added znv %.3f..%.3f  y %.3f..%.3f  |x-xc| max %.3f  FRONT(y<-0.02)=%d' % (
        fznv[am].min(), fznv[am].max(), fc[am, 1].min(), fc[am, 1].max(),
        float(np.abs(fc[am, 0] - xc).max()), int((am & (fc[:, 1] < -0.02)).sum())))
np.savez(os.environ.get('EH_OUT', '_Amesh_ext.npz'), co=co, fv=fv, nv=nv, nf=m['nf'], hair=new)
print('saved', os.environ.get('EH_OUT', '_Amesh_ext.npz'))
