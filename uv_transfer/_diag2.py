"""Diagnose why the drape seam stops: how low does HAIR go, how low does the hair<->body
boundary go, and below that, what is the hair adjacent to (free-hanging vs against body)."""
import numpy as np, os
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as ccomp

mesh = np.load('_Amesh.npz', allow_pickle=True)
co = mesh['co']; fv = mesh['fv']; nv = int(mesh['nv'])
hair = mesh['hair'] if 'hair' in mesh.files else np.zeros(len(fv), bool)
znv = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)
xc = 0.5 * (co[:, 0].min() + co[:, 0].max())

# face centroid znv
fcz = np.array([znv[[int(a) for a in f]].mean() for f in fv])
print('HAIR faces: %d, znv range %.3f..%.3f' % (hair.sum(), fcz[hair].min(), fcz[hair].max()))
print('  hair face znv percentiles 5/25/50: %.3f %.3f %.3f' % tuple(np.percentile(fcz[hair], [5, 25, 50])))

e2f = defaultdict(list)
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)

# all hair<->nonhair boundary verts, with z
bnd_mask = np.zeros(nv, bool)
for (_u, _w), _fs in e2f.items():
    if len(_fs) == 2 and bool(hair[_fs[0]]) != bool(hair[_fs[1]]):
        bnd_mask[_u] = True; bnd_mask[_w] = True
bm = np.where(bnd_mask)[0]
print('bnd_mask (ALL hair<->nonhair) verts: %d, znv range %.3f..%.3f' % (len(bm), znv[bm].min(), znv[bm].max()))

# body-component bnd_outer
_rr = []; _ccl = []
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2 and (not hair[_fs[0]]) and (not hair[_fs[1]]):
        _rr.append(_fs[0]); _ccl.append(_fs[1])
_Gf = sp.csr_matrix((np.ones(len(_rr)), (_rr, _ccl)), shape=(len(fv), len(fv))); _Gf = _Gf + _Gf.T
_nc, _lab = ccomp(_Gf, directed=False)
_bodyc = int(np.bincount(_lab[~hair]).argmax())
print('non-hair components: %d, body component size %d / %d' % (_nc, int((_lab[~hair] == _bodyc).sum()), int((~hair).sum())))
bnd_outer = np.zeros(nv, bool)
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2 and (bool(hair[_fs[0]]) != bool(hair[_fs[1]])):
        _nh = _fs[0] if not hair[_fs[0]] else _fs[1]
        if _lab[_nh] == _bodyc:
            bnd_outer[_a] = True; bnd_outer[_b] = True
bo = np.where(bnd_outer)[0]
print('bnd_outer (borders BODY) verts: %d, znv range %.3f..%.3f' % (len(bo), znv[bo].min(), znv[bo].max()))

# In the lower drape (znv<0.71) right side: are there hair faces? what non-hair components touch them?
lowmask = (fcz < 0.71) & hair & (np.array([co[[int(a) for a in f]][:, 0].mean() for f in fv]) > xc)
print('lower-drape (znv<0.71, x>xc) HAIR faces: %d' % int(lowmask.sum()))
# what non-hair faces are adjacent to lower-drape hair, and their component
adj_comp = defaultdict(int)
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2 and bool(hair[_fs[0]]) != bool(hair[_fs[1]]):
        hf = _fs[0] if hair[_fs[0]] else _fs[1]
        nf = _fs[1] if hair[_fs[0]] else _fs[0]
        if fcz[hf] < 0.71 and co[[int(a) for a in fv[hf]]][:, 0].mean() > xc:
            adj_comp[('body' if _lab[nf] == _bodyc else 'comp%d' % _lab[nf])] += 1
print('lower-drape hair boundary by adjacent non-hair component:', dict(adj_comp))
# count free hair edges in lower drape (hair-hair edges only => no boundary => free)
free = 0; tot = 0
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2:
        if (fcz[_fs[0]] < 0.71 or fcz[_fs[1]] < 0.71):
            if hair[_fs[0]] and hair[_fs[1]]:
                free += 1
print('lower (znv<0.71) hair-hair interior edges: %d' % free)
