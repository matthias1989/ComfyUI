"""Step 1: detect CONFIDENT hairline edges = sharp CONCAVE creases (the fold where the hair shell
meets skin). Newell face normals; an edge is concave if the neighbour face centroid sits in front
of the normal. Restrict to the head band. Output crease edges for a 3D render. No region mask used."""
import numpy as np, os
from collections import defaultdict

d = np.load(os.environ.get('CR_MESH', '_Bmesh.npz'), allow_pickle=True)
co = d['co']; fv = d['fv']; nv = int(d['nv']); nf = len(fv)
THR = float(os.environ.get('CR_THR', '0.06'))   # crease sharpness 1-dot(n1,n2)
SIGN = float(os.environ.get('CR_SIGN', '1'))     # +1 concave, -1 convex
ZMIN = float(os.environ.get('CR_ZMIN', '0.08'))  # safety floor only; BAND keeps creases near hair (incl. neck drape)

fn = np.zeros((nf, 3)); fc = np.zeros((nf, 3))
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; p = co[vs]; fc[fi] = p.mean(0)
    n = np.zeros(3); m = len(vs)
    for a in range(m):
        cur = p[a]; nxt = p[(a + 1) % m]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    ln = np.linalg.norm(n); fn[fi] = n / ln if ln > 1e-12 else 0

e2f = defaultdict(list)
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)

# FILTER (step 1.5): drop "wrong" concaves OUTSIDE the hair region (eyes/nose/mouth creases).
# Keep a crease only if both its faces lie within CR_BAND rings of the (geometric) hair region.
hair = d['hair']
BAND = int(os.environ.get('CR_BAND', '6'))      # 0 = no filter
if BAND > 0:
    # SYMMETRIZE the hair mask across the x-center first: the raw Trellis mask is lopsided
    # (smaller on one side), which made "near hair" asymmetric and dropped one neck's creases.
    # Mirror each face to its x-reflection; a face counts as hair if it OR its mirror is hair.
    from scipy.spatial import cKDTree
    xc = 0.5 * (fc[:, 0].min() + fc[:, 0].max())
    mc = fc.copy(); mc[:, 0] = 2 * xc - fc[:, 0]
    _, mir = cKDTree(fc).query(mc)
    hair_sym = hair | hair[mir]
    fpairs = np.array([fs for fs in e2f.values() if len(fs) == 2], dtype=np.int64)
    near = hair_sym.copy()
    for _ in range(BAND):
        nn = near.copy()
        np.logical_or.at(nn, fpairs[:, 0], near[fpairs[:, 1]])
        np.logical_or.at(nn, fpairs[:, 1], near[fpairs[:, 0]])
        near = nn
    print('  hair faces=%d  symmetrized=%d' % (int(hair.sum()), int(hair_sym.sum())))
else:
    near = np.ones(nf, bool)

# EAR exclusion: drop creases ON the ears (lateral protrusions) so the ear-base fold doesn't get
# treated as hairline and tangle the loop. Detect ears as the lateral protrusion in the head band,
# only if there's a clear lateral spike (so non-eared heads are untouched). Dilate to cover the base.
EAR_ZMIN = float(os.environ.get('CR_EAR_ZMIN', '0.80'))
EAR_PCTL = float(os.environ.get('CR_EAR_PCTL', '90'))
EAR_DILATE = int(os.environ.get('CR_EAR_DILATE', '3'))
zmn, zmx = co[:, 2].min(), co[:, 2].max()
znf = (fc[:, 2] - zmn) / (zmx - zmn)
xcen = 0.5 * (co[:, 0].min() + co[:, 0].max())
axc = np.abs(fc[:, 0] - xcen)
hb = znf > EAR_ZMIN
ear_faces = np.zeros(nf, bool)
if int(hb.sum()) > 0:
    p = np.percentile(axc[hb], EAR_PCTL); pmax = float(axc[hb].max())
    if pmax > 1.3 * p:                       # clear lateral spike -> ears present
        thr = max(p, 0.7 * pmax)
        ear_faces = hb & (axc > thr)
        _fp = np.array([fs for fs in e2f.values() if len(fs) == 2], dtype=np.int64)
        for _ in range(EAR_DILATE):
            ee = ear_faces.copy()
            np.logical_or.at(ee, _fp[:, 0], ear_faces[_fp[:, 1]])
            np.logical_or.at(ee, _fp[:, 1], ear_faces[_fp[:, 0]])
            ear_faces = ee
print('ear faces=%d' % int(ear_faces.sum()))

crease = []; dropped = 0
for e, fs in e2f.items():
    if len(fs) != 2:
        continue
    f1, f2 = fs
    if fc[f1, 2] < ZMIN or fc[f2, 2] < ZMIN:
        continue
    sharp = 1.0 - float(np.dot(fn[f1], fn[f2]))
    conc = float(np.dot(fc[f2] - fc[f1], fn[f1]))   # >0 concave (outward normals)
    if sharp > THR and SIGN * conc > 0:
        if (near[f1] and near[f2]) and not (ear_faces[f1] or ear_faces[f2]):
            crease.append((e[0], e[1]))
        else:
            dropped += 1
np.savez(os.environ.get('CR_OUT', '_Bcrease.npz'), edges=np.array(crease, dtype=np.int64))
print('crease edges=%d (dropped %d outside hair, BAND=%d)  THR=%.3f SIGN=%+d ZMIN=%.2f'
      % (len(crease), dropped, BAND, THR, SIGN, ZMIN))
