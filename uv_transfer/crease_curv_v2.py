"""Precise concave detection via mesh CURVATURE (pyvista/VTK) -- integrates over a neighborhood so
it catches the small hairline height-step a per-edge dihedral misses. Then FILTER the noise the same
way as crease.py: keep only curvature near the hair-region edge (symmetrized band) and off the ears.
The junk curvature lives INSIDE the hair (strands) / INSIDE the face (features); the hairline is on
the boundary, so the band keeps it and drops the interior noise. Env CR_MESH, CR_OUT, CV_SIGN,
CV_PCT, CR_BAND."""
import pyvista as pv, numpy as np, os
from collections import defaultdict
from scipy.spatial import cKDTree

d = np.load(os.environ.get('CR_MESH', '_cmesh.npz'), allow_pickle=True)
co = d['co']; fv = d['fv']; hair = d['hair']; nv = int(d['nv']); nf = len(fv)

faces = []
for verts in fv:
    vs = [int(x) for x in verts]; faces.append(len(vs)); faces += vs
m = pv.PolyData(co.astype(np.float64), np.array(faces, dtype=np.int64)).triangulate()
c = np.asarray(m.curvature('mean'))
SGN = float(os.environ.get('CV_SIGN', '1')); PCT = float(os.environ.get('CV_PCT', '85'))
hi = (SGN * c) > np.percentile(SGN * c, PCT)

e2f = defaultdict(list); fc = np.zeros((nf, 3))
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; fc[fi] = co[vs].mean(0); k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)
fpairs = np.array([fs for fs in e2f.values() if len(fs) == 2], dtype=np.int64)

# symmetrized band around the hair region (BAND<=0 disables the band filter entirely)
BAND = int(os.environ.get('CR_BAND', '6'))
xc = 0.5 * (fc[:, 0].min() + fc[:, 0].max())
if BAND > 0:
    mc = fc.copy(); mc[:, 0] = 2 * xc - fc[:, 0]
    _, mir = cKDTree(fc).query(mc); hair_sym = hair | hair[mir]
    near = hair_sym.copy()
    for _ in range(BAND):
        nn = near.copy()
        np.logical_or.at(nn, fpairs[:, 0], near[fpairs[:, 1]]); np.logical_or.at(nn, fpairs[:, 1], near[fpairs[:, 0]])
        near = nn
else:
    near = np.ones(nf, bool)

# ear exclusion (lateral protrusion). Band starts at the AUTO NECK FLOOR (width-pinch, same method as
# path3) so the A-pose ARMS below the neck can't skew the skull-edge threshold. Fully self-calibrating:
# no hardcoded band z and no hardcoded |x| -- both derive from each mesh.
znf = (fc[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min()); axc = np.abs(fc[:, 0] - xc)
_znv = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min()); _NB = 60
_wd = np.array([float(np.ptp(co[(_znv >= i / _NB) & (_znv < (i + 1) / _NB), 0])) if ((_znv >= i / _NB) & (_znv < (i + 1) / _NB)).sum() > 20 else 0.0 for i in range(_NB)])
_ws = _wd.copy()
for _k in range(1, _NB - 1):
    _ws[_k] = (_wd[_k - 1] + _wd[_k] + _wd[_k + 1]) / 3.0
_arm = int(np.argmax(_ws)); _twr = 0.3 * _ws[_arm]; _i = _arm
while _i < _NB - 1 and _ws[_i + 1] >= _twr:
    _i += 1
_neck = (_i + 1 + 0.5) / _NB
EAR_ZMIN = float(os.environ['CR_EAR_ZMIN']) if 'CR_EAR_ZMIN' in os.environ else _neck
ear_faces = np.zeros(nf, bool)
# PER-Z-SLICE: the ear protrudes laterally at its OWN heights; the sideburn does not. In each head
# slice cut only the lateral tail BEYOND that slice's skull edge, and ONLY where that slice actually
# has a protrusion (max >> edge). This separates the ear (protrudes) from the sideburn (no protrusion
# at the jaw). A single global percentile can't (ear & sideburn overlap in |x|). Fully self-calibrating.
P_EAR = float(os.environ.get('CR_EAR_P', '80')); NBE = int(os.environ.get('CR_EAR_NB', '24'))
_ne = 0
for _b in range(NBE):
    _lo = EAR_ZMIN + (1.0 - EAR_ZMIN) * _b / NBE; _hi = EAR_ZMIN + (1.0 - EAR_ZMIN) * (_b + 1) / NBE
    sl = (znf >= _lo) & (znf < _hi)
    if int(sl.sum()) < 20:
        continue
    p = np.percentile(axc[sl], P_EAR); pmax = float(axc[sl].max())
    if pmax > 1.4 * p:                       # this slice has a lateral protrusion = ear
        ear_faces |= sl & (axc > p); _ne += 1
print('[ear] band z>=%.3f(neck-auto) per-slice, %d slices w/ protrusion, %d ear faces' % (EAR_ZMIN, _ne, int(ear_faces.sum())))
for _ in range(4):
    ee = ear_faces.copy()
    np.logical_or.at(ee, fpairs[:, 0], ear_faces[fpairs[:, 1]]); np.logical_or.at(ee, fpairs[:, 1], ear_faces[fpairs[:, 0]])
    ear_faces = ee
if os.environ.get('CR_EAR_OUT'):   # env-guarded export for the standalone hairline module (no behavior change)
    np.savez(os.environ['CR_EAR_OUT'], ear_faces=ear_faces, neck_floor=np.float64(_neck))
    print('[ear] exported ear_faces + neck_floor=%.3f -> %s' % (_neck, os.environ['CR_EAR_OUT']))

ZMIN = float(os.environ.get('CR_ZMIN', '0.0')); XMAX = float(os.environ.get('CR_XMAX', '9.0'))
FCLR = int(os.environ.get('CR_FACECLEAR', '0'))
FZ_TOP = float(os.environ.get('CR_FZ_TOP', '0.62')); FZ_BOT = float(os.environ.get('CR_FZ_BOT', '-1.0'))  # HEAD-RELATIVE (0=neck,1=crown)
FX = float(os.environ.get('CR_FX', '0.05')); SKINONLY = int(os.environ.get('CR_FACE_SKINONLY', '1'))
FRONT_FRAC = float(os.environ.get('CR_FRONT', '0.40'))   # how far back the "front face plane" reaches (frac of head depth)
yc = float(np.median(fc[znf > 0.78, 1])) if (znf > 0.78).any() else 0.0   # head front/back split
crease = []
for (u, w), fs in e2f.items():
    if len(fs) != 2:
        continue
    f1, f2 = fs
    if znf[f1] < ZMIN or abs(fc[f1, 0] - xc) > XMAX:   # region gate: drop body/arm noise only
        continue
    # FACE-FEATURE CLEAR: a fold within SKIN (BOTH faces skin) in the front-central-mid zone = eye/
    # nose/mouth. The hairline is a hair|skin boundary (one hair face) -> never both-skin -> kept,
    # even next to the eye. Top/sides are OUTSIDE the zone -> protected (mask may under-detect there).
    if (hi[u] and hi[w]) and (near[f1] and near[f2]) and not (ear_faces[f1] or ear_faces[f2]):
        crease.append((u, w))
crease = np.array(crease, dtype=np.int64)
_n0 = len(crease)
# FACE-FEATURE CLEAR (non-square, connectivity-based): drop crease COMPONENTS CONFINED to the
# central-mid-front face (isolated eye/nose/mouth). A component reaching the hairline frame
# (z>=FZ_TOP forehead, |x|>=FX temple, or z<=FZ_BOT neck) extends OUT -> it's the hairline passing
# through -> KEPT. So the box can never cut the hairline; only isolated central features go.
if FCLR and len(crease):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    vs = np.unique(crease); idx = {int(v): i for i, v in enumerate(vs.tolist())}
    rr2 = [idx[int(u)] for u, w in crease]; cc2 = [idx[int(w)] for u, w in crease]
    Gf = sp.csr_matrix((np.ones(len(rr2)), (rr2, cc2)), shape=(len(vs), len(vs))); Gf = Gf + Gf.T
    nc, lab = connected_components(Gf, directed=False)
    # DEPTH rule: face features (eyes/nose/mouth) sit on the FRONT face plane; the side hairline is
    # set BACK; the forehead is front but HIGH. A component CONFINED to the front plane (y<Y_FRONT)
    # AND below the forehead (zh<FZ_TOP) = a face feature -> remove. The forehead extends up
    # (zh>FZ_TOP), the sides extend back (y>Y_FRONT) -> kept.
    # z is normalized WITHIN THE HEAD [neck, crown] -> the window generalizes across body proportions
    # (the head can be the top ~12% of a full body; a fixed whole-body z window only clears the chin).
    zn_all = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)
    def _neck():
        nb = 60
        w = np.array([float(np.ptp(co[(zn_all >= i / nb) & (zn_all < (i + 1) / nb), 0]))
                      if ((zn_all >= i / nb) & (zn_all < (i + 1) / nb)).sum() > 20 else 0.0 for i in range(nb)])
        ws = w.copy()
        for i in range(1, nb - 1):
            ws[i] = (w[i - 1] + w[i] + w[i + 1]) / 3.0
        arm = int(np.argmax(ws)); thr = 0.3 * ws[arm]; i = arm
        while i < nb - 1 and ws[i + 1] >= thr:
            i += 1
        return (i + 1 + 0.5) / nb
    neck = _neck()
    zh = (zn_all - neck) / (1.0 - neck + 1e-9)        # 0 at neck, 1 at crown
    zhv = zh[vs]; yv = co[vs, 1]; xv = np.abs(co[vs, 0] - xc)
    hy = co[zn_all > neck, 1]                          # head y-range (above the neck)
    Y_FRONT = float(hy.min() + FRONT_FRAC * (hy.max() - hy.min()))
    hw = 0.5 * float(np.ptp(co[zn_all > neck, 0]))     # head half-width
    X_INNER = float(os.environ.get('CR_FACE_XIN', '0.3')) * hw   # eyes/nose/mouth reach the centerline; the temple hairline does NOT (narrower = keep more side creases)
    rem = set()
    for ci in range(nc):
        m = lab == ci
        # remove only a front + low-head component that ALSO reaches the central face (small |x|).
        # the lateral temple/forehead-side hairline stays low+front but never reaches center -> KEPT.
        if zhv[m].max() < FZ_TOP and zhv[m].min() > FZ_BOT and yv[m].max() < Y_FRONT and xv[m].min() < X_INNER:
            rem.add(ci)
    elab = np.array([lab[idx[int(u)]] for u, w in crease])
    crease = crease[np.array([l not in rem for l in elab], dtype=bool)]
    print('face-clear(head-rel neck=%.3f FZ_TOP=%.2f): removed %d comps (%d->%d edges)' % (neck, FZ_TOP, len(rem), _n0, len(crease)))

# FRONT-MOST filter (user's idea): the hairline is the FRONT edge of the hair; per (height,lateral)
# column keep only creases within FMARGIN of the most-forward one (min y; face=-Y), drop the ones
# behind (back-of-hair strands / drape interior) that pull the loop to the back and skip the forehead.
FRONTMOST = int(os.environ.get('CR_FRONTMOST', '0'))
if FRONTMOST and len(crease):
    from collections import defaultdict as _dd
    FCELL = float(os.environ.get('CR_FCELL', '0.012')); FMARGIN = float(os.environ.get('CR_FMARGIN', '0.025'))
    vv = np.unique(crease)
    FXBIN = float(os.environ.get('CR_FXBIN', '0'))   # >0: bin by fine x-columns (keeps the lateral temple
    # hairline, which x-SIGN binning drops because the forward forehead-CENTER dominates the whole side)
    zi = np.round(co[vv, 2] / FCELL).astype(int); yvv = co[vv, 1]
    xi = np.round(co[vv, 0] / FXBIN).astype(int) if FXBIN > 0 else (co[vv, 0] > xc).astype(int)   # per height, per x-COLUMN (or SIDE)
    cellmin = _dd(lambda: 1e9)
    for k in range(len(vv)):
        key = (int(zi[k]), int(xi[k]))
        if yvv[k] < cellmin[key]:
            cellmin[key] = yvv[k]
    front = np.array([yvv[k] < cellmin[(int(zi[k]), int(xi[k]))] + FMARGIN for k in range(len(vv))])
    fset = set(vv[front].tolist())
    _b = len(crease)
    crease = crease[np.array([(int(u) in fset and int(w) in fset) for u, w in crease], dtype=bool)]
    print('front-most filter: kept %d/%d front verts (%d->%d edges)' % (int(front.sum()), len(vv), _b, len(crease)))
# connected-component length filter: the hairline is one long connected curve; strand bits are short.
# DENSITY-RELATIVE (was a fixed 40-edge count = a different physical size on every remesh): require a crease
# component to span a meaningful fraction of the HEAD WIDTH, converted to an edge count via the mesh's own
# median edge length. Same real-world minimum length on any character / any mesh density.
_emed = np.sort(np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], 0), 1)
_elen = float(np.median(np.linalg.norm(co[_emed[:, 0]] - co[_emed[:, 1]], axis=1)))
_hwid = float(np.ptp(fc[znf >= _neck, 0])) if (znf >= _neck).any() else float(np.ptp(co[:, 0]))
_minlf = float(os.environ.get('CV_MINLEN_FRAC', '0.44'))   # min crease length as a fraction of head width
# (0.44 reproduces the old fixed MINC=40 on the reference mesh, now density-scalable to any mesh)
MINC = int(os.environ['CV_MINCOMP']) if 'CV_MINCOMP' in os.environ else max(3, int(round(_minlf * _hwid / max(_elen, 1e-9))))
print('[mincomp] density-relative: head_w=%.4f median_edge=%.5f -> MINC=%d edges (>=%.0f%% head width)' % (_hwid, _elen, MINC, 100 * _minlf))
if MINC > 0 and len(crease):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    vs = np.unique(crease); idx = {int(v): i for i, v in enumerate(vs.tolist())}
    r = [idx[int(u)] for u, w in crease]; cc = [idx[int(w)] for u, w in crease]
    G = sp.csr_matrix((np.ones(len(r)), (r, cc)), shape=(len(vs), len(vs))); G = G + G.T
    ncomp, lab = connected_components(G, directed=False)
    elab = np.array([lab[idx[int(u)]] for u, w in crease])
    sizes = np.bincount(elab)
    crease = crease[sizes[elab] >= MINC]
np.savez(os.environ.get('CR_OUT', '_ccrease.npz'), edges=crease)
print('curv+filter crease edges=%d -> %d after comp>=%d  (SGN=%g PCT=%g BAND=%d ear=%d)' % (_n0, len(crease), MINC, SGN, PCT, BAND, int(ear_faces.sum())))
