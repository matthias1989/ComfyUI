"""Step 2 v2 (front-most loop): the loop = the MOST-FORWARD crease at each height, per side, connected.
Not threading every arc (that grabbed back lines). Per (z-bin, side) pick the front-most crease vertex
(min y; face=-Y); order left top->bottom then right bottom->top; connect consecutive with Dijkstra on a
strongly FRONT-weighted, crease-cheap graph. Env P2_MESH, P2_CRE, P2_OUT, P2_FWD, P2_NZ."""
import numpy as np, scipy.sparse as sp, os
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from collections import defaultdict

mesh = np.load(os.environ['P2_MESH'], allow_pickle=True)
co = mesh['co']; fv = mesh['fv']; nv = int(mesh['nv'])
cre = np.load(os.environ['P2_CRE'])['edges']
FWD = float(os.environ.get('P2_FWD', '8.0')); NZ = int(os.environ.get('P2_NZ', '45'))
CRD = float(os.environ.get('P2_CRD', '0.5'))   # crease-ride discount AT THE FRONT (head-rel y<=DY0)
DY0 = float(os.environ.get('P2_DY0', '0.30'))  # below this head-rel depth: full discount
DY1 = float(os.environ.get('P2_DY1', '0.55'))  # above this head-rel depth: NO discount (deep creases = ear/interior get no pull)

e2f = defaultdict(list)
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)
creset = set((int(u), int(w)) if u < w else (int(w), int(u)) for u, w in cre)

_fmp = os.environ.get('P2_FACEMASK')
if _fmp and os.path.exists(_fmp):
    _fm = np.load(_fmp).astype(bool)
    if len(_fm) == nv:
        _b4 = len(creset)
        creset = set((u, w) for (u, w) in creset if not (_fm[u] and _fm[w]))
        cre = np.array(sorted(creset), dtype=np.int64) if creset else np.zeros((0, 2), np.int64)
        print('[facemask] dropped %d/%d creases inside the face' % (_b4 - len(creset), _b4))

# HAIR-MASK BOUNDARY (the drape edge). The mask is unreliable at the hairline (we use creases there) but
# good on the bulk DRAPE below the neck -> below the floor, follow this boundary so the line traces down
# the actual hair, not the bare body in front of it.
hair = mesh['hair'] if 'hair' in mesh.files else np.zeros(len(fv), bool)
bnd_mask = np.zeros(nv, bool); bnd_edges = []
for (_u, _w), _fs in e2f.items():
    if len(_fs) == 2 and bool(hair[_fs[0]]) != bool(hair[_fs[1]]):
        bnd_mask[_u] = True; bnd_mask[_w] = True; bnd_edges.append((_u, _w))
# OUTER drape edge = the hair-mask boundary that borders the BODY (the largest non-hair region), NOT the
# inner holes inside the hair (small non-hair components) which made the loop thread up inside the tube.
# Region-based, no y-margin -> keeps the whole outer drape edge.
from scipy.sparse.csgraph import connected_components as _ccomp
_rr = []; _ccl = []
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2 and (not hair[_fs[0]]) and (not hair[_fs[1]]):
        _rr.append(_fs[0]); _ccl.append(_fs[1])
_Gf = sp.csr_matrix((np.ones(len(_rr)), (_rr, _ccl)), shape=(len(fv), len(fv))); _Gf = _Gf + _Gf.T
_nc, _lab = _ccomp(_Gf, directed=False)
_bodyc = int(np.bincount(_lab[~hair]).argmax()) if (~hair).any() else -1
bnd_outer = np.zeros(nv, bool)
for (_a, _b), _fs in e2f.items():
    if len(_fs) == 2 and (bool(hair[_fs[0]]) != bool(hair[_fs[1]])):
        _nh = _fs[0] if not hair[_fs[0]] else _fs[1]
        if _lab[_nh] == _bodyc:
            bnd_outer[_a] = True; bnd_outer[_b] = True

ymin, ymax = float(co[:, 1].min()), float(co[:, 1].max())
xc = 0.5 * (float(co[:, 0].min()) + float(co[:, 0].max()))
znv = (co[:, 2] - float(co[:, 2].min())) / (float(co[:, 2].max()) - float(co[:, 2].min()) + 1e-9)
ynv = (co[:, 1] - ymin) / (ymax - ymin + 1e-9)

# --- proportion-independent NECK FLOOR: confine the loop to the head, not the arms/torso ---
def _neck_floor():
    nb = 60
    cnt = np.array([int(((znv >= i / nb) & (znv < (i + 1) / nb)).sum()) for i in range(nb)])
    w = np.array([float(np.ptp(co[(znv >= i / nb) & (znv < (i + 1) / nb), 0])) if cnt[i] > 20 else 0.0 for i in range(nb)])
    ws = w.copy()
    for i in range(1, nb - 1):               # smooth out per-bin noise
        ws[i] = (w[i - 1] + w[i] + w[i + 1]) / 3.0
    arm = int(np.argmax(ws))                 # widest band = shoulders + (A-pose) arms
    thr = float(os.environ.get('P2_NECK_FRAC', '0.3')) * ws[arm]   # arms "end" where width drops below this
    i = arm
    while i < nb - 1 and ws[i + 1] >= thr:   # climb until just above the shoulders = neck base (no overshoot to crown)
        i += 1
    return (i + 1 + 0.5) / nb

_env = os.environ.get('P2_ZMIN', 'auto')
P2_ZMIN = _neck_floor() if _env == 'auto' else float(_env)

# --- THE RULE: a crease IN FRONT (lower y) makes any crease BEHIND it (higher y) NEARBY irrelevant. ---
# Relative front-suppression (no absolute Y cap): drop a crease vertex if another crease vertex within a
# small (x,z) ball is more FORWARD (lower y) by > FSM. This is general: the forehead suppresses the crown
# directly behind it; the front hairline suppresses interior/ear creases behind it; but the temple and
# the below-ear line survive because they ARE the front-most where they sit (nothing in front of them).
_cvall = np.unique(cre)
_xz = np.column_stack([co[_cvall, 0], co[_cvall, 2]])
_yv = co[_cvall, 1]
# LATERAL-TIGHT, head-relative suppression. The old version used an ABSOLUTE, isotropic (x,z) radius
# (R=0.10) that was WIDER than the head half-width -> the central front face (most-forward near x=xc)
# suppressed the laterally-offset under-ear / side-of-neck hairline right across the head, even though
# nothing sat in front of it IN ITS OWN COLUMN. Fix: compare a crease only to more-forward creases in
# (nearly) the same lateral column (tight x, NO x-neighbour) and nearby height (z, +/-1). Cells are sized
# off the head's own width -> fully self-calibrating, no per-character constants.
_hwest = float(np.ptp(co[znv >= P2_ZMIN, 0]))      # head width from the neck floor (excludes the A-pose arms)
# SELF-CALIBRATING from the mesh's OWN resolution (median edge length): the lateral column width and the
# forward band are both one mesh cell -> "same lateral position" and "same depth" at the finest scale the
# geometry resolves. No tuned ratios, no head-relative magic numbers.
_el = float(np.median(np.linalg.norm(co[fv[:, 0]] - co[fv[:, 1]], axis=1)))
cellx = _el; FSM = _el
_gx = np.floor(_xz[:, 0] / cellx).astype(np.int64)
# COLUMN-WIDE relative front-suppression = the user's rule "nothing behind an already-defined
# hairline point". The most-FORWARD crease in each lateral (x) column suppresses everything behind
# it (higher y) by > FSM, across the WHOLE height of that column -> the forehead kills the crown,
# the side hairline kills the hair behind it. It's purely relational (front-vs-behind), no distance
# cap. FSM is just a head-relative band thickness so the hairline arc doesn't suppress itself.
# Ears are separate seams now (old under-ear height-coupling objection is moot); the deleted Y-cap
# used to be what stopped the crown, so this column rule now carries that. Drape (below floor) is exempt.
_colmin = {}
for _ii in range(len(_cvall)):
    _k = int(_gx[_ii])
    if _yv[_ii] < _colmin.get(_k, 1e9):
        _colmin[_k] = float(_yv[_ii])
_lmin = np.array([_colmin[int(_gx[_ii])] for _ii in range(len(_cvall))], dtype=float)
_keep = (_yv - _lmin) <= FSM
front_v = set(int(v) for v in _cvall[_keep].tolist())
print('front-suppression (self-calibrated cell=median-edge=%.4f): kept %d/%d' % (_el, int(_keep.sum()), len(_cvall)))
creset = set(e for e in creset if e[0] in front_v and e[1] in front_v)   # only FRONT creases get the ride-discount
if os.environ.get('P2_FILT_OUT'):   # save the front-suppressed crease set so the diagnostic BLUE = what the loop sees
    np.savez(os.environ['P2_FILT_OUT'], edges=(np.array(sorted(creset), dtype=np.int64) if creset else np.zeros((0, 2), np.int64)))

front_below = (znv < P2_ZMIN)
# Front-suppression is a HAIRLINE rule (it drops creases sitting BEHIND the forehead). It must NOT run on
# the neck/drape and kill the CONTINUATION. Add the below-floor creases back into the kept (blue) set.
if int(os.environ.get('P2_DRAPE_KEEPCRE', '0')):   # default OFF: front-suppress the drape too (keep its FRONT edge,
    _rawf = os.environ.get('P2_CRE_RAW')                 # RAW creases (pre face-clear) below the floor
    cre_raw = np.load(_rawf)['edges'] if _rawf else cre
    _rawv = np.unique(cre_raw) if len(cre_raw) else np.array([], np.int64)
    _belowsel = _rawv[front_below[_rawv]]
    front_v |= set(int(v) for v in _belowsel.tolist())
    _alle = list(cre) + (list(cre_raw) if _rawf else [])
    creset = set((int(a), int(b)) if a < b else (int(b), int(a)) for a, b in _alle
                 if int(a) in front_v and int(b) in front_v)
    print('keep-below-floor creases (raw=%s): front_v now %d' % (bool(_rawf), len(front_v)))
    if os.environ.get('P2_FILT_OUT'):
        np.savez(os.environ['P2_FILT_OUT'], edges=(np.array(sorted(creset), dtype=np.int64) if creset else np.zeros((0, 2), np.int64)))
P2_YFRONT = float(os.environ.get('P2_YFRONT', '0.5'))   # front/back split (face=-Y, front=low y)
XK = float(os.environ.get('P2_XKEEP', '2.0'))           # below-floor keep half-width, in head-half-widths
hw = 0.5 * float(np.ptp(co[znv >= P2_ZMIN, 0])) if (znv >= P2_ZMIN).any() else 0.1
xkeep = XK * hw
# Above the neck floor = head -> keep all. BELOW the floor keep ONLY the central-BACK column within a
# band just under the neck (nape / back-of-head hair); drop the FRONT body (chest), the WIDE
# arms/shoulders, AND the deep torso/spine (a central-back column with no lower bound otherwise lets the
# loop run all the way down the back). A long back drape below this band is deferred (back-hair = later).
ZLOW = P2_ZMIN - float(os.environ.get('P2_ZBAND', '0.10'))
body = (znv < P2_ZMIN) & ~((ynv >= P2_YFRONT) & (np.abs(co[:, 0] - xc) <= xkeep) & (znv >= ZLOW))
# HEAD-relative depth: 0 = front of the head (face), 1 = back (occiput). Normalizing over the head
# (not the whole body) gives front-weighting real leverage and lets us cap the hairline to the FRONT.
_hm = znv >= P2_ZMIN
hymin = float(co[_hm, 1].min()); hymax = float(co[_hm, 1].max())
yhv = (co[:, 1] - hymin) / (hymax - hymin + 1e-9)
YFMAX = float(os.environ.get('P2_YFMAX', '0.55'))   # drop anything behind this (head-rel depth) = crown/occiput +
#                                                      the blue-behind under the ear; under-ear FRONT line is forward of it
# BOUNDARY-PROXIMITY GUARD (self-calibrating, no fixed box): the hairline IS the hair<->skin boundary, so
# its creases sit within a few rings of that boundary EDGE. Two things are far from it and must drop: (a)
# bare-skin eye/brow/face creases (no hair near them), and (b) the INTERIOR of the hair -- the back/crown of
# the head, which is all-hair with no skin edge (this is what flooded the BLUE in the back). Seeding the
# proximity from the boundary edge (bnd_outer) instead of every hair face drops BOTH. For long hair the back
# of the head has no hair<->skin edge -> its interior creases are far from bnd_outer -> dropped.
# ── PROXIMITY GUARD REMOVED (KAN-8): it kept creases within a hand-tuned 0.40*head_width distance of the
# hair seed — a magic constant, forbidden. The face (P2_FACE), the front-facing guard below, and the
# neck-floor/body exclusion drop face/back/body creases; hair-interior is handled by the clean
# "hairline = sides + bottom only" boundary rule, not a distance ball. front_v is left untouched here.
print('proximity guard REMOVED (no distance constant): front_v = %d creases' % len(front_v))
# IGNORE THE BACK (self-calibrating, no fixed value): a crease is kept only if its surface faces the FRONT
# (outward normal points toward the face, -Y). The whole BACK of the hair/head faces away (+Y) -> dropped.
# This is the "remove the spots on the back" the user has asked for repeatedly -- decided by facing
# direction (geometry), not a y-threshold. Orientation is made outward via the head centre.
_t3 = np.array([[int(f[0]), int(f[1]), int(f[2])] for f in fv])
_fnf = np.cross(co[_t3[:, 1]] - co[_t3[:, 0]], co[_t3[:, 2]] - co[_t3[:, 0]])
_fcen = co[_t3].mean(1)
_hc = co[znv >= P2_ZMIN].mean(0)
_flip = np.sum(_fnf * (_fcen - _hc), axis=1) < 0
_fnf[_flip] = -_fnf[_flip]                          # orient outward (away from head centre)
_vn = np.zeros((nv, 3))
np.add.at(_vn, _t3[:, 0], _fnf); np.add.at(_vn, _t3[:, 1], _fnf); np.add.at(_vn, _t3[:, 2], _fnf)
_vln = np.linalg.norm(_vn, axis=1, keepdims=True); _vnn = _vn / np.where(_vln > 0, _vln, 1.0)
_faceT = float(os.environ.get('P2_FACE_T', '0.30'))   # drop only STRONGLY back-facing (normal_y>T) = the back of the
backface = (_vnn[:, 1] > _faceT) if int(os.environ.get('P2_FRONT_FACE', '1')) else np.zeros(nv, bool)  # head; the under-ear faces front (median -0.6) so relaxed T keeps it (was >0 -> ate the under-ear)
if int(os.environ.get('P2_FRONT_FACE', '1')) and front_v:
    _fa2 = np.fromiter(front_v, dtype=np.int64, count=len(front_v))
    _nb = int(backface[_fa2].sum())
    front_v -= set(int(v) for v in _fa2[backface[_fa2]].tolist())
    print('front-facing guard: dropped %d back-facing creases (the back of the hair)' % _nb)
# DEEP-BACK Y-CAP (head only): drop creases past YFMAX of the head's depth = the crown/occiput. The facing
# guard catches the BACK (faces +Y) but NOT the up-facing CROWN (faces +Z); this removes it. The under-ear
# hairline sits well forward of YFMAX (~0.6 vs 0.72) so it's untouched. Head-relative -> self-calibrating.
# Applied only ABOVE the floor (the head); the drape below is handled separately.
if int(os.environ.get('P2_YCAP', '1')) and front_v:
    _fa3 = np.fromiter(front_v, dtype=np.int64, count=len(front_v))
    _capped = _fa3[(znv[_fa3] >= P2_ZMIN) & (yhv[_fa3] > YFMAX)]
    front_v -= set(int(v) for v in _capped.tolist())
    print('deep-back Y-cap (YFMAX=%.2f, head only): dropped %d crown/occiput creases' % (YFMAX, len(_capped)))
# Re-save the diagnostic BLUE so it reflects the FINAL kept creases (AFTER the guards) -- otherwise the
# diag shows creases the loop no longer uses, which looks like the back wasn't cleaned.
if os.environ.get('P2_FILT_OUT'):
    _alle2 = _alle if '_alle' in dir() else list(cre)
    creset = set((int(a), int(b)) if a < b else (int(b), int(a)) for a, b in _alle2 if int(a) in front_v and int(b) in front_v)
    np.savez(os.environ['P2_FILT_OUT'], edges=(np.array(sorted(creset), dtype=np.int64) if creset else np.zeros((0, 2), np.int64)))
# THROAT clear: the CENTRAL FRONT of the neck (a band just above the floor) is bare skin, never a
# hairline -- the loop was falling right on it. Clear it (head-rel FRONT + CENTRAL). Narrow in x so the
# lateral sideburn/jaw hairline is untouched; the BACK (nape) is kept for back-hair.
THZ = float(os.environ.get('P2_THROAT_Z', '0.06'))
XN = float(os.environ.get('P2_THROAT_X', '0.5')) * hw
body = body | ((znv >= P2_ZMIN) & (znv < P2_ZMIN + THZ) & (yhv < 0.5) & (np.abs(co[:, 0] - xc) < XN))
# DRAPE keep (MASK-based): below the floor the front-most crease is the bare BODY (in front of the hair),
# so creases can't trace the drape. Instead follow the HAIR-MASK BOUNDARY (the drape edge): below the
# floor keep ONLY the hair-mask boundary verts; everything else below stays 'body'. The mask is reliable
# on the bulk drape (unlike at the hairline, where we keep using creases).
if int(os.environ.get('P2_DRAPE', '1')):
    body = body & ~(front_below & bnd_outer & ~backface)   # FRONT-FACING drape edge only (no back spots)
# FOLLOW THE RED CREASES: keep the (now un-suppressed) below-floor creases as anchors so the loop rides
# them down. No front/back/lateral pre-filter -- the connected-component filter below keeps only the chain
# linked to the hairline and drops disconnected torso/strand creases, so the line continues and stops
# where the creases actually run out.
_fvm = np.zeros(nv, bool)
if front_v:
    _fvm[np.fromiter(front_v, dtype=np.int64, count=len(front_v))] = True
if int(os.environ.get('P2_DRAPE_FRONT', '1')):
    body = body & ~((znv < P2_ZMIN) & _fvm)
# BRIDGE: the blue creases are isolated points; the smooth hair surface between them is 'body', so the
# green line can't hop from one to the next. Dilate the below-floor creases a few rings along the mesh to
# open that connecting hair surface, so the line can travel down the hair and REACH the lower blue. The
# component filter + outer-edge selection keep it on the hair edge, not diving through the interior.
if int(os.environ.get('P2_DRAPE_BRIDGE', '0')):
    _vadj = defaultdict(list)
    for (_a, _b) in e2f.keys():
        _vadj[_a].append(_b); _vadj[_b].append(_a)
    _seed = set(int(v) for v in np.where((znv < P2_ZMIN) & _fvm & ~body)[0].tolist())
    _cur = set(_seed)
    for _ in range(int(os.environ.get('P2_DRAPE_RINGS', '2'))):
        _nx = set()
        for v in _cur:
            _nx.update(_vadj[v])
        _cur |= _nx
    _bb = np.fromiter(_cur, dtype=np.int64, count=len(_cur))
    _bb = _bb[znv[_bb] < P2_ZMIN]                  # stay below the floor (don't reopen the head)
    body[_bb] = False
print('neck floor z_norm=%.3f  head_halfwidth=%.3f  xkeep=%.3f  below-floor crease verts=%d' % (
    P2_ZMIN, hw, xkeep, int(((znv < P2_ZMIN) & _fvm & ~body).sum())))
# ── STEP 2 — REMOVE THE FACE (MediaPipe landmarks): a SOFT barrier on the face region (eyes/nose/
# mouth/cheeks, NOT the hairline). The face is made EXPENSIVE to cross in the graph (so the loop
# traces AROUND it = the hairline, no nose line) and is barred as ANCHORS, but its edges REMAIN so
# connectivity is preserved. (A HARD body-barrier disconnected the side neck-line -> the component
# filter dropped it as an orphan.) Per-vert mask from the texturing face detector
# (compute_face_region_faces) -- geometry/landmarks, independent of the hair detection. P2_FACE npy.
faceb = np.zeros(nv, bool)
if os.environ.get('P2_FACE') and os.path.exists(os.environ['P2_FACE']):
    _frv = np.load(os.environ['P2_FACE'])
    if len(_frv) == nv:
        faceb = _frv.astype(bool)
        print('[step2-face] face-region SOFT barrier: %d verts (graph-penalized + anchor-barred, edges kept for connectivity)' % int(faceb.sum()))
    else:
        print('[step2-face] P2_FACE len %d != nv %d -> skipped' % (len(_frv), nv))
# --- DRAPE = OUTER edge. Above the floor the hairline rides the FRONT (min y); below it the seam must
# ride the OUTER (posterior) edge of the hair drape, so the drape anchors pick MAX y (back), not min y.
# This is the fix for "the algorithm always refuses the outer parts": front-most was right for the
# hairline but pulled the drape onto the inner (side-neck) edge instead of the outer hair edge. ---
belowf = znv < P2_ZMIN
FACE_PEN = float(os.environ.get('P2_FACE_PEN', '30.0'))   # STEP 2: cost multiplier to cross the face region
rows = []; cols = []; wts = []
for (u, w) in e2f.keys():
    if body[u] or body[w]:    # front body only -> bridges can't dip into the chest; back hair kept
        continue
    L = float(np.linalg.norm(co[u] - co[w]))
    yn = 0.5 * (yhv[u] + yhv[w])          # HEAD-relative front-weighting (real leverage within the head)
    if yn < 0.0:                          # drape/body verts can sit more front than the head -> clamp so
        yn = 0.0                          # the weight never goes negative (negative weights break/​hang Dijkstra)
    L *= (1.0 + FWD * yn)                 # strongly prefer the FRONT
    if (u, w) in creset:
        t = (yn - DY0) / (DY1 - DY0)      # depth-graded discount: deeper-in-Y creases lose their pull
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        L *= CRD + (1.0 - CRD) * t        # front crease -> CRD (full discount); deep crease -> 1.0 (no discount)
    if faceb[u] or faceb[w]:              # STEP 2 soft barrier: face region is expensive to cross -> the
        L *= FACE_PEN                     # loop routes around it (no nose line); edge stays -> neck connected
    rows += [u, w]; cols += [w, u]; wts += [L, L]
G = sp.csr_matrix((wts, (rows, cols)), shape=(nv, nv))

cv = np.unique(np.concatenate([                  # creases (incl. the un-suppressed below-floor ones) + hair-mask drape edge
    np.array(sorted(front_v), dtype=np.int64),
    np.where(bnd_outer & front_below & ~backface)[0].astype(np.int64)]))
cv = cv[~body[cv] & ~faceb[cv]]                   # + neck-floor / body exclusion + no anchors on the face
# CONNECTED-COMPONENT filter: keep only anchors reachable (in the graph) from the hairline crown, so the
# front-edge chain that links to the hairline survives but disconnected TORSO creases (and stray islands)
# are dropped -> the loop rides the red hair-neck line and stops where it disconnects (the collarbone).
from scipy.sparse.csgraph import connected_components as _cc2
_ncomp, _lbl = _cc2(G, directed=False)
# CROWN = topmost HAIR vertex, NOT the topmost crease. On elf characters the highest crease sits on
# the EAR TIP (a peripheral component), so anchoring the component filter there keeps only the ear's
# tiny island and drops the whole hairline. The hair crown is always in the main head component.
_hairv = (np.unique(np.concatenate([np.asarray(fv[_fi], dtype=np.int64) for _fi in np.where(hair)[0]]))
          if hair.any() else np.array([], dtype=np.int64))
if len(_hairv):
    _topv = int(_hairv[np.argmax(znv[_hairv])])
else:
    _topv = max(front_v, key=lambda v: znv[v]) if front_v else int(cv[np.argmax(znv[cv])])
_keepc = _lbl[cv] == _lbl[_topv]
if int(_keepc.sum()) < max(8, int(0.05 * len(cv))):   # live-mesh (pipeline) graph fragments -> the
    print('[port] component filter dropped ~all (%d/%d) -> keeping all anchors (fragmented graph)'
          % (int(_keepc.sum()), len(cv)))                # component filter would nuke everything; skip it
    _keepc = np.ones(len(cv), dtype=bool)
print('component filter: kept %d/%d anchors (%d comps)' % (int(_keepc.sum()), len(cv), _ncomp))
cv = cv[_keepc] if not os.environ.get('P2_NOCOMP') else cv
np.save('_gb_kept.npy', (np.zeros(nv,bool).__class__ if False else __import__('numpy').isin(np.arange(nv),cv)))  # DEBUG kept points after ALL filters
# HEAD/DRAPE-DECOUPLED z-binning: a long hair drape (deep below the neck floor) must NOT steal the
# head's bins. The OLD binning spread NZ bins over the FULL cv z-range [drape_bottom .. crown], so a
# long drape left the head (top ~14%) only a few bins -> few hairline anchors -> collapsed loop. Bin
# the HEAD (>= neck floor) over its OWN z-range with the full NZ bins, and the drape below with its
# own NZ bins -> head anchor density is independent of drape length (dynamic for any character).
zmn, zmx = float(co[cv, 2].min()), float(co[cv, 2].max())
_floorz = float(co[:, 2].min()) + P2_ZMIN * (float(co[:, 2].max()) - float(co[:, 2].min()))
_zcv = co[cv, 2]; _ishead = _zcv >= _floorz
_hb = np.clip(((_zcv - _floorz) / (zmx - _floorz + 1e-9) * NZ).astype(int), 0, NZ - 1)
_db = np.clip(((_zcv - zmn) / (_floorz - zmn + 1e-9) * NZ).astype(int), 0, NZ - 1)
zb = np.where(_ishead, NZ + _hb, _db)          # head -> bins [NZ, 2NZ);  drape -> bins [0, NZ)
aL = {}; aR = {}
for i, v in enumerate(cv):
    d = aR if co[v, 0] > xc else aL
    k = int(zb[i])
    if k not in d:
        d[k] = int(v)
    elif k >= NZ and co[v, 1] < co[d[k], 1]:     # HEAD hairline -> front-most (min y; forward = -Y)
        d[k] = int(v)
    elif k < NZ and co[v, 1] > co[d[k], 1]:       # DRAPE -> OUTER/back edge (max y), NOT the front-neck
        d[k] = int(v)

# FRONT-ENVELOPE filter: drop an anchor whose y is far behind its z-neighbours' front-most (a back
# spike = interior strand band / ear). Adaptive tolerance from STRANDINESS (head-face normal
# dispersion): strandy heads need it LOOSE, smooth heads TIGHT. Override with P2_YTOL.
tri = np.array([[int(v[0]), int(v[1]), int(v[2])] for v in fv])
_nrm = np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]])
_ln = np.linalg.norm(_nrm, axis=1, keepdims=True); fnrm = _nrm / np.where(_ln > 0, _ln, 1.0)
_znf = (co[tri].mean(1)[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)
_pairs = np.array([fs for fs in e2f.values() if len(fs) == 2])
_hp = _pairs[(_znf[_pairs[:, 0]] > 0.86) & (_znf[_pairs[:, 1]] > 0.86)]
strand = float((1 - np.abs(np.sum(fnrm[_hp[:, 0]] * fnrm[_hp[:, 1]], axis=1))).mean()) if len(_hp) else 0.025
ytf = float(os.environ['P2_YTOL']) if 'P2_YTOL' in os.environ else min(0.32, max(0.10, 0.10 + 9.5 * (strand - 0.019)))
print('strandiness=%.4f -> envelope YTOL frac=%.3f' % (strand, ytf))
YTOL = ytf * (ymax - ymin)
ENVW = int(os.environ.get('P2_ENVW', '4'))
def _envelope(d):
    # Relative front-envelope, applied EVERYWHERE (self-calibrating, no fixed y): drop an anchor whose y is
    # far BEHIND the front-most of its z-neighbours by > YTOL. That removes the BACK drape (far behind the
    # front hairline) while keeping the front-side hairline (incl. under-ear/above-shoulder, which is NOT
    # far behind). YTOL is set from strandiness, not hardcoded.
    if not d:
        return d
    ks = sorted(d); yk = {k: float(co[d[k], 1]) for k in ks}
    out = {}
    for k in ks:
        if k < NZ:                 # drape bin -> keep (front-envelope is wrong for the back drape)
            out[k] = d[k]; continue
        nb = [yk[j] for j in ks if abs(j - k) <= ENVW and j >= NZ]
        if (not nb) or yk[k] <= min(nb) + YTOL:
            out[k] = d[k]
    return out
_nL0, _nR0 = len(aL), len(aR)
import copy as _copy
aL0 = _copy.copy(aL); aR0 = _copy.copy(aR)
aL = _envelope(aL); aR = _envelope(aR)
if os.environ.get('P2_DEBUG'):
    fl = P2_ZMIN
    rmask = (co[cv, 0] > xc) & (znv[cv] < fl)
    rv = cv[rmask]; rzb = zb[rmask]
    print('--- DEBUG: right-side bnd_outer candidates BELOW floor (znv<%.3f): %d verts ---' % (fl, len(rv)))
    for k in sorted(set(rzb.tolist())):
        sel = rv[rzb == k]; ys = co[sel, 1]
        print('  zbin %2d znv~%.3f n=%2d y=%.3f..%.3f  (min-y SELECTED=%.3f)  x-xc=%.3f..%.3f' % (
            k, float(znv[sel].mean()), len(sel), ys.min(), ys.max(), ys.min(),
            float((co[sel, 0] - xc).min()), float((co[sel, 0] - xc).max())))
    r0 = sorted([k for k, v in aR0.items() if znv[v] < fl]); r1 = sorted([k for k, v in aR.items() if znv[v] < fl])
    print('  aR below-floor zbins PRE-envelope :', r0)
    print('  aR below-floor zbins POST-envelope:', r1)
    print('  envelope DROPPED below-floor R zbins:', sorted(set(r0) - set(r1)))
CLOSE = int(os.environ.get('P2_CLOSE', '1'))
if CLOSE:
    order = [aL[k] for k in sorted(aL, reverse=True)] + [aR[k] for k in sorted(aR)]
else:   # OPEN front-hairline arc: left sideburn -> up the temple -> over the forehead -> down to the
    # right sideburn. No bottom bridge, so the loop never closes across the bare front of the neck.
    order = [aL[k] for k in sorted(aL)] + [aR[k] for k in sorted(aR, reverse=True)]
print('anchors L=%d R=%d (envelope dropped %d) close=%d' % (len(aL), len(aR), (_nL0 - len(aL)) + (_nR0 - len(aR)), CLOSE))

def path(a, b):
    dist, pred = dijkstra(G, indices=a, return_predecessors=True)
    if a != b and pred[b] < 0:
        return [a, b]
    p = [b]
    while p[-1] != a and p[-1] >= 0:
        p.append(int(pred[p[-1]]))
    return p[::-1]

loopverts = [order[0]]
_seg_fail = 0
for i in range(1, len(order)):
    pp = path(loopverts[-1], order[i])
    if len(pp) == 2 and ((pp[0], pp[1]) not in e2f) and ((pp[1], pp[0]) not in e2f):
        _seg_fail += 1
        if os.environ.get('P2_DEBUG'):
            print('  SEG FAIL %d->%d  znv %.3f->%.3f' % (pp[0], pp[1], znv[pp[0]], znv[pp[1]]))
    loopverts += pp[1:]
if CLOSE:
    loopverts += path(loopverts[-1], order[0])[1:]
if os.environ.get('P2_DEBUG'):
    lv = np.array(loopverts)
    print('loopverts=%d znv %.3f..%.3f below-floor=%d  failed-segments=%d' % (
        len(lv), znv[lv].min(), znv[lv].max(), int((znv[lv] < P2_ZMIN).sum()), _seg_fail))
loopedges = [(loopverts[i], loopverts[i + 1]) for i in range(len(loopverts) - 1) if loopverts[i] != loopverts[i + 1]]
np.savez(os.environ['P2_OUT'], loop=np.array(loopedges, dtype=np.int64), anchors=np.array(order, dtype=np.int64),
         neck_floor=np.float64(P2_ZMIN))   # so _hairfill can block the body (non-hair) from crossing the neck
print('front-most loop: anchors=%d edges=%d' % (len(order), len(loopedges)))
