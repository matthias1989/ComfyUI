"""KAN-18 pelvic/genital donor graft -- the consolidated, validated recipe (B40).

Runs in headless Blender. Grafts the high-res donor genital (donor_pelvis_v2.npz) onto a
SMOOTH remeshed body and returns a clean, border-free, uniform-density result.

The seam saga (B1-B40) distilled to 3 stages, each fixing a distinct failure:
  1. BUILD  - insert donor + flush the outer margin flush to the body (no height step) +
              bridge + beautify (kill slivers) + HC-Laplacian wide smooth (flatten the
              seam without a moat, protecting the central vulva detail).
  2. FUSE   - voxel remesh @ FUSE_VOX. This makes the whole region UNIFORM density, which
              ELIMINATES the dense-donor/coarse-body density boundary. That boundary is the
              root of the persistent "border" (a vertex-normal artifact that survives every
              smoothing/normal trick) -- only fusing removes it.
  3. SEAM   - Taubin (anti-shrink) smooth of the fusion-seam ring. This now WORKS headless
              precisely because step 2 made the density uniform (the same smooth failed on the
              un-fused mesh because the density boundary fought it). Vulva + far body protected.

Detail intentionally ends up slightly soft (voxel) -> the fine micro-detail is restored by a
separate genital normal-map bake from the high-poly donor (Body_LP_Ref). See project memory.

Usage (standalone test):
  blender --background --python pelvis_graft.py -- --body handgraft_out.npz --voxel_body --out out.npz --blend out.blend
Pipeline:
  call graft_pelvis(body_V, body_F, voxel_body=False)  # body already the gen_seams remesh
"""
import os, sys, numpy as np
import bpy, bmesh
from collections import defaultdict
from mathutils import Vector, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
A_DONOR = os.path.join(HERE, "assets", "donor_pelvis_v2.npz")
A_ZONES = os.path.join(HERE, "assets", "genital_donor_zones.npy")  # per-donor-vert label: 0 skin, 1-2 outer labia, 3 inner labia, 4 opening
A_PLACE = os.path.join(HERE, "assets", "pelvis_place_fixed.npz")

# ---- recipe parameters (validated B38/B39/B40) ----
GS         = float(os.environ.get("PELVIS_GS", "1.10"))  # genital scale (1.0=anatomical, 1.10=+10%)
PLACE_DY   = 0.0       # placement nudge knobs (mesh Y=up/height, Z=depth). 0/0 = pg14. (Recess attempts
PLACE_DZ   = 0.0       #   didn't visibly help, reverted to 0.)
BODY_VOX   = 0.0028    # body voxel (matches gen_seams target density) -- only if voxel_body
DONOR_X_SCALE = 0.80   # narrow the placed donor in X (width) about the midline; <1 pulls the seam off the thighs
MORPH_R    = 0.05      # snap: full weight within MORPH_R of the anchor
MORPH_FALL = 0.022     # ... soft falloff to 0 over this width (blends the genital into the body)
SNAP_SUBDIV = 2        # (legacy snap) subdivide the genital region this many cuts
SNAP_CLAMP  = 0.012    # (legacy snap) clamp distance
SNAP_SMOOTH_ITERS = 40 # (legacy snap) ring iterations
GRAFT_SEAM_BAND  = 0.009  # GRAFT seam smooth: smooth verts within this of the donor rim line (both sides)
GRAFT_SEAM_ITERS = 120    # ... iterations (inner detail beyond the band is protected, so this can be high)
GRAFT_PROTECT_R  = 0.003  # NEVER smooth verts within this of a LABELED detail vert (outer/inner labia + opening,
                          # donor zones 1-4). Distance-to-rim alone fails near the anus where the rim hugs the
                          # labia -> the outer labia got eaten. Labeled detail = 100% protected; smooth only outside it.
MONS_R     = 0.020        # gentle pubic-mound de-bump: radius from the vulva centre. The anus sits ~0.025-0.03 back
MONS_ITERS = 12           # so it stays untouched; labia stay protected. Light de-bump, not a flatten.
MONS_W     = 0.30
FLUSH_HI   = 0.032     # flush the outer margin: weight 1 for dmin<FLUSH_LO ...
FLUSH_W    = 0.014     #   ... ramping to 0 by FLUSH_HI (dmin = dist from donor rim)
# A-fix (depth-stretch walls): the stretch lifts the back perineum off the body surface -> a height step
# (wall) the smoother can't remove. So also SHRINKWRAP the stretched back perineum onto the body so it
# follows the curve. Ramp 0 at the vulva (front, protected) -> BACK_STRENGTH at the back; keep the anus dimple.
BACK_STRENGTH = 0.0             # A-fix DISABLED (didn't kill the stretch walls). Option B = gentler stretch.
BACK_Z0, BACK_W = 0.008, 0.045   # back-region ramp: starts 0.008 behind vulva, full by +0.045 (body +Z)
BACK_ANUS_KEEP = 0.012           # don't shrinkwrap within this radius of the anus (keep the dimple)
CUT        = 1.08      # cut the body hole just BEYOND the donor rim (no overlap)
BRIDGE_CUTS= 2
BEAUTY_R   = 0.040     # seam band (from rim) to triangulate+beautify
# HC-Laplacian wide smooth (flatten seam, protect central vulva)
HC_REG, HC_PX, HC_PR = 0.084, 0.016, 0.044
HC_ALPHA, HC_BETA, HC_ITERS = 0.1, 0.6, 45
FUSE_VOX   = 0.0012    # fine fuse -> uniform density (kills the density border)
# seam-smooth of the fused crease: tangential RELAX (even spacing) + Taubin (flatten)
# smooth the WHOLE surround (mons + inner thighs + perineum) -- not just the seam ring -- to match
# B40; the surround carries the original body-voxel waviness the fine fuse preserved.
# GPU-sculpt mask. Protect the whole DONOR interior (<INNER; rim/seam is at dA~0.074) so the strong
# smooth works OUTWARD on the body/walls instead of dishing the donor. SOFT edges (wide falloffs) so no
# step/wall forms where smooth meets protected.
# pg14 mask (proven) + small anus guard. The walls are fixed at the SOURCE by the back-flush (A-fix),
# not by the smoother, so the mask stays at pg14's values.
SEAM_INNER, SEAM_FAR = 0.028, 0.130   # (legacy wide-surround band -- standalone/non-local only)
INNER_FALL, FAR_FALL = 0.008, 0.025
# LOCAL/pipeline: smooth ONLY a thin band at the donor->body SEAM (dist-from-rim), NOT the wide surround --
# the pipeline body has real detail (buttocks cleft, mons) a wide smooth destroys. We only blend the graft line.
SEAM_BAND, SEAM_BAND_FALL = 0.030, 0.024   # front/legs: feathered band -> blends the seam without a band-edge crease
SEAM_BELLY_CUT = 0.5                  # belly side (above the anchor, +Y) smoothed less
# BACK: tight band + tight falloff -> smooth ONLY the immediate seam line, leave the buttocks raw
SEAM_BACK_NARROW = 0.024             # COMBINE (v7 back + v13 front): tight back band ~0.006 -> minimal back
                                     #   smoothing = preserves the BUTT SHAPE (user preferred v7's back). Front
                                     #   unaffected (front-only belly-cut keeps v13's front). (v13 used 0.008/wide.)
SEAM_BACK_FALL = 0.005              # back falloff tight (v7) -> butt stays raw
CLEFT_W = 0.013                      # PROTECT the gluteal cleft: a midline strip (|X|<CLEFT_W) on the back
                                      #   stays unsmoothed so the natural cleft groove is kept (like the brush)
ANUS_R, ANUS_FALL = 0.016, 0.016      # small guard: keep the anus dimple from the smoother
RELAX_ITERS, SEAM_ITERS = 60, 200   # v13 locked (280 added no real smoothing gain)
SURR_ITERS = 130                      # LOCAL/headless surround polish: Taubin lambda/mu pairs
SURR_LAM, SURR_MU = 0.5, -0.53        #   (volume-preserving -> no crater/shrink)
SURR_RELAX = 0                        #   tangential RELAX pre-pass OFF: with stale normals it INFLATED the
                                      #   surround into a raised welt around the genital (much worse). Don't.
# LOCAL border fix: grade the donor's RIM density down toward the body's so the dense-donor/coarse-body
# density jump (-> the vertex-normal "border") disappears. Protect the detailed center; decimate the rim.
GRADE_RATIO = 1.0                     # decimate the rim to this fraction (1.0 = OFF: collapse-decimating the
                                      #   rim made JAGGED slivers = a worse seam. Border -> handled by the
                                      #   genital normal-map bake instead, which overrides geometry normals.)
GRADE_KEEP, GRADE_RAMP = 0.045, 0.025 # protect dA<KEEP (vulva/anus); ramp to full decimate by KEEP+RAMP
GENITAL_HUG = 0.013                   # TIGHT genital flag: face within this of the donor SURFACE = genital
                                      #   (donor + ~1 border ring) -> UV island/material hugs only the genitals,
                                      #   surrounding skin stays part of the body. (replaces the big GENITAL_R sphere)
GENITAL_R = 0.085                     # LOCAL: faces within this dA of the anchor = the genital region ->
                                      #   saved as a per-face flag so gen_seams cuts it into its own UV
                                      #   island (separate texturing; body front-image can't see it)
FINAL_SMOOTH_FAC, FINAL_SMOOTH_ITERS = 0.5, 0   # global pass off


def _leg_junction_Y(V):
    ymin, ymax = V[:, 1].min(), V[:, 1].max(); st = (ymax - ymin) / 60
    for i in range(60):
        b = V[(V[:, 1] >= ymin + st * i) & (V[:, 1] < ymin + st * (i + 1))]
        if len(b) and (np.abs(b[:, 0]) < 0.04).sum() > 20:
            return ymin + st * i
    return (ymin + ymax) / 2


def _place(Vt):
    """Auto-place the donor on this body: anchor at the leg junction, scale to pelvis width."""
    f = np.load(A_PLACE); L = f["L"]; A = f["A"].copy(); ref_w = float(f["ref_width"]); Gc = f["Gc"]
    A[1] += PLACE_DY; A[2] += PLACE_DZ   # recess the whole region (reduce sticking out)
    dd = np.load(A_DONOR); Vd = dd["V"].astype(float); Fd = dd["F"].astype(int)
    # Scale the donor by the body width WHERE THE DONOR ACTUALLY SITS -- the anchor -- not at a
    # separately detected "leg junction".
    # _leg_junction_Y returns the FIRST y (scanning up from the feet) with >20 verts near the
    # midline, which is where the inner THIGHS touch, not the crotch. On a legs-together pose
    # that is far below the pelvis: on character_posed_00003_ it landed at 28% of body height,
    # so the band was measured across the thighs (X width 0.2700) and scaled the donor to
    # ratio 1.1819 -- 18% oversize. An oversized donor rim cannot match the hole cut in the
    # body, and the stitch fans out into long spikes at the vulva.
    # ref_w is itself calibrated at the anchor: on this body the anchor band measures 0.2286
    # against ref_width 0.2284, i.e. ratio 1.0005 -- the donor was authored for this size.
    band = Vt[(Vt[:, 1] > A[1] - 0.05) & (Vt[:, 1] < A[1] + 0.06)]
    ratio = np.ptp(band[:, 0]) / ref_w
    W = ((Vd - Gc) * GS) @ (L.T * ratio) + A
    return W, Fd, A


def _new_obj(V, F, name="m"):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], [list(map(int, f)) for f in F]); me.update()
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    return o, me


def _voxel(o, vox):
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    m = o.modifiers.new("vr", 'REMESH'); m.mode = 'VOXEL'; m.voxel_size = vox; m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="vr")


def _np_verts(me):
    return np.array([v.co for v in me.vertices])


def _adj(F2, mask, pad=0.01, dA=None, lim=None):
    a = defaultdict(set)
    for fl in F2:
        if (dA is None) or (dA[list(fl)].min() < lim + pad):
            n = len(fl)
            for k in range(n):
                x, y = fl[k], fl[(k + 1) % n]; a[x].add(y); a[y].add(x)
    return a


def _graft_cutjoin(ob, W, Fd, A):
    """Cut-and-JOIN graft (KAN-18, 2026-06-24, HEADLESS). Flood-cut the body under the donor (this OPENS the
    vagina/anus recesses instead of the body meshing a wall behind them), stitch the donor OUTER rim to the
    hole by angle (proportional -> cannot twist), weld leftover branch edges, then smooth ONLY a 7mm ring at
    the seam (vulva + surround never touched). Body MUST be watertight at the crotch (gen_seams remesh /
    voxel_body) -- raw meshes are open there and leave gaps. Modifies ob.data in place; returns (ob, Wp)."""
    # ===== GRAFT donor geometry + seam-RING smooth (2026-06-25, user-directed) =====
    # Snapping re-sampled the donor with the body's coarse verts -> jagged interior (spikes by the vagina) and
    # the smooth ate the labia. Instead GRAFT the donor's OWN geometry (= the user's sculpt = clean detail):
    # flood-cut the body under the donor footprint, stitch the donor rim to the hole, then smooth ONLY the
    # seam RING (verts within SEAM_BAND of the rim line, BOTH sides). The inner detail (labia/vagina/anus,
    # farther than SEAM_BAND from the rim) is 100% PROTECTED. Everything below `return ob, W` is DEAD code.
    from collections import defaultdict, deque
    A = np.asarray(A, float); W = np.asarray(W, float); me = ob.data
    Vb = _np_verts(me)
    fn = np.zeros(3)
    for f in Fd:
        a, b, c = W[int(f[0])], W[int(f[1])], W[int(f[2])]; fn = fn + np.cross(b - a, c - a)
    outw = fn / (np.linalg.norm(fn) + 1e-12)
    if outw @ (A - Vb.mean(0)) < 0: outw = -outw
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adj[a].append(b); adj[b].append(a)
    seen = set(); loops = []
    for s in list(adj):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; cur = s
        while True:
            nx = [w for w in adj[cur] if w != prev and w not in seen]
            if not nx: break
            n = nx[0]; lp.append(n); seen.add(n); prev = cur; cur = n
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); loop = loops[0]
    for li in loop:
        o = Vector(W[li].tolist()); ok, cp, nn, ix = ob.closest_point_on_mesh(o)
        if ok: W[li] = np.array(cp.to_tuple())
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in loop])
    def _inp(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bmb = bmesh.new(); bmb.from_mesh(me); bmb.faces.ensure_lookup_table()
    cand = set()
    for f in bmb.faces:
        cc = np.array(f.calc_center_median())
        if np.linalg.norm(cc - A) > 0.12: continue
        dd = cc - A
        if _inp(dd @ u, dd @ v, poly): cand.add(f.index)
    if not cand:
        bmb.free(); print("[cutjoin] WARN: no body faces under donor footprint", flush=True); return ob, W
    seed = min((f for f in bmb.faces if f.index in cand), key=lambda f: np.linalg.norm(np.array(f.calc_center_median()) - A))
    flood = set([seed.index]); dq = deque([seed])
    while dq:
        f = dq.popleft()
        for e in f.edges:
            for nf in e.link_faces:
                if nf.index in cand and nf.index not in flood: flood.add(nf.index); dq.append(nf)
    hb = set()
    for fi in flood:
        for e in bmb.faces[fi].edges:
            if any(lf.index not in flood for lf in e.link_faces): hb.add(e.verts[0]); hb.add(e.verts[1])
    bmesh.ops.delete(bmb, geom=[bmb.faces[fi] for fi in flood], context='FACES')
    dvm = [bmb.verts.new(W[i].tolist()) for i in range(len(W))]; bmb.verts.ensure_lookup_table()
    donset = set(id(x) for x in dvm)
    for f in Fd:
        try: bmb.faces.new([dvm[int(f[0])], dvm[int(f[1])], dvm[int(f[2])]])
        except Exception: pass
    bmb.edges.ensure_lookup_table()
    he = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hb and e.verts[1] in hb]
    hadj = defaultdict(list)
    for e in he: hadj[e.verts[0]].append(e.verts[1]); hadj[e.verts[1]].append(e.verts[0])
    hs = he[0].verts[0]; H = [hs]; prev = None; cur = hs
    while True:
        nx = [w for w in hadj[cur] if w is not prev]
        if not nx: break
        if len(nx) == 1: n = nx[0]
        else:
            cc = np.array(cur.co); din = (cc - np.array(prev.co)) if prev is not None else np.array([1.0, 0, 0])
            din = din / (np.linalg.norm(din) + 1e-9)
            n = min(nx, key=lambda w: -float(np.dot(din, (np.array(w.co) - cc) / (np.linalg.norm(np.array(w.co) - cc) + 1e-9))))
        if n is hs or n in H: break
        H.append(n); prev = cur; cur = n
    R = [dvm[li] for li in loop]
    def _ang(x): p = np.array(x.co) - A; return float(np.arctan2(p @ v, p @ u))
    def _sar(L):
        p = np.array([[(np.array(x.co) - A) @ u, (np.array(x.co) - A) @ v] for x in L])
        return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    if _sar(R) * _sar(H) < 0: H = H[::-1]
    a0 = _ang(R[0]); j0 = int(np.argmin([abs(((_ang(h) - a0 + np.pi) % (2 * np.pi)) - np.pi) for h in H])); H = H[j0:] + H[:j0]
    nR = len(R); nH = len(H); i = j = 0
    for _ in range(nR + nH):
        if i < nR and (j >= nH or (i + 1) / nR <= (j + 1) / nH):
            try: bmb.faces.new([R[i % nR], R[(i + 1) % nR], H[j % nH]])
            except Exception: pass
            i += 1
        else:
            try: bmb.faces.new([H[j % nH], R[i % nR], H[(j + 1) % nH]])
            except Exception: pass
            j += 1
    bmb.edges.ensure_lookup_table()
    rimco = np.array([W[li] for li in loop]); rimv = [dvm[li] for li in loop]; rimset = set(rimv)
    opv = set()
    for e in bmb.edges:
        if len(e.link_faces) == 1:
            for vv in e.verts:
                if vv.is_valid and vv not in rimset and id(vv) not in donset and np.linalg.norm(np.array(vv.co) - A) < 0.12: opv.add(vv)
    tm = {}
    for vv in opv:
        p = np.array(vv.co); k = int(np.argmin(((rimco - p) ** 2).sum(1)))
        if np.linalg.norm(rimco[k] - p) < 0.02: tm[vv] = rimv[k]
    if tm: bmesh.ops.weld_verts(bmb, targetmap=tm)
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    bmb.to_mesh(me); bmb.free(); me.update()
    bm2 = bmesh.new(); bm2.from_mesh(me); bm2.verts.ensure_lookup_table()
    allv = list(bm2.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    # PROTECT labeled detail (outer/inner labia + opening = donor zones 1-4): KD-tree of those donor verts'
    # placed positions so they are NEVER smoothed, even where the rim hugs them (near the anus). User spec:
    # inner parts 100% protected, smoothing happens only OUTWARD of them.
    _detail_kd = None
    try:
        _zn = np.load(A_ZONES)
        if len(_zn) == len(W):
            _dp = W[np.isin(_zn, [1, 2, 3, 4])]
            if len(_dp):
                _detail_kd = kdtree.KDTree(len(_dp))
                for _i in range(len(_dp)): _detail_kd.insert(Vector(_dp[_i].tolist()), _i)
                _detail_kd.balance()
    except Exception as _ze:
        print(f"[cutjoin] WARN: detail-zone protect off ({_ze})", flush=True)
    zone = set(); _prot = 0
    for i in range(len(allv)):
        if dA2[i] > 0.1: continue
        dd = co[i] - A; d2r = float(np.min(np.sqrt(((poly - np.array([dd @ u, dd @ v])) ** 2).sum(1))))
        if d2r < GRAFT_SEAM_BAND:
            if _detail_kd is not None and _detail_kd.find(Vector(co[i].tolist()))[2] < GRAFT_PROTECT_R:
                _prot += 1; continue   # labeled labia/opening detail -> 100% protected, never smoothed
            zone.add(i)   # seam RING (body side + outer margin); detail beyond the band ALSO protected
    print(f"[cutjoin] detail-zone protect: spared {_prot} labeled labia/opening verts from the seam smooth", flush=True)
    nb = {i: set() for i in zone}
    for e in bm2.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in zone: nb[a].add(b)
        if b in zone: nb[b].add(a)
    P = co.copy()
    for _ in range(GRAFT_SEAM_ITERS):
        nP = P.copy()
        for i in zone:
            if nb[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nb[i]], axis=0)
        P = nP
    for i in zone: allv[i].co = Vector(P[i].tolist())
    # gentle MONS de-bump (KAN-18): the pubic mound just above the vulva reads bumpy. Lightly smooth the FRONT
    # skin (donor zone 0) within MONS_R of the vulva centre. Labia (zones 1-4) stay protected; the anus/perineum
    # are further back than MONS_R, untouched. Light (MONS_ITERS x MONS_W) so it de-bumps without flattening.
    if _detail_kd is not None:
        _co2 = np.array([list(x.co) for x in allv])
        _vc = W[np.isin(_zn, [3, 4])].mean(0)            # vulva centre (inner labia + opening)
        _skin = W[_zn == 0]
        _skd = kdtree.KDTree(len(_skin))
        for _i in range(len(_skin)): _skd.insert(Vector(_skin[_i].tolist()), _i)
        _skd.balance()
        mons = set()
        for i in range(len(allv)):
            p = _co2[i]
            if np.linalg.norm(p - _vc) > MONS_R: continue                              # near the vulva only
            if _detail_kd.find(Vector(p.tolist()))[2] < GRAFT_PROTECT_R: continue      # protect labia
            if _skd.find(Vector(p.tolist()))[2] < 0.004: mons.add(i)                   # donor skin (zone 0), not body
        mnb = {i: set() for i in mons}
        for e in bm2.edges:
            a = idx[e.verts[0]]; b = idx[e.verts[1]]
            if a in mons: mnb[a].add(b)
            if b in mons: mnb[b].add(a)
        Pm = _co2.copy()
        for _ in range(MONS_ITERS):
            nPm = Pm.copy()
            for i in mons:
                if mnb[i]: nPm[i] = (1 - MONS_W) * Pm[i] + MONS_W * np.mean([Pm[k] for k in mnb[i]], axis=0)
            Pm = nPm
        for i in mons: allv[i].co = Vector(Pm[i].tolist())
        print(f"[cutjoin] mons de-bump: gently smoothed {len(mons)} pubic-mound skin verts", flush=True)
    bm2.normal_update(); bm2.to_mesh(me); bm2.free()
    for p in me.polygons: p.use_smooth = True
    print(f"[cutjoin] graft + seam-ring smooth: {len(me.vertices)} verts, ring {len(zone)} (inner detail protected)", flush=True)
    return ob, W
    # ===== DEAD (snap / shrinkwrap-morph / cut-bridge) =====
    from collections import defaultdict
    A = np.asarray(A, float); W = np.asarray(W, float); me = ob.data
    # (1) subdivide the genital region
    _bm = bmesh.new(); _bm.from_mesh(me); _bm.faces.ensure_lookup_table()
    _gf = [f for f in _bm.faces if np.linalg.norm(np.array(f.calc_center_median()) - A) < 0.055]
    _ge = list({e for f in _gf for e in f.edges})
    if _ge:
        bmesh.ops.subdivide_edges(_bm, edges=_ge, cuts=SNAP_SUBDIV, use_grid_fill=True)
        bmesh.ops.triangulate(_bm, faces=[f for f in _bm.faces if len(f.verts) > 4])
    _bm.to_mesh(me); _bm.free()
    # (2) controlled snap onto the donor (clamped)
    dme = bpy.data.meshes.new("don"); dme.from_pydata(W.tolist(), [], [list(map(int, f)) for f in Fd]); dme.update()
    don = bpy.data.objects.new("don", dme); bpy.context.scene.collection.objects.link(don)
    V2 = np.array([list(v.co) for v in me.vertices]); dA = np.linalg.norm(V2 - A, axis=1)
    for i in np.where(dA < 0.06)[0]:
        w = float(np.clip((MORPH_R - dA[i]) / MORPH_FALL, 0, 1))
        if w <= 0: continue
        p = Vector(V2[i].tolist()); ok, cp, nn, ix = don.closest_point_on_mesh(p)
        if ok and (cp - p).length < SNAP_CLAMP: me.vertices[i].co = p.lerp(cp, w)
    me.update(); bpy.data.objects.remove(don, do_unlink=True)
    # (3) outward-ring smooth; outward = donor mean normal (robust on any body), donor 2D footprint = in/out
    fn = np.zeros(3)
    for f in Fd:
        a, b, c = W[int(f[0])], W[int(f[1])], W[int(f[2])]; fn = fn + np.cross(b - a, c - a)
    outw = fn / (np.linalg.norm(fn) + 1e-12)
    if outw @ (A - V2.mean(0)) < 0: outw = -outw
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adjr = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adjr[a].append(b); adjr[b].append(a)
    seen = set(); loops = []
    for s in list(adjr):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; cur = s
        while True:
            nx = [ww for ww in adjr[cur] if ww != prev and ww not in seen]
            if not nx: break
            nn2 = nx[0]; lp.append(nn2); seen.add(nn2); prev = cur; cur = nn2
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); rim = loops[0]
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in rim])
    def _inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    allv = list(bm.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    zone = set()
    for i in range(len(allv)):
        if dA2[i] > 0.1: continue
        dd = co[i] - A; px = dd @ u; py = dd @ v
        d2r = float(np.min(np.sqrt(((poly - np.array([px, py])) ** 2).sum(1))))
        inside = _inpoly(px, py, poly)
        if (not inside and d2r < 0.016) or (inside and d2r < 0.010): zone.add(i)  # outward ring + tiny boundary band
    nb = {i: set() for i in zone}
    for e in bm.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in zone: nb[a].add(b)
        if b in zone: nb[b].add(a)
    P = co.copy()
    for _ in range(SNAP_SMOOTH_ITERS):
        nP = P.copy()
        for i in zone:
            if nb[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nb[i]], axis=0)
        P = nP
    for i in zone: allv[i].co = Vector(P[i].tolist())
    bm.normal_update(); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    print(f"[cutjoin] subdivide+snap graft: {len(me.vertices)} verts, ring {len(zone)} (interior protected)", flush=True)
    return ob, W
    # ===== DEAD (old shrinkwrap-morph + cut-bridge) =====
    A = np.asarray(A, float); W = np.asarray(W, float)
    dme = bpy.data.meshes.new("don"); dme.from_pydata(W.tolist(), [], [list(map(int, f)) for f in Fd]); dme.update()
    don = bpy.data.objects.new("don", dme); bpy.context.scene.collection.objects.link(don)
    me = ob.data
    V2 = np.array([list(v.co) for v in me.vertices]); dA = np.linalg.norm(V2 - A, axis=1)
    vg = ob.vertex_groups.new(name="gen")
    for i in range(len(V2)):
        w = float(np.clip((MORPH_R - dA[i]) / MORPH_FALL, 0, 1))   # 1 in the centre -> 0 (soft blend) at MORPH_R
        if w > 0: vg.add([i], w, 'REPLACE')
    sw = ob.modifiers.new("sw", 'SHRINKWRAP'); sw.target = don; sw.vertex_group = "gen"
    # PROJECT (along the body's own normals) NOT nearest-surface: on a coarse body NEAREST folds the surface
    # (adjacent verts snap to non-adjacent donor points) -> lumps/waves; PROJECT is a coherent per-vertex move.
    sw.wrap_method = 'PROJECT'; sw.use_negative_direction = True; sw.use_positive_direction = True
    sw.subsurf_levels = 0; sw.wrap_mode = 'ON_SURFACE'
    bpy.context.view_layer.objects.active = ob
    for x in bpy.context.scene.objects: x.select_set(x == ob)
    bpy.ops.object.modifier_apply(modifier="sw")
    bpy.data.objects.remove(don, do_unlink=True)
    # smooth ONLY the BODY side of the genital boundary ("from the outside") to blend the donor-edge crack,
    # leaving the genital interior (vulva detail) fully untouched. The donor 2D footprint separates in/out.
    from collections import defaultdict
    Vb = _np_verts(ob.data); _cen = Vb.mean(0); outw = A - _cen; outw = outw / (np.linalg.norm(outw) + 1e-12)
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adjr = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adjr[a].append(b); adjr[b].append(a)
    seen = set(); loops = []
    for s in list(adjr):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; curr = s
        while True:
            nx = [ww for ww in adjr[curr] if ww != prev and ww not in seen]
            if not nx: break
            nn = nx[0]; lp.append(nn); seen.add(nn); prev = curr; curr = nn
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); rim = loops[0]
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in rim])
    def _inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    allv = list(bm.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    zone = set()
    for i in range(len(allv)):
        if dA2[i] > 0.085: continue
        d = co[i] - A
        if not _inpoly(d @ u, d @ v, poly): zone.add(i)   # OUTSIDE the genital footprint = body side -> smooth
    nb = {i: set() for i in zone}
    for e in bm.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in zone: nb[a].add(b)
        if b in zone: nb[b].add(a)
    P = co.copy()
    for _ in range(12):
        nP = P.copy()
        for i in zone:
            if nb[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nb[i]], axis=0)
        P = nP
    for i in zone: allv[i].co = Vector(P[i].tolist())
    bm.normal_update(); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    print("[cutjoin] shrinkwrap-morph graft (no bridge -> no shards)", flush=True)
    return ob, W
    # ---- DEAD CODE BELOW (cut+bridge -- shards) ----
    from collections import defaultdict, deque
    A = np.asarray(A, float)
    Vb = _np_verts(ob.data); cen = Vb.mean(0)
    outw = A - cen; outw = outw / (np.linalg.norm(outw) + 1e-12); OW = Vector(outw.tolist())
    t = np.array([1.0, 0, 0])
    if abs(outw @ t) > 0.9: t = np.array([0, 1.0, 0])
    u = np.cross(outw, t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    # donor OUTER rim = the longest boundary loop
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adj[a].append(b); adj[b].append(a)
    seen = set(); loops = []
    for s in list(adj):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; curr = s
        while True:
            nx = [w for w in adj[curr] if w != prev and w not in seen]
            if not nx: break
            n = nx[0]; lp.append(n); seen.add(n); prev = curr; curr = n
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); loop = loops[0]
    # snap the rim onto the body along -outward (near crotch surface, never the opposite thigh)
    W = np.asarray(W, float); Wp = W.copy()
    for li in loop:
        o = Vector(W[li].tolist()); hit, lp2, nr, ix = ob.ray_cast(o, -OW)
        if not hit or (Vector(lp2) - o).length > 0.08: hit, lp2, nr, ix = ob.ray_cast(o, OW)
        if hit and (Vector(lp2) - o).length < 0.08: Wp[li] = np.array(lp2.to_tuple())
        else:
            ok, cp, nn, ix = ob.closest_point_on_mesh(o)
            if ok: Wp[li] = np.array(cp.to_tuple())
    poly = np.array([[Wp[li] @ u, Wp[li] @ v] for li in loop])
    def inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bmb = bmesh.new(); bmb.from_mesh(ob.data); bmb.faces.ensure_lookup_table()
    cand = set()
    for f in bmb.faces:
        c = np.array(f.calc_center_median())
        if np.linalg.norm(c - A) > 0.12: continue
        if abs(np.dot(c - A, outw)) > 0.045: continue
        if inpoly(c @ u, c @ v, poly): cand.add(f.index)
    if not cand:
        bmb.free(); print("[cutjoin] WARN: no body faces under donor footprint", flush=True); return ob, Wp
    seedf = min((f for f in bmb.faces if f.index in cand),
                key=lambda f: np.linalg.norm(np.array(f.calc_center_median()) - A))
    flood = set([seedf.index]); dq = deque([seedf])
    while dq:
        f = dq.popleft()
        for e in f.edges:
            for nf in e.link_faces:
                if nf.index in cand and nf.index not in flood: flood.add(nf.index); dq.append(nf)
    changed = True
    while changed:
        changed = False
        for f in bmb.faces:
            if f.index in flood or np.linalg.norm(np.array(f.calc_center_median()) - A) > 0.12: continue
            nbf = [lf.index for e in f.edges for lf in e.link_faces if lf.index != f.index]
            if nbf and all(x in flood for x in nbf): flood.add(f.index); changed = True
    hole_bverts = set()
    for fi in flood:
        for e in bmb.faces[fi].edges:
            if any(lf.index not in flood for lf in e.link_faces):
                hole_bverts.add(e.verts[0]); hole_bverts.add(e.verts[1])
    bmesh.ops.delete(bmb, geom=[bmb.faces[fi] for fi in flood], context='FACES')
    # RELAX the jagged cut boundary into a smooth, evenly-spaced loop BEFORE bridging -> the bridge to the
    # clean donor rim is regular -> kills the long shard triangles at the source. Tangential Laplacian along
    # the boundary loop (move toward midpoint of the two boundary neighbours), kept on the body surface.
    bmb.edges.ensure_lookup_table()
    he0 = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hole_bverts and e.verts[1] in hole_bverts]
    hnb = defaultdict(list)
    for e in he0: hnb[e.verts[0]].append(e.verts[1]); hnb[e.verts[1]].append(e.verts[0])
    hbv = [hv for hv in hole_bverts if hv.is_valid and len(hnb[hv]) == 2]   # clean loop verts only (skip branches)
    for _ in range(12):
        newp = {}
        for hv in hbv:
            mid = 0.5 * (np.array(hnb[hv][0].co) + np.array(hnb[hv][1].co))
            newp[hv] = 0.5 * np.array(hv.co) + 0.5 * mid
        for hv, p in newp.items():
            ok, cp, nn, ix = ob.closest_point_on_mesh(Vector(p.tolist()))
            hv.co = cp if ok else Vector(p.tolist())         # keep the relaxed vert ON the body surface
    dvm = [bmb.verts.new(Wp[i].tolist()) for i in range(len(Wp))]; bmb.verts.ensure_lookup_table()
    for f in Fd:
        try: bmb.faces.new([dvm[int(f[0])], dvm[int(f[1])], dvm[int(f[2])]])
        except Exception: pass
    bmb.edges.ensure_lookup_table()
    # bridge the body hole loop <-> donor rim (watertight base; its shard triangles get evened by the
    # body-side seam smooth below). Branch-aware hole-loop walk + fraction stitch.
    he = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hole_bverts and e.verts[1] in hole_bverts]
    hadj = defaultdict(list)
    for e in he: hadj[e.verts[0]].append(e.verts[1]); hadj[e.verts[1]].append(e.verts[0])
    hstart = he[0].verts[0]; H = [hstart]; prev = None; curr = hstart
    while True:
        nx = [w for w in hadj[curr] if w is not prev]
        if not nx: break
        if len(nx) == 1: n = nx[0]
        else:
            cc = np.array(curr.co); din = (cc - np.array(prev.co)) if prev is not None else np.array([1.0, 0, 0])
            din = din / (np.linalg.norm(din) + 1e-9)
            n = min(nx, key=lambda w: -float(np.dot(din, (np.array(w.co) - cc) / (np.linalg.norm(np.array(w.co) - cc) + 1e-9))))
        if n is hstart or n in H: break
        H.append(n); prev = curr; curr = n
    R = [dvm[li] for li in loop]
    def sarea(L):
        p = np.array([[(np.array(x.co) - A) @ u, (np.array(x.co) - A) @ v] for x in L])
        return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    if sarea(R) * sarea(H) < 0: H = H[::-1]
    def ang(x):
        p = np.array(x.co) - A; return float(np.arctan2(p @ v, p @ u))
    a0 = ang(R[0]); j0 = int(np.argmin([abs(((ang(h) - a0 + np.pi) % (2 * np.pi)) - np.pi) for h in H])); H = H[j0:] + H[:j0]
    nR = len(R); nH = len(H); i = j = 0
    for _ in range(nR + nH):
        fi = (i + 1) / nR; fj = (j + 1) / nH
        if i < nR and (j >= nH or fi <= fj):
            try: bmb.faces.new([R[i % nR], R[(i + 1) % nR], H[j % nH]])
            except Exception: pass
            i += 1
        else:
            try: bmb.faces.new([H[j % nH], R[i % nR], H[(j + 1) % nH]])
            except Exception: pass
            j += 1
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    rim_co = np.array([Wp[li] for li in loop]); rim_v = [dvm[li] for li in loop]; rimset = set(rim_v)
    bmb.edges.ensure_lookup_table()
    openv = set()
    for e in bmb.edges:
        if len(e.link_faces) == 1:
            for vv in e.verts:
                if vv.is_valid and vv not in rimset and np.linalg.norm(np.array(vv.co) - A) < 0.12: openv.add(vv)
    tmap = {}
    for vv in openv:
        p = np.array(vv.co); k = int(np.argmin(((rim_co - p) ** 2).sum(1)))
        if np.linalg.norm(rim_co[k] - p) < 0.02: tmap[vv] = rim_v[k]
    if tmap: bmesh.ops.weld_verts(bmb, targetmap=tmap)
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    # flip long sliver bridge edges to well-shaped triangles (Delaunay-like) -> removes most shard faces,
    # without moving the surface (edge flips preserve the shape; the genital tessellation just gets cleaner).
    bmb.faces.ensure_lookup_table()
    sf = [f for f in bmb.faces if len(f.verts) == 3 and np.linalg.norm(np.array(f.calc_center_median()) - A) < 0.09]
    if sf:
        se = list({e for f in sf for e in f.edges})
        bmesh.ops.beautify_fill(bmb, faces=sf, edges=se)
        bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    bmb.to_mesh(ob.data); bmb.free(); ob.data.update()
    # BODY-SIDE seam smooth: even out the bridge shards, but ONLY from the rim OUTWARD (~2.5cm onto the
    # body). The genital interior (anything inside the rim radius) and the far surround are PROTECTED, so the
    # vulva/anus detail is never touched. Per-angle rim radius decides inside(genital) vs outside(body).
    bm3 = bmesh.new(); bm3.from_mesh(ob.data); bm3.verts.ensure_lookup_table()
    allv = list(bm3.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv])
    rimpts = np.array([Wp[li] for li in loop])
    rim2d = np.stack([(rimpts - A) @ u, (rimpts - A) @ v], 1)
    rim_ang = np.arctan2(rim2d[:, 1], rim2d[:, 0]); rim_rad = np.linalg.norm(rim2d, axis=1)
    near = np.where(np.linalg.norm(co - A, axis=1) < 0.13)[0]
    seam = set()
    for i in near:
        d = co[i] - A; p2 = np.array([d @ u, d @ v]); r = float(np.linalg.norm(p2))
        th = float(np.arctan2(p2[1], p2[0]))
        Rth = rim_rad[int(np.argmin(np.abs(((rim_ang - th + np.pi) % (2 * np.pi)) - np.pi)))]
        if (r > Rth - 0.004) and (r < Rth + 0.030):     # rim + BODY side only; genital interior protected
            seam.add(int(i))
    nbrs = {i: set() for i in seam}
    for e in bm3.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in seam: nbrs[a].add(b)
        if b in seam: nbrs[b].add(a)
    P = co.copy()
    for _ in range(18):
        nP = P.copy()
        for i in seam:
            if nbrs[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nbrs[i]], axis=0)
        P = nP
    for i in seam: allv[i].co = Vector(P[i].tolist())
    bm3.normal_update(); bm3.to_mesh(ob.data); bm3.free(); ob.data.update()
    print(f"[cutjoin] body-side seam smooth on {len(seam)} verts", flush=True)
    # final: close last micro-gaps at the old branch spot
    bmz = bmesh.new(); bmz.from_mesh(ob.data); bmz.edges.ensure_lookup_table()
    rez = [e for e in bmz.edges if len(e.link_faces) == 1 and np.linalg.norm(np.array((e.verts[0].co + e.verts[1].co) / 2) - A) < 0.12]
    if rez:
        bmesh.ops.holes_fill(bmz, edges=rez, sides=6)
        bmesh.ops.triangulate(bmz, faces=[f for f in bmz.faces if len(f.verts) > 4])
        bmesh.ops.recalc_face_normals(bmz, faces=list(bmz.faces))
    bmz.to_mesh(ob.data); bmz.free(); ob.data.update()
    print(f"[cutjoin] grafted: cut {len(flood)} body faces, rim {len(loop)} verts", flush=True)
    return ob, Wp


def graft_pelvis(Vt, Ft, voxel_body=True, local=False):
    """Vt,Ft = body mesh. Returns (V,F) of the clean grafted genital body.

    local=True (PIPELINE): keep the body at its OWN density (~150k), graft the dense donor with ONLY the
    HC seam-flatten, and SKIP the whole-body fuse + GPU sculpt. The fuse would re-voxel the entire body to
    ~1.2M (violates 'detail only in the genital, lean body'); the body is already the smooth gen_seams
    remesh so it needs no surround sculpt. Result stays ~body+donor faces (<200k)."""
    Vt = np.asarray(Vt, float)   # Ft kept as-is (may be a list of QUADS/mixed faces -> don't triangulate the body)
    # 0) body -> smooth remesh (skip if caller already provides the gen_seams remesh)
    ob, _ = _new_obj(Vt, Ft, "body")
    if voxel_body:
        _voxel(ob, BODY_VOX)
    Vb = _np_verts(ob.data); Fb = [list(p.vertices) for p in ob.data.polygons]
    A = np.array([0.0, 0.0, 0.0])

    # 1) place donor
    W, Fd, A = _place(Vt)
    # ===== CUT-AND-JOIN graft (2026-06-24): replaces the old conform+cut+bridge+HC+fuse+GPU-sculpt below.
    # Headless, no surround smoothing -> the body outside the genital is untouched; vulva/anus detail kept.
    # Everything from here to `return V, F, o` supersedes the (now-dead) block beneath this return. =====
    A = np.asarray(A, float)
    # narrow the donor in X (width) about the midline -> rim sits inboard of the thigh junction (the concave
    # spot where the seam fights) -> cleaner seam, off the thighs. Tune DONOR_X_SCALE.
    W = np.asarray(W, float).copy(); W[:, 0] *= DONOR_X_SCALE
    o, W = _graft_cutjoin(ob, W, Fd, A)
    me = o.data
    for p in me.polygons: p.use_smooth = True
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects: x.select_set(x == o)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    if not local:
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False); bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    V = _np_verts(me).astype(np.float32); F = [list(p.vertices) for p in me.polygons]
    if local:
        cents = np.array([list(p.center) for p in me.polygons], np.float64)
        _gkd = kdtree.KDTree(len(W))
        for _i in range(len(W)): _gkd.insert(Vector(W[_i].tolist()), _i)
        _gkd.balance()
        isgen = np.array([_gkd.find(Vector(c.tolist()))[2] < GENITAL_HUG for c in cents])
        try:
            _gp = os.path.normpath(os.path.join(HERE, "..", "..", "uv_transfer", "_gs_isgenital.npy"))
            np.save(_gp, isgen)
            print(f"[pelvis_graft] genital flag: {int(isgen.sum())}/{len(isgen)} faces -> {_gp}", flush=True)
        except Exception as _e:
            print("[pelvis_graft] genital flag save failed:", _e, flush=True)
    return V, F, o
    # ----- DEAD CODE BELOW (old pg14 cut+bridge+GPU-sculpt; kept for reference, never executed) -----
    anchor = Vector((0.0, float(A[1]), float(A[2])))
    if local and GRADE_RATIO < 1.0:
        # GRADE: decimate the donor's RIM toward the body's density (protect the detailed center) so the
        # donor->body seam has no density jump -> no vertex-normal "border". Headless.
        og, _ = _new_obj(W, Fd, "grade")
        _dAg = np.linalg.norm(W - np.array([0.0, A[1], A[2]]), axis=1)
        _vgg = og.vertex_groups.new(name="g")
        for _i in range(len(W)):
            _gw = float(np.clip((_dAg[_i] - GRADE_KEEP) / GRADE_RAMP, 0, 1))   # 0 center -> 1 rim
            _vgg.add([_i], _gw, 'REPLACE')
        _dm = og.modifiers.new("g", 'DECIMATE'); _dm.decimate_type = 'COLLAPSE'
        _dm.ratio = GRADE_RATIO; _dm.vertex_group = "g"; _dm.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = og
        for x in bpy.context.scene.objects:
            x.select_set(x == og)
        bpy.ops.object.modifier_apply(modifier="g")
        W = _np_verts(og.data); Fd = np.array([list(p.vertices) for p in og.data.polygons], np.int64)
        bpy.data.objects.remove(og, do_unlink=True)
        print(f"[graft] graded donor rim -> {len(W)}v/{len(Fd)}f", flush=True)
    # anus anchor = centroid of the placed donor's BACK midline strip (the perineum/anus feature). Used
    # in step 6 to protect the anus from the seam smooth (the depth-stretch slid it into the smooth zone).
    _mid = W[np.abs(W[:, 0]) < 0.02]
    _back = _mid[_mid[:, 2] >= np.percentile(_mid[:, 2], 72)]
    anus_anchor = np.array([0.0, float(_back[:, 1].mean()), float(_back[:, 2].mean())])
    ec = defaultdict(int)
    for t in Fd:
        for a, b in [(int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    bidx = sorted({v for (a, b), c in ec.items() if c == 1 for v in (a, b)})
    bco = W[bidx]
    dmin = np.sqrt(((W[:, None, :] - bco[None, :, :]) ** 2).sum(-1).min(1))
    # scale the rim-flush band to the DONOR size: a fixed FLUSH_HI over-reaches on a smaller donor and
    # shrinkwraps the LABIA (not just the outer margin) onto the recessed crotch -> crumple. 0.132 = original
    # donor X-width; _dsc<1 for a tighter cut shrinks the band proportionally so only the rim conforms.
    _dsc = min(1.0, float(np.ptp(np.load(A_DONOR)["V"][:, 0]) / 0.132))
    wgt = np.clip((FLUSH_HI * _dsc - dmin) / (FLUSH_W * _dsc), 0, 1)
    # A-fix: also pull the stretched back perineum onto the body (follow the curve), keeping vulva + anus
    backw = np.clip((W[:, 2] - A[2] - BACK_Z0) / BACK_W, 0, 1)           # 0 at vulva -> 1 toward the back
    backw *= np.clip(np.linalg.norm(W - anus_anchor, axis=1) / BACK_ANUS_KEEP, 0, 1)   # keep the anus dimple
    wgt = np.clip(np.maximum(wgt, backw * BACK_STRENGTH), 0, 1)
    # DIRECTIONAL conform (replaces the NEAREST_SURFACEPOINT shrinkwrap): for each weighted donor vert, ray-cast
    # toward the body interior (both ways) and seat it on the NEAREST hit. NEAREST_SURFACEPOINT grabbed the
    # OPPOSITE thigh in the concave between-legs gap and folded the edge into WALLS; a directional cast hits the
    # surface the vert actually sits over, so a TIGHT cut seats flat with no walls + needs no skin margin.
    _bcen = Vector(Vt.mean(0).tolist()); _mwi = ob.matrix_world.inverted(); _m3 = _mwi.to_3x3()
    Wp = W.copy()
    for _i in np.where(wgt > 0.01)[0]:
        _o = Vector(W[_i].tolist()); _dd0 = _bcen - _o
        if _dd0.length < 1e-6:
            continue
        _dd0.normalize(); _ol = _mwi @ _o; _dl = (_m3 @ _dd0).normalized()
        _best = None; _bd = 1e9
        for _s in (1.0, -1.0):
            _hit, _loc, _nrm, _idx = ob.ray_cast(_ol, _dl * _s)
            if _hit:
                _wl = ob.matrix_world @ _loc; _dist = (_wl - _o).length
                if _dist < _bd:
                    _bd = _dist; _best = _wl
        if _best is not None and _bd < 0.05:
            Wp[_i] = np.array((_o.lerp(_best, float(wgt[_i]))).to_tuple())
    od, _ = _new_obj(Wp, Fd, "don")
    Wp = _np_verts(od.data); rim_pts = Wp[bidx]
    bpy.data.objects.remove(od, do_unlink=True); bpy.data.objects.remove(ob, do_unlink=True)

    # 2) cut the body hole just beyond the donor rim (no overlap)
    gcy = Wp[:, 1].mean(); ax, ay = np.ptp(Wp[:, 0]) / 2, np.ptp(Wp[:, 1]) / 2
    cen = np.array([Vb[fl].mean(0) for fl in Fb])
    rm = ((cen[:, 0] / ax) ** 2 + ((cen[:, 1] - gcy) / ay) ** 2 < CUT ** 2) \
        & (cen[:, 2] > Wp[:, 2].min() - 0.02) & (cen[:, 2] < Wp[:, 2].max() + 0.02)
    Fb_keep = [fl for fl, k in zip(Fb, (~rm).tolist()) if k]

    # 3) join + bridge donor rim to body hole
    nb = len(Vb)
    Vall = np.vstack([Vb, Wp])
    Fall = Fb_keep + [[int(t[0] + nb), int(t[1] + nb), int(t[2] + nb)] for t in Fd]
    o, me = _new_obj(Vall, Fall, "m")
    bpy.ops.object.mode_set(mode='EDIT'); bm = bmesh.from_edit_mesh(me); bm.edges.ensure_lookup_table()
    for e in bm.edges:
        e.select = (len(e.link_faces) == 1 and ((e.verts[0].co + e.verts[1].co) * 0.5 - anchor).length < 0.17)
    bmesh.update_edit_mesh(me)
    bpy.ops.mesh.bridge_edge_loops(number_cuts=BRIDGE_CUTS)
    # beautify the seam band (kill slivers)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(me); bm.faces.ensure_lookup_table()
    kd = kdtree.KDTree(len(rim_pts))
    for i, p in enumerate(rim_pts):
        kd.insert(Vector(p.tolist()), i)
    kd.balance()
    for fc in bm.faces:
        _, _, d = kd.find(fc.calc_center_median())
        if d is not None and d < BEAUTY_R + 0.008:
            fc.select = True
    bmesh.update_edit_mesh(me)
    bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.mesh.beautify_fill(angle_limit=3.14159)
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode='OBJECT')

    if os.environ.get("GRAFT_BRIDGE_ONLY"):   # DEBUG: stop right after the bridge to inspect the attachment
        return _np_verts(me), [list(p.vertices) for p in me.polygons], o   # (skips smooth + crash-prone GPU sculpt)

    # 4) HC-Laplacian wide smooth (flatten seam, no moat; protect central vulva)
    V2 = _np_verts(me); F2 = [list(p.vertices) for p in me.polygons]
    Anp = np.array([0.0, A[1], A[2]])
    dAv = np.linalg.norm(V2 - Anp, axis=1); Xv = np.abs(V2[:, 0])
    protect = np.clip((HC_PX - Xv) / HC_PX, 0, 1) * np.clip((HC_PR - dAv) / 0.020, 0, 1)
    edge = np.clip((HC_REG - dAv) / 0.030, 0, 1)
    sm = np.clip((1.0 - protect) * edge, 0, 1); sm[dAv >= HC_REG] = 0.0
    adj = defaultdict(list); seen = defaultdict(set)
    for fl in F2:
        if dAv[list(fl)].min() < HC_REG + 0.01:
            n = len(fl)
            for k in range(n):
                a, b = fl[k], fl[(k + 1) % n]
                if b not in seen[a]: adj[a].append(b); seen[a].add(b)
                if a not in seen[b]: adj[b].append(a); seen[b].add(a)
    active = [int(i) for i in np.where(sm > 0.01)[0]]
    O = V2.copy(); P = V2.copy()
    for _ in range(HC_ITERS):
        Q = P.copy(); Lap = P.copy()
        for i in active:
            nn = adj[i]
            if nn: Lap[i] = np.mean([P[j] for j in nn], axis=0)
        B = np.zeros_like(P)
        for i in active:
            B[i] = Lap[i] - (HC_ALPHA * O[i] + (1 - HC_ALPHA) * Q[i])
        for i in active:
            nn = adj[i]; bn = np.mean([B[j] for j in nn], axis=0) if nn else B[i]
            P[i] = Q[i] + sm[i] * ((Lap[i] - (HC_BETA * B[i] + (1 - HC_BETA) * bn)) - Q[i])
    for i in active:
        me.vertices[i].co = Vector(P[i].tolist())
    me.update()

    if not local:
        # 5) FUSE -> uniform density (eliminates the density border). LOCAL skips it (keeps body lean).
        _voxel(o, FUSE_VOX)
        me = o.data

    # 6) surround smooth = Blender's GPU sculpt mesh_filter (RELAX + SMOOTH) -- the PROVEN smooth (pg14).
    #    Runs for BOTH now: non-local on the fused mesh; LOCAL on the lean non-fused mesh (smooths the
    #    mons/perineum/inner-thigh surround on the body's OWN density). SMOOTH is volume-preserving so it
    #    does NOT crater; the headless numpy Taubin/RELAX welted + striated when pushed, so the surround
    #    smooth stays on the GPU sculpt. Needs a GPU/VIEW_3D context -> WINDOWED Blender (caller wraps it
    #    in a VIEW_3D temp_override).
    me.update()
    V3 = _np_verts(me)
    dA3 = np.linalg.norm(V3 - Anp, axis=1)
    anus_p = np.clip((ANUS_R - np.linalg.norm(V3 - anus_anchor, axis=1)) / ANUS_FALL, 0, 1)  # keep anus dimple
    if local:
        # SEAM-ONLY: smooth a THIN band at the donor->body seam (distance from the donor rim), protecting
        # EVERYTHING else -- the body's detail (buttock cleft, mons) AND the genital interior. Only the
        # graft transition line gets blended. dist-from-rim only for verts near the genital (rest is far).
        _rp = np.asarray(rim_pts, float)
        dmin_rim = np.full(len(V3), 1e9)
        _near = dA3 < 0.14
        if _near.any():
            _Vn = V3[_near]
            dmin_rim[_near] = np.sqrt(((_Vn[:, None, :] - _rp[None, :, :]) ** 2).sum(-1).min(1))
        # back side is -Z in the graft frame (buttocks; +Z is the vulva/front). Use a NARROWER band there so
        # the smooth never reaches up into the gluteal cleft/buttocks (no flattened cleft, no creases).
        back = np.clip((Anp[2] - V3[:, 2]) / 0.03, 0, 1)
        band_eff = np.clip(SEAM_BAND - SEAM_BACK_NARROW * back, 0.004, None)
        fall_eff = SEAM_BAND_FALL - (SEAM_BAND_FALL - SEAM_BACK_FALL) * back   # back: tight falloff (no feather out)
        seam_w = np.clip((band_eff - dmin_rim) / fall_eff, 0, 1)          # 1 on the seam line -> 0 away
        # belly-cut: smooth LESS only on the FRONT belly (+Y above the anchor AND +Z front) -- it must NOT
        # touch the back, where +Y is the buttock seam arc that needs full blending.
        above = np.clip((V3[:, 1] - Anp[1]) / 0.05, 0, 1)
        front = np.clip((V3[:, 2] - Anp[2]) / 0.015, 0, 1)        # +Z = front (vulva side)
        seam_w *= (1.0 - SEAM_BELLY_CUT * above * front)
        # PROTECT the gluteal cleft: midline strip (|X|<CLEFT_W) on the BACK (-Z), but only the UPPER cleft
        # (+Y, above the anchor) -> the natural groove is kept while the LOWER perineum centre still blends.
        cleft_p = (np.clip((CLEFT_W - np.abs(V3[:, 0])) / CLEFT_W, 0, 1)
                   * np.clip((Anp[2] - V3[:, 2]) / 0.015, 0, 1)
                   * np.clip((V3[:, 1] - Anp[1] - 0.005) / 0.02, 0, 1))
        seam_w *= (1.0 - cleft_p)
        mask = 1.0 - seam_w * (1.0 - anus_p)                             # 1 = protect (everything but the line)
    else:
        # standalone: wide surround smooth (matches B40 on the featureless coarse body)
        inner = np.clip((SEAM_INNER - dA3) / INNER_FALL, 0, 1); far = np.clip((dA3 - SEAM_FAR) / FAR_FALL, 0, 1)
        mask = np.clip(1.0 - (1.0 - inner) * (1.0 - far) * (1.0 - anus_p), 0, 1)
    if '.sculpt_mask' not in me.attributes:
        me.attributes.new('.sculpt_mask', 'FLOAT', 'POINT')
    ma = me.attributes['.sculpt_mask']
    for i in range(len(me.vertices)):
        ma.data[i].value = float(mask[i])
    me.update()
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    bpy.ops.object.mode_set(mode='SCULPT')
    for _ in range(RELAX_ITERS):
        bpy.ops.sculpt.mesh_filter(type='RELAX', strength=1.0)
    for _ in range(GRAFT_SEAM_ITERS):
        bpy.ops.sculpt.mesh_filter(type='SMOOTH', strength=1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    # (the headless numpy Taubin/RELAX surround polish was removed -- it welted/striated when pushed;
    #  the surround smooth is the GPU sculpt above, for local too.)

    # 7) final light GLOBAL smooth -> remove residual voxel-surface roughness over the whole
    #    fused region uniformly (broad form survives; fine micro-detail = normal-map bake)
    if FINAL_SMOOTH_ITERS > 0:
        bpy.context.view_layer.objects.active = o
        for x in bpy.context.scene.objects:
            x.select_set(x == o)
        fs = o.modifiers.new("fsmooth", 'SMOOTH'); fs.factor = FINAL_SMOOTH_FAC; fs.iterations = FINAL_SMOOTH_ITERS
        bpy.ops.object.modifier_apply(modifier="fsmooth")
        me = o.data
    for p in me.polygons:
        p.use_smooth = True

    # round-trip cleanly. local=True KEEPS the body's quads (only the donor region is tris) so the body
    # stays lean -- do NOT triangulate the whole mesh (that doubles the 150k body). Standalone -> all tris.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if not local:
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    V = _np_verts(me).astype(np.float32)
    F = [list(p.vertices) for p in me.polygons]   # list (may be mixed quad/tri in local mode)
    if local:
        # save a per-FACE genital flag (centroid within GENITAL_R of the vulva anchor) aligned to THIS
        # mesh's face order, for gen_seams to cut the genital into its own UV island + (later) material.
        # Computed here in the Y-up graft frame (the in_blend rotation that follows preserves face order).
        cents = np.array([list(p.center) for p in me.polygons], np.float64)
        # TIGHT: hug the placed donor surface (the actual genital geometry) + a thin border, NOT a big sphere
        # -> the UV island + slot-2 material cover only the genitals; surrounding skin stays part of the body.
        _gkd = kdtree.KDTree(len(W))
        for _i in range(len(W)):
            _gkd.insert(Vector(W[_i].tolist()), _i)
        _gkd.balance()
        isgen = np.array([_gkd.find(Vector(c.tolist()))[2] < GENITAL_HUG for c in cents])
        try:
            _gp = os.path.normpath(os.path.join(HERE, "..", "..", "uv_transfer", "_gs_isgenital.npy"))
            np.save(_gp, isgen)
            print(f"[pelvis_graft] genital flag: {int(isgen.sum())}/{len(isgen)} faces -> {_gp}", flush=True)
        except Exception as _e:
            print("[pelvis_graft] genital flag save failed:", _e, flush=True)
    return V, F, o


def _argval(flag, default=None):
    a = sys.argv
    if "--" in a:
        a = a[a.index("--") + 1:]
    return a[a.index(flag) + 1] if flag in a else default


def _view3d():
    # GPU sculpt needs a VIEW_3D/region context -> run WINDOWED Blender (not --background). DON'T
    # read_homefile(use_empty=True) -- that wipes the screen. Returns (window, VIEW_3D area, region).
    win = bpy.context.window
    area = next((a for a in (win.screen.areas if (win and win.screen) else []) if a.type == 'VIEW_3D'), None)
    region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
    if area is None:
        raise RuntimeError("no VIEW_3D area -- run WINDOWED Blender (not --background), GPU sculpt needs it")
    return win, area, region


if __name__ == "__main__":
    import traceback
    from mathutils import Matrix
    from math import radians
    in_blend = _argval("--in_blend")   # PIPELINE mode: graft the genital into an existing (remeshed) blend
    body = _argval("--body"); out = _argval("--out"); blend = _argval("--blend")
    vox_body = "--voxel_body" in (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv)
    try:
        if in_blend:
            # PIPELINE: graft the high-detail genital onto the ALREADY-REMESHED body in `in_blend`
            # (gen_seams' last_seams.blend), in place, BEFORE gen_seams UVs it -> the genital keeps its
            # full fuse detail (it is never re-remeshed). We APPEND the body mesh into the windowed STARTUP
            # scene (NOT open_mainfile -- that drops the VIEW_3D the GPU sculpt needs), graft, then save it
            # back as the SAME object name ('geometry_0') with identity matrix so gen_seams still finds it.
            for ob in list(bpy.data.objects):          # clear startup default cube/camera/light
                bpy.data.objects.remove(ob, do_unlink=True)
            with bpy.data.libraries.load(in_blend) as (_df, _dt):
                _dt.objects = list(_df.objects)
            src = None
            for _ob in _dt.objects:
                if _ob is None:
                    continue
                bpy.context.scene.collection.objects.link(_ob)
                if _ob.type == 'MESH' and (src is None or len(_ob.data.polygons) > len(src.data.polygons)):
                    src = _ob
            if src is None:
                raise RuntimeError("no mesh object in " + in_blend)
            name = src.name
            mw = src.matrix_world
            R = np.array([[mw[r][c] for c in range(3)] for r in range(3)], float)
            T = np.array([mw[r][3] for r in range(3)], float)
            # Extract world V + faces AS-IS (KEEP the body's quads -- triangulating here would double the
            # ~150k body before we even graft). graft_pelvis(local=True) keeps the body quads.
            nv = len(src.data.vertices); _co = np.empty(nv * 3, np.float64)
            src.data.vertices.foreach_get('co', _co)
            Vw = _co.reshape(nv, 3) @ R.T + T
            Fw = [list(p.vertices) for p in src.data.polygons]
            # The pipeline body comes out of the OBJ import Z-UP, but graft_pelvis/_place are tuned for the
            # Y-UP trimesh frame (handgraft_out). If Z is the height axis, rotate -90deg about X into Y-up
            # for the graft, then rotate the RESULT back +90deg about X before saving (gen_seams' frame).
            _zup = float(np.ptp(Vw[:, 2])) > float(np.ptp(Vw[:, 1]))
            if _zup:
                Vw = Vw @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # -90 about X: (x,y,z)->(x,z,-y)
            # cut-and-join graft is fully HEADLESS (no GPU sculpt) -> no VIEW_3D / windowed Blender needed.
            V, F, o = graft_pelvis(Vw, Fw, voxel_body=False, local=True)
            print(f"[pelvis_graft] done (pipeline): {len(V)} verts, {len(F)} faces", flush=True)
            for x in list(bpy.data.objects):           # keep ONLY the grafted result, as 'geometry_0'
                if x is not o:
                    bpy.data.objects.remove(x, do_unlink=True)
            for _lib in list(bpy.data.libraries):      # drop the appended-from library -> can overwrite the file
                bpy.data.libraries.remove(_lib)
            if _zup:                                    # rotate the grafted result back to the pipeline's Z-up frame
                o.data.transform(Matrix.Rotation(radians(90), 4, 'X'))   # +90 about X: (x,y,z)->(x,-z,y)
            o.name = name; o.matrix_world = Matrix.Identity(4)
            bpy.ops.wm.save_as_mainfile(filepath=in_blend)
            # GENITAL TEXTURING (approach A) runs as a SEPARATE HEADLESS launch (genital_texture.py) after this
            # -- it doesn't need the GPU, and the windowed graft session is too unstable for its unwrap.
        else:
            for ob in list(bpy.data.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            d = np.load(body)
            V, F, o = graft_pelvis(d["V"], d["F"], voxel_body=vox_body)
            print(f"[pelvis_graft] done: {len(V)} verts, {len(F)} faces", flush=True)
            if out:
                np.savez(out, V=np.asarray(V, np.float32), F=np.asarray(F, np.int64))
            if blend:
                bpy.ops.wm.save_as_mainfile(filepath=blend)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)   # quit the windowed Blender so the subprocess returns (no lingering window/prompt)
