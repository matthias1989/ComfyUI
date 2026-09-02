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
A_PLACE = os.path.join(HERE, "assets", "pelvis_place_fixed.npz")

# ---- recipe parameters (validated B38/B39/B40) ----
GS         = float(os.environ.get("PELVIS_GS", "1.10"))  # genital scale (1.0=anatomical, 1.10=+10%)
BODY_VOX   = 0.0028    # body voxel (matches gen_seams target density) -- only if voxel_body
FLUSH_HI   = 0.032     # flush the outer margin: weight 1 for dmin<FLUSH_LO ...
FLUSH_W    = 0.014     #   ... ramping to 0 by FLUSH_HI (dmin = dist from donor rim)
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
SEAM_INNER, SEAM_FAR = 0.028, 0.130   # GPU-sculpt mask: protect vulva (<INNER) + far body (>FAR)
RELAX_ITERS, SEAM_ITERS = 60, 200     # mesh_filter RELAX then SMOOTH counts -> matches B40 (pg_test13)
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
    dd = np.load(A_DONOR); Vd = dd["V"].astype(float); Fd = dd["F"].astype(int)
    lj = _leg_junction_Y(Vt)
    band = Vt[(Vt[:, 1] > lj - 0.05) & (Vt[:, 1] < lj + 0.06)]
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


def graft_pelvis(Vt, Ft, voxel_body=True):
    """Vt,Ft = body mesh. Returns (V,F) of the clean grafted genital body."""
    Vt = np.asarray(Vt, float); Ft = np.asarray(Ft, int)
    # 0) body -> smooth remesh (skip if caller already provides the gen_seams remesh)
    ob, _ = _new_obj(Vt, Ft, "body")
    if voxel_body:
        _voxel(ob, BODY_VOX)
    Vb = _np_verts(ob.data); Fb = [list(p.vertices) for p in ob.data.polygons]
    A = np.array([0.0, 0.0, 0.0])

    # 1) place donor + flush the outer margin onto the body (shrinkwrap by dist-from-rim weight)
    W, Fd, A = _place(Vt)
    anchor = Vector((0.0, float(A[1]), float(A[2])))
    ec = defaultdict(int)
    for t in Fd:
        for a, b in [(int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    bidx = sorted({v for (a, b), c in ec.items() if c == 1 for v in (a, b)})
    bco = W[bidx]
    dmin = np.sqrt(((W[:, None, :] - bco[None, :, :]) ** 2).sum(-1).min(1))
    wgt = np.clip((FLUSH_HI - dmin) / FLUSH_W, 0, 1)
    od, _ = _new_obj(W, Fd, "don")
    vg = od.vertex_groups.new(name="blend")
    for i, w in enumerate(wgt.tolist()):
        if w > 0:
            vg.add([i], float(w), 'REPLACE')
    sw = od.modifiers.new("sw", 'SHRINKWRAP'); sw.target = ob; sw.vertex_group = "blend"
    sw.wrap_method = 'NEAREST_SURFACEPOINT'; sw.wrap_mode = 'ON_SURFACE'
    bpy.context.view_layer.objects.active = od
    for x in bpy.context.scene.objects:
        x.select_set(x == od)
    bpy.ops.object.modifier_apply(modifier="sw")
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

    # 5) FUSE -> uniform density (eliminates the density border)
    _voxel(o, FUSE_VOX)
    me = o.data

    # 6) seam + surround smooth = Blender's GPU sculpt mesh_filter (RELAX + SMOOTH) -- the PROVEN
    #    pg_test7 method. SMOOTH is volume-preserving so it flattens the fusion crease AND the body-
    #    voxel waviness over the wide surround WITHOUT caving it into a crater (headless Laplacian
    #    craters; LaplacianSmooth-volume-preserve under-smooths). Needs a GPU/VIEW_3D context, so
    #    pelvis_graft runs in a WINDOWED Blender (user accepted this; node launches without --background).
    #    The caller wraps this in a VIEW_3D temp_override so the ops have context.
    me.update()
    V3 = _np_verts(me)
    dA3 = np.linalg.norm(V3 - Anp, axis=1)
    inner = np.clip((SEAM_INNER - dA3) / 0.008, 0, 1); far = np.clip((dA3 - SEAM_FAR) / 0.025, 0, 1)
    mask = np.clip(1.0 - (1.0 - inner) * (1.0 - far), 0, 1)   # 1 = protect (vulva + far body), 0 = smooth
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
    for _ in range(SEAM_ITERS):
        bpy.ops.sculpt.mesh_filter(type='SMOOTH', strength=1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data

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

    # triangulate so the result round-trips cleanly as a triangle mesh (trimesh / FBX)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    V = _np_verts(me).astype(np.float32)
    F = np.array([list(p.vertices) for p in me.polygons], dtype=np.int64)
    return V, F, o


def _argval(flag, default=None):
    a = sys.argv
    if "--" in a:
        a = a[a.index("--") + 1:]
    return a[a.index(flag) + 1] if flag in a else default


if __name__ == "__main__":
    import traceback
    body = _argval("--body"); out = _argval("--out"); blend = _argval("--blend")
    vox_body = "--voxel_body" in (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv)
    try:
        # Run WINDOWED (node launches without --background) so the GPU sculpt works. DON'T
        # read_homefile(use_empty=True) -- that wipes the screen -> no VIEW_3D. Keep the default
        # startup screen and just clear its objects.
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        d = np.load(body)
        win = bpy.context.window
        area = next((a for a in (win.screen.areas if (win and win.screen) else []) if a.type == 'VIEW_3D'), None)
        region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
        if area is None:
            raise RuntimeError("no VIEW_3D area -- run WINDOWED Blender (not --background), GPU sculpt needs it")
        with bpy.context.temp_override(window=win, area=area, region=region):
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
