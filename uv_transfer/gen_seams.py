"""
Anatomical UV seams for voxel-remeshed humanoid meshes.

Ring cuts use face-centroid crossing detection: an edge is a ring seam edge
iff its two adjacent face centroids lie on opposite sides of the cut plane.
On a clean manifold mesh (produced by Blender Voxel Remesh) every such edge
belongs to a complete closed loop, so ALL crossing edges form a valid UV
barrier with no Dijkstra required.

Length seams (arms/legs/torso) still use per-component Dijkstra paths to
unroll the cylinders.
"""
import bpy, bmesh, heapq, sys, os
import numpy as np
from collections import defaultdict, deque

# ── Mesh ─────────────────────────────────────────────────────────────────────
obj = bpy.data.objects.get('geometry_0')
if obj is None:
    obj = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
if obj is None: print("ERROR: no mesh found"); sys.exit(1)
print(f"Using mesh object: '{obj.name}'")
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data

n_verts = len(me.vertices)
co_flat = np.empty(n_verts * 3, dtype=np.float64)
me.vertices.foreach_get('co', co_flat)
co = co_flat.reshape(n_verts, 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
T = np.array([mat[r][3] for r in range(3)], dtype=np.float64)
co = co @ R.T + T

# ── FACE BRIDGE pass A (KAN-9) ────────────────────────────────────────────────
# Blender lacks MediaPipe/nvdiffrast, but ComfyUI's python_embeded has both. So we
# export the mesh here, let python_embeded detect the face region (mesh-only MediaPipe),
# and read it back below to drive the face UV chart (and later the hairline) ROBUSTLY
# per-character — replacing the fragile geometric thresholds. Export world verts (Y-up,
# pre-normalization) + triangle faces in me.polygons order so the per-face mask maps 1:1.
if os.environ.get('GS_FACE_EXPORT'):
    _fb = os.path.join(os.path.dirname(__file__), '_gs_facemesh.npz')
    _tris = np.array([tuple(p.vertices)[:3] for p in me.polygons], dtype=np.int32)
    np.savez(_fb, verts=co.astype('float32'), tris=_tris)
    print(f"[face-bridge] pass A: exported {len(co)} verts, {len(_tris)} faces -> {_fb}")
    sys.exit(0)

# ── Auto-normalize to calibrated coordinate space ────────────────────────────
# gen_seams cut positions were tuned for: Z in [-0.41, ~0.50], X in [-0.46, 0.46].
_z_min  = float(co[:, 2].min()); _z_max = float(co[:, 2].max())
_x_ctr  = float((co[:, 0].min() + co[:, 0].max()) / 2.0)
_x_span = float(co[:, 0].max() - co[:, 0].min())
_z_scale = 0.91 / (_z_max - _z_min)
_z_off   = -0.41 - _z_min * _z_scale
_x_scale = 0.92 / _x_span
co[:, 2] = co[:, 2] * _z_scale + _z_off
co[:, 0] = (co[:, 0] - _x_ctr) * _x_scale
co[:, 1] = co[:, 1] * _z_scale
print(f"[gen_seams] Normalized: Z=[{co[:,2].min():.3f},{co[:,2].max():.3f}]  "
      f"X=[{co[:,0].min():.3f},{co[:,0].max():.3f}]")

n_edges = len(me.edges)
ev_flat = np.empty(n_edges * 2, dtype=np.int32)
me.edges.foreach_get('vertices', ev_flat)
EV = ev_flat.reshape(n_edges, 2)
ELEN = np.sqrt(((co[EV[:, 0]] - co[EV[:, 1]]) ** 2).sum(axis=1))

xc = 0.0
print(f"Mesh: {n_verts} verts, {n_edges} edges, {len(me.polygons)} faces")

# ── Build face-centroid arrays and edge→faces map ─────────────────────────────
n_faces = len(me.polygons)
n_loops = len(me.loops)

# Read loop data via foreach_get (these attributes exist on MeshLoop)
lv_flat = np.empty(n_loops, dtype=np.int32)  # vertex index per loop
le_flat = np.empty(n_loops, dtype=np.int32)  # edge index per loop
me.loops.foreach_get('vertex_index', lv_flat)
me.loops.foreach_get('edge_index',   le_flat)

# Polygon → loop range (loop_start + loop_total define the face's loops)
loop_starts = np.empty(n_faces, dtype=np.int32)
loop_totals = np.empty(n_faces, dtype=np.int32)
me.polygons.foreach_get('loop_start', loop_starts)
me.polygons.foreach_get('loop_total', loop_totals)

# Face centroids in normalized coordinate space.
# Computed from co[] (already world-transformed + normalized) via loop vertices,
# avoiding foreach_get('center',...) which fails for computed RNA properties.
face_centroid = np.zeros((n_faces, 3), dtype=np.float64)
for fidx in range(n_faces):
    ls, lt = int(loop_starts[fidx]), int(loop_totals[fidx])
    face_centroid[fidx] = co[lv_flat[ls:ls + lt]].mean(axis=0)
face_z = face_centroid[:, 2]
face_x = face_centroid[:, 0]

def _tdbg(tag):   # diagnostic trace (face-plane vs side-scalp hair per step); set GS_TDBG=1 to enable
    if not os.environ.get('GS_TDBG'):
        return
    try:
        _f = is_hair & (face_normal[:, 1] < -0.40) & (face_z > 0.38) & (face_z < 0.47) & (np.abs(face_x) < 0.10)
        _s = is_hair & (face_z > 0.40) & (face_z < 0.50) & (np.abs(face_x) > 0.06) & (np.abs(face_x) < 0.13)
        print(f"[TDBG {tag:14s}] FACEplane-hair={int(_f.sum()):5d}  SIDEscalp-hair={int(_s.sum()):5d}")
    except NameError:
        pass

# ── BRIDGE pass A: export face centroids (gen normalized space) and stop ───────
# The real hairline lives in the texturing's Canny detector (needs cv2/scipy,
# which Blender lacks).  So we export centroids here, let python_embeded compute
# the per-face hair classification from that detector, then read it back below.
if os.environ.get('GS_EXPORT_CEN'):
    _cen_path = os.path.join(os.path.dirname(__file__), '_gs_cen.npy')
    np.save(_cen_path, face_centroid.astype('float32'))
    print(f"[bridge] exported {n_faces} face centroids -> {_cen_path}")
    sys.exit(0)

# ── KAN-8 ear-band diagnostic (print-only; remove once cut planes are set) ────
# Locates the ear protrusion in the calibrated space so the ear ring-cut plane
# can be placed precisely.  Ears = lateral (high |X|) faces in the head z-band.
# Head z-band runs from the chin seam (0.385) to the crown (~0.50).
_kan8_head = (face_z >= 0.385) & (face_z <= 0.50)
if int(_kan8_head.sum()) > 0:
    _kan8_hx = np.abs(face_x[_kan8_head])
    print("[KAN8-ear] head-band faces (z in [0.385,0.50]): "
          f"{int(_kan8_head.sum())}  |X| "
          + "  ".join(f"p{p}={np.percentile(_kan8_hx, p):.3f}"
                      for p in (50, 75, 90, 95, 99, 100)))
    print("[KAN8-ear] per z-slice |X| extent (find ear-root x and ear z-band):")
    for _zl, _zh in [(0.385, 0.40), (0.40, 0.42), (0.42, 0.44),
                     (0.44, 0.46), (0.46, 0.50)]:
        _m = _kan8_head & (face_z >= _zl) & (face_z < _zh)
        if int(_m.sum()) == 0:
            print(f"   z=[{_zl:.3f},{_zh:.3f}) faces=0"); continue
        _ax = np.abs(face_x[_m])
        print(f"   z=[{_zl:.3f},{_zh:.3f}) faces={int(_m.sum()):5d}  "
              f"|X|max={_ax.max():.3f}  p90={np.percentile(_ax, 90):.3f}  "
              f"p99={np.percentile(_ax, 99):.3f}  frac|X|>0.10={np.mean(_ax > 0.10):.3f}")
    # Full head front/back (Y) extent — needed to place a coronal behind-ears cut.
    _face_y = face_centroid[:, 1]
    _hy = _face_y[_kan8_head]
    print(f"[KAN8-ear] head-band Y (front=low, back=high): "
          f"min={_hy.min():.3f}  p25={np.percentile(_hy,25):.3f}  "
          f"mid={np.percentile(_hy,50):.3f}  p75={np.percentile(_hy,75):.3f}  "
          f"max={_hy.max():.3f}")
    # Ear-root probe: lateral faces beyond the head core, with FULL 3-D bbox
    # (X/Y/Z) so the cut planes can be placed precisely.
    for _thr in (0.06, 0.07, 0.085):
        _ear = _kan8_head & (np.abs(face_x) > _thr)
        if int(_ear.sum()) > 0:
            _ez = face_z[_ear]; _ey = _face_y[_ear]; _ex = np.abs(face_x[_ear])
            print(f"[KAN8-ear] lateral |X|>{_thr:.3f}: faces={int(_ear.sum()):5d}  "
                  f"x=[{_ex.min():.3f},{_ex.max():.3f}]  "
                  f"y=[{_ey.min():.3f},{_ey.max():.3f}]  "
                  f"z=[{_ez.min():.3f},{_ez.max():.3f}]")
        else:
            print(f"[KAN8-ear] lateral |X|>{_thr:.3f}: faces=0")
    # Per-side ear Y-center (left x<0 vs right x>0) at the ear z-band — the
    # coronal seam should sit just BEHIND these (toward higher Y).
    for _side, _sel in (("L", face_x < -0.06), ("R", face_x > 0.06)):
        _m = _kan8_head & _sel & (face_z >= 0.42) & (face_z <= 0.47)
        if int(_m.sum()) > 0:
            _yy = _face_y[_m]
            print(f"[KAN8-ear] {_side} ear-band Y: faces={int(_m.sum()):4d}  "
                  f"y=[{_yy.min():.3f},{_yy.max():.3f}]  ymid={np.percentile(_yy,50):.3f}")
        else:
            print(f"[KAN8-ear] {_side} ear-band Y: faces=0")
else:
    print("[KAN8-ear] no head-band faces found (z calibration off?)")
# ── end KAN-8 diagnostic ──────────────────────────────────────────────────────

# Edge → face index map (using loop edge_index, not poly.edge_indices which doesn't exist)
edge_faces = defaultdict(list)
for fidx in range(n_faces):
    ls, lt = int(loop_starts[fidx]), int(loop_totals[fidx])
    for eidx in le_flat[ls:ls + lt].tolist():
        edge_faces[eidx].append(fidx)

def _edge_key(eidx):
    v0, v1 = me.edges[eidx].vertices
    return (min(v0, v1), max(v0, v1))

# ── Face-centroid ring cut — Z plane ─────────────────────────────────────────
def face_cut_z(label, z_tgt, x_lo=-9.0, x_hi=9.0):
    """Mark every manifold edge whose two adjacent face centroids straddle z_tgt.
    Both face centroids must also fall in [x_lo, x_hi] (body-part filter).
    On a clean manifold mesh these edges form one or more complete closed loops
    that reliably separate UV islands."""
    s = set()
    for eidx, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        f0, f1 = faces
        # Must be on opposite sides of cut plane
        if (face_z[f0] < z_tgt) == (face_z[f1] < z_tgt):
            continue
        # Both face centroids must be in the x filter range
        if not (x_lo <= face_x[f0] <= x_hi and x_lo <= face_x[f1] <= x_hi):
            continue
        s.add(_edge_key(eidx))
    print(f"  {label}: {len(s)} face-crossing seam edges (z={z_tgt:.3f})")
    return s

# ── Face-centroid ring cut — X plane ─────────────────────────────────────────
def face_cut_x(label, x_tgt, z_lo, z_hi):
    """Same as face_cut_z but for X-plane ring cuts (shoulder / wrist)."""
    s = set()
    for eidx, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        f0, f1 = faces
        if (face_x[f0] < x_tgt) == (face_x[f1] < x_tgt):
            continue
        # Both face centroids must be in the z filter range
        if not (z_lo <= face_z[f0] <= z_hi and z_lo <= face_z[f1] <= z_hi):
            continue
        s.add(_edge_key(eidx))
    print(f"  {label}: {len(s)} face-crossing seam edges (x={x_tgt:.3f})")
    return s

# ── Boxed X-plane cut — localized to a Y/Z box (for isolating ears) ───────────
def face_cut_x_box(label, x_tgt, y_lo, y_hi, z_lo, z_hi):
    """Like face_cut_x but additionally requires both face centroids to fall in
    the [y_lo,y_hi] × [z_lo,z_hi] box.  Used to ring a small lateral protrusion
    (the ear) without slicing the whole side of the head.  On a manifold mesh the
    ear cap is a topological disk, so the straddling edges close into a loop
    around its root → the ear becomes its own UV island."""
    s = set()
    for eidx, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        f0, f1 = faces
        if (face_x[f0] < x_tgt) == (face_x[f1] < x_tgt):
            continue
        fy0 = face_centroid[f0, 1]; fy1 = face_centroid[f1, 1]
        if not (y_lo <= fy0 <= y_hi and y_lo <= fy1 <= y_hi):
            continue
        if not (z_lo <= face_z[f0] <= z_hi and z_lo <= face_z[f1] <= z_hi):
            continue
        s.add(_edge_key(eidx))
    print(f"  {label}: {len(s)} face-crossing seam edges "
          f"(x={x_tgt:.3f}, y=[{y_lo:.3f},{y_hi:.3f}], z=[{z_lo:.3f},{z_hi:.3f}])")
    return s

# ── Dijkstra (used by length seams) ──────────────────────────────────────────
def dijkstra(adj, start, goal_set):
    """Shortest path from start to any vertex in goal_set."""
    if start not in adj: return None
    dist = {start: 0.0}; prev = {}; heap = [(0.0, start)]
    while heap:
        if len(heap) > 500000: print("  WARN heap too large"); return None
        d, u = heapq.heappop(heap)
        if u in goal_set:
            path = []; cur = u
            while cur in prev: path.append(cur); cur = prev[cur]
            path.append(start); return list(reversed(path))
        if d > dist.get(u, 1e18): continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u; heapq.heappush(heap, (nd, v))
    return None

def pe(path):
    if not path: return set()
    return {(min(a, b), max(a, b)) for a, b in zip(path, path[1:])}

# ── Dijkstra length seam helpers ──────────────────────────────────────────────
def bfs_component(start, adj):
    visited = {start}; q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            vv = v[0] if isinstance(v, tuple) else v
            if vv not in visited: visited.add(vv); q.append(vv)
    return visited

def build_adj(idx_arr, pen=None):
    vmask = np.zeros(n_verts, dtype=bool); vmask[idx_arr] = True
    keep = vmask[EV[:, 0]] & vmask[EV[:, 1]]
    e_s = EV[keep]; w_s = ELEN[keep]
    adj = defaultdict(list); pen = pen or {}
    for i0, i1, w in zip(e_s[:, 0].tolist(), e_s[:, 1].tolist(), w_s.tolist()):
        adj[i0].append((i1, w + pen.get(i1, 0.0)))
        adj[i1].append((i0, w + pen.get(i0, 0.0)))
    return adj

def low_y_pen(idx_arr):
    ys = co[idx_arr, 1]; y_min = float(ys.min())
    return {int(i): float(co[i, 1] - y_min) * 4.0 for i in idx_arr}

def high_y_pen(idx_arr):
    """Penalise LOW Y (front of character) so Dijkstra routes through HIGH Y (back).
    After GLTF import into Blender the character's front face sits at NEGATIVE / LOW Y
    because GLTF +Z (toward viewer) maps to Blender -Y.  low_y_pen therefore routes
    through the FRONT.  This function inverts the bias to route through the BACK."""
    ys = co[idx_arr, 1]; y_max = float(ys.max())
    return {int(i): float(y_max - co[i, 1]) * 4.0 for i in idx_arr}

def back_center_pen(idx_arr, x_center=0.0, y_weight=4.0, x_weight=20.0):
    """Combined penalty: penalise FRONT (low Y) + X deviation from centerline.
    The y_weight keeps the path on the back surface; the x_weight prevents the
    Dijkstra path from drifting sideways by strongly favouring vertices near x=x_center."""
    ys = co[idx_arr, 1]; y_max = float(ys.max())
    result = {}
    for i in idx_arr:
        ii = int(i)
        y_pen = float(y_max - co[ii, 1]) * y_weight
        x_pen = float(abs(co[ii, 0] - x_center)) * x_weight
        result[ii] = y_pen + x_pen
    return result

def low_z_pen(idx_arr):
    zs = co[idx_arr, 2]; z_min = float(zs.min())
    return {int(i): float(co[i, 2] - z_min) * 4.0 for i in idx_arr}

def all_components(idx_arr, adj, min_size=8):
    processed = set(); comps = []
    for v in idx_arr.tolist():
        if v in processed or v not in adj: processed.add(v); continue
        comp = bfs_component(v, adj); processed |= comp
        if len(comp) >= min_size: comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps

def seam_per_component_x(label, comps, x_arm_max, x_arm_min):
    arm_span = x_arm_max - x_arm_min
    tol = max(0.015, arm_span * 0.03)
    all_s = set()
    for comp in comps:
        ca = np.array(list(comp)); xv = co[ca, 0]
        x_hi = float(xv.max()); x_lo = float(xv.min())
        if (x_hi - x_lo) < 0.05: continue
        sv = sorted([v for v in comp if co[v, 0] >= x_arm_max - tol], key=lambda i: co[i, 2])
        if not sv:
            sv = sorted([v for v in comp if co[v, 0] >= x_hi - tol], key=lambda i: co[i, 2])
        if not sv: continue
        gv = {v for v in comp if co[v, 0] <= x_arm_min + tol}
        if not gv:
            gv = {v for v in comp if co[v, 0] <= x_lo + tol}
        if not gv: continue
        r = pe(dijkstra(adj_g, sv[0], gv))
        if len(r) > 0: all_s |= r
    return all_s

def seam_per_component_z(label, comps, z0, z1):
    all_s = set()
    for comp in comps:
        ca = np.array(list(comp)); zv = co[ca, 2]
        z_top = float(zv.max()); z_bot = float(zv.min())
        z_thresh_top = z_top - max(0.05, (z_top - z_bot) * 0.10)
        # Sort top vertices by Y descending so sv[-1] is highest-Y = BACK of character.
        # (After GLTF→Blender import the front of the character is at LOW Y, back at HIGH Y.)
        # Using sv[-1] starts Dijkstra from the back; combined with high_y_pen it stays there.
        sv = sorted([v for v in comp if co[v, 2] >= z_thresh_top], key=lambda i: co[i, 1])
        z_thresh_bot = z_bot + max(0.05, (z_top - z_bot) * 0.10)
        gv = {v for v in comp if co[v, 2] <= z_thresh_bot}
        if not sv or not gv: continue
        r = pe(dijkstra(adj_g, sv[-1], gv))  # sv[-1] = highest-Y = back of character
        all_s |= r
    return all_s

def length_seam(label, x0, x1, z_lo, z_hi):
    """Arm length seam: Dijkstra from shoulder-end to wrist-end (underside routing)."""
    global adj_g
    x_min, x_max = min(x0, x1), max(x0, x1)
    idx = np.where((co[:, 0] >= x_min - 0.01) & (co[:, 0] <= x_max + 0.01) &
                   (co[:, 2] >= z_lo) & (co[:, 2] <= z_hi))[0]
    if len(idx) < 2: print(f"  {label}: no verts"); return set()
    x_arm_max = float(co[idx, 0].max())
    x_arm_min = float(co[idx, 0].min())
    print(f"  {label}: {len(idx)} verts, X=[{x_arm_min:.3f},{x_arm_max:.3f}]")
    pen = low_z_pen(idx); adj_g = build_adj(idx, pen)
    comps = all_components(idx, adj_g)
    print(f"  {label}: {len(comps)} components")
    r = seam_per_component_x(label, comps, x_arm_max, x_arm_min)
    print(f"  {label}: -> {len(r)} edges"); return r

def back_seam(label, z0, z1, x_lo, x_hi):
    """Leg/torso/head back seam: Dijkstra from top to bottom routing through HIGH Y (back)
    and staying near the X centerline.
    After GLTF→Blender import the character's back (spine/occiput) is at HIGH Y.
    back_center_pen penalises LOW Y (front) AND sideways X deviation so the path
    stays on the back surface along the midline and does not drift sideways."""
    global adj_g
    z_lo_f, z_hi_f = min(z0, z1) - 0.06, max(z0, z1) + 0.06
    idx = np.where((co[:, 0] >= x_lo) & (co[:, 0] <= x_hi) &
                   (co[:, 2] >= z_lo_f) & (co[:, 2] <= z_hi_f))[0]
    if len(idx) < 2: print(f"  {label}: no verts"); return set()
    x_center = float((x_lo + x_hi) / 2.0)
    pen = back_center_pen(idx, x_center=x_center); adj_g = build_adj(idx, pen)
    comps = all_components(idx, adj_g)
    print(f"  {label}: {len(idx)} verts, {len(comps)} components")
    r = seam_per_component_z(label, comps, z0, z1)
    print(f"  {label}: -> {len(r)} edges"); return r

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Computing anatomical seams ===")
all_seam = set()
_struct_seams = set()  # limb/ear rings that must stay COMPLETE even inside hair
_ear_seams = set()     # ear ring-cuts (kept); all other body cuts are dropped below

# ── RING CUTS (face-centroid crossing — guaranteed UV barriers on manifold mesh) ─

# Chin seam — ring just BELOW the chin separating the head/face island from the neck.
# z=0.385: above arm z_hi=0.38 so full-ring cut (no x filter) is safe.
# DISABLED (full ring wraps the back head/neck junction through the hair drape;
# re-enable when hair detection is solid). NOTE: with this off, the head/face is no
# longer split from the neck island.
# all_seam |= face_cut_z("Chin seam",      0.385)

# Neck ring at z=0.35 — sits at the neck-shoulder junction, just BELOW the neck.
# Arms span z=0.24–0.38 but are at |x|>0.075; x_lo/hi=±0.10 selects only neck faces.
# DISABLED (cuts into the back hair; re-enable when hair detection is solid):
# all_seam |= face_cut_z("Neck ring",      0.35, x_lo=-0.10, x_hi=0.10)

# Ear isolation (KAN-8) — boxed X-plane cut around each ear root so the ear
# becomes its own UV island instead of folding onto the cheek inside the head
# island.  Boxes derived from the [KAN8-ear] diagnostic (per-character ear bbox).
#   L ear: x∈[0.06,0.10], y∈[-0.014,0.009], z∈[0.426,0.464]
#   R ear: x∈[0.06,0.10], y∈[ 0.000,0.027], z∈[0.426,0.464]
# KAN-9: ear cuts clutter the face area (user request) -> OFF by default. Set GS_HEADNECK_CUTS=1 to
# restore (the elf ears then unwrap as their own islands instead of folding onto the head).
if int(os.environ.get('GS_HEADNECK_CUTS', '0')):
    _ec = face_cut_x_box("L ear cut", -0.058, -0.030, 0.020, 0.420, 0.470); all_seam |= _ec; _struct_seams |= _ec; _ear_seams |= _ec
    _ec = face_cut_x_box("R ear cut", +0.058, -0.010, 0.040, 0.420, 0.470); all_seam |= _ec; _struct_seams |= _ec; _ear_seams |= _ec

# Torso — navel cut (above buttocks)
all_seam |= face_cut_z("Navel cut",      0.12)

# KAN-9: UPPER-CHEST cut — a horizontal ring just BELOW THE ARMS that peels the head+neck+
# upper-chest off the torso, so the face gets a SHORT island (it was UV-starved in the tall
# head→navel island → collapsed/smeared). Added to all_seam ONLY (not _struct_seams) so the
# hair-strip pass below CLIPS it at the hairline — it stops where it hits the draping hair instead
# of cutting through it (user spec). z is DETECTED just below the arm-bottom (armpit) so the ring
# clears the arms and spans the full torso width uninterrupted (a higher ring runs INTO the T-pose
# arms on the sides and stops short). Arm-relative so it adapts to body offset / proportions.
def _upper_chest_cut_z():
    _arm = (np.abs(co[:, 0]) > 0.22) & (co[:, 2] > 0.10) & (co[:, 2] < 0.45)   # lateral = the T-pose arms
    if int(_arm.sum()) < 20:
        return 0.30
    return float(co[_arm, 2].min()) - 0.015                        # just below the armpit
_ucz = _upper_chest_cut_z()
print(f"[KAN-9 upper-chest] cut z={_ucz:.3f} (just below the arms; hair-strip stops it at the hairline)")
all_seam |= face_cut_z("Upper-chest cut", _ucz)

# Arm/body separation (shoulder and wrist X-plane rings)
# KAN-9: RE-ENABLED (user request) + BODY-X-OFFSET ROBUST. These shoulder ring loops isolate each
# arm+shoulder from the torso (without them head+chest+arms fuse into ONE giant island → face UV
# collapses). The shoulder x is DETECTED per side (the arm-start = where the mid-chest runs out
# laterally) instead of a fixed ±0.075: the Trellis body is often x-offset (here torso centroid
# ~+0.05), so a symmetric cut lands on the armpit one side but the central chest the other → one arm
# stays fused. Detection makes both land on the real armpit regardless of offset.
def _arm_start_x(_side):
    _mid = (co[:, 2] > 0.15) & (co[:, 2] < 0.27)       # mid-chest band, below the T-pose arm
    for _k in range(4, 46):
        _x = 0.01 * _k * _side
        if int((_mid & (np.abs(co[:, 0] - _x) < 0.02)).sum()) < 8:
            return _x + 0.025 * _side                  # step slightly INTO the arm for a clean ring
    return 0.075 * _side                               # fallback to the old fixed value
_lsx = _arm_start_x(-1); _rsx = _arm_start_x(+1)
print(f"[KAN-9 arm-start] L shoulder x={_lsx:+.3f}  R shoulder x={_rsx:+.3f}  (body-offset-robust; was ±0.075)")
_sc = face_cut_x("L shoulder cut", _lsx, 0.24, 0.38); all_seam |= _sc; _struct_seams |= _sc
_sc = face_cut_x("R shoulder cut", _rsx, 0.24, 0.38); all_seam |= _sc; _struct_seams |= _sc
_sc = face_cut_x("L wrist cut",    -0.40,  0.24, 0.38); all_seam |= _sc; _struct_seams |= _sc
_sc = face_cut_x("R wrist cut",    +0.40,  0.24, 0.38); all_seam |= _sc; _struct_seams |= _sc

# Hip cut — DISABLED (KAN-9): this full front ring was the visible "thigh ring" line.
# Its original purpose was to close the ring around an OPEN crotch cross-section; the pelvis
# donor now closes that geometry, so the ring is no longer needed. With it gone the pelvis+legs
# unwrap as one "pants" island cut down the BACK (boxer + leg back-seams already do this),
# so there's no horizontal seam crossing the front of the thighs. Re-enable if a crotch/belly
# UV artifact returns (donor didn't fully close the front geometry).
# all_seam |= face_cut_z("Hip cut",        -0.03)

# Ankle separation
all_seam |= face_cut_z("L ankle cut",    -0.36, x_lo=-0.18, x_hi=0.0)
all_seam |= face_cut_z("R ankle cut",    -0.36, x_lo= 0.0,  x_hi=0.18)

# ── LENGTH SEAMS (allow cylinders to unroll flat) ─────────────────────────────
_as = length_seam("L arm seam", _lsx, -0.40, 0.24, 0.38); all_seam |= _as; _struct_seams |= _as
_as = length_seam("R arm seam", _rsx, +0.40, 0.24, 0.38); all_seam |= _as; _struct_seams |= _as

all_seam |= back_seam("L leg seam",  -0.03, -0.41, -0.22, -0.002)
all_seam |= back_seam("R leg seam",  -0.03, -0.41,  0.002,  0.22)

# Head seam: back-center cut from chin seam up to crown of head.
# x filter widened to ±0.08 so the crown (which can sit slightly off xc due to
# voxel discretisation) is included, giving the seam a higher starting point.
all_seam |= back_seam("Head seam",   0.50,  0.385, xc - 0.08, xc + 0.08)
# Neck seam: back-center cut through the neck strip between chin and shoulder rings
# DISABLED (back-center neck cut runs up into the back hair; re-enable later):
# all_seam |= back_seam("Neck seam",   0.385, 0.35,  xc - 0.05, xc + 0.05)
all_seam |= back_seam("Torso seam",  0.35,  0.12, xc - 0.06, xc + 0.06)
all_seam |= back_seam("Boxer seam",  0.12, -0.03, xc - 0.06, xc + 0.06)

# ── DROP body UV-cuts (user: "we don't need all that") ────────────────────────
# Keep ONLY the ear ring-cuts; the hair-shell seam (the hairline) is added below.
# Removes the chin/neck/navel/arm/leg/torso/hip/ankle cuts so the seam view is just
# hair + ears.  TO RESTORE the full body unwrap for body-skin texturing, delete the
# next line (everything above still computes the cuts).
# all_seam = set(_ear_seams)   # (was: drop ALL body cuts, keep only ears)
# RESTORED: keep the full anatomical body seam set computed above so the body
# unwraps into proper islands.  The hair-margin strip (further below) clips every
# body cut at the hairline, so no body seam cuts through the hair, and the hairline
# / hair-shell seam itself is preserved untouched.
print(f"[seams] body UV-cuts RESTORED: {len(all_seam)} edges (will be clipped at the hair below)")

# ── HAIR-SHELL SEAM (KAN-8) ───────────────────────────────────────────────────
# The long hair is a thin shell draped OVER the head/body and welded to it, so it
# shares UV atlas space with the body → UV collision (the right-drape edge ends
# up with no atlas texels, and hair bleeds onto the neck/face during texturing).
# Seam the hair shell off the body so it unwraps into its OWN island(s).
#
# Detect hair faces by LAYERING: a face is hair if another mesh surface lies a
# SHORT distance behind it along its inward normal.  The small max-distance is the
# discriminator — it catches the thin hair↔body gap but NOT the far face→skull
# span (so the face/front is not flagged).  Restricted to the upper body (above
# the hips, inboard of the arms).  Mark the hair↔body boundary edges as seams.
# Wrapped in try/except: on any failure it simply skips, leaving the existing
# unwrap untouched.
try:
    import mathutils
    from mathutils.bvhtree import BVHTree

    # Per-face outward normals in the (normalized) co space.
    face_normal = np.zeros((n_faces, 3), dtype=np.float64)
    for _fi in range(n_faces):
        _ls, _lt = int(loop_starts[_fi]), int(loop_totals[_fi])
        _vp = co[lv_flat[_ls:_ls + _lt]]
        _nv = np.cross(_vp[1] - _vp[0], _vp[2] - _vp[0])
        _nl = np.linalg.norm(_nv)
        if _nl > 1e-12:
            face_normal[_fi] = _nv / _nl
    # Orient outward: the crown (top, high Z) should point +Z.
    _crown = face_z > np.percentile(face_z, 98)
    if face_normal[_crown, 2].mean() < 0.0:
        face_normal = -face_normal

    # BVH over the normalized mesh for the inward "surface behind me?" test.
    _bverts = [mathutils.Vector(co[_i].tolist()) for _i in range(n_verts)]
    _bpolys = [lv_flat[int(loop_starts[_f]):int(loop_starts[_f]) + int(loop_totals[_f])].tolist()
               for _f in range(n_faces)]
    _bvh = BVHTree.FromPolygons(_bverts, _bpolys, all_triangles=False)

    HAIR_HIT_D = 0.041    # normalized: > hair↔body gap, < head front↔back span
    HAIR_Z_MIN = -0.150    # above the hips/waist (exclude legs + lower torso)
    HAIR_X_MAX = 0.200    # inboard of the arms
    # SHELL_ALIGN: the surface hit behind a face must face ~the SAME way as the face (a real
    # LAYER over the body -> hair sitting on scalp/body, both outward normals aligned). A FACE
    # CREASE/feature (eye socket, lip, nostril) hits a wall facing BACK -> dot<0 -> rejected.
    # General/dynamic (pure normals, no coords): kills the face-as-hair bleed across characters.
    SHELL_ALIGN = float(os.environ.get('GS_SHELL_ALIGN', '0.30'))
    _cand = np.where((face_z > HAIR_Z_MIN) & (np.abs(face_x) < HAIR_X_MAX))[0]
    is_hair = np.zeros(n_faces, dtype=bool)
    for _fi in _cand.tolist():
        _o = mathutils.Vector((face_centroid[_fi] - 0.0015 * face_normal[_fi]).tolist())
        _dir = mathutils.Vector((-face_normal[_fi]).tolist())
        _hit = _bvh.ray_cast(_o, _dir, HAIR_HIT_D)
        if (_hit[0] is not None and _hit[2] is not None and _hit[2] != _fi
                and _hit[3] is not None and _hit[3] > 0.0021
                and float(np.dot(face_normal[_fi], face_normal[_hit[2]])) > SHELL_ALIGN):
            is_hair[_fi] = True

    # ── COMBINE shell with surface ROUGHNESS ──────────────────────
    # Two geometric signals, each catching what the other misses:
    #   SHELL (above): a body layer sits a short distance behind a face -> the long
    #     DRAPE and the face-framing locks (shoulder/back/skull behind).  It does
    #     NOT fire on the ear/cheek/nose (solid body, nothing close behind), so it
    #     never "jumps to the ear".  But it MISSES the crown, where hair and scalp
    #     are fused into one surface (no gap behind).
    #   ROUGH (here): hair has strand/clump detail -> high local normal variation;
    #     smooth skin is low.  Catches the textured crown the shell misses -- but
    #     ALSO fires on bumpy features (ear/nose/lips), so it is never used alone.
    # hair = the connected component(s) of (shell | rough), inside the central
    # column, that CONTAIN shell support and are large.  The textured crown is
    # connected to the shell drape -> kept.  Ear/nose rough patches have no shell
    # and are cut off from the hairline by SMOOTH skin -> never connected -> dropped.
    # Fully automatic: no per-character tuning, no flat z-cuts.
    _shell = is_hair.copy()
    _pairs = np.array([fc for fc in edge_faces.values() if len(fc) == 2], dtype=np.int64)
    # CONSOLIDATE the shell: the raw inward ray-cast is speckled at the temple/side
    # (the hair there is part-fused, part-gapped) -> a checkerboard that fragments
    # the boundary.  A bounded, SYMMETRIC close (dilate then equal erode) fills those
    # small gaps without bridging the wide gap to the ear, so the side hair becomes a
    # solid region while the ear stays out.
    def _close(m, k):
        for _ in range(k):
            _c = np.zeros(n_faces)
            np.add.at(_c, _pairs[:, 0], m[_pairs[:, 1]].astype(float))
            np.add.at(_c, _pairs[:, 1], m[_pairs[:, 0]].astype(float))
            m = m | (_c >= 1.0)
        for _ in range(k):
            _c = np.zeros(n_faces)
            np.add.at(_c, _pairs[:, 0], (~m[_pairs[:, 1]]).astype(float))
            np.add.at(_c, _pairs[:, 1], (~m[_pairs[:, 0]]).astype(float))
            m = m & (_c == 0)
        return m
    _shell = _close(_shell, 3)
    _rij = 1.0 - np.sum(face_normal[_pairs[:, 0]] * face_normal[_pairs[:, 1]], axis=1)
    _rough = np.zeros(n_faces); _cntr = np.zeros(n_faces)
    np.add.at(_rough, _pairs[:, 0], _rij); np.add.at(_rough, _pairs[:, 1], _rij)
    np.add.at(_cntr, _pairs[:, 0], 1.0); np.add.at(_cntr, _pairs[:, 1], 1.0)
    _rough /= np.maximum(_cntr, 1)
    for _ in range(3):                      # light denoise only (no heavy diffusion)
        _acc = np.zeros(n_faces); _c2 = np.zeros(n_faces)
        np.add.at(_acc, _pairs[:, 0], _rough[_pairs[:, 1]]); np.add.at(_acc, _pairs[:, 1], _rough[_pairs[:, 0]])
        np.add.at(_c2, _pairs[:, 0], 1.0); np.add.at(_c2, _pairs[:, 1], 1.0)
        _rough = 0.5 * _rough + 0.5 * _acc / np.maximum(_c2, 1)
    _zone = (np.abs(face_x) < 0.160) & (face_z > -0.10)   # head + drape; no arms/legs
    # ADAPTIVE rough threshold (KAN-9): the fixed 0.009 was tuned for a SMOOTH mesh. On a
    # jittery/faceted generated mesh the per-face normal variation is high EVERYWHERE
    # (median in-zone ~0.07+ vs ~0.003 on a clean mesh), so 0.009 marks ~every face rough
    # and the grow floods the whole body -> the white-body blow-up. Scale the threshold by
    # the mesh's OWN noise floor (median rough in-zone) so only faces genuinely rougher than
    # typical (the real strands) qualify, on ANY mesh: a clean mesh keeps ~0.009, a noisy
    # one self-raises. Verified locally on a noisy mesh: grow flood 45% -> 3%.
    _zmed = float(np.median(_rough[_zone])) if _zone.any() else 0.0
    ROUGH_THR = max(0.009, 4.0 * _zmed)
    _is_rough = _rough > ROUGH_THR
    print(f"[hair-rough] adaptive: median-rough(zone)={_zmed:.4f} -> ROUGH_THR={ROUGH_THR:.4f} "
          f"({int(_is_rough.sum())} rough / {int(_zone.sum())} zone faces)")
    # ── GROW the reliable shell band into the connected textured CROWN ────────
    # Shell misses the crown (hair fused to scalp, no gap behind), but the crown is
    # ROUGH and physically continuous with the shell band over the sides/back.  Seed
    # with the shell, then flood into adjacent ROUGH faces: this fills the crown, yet
    # CANNOT cross the smooth-skin gap to the ear/face (their rough patches have no
    # rough bridge back to the seed) -> no ear-jump, no face bleed.
    # STRONGLY forward-facing faces are facial features (eyes/nose/mouth/brow); the
    # grow may not enter them, so even though they are rough it cannot leak across
    # into the face.  Threshold is deliberately steep (only true face features), so
    # the gently forward-tilted crown/forehead-edge hair is NOT cut.
    _front = face_normal[:, 1] < -0.40
    # the crown cap = UP-facing faces high on the head; the smooth spots between
    # strands point up too, so growing into them (not just the rough strands) fills
    # the crown solid.  Restricted to high z so it can't bleed onto the (also
    # up-facing) shoulders.
    _up = (face_normal[:, 2] > 0.25) & (face_z > 0.36)
    # The grow may enter forward-facing rough faces ONLY high up (forward-sweeping
    # bangs at the top of the forehead); below the brow line, forward-facing faces
    # are the actual face (eyes/nose/cheeks) and are blocked, so the grow can't creep
    # down across the temple into them.  Non-forward faces grow freely.
    # Rough-grow is allowed only HIGH (the crown); on the sides (z<0.40) the noisy
    # rough patches interleave with skin and shatter the boundary, so there the side
    # hair is left to the (consolidated) shell lock alone -> clean side hairline.
    _growable = (_is_rough | _up) & (~_front | (face_z > 0.43))
    is_hair = _shell & _zone
    _tdbg("1-shell")
    for _ in range(80):
        _hn = np.zeros(n_faces)
        np.add.at(_hn, _pairs[:, 0], is_hair[_pairs[:, 1]].astype(float))
        np.add.at(_hn, _pairs[:, 1], is_hair[_pairs[:, 0]].astype(float))
        _grow = (~is_hair) & _growable & _zone & (_hn >= 1.0)
        if not _grow.any():
            break
        is_hair |= _grow
    print(f"[hair-combo] shell={int(_shell.sum())} rough={int(_is_rough.sum())} -> grown {int(is_hair.sum())}")
    _tdbg("2-roughgrow")
    # crown cap: fill faces down to ny>-0.45 (forward onto the forehead).  Use a
    # POSITION test (central column |x|<0.10) instead of the normal test -- the bumpy
    # crown has many sideways-pointing faces that the normal test wrongly skipped,
    # leaving the skin-poke gaps/islands you saw.  Lateral faces are filled only where
    # the shell accepts them (side-scalp), so the lateral ear stays skin.
    _fillable = ((face_z > 0.34) & (face_normal[:, 1] > -0.20) & _zone &
                 ((np.abs(face_x) < 0.10) | _shell))
    is_hair |= _fillable
    _tdbg("3-crowncap")
    def _cc_keep(mask, minsz):
        _p = list(range(n_faces))
        def _f(a):
            while _p[a] != a:
                _p[a] = _p[_p[a]]; a = _p[a]
            return a
        for _a, _b in _pairs.tolist():
            if mask[_a] and mask[_b]:
                _ra, _rb = _f(_a), _f(_b)
                if _ra != _rb: _p[_ra] = _rb
        _r = np.array([_f(_x) for _x in range(n_faces)])
        _uu, _ii, _cc = np.unique(_r, return_inverse=True, return_counts=True)
        return mask & (_cc[_ii] >= minsz)
    is_hair = _cc_keep(is_hair, 1200)
    _tdbg("4-cckeep")
    # FACE carve (curved, shell-aware): forward-facing + no body layer behind = the
    # face -> skin.  Follows the normal-transition, no flat cut.  (Eye-through remains
    # because the shell fires on the eyeball behind the lid -- that is the known issue
    # of this state, which the user accepts for now.)
    # carve only the STEEPLY forward faces (the forehead/face proper, ny<-0.45);
    # the gently-forward crown-front (ny~-0.3) is hair and stays, so the hairline
    # sits at the forehead curve instead of climbing up into the crown.  No z cap,
    # so the line follows the forward->up transition (curved, not flat).
    # carve only the STEEPLY forward faces (ny<-0.52) -> hairline sits a bit further
    # FORWARD on the forehead/sides (less is carved = more hair kept at the front).
    _facefwd = ((face_normal[:, 1] < -0.52) & ~_shell &
                (face_z > 0.30) & (np.abs(face_x) < 0.15))
    is_hair &= ~_facefwd
    # EYE/TEMPLE carve (the residual "yellow on the face").  The face carve above
    # needs ~_shell, but the shell fires on the eyeball behind the lid, so it can
    # never reach the eye/temple -> those stay hair.  Discriminate by NORMAL, not
    # height: the eye/temple is the FORWARD-and-DOWN-facing face plane (normal z<0),
    # while the scalp/hairline domes UP (normal z>0) even at its lowest -> carving
    # the down-facing faces removes the eye/temple WITHOUT touching the hairline.
    # |x|>0.045 keeps the central forehead hairline completely untouched.
    # GATE to the front-face surface (depth fy<-0.02): the eye/temple lives on the
    # front of the face, while the side/jaw HAIR DRAPE sits at the side (fy~0).  This
    # gate catches 100% of the eye/temple yet 0% of the jaw drape -> the carve can no
    # longer eat the side hair below the ear (and it only SHRINKS, never reaching the
    # hairline).
    _eyecarve = ((face_normal[:, 1] < -0.20) & (face_normal[:, 2] < -0.10) &
                 (face_centroid[:, 1] < -0.02) &
                 (face_z > 0.30) & (face_z < 0.47) &
                 (np.abs(face_x) > 0.045) & (np.abs(face_x) < 0.16))
    is_hair &= ~_eyecarve
    _tdbg("5-face/eyecarve")
    _prefill = is_hair.copy()   # base detection (shell-align + strand-grow + carve), BEFORE the fills; the face-guard trusts this and only undoes fill-added forward face
    # ── ISSUE-2 (KAN): per-character hairline raise, AUTO-CLASSIFIED. Per-face geometry on the
    #    forehead is identical between a STRANDY head (charA: framing hair MUST be kept) and a
    #    SMOOTH head (charB: smooth hair lies on the forehead, so the hairline reads too low) --
    #    both are forward-facing, shell-supported and rough.  They differ only at SURFACE scale:
    #    strandy hair has high local normal dispersion, smooth hair is a flat shell.  So classify
    #    by whole-mesh strandiness, and ONLY on a SMOOTH head apply a forward-facing forehead
    #    carve to recede the hairline UP the forward->up curve (no z cap -> it stays a dome, not a
    #    flat/translated line).  A strandy head is gated out entirely, so its pipeline is byte-for-
    #    byte unchanged (charA gold preserved by construction).  Re-applied after the smoothing.
    _nrm = face_normal / np.maximum(np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-9)
    _dadj = 1.0 - np.abs(np.einsum('ij,ij->i', _nrm[_pairs[:, 0]], _nrm[_pairs[:, 1]]))
    _dsum = np.zeros(n_faces); _dnum = np.zeros(n_faces)
    np.add.at(_dsum, _pairs[:, 0], _dadj); np.add.at(_dsum, _pairs[:, 1], _dadj)
    np.add.at(_dnum, _pairs[:, 0], 1.0);  np.add.at(_dnum, _pairs[:, 1], 1.0)
    _disp = _dsum / np.maximum(_dnum, 1.0)
    _hb = is_hair & (face_z > 0.30)
    _strand = float(_disp[_hb].mean()) if _hb.any() else 0.0
    I2_STRAND_THR = float(os.environ.get('GS_I2_STRAND_THR', '0.026'))   # below = SMOOTH; centred between charB 0.017 / charA 0.038
    _is_smooth = _strand < I2_STRAND_THR
    print(f"[hairtype] strandiness(head-hair normal-dispersion)={_strand:.4f}  thr={I2_STRAND_THR} -> "
          f"{'SMOOTH: raise hairline' if _is_smooth else 'STRANDY: keep framing (no change)'}")
    # NOTE: the smooth-type hairline RAISE is done AFTER the smoothing below, as a morphological
    # erosion of the forward-facing forehead boundary, so the fills/diffusion can't undo it.
    # fill small enclosed holes (strand specks)
    _parH = list(range(n_faces))
    def _findH(a):
        while _parH[a] != a:
            _parH[a] = _parH[_parH[a]]; a = _parH[a]
        return a
    for _a, _b in _pairs.tolist():
        if (not is_hair[_a]) and (not is_hair[_b]):
            _ra, _rb = _findH(_a), _findH(_b)
            if _ra != _rb: _parH[_ra] = _rb
    _rH = np.array([_findH(_f) for _f in range(n_faces)])
    _uH, _iH, _cH = np.unique(_rH, return_inverse=True, return_counts=True)
    # fill enclosed skin holes between strands (the little islands on the crown).
    # cap is large enough to close them but FAR below the face/neck/body (one big
    # connected region of thousands of faces), which is never a "hole" -> never filled.
    is_hair = is_hair | ((~is_hair) & (_cH[_iH] < 4000))
    # light de-jag
    _degF = np.zeros(n_faces)
    np.add.at(_degF, _pairs[:, 0], 1.0); np.add.at(_degF, _pairs[:, 1], 1.0)
    for _ in range(2):
        _hcF = np.zeros(n_faces)
        np.add.at(_hcF, _pairs[:, 0], is_hair[_pairs[:, 1]].astype(float))
        np.add.at(_hcF, _pairs[:, 1], is_hair[_pairs[:, 0]].astype(float))
        is_hair = (_hcF > (0.5 * _degF)) & _zone & ~_facefwd & ~_eyecarve
    # ── SMOOTH the hairline into a continuous line (kill the quad staircase) ─────
    # The hairline boundary follows the mesh quads in stair-steps.  A morphological
    # close+open rounds those steps into a clean continuous line that rides the drape
    # edge -- and unlike majority voting it does NOT erode the diagonal boundary
    # (majority voting recedes 45° lines).  Re-mask so it can never grow into the
    # face/eye carve or out of the zone.
    def _open(m, k):
        for _ in range(k):
            _c = np.zeros(n_faces)
            np.add.at(_c, _pairs[:, 0], (~m[_pairs[:, 1]]).astype(float))
            np.add.at(_c, _pairs[:, 1], (~m[_pairs[:, 0]]).astype(float))
            m = m & (_c == 0)
        for _ in range(k):
            _c = np.zeros(n_faces)
            np.add.at(_c, _pairs[:, 0], m[_pairs[:, 1]].astype(float))
            np.add.at(_c, _pairs[:, 1], m[_pairs[:, 0]].astype(float))
            m = m | (_c >= 1.0)
        return m
    is_hair = _open(_close(is_hair, 2), 2)
    is_hair &= _zone & ~_facefwd & ~_eyecarve
    # ── DRIVE THE HAIRLINE WITH THE CONCAVE+CREASE BORDER (version A) ────────────
    # Push the hair edge OUT to the sharp concave/crease line the user confirmed: a
    # small-scale concavity (smoothed-centroid vs normal) OR a hard crease (dihedral).
    # Bounded to <=3 rings beyond the current edge, inside the zone, and OFF the front
    # face (depth fy>-0.03) so the crease that also fires on the nose/lips cannot leak
    # onto the face.  The hair/skin boundary (the seam) then lands ON that line.
    _cnt = np.zeros(n_faces)
    np.add.at(_cnt, _pairs[:, 0], 1.0); np.add.at(_cnt, _pairs[:, 1], 1.0)
    _cnt = np.maximum(_cnt, 1.0)
    _P = face_centroid.copy()
    for _ in range(4):                                   # small-scale smoothed centroid
        _acc = np.zeros_like(_P)
        np.add.at(_acc, _pairs[:, 0], _P[_pairs[:, 1]]); np.add.at(_acc, _pairs[:, 1], _P[_pairs[:, 0]])
        _P = 0.5 * _P + 0.5 * _acc / _cnt[:, None]
    _conc = np.einsum('ij,ij->i', _P - face_centroid, face_normal)   # <0 = sharp convex ridge
    _dih = 1.0 - np.sum(face_normal[_pairs[:, 0]] * face_normal[_pairs[:, 1]], axis=1)
    _crz = np.zeros(n_faces); np.maximum.at(_crz, _pairs[:, 0], _dih); np.maximum.at(_crz, _pairs[:, 1], _dih)
    _wallA = (_conc < np.percentile(_conc, 18)) | (_crz > 0.25)
    _wallA = _close(_wallA, 3)   # close gaps so the concave/crease line is UNINTERRUPTED
    # (the version-A drive itself happens in the FINAL pass below, after the island cull)
    # SHELL-AWARE island cull: keep the main mass AND any sizeable, strongly
    # shell-supported drape that the ear/topology split off from it (the side/jaw
    # drape below the ear is genuine LAYERED hair, not a stray speck).  Plain size
    # culling deleted it because the protruding ear breaks the shell band and isolates
    # it (~1700 faces < 2500).  Shell density tells real hair from noise.
    def _cc_keep_shell(mask, big, small, sfrac):
        _p = np.arange(n_faces)
        def _f(a):
            while _p[a] != a:
                _p[a] = _p[_p[a]]; a = _p[a]
            return a
        for _a, _b in _pairs.tolist():
            if mask[_a] and mask[_b]:
                _ra, _rb = _f(int(_a)), _f(int(_b))
                if _ra != _rb: _p[_ra] = _rb
        _r = np.array([_f(_x) for _x in range(n_faces)])
        _r[~mask] = -1
        _u, _inv, _c = np.unique(_r, return_inverse=True, return_counts=True)
        _sc = np.zeros(len(_u))
        np.add.at(_sc, _inv[mask], _shell[mask].astype(float))
        _frac = _sc / np.maximum(_c, 1)
        _keep = (_c >= big) | ((_c >= small) & (_frac >= sfrac))
        return _keep[_inv] & mask
    is_hair = _cc_keep_shell(is_hair, 2500, 500, 0.40)
    # ── KAN-8: DRIVE THE HAIR REGION FROM VERSION A (smoothed density field) ─────
    # Draw the seams FULLY from the concave/crease border (version A).  version A is
    # DENSE on textured hair, SPARSE+scattered on smooth body.  Treat it as a continuous
    # "hair-ness" FIELD: diffuse the version-A indicator, then threshold.  The diffusion
    # is what makes this general (no symmetry assumption, no per-character constant): it
    # SOLIDIFIES the streaky strand pattern (dense strands bleed across the inter-strand
    # valleys) AND self-balances L/R (a slightly smoother side inherits density from the
    # dense crown it is connected to, so it is kept on its own merit).  Then: keep the
    # main connected mass(es) -- this DROPS the isolated body bumps version A also fires
    # on (breast/shoulder convex curvature) -- fill small enclosed holes, de-speckle.
    _wallA_raw = (_conc < np.percentile(_conc, 18)) | (_crz > 0.25)
    _densA = _wallA_raw.astype(float)                 # version-A indicator as a field
    for _ in range(8):                                # DIFFUSE -> solid + L/R self-balanced
        _acc = np.zeros(n_faces)
        np.add.at(_acc, _pairs[:, 0], _densA[_pairs[:, 1]])
        np.add.at(_acc, _pairs[:, 1], _densA[_pairs[:, 0]])
        _densA = 0.5 * _densA + 0.5 * _acc / _cnt
    _faceA = ((face_normal[:, 1] < -0.40) & (face_centroid[:, 1] < -0.01) &
              (face_z > 0.20) & (face_z < 0.46) & (np.abs(face_x) < 0.12))   # front face -> skip
    is_hair = (_densA > 0.30) & _zone & ~_faceA
    is_hair = _close(is_hair, 2)                      # close small gaps -> solid region
    # ── BRIDGE the scalp dimple (HIGH on the head only) so it is NOT read as an edge
    # The detector keys on CONVEX strand texture, so the INWARD dimple where the side
    # hair meets the scalp reads as low-density and gets cut -- disconnecting the side
    # lobe so the size-cull drops it (the bald temple).  Grow hair into adjacent strongly
    # CONCAVE faces, but ONLY above BRIDGE_Z so it physically cannot reach the nape/neck
    # (that band is z<=0.42; the neck-bleed earlier came from gating too low).  The true
    # hairline (hair over the CONVEX forehead) is not concave -> never bridged.
    BRIDGE_Z = 0.42
    _concaveD = (_conc > np.percentile(_conc, 80)) & (face_z > BRIDGE_Z) & _zone & ~_faceA
    for _ in range(6):
        _bn = np.zeros(n_faces)
        np.add.at(_bn, _pairs[:, 0], is_hair[_pairs[:, 1]].astype(float))
        np.add.at(_bn, _pairs[:, 1], is_hair[_pairs[:, 0]].astype(float))
        _br = (~is_hair) & _concaveD & (_bn >= 1.0)
        if not _br.any():
            break
        is_hair |= _br
    # keep the main connected mass(es): components >= 15% of the largest -> drops the
    # isolated body bumps version A also fires on (breast/shoulder convex curvature).
    _pA = np.arange(n_faces)
    def _fA(a):
        while _pA[a] != a:
            _pA[a] = _pA[_pA[a]]; a = _pA[a]
        return a
    for _a, _b in _pairs.tolist():
        if is_hair[_a] and is_hair[_b]:
            _ra, _rb = _fA(int(_a)), _fA(int(_b))
            if _ra != _rb: _pA[_ra] = _rb
    _rA = np.array([_fA(_x) for _x in range(n_faces)]); _rA[~is_hair] = -1
    _uA, _cA = np.unique(_rA[is_hair], return_counts=True)
    if len(_cA):
        _keepA = set(_uA[_cA >= _cA.max() * 0.15].tolist())
        is_hair = np.array([is_hair[_i] and (_rA[_i] in _keepA) for _i in range(n_faces)])
    # ── SOLIDIFY + ROUND into ONE clean closed hairline ─────────────────────────
    # Make the hair SOLID: keep only the single biggest skin region; every OTHER skin
    # region is enclosed by hair -> fill it.  This removes ALL internal seam loops (the
    # back blotches AND the temple notches that made it "not close") with NO size cap.
    # Verified offline on this mesh: 0 internal holes, 0 ear bleed, no body bleed (the
    # body/face/neck is one huge skin region, so it is never filled).  Then a bounded
    # morphological close+open rounds the jagged hairline (de-jags the temple), then
    # re-solidify and drop sub-200 specks -> a single clean closed boundary.
    # SOLID_CAP: fill ENCLOSED skin only if its component is SMALL (a bald speck/notch). A large
    # enclosed region is the FACE (the hairline rings it) -> must NOT be filled. Cap is a fraction
    # of the mesh -> dynamic across characters, no coords. The biggest skin comp (open body) is
    # never filled regardless.
    SOLID_CAP = float(os.environ.get('GS_SOLID_CAP', '0.012'))
    def _solid(m):
        _p = np.arange(n_faces)
        def _ff(a):
            while _p[a] != a:
                _p[a] = _p[_p[a]]; a = _p[a]
            return a
        for _a, _b in _pairs.tolist():
            if (not m[_a]) and (not m[_b]):
                _ra, _rb = _ff(int(_a)), _ff(int(_b))
                if _ra != _rb: _p[_ra] = _rb
        _r = np.array([_ff(_x) for _x in range(n_faces)])
        _lab = _r[~m]
        if not len(_lab):
            return m
        _u, _c = np.unique(_lab, return_counts=True)
        _big = _u[int(np.argmax(_c))]
        _sz = _c[np.clip(np.searchsorted(_u, _r), 0, len(_u) - 1)]   # per-face enclosed-component size
        return m | ((~m) & (_r != _big) & _zone & (_sz < int(n_faces * SOLID_CAP)))
    is_hair = _solid(is_hair)                          # fill ALL interior holes -> solid hair
    # CLOSE-ONLY (no erosion): fills notches + bridges the hairline gap so the loop
    # closes over the temple, while PRESERVING the thin temple/face-frame hair (erosion
    # would shave it off and leave the temple bald).  Then re-solidify + speck-cull.
    is_hair = _close(is_hair, 2)
    is_hair = _solid(is_hair)                          # re-solidify (fill any notch left open)
    is_hair = _cc_keep(is_hair, 200)                   # drop tiny specks
    is_hair &= _zone & ~_faceA
    # CLOSE the internal scalp channel (the depression groove the user marked in yellow):
    # a skin strip HEMMED by hair but OPEN at its ends, so _solid can't fill it (it stays
    # connected to the outer skin).  Fill skin faces whose local neighbourhood is MOSTLY
    # hair -- this catches the groove but NOT the forehead edge (which is open to the big
    # face skin = low hair fraction).  Gated neck-safe: z>0.42 everywhere, extended down
    # to z>0.36 only on the FRONT (temple, fy<0) where there is no neck.  The back/neck
    # below 0.42 is hard-excluded so it cannot bleed.
    # FILL bald regions that sit WITHIN the hair borders (user's rule).  Such a patch is
    # connected to the outer face-skin only through a NARROW channel (the Trellis
    # depression), so it's technically "outside" and plain hole-fill misses it.  Bridge
    # that channel with a temporary CLOSE so the patch becomes enclosed, let _solid fill
    # the enclosed patch, then add back ONLY those filled patch-faces (NOT the close's
    # boundary growth) -> the hairline is unchanged, the bald patch becomes hair.  Gated
    # neck-safe (z>0.42 back, z>0.36 front) so a bridged neck region can never be filled.
    # ITERATIVE: a bald "bay" of face-skin pokes up into the hair through a narrow neck in
    # the jagged hairline (BOTH temples).  Each pass: CLOSE bridges the bay's narrow
    # opening so it becomes enclosed, _solid fills it, add back ONLY the filled faces (not
    # the boundary growth).  Filling widens the hair and narrows the next bay, so repeated
    # passes eat the jagged bays on both sides.  The broad forehead/cheek have a HUGE
    # opening to the face -> never enclosed by a small close -> never filled.  Gated to the
    # whole head EXCEPT the central-back-low NECK strip, so it can never re-bleed the neck.
    _neck = (face_centroid[:, 1] > 0.0) & (np.abs(face_x) < 0.13) & (face_z < 0.42)
    _safe = (face_z > 0.34) & _zone & ~_faceA & ~_neck
    _tot = 0
    for _it in range(10):
        _closedK = _close(is_hair, 3)
        _patch = _solid(_closedK) & (~_closedK) & _safe
        if not _patch.any():
            break
        is_hair |= _patch; _tot += int(_patch.sum())
        is_hair = _solid(is_hair)
    is_hair &= _zone & ~_faceA
    # FINAL: fill bald ISLANDS that are fully ringed by hair, even inside the face zone.
    # These small skin islands at the jagged hairline are enclosed by hair, so they are
    # hair-INTERIOR, not open face -- the ~_faceA protection above was wrongly forcing them
    # back to skin, which left the bald spots + closed seam-loops the user marked.  _solid
    # fills ONLY non-biggest (enclosed) skin, so the open forehead/cheek (the one huge skin
    # component) is untouched; only the ringed-by-hair islands flip to hair.
    is_hair = _solid(is_hair)
    is_hair &= _zone
    print(f"[engulf-fill] filled {_tot} bald-bay faces; final _solid fills enclosed islands")
    _tdbg("6-engulf")
    # ── OUTER-TEMPLE strand fill (narrow, eye-safe) ──────────────────────────────
    # Fill the forehead strand at the OUTER TEMPLE only: z 0.40-0.44, |x| 0.08-0.12 -- the
    # eye center sits at |x|<0.075, so this stays clear of it.  Both sides (anatomical).
    # The user OK'd "all hair is fine" here.
    _TF_XMAX = float(os.environ.get('GS_TEMPLE_FILL_XMAX', '0.05'))   # outer-temple fill lateral cap (was 0.12 -- it pulled the hairline toward the ear)
    _ttzone = ((face_z > 0.405) & (face_z < 0.44) & (face_centroid[:, 1] < -0.01) &
               (np.abs(face_x) > 0.04) & (np.abs(face_x) < _TF_XMAX) & _zone)
    _tttot = int(((~is_hair) & _ttzone).sum())
    is_hair |= _ttzone
    is_hair = _solid(is_hair)
    is_hair = _cc_keep(is_hair, 60)    # drop tiny stray specks; keep the one connected mass
    is_hair &= _zone
    print(f"[temple-fill] solid-filled {_tttot} outer-temple faces (z 0.40-0.44, |x| .08-.12)")
    _tdbg("7-templefill")
    # ── SHELL-GROW: fill smooth hair that the strand-detector under-fires on ──────────
    # version-A keys off CONVEX STRAND detail, so a smooth/slicked cap or a free-hanging
    # drape fills as streaks (a different character revealed this).  The depth shell
    # (_shell) catches the smooth hair SURFACE regardless of strands, but it also fires on
    # the arms (the torso sits behind them).  Growing the shell only where it is CONNECTED
    # to the confident version-A hair drops the arms (a separate surface component) and
    # fills the cap+drape solid.  Fully general: depth + connectivity, no per-char coords.
    # (_zone already excludes arms/legs, a structural backstop on top of connectivity.)
    try:
        # & ~_faceA: shell-grow may grow into the central forward FACE (the shell fires on
        # the skull behind the forehead and connects to the hairline) -> a forehead spike.
        # Excluding _faceA from the GROW TARGET prevents that creep at the source WITHOUT
        # removing anything (pre-existing hair stays via is_hair).  Keeps A == gold + non-face
        # fills, and the face is skin on B anyway so its crown/drape fill is unaffected.
        _sg_region = is_hair | (_shell & _zone & ~_faceA)
        _sg_adj = defaultdict(list)
        for _eid, _efc in edge_faces.items():
            if len(_efc) == 2 and _sg_region[_efc[0]] and _sg_region[_efc[1]]:
                _sg_adj[_efc[0]].append(_efc[1]); _sg_adj[_efc[1]].append(_efc[0])
        _grown = is_hair.copy()
        _sgq = deque(np.where(is_hair)[0].tolist())
        while _sgq:
            _u = _sgq.popleft()
            for _w in _sg_adj[_u]:
                if not _grown[_w]:
                    _grown[_w] = True; _sgq.append(_w)
        _sg_added = int(_grown.sum()) - int(is_hair.sum())
        is_hair = _grown
        print(f"[shell-grow] filled {_sg_added} smooth-hair faces via shell+connectivity (total {int(is_hair.sum())})")
        _tdbg("8-shellgrow")
    except Exception as _sge:
        print(f"[shell-grow] skipped: {_sge!r}")
    # shell-grow newly ENCLOSES skin islands (gaps fully ringed by the grown hair); the
    # earlier _solid ran before the grow, so re-run it here to fill them (general: fills
    # only non-biggest, i.e. enclosed, skin -- the open face/neck is untouched).
    # ROBUST enclosed-island fill.  The face graph (edge_faces) bridges a hair-ringed
    # bay to the open face through thin/degenerate faces (the viz triangulation drops
    # those, which is exactly why the bay reads as a clean gap to the eye).  So detect
    # enclosure on the SAME fan-triangulated mesh the render/eye sees: a skin TRI not in
    # the biggest skin tri-component is enclosed; flip its parent face to hair.  Additive
    # + zone-gated, so it can only fill skin already ringed by hair (no open-face bleed).
    try:
        _ft = []; _fp = []
        for _f in range(n_faces):
            _vs = lv_flat[loop_starts[_f]:loop_starts[_f] + loop_totals[_f]]
            for _k in range(1, len(_vs) - 1):
                _ft.append((int(_vs[0]), int(_vs[_k]), int(_vs[_k + 1]))); _fp.append(_f)
        _ft = np.asarray(_ft, dtype=np.int64); _fp = np.asarray(_fp, dtype=np.int64)
        _tn = len(_ft); _thair = is_hair[_fp]; _sk = ~_thair
        _Em = np.sort(np.vstack([_ft[:, [0, 1]], _ft[:, [1, 2]], _ft[:, [2, 0]]]), 1)
        _tt = np.tile(np.arange(_tn), 3)
        _od = np.lexsort((_Em[:, 1], _Em[:, 0])); _Es = _Em[_od]; _ts = _tt[_od]
        _sm = np.all(_Es[1:] == _Es[:-1], 1)
        _e0 = _ts[:-1][_sm]; _e1 = _ts[1:][_sm]
        _ke = _sk[_e0] & _sk[_e1]; _u0 = _e0[_ke]; _u1 = _e1[_ke]    # skin-skin tri edges
        _par = np.arange(_tn)
        def _tff(a):
            while _par[a] != a: _par[a] = _par[_par[a]]; a = _par[a]
            return a
        for _x, _y in zip(_u0.tolist(), _u1.tolist()):
            _rx = _tff(_x); _ry = _tff(_y)
            if _rx != _ry: _par[_rx] = _ry
        _root = np.array([_tff(_i) for _i in range(_tn)])
        _skroot = _root[_sk]
        if len(_skroot):
            _ur, _cr = np.unique(_skroot, return_counts=True); _bigr = _ur[int(np.argmax(_cr))]
            _isltri = _sk & (_root != _bigr)
            _islf = np.zeros(n_faces, bool); _islf[_fp[_isltri]] = True
            _islf &= (~is_hair) & _zone
            _nf = int(_islf.sum()); is_hair = is_hair | _islf
            print(f"[island-fill] filled {_nf} enclosed bay faces (tri-CC, total {int(is_hair.sum())})")
    except Exception as _ife:
        print(f"[island-fill] skipped: {_ife!r}")
    # NOTE: the post-shell-grow `~_faceA` re-apply AND the ear-carve are intentionally REMOVED
    # here -- both CHANGE character A vs its approved gold (the re-apply drops ~545 central-
    # forehead faces; the ear-carve drops ~908 ear faces).  shell-grow + island-fill above are
    # purely ADDITIVE, so A = gold + fills only (verified 0 hair->skin).  The ear-carve can be
    # re-added later as a separately-verified change if skin ears are wanted.
    # ── HAIRLINE SMOOTHING (KAN): round off the jagged hair boundary so the seam follows a
    #    smooth line (fixes the zigzag/notch the seam picks up near ears/jaw/the forehead dip).
    #    Diffuse the mask over face adjacency, re-threshold at 0.5 -> interior/exterior stay,
    #    only the jagged BOUNDARY rounds. Applied ONLY in the head/hairline band (face_z>0.30)
    #    so the lower drape/back is left alone; an already-smooth seam (charA) has little
    #    jaggedness to grab, so it is barely touched. Tune HAIR_SMOOTH up/down.
    HAIR_SMOOTH = int(os.environ.get('GS_HAIR_SMOOTH', '80'))   # fixes seam wander 1/3/4/5; env-tunable
    if HAIR_SMOOTH > 0:
        _sm = is_hair.astype(float)
        for _ in range(HAIR_SMOOTH):
            _acc = np.zeros(n_faces)
            np.add.at(_acc, _pairs[:, 0], _sm[_pairs[:, 1]])
            np.add.at(_acc, _pairs[:, 1], _sm[_pairs[:, 0]])
            _sm = _acc / _cnt
        _b4 = int(is_hair.sum())
        is_hair = np.where(face_z > 0.30, _sm > 0.5, is_hair)
        print(f"[hairline-smooth] {HAIR_SMOOTH} passes (head band z>0.30): "
              f"{_b4} -> {int(is_hair.sum())} hair faces")
    _tdbg("9-smooth")
    # ── ISSUE-2 hairline RAISE (SMOOTH heads only): morphologically ERODE the forward-facing
    #    forehead boundary into a DOME.  The erosion depth is CENTRE-WEIGHTED -- full I2_RINGS at
    #    the centre (x=0), tapering with a parabola to 0 at |x|=I2_XW -- so the hairline ARCHES up
    #    instead of rising as a flat band (the 80-pass smoothing straightens the line; this re-
    #    introduces the curve).  Gated to forward-facing (ny<I2_NY) forehead (z>I2_ZMIN); lateral
    #    temples and the lower front are left alone.  Knobs: I2_DZ = centre peak height (normalized
    #    z, resolution-independent), I2_XW = dome width, I2_NY = how forward a face must be to count
    #    as forehead.  Strandy heads skip this entirely (charA safe).  Post-smooth so fills can't undo it.
    if _is_smooth:
        # Resolution-INDEPENDENT peak: target a centre rise of I2_DZ in normalized-z and convert
        # to face-rings via the mesh's OWN median edge length (the voxel-remesh density varies a
        # little between characters; this keeps the physical raise constant rather than the ring
        # count).  GS_I2_RINGS>0 overrides with a fixed ring count for manual tuning.
        _facesz  = float(np.median(ELEN))                      # global median edge ~ remesh resolution
        I2_DZ    = float(os.environ.get('GS_I2_DZ', '0.0115')) # centre peak raise, normalized-z (~6 rings on charB)
        I2_RINGS = int(os.environ.get('GS_I2_RINGS', '0'))     # >0 = fixed ring count (overrides I2_DZ)
        if I2_RINGS <= 0:
            I2_RINGS = max(1, int(round(I2_DZ / max(_facesz, 1e-6))))
        I2_NY    = float(os.environ.get('GS_I2_NY', '-0.45'))  # forehead = faces this forward-facing
        I2_ZMIN  = float(os.environ.get('GS_I2_ZMIN', '0.42')) # forehead only -> never the lower front/sideburns
        I2_XW    = float(os.environ.get('GS_I2_XW', '0.04'))   # dome half-width: raise tapers to 0 here
        _erodable = (face_normal[:, 1] < I2_NY) & (face_z > I2_ZMIN) & (np.abs(face_x) < I2_XW)
        _wdome = np.clip(1.0 - (np.abs(face_x) / max(I2_XW, 1e-6)) ** 2, 0.0, 1.0)   # 1 centre -> 0 sides
        _b2 = int(is_hair.sum()); _ea, _eb = _pairs[:, 0], _pairs[:, 1]
        for _r in range(I2_RINGS):
            _active = _erodable & (_wdome > (_r / max(I2_RINGS, 1)))   # sides drop out as _r grows -> arch
            _bnd = np.zeros(n_faces, bool)
            np.logical_or.at(_bnd, _ea, is_hair[_ea] & ~is_hair[_eb])
            np.logical_or.at(_bnd, _eb, is_hair[_eb] & ~is_hair[_ea])
            is_hair &= ~(_bnd & _active)
        print(f"[hairline-raise] dome erosion: peak {I2_RINGS} rings (dz={I2_DZ}, face_sz={_facesz:.5f}), "
              f"half-width {I2_XW} (ny<{I2_NY}, z>{I2_ZMIN}): {_b2} -> {int(is_hair.sum())} hair faces")
    # ── FINAL FACE/BODY-SKIN GUARD (general; normals + shell only, NO coords): a face that is
    #    FORWARD-facing yet has NO aligned layer behind it (~_shell) is skin — the actual face, or a
    #    body-skin bleed — never hair. Carve it. A drape/fringe over the body IS a layer (_shell) so
    #    it is kept; the up-facing scalp is untouched. Undoes any face/body the fills over-reached.
    #    Only carve forward/no-layer faces the FILLS ADDED (~_prefill) -- the base detection
    #    (shell-align + strand-grow) is trusted, so a strandy head's real forward framing (in
    #    _prefill) is kept; the face that the enclosed-fill/close over-reached onto is removed.
    #    General, dynamic, no coords, no per-character branch.
    #    Per-FACE roughness can't separate framing from face (the distributions overlap), but at the
    #    REGION level it does: charA's framing components are strandy (mean rough ~0.07), the enclosed
    #    face is smooth (mean rough ~0.03). So group the ambiguous forward+no-layer hair into connected
    #    components and carve only the SMOOTH ones (face); strandy components (framing) are kept.
    FACE_NY = float(os.environ.get('GS_FACE_NY', '-0.40'))
    FACE_STRAND_THR = float(os.environ.get('GS_FACE_STRAND', '0.045'))
    _amb = (face_normal[:, 1] < FACE_NY) & (~_shell) & is_hair
    if _amb.any():
        _pp = np.arange(n_faces)
        def _ff2(a):
            while _pp[a] != a:
                _pp[a] = _pp[_pp[a]]; a = _pp[a]
            return a
        for _a, _b in _pairs.tolist():
            if _amb[_a] and _amb[_b]:
                _ra, _rb = _ff2(int(_a)), _ff2(int(_b))
                if _ra != _rb: _pp[_ra] = _rb
        _root = np.array([_ff2(i) for i in range(n_faces)])
        FACE_MIN = int(os.environ.get('GS_FACE_MIN', '600'))
        _fg0 = int(is_hair.sum()); _carved = 0; _comps = []
        for _rt in np.unique(_root[_amb]):
            _comp = _amb & (_root == _rt)
            _sz = int(_comp.sum()); _mr = float(_rough[_comp].mean())
            _comps.append((_sz, round(_mr, 3)))
            if _mr < FACE_STRAND_THR and _sz > FACE_MIN:
                is_hair &= ~_comp; _carved += _sz
        _comps.sort(reverse=True)
        print(f"[face-guard] region carve (<{FACE_STRAND_THR}, >{FACE_MIN}): {_fg0}->{int(is_hair.sum())} ({_carved}); "
              f"top(sz,rough)={_comps[:6]}")
    if os.environ.get('GS_DUMP'):
        np.savez(os.environ['GS_DUMP'], rough=_rough, shell=_shell, prefill=_prefill,
                 ny=face_normal[:, 1], fz=face_z, fx=face_x, ishair=is_hair, pairs=_pairs)
        print(f"[GS_DUMP] wrote {os.environ['GS_DUMP']}")
    _tdbg("10-faceguard")
    # ── ROBUST PLAUSIBILITY CAP (KAN-9) ───────────────────────────────────────────
    # Hair on a humanoid is a MINORITY of the surface (~5-15%). If detection flags an
    # implausible fraction, the upstream mesh is degenerate — doubled/self-intersecting
    # surfaces that the shell-layer test cannot tell apart from a real hair layer (on a
    # welded mesh they are geometrically identical) — and the result buries the whole body
    # in Hair material. So REJECT it: keep only hair in the HEAD band (top of the figure),
    # so the body projects as skin instead of going white. This is a self-calibrating
    # SAFEGUARD: a sane detection sits far below the cap and is left byte-for-byte unchanged;
    # only a blow-up triggers it, and even then the scalp/hairline survives.
    _HAIR_CAP = 0.35
    if is_hair.sum() > _HAIR_CAP * n_faces:
        _before = int(is_hair.sum())
        _zhead = float(np.percentile(face_z, 80.0))      # top ~20% of height = head/scalp
        is_hair = is_hair & (face_z >= _zhead)
        print(f"[hair-cap] IMPLAUSIBLE hair {_before}/{n_faces} (>{_HAIR_CAP:.0%}) — upstream "
              f"mesh degenerate; restricted to head band (z>={_zhead:.3f}) -> {int(is_hair.sum())} "
              f"(body now projects as skin instead of white Hair material)")
    # ROBUST FACE RE-CARVE (KAN-9): the island/solid fills re-flood the face (it's enclosed by hair, so it
    # fills back as a "bay"), and the roughness-gated face-guard above MISSES it on a coarse/strandy mesh
    # (every region reads rough>thr) -> the intermittent "no face" bake. Re-apply the SAME geometric face +
    # eye/temple carves used at detection, unconditionally, so the face is NEVER hair regardless of
    # roughness/strandiness. Only removes the steeply-forward central face faces the fills re-added
    # (geometric, same criterion as detection -> charA framing / charC bob are untouched).
    if int(os.environ.get('GS_FACE_RECARVE', '1')):
        _rf0 = int(is_hair.sum())
        is_hair = is_hair & ~_facefwd & ~_eyecarve
        print(f"[face-recarve] {_rf0} -> {int(is_hair.sum())} (geometric face+eye carve, roughness-independent)")
    # ── GREEN-BORDER OVERRIDE (KAN-9) ─────────────────────────────────────────
    # If the green-border bridge produced a per-face hair region (path3 crease hairline computed on the
    # DENSE high-res mesh, transferred to THIS mesh's faces in me.polygons order), use it as is_hair so
    # the SEAM + UV island follow the green border instead of gen_seams' coarse 2-line hairline. The
    # high-res chain gives a clean hairline that the coarse mesh can't produce on its own.
    if os.environ.get('GS_USE_ISHAIR'):
        _ish = os.path.join(os.path.dirname(__file__), '_gs_ishair.npy')
        if os.path.exists(_ish):
            _ext = np.load(_ish)
            if len(_ext) == n_faces:
                is_hair = _ext.astype(bool)
                print(f"[greenborder] is_hair OVERRIDDEN by green-border bridge: {int(is_hair.sum())} hair faces")
            else:
                print(f"[greenborder] _gs_ishair len {len(_ext)} != n_faces {n_faces}; kept gen_seams is_hair")
        else:
            print("[greenborder] GS_USE_ISHAIR set but _gs_ishair.npy missing; kept gen_seams is_hair")
    _cand = np.where(is_hair)[0]
    print(f"[versionA-drive] smoothed-density region: {int(is_hair.sum())} hair faces "
          f"(diffuse8, thr0.30, mass-cull, SOLID+close2, speck-cull -> closed hairline, temple kept)")

    # ── KAN-8 VIZ EXPORT ──────────────────────────────────────────────────────
    # Save the hair/body classification (the SEAMS source) as a renderable mesh so
    # the boundary can be drawn and checked offline, with no full pipeline run.
    try:
        _vz_tris = []; _vz_hair = []; _vz_shell = []; _vz_rough = []; _vz_fn = []; _vz_face = []
        for _vf in range(n_faces):
            _vs = lv_flat[loop_starts[_vf]:loop_starts[_vf] + loop_totals[_vf]]
            for _vk in range(1, len(_vs) - 1):
                _vz_tris.append((int(_vs[0]), int(_vs[_vk]), int(_vs[_vk + 1])))
                _vz_hair.append(bool(is_hair[_vf]))
                _vz_shell.append(bool(_shell[_vf]))
                _vz_rough.append(float(_rough[_vf]))
                _vz_fn.append(face_normal[_vf])
                _vz_face.append(_vf)
        np.savez(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz',
                 co=co.astype('float32'),
                 tris=np.array(_vz_tris, dtype=np.int32),
                 tris_hair=np.array(_vz_hair, dtype=bool),
                 tris_shell=np.array(_vz_shell, dtype=bool),
                 tris_rough=np.array(_vz_rough, dtype='float32'),
                 tris_fn=np.array(_vz_fn, dtype='float32'),
                 tris_face=np.array(_vz_face, dtype=np.int32))
        print(f"[KAN8-hairUV] viz export -> _genseams_viz.npz ({int(is_hair.sum())} hair faces)")
    except Exception as _ve:
        print(f"[KAN8-hairUV] viz export failed: {_ve!r}")
    # ── GS_DUMP_VIZ early-exit (KAN-9) ────────────────────────────────────────
    # The green-border bridge needs is_hair + _genseams_viz, but NOT a full UV pass. Exit here so the
    # pipeline runs gen_seams ONCE: this dump-only pass (no UV, no save) feeds the bridge, then a single
    # GS_USE_ISHAIR pass does the UV/seam/save. Running gen_seams twice as full passes (UV then UV again
    # on its own output) is what corrupted the geometry (the -8.6e28 vertex). One full pass = no corruption.
    if os.environ.get('GS_DUMP_VIZ'):
        print("[gen_seams] GS_DUMP_VIZ -> dumped is_hair + _genseams_viz, exiting before UV (no save)")
        sys.exit(0)
    # KAN-8 (material split for the FBX export): save per-FACE hair flags, aligned to
    # me.polygons == last_seams.obj face order.  The FBX exporter (ComfyUI-FlattenLight)
    # can then assign hair faces a BLONDE material instead of relying on the
    # collision-prone UV texture -- the per-face detection is on the SAME mesh the FBX
    # uses, so no re-mapping is needed.
    try:
        np.save(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams_hairfaces.npy',
                np.asarray(is_hair, dtype=bool))
        print(f"[KAN8-material] saved {int(is_hair.sum())} hair-face flags / {n_faces} faces "
              f"-> last_seams_hairfaces.npy")
    except Exception as _me:
        print(f"[KAN8-material] hair-face flag save failed: {_me!r}")

    # OLD hairline seam (is_hair boundary) REMOVED -- it produced a ragged band, not a hairline.
    # is_hair is still used below for the FBX material + the body-cut stripping; the SEAM itself is now
    # the path3 LOOP (post-fixed around the face), marked by _hairline_pipe.mark_loop_seam after the
    # body seams (see "Mark seams"). This is the v8 chain, living in the pipeline.
    print(f"[KAN8-hairUV] hair-shell faces={int(is_hair.sum())}/{n_faces} "
          f"(material + body-strip only; hairline seam = path3 loop, marked below)")
except Exception as _e:
    print(f"[KAN8-hairUV] FAILED ({_e!r}) — hair-shell seam skipped")

# ── Keep body-cuts OUT of the hair ────────────────────────────────────────────
# The anatomical cuts (chin ring, back-center head/neck lines, etc.) were built to
# unwrap a humanoid BODY and run straight through the head — and thus through the
# hair.  Remove every seam edge whose both faces fall inside the hair (grown by a
# small margin), so those cuts stop SHORT of the hairline.  The hairline boundary
# itself is one-hair-one-skin per edge so it survives — the hair ends up a single
# island bounded by the hairline.
try:
    _hd = is_hair; _hair_ok = True
except NameError:
    _hair_ok = False
if _hair_ok:
    # Keep ARM/SHOULDER/WRIST rings complete (they isolate the limbs), but DO strip the
    # EAR ring-cut where it is BURIED inside hair (both adjacent faces hair): a hair-
    # covered ear is part of the hair island, so its ring-cut there is just an internal
    # seam splitting the hair UV (the user's "seam through solid hair").  Where the ear is
    # skin, its cut edges are hair|skin (never both-hair) -> never stripped -> ear stays
    # isolated exactly as before.
    _keep_struct = _struct_seams - _ear_seams
    # ── MARGIN: grow the hair mask a few face-rings so we strip not just edges
    #    strictly INSIDE the hair but also any seam that comes within touching
    #    distance of it (handles is_hair detection gaps + ear cuts creeping in).
    #    The TRUE hairline (one hair-face, one skin-face) is preserved explicitly
    #    below, so the hair still ends up a single bounded island.
    _HAIR_MARGIN = int(os.environ.get('GS_HAIR_MARGIN', '2'))   # face-rings; 0 = strict inside-only
    _hair_margin = is_hair.copy()
    for _ in range(max(0, _HAIR_MARGIN)):
        _gm = np.zeros(n_faces, dtype=bool)
        np.logical_or.at(_gm, _pairs[:, 0], _hair_margin[_pairs[:, 1]])
        np.logical_or.at(_gm, _pairs[:, 1], _hair_margin[_pairs[:, 0]])
        _hair_margin |= _gm
    _rm = 0; _rm_ear = 0
    for _ei, _fcs in edge_faces.items():
        if len(_fcs) != 2:
            continue
        _fa, _fb = _fcs[0], _fcs[1]
        # Preserve the real hairline boundary (hair on one side, skin on the other).
        if is_hair[_fa] != is_hair[_fb]:
            continue
        # Strip edges whose BOTH faces are inside the hair-or-margin band.
        if _hair_margin[_fa] and _hair_margin[_fb]:
            _k = _edge_key(_ei)
            if _k in all_seam and _k not in _keep_struct:
                if _k in _ear_seams: _rm_ear += 1
                all_seam.discard(_k); _rm += 1
    print(f"[hair-island] margin={_HAIR_MARGIN} rings: stripped {_rm} edges in/near hair "
          f"({_rm_ear} were ear-ring edges) "
          f"(kept {len(_keep_struct & all_seam)} arm/shoulder ring edges complete)")
    # ── KAN-8 FINAL HAIR-ISLAND INVARIANT (deterministic; verified on last_seams) ────
    # The margin-strip above only REMOVES cuts near the hair — it never CLOSES the
    # boundary, so wherever no body-cut happened to run along the hair edge (the whole
    # crown-top + back + sides), the hair stayed connected to the body: select-by-seam
    # then leaks the body OR the only thing bounding a shell are stray interior cuts that
    # shatter it into many pieces (the "side missing" / 27-piece shatter). Enforce the
    # invariant the user asked for directly: (1) EVERY hair⇄skin edge IS a seam → each
    # natural hair shell is fully bounded, split off from the body, no leak; (2) NO
    # hair⇄hair edge is a seam → no internal/closing cut splits a shell. Nothing is
    # dropped (no keep-largest) so no side goes missing; body⇄body seams are untouched so
    # the body UV is unchanged. Verified on last_seams: 8 clean leak-free shells, biggest
    # spans both sides. The front forehead hairline is already seamed (path3 loop /
    # face-chart) so it is preserved — the closures here are crown-top/back/side only.
    _added_b = 0; _stripped_i = 0
    for _ei, _fcs in edge_faces.items():
        if len(_fcs) != 2:
            continue
        _fa, _fb = _fcs[0], _fcs[1]
        _k = _edge_key(_ei)
        if is_hair[_fa] != is_hair[_fb]:               # hair⇄skin boundary → CLOSE it
            if _k not in all_seam:
                all_seam.add(_k); _added_b += 1
        elif is_hair[_fa] and is_hair[_fb]:            # hair⇄hair interior → STRIP it
            if _k in all_seam:
                all_seam.discard(_k); _stripped_i += 1
    print(f"[hair-island] FINAL invariant: closed boundary +{_added_b}, stripped interior "
          f"-{_stripped_i}  (each hair shell = one clean island, nothing dropped)")
    # DIAGNOSTIC: any hair|hair seam edges left, and any SKIN region sitting inside the hair?
    _hh = sum(1 for _e2, _f2 in edge_faces.items()
              if len(_f2) == 2 and is_hair[_f2[0]] and is_hair[_f2[1]] and _edge_key(_e2) in all_seam)
    print(f"[verify] hair|hair seam edges STILL in all_seam: {_hh}")
    try:
        _sp = np.arange(n_faces)
        def _sf2(a):
            while _sp[a] != a:
                _sp[a] = _sp[_sp[a]]; a = _sp[a]
            return a
        for _a2, _b2 in _pairs.tolist():
            if (not is_hair[_a2]) and (not is_hair[_b2]):
                _r1, _r2 = _sf2(int(_a2)), _sf2(int(_b2))
                if _r1 != _r2: _sp[_r1] = _r2
        _sr = np.array([_sf2(_x) for _x in range(n_faces)]); _msk = ~is_hair
        _su, _sc = np.unique(_sr[_msk], return_counts=True); _o = np.argsort(_sc)[::-1]
        print("[verify] top skin components (size | z-range | ymid | inside-hair?):")
        for _i in _o[:6]:
            _m = (_sr == _su[_i]) & _msk
            _ih = (face_z[_m].min() > 0.30)   # wholly above the chin => sitting up in the hair
            print(f"   {_sc[_i]:6d} | z[{face_z[_m].min():.2f},{face_z[_m].max():.2f}] | "
                  f"ymid {np.median(face_centroid[_m,1]):+.2f} | inside_hair={_ih}")
    except Exception as _de:
        print("[verify] skin-comp diag failed:", repr(_de))

print(f"\nTotal seam edges: {len(all_seam)}")

# ── FACE-ISLAND SEAM (KAN-9) ──────────────────────────────────────────────────
# The front face collapses to ~0 UV area inside the hair island: its convex bump
# squishes to a sliver under the ANGLE_BASED unwrap, so it can't be textured (no
# texels). Seam the front face off into its OWN UV chart so it unwraps with real,
# proportional area. This only ADDS a chart boundary — it does NOT change is_hair,
# so the body/hair classification (and the seam-stripping that broke the body before)
# is untouched. Region = forward-facing + central + head-height (generous front face).
try:
    # Prefer the MediaPipe face region (FACE BRIDGE pass B — robust per-character); fall
    # back to the geometric front-face region if the bridge mask isn't present/valid.
    if os.environ.get('GS_NO_FACECHART'):
        raise RuntimeError('face-chart disabled (A/B test)')
    _geo_face = ((face_normal[:, 1] < -0.30) & (face_z > 0.36) & (np.abs(face_x) < 0.15))
    _mp_path = os.path.join(os.path.dirname(__file__), '_gs_faceregion.npy')
    _mp = None
    if os.path.exists(_mp_path):
        _m = np.load(_mp_path).astype(bool)
        if _m.shape[0] == n_faces and int(_m.sum()) > 100:
            _mp = _m
    if _mp is not None:
        # MediaPipe is ALREADY a clean face oval (forehead->chin, no ear). Only fill interior
        # holes (eyes/nostrils). Do NOT dilate/_close it — that grew it past the hairline into
        # the hair+ear and split the lower face (the bad seam through the lips).
        _facechart = _solid(_mp)
        print(f"[face-chart] MediaPipe face region (clean, no dilate): {int(_facechart.sum())} faces")
    else:
        # geometric region is speckled -> close + solid-fill into one contiguous patch, drop specks.
        _facechart = _cc_keep(_solid(_close(_geo_face, 3)), 300)
        print(f"[face-chart] geometric fallback: {int(_facechart.sum())} faces")
    # (1) STRIP existing seam edges INSIDE the face (anatomical cuts that run through it and
    # split it — the lip/chin seam). (2) ADD the face boundary. Result: the face is ONE clean
    # island bounded only by its outline.
    _fc_strip = 0
    for _eidx, _fcs in edge_faces.items():
        if len(_fcs) == 2 and bool(_facechart[_fcs[0]]) and bool(_facechart[_fcs[1]]):
            _k = _edge_key(_eidx)
            if _k in all_seam:
                all_seam.discard(_k); _fc_strip += 1
    # (2) The face boundary oval makes the face its OWN UV island, but it draws a visible cut
    # right across the front face (cheeks/jaw). The user wants the front face free of seams
    # (only the hairline). DROP the oval by default: the strip above already removed every
    # anatomical cut inside the face, and without the oval the face stays connected down through
    # the neck to the body-skin island (it is NOT enclosed by the hairline, so it does not
    # collapse). GS_FACE_OVAL=1 restores the dedicated face island.
    _fc_added = 0
    if os.environ.get('GS_FACE_OVAL'):
        for _eidx, _fcs in edge_faces.items():
            if len(_fcs) == 2 and (bool(_facechart[_fcs[0]]) != bool(_facechart[_fcs[1]])):
                all_seam.add(_edge_key(_eidx)); _fc_added += 1
    print(f"[face-chart] region={int(_facechart.sum())} faces; stripped {_fc_strip} internal "
          f"cuts, +{_fc_added} boundary (oval {'ON' if _fc_added else 'OFF'}) -> face seam-free")
except Exception as _fce:
    print(f"[face-chart] skipped: {_fce!r}")

# ── GENITAL-ISLAND SEAM (KAN-18) ──────────────────────────────────────────────
# The donor genital must be its OWN UV island so it can be textured separately (the body's front-image
# projection never sees between the legs -> it smears a dark blob there). The pelvis graft saves a per-FACE
# genital flag (_gs_isgenital.npy, aligned to THIS mesh). Strip anatomical cuts that run through it (the
# hip ring crosses the crotch), then add its boundary -> one clean genital chart. (Mirrors the face-chart.)
try:
    _gn_path = os.path.join(os.path.dirname(__file__), '_gs_isgenital.npy')
    _gen = None
    if os.path.exists(_gn_path):
        _g = np.load(_gn_path).astype(bool)
        if _g.shape[0] == n_faces and int(_g.sum()) > 20:
            _gen = _g
    if _gen is not None:
        _gn_strip = 0; _gn_add = 0
        for _eidx, _fcs in edge_faces.items():
            if len(_fcs) == 2:
                _ia, _ib = bool(_gen[_fcs[0]]), bool(_gen[_fcs[1]])
                _k = _edge_key(_eidx)
                if _ia and _ib:                       # interior to the genital -> strip anatomical cuts
                    if _k in all_seam: all_seam.discard(_k); _gn_strip += 1
                elif _ia != _ib:                      # boundary -> add (genital becomes its own island)
                    all_seam.add(_k); _gn_add += 1
        print(f"[genital-chart] region={int(_gen.sum())} faces; stripped {_gn_strip} internal cuts, "
              f"+{_gn_add} boundary -> genital is its own UV island")
    else:
        print(f"[genital-chart] no valid _gs_isgenital.npy at {_gn_path} (skipped)")
except Exception as _gne:
    print(f"[genital-chart] skipped: {_gne!r}")

# ── Mark seams ────────────────────────────────────────────────────────────────
for edge in me.edges: edge.use_seam = False
eidx = {}
for edge in me.edges:
    k = (min(edge.vertices[0], edge.vertices[1]), max(edge.vertices[0], edge.vertices[1]))
    eidx[k] = edge.index
marked = 0
for k in all_seam:
    if k in eidx: me.edges[eidx[k]].use_seam = True; marked += 1
print(f"Marked {marked} seam edges")

# ── HAIRLINE SEAM (KAN-9): the path3 LOOP, post-fixed around the face, marked here ON TOP of the body
#    seams (replaces the old is_hair boundary). All logic in _hairline_pipe so a bake runs EXACTLY this.
try:
    import importlib.util as _hpilu
    _hpp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_hairline_pipe.py')
    _hps = _hpilu.spec_from_file_location('_hairline_pipe', _hpp)
    _hpm = _hpilu.module_from_spec(_hps); _hps.loader.exec_module(_hpm)
    _hpm.mark_loop_seam(me, eidx, os.path.dirname(os.path.abspath(__file__)))
except Exception as _hpe:
    import traceback as _tb; _tb.print_exc()
    print(f"[hairline-pipe] FAILED ({_hpe!r}) -- no hairline seam this run")

# ── BACK-FOLD seam: marked AFTER mark_loop_seam so its de-double/prune NEVER touches these edges (that is
#    why the loop-merged version erased the real seams). The fold edges are gb-mesh (== export) indices ->
#    eidx maps them straight. Add-only: it can only set use_seam=True, never clear it. ──
try:
    _foldp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gb_fold.npz')
    if os.path.exists(_foldp):
        _fe = np.load(_foldp)['edges']; _nfm = 0; _nbr = 0
        _bvadj = defaultdict(set)                                      # export-mesh vertex adjacency, to bridge
        for _e in me.edges:                                           # quad diagonals (gb-mesh tri edge that is a
            _bvadj[int(_e.vertices[0])].add(int(_e.vertices[1]))      # quad's diagonal has no export edge -> gap)
            _bvadj[int(_e.vertices[1])].add(int(_e.vertices[0]))
        for _u, _w in _fe:
            _u, _w = int(_u), int(_w); _k = (min(_u, _w), max(_u, _w))
            if _k in eidx:
                me.edges[eidx[_k]].use_seam = True; _nfm += 1
            else:
                _common = _bvadj[_u] & _bvadj[_w]                     # detour through the quad's shared vertex
                if _common:
                    _mid = min(_common, key=lambda m: (me.vertices[_u].co - me.vertices[m].co).length + (me.vertices[m].co - me.vertices[_w].co).length)
                    for _a, _b in ((_u, _mid), (_mid, _w)):
                        _ee = eidx.get((min(_a, _b), max(_a, _b)))
                        if _ee is not None:
                            me.edges[_ee].use_seam = True
                    _nbr += 1
        print(f"[back-fold] marked {_nfm} + bridged {_nbr} of {len(_fe)} back-seam edges (separate, after the loop)")
except Exception as _bfe:
    import traceback as _bft; _bft.print_exc(); print(f"[back-fold] skipped ({_bfe!r})")

# ── KAN-8 FINAL MESH GUARD: enforce the hair-island invariant on the FINAL use_seam, AFTER
#    the loop + back-fold (both mark edges directly, bypassing all_seam, so they can leave
#    hair⇄hair cuts the all_seam pass never sees — the back-fold IS a back-of-hair relief
#    cut). The user's rule, guaranteed on what actually ships: (1) EVERY hair⇄skin edge is a
#    seam → each natural hair shell is fully bounded / split off from the body / no leak;
#    (2) NO hair⇄hair edge is a seam → no internal cut splits a shell (back hair stays one
#    clean fillable piece). Nothing is dropped (no side goes missing); body⇄body and the
#    skin-side face-chart seams are untouched. This is exactly the state verified on
#    last_seams (8 clean leak-free shells, biggest spans both sides).
try:
    _gb_add = 0; _gb_strip = 0
    for _ei2, _fcs2 in edge_faces.items():
        if len(_fcs2) != 2:
            continue
        _fa2, _fb2 = _fcs2[0], _fcs2[1]
        _e2 = eidx.get(_edge_key(_ei2))
        if _e2 is None:
            continue
        if is_hair[_fa2] != is_hair[_fb2]:                 # hair⇄skin boundary → must be a seam
            if not me.edges[_e2].use_seam:
                me.edges[_e2].use_seam = True; _gb_add += 1
        elif is_hair[_fa2] and is_hair[_fb2]:              # hair⇄hair interior → must NOT be a seam
            if me.edges[_e2].use_seam:
                me.edges[_e2].use_seam = False; _gb_strip += 1
    print(f"[hair-island] FINAL MESH GUARD: closed boundary +{_gb_add}, stripped interior "
          f"-{_gb_strip}  (hair = clean bounded shells on the shipped seams)")
except Exception as _ge:
    import traceback as _gt; _gt.print_exc(); print(f"[hair-island] mesh guard skipped ({_ge!r})")

# ── UV Unwrap ─────────────────────────────────────────────────────────────────
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.004)
try:
    bpy.ops.uv.pack_islands(margin=0.002)
    print("UV islands packed")
except Exception as e:
    print(f"pack_islands: {e}")
bpy.ops.object.mode_set(mode='OBJECT')

# ── Guarantee UVs fill [0,1] — normalize if pack left them in a tiny corner ──
uv_layer = me.uv_layers.active
if uv_layer:
    raw = np.array([list(ud.uv) for ud in uv_layer.data], dtype=np.float32)
    u_min, u_max = float(raw[:, 0].min()), float(raw[:, 0].max())
    v_min, v_max = float(raw[:, 1].min()), float(raw[:, 1].max())
    u_range = u_max - u_min; v_range = v_max - v_min
    fill = max(u_range, v_range)
    print(f"UV fill: u=[{u_min:.3f},{u_max:.3f}] v=[{v_min:.3f},{v_max:.3f}]  fill={fill:.3f}")
    if fill < 0.8:
        print("UV fill < 0.8, normalizing to [0,1]...")
        scale = 0.98 / fill
        for i, ud in enumerate(uv_layer.data):
            ud.uv = ((raw[i, 0] - u_min) * scale, (raw[i, 1] - v_min) * scale)
        print("UV normalized")
print("UV unwrap done")

# ── Island count ──────────────────────────────────────────────────────────────
bm2 = bmesh.new(); bm2.from_mesh(me)
bm2.faces.ensure_lookup_table(); bm2.edges.ensure_lookup_table()
uv_layer = bm2.loops.layers.uv.active
visited = set(); island_sizes = []
fadj = {f.index: set() for f in bm2.faces}
for e in bm2.edges:
    if len(e.link_faces) == 2:
        fa, fb = e.link_faces; vs = {e.verts[0].index, e.verts[1].index}
        uva = {lp.vert.index: tuple(lp[uv_layer].uv) for lp in fa.loops if lp.vert.index in vs}
        uvb = {lp.vert.index: tuple(lp[uv_layer].uv) for lp in fb.loops if lp.vert.index in vs}
        same = all(abs(uva[vi][0] - uvb[vi][0]) < 1e-5 and abs(uva[vi][1] - uvb[vi][1]) < 1e-5
                   for vi in vs if vi in uva and vi in uvb)
        if same: fadj[fa.index].add(fb.index); fadj[fb.index].add(fa.index)
for fi in range(len(bm2.faces)):
    if fi in visited: continue
    stack = [fi]; size = 0
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur); size += 1; stack.extend(fadj[cur] - visited)
    island_sizes.append(size)
bm2.free()
island_sizes.sort(reverse=True)
n_islands = len(island_sizes)
print(f"\nUV islands: {n_islands}")
print(f"Top 15: {island_sizes[:15]}")
print(f"Avg: {sum(island_sizes) / max(n_islands, 1):.1f}")
print(f"Top 10 cover {sum(island_sizes[:10])}/{sum(island_sizes)} faces "
      f"({100 * sum(island_sizes[:10]) / max(sum(island_sizes), 1):.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
# Save back to the blend we actually opened (so testing on last_seams_A/B.blend stays
# per-character and never clobbers). In the pipeline this is still last_seams.blend.
out = bpy.data.filepath or r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.blend'
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f"Saved: {out}")
