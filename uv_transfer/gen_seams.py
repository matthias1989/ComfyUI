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
import bpy, bmesh, heapq, sys
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

# ── RING CUTS (face-centroid crossing — guaranteed UV barriers on manifold mesh) ─

# Chin seam — ring just BELOW the chin separating the head/face island from the neck.
# z=0.385: above arm z_hi=0.38 so full-ring cut (no x filter) is safe.
all_seam |= face_cut_z("Chin seam",      0.385)

# Neck ring at z=0.35 — sits at the neck-shoulder junction, just BELOW the neck.
# Arms span z=0.24–0.38 but are at |x|>0.075; x_lo/hi=±0.10 selects only neck faces.
all_seam |= face_cut_z("Neck ring",      0.35, x_lo=-0.10, x_hi=0.10)

# Torso — navel cut (above buttocks)
all_seam |= face_cut_z("Navel cut",      0.12)

# Arm/body separation (shoulder and wrist X-plane rings)
all_seam |= face_cut_x("L shoulder cut", -0.075, 0.24, 0.38)
all_seam |= face_cut_x("R shoulder cut", +0.075, 0.24, 0.38)
all_seam |= face_cut_x("L wrist cut",    -0.40,  0.24, 0.38)
all_seam |= face_cut_x("R wrist cut",    +0.40,  0.24, 0.38)

# Hip cut — no x filter: closes the ring around the full pelvis/crotch cross-section,
# eliminating the extra belly island caused by open front geometry at crotch level
all_seam |= face_cut_z("Hip cut",        -0.03)

# Ankle separation
all_seam |= face_cut_z("L ankle cut",    -0.36, x_lo=-0.18, x_hi=0.0)
all_seam |= face_cut_z("R ankle cut",    -0.36, x_lo= 0.0,  x_hi=0.18)

# ── LENGTH SEAMS (allow cylinders to unroll flat) ─────────────────────────────
all_seam |= length_seam("L arm seam", -0.075, -0.40, 0.24, 0.38)
all_seam |= length_seam("R arm seam", +0.075, +0.40, 0.24, 0.38)

all_seam |= back_seam("L leg seam",  -0.03, -0.41, -0.22, -0.002)
all_seam |= back_seam("R leg seam",  -0.03, -0.41,  0.002,  0.22)

# Head seam: back-center cut from chin seam up to crown of head.
# x filter widened to ±0.08 so the crown (which can sit slightly off xc due to
# voxel discretisation) is included, giving the seam a higher starting point.
all_seam |= back_seam("Head seam",   0.50,  0.385, xc - 0.08, xc + 0.08)
# Neck seam: back-center cut through the neck strip between chin and shoulder rings
all_seam |= back_seam("Neck seam",   0.385, 0.35,  xc - 0.05, xc + 0.05)
all_seam |= back_seam("Torso seam",  0.35,  0.12, xc - 0.06, xc + 0.06)
all_seam |= back_seam("Boxer seam",  0.12, -0.03, xc - 0.06, xc + 0.06)

print(f"\nTotal seam edges: {len(all_seam)}")

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
out = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.blend'
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f"Saved: {out}")
