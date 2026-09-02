"""KAN-8 EAR seams — a SEPARATE after-action on last_seams.blend (the export FBX mesh).  Does NOT touch
the hair seams.  KNOWN-INCOMPLETE (2026-06-21): geometric ear detection here was unreliable — the
protrusion test grabbed the jaw concavity behind the ear, not the ear flap; |x|-threshold swept the
head side.  Ear-skin vs hair-skin is a material distinction geometry can't make cleanly (the documented
"geometry wall").  Kept for reference; needs a different approach (head-relative ring, or front-image).
Run: blender last_seams.blend --background --python gen_ear_seams.py
Env (dev): GS_EAR_GROW (def 1)"""
import bpy, sys, os, math
import numpy as np
from collections import defaultdict

UV = os.path.dirname(os.path.abspath(__file__))

obj = bpy.data.objects.get('geometry_0') or next((o for o in bpy.data.objects if o.type == 'MESH'), None)
me = obj.data
n_verts = len(me.vertices); n_faces = len(me.polygons)
co_flat = np.empty(n_verts * 3); me.vertices.foreach_get('co', co_flat); co = co_flat.reshape(n_verts, 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T = np.array([mat[r][3] for r in range(3)])
co = co @ R.T + T
_zmn, _zmx = float(co[:, 2].min()), float(co[:, 2].max())
_xc = float((co[:, 0].min() + co[:, 0].max()) / 2.0); _xsp = float(co[:, 0].max() - co[:, 0].min())
_zs = 0.91 / (_zmx - _zmn); _zo = -0.41 - _zmn * _zs; _xs = 0.92 / _xsp
co[:, 2] = co[:, 2] * _zs + _zo; co[:, 0] = (co[:, 0] - _xc) * _xs; co[:, 1] = co[:, 1] * _zs

lv = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get('vertex_index', lv)
ls = np.empty(n_faces, dtype=np.int32); lt = np.empty(n_faces, dtype=np.int32)
me.polygons.foreach_get('loop_start', ls); me.polygons.foreach_get('loop_total', lt)
fc = np.zeros((n_faces, 3))
for f in range(n_faces):
    fc[f] = co[lv[ls[f]:ls[f] + lt[f]]].mean(0)
face_z = fc[:, 2]; face_x = fc[:, 0]
face_normal = np.zeros((n_faces, 3))
for f in range(n_faces):
    vp = co[lv[ls[f]:ls[f] + lt[f]]]
    nv = np.cross(vp[1] - vp[0], vp[2] - vp[0]); nl = np.linalg.norm(nv)
    if nl > 1e-12:
        face_normal[f] = nv / nl
edge_faces = defaultdict(list)
for f in range(n_faces):
    vs = lv[ls[f]:ls[f] + lt[f]]
    for j in range(len(vs)):
        a, b = int(vs[j]), int(vs[(j + 1) % len(vs)])
        edge_faces[(a, b) if a < b else (b, a)].append(f)
_pairs = np.array([fc2 for fc2 in edge_faces.values() if len(fc2) == 2], dtype=np.int64)

hair = np.zeros(n_faces, dtype=bool)
hp = os.path.join(UV, '_gs_ishair.npy')
if os.path.exists(hp):
    h = np.load(hp)
    if len(h) == n_faces:
        hair = h.astype(bool)
print(f"[ear] hair region loaded: {int(hair.sum())} faces")

# ── EAR detect by PROTRUSION: an ear flap has a skull surface behind an air gap when you march toward
# the head's vertical axis; the smooth skull side does not (ray goes into the head interior). ──
import mathutils
from mathutils.bvhtree import BVHTree
_bverts = [mathutils.Vector(co[i].tolist()) for i in range(n_verts)]
_bpolys = [lv[ls[f]:ls[f] + lt[f]].tolist() for f in range(n_faces)]
_bvh = BVHTree.FromPolygons(_bverts, _bpolys, all_triangles=False)
face_y = fc[:, 1]
_zmax = float(face_z.max())
HEAD_Z0 = _zmax - 0.16 * (_zmax - face_z.min())
_hb = (face_z > HEAD_Z0) & (np.abs(face_x) < 0.20)
head_x_max = float(np.percentile(np.abs(face_x[_hb]), 97)) if _hb.any() else 0.14
GAP = 0.003
cand = np.zeros(n_faces, dtype=bool)
_test = (face_z > HEAD_Z0) & (np.abs(face_x) > 0.45 * head_x_max) & (np.abs(face_x) < 1.6 * head_x_max)
for f in np.where(_test)[0].tolist():
    c = fc[f]; axis = np.array([0.0, c[1], c[2]]); vec = axis - c; dist = float(np.linalg.norm(vec))
    if dist < 1e-4:
        continue
    dirn = vec / dist
    h = _bvh.ray_cast(mathutils.Vector((c + GAP * dirn).tolist()), mathutils.Vector(dirn.tolist()), dist * 0.92 - GAP)
    if h[0] is not None and h[2] is not None and h[2] != f and h[3] is not None and h[3] > GAP:
        cand[f] = True
print(f"[ear] HEAD_Z0={HEAD_Z0:.3f} head_x_max={head_x_max:.3f} protruding candidates={int(cand.sum())}")
XCAP = 1.6 * head_x_max
GROW = int(os.environ.get('GS_EAR_GROW', '1'))

def _components(mask):
    parent = np.arange(n_faces)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    both = mask[_pairs[:, 0]] & mask[_pairs[:, 1]]
    for a, b in _pairs[both].tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_faces)])
_lab = _components(cand)
ear = np.zeros(n_faces, dtype=bool)
for side in (+1, -1):
    m = cand & (np.sign(face_x) == side)
    if not m.any():
        continue
    u, c = np.unique(_lab[m], return_counts=True)
    root = u[c.argmax()]
    ear |= (_lab == root) & cand
for _ in range(GROW):
    g = np.zeros(n_faces, dtype=bool)
    np.logical_or.at(g, _pairs[:, 0], ear[_pairs[:, 1]])
    np.logical_or.at(g, _pairs[:, 1], ear[_pairs[:, 0]])
    ear = ear | (g & (face_z > HEAD_Z0) & (np.abs(face_x) > 0.30 * head_x_max) & (np.abs(face_x) < XCAP))
np.save(os.path.join(UV, '_hs_ear.npy'), ear)
print(f"[ear] FINAL ear faces = {int(ear.sum())} -> _hs_ear.npy")

# ── render: hair=red, ear=cyan, skin=gray ──
me.materials.clear()
def m(name, col):
    mm = bpy.data.materials.new(name); mm.use_nodes = False; mm.diffuse_color = col; me.materials.append(mm)
m("skin", (0.78, 0.74, 0.70, 1)); m("hair", (0.85, 0.12, 0.12, 1)); m("ear", (0.15, 0.85, 0.90, 1))
midx = np.zeros(n_faces, dtype=np.int32); midx[hair] = 1; midx[ear] = 2
me.polygons.foreach_set("material_index", midx); me.update()
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 900; sc.render.resolution_y = 1100
if sc.world is None: sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.12, 0.12, 0.14)
from mathutils import Vector
bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ctr = (mn + mx) * 0.5; dim = mx - mn
cam_d = bpy.data.cameras.new("c"); cam_d.type = 'ORTHO'
cam = bpy.data.objects.new("c", cam_d); sc.collection.objects.link(cam); sc.camera = cam
def shot(view, fn):
    c = Vector((ctr.x, ctr.y, mn.z + 0.82 * dim.z)); span = 0.34 * dim.z
    cam_d.ortho_scale = span * 2.0; dd = max(dim.x, dim.y, dim.z) * 3
    if view == 'front':
        cam.location = c + Vector((0, -dd, 0)); cam.rotation_euler = (math.radians(90), 0, 0)
    elif view == 'right':
        cam.location = c + Vector((dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    else:
        cam.location = c + Vector((0, dd, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(180))
    sc.render.filepath = fn; bpy.ops.render.render(write_still=True); print(f"[ear] wrote {fn}")
for v in ('front', 'right', 'back'):
    shot(v, os.path.join(UV, f'_hs_ear_{v}.png'))
