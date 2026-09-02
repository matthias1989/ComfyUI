"""KAN-8 OFFLINE hair seed module. Runs on last_seams.blend = THE EXPORTED FBX MESH.
NOTE (2026-06-21): the SEED detection below (drape) was the working part — it catches the full hair
(crown + ponytail) where the old shell detector collapsed to 3%. The downstream seam/ear assembly I
built on top REGRESSED the hairline and was scrapped. Treat this as the seed stage only; start hairline
work from the existing pipeline's path3 loop, not a region-boundary replacement.

  shell  -- body surface a short distance behind a face (existing gen_seams signal; misses free hair)
  rough  -- strand/clump normal variation (existing; catches the textured crown only)
  drape  -- march inward (-normal), SKIP the hair shell's own anti-aligned backface, flag if an ALIGNED
            body layer sits within reach -> catches free-hanging smooth hair. crown-anchored component
            + forward/face/ear cuts remove torso/face/ear false positives.

Run:  blender last_seams.blend --background --python gen_hair_seams.py
Env (dev): GS_DRAPE_DIST(0.16) GS_DRAPE_ALIGN(0.25) GS_NO_VIZ(skip viz write)
"""
import bpy, bmesh, sys, os, math
import numpy as np
from collections import defaultdict

UV = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

# ── mesh + normalized co (IDENTICAL to gen_seams.py) ──────────────────────────
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
face_centroid = np.zeros((n_faces, 3))
for f in range(n_faces):
    face_centroid[f] = co[lv[ls[f]:ls[f] + lt[f]]].mean(0)
face_z = face_centroid[:, 2]; face_x = face_centroid[:, 0]; face_y = face_centroid[:, 1]

face_normal = np.zeros((n_faces, 3))
for f in range(n_faces):
    vp = co[lv[ls[f]:ls[f] + lt[f]]]
    nv = np.cross(vp[1] - vp[0], vp[2] - vp[0]); nl = np.linalg.norm(nv)
    if nl > 1e-12:
        face_normal[f] = nv / nl
if face_normal[face_z > np.percentile(face_z, 98), 2].mean() < 0:
    face_normal = -face_normal

edge_faces = defaultdict(list)
for f in range(n_faces):
    vs = lv[ls[f]:ls[f] + lt[f]]
    for j in range(len(vs)):
        a, b = int(vs[j]), int(vs[(j + 1) % len(vs)])
        edge_faces[(a, b) if a < b else (b, a)].append(f)
_pairs = np.array([fc for fc in edge_faces.values() if len(fc) == 2], dtype=np.int64)

import mathutils
from mathutils.bvhtree import BVHTree
_bverts = [mathutils.Vector(co[i].tolist()) for i in range(n_verts)]
_bpolys = [lv[ls[f]:ls[f] + lt[f]].tolist() for f in range(n_faces)]
_bvh = BVHTree.FromPolygons(_bverts, _bpolys, all_triangles=False)

def _close(m, k):
    for _ in range(k):
        c = np.zeros(n_faces); np.add.at(c, _pairs[:, 0], m[_pairs[:, 1]].astype(float)); np.add.at(c, _pairs[:, 1], m[_pairs[:, 0]].astype(float))
        m = m | (c >= 1.0)
    for _ in range(k):
        c = np.zeros(n_faces); np.add.at(c, _pairs[:, 0], (~m[_pairs[:, 1]]).astype(float)); np.add.at(c, _pairs[:, 1], (~m[_pairs[:, 0]]).astype(float))
        m = m & (c == 0)
    return m

# ── SHELL (existing) ──────────────────────────────────────────────────────────
HAIR_HIT_D = 0.041; SHELL_ALIGN = 0.30
_cand = np.where((face_z > -0.150) & (np.abs(face_x) < 0.200))[0]
shell = np.zeros(n_faces, dtype=bool)
for f in _cand.tolist():
    o = mathutils.Vector((face_centroid[f] - 0.0015 * face_normal[f]).tolist())
    d = mathutils.Vector((-face_normal[f]).tolist())
    h = _bvh.ray_cast(o, d, HAIR_HIT_D)
    if (h[0] is not None and h[2] is not None and h[2] != f and h[3] is not None and h[3] > 0.0021
            and float(np.dot(face_normal[f], face_normal[h[2]])) > SHELL_ALIGN):
        shell[f] = True
shell = _close(shell, 3)

# ── ROUGH (existing) ──────────────────────────────────────────────────────────
_rij = 1.0 - np.sum(face_normal[_pairs[:, 0]] * face_normal[_pairs[:, 1]], axis=1)
rough_v = np.zeros(n_faces); cnt = np.zeros(n_faces)
np.add.at(rough_v, _pairs[:, 0], _rij); np.add.at(rough_v, _pairs[:, 1], _rij)
np.add.at(cnt, _pairs[:, 0], 1.0); np.add.at(cnt, _pairs[:, 1], 1.0)
rough_v /= np.maximum(cnt, 1)
for _ in range(3):
    acc = np.zeros(n_faces); c2 = np.zeros(n_faces)
    np.add.at(acc, _pairs[:, 0], rough_v[_pairs[:, 1]]); np.add.at(acc, _pairs[:, 1], rough_v[_pairs[:, 0]])
    np.add.at(c2, _pairs[:, 0], 1.0); np.add.at(c2, _pairs[:, 1], 1.0)
    rough_v = 0.5 * rough_v + 0.5 * acc / np.maximum(c2, 1)
_zone = (np.abs(face_x) < 0.160) & (face_z > -0.10)
_zmed = float(np.median(rough_v[_zone])) if _zone.any() else 0.0
ROUGH_THR = max(0.009, 4.0 * _zmed)
rough = rough_v > ROUGH_THR

# ── DRAPE: aligned body layer behind a face (long shell + backface march) ─────
D_ZMIN  = float(os.environ.get('GS_DRAPE_ZMIN', '-0.15'))
D_XMAX  = float(os.environ.get('GS_DRAPE_XMAX', '0.20'))
D_LONG  = float(os.environ.get('GS_DRAPE_DIST', '0.16'))
D_ALIGN = float(os.environ.get('GS_DRAPE_ALIGN', '0.25'))
D_GAP   = float(os.environ.get('GS_DRAPE_GAPMIN', '0.003'))
_dz = (face_z > D_ZMIN) & (np.abs(face_x) < D_XMAX)
drape = np.zeros(n_faces, dtype=bool)
for f in np.where(_dz)[0].tolist():
    nrm = face_normal[f]; dirn = mathutils.Vector((-nrm).tolist())
    pos = face_centroid[f] - D_GAP * nrm
    remaining = D_LONG; steps = 0
    while remaining > 0 and steps < 6:
        steps += 1
        h = _bvh.ray_cast(mathutils.Vector(pos.tolist()), dirn, remaining)
        if h[0] is None or h[2] is None or h[3] is None:
            break
        if h[2] != f and float(np.dot(nrm, face_normal[h[2]])) > D_ALIGN and h[3] > D_GAP:
            drape[f] = True; break
        adv = max(h[3], 1e-4) + D_GAP
        pos = np.array(h[0][:]) - D_GAP * nrm; remaining -= adv
drape = _close(drape, 2)

print(f"[seed] shell={int(shell.sum())} ({100*shell.mean():.1f}%)  "
      f"rough={int(rough.sum())} ({100*rough.mean():.1f}%)  "
      f"drape={int(drape.sum())} ({100*drape.mean():.1f}%)  dist={D_LONG} align={D_ALIGN}")
for nm, m in (('shell', shell), ('rough', rough), ('drape', drape)):
    if m.any():
        c = face_centroid[m]
        print(f"   {nm}: z[{c[:,2].min():.2f},{c[:,2].max():.2f}] x[{c[:,0].min():.2f},{c[:,0].max():.2f}] "
              f"y[{c[:,1].min():.2f},{c[:,1].max():.2f}]")

# ── COMBINE: drape = reliable hair layer; keep crown-anchored component, cut torso/face/ear FPs ──
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

_zmax = float(face_z.max()); _zmin = float(face_z.min())
T_chin = _zmax - 0.22 * (_zmax - _zmin)
_frontlow = (face_normal[:, 1] < -0.15) & (face_z < T_chin)
_facecarve = (face_normal[:, 1] < -0.45) & (np.abs(face_x) < 0.15)   # face skin, never hair
_headband = (face_z > T_chin) & (np.abs(face_x) < 0.22)
head_x_max = float(np.abs(face_x[_headband]).max()) if _headband.any() else 0.18
_ear = (face_z > T_chin) & (np.abs(face_x) > 0.62 * head_x_max) & (np.abs(face_normal[:, 0]) > 0.40)
_cut = _frontlow | _facecarve | _ear
_nrm_cut = int((drape & _cut).sum())
drape = drape & ~_cut
print(f"[seed] cut fwd-low + face + ears: T_chin={T_chin:.3f} head_x_max={head_x_max:.3f} "
      f"face={int(_facecarve.sum())} ear={int(_ear.sum())}, removed {_nrm_cut} drape faces")

_lab = _components(drape)
_crown = drape & (face_z > 0.40)
_keep_roots = set(_lab[_crown].tolist())
if _keep_roots:
    hair = drape & np.isin(_lab, list(_keep_roots))
else:
    u, c = np.unique(_lab[drape], return_counts=True)
    hair = drape & (_lab == u[c.argmax()])
print(f"[seed] drape crown-anchored component(s): {int(hair.sum())} ({100*hair.mean():.1f}%)")

for _ in range(2):                                      # grow crown strand detail into adjacent rough
    g = np.zeros(n_faces, dtype=bool)
    np.logical_or.at(g, _pairs[:, 0], hair[_pairs[:, 1]] & rough[_pairs[:, 0]])
    np.logical_or.at(g, _pairs[:, 1], hair[_pairs[:, 1]] & rough[_pairs[:, 0]])
    np.logical_or.at(g, _pairs[:, 1], hair[_pairs[:, 0]] & rough[_pairs[:, 1]])
    np.logical_or.at(g, _pairs[:, 0], hair[_pairs[:, 0]] & rough[_pairs[:, 1]])
    hair = hair | (g & rough)
hair = _close(hair, 2) & _zone
seed = hair
np.save(os.path.join(UV, '_hs_seed.npy'), seed)
np.save(os.path.join(UV, '_hs_signals.npy'), np.stack([shell, rough, drape]))
print(f"[seed] FINAL crown-anchored hair = {int(seed.sum())} ({100*seed.mean():.1f}%) -> _hs_seed.npy")

# ── write path3-bridge mesh+seed (CAUTION: overwrites the shared _genseams_viz.npz; GS_NO_VIZ to skip) ──
if not os.environ.get('GS_NO_VIZ'):
    _tris = []; _tface = []
    for f in range(n_faces):
        vs = lv[ls[f]:ls[f] + lt[f]]
        for j in range(1, len(vs) - 1):
            _tris.append((int(vs[0]), int(vs[j]), int(vs[j + 1]))); _tface.append(f)
    _tris = np.array(_tris, dtype=np.int32); _tface = np.array(_tface, dtype=np.int32)
    _seed_tri = seed[_tface]
    vp = os.path.join(UV, '_genseams_viz.npz')
    bak = os.path.join(UV, '_genseams_viz_PREHS.npz')
    if os.path.exists(vp) and not os.path.exists(bak):
        import shutil; shutil.copy2(vp, bak)
    np.savez(vp, co=co.astype('float32'), tris=_tris, tris_face=_tface, tris_hair=_seed_tri,
             tris_shell=shell[_tface], tris_rough=rough_v[_tface], tris_fn=face_normal[_tface].astype('float32'))
    print(f"[seed] wrote _genseams_viz.npz self-contained ({len(_tris)} tris, {int(_seed_tri.sum())} hair)")

# ── diagnostic render: final hair seed (yellow) + detected ears (blue) on the export mesh ──
def render_signals():
    me.materials.clear()
    cols = {'skin': (0.78, 0.74, 0.70, 1), 'shell': (0.20, 0.45, 0.95, 1),
            'rough': (0.20, 0.85, 0.30, 1), 'drape': (0.95, 0.20, 0.20, 1), 'multi': (0.97, 0.95, 0.25, 1)}
    for k in ['skin', 'shell', 'rough', 'drape', 'multi']:
        mm = bpy.data.materials.new(k); mm.use_nodes = False; mm.diffuse_color = cols[k]; me.materials.append(mm)
    midx = np.zeros(n_faces, dtype=np.int32)
    midx[_ear] = 1
    midx[seed] = 4
    me.polygons.foreach_set("material_index", midx); me.update()
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
    sc.render.resolution_x = 900; sc.render.resolution_y = 1200
    if sc.world is None: sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.12, 0.12, 0.14)
    from mathutils import Vector
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    ctr = (mn + mx) * 0.5; dim = mx - mn
    cam_d = bpy.data.cameras.new("c"); cam_d.type = 'ORTHO'
    cam = bpy.data.objects.new("c", cam_d); sc.collection.objects.link(cam); sc.camera = cam
    def shot(view, fn, focus='full'):
        if focus == 'head':
            c = Vector((ctr.x, ctr.y, mn.z + 0.82 * dim.z)); span = 0.34 * dim.z
        else:
            c = ctr; span = 1.05 * max(dim.x, dim.z)
        cam_d.ortho_scale = span * 2.0; dd = max(dim.x, dim.y, dim.z) * 3
        if view == 'front':
            cam.location = c + Vector((0, -dd, 0)); cam.rotation_euler = (math.radians(90), 0, 0)
        elif view == 'back':
            cam.location = c + Vector((0, dd, 0));  cam.rotation_euler = (math.radians(90), 0, math.radians(180))
        elif view == 'right':
            cam.location = c + Vector((dd, 0, 0));  cam.rotation_euler = (math.radians(90), 0, math.radians(90))
        else:
            cam.location = c + Vector((-dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(-90))
        sc.render.filepath = fn; bpy.ops.render.render(write_still=True); print(f"[seed] wrote {fn}")
    for v in ('front', 'back', 'right', 'left'):
        shot(v, os.path.join(UV, f'_hs_seed_{v}.png'))
    for v in ('front', 'right', 'left'):
        shot(v, os.path.join(UV, f'_hs_head_{v}.png'), focus='head')

render_signals()
print("[seed] done.  yellow=final hair seed  blue=detected ears")
