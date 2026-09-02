"""Final cleanup: un-mark the central neck-band seam edges in a baked blend (the horizontal seam
across the neck that makes the neck part of the hair UV island). Keeps the lateral side seams and
the forehead. Region = central |x-xc| < XW and neck-base z-norm in [ZLO, ZHI]. Saves in place."""
import bpy, numpy as np, os
o = bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type == 'MESH')
me = o.data
co = np.array([v.co[:] for v in me.vertices])
zmin, zmax = co[:, 2].min(), co[:, 2].max()
znv = (co[:, 2] - zmin) / (zmax - zmin)
xc = float((co[:, 0].min() + co[:, 0].max()) / 2.0)
XW = float(os.environ.get('UB_XW', '0.07')); ZLO = float(os.environ.get('UB_ZLO', '0.72')); ZHI = float(os.environ.get('UB_ZHI', '0.90'))
ymid = float((co[:, 1].min() + co[:, 1].max()) / 2.0)   # face = -Y, so FRONT = y < ymid
n = 0; lat = 0
for e in me.edges:
    if not e.use_seam:
        continue
    a, b = e.vertices[0], e.vertices[1]
    mx = abs((co[a, 0] + co[b, 0]) / 2.0 - xc); mz = (znv[a] + znv[b]) / 2.0; my = (co[a, 1] + co[b, 1]) / 2.0
    # un-mark only the FRONT (low-y) central neck-level edges = the front-neck/throat band. Keeps the
    # nape (high y), the lateral sides, and the forehead (z-norm above ZHI).
    if mx < XW and ZLO < mz < ZHI and my < ymid:
        e.use_seam = False; n += 1
    else:
        lat += 1
print('[unmark-band] un-marked %d central neck-band edges; %d lateral side edges remain' % (n, lat))
# REBUILD the green HAIR_SEAM overlay from the CURRENT use_seam so the visible tube matches the edits
# (the overlay is a separate object; without this it still shows the pre-edit path).
_lv = set()
for e in me.edges:
    if e.use_seam:
        _lv.add(int(e.vertices[0])); _lv.add(int(e.vertices[1]))
_old = bpy.data.objects.get('HAIR_SEAM')
if _old:
    bpy.data.objects.remove(_old, do_unlink=True)
_eps = 0.0018 * float((co.max(0) - co.min(0)).max())
_vmap = {}; _nv = []; _nf = []
for p in me.polygons:
    if not any(int(v) in _lv for v in p.vertices):
        continue
    _fi = []
    for v in p.vertices:
        v = int(v)
        if v not in _vmap:
            _vmap[v] = len(_nv); _nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * _eps)[:]))
        _fi.append(_vmap[v])
    _nf.append(_fi)
_om = bpy.data.meshes.new('HAIR_SEAM'); _om.from_pydata(_nv, [], _nf)
_oo = bpy.data.objects.new('HAIR_SEAM', _om); bpy.context.scene.collection.objects.link(_oo)
_oo.matrix_world = o.matrix_world.copy()
_mat = bpy.data.materials.new('SEAMMAT'); _mat.use_nodes = False; _mat.diffuse_color = (0.0, 1.0, 0.15, 1.0); _om.materials.append(_mat)
print('[unmark-band] rebuilt HAIR_SEAM overlay: %d faces (matches use_seam now)' % len(_nf))
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
