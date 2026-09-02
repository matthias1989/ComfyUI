"""Paint the ACTUAL bake output: the _hairfill HAIR REGION (blue) + its boundary = the real UV seam (green).
This is what the bake marks, NOT the path3 loop. Build mesh from _frozen/gb_mesh.npz. Render 4 views."""
import bpy, os, math, numpy as np
from mathutils import Vector
UV = os.path.dirname(os.path.abspath(__file__))
for _o in list(bpy.data.objects):
    bpy.data.objects.remove(_o, do_unlink=True)
d = np.load(os.path.join(UV, "_frozen", "gb_mesh.npz"))
co = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); nv = len(co); nf = len(fv)
fill = np.load(os.path.join(UV, "_frozen", "fill.npz"))["faces"]
ishair = np.zeros(nf, bool); ishair[fill[fill < nf]] = True
# boundary faces = faces on a hair<->non-hair edge (the SEAM)
e = np.sort(np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], 0), 1)
fid = np.tile(np.arange(nf), 3)
order = np.lexsort((e[:, 1], e[:, 0])); es = e[order]; fids = fid[order]
same = (es[1:] == es[:-1]).all(1)
fa = fids[:-1][same]; fb = fids[1:][same]
bnd = ishair[fa] != ishair[fb]
seamf = np.zeros(nf, bool); seamf[fa[bnd]] = True; seamf[fb[bnd]] = True
me = bpy.data.meshes.new("gb"); me.from_pydata(co.tolist(), [], fv.tolist()); me.update()
obj = bpy.data.objects.new("gb", me); sc = bpy.context.scene; sc.collection.objects.link(obj)
me.materials.clear()
def m(n, c):
    x = bpy.data.materials.new(n); x.use_nodes = False; x.diffuse_color = c; me.materials.append(x)
m('skin', (0.80, 0.78, 0.74, 1)); m('hair', (0.45, 0.60, 0.88, 1)); m('seam', (0.05, 0.85, 0.15, 1))
midx = np.zeros(nf, dtype=np.int32); midx[ishair] = 1; midx[seamf] = 2
me.polygons.foreach_set('material_index', midx); me.update()
print('[region] hair %d faces, seam-boundary %d faces' % (int(ishair.sum()), int(seamf.sum())))
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 700; sc.render.resolution_y = 800
if sc.world is None: sc.world = bpy.data.worlds.new('w')
sc.world.color = (0.12, 0.12, 0.14)
mn = Vector((float(co[:, 0].min()), float(co[:, 1].min()), float(co[:, 2].min())))
mx = Vector((float(co[:, 0].max()), float(co[:, 1].max()), float(co[:, 2].max())))
ctr = (mn + mx) * 0.5; dim = mx - mn
cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
cd.ortho_scale = 1.05 * max(dim.x, dim.y, dim.z); dd = max(dim.x, dim.y, dim.z) * 3
def shot(view, fn2):
    c = ctr
    if view == 'front': cam.location = c + Vector((0, -dd, 0)); cam.rotation_euler = (math.radians(90), 0, 0)
    if view == 'back':  cam.location = c + Vector((0,  dd, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(180))
    if view == 'left':  cam.location = c + Vector((-dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(-90))
    if view == 'right': cam.location = c + Vector(( dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    sc.render.filepath = fn2; bpy.ops.render.render(write_still=True)
for v in ('front', 'back', 'left', 'right'):
    shot(v, os.path.join(UV, "_frozen", f"region_{v}.png"))
for _sc in bpy.data.screens:
    for _ar in _sc.areas:
        if _ar.type == 'VIEW_3D':
            for _sp in _ar.spaces:
                if _sp.type == 'VIEW_3D': _sp.shading.type = 'SOLID'; _sp.shading.color_type = 'MATERIAL'; _sp.shading.light = 'FLAT'
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(UV, "_frozen", "region.blend"))
print('[region] rendered + saved _frozen/region.blend')
