"""DIAGNOSTIC: show the hair mask and the raw hair<->skin BOUNDARY (the proposed dynamic hairline).
light-blue = hair faces (so we can judge hair detection), green = boundary verts (hair meets skin).
No path3, no creases, no facemask. Build mesh straight from _frozen/gb_mesh.npz. Render 4 views + head zoom + save blend."""
import bpy, os, math, numpy as np
from mathutils import Vector
UV = os.path.dirname(os.path.abspath(__file__))
for _o in list(bpy.data.objects):
    bpy.data.objects.remove(_o, do_unlink=True)
d = np.load(os.path.join(UV, "_frozen", "gb_mesh.npz"))
co = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); hair = d["hair"].astype(bool)
nv = len(co); nf = len(fv)
# hair<->skin boundary = adjacent face pairs where exactly one is hair
e = np.sort(np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], 0), 1)
fid = np.tile(np.arange(nf), 3)
order = np.lexsort((e[:, 1], e[:, 0])); es = e[order]; fids = fid[order]
same = (es[1:] == es[:-1]).all(1)
fa = fids[:-1][same]; fb = fids[1:][same]; eb = es[:-1][same]   # shared edge -> its two faces
bnd = hair[fa] != hair[fb]
bnd_edges = eb[bnd]
bvert = np.zeros(nv, bool); bvert[np.unique(bnd_edges.reshape(-1))] = True
print('[diag] hair faces %d/%d  boundary edges %d  boundary verts %d' % (int(hair.sum()), nf, int(bnd.sum()), int(bvert.sum())))
me = bpy.data.meshes.new("gb"); me.from_pydata(co.tolist(), [], fv.tolist()); me.update()
obj = bpy.data.objects.new("gb", me); sc = bpy.context.scene; sc.collection.objects.link(obj)
def faces_of(vmask): return vmask[fv].any(1)
bndf = faces_of(bvert)
me.materials.clear()
def m(n, c):
    x = bpy.data.materials.new(n); x.use_nodes = False; x.diffuse_color = c; me.materials.append(x)
m('skin', (0.80, 0.78, 0.74, 1)); m('hair', (0.55, 0.72, 0.92, 1)); m('boundary', (0.05, 0.85, 0.15, 1))
midx = np.zeros(nf, dtype=np.int32); midx[hair] = 1; midx[bndf] = 2
me.polygons.foreach_set('material_index', midx); me.update()
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 700; sc.render.resolution_y = 800
if sc.world is None: sc.world = bpy.data.worlds.new('w')
sc.world.color = (0.12, 0.12, 0.14)
mn = Vector((float(co[:, 0].min()), float(co[:, 1].min()), float(co[:, 2].min())))
mx = Vector((float(co[:, 0].max()), float(co[:, 1].max()), float(co[:, 2].max())))
ctr = (mn + mx) * 0.5; dim = mx - mn
cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
def shot(view, fn2, head):
    cz = (mx.z - 0.11 * dim.z) if head else ctr.z
    cd.ortho_scale = (0.34 * dim.z) if head else (1.10 * max(dim.x, dim.y, dim.z))
    c = Vector((ctr.x, ctr.y, cz)); dd = max(dim.x, dim.y, dim.z) * 3
    if view == 'front': cam.location = c + Vector((0, -dd, 0)); cam.rotation_euler = (math.radians(90), 0, 0)
    if view == 'back':  cam.location = c + Vector((0,  dd, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(180))
    if view == 'left':  cam.location = c + Vector((-dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(-90))
    if view == 'right': cam.location = c + Vector(( dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    sc.render.filepath = fn2; bpy.ops.render.render(write_still=True)
for v in ('front', 'back', 'left', 'right'):
    shot(v, os.path.join(UV, "_frozen", f"diag_{v}.png"), head=True)
shot('front', os.path.join(UV, "_frozen", "diag_body.png"), head=False)
for _sc in bpy.data.screens:
    for _ar in _sc.areas:
        if _ar.type == 'VIEW_3D':
            for _sp in _ar.spaces:
                if _sp.type == 'VIEW_3D': _sp.shading.type = 'SOLID'; _sp.shading.color_type = 'MATERIAL'; _sp.shading.light = 'FLAT'
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(UV, "_frozen", "diag.blend"))
print('[diag] rendered + saved _frozen/diag.blend')
