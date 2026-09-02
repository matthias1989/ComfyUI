"""Build a debug mesh straight from _frozen/gb_mesh.npz (so vertex order == the npy masks) and paint
whatever layers exist: blue=facemask(facevert), orange=concave(crease), magenta=kept. Render 4 views
(the head is turned, so front/back/left/right all help). Run: blender --background --python _frozen_paint.py"""
import bpy, os, math, numpy as np
from mathutils import Vector
UV = os.path.dirname(os.path.abspath(__file__))
for _o in list(bpy.data.objects):            # drop the default startup Cube/Camera/Light
    bpy.data.objects.remove(_o, do_unlink=True)
d = np.load(os.path.join(UV, "_frozen", "gb_mesh.npz"))
co = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); nv = len(co); nf = len(fv)
me = bpy.data.meshes.new("gb"); me.from_pydata(co.tolist(), [], fv.tolist()); me.update()
obj = bpy.data.objects.new("gb", me)
sc = bpy.context.scene; sc.collection.objects.link(obj)

def load(name):
    p = os.path.join(UV, "_frozen", name)
    return np.load(p).astype(bool) if os.path.exists(p) else None
fm = load("filtered.npy")          # all filter regions (face + throat + ears) for the blue layer
if fm is None: fm = load("facevert.npy")
loop = None
lp = os.path.join(UV, "_frozen", "loop.npz")
if os.path.exists(lp):
    led = np.load(lp)["loop"]; loop = np.zeros(nv, bool); loop[np.unique(led.reshape(-1))] = True
cre = None
cp = os.path.join(UV, "_frozen", "crease.npz")
if os.path.exists(cp):
    ced = np.load(cp)["edges"]; cre = np.zeros(nv, bool); cre[np.unique(ced.reshape(-1))] = True

def faces_of(vmask):
    if vmask is None: return np.zeros(nf, bool)
    return vmask[fv].any(1)
fmf = faces_of(fm); loopf = faces_of(loop); cref = faces_of(cre)
me.materials.clear()
def m(n, c):
    x = bpy.data.materials.new(n); x.use_nodes = False; x.diffuse_color = c; me.materials.append(x)
m('skin', (0.78, 0.76, 0.72, 1)); m('facemask', (0.45, 0.60, 0.85, 1)); m('concave', (0.95, 0.55, 0.10, 1)); m('loop', (0.05, 0.85, 0.15, 1))
midx = np.zeros(nf, dtype=np.int32); midx[fmf] = 1; midx[cref] = 2; midx[loopf] = 3   # all concave (orange) over face mask, green seam on top
me.polygons.foreach_set('material_index', midx); me.update()
print('[fz-paint] facemask %d, concave %d, loop %d faces' % (int(fmf.sum()), int(cref.sum()), int(loopf.sum())))

sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x = 700; sc.render.resolution_y = 800
if sc.world is None: sc.world = bpy.data.worlds.new('w')
sc.world.color = (0.12, 0.12, 0.14)
mn = Vector((float(co[:, 0].min()), float(co[:, 1].min()), float(co[:, 2].min())))
mx = Vector((float(co[:, 0].max()), float(co[:, 1].max()), float(co[:, 2].max())))
ctr = (mn + mx) * 0.5; dim = mx - mn
cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
# frame on the HEAD (top of Z)
hz = mx.z - 0.11 * dim.z; cd.ortho_scale = 0.34 * dim.z; dd = max(dim.x, dim.y, dim.z) * 3   # HEAD zoom
def shot(view, fn2):
    c = Vector((ctr.x, ctr.y, hz))
    if view == 'front':  cam.location = c + Vector((0, -dd, 0)); cam.rotation_euler = (math.radians(90), 0, 0)
    if view == 'back':   cam.location = c + Vector((0,  dd, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(180))
    if view == 'left':   cam.location = c + Vector((-dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(-90))
    if view == 'right':  cam.location = c + Vector(( dd, 0, 0)); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    sc.render.filepath = fn2; bpy.ops.render.render(write_still=True)
for v in ('front', 'back', 'left', 'right'):
    shot(v, os.path.join(UV, "_frozen", f"paint_{v}.png"))
for _sc in bpy.data.screens:
    for _ar in _sc.areas:
        if _ar.type == 'VIEW_3D':
            for _sp in _ar.spaces:
                if _sp.type == 'VIEW_3D': _sp.shading.type = 'SOLID'; _sp.shading.color_type = 'MATERIAL'; _sp.shading.light = 'FLAT'
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(UV, "_frozen", "painted.blend"))
print('[fz-paint] rendered 4 views + saved _frozen/painted.blend')
