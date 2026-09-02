import bpy, numpy as np, os
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=D + r"\last_seams.blend")
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
nf = len(me.polygons); nv = len(me.vertices)
co = np.empty(nv * 3); me.vertices.foreach_get('co', co); co = co.reshape(nv, 3)
hair = np.load(D + r"\last_seams_hairfaces.npy").astype(bool)
print("polys=%d hairfaces=%d hair=%d" % (nf, len(hair), int(hair.sum())))
cen = np.zeros((nf, 3))
for p in me.polygons: cen[p.index] = p.center[:]
# normalize z like gen_seams to read bands
zmn, zmx = co[:, 2].min(), co[:, 2].max(); zs = 0.91 / (zmx - zmn); zoff = -0.41 - zmn * zs
cz = cen[:, 2] * zs + zoff
xsp = co[:, 0].max() - co[:, 0].min(); xc = (co[:, 0].min() + co[:, 0].max()) / 2
cx = (cen[:, 0] - xc) * (0.92 / xsp)
print("hair faces by normalized-z band (chin~0.385, crown~0.50; body is z<0.34):")
for lo, hi in [(0.40, 0.55), (0.34, 0.40), (0.28, 0.34), (0.15, 0.28), (-0.10, 0.15), (-0.5, -0.10)]:
    m = hair & (cz >= lo) & (cz < hi)
    lat = hair & (cz >= lo) & (cz < hi) & (np.abs(cx) > 0.16)
    print("  z[%+.2f,%+.2f) hair=%5d  of which |x|>0.16 (arms/shoulders)=%5d" % (lo, hi, int(m.sum()), int(lat.sum())))
# color + render
lf = np.empty(len(me.loops), dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total): lf[p.loop_start + k] = p.index
GOLD = np.array([0.93, 0.74, 0.27]); SK = np.array([0.62, 0.6, 0.58])
fc = np.where(hair[lf][:, None], GOLD[None], SK[None])
rgba = np.concatenate([fc, np.ones((len(lf), 1))], axis=1).astype(np.float32)
col = me.color_attributes.new(name="B", type='BYTE_COLOR', domain='CORNER')
col.data.foreach_set("color", rgba.ravel()); me.update()
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
sc.render.resolution_x = 900; sc.render.resolution_y = 1200
zc2 = co[:, 2]; tgt = Vector((0.0, 0.0, (zc2.max() + zc2.min()) * 0.5 + (zc2.max() - zc2.min()) * 0.15))

def shot(loc, nm):
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = (zc2.max() - zc2.min()) * 0.75
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector(loc); f = (tgt - cam.location).normalized(); cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = D + "\\" + nm; bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)
shot((0, -3, 0), "_bleed_front.png")
shot((3, -1.5, 0), "_bleed_34.png")
print("done")
