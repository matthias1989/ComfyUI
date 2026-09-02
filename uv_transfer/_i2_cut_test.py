"""Prototype the center-only orientation cut for issue-2 (offline, on charB viz).
Remove hair that is forward-facing (nz<NZ) in the center column (|x|<XW) over the forehead
band -> the center hairline recedes up to the forehead->crown flip (~the green arc), while
the side framing (|x|>XW) is untouched. Renders forehead close-up before vs after.
Args: -- NZ XW   (defaults 0.20 0.06)"""
import bpy, os, sys, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
XW = float(a[0]) if len(a) > 0 else 0.07    # center half-width
ZC = float(a[1]) if len(a) > 1 else 0.48    # hairline height: cut front forehead hair below this
vz = np.load(os.path.join(D, "_viz_B80.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64); fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]; ny = fn[:, 1]
cut = H & (np.abs(x) < XW) & (z > 0.40) & (z < ZC) & (ny < -0.30)   # front-facing center forehead below ZC
newH = H & ~cut
print("[i2cut] XW=%.2f ZC=%.2f  cut %d faces  (hair %d -> %d)" % (XW, ZC, int(cut.sum()), int(H.sum()), int(newH.sum())))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56])

def shot(mask, nm):
    fc = np.where(mask[:, None], BL[None, :], SK[None, :])
    rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel()); me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1200; sc.render.resolution_y = 1150
    tgt = Vector((0.0, 0.0, 0.45))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.24
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector((0.0, -3.0, 0.0)); f = (tgt - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

shot(H,    "_i2c_before.png")
shot(newH, "_i2c_after.png")
print("[i2cut] DONE")
