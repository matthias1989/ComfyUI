"""Render a given viz file's hair mask (front + 3/4) so before/after can be compared.
Usage: blender --background --python _show_viz.py -- <viz.npz> <tag>"""
import bpy, os, sys, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
args = sys.argv[sys.argv.index('--') + 1:]
vizf, tag = args[0], args[1]
vz = np.load(os.path.join(D, vizf))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64); H = vz['tris_hair'].astype(bool)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56])
fc = np.where(H[:, None], BL[None, :], SK[None, :])
rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
col.data.foreach_set("color", rgba.ravel()); me.update()

def shot(off, nm):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1100; sc.render.resolution_y = 1250
    zmax = co[:, 2].max(); tgt = Vector((0.0, 0.0, zmax - 0.12))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.40
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector(off); f = (tgt - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

shot((0.0, -3.0, 0.3), f"_sv_{tag}_front.png")
shot((2.3, -2.0, 0.3), f"_sv_{tag}_qR.png")
print(f"[show_viz] {vizf} -> _sv_{tag}_*  hair {int(H.sum())}")
