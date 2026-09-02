"""Investigate issue-2: the forehead-centre hairline notch on the smoothed charB.
Find the non-hair faces in the forehead-centre (the dip), report their concavity, and
render a tight forehead close-up with the notch highlighted red — so the fill is grounded."""
import bpy, os, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_viz_B80.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64); fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]
nv = co.shape[0]; flat = tris.ravel()
P = cen.copy()
for _ in range(4):
    vs = np.zeros((nv, 3)); vc = np.zeros(nv)
    np.add.at(vs, flat, np.repeat(P, 3, axis=0)); np.add.at(vc, flat, 1.0)
    P = (vs / np.maximum(vc, 1)[:, None])[tris].mean(1)
conc = np.einsum('ij,ij->i', P - cen, fn)
front = -fn[:, 1] > 0.25
notch = (~H) & front & (np.abs(x) < 0.07) & (z > 0.43) & (z < 0.50)
allc = conc[(np.abs(x) < 0.07) & (z > 0.43) & (z < 0.50)]
print("[issue2] forehead-centre non-hair (notch) faces: %d  mean conc %.4f  (region conc median %.4f, p80 %.4f)"
      % (int(notch.sum()), conc[notch].mean() if notch.any() else 0, np.median(allc), np.percentile(conc, 80)))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56]); RD = np.array([0.9, 0.1, 0.1])
fc = np.where(H[:, None], BL[None, :], SK[None, :])
fc = np.where(notch[:, None], RD[None, :], fc)
rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
col.data.foreach_set("color", rgba.ravel()); me.update()
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
sc.render.resolution_x = 1200; sc.render.resolution_y = 1100
tgt = Vector((0.0, 0.0, 0.455))
cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.20
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
cam.location = tgt + Vector((0.0, -3.0, 0.0)); f = (tgt - cam.location).normalized()
cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
sc.render.filepath = os.path.join(D, "_i2_forehead.png"); bpy.ops.render.render(write_still=True)
print("[issue2] DONE")
