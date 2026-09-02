"""Show (don't remove) the faces I'd treat as EAR and JAW-CREASE, so the user can confirm
the detection before any exclusion.
  EAR  = points laterally OUTWARD (normal x aligned with position x) + lateral + head height.
  JAW  = high crease at jaw/neck height.
Renders the hair mask with ear=green, jaw=cyan, 3/4 + side."""
import bpy, os, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64); fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]
nx, ny, nz = fn[:, 0], fn[:, 1], fn[:, 2]
nv = co.shape[0]; flat = tris.ravel()
vn = np.zeros((nv, 3)); np.add.at(vn, flat, np.repeat(fn, 3, axis=0)); vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
sm = vn[tris].mean(1); sm /= np.linalg.norm(sm, axis=1, keepdims=True) + 1e-9
crease = 1 - np.einsum('ij,ij->i', fn, sm)

ear = (np.abs(nx) > 0.6) & (np.sign(nx) == np.sign(x)) & (np.abs(x) > 0.05) & (z > 0.36) & (z < 0.52)
jaw = (crease > 0.03) & (z > 0.28) & (z < 0.40) & (np.abs(x) > 0.03) & (np.abs(x) < 0.13) & (nz < 0.0)
print("[ear] ear faces=%d  jaw faces=%d" % (int(ear.sum()), int(jaw.sum())))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56])
GRN = np.array([0.1, 0.95, 0.1]); CYN = np.array([0.1, 0.85, 0.95])
fc = np.where(H[:, None], BL[None, :], SK[None, :])
fc = np.where(jaw[:, None], CYN[None, :], fc)
fc = np.where(ear[:, None], GRN[None, :], fc)

def render(off, nm):
    rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel()); me.update()
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

render((2.3, -2.0, 0.3), "_ed_qR.png")
render((3.0, 0.0, 0.2), "_ed_side.png")
print("[ear] DONE")
