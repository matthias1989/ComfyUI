"""Understanding-only: show WHERE the version-A detector fires (convex + crease) next to
where the hair mask boundary lands. If the detector lights up on the ears/jaw, that
confirms it's dragging the hair boundary (the seam) toward them. No changes."""
import bpy, os, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
nv = co.shape[0]; flat = tris.ravel()
P = cen.copy()
for _ in range(4):
    vs = np.zeros((nv, 3)); vc = np.zeros(nv)
    np.add.at(vs, flat, np.repeat(P, 3, axis=0)); np.add.at(vc, flat, 1.0)
    P = (vs / np.maximum(vc, 1)[:, None])[tris].mean(1)
conc = np.einsum('ij,ij->i', P - cen, fn)
# crease proxy: variation of face normal from its vertex-averaged normal
vn = np.zeros((nv, 3)); np.add.at(vn, flat, np.repeat(fn, 3, axis=0)); vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
crease = 1 - np.einsum('ij,ij->i', fn, vn[tris].mean(1) / (np.linalg.norm(vn[tris].mean(1), axis=1, keepdims=True) + 1e-9))
wallA = (conc < np.percentile(conc, 18)) | (crease > 0.02)   # version-A 'hair-ness' fires here
print("[diag] version-A fires on %d faces (%.0f%%)" % (int(wallA.sum()), 100 * wallA.mean()))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')

def render(fc, off, nm):
    rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel()); me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1100; sc.render.resolution_y = 1250
    zmax = co[:, 2].max(); tgt = Vector((0.0, 0.0, zmax - 0.11))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.34
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector(off); f = (tgt - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

RED = np.array([0.85, 0.15, 0.15]); GRY = np.array([0.6, 0.6, 0.62])
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56])
wmap = np.where(wallA[:, None], RED[None, :], GRY[None, :])
hmap = np.where(H[:, None], BL[None, :], SK[None, :])
render(wmap, (0.0, -3.0, 0.3), "_sd_wallA_front.png")
render(wmap, (2.3, -2.0, 0.3), "_sd_wallA_qR.png")     # 3/4 to see the ear
render(hmap, (2.3, -2.0, 0.3), "_sd_hair_qR.png")
print("[diag] DONE")
