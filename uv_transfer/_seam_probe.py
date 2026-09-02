"""FIND the 'depression on top of the hair' the user described, and show it next to the
current hair mask — no changes, just looking. Concavity = smoothed-centroid vs centroid
along the normal (gen_seams' own measure): >0 = concave (a dip/depression), <0 = convex.
Renders front + a tilted top view of the crown."""
import bpy, os, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
nv = co.shape[0]; flat = tris.ravel()
P = cen.copy()
for _ in range(12):                                 # smooth over a broader area -> catch broad dips
    vs = np.zeros((nv, 3)); vc = np.zeros(nv)
    np.add.at(vs, flat, np.repeat(P, 3, axis=0)); np.add.at(vc, flat, 1.0)
    P = (vs / np.maximum(vc, 1)[:, None])[tris].mean(1)
conc = np.einsum('ij,ij->i', P - cen, fn)           # >0 concave (depression)
print("[sp] conc range %.4f .. %.4f  (median %.4f)" % (conc.min(), conc.max(), np.median(conc)))

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
    sc.render.resolution_x = 1100; sc.render.resolution_y = 1100
    zmax = co[:, 2].max(); tgt = Vector((0.0, 0.0, zmax - 0.07))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.34
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector(off); f = (tgt - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

# concavity heatmap: red = concave (depression), blue = convex, grey = flat (auto-scaled)
K = 0.55 / (np.percentile(np.abs(conc), 96) + 1e-9)
print("[sp] color scale K=%.0f" % K)
r = 0.45 + np.clip(conc * K, 0, 0.55)
b = 0.45 + np.clip(-conc * K, 0, 0.55)
g = np.full_like(r, 0.45)
cmap = np.stack([r, g, b], axis=1)
# hair mask
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.60, 0.58, 0.56])
hmap = np.where(H[:, None], BL[None, :], SK[None, :])

render(cmap, (0.0, -3.0, 0.4), "_sp_conc_front.png")
render(hmap, (0.0, -3.0, 0.4), "_sp_hair_front.png")

# candidate "depression" faces: NOT hair, concave (a dip), upper head, embedded in hair
hf = H.astype(np.float64)
for _ in range(4):                                  # fraction of nearby faces that are hair
    vs = np.zeros(nv); vc = np.zeros(nv)
    np.add.at(vs, flat, np.repeat(hf, 3)); np.add.at(vc, flat, 1.0)
    hf = (vs / np.maximum(vc, 1))[tris].mean(1)
cand = (~H) & (cen[:, 2] > 0.44) & (hf > 0.55)      # non-hair faces embedded in the upper hair
print("[sp] depression candidates: %d  (mean conc %.4f)" % (int(cand.sum()), conc[cand].mean() if cand.any() else 0))
GRN = np.array([0.1, 0.95, 0.1])
cmap2 = np.where(cand[:, None], GRN[None, :], hmap)
render(cmap2, (0.0, -3.0, 0.4), "_sp_cand_front.png")
render(cmap2, (0.0, -1.6, 2.6), "_sp_cand_top.png")
print("[sp] DONE")
