"""Geometry-only hairline test: remove forward-facing 'hair' in the forehead band
(the bleed) and keep up-facing crown. Renders before vs after, head-zoom, so we can
see whether the orientation gate gives a clean natural hairline. No image involved."""
import bpy, os, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0

# ORIENTATION GATE: forehead band, hair must point up; forward-facing -> skin
band = (cen[:, 2] > 0.40) & (cen[:, 2] < 0.47) & (np.abs(cen[:, 0]) < 0.08)
fwd  = fn[:, 2] < 0.20
newH = H & ~(H & band & fwd)
print("[geomtest] hair %d -> %d (removed %d forward-facing forehead)"
      % (int(H.sum()), int(newH.sum()), int((H & band & fwd).sum())))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BLOND = np.array([0.92, 0.76, 0.28]); SKIN = np.array([0.60, 0.58, 0.56])

def shot(mask, off, nm):
    fc = np.where(mask[:, None], BLOND[None, :], SKIN[None, :])
    rgba = np.concatenate([np.repeat(fc, 3, axis=0), np.ones((3 * len(tris), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel()); me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1100; sc.render.resolution_y = 1250
    zmax = co[:, 2].max(); tgt = Vector((0.0, 0.0, zmax - 0.105))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = 0.32
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = tgt + Vector(off); fwd_v = (tgt - cam.location).normalized()
    cam.rotation_euler = fwd_v.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

shot(H,    (0.0, -3.0, 0.0), "_gt_before_front.png")
shot(newH, (0.0, -3.0, 0.0), "_gt_after_front.png")
shot(newH, (2.0, -2.2, 0.0), "_gt_after_qR.png")
print("[geomtest] DONE")
