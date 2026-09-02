"""Conservative hair-boundary smoothing test (charB). Diffuse the hair mask N passes and
re-threshold at 0.5: interior stays hair, exterior stays skin, only the jagged BOUNDARY
gets rounded. Low N = only small-scale jaggedness touched (smooth seams ~unchanged).
Renders before vs after, front + 3/4, and reports how many faces changed."""
import bpy, os, sys, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
ITERS = int(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else 2   # low to start
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64); H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
nv = co.shape[0]; flat = tris.ravel()

def smooth(mask, iters):
    m = mask.astype(np.float64)
    for _ in range(iters):
        vs = np.zeros(nv); vc = np.zeros(nv)
        np.add.at(vs, flat, np.repeat(m, 3)); np.add.at(vc, flat, 1.0)
        m = (vs / np.maximum(vc, 1))[tris].mean(1)
    return m > 0.5

newH = smooth(H, ITERS)
chg = newH != H
print("[smooth] iters=%d  hair %d -> %d  changed %d (added %d, removed %d)  in-head changed %d"
      % (ITERS, int(H.sum()), int(newH.sum()), int(chg.sum()),
         int((newH & ~H).sum()), int((~newH & H).sum()),
         int((chg & (cen[:, 2] > 0.30) & (np.abs(cen[:, 0]) < 0.16)).sum())))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
BL = np.array([0.92, 0.76, 0.28]); SK = np.array([0.6, 0.58, 0.56])

def render(mask, off, nm):
    fc = np.where(mask[:, None], BL[None, :], SK[None, :])
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

render(H,    (0.0, -3.0, 0.3), "_st_before_front.png")
render(newH, (0.0, -3.0, 0.3), "_st_after_front.png")
render(H,    (2.3, -2.0, 0.3), "_st_before_qR.png")
render(newH, (2.3, -2.0, 0.3), "_st_after_qR.png")
print("[smooth] DONE")
