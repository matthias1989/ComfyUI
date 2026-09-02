"""Render the saved seam blend (already has HAIR_SEAM green overlay) framed on the FULL head+neck+drape
so the back-neck drape is visible (the head-framed shots crop it). Env: RW_OUT (prefix)."""
import bpy, numpy as np, os
from mathutils import Vector

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH' and o.name != 'HAIR_SEAM')
me = obj.data
co = np.array([v.co[:] for v in me.vertices])
zn = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min())
# frame from upper chest (znv>0.55) up = head + neck + drape
cen = co[zn > 0.55]
lo = np.percentile(cen, 1, axis=0); hi = np.percentile(cen, 99, axis=0)
ctr = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2))
ext = max(hi[0]-lo[0], hi[2]-lo[2]) * 1.25
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.render.film_transparent = True
out = os.environ['RW_OUT']

def shot(dv, nm):
    sc.render.resolution_x = 800; sc.render.resolution_y = 1000
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = ctr + Vector(dv).normalized() * 10.0
    cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)

shot((1, 0.0, 0.0), out + '_sideR.png')
shot((1, 0.7, 0.1), out + '_qback.png')
shot((0, 1, 0.05), out + '_back.png')
print('rendered wide', out)
