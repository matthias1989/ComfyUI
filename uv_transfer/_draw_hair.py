"""Fill the HAIR faces (the hair mask from the mesh npz) as a YELLOW overlay so the hair region is
visible directly on the character. Env: DH_MESH (npz with 'hair' per-face bool), DH_RENDER (prefix)."""
import bpy, numpy as np, os
from mathutils import Vector

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH' and o.name != 'HAIR_FILL')
me = obj.data
hair = np.load(os.environ['DH_MESH'], allow_pickle=True)['hair'].astype(bool)
co = np.array([v.co[:] for v in me.vertices])
size = float((co.max(0) - co.min(0)).max())
eps = 0.0015 * size
print('mesh polygons %d, hair-flag len %d, hair faces %d' % (len(me.polygons), len(hair), int(hair.sum())))

vmap = {}; nv = []; nf = []
for p in me.polygons:
    if p.index >= len(hair) or not hair[p.index]:
        continue
    fi = []
    for v in p.vertices:
        v = int(v)
        if v not in vmap:
            vmap[v] = len(nv)
            nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * eps)[:]))
        fi.append(vmap[v])
    nf.append(fi)
old = bpy.data.objects.get('HAIR_FILL')
if old:
    bpy.data.objects.remove(old, do_unlink=True)
om = bpy.data.meshes.new('HAIR_FILL'); om.from_pydata(nv, [], nf)
oo = bpy.data.objects.new('HAIR_FILL', om); bpy.context.scene.collection.objects.link(oo)
oo.matrix_world = obj.matrix_world.copy()
mat = bpy.data.materials.new('HAIRFILLMAT'); mat.use_nodes = False; mat.diffuse_color = (1.0, 0.9, 0.05, 1.0)
om.materials.append(mat)
print('hair fill faces %d' % len(nf))

if os.environ.get('DH_RENDER'):
    out = os.environ['DH_RENDER']
    zn = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min())
    cen = co[zn > 0.5]                       # frame head + neck + drape
    lo = np.percentile(cen, 1, axis=0); hi = np.percentile(cen, 99, axis=0)
    ctr = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2))
    ext = max(hi[0]-lo[0], hi[2]-lo[2]) * 1.25
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.render.film_transparent = True

    def shot(dv, nm):
        sc.render.resolution_x = 800; sc.render.resolution_y = 1000
        cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
        cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = ctr + Vector(dv).normalized() * 10.0
        cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
    shot((0, -1, 0), out + '_front.png'); shot((1, 0, 0), out + '_sideR.png')
    shot((1, 0.7, 0.1), out + '_qback.png'); shot((0, 1, 0), out + '_back.png')
    print('rendered', out)
