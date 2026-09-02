"""Overlay BOTH the hair mask (yellow, faint) AND a loop (green) in one frame so we can see whether the
boundary green sits on the FRONT EDGE of the hair. Env: DB_MESH, DB_LOOP, DB_RENDER."""
import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
co = np.array([v.co[:] for v in me.vertices])
zn = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min())
size = float((co.max(0) - co.min(0)).max())
m = np.load(os.environ['DB_MESH'], allow_pickle=True)
hair = m['hair'].astype(bool)

v2f = defaultdict(list)
for p in me.polygons:
    for v in p.vertices:
        v2f[int(v)].append(p.index)

def add_obj(name, faces, color, eps):
    vmap = {}; nv = []; nf = []
    for f in faces:
        fi = []
        for v in me.polygons[f].vertices:
            v = int(v)
            if v not in vmap:
                vmap[v] = len(nv)
                nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * eps)[:]))
            fi.append(vmap[v])
        nf.append(fi)
    old = bpy.data.objects.get(name)
    if old: bpy.data.objects.remove(old, do_unlink=True)
    mm = bpy.data.meshes.new(name); mm.from_pydata(nv, [], nf)
    oo = bpy.data.objects.new(name, mm); bpy.context.scene.collection.objects.link(oo)
    oo.matrix_world = obj.matrix_world.copy()
    mat = bpy.data.materials.new(name + 'M'); mat.use_nodes = False; mat.diffuse_color = color
    mm.materials.append(mat)

# yellow hair fill (faint)
hf = [p.index for p in me.polygons if p.index < len(hair) and hair[p.index]]
add_obj('HFILL', hf, (0.95, 0.85, 0.10, 1.0), 0.0012 * size)
# green loop
lp = np.load(os.environ['DB_LOOP'])['loop']; lv = set(int(x) for x in np.unique(lp))
lf = sorted({f for v in lv for f in v2f[v]})
add_obj('LOOP', lf, (0.05, 1.0, 0.15, 1.0), 0.004 * size)

out = os.environ['DB_RENDER']
cen = co[zn > 0.55]
lo = np.percentile(cen, 2, axis=0); hi = np.percentile(cen, 98, axis=0)
ctr = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2))
ext = max(hi[0]-lo[0], hi[2]-lo[2]) * 1.45
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.render.film_transparent = True
def shot(dv, nm):
    sc.render.resolution_x = 900; sc.render.resolution_y = 1000
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = ctr + Vector(dv).normalized() * 10.0
    cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
shot((1, 0, 0), out + '_sideR.png'); shot((1, -0.5, 0.1), out + '_q34.png')
print('rendered', out)
