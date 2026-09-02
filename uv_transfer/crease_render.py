"""Clean render of step-1 creases: bare grey mesh + RED crease edges (RS_SEAM). Framed on the
crease region (head). No hair mask shown. Env RS_SEAM, RS_OUT."""
import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data; nf = len(me.polygons)
edges = np.load(os.environ['RS_SEAM'])['edges']
out = os.environ['RS_OUT']
vp2f = defaultdict(list)
for p in me.polygons:
    vs = list(p.vertices); k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        vp2f[(u, w) if u < w else (w, u)].append(p.index)
seamface = np.zeros(nf, bool)
for u, w in edges:
    for f in vp2f.get((int(u), int(w)) if u < w else (int(w), int(u)), []):
        seamface[f] = True
mat = obj.matrix_world; cen = np.zeros((nf, 3))
for p in me.polygons:
    c = mat @ p.center; cen[p.index] = (c.x, c.y, c.z)
lf = np.empty(len(me.loops), dtype=np.int64)
for p in me.polygons:
    for k2 in range(p.loop_total):
        lf[p.loop_start + k2] = p.index
S = np.array([0.70, 0.68, 0.66]); R = np.array([0.95, 0.05, 0.05])
fcol = np.where(seamface[:, None], R[None], np.tile(S, (nf, 1)))
rgba = np.concatenate([fcol[lf], np.ones((len(lf), 1))], 1).astype(np.float32)
for ca in list(me.color_attributes):
    if ca.name == 'RS':
        me.color_attributes.remove(ca)
col = me.color_attributes.new(name='RS', type='BYTE_COLOR', domain='CORNER')
col.data.foreach_set('color', rgba.ravel()); me.color_attributes.active_color = col; me.update()
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'
sc.display.shading.color_type = 'VERTEX'; sc.render.film_transparent = True
hc = cen[seamface] if seamface.any() else cen
lo = np.percentile(hc, 5, axis=0); hi = np.percentile(hc, 95, axis=0)
ctr = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)); ext = max(hi[0] - lo[0], hi[2] - lo[2]) * 1.6
def shot(dv, nm):
    sc.render.resolution_x = 1000; sc.render.resolution_y = 1000
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = ctr + Vector(dv).normalized() * 10.0
    cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
shot((0, -1, 0), out + '_front.png'); shot((1, 0, 0), out + '_sideR.png'); shot((0.85, -0.85, 0.12), out + '_q34.png')
print('crease-render', out, 'creasefaces', int(seamface.sum()))
