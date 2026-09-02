"""Render the pathfinding clean seam (_Bcleanseam.npz edges) in RED over the hair mask
(gold=hair, grey=skin). Env RS_NPY (hair mask), RS_SEAM (clean edges npz), RS_OUT."""
import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data; nf = len(me.polygons)
hair = np.load(os.environ['RS_NPY']).astype(bool)
edges = np.load(os.environ['RS_SEAM'])['edges']
out = os.environ['RS_OUT']

vpair2f = defaultdict(list)
for p in me.polygons:
    vs = list(p.vertices); k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        vpair2f[(u, w) if u < w else (w, u)].append(p.index)
seamface = np.zeros(nf, bool)
for u, w in edges:
    for f in vpair2f.get((int(u), int(w)) if u < w else (int(w), int(u)), []):
        seamface[f] = True

mat = obj.matrix_world; cen = np.zeros((nf, 3))
for p in me.polygons:
    c = mat @ p.center; cen[p.index] = (c.x, c.y, c.z)
lf = np.empty(len(me.loops), dtype=np.int64)
for p in me.polygons:
    for k2 in range(p.loop_total):
        lf[p.loop_start + k2] = p.index
G = np.array([0.82, 0.64, 0.20]); S = np.array([0.66, 0.63, 0.61]); R = np.array([0.95, 0.06, 0.06])
fcol = np.where(hair[:, None], G[None], S[None]); fcol = np.where(seamface[:, None], R[None], fcol)
rgba = np.concatenate([fcol[lf], np.ones((len(lf), 1))], 1).astype(np.float32)
for ca in list(me.color_attributes):
    if ca.name == 'RS':
        me.color_attributes.remove(ca)
col = me.color_attributes.new(name='RS', type='BYTE_COLOR', domain='CORNER')
col.data.foreach_set('color', rgba.ravel()); me.color_attributes.active_color = col; me.update()
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'
sc.display.shading.color_type = 'VERTEX'; sc.render.film_transparent = True
hc = cen[hair] if hair.any() else cen
ctr = Vector((hc[:, 0].mean(), hc[:, 1].mean(), hc[:, 2].mean())); ext = max(np.ptp(hc[:, 0]), np.ptp(hc[:, 2])) * 1.4
def shot(dv, nm):
    sc.render.resolution_x = 1000; sc.render.resolution_y = 1000
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = ctr + Vector(dv).normalized() * 10.0
    cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
shot((1, 0, 0), out + '_sideR.png'); shot((0.85, -0.85, 0.12), out + '_q34.png')
print('cleanseam-render', out, 'seamfaces', int(seamface.sum()))
