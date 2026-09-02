"""Classify seam edges and render: RED = hairline (hair|skin boundary), BLUE = cut
(same-class seam edge, e.g. the ear ring-cut), gold=hair, grey=skin. Env RS_NPY, RS_OUT."""
import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
nf = len(me.polygons)
hair = np.load(os.environ['RS_NPY']).astype(bool)
out = os.environ['RS_OUT']

e2f = defaultdict(list)
for p in me.polygons:
    for ek in p.edge_keys:
        e2f[ek].append(p.index)
hairline_f = np.zeros(nf, bool); cut_f = np.zeros(nf, bool)
for e in me.edges:
    if not e.use_seam:
        continue
    fs = e2f.get(e.key, [])
    if len(fs) == 2:
        if hair[fs[0]] != hair[fs[1]]:
            hairline_f[fs[0]] = hairline_f[fs[1]] = True
        else:
            cut_f[fs[0]] = cut_f[fs[1]] = True
    else:
        for f in fs:
            cut_f[f] = True

mat = obj.matrix_world
cen = np.zeros((nf, 3))
for p in me.polygons:
    c = mat @ p.center
    cen[p.index] = (c.x, c.y, c.z)

lf = np.empty(len(me.loops), dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total):
        lf[p.loop_start + k] = p.index
GOLD = np.array([0.82, 0.64, 0.20]); SK = np.array([0.66, 0.63, 0.61]); RED = np.array([0.95, 0.06, 0.06]); BLUE = np.array([0.12, 0.45, 0.97])
facecol = np.where(hair[:, None], GOLD[None], SK[None])
facecol = np.where(hairline_f[:, None], RED[None], facecol)
facecol = np.where(cut_f[:, None], BLUE[None], facecol)  # cut on top so the ear ring stands out
rgba = np.concatenate([facecol[lf], np.ones((len(lf), 1))], axis=1).astype(np.float32)
for ca in list(me.color_attributes):
    if ca.name == 'RS':
        me.color_attributes.remove(ca)
col = me.color_attributes.new(name='RS', type='BYTE_COLOR', domain='CORNER')
col.data.foreach_set('color', rgba.ravel())
me.color_attributes.active_color = col
me.update()

sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'FLAT'
sc.display.shading.color_type = 'VERTEX'
sc.render.film_transparent = True

hc = cen[hair] if hair.any() else cen
ctr = Vector((hc[:, 0].mean(), hc[:, 1].mean(), hc[:, 2].mean()))
ext = max(hc[:, 0].max() - hc[:, 0].min(), hc[:, 2].max() - hc[:, 2].min()) * 1.35


def shot(dir_vec, nm):
    sc.render.resolution_x = 1000; sc.render.resolution_y = 1000
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    d = Vector(dir_vec).normalized()
    cam.location = ctr + d * 10.0
    f = (ctr - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


shot((1, 0, 0), out + '_sideR.png')
shot((0.85, -0.85, 0.12), out + '_q34.png')
print('seam-class', out, 'hairline_f=', int(hairline_f.sum()), 'cut_f=', int(cut_f.sum()))
