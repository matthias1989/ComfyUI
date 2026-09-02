"""Render the hair mask (gold=hair, grey=skin) of the currently-open blend.
Env: RS_NPY = hairfaces .npy (face order == me.polygons), RS_OUT = output prefix.
Produces <prefix>_front.png (full body) and <prefix>_q34.png (head 3/4 zoom)."""
import bpy, numpy as np, os
from mathutils import Vector

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
nf = len(me.polygons)
hair = np.load(os.environ['RS_NPY']).astype(bool)
out = os.environ['RS_OUT']

mat = obj.matrix_world
cen = np.zeros((nf, 3))
for p in me.polygons:
    c = mat @ p.center
    cen[p.index] = (c.x, c.y, c.z)

lf = np.empty(len(me.loops), dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total):
        lf[p.loop_start + k] = p.index
GOLD = np.array([0.96, 0.74, 0.18]); SK = np.array([0.66, 0.63, 0.61])
fc = np.where(hair[lf][:, None], GOLD[None], SK[None])
rgba = np.concatenate([fc, np.ones((len(lf), 1))], axis=1).astype(np.float32)
for ca in list(me.color_attributes):
    if ca.name == 'RS':
        me.color_attributes.remove(ca)
col = me.color_attributes.new(name='RS', type='BYTE_COLOR', domain='CORNER')
col.data.foreach_set('color', rgba.ravel())
me.color_attributes.active_color = col
me.update()

sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'VERTEX'
sc.render.film_transparent = True

bb = cen  # world face centroids
hc = cen[hair] if hair.any() else cen


def shot(dir_vec, tgt, scale, nm, rx=900, ry=1100):
    sc.render.resolution_x = rx; sc.render.resolution_y = ry
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = scale; cd.clip_start = 0.001; cd.clip_end = 1000
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    d = Vector(dir_vec).normalized()
    cam.location = Vector(tgt) + d * 10.0
    f = (Vector(tgt) - cam.location).normalized()
    cam.rotation_euler = f.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = nm
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


# full-body front
ctr_full = Vector((bb[:, 0].min() + bb[:, 0].max(), 0, bb[:, 2].min() + bb[:, 2].max())) * 0.5
ctr_full.y = (bb[:, 1].min() + bb[:, 1].max()) * 0.5
h_full = (bb[:, 2].max() - bb[:, 2].min()) * 1.06
shot((0, -1, 0), ctr_full, h_full, out + '_front.png', 760, 1180)

# head 3/4 zoom (framed on the hair)
ctr_head = Vector((hc[:, 0].mean(), hc[:, 1].mean(), hc[:, 2].mean()))
ext = max(hc[:, 0].max() - hc[:, 0].min(), hc[:, 2].max() - hc[:, 2].min()) * 1.55
shot((0.85, -0.85, 0.12), ctr_head, ext, out + '_q34.png', 1000, 1000)
print('rendered', out, 'hair=', int(hair.sum()))
