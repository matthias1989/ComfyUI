import bpy, numpy as np
o = bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type == 'MESH')
me = o.data
co = np.array([v.co[:] for v in me.vertices])
se = np.array([(e.vertices[0], e.vertices[1]) for e in me.edges if e.use_seam])
print('FILE=%s' % bpy.data.filepath)
print('VERTS=%d EDGES=%d SEAMS=%d' % (len(me.vertices), len(me.edges), len(se)))
print('mesh bbox: X[%.3f,%.3f] Y[%.3f,%.3f] Z[%.3f,%.3f]' % (
    co[:, 0].min(), co[:, 0].max(), co[:, 1].min(), co[:, 1].max(), co[:, 2].min(), co[:, 2].max()))
if len(se):
    mid = (co[se[:, 0]] + co[se[:, 1]]) / 2
    rng = co.max(0) - co.min(0); ha = int(np.argmax(rng))
    thr = co[:, ha].min() + 0.78 * rng[ha]
    h = mid[:, ha] > thr
    print('height axis=%d range=%.3f  top-region thr=%.3f' % (ha, rng[ha], thr))
    print('seam edges in TOP 22%% (head/hairline band): %d' % int(h.sum()))
    print('seam mid bbox: X[%.3f,%.3f] Y[%.3f,%.3f] Z[%.3f,%.3f]' % (
        mid[:, 0].min(), mid[:, 0].max(), mid[:, 1].min(), mid[:, 1].max(), mid[:, 2].min(), mid[:, 2].max()))
