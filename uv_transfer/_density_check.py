import bpy, numpy as np, os
UV = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
o = bpy.data.objects.get('geometry_0'); me = o.data
uvl = me.uv_layers.active.data
nf = len(me.polygons)
a3 = np.array([p.area for p in me.polygons])
auv = np.zeros(nf)
for p in me.polygons:
    uvs = [uvl[li].uv for li in p.loop_indices]; n = len(uvs); s = 0.0
    for i in range(n):
        x1, y1 = uvs[i]; x2, y2 = uvs[(i + 1) % n]; s += x1 * y2 - x2 * y1
    auv[p.index] = abs(s) * 0.5
mesh_d = auv.sum() / max(a3.sum(), 1e-12)
fr = np.load(os.path.join(UV, '_gs_faceregion.npy'))
if len(fr) == nf:
    fm = fr.astype(bool)
    face_d = auv[fm].sum() / max(a3[fm].sum(), 1e-12)
    print('>>> FACE texel-density ratio = %.3f  (face/mesh; 1.0=avg)   [%d face faces]' % (face_d / mesh_d, int(fm.sum())))
else:
    print('>>> faceregion len mismatch', len(fr), nf)
