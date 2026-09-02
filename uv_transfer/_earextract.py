import bpy, numpy as np, os
o = bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type == 'MESH')
me = o.data
co = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
seam = np.array([(min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1])) for e in me.edges if e.use_seam], dtype=np.int64)
fv = np.array([[int(v) for v in p.vertices[:3]] for p in me.polygons], dtype=np.int64)
np.savez(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_earlook.npz', co=co, seam=seam, fv=fv)
print('SAVED co=%d seam=%d fv=%d' % (len(co), len(seam), len(fv)))
