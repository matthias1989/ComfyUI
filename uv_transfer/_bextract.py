import bpy, numpy as np, os
o=bpy.data.objects.get('geometry_0'); me=o.data
me.calc_loop_triangles()
mw=o.matrix_world
co=np.array([(mw @ v.co)[:] for v in me.vertices], dtype=np.float64)
tris=np.array([list(t.vertices) for t in me.loop_triangles], dtype=np.int64)
# polygon sides histogram (quad vs tri)
sides={}
for p in me.polygons: sides[len(p.vertices)]=sides.get(len(p.vertices),0)+1
print('B_diag geometry_0: %d verts  %d polys  %d tris(after triangulation)'%(len(co),len(me.polygons),len(tris)))
print('polygon sides:', sides)
np.savez(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_bmesh_raw.npz', co=co, fv=tris)
