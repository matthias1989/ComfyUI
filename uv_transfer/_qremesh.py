import bpy, numpy as np, os
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
d=np.load(os.path.join(UV,'_gs_hires.npz')); co=d['co'].astype(float); fv=d['fv'].astype(np.int64)
me=bpy.data.meshes.new('m'); me.from_pydata(co.tolist(),[],fv.tolist()); me.update()
o=bpy.data.objects.new('m',me); bpy.context.scene.collection.objects.link(o)
bpy.context.view_layer.objects.active=o; o.select_set(True)
print('input: %d verts %d tris'%(len(co),len(fv)))
try:
    bpy.ops.object.quadriflow_remesh(target_faces=150000, use_preserve_sharp=False, use_preserve_boundary=False)
except Exception as e:
    print('quadriflow FAILED:', repr(e))
me2=o.data; me2.calc_loop_triangles()
co2=np.array([v.co[:] for v in me2.vertices],dtype=np.float64)
tris=np.array([list(t.vertices) for t in me2.loop_triangles],dtype=np.int64)
sides={}
for p in me2.polygons: sides[len(p.vertices)]=sides.get(len(p.vertices),0)+1
print('remeshed: %d verts  polys %d  sides %s  %d tris'%(len(co2),len(me2.polygons),sides,len(tris)))
np.savez(os.path.join(UV,'_qmesh_raw.npz'), co=co2, fv=tris)
