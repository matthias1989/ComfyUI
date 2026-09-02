import bpy, numpy as np, os
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
d=np.load(os.path.join(UV,'_gs_hires.npz')); co=d['co'].astype(float); fv=d['fv'].astype(np.int64)
bbox=co.max(0)-co.min(0); diag=float(np.linalg.norm(bbox))
vsize=diag/float(os.environ.get('VOX_DIV','350'))
bpy.ops.wm.read_factory_settings(use_empty=True)
me=bpy.data.meshes.new('m'); me.from_pydata(co.tolist(),[],fv.tolist()); me.update()
o=bpy.data.objects.new('m',me); bpy.context.scene.collection.objects.link(o)
bpy.context.view_layer.objects.active=o; o.select_set(True)
o.data.remesh_voxel_size=vsize; o.data.remesh_voxel_adaptivity=0.0
print('voxel_size=%.5f (diag=%.3f)'%(vsize,diag))
bpy.ops.object.voxel_remesh()
me2=o.data; sides={}
for p in me2.polygons: sides[len(p.vertices)]=sides.get(len(p.vertices),0)+1
me2.calc_loop_triangles()
co2=np.array([v.co[:] for v in me2.vertices],dtype=np.float64); tris=np.array([list(t.vertices) for t in me2.loop_triangles],dtype=np.int64)
print('voxel-remeshed: %d verts %d polys sides %s %d tris'%(len(co2),len(me2.polygons),sides,len(tris)))
np.savez(os.path.join(UV,'_vmesh_raw.npz'), co=co2, fv=tris)
