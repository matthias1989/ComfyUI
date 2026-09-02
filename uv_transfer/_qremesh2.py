import bpy, numpy as np, os
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
d=np.load(os.path.join(UV,'_gs_hires.npz')); co=d['co'].astype(float); fv=d['fv'].astype(np.int64)
bpy.ops.wm.read_factory_settings(use_empty=True)
me=bpy.data.meshes.new('m'); me.from_pydata(co.tolist(),[],fv.tolist()); me.update()
o=bpy.data.objects.new('m',me); bpy.context.scene.collection.objects.link(o)
bpy.context.view_layer.objects.active=o; o.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0003)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
print('after clean: %d verts %d polys'%(len(o.data.vertices),len(o.data.polygons)))
try:
    res=bpy.ops.object.quadriflow_remesh(target_faces=120000, use_mesh_symmetry=False, use_preserve_sharp=False, use_preserve_boundary=False, smooth_normals=False, seed=0)
    print('quadriflow result:', res)
except Exception as e:
    print('quadriflow EXCEPTION:', repr(e))
me2=o.data; sides={}
for p in me2.polygons: sides[len(p.vertices)]=sides.get(len(p.vertices),0)+1
print('after remesh: %d verts %d polys  sides %s'%(len(me2.vertices),len(me2.polygons),sides))
me2.calc_loop_triangles()
co2=np.array([v.co[:] for v in me2.vertices],dtype=np.float64); tris=np.array([list(t.vertices) for t in me2.loop_triangles],dtype=np.int64)
np.savez(os.path.join(UV,'_qmesh_raw.npz'), co=co2, fv=tris)
