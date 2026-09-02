import bpy
F = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00196_\run_20260614_020846\character_posed_00196_.fbx"
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=F)
ms=[o for o in bpy.data.objects if o.type=='MESH']
for o in ms:
    me=o.data
    q=sum(1 for p in me.polygons if len(p.vertices)==4); t=sum(1 for p in me.polygons if len(p.vertices)==3); ng=sum(1 for p in me.polygons if len(p.vertices)>4)
    import numpy as np
    n=len(me.vertices); co=np.empty(n*3); me.vertices.foreach_get('co',co); co=co.reshape(n,3)
    print("MESH %r verts=%d faces=%d quads=%d tris=%d ngons=%d  bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]"%(o.name,n,len(me.polygons),q,t,ng,co[:,0].min(),co[:,0].max(),co[:,1].min(),co[:,1].max(),co[:,2].min(),co[:,2].max()))
    print("  uv_layers:",[l.name for l in me.uv_layers],"  materials:",[m.name if m else None for m in me.materials])
