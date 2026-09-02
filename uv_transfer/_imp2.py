import bpy, numpy as np
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
def do(fbx, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
    bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True); bpy.context.view_layer.objects.active=obj
    try: bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    except Exception: pass
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me=obj.data; n=len(me.vertices); co=np.empty(n*3); me.vertices.foreach_get('co',co); co=co.reshape(n,3)
    print("%s after-apply bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]"%(out,co[:,0].min(),co[:,0].max(),co[:,1].min(),co[:,1].max(),co[:,2].min(),co[:,2].max()))
    for o in list(bpy.data.objects):
        if o.type!='MESH': bpy.data.objects.remove(o,do_unlink=True)
    bpy.ops.wm.save_as_mainfile(filepath=D+"\\"+out)
do(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00196_\run_20260614_020846\character_posed_00196_.fbx","_work_00196.blend")
do(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00184_\run_20260614_021516\character_posed_00184_.fbx","_work_00184.blend")
