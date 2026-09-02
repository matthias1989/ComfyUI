"""Run INSIDE Blender (--background --python). Imports last_seams.obj as 'geometry_0',
runs gen_seams.py (marks seams + unwraps), exports the UV'd result to _gs_test_out.obj.
Lets me verify gen_seams UV changes locally (the side files it writes are backed up/restored
by the PowerShell wrapper)."""
import bpy, os
UV = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
OBJ_IN  = os.environ.get("GS_OBJ_IN", os.path.join(UV, "last_seams.obj"))
OBJ_OUT = os.path.join(UV, "_gs_test_out.obj")
GS      = os.path.join(UV, "gen_seams.py")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=OBJ_IN)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
print(f"RUNNER: imported {len(objs)} mesh objects")
obj = objs[0]; obj.name = 'geometry_0'
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
try:
    exec(compile(open(GS, encoding='utf-8').read(), GS, 'exec'))
except SystemExit as e:
    print(f"RUNNER: gen_seams sys.exit({e})")
except Exception as e:
    import traceback; traceback.print_exc(); print(f"RUNNER: gen_seams FAILED {e!r}")
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
o = bpy.data.objects.get('geometry_0')
o.select_set(True); bpy.context.view_layer.objects.active = o
bpy.ops.wm.obj_export(filepath=OBJ_OUT, export_uv=True, export_selected_objects=True, export_materials=False)
print(f"RUNNER: exported -> {OBJ_OUT}")
