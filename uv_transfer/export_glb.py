"""
Export geometry_0 from last_seams.blend to a GLB file,
preserving the custom UV layout from gen_seams.py.

Run:
  blender.exe last_seams.blend --background --python export_glb.py
"""
import bpy
import os

blend_dir = os.path.dirname(bpy.data.filepath)
out_path  = os.path.join(blend_dir, "last_seams.glb")

# Select only geometry_0
for obj in bpy.context.scene.objects:
    obj.select_set(False)

target = bpy.data.objects.get("geometry_0")
if target is None:
    target = max((o for o in bpy.context.scene.objects if o.type == 'MESH'),
                 key=lambda o: len(o.data.polygons))

target.select_set(True)
bpy.context.view_layer.objects.active = target

print(f"\n=== Exporting to GLB ===")
print(f"  Object: {target.name}, UV layers: {[l.name for l in target.data.uv_layers]}")
print(f"  Verts: {len(target.data.vertices)}, Faces: {len(target.data.polygons)}")

# Assign a plain material so GLTF exporter includes UV data
mat = bpy.data.materials.get("__uv_export_mat__")
if mat is None:
    mat = bpy.data.materials.new("__uv_export_mat__")
    mat.use_nodes = True

if len(target.data.materials) == 0:
    target.data.materials.append(mat)
else:
    target.data.materials[0] = mat

bpy.ops.export_scene.gltf(
    filepath=out_path,
    export_format='GLB',
    use_selection=True,
    export_apply=True,
    export_normals=True,
    export_tangents=False,
    export_texcoords=True,
    export_materials='EXPORT',   # Must be EXPORT so UV coords are included
    export_animations=False,
    export_skins=False,
    export_morph=False,
)

print(f"Exported: {out_path}")

# Also export OBJ so FBXExport can import it (it expects OBJ, not GLB)
out_obj = os.path.join(blend_dir, "last_seams.obj")
bpy.ops.wm.obj_export(
    filepath=out_obj,
    export_selected_objects=True,
    export_uv=True,
    export_normals=True,
    export_materials=False,
    export_triangulated_mesh=False,
)
print(f"Exported OBJ: {out_obj}")
