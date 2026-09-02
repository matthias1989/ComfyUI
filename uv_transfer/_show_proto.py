"""Render a prototype mask (_proto_result.npy) over the viz mesh: hair=blonde,
skin=grey. front/back/side. Rebuilt from _genseams_viz.npz so it's the current char."""
import bpy, os, numpy as np, sys
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
MASK=sys.argv[-1] if sys.argv[-1].endswith('.npy') else "_proto_result.npy"
TAG=os.path.splitext(os.path.basename(MASK))[0]
vz=np.load(os.path.join(D,"_genseams_viz.npz"))
co=vz['co'].astype(np.float64); tris=vz['tris']
th=np.load(os.path.join(D,MASK)).astype(bool)
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.scene.collection.objects.link(ob)
mg=bpy.data.materials.new("g"); mg.diffuse_color=(0.60,0.58,0.56,1)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.92,0.76,0.28,1)
me.materials.append(mg); me.materials.append(mb)
me.polygons.foreach_set("material_index",th.astype(np.int32)); me.update()
zmin,zmax=co[:,2].min(),co[:,2].max(); H=zmax-zmin
tgt=Vector((0.0,0.0,zmin+0.86*H)); S=0.55*H
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=820; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
def shot(off,nm):
    loc=tgt+Vector(off)
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=S
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=loc; fwd=(tgt-loc).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
shot((0,-3.0*H,0.05*H),f"_p_{TAG}_front.png")
shot((0, 3.0*H,0.05*H),f"_p_{TAG}_back.png")
shot(( 3.0*H,0,0.05*H),f"_p_{TAG}_side.png")
print(f"DONE_PROTO_RENDER {TAG} hair {int(th.sum())}")
