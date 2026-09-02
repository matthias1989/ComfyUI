"""Tight FACE/forehead render of the current viz hair mask: front + top-down, to judge
the forehead-middle bleed and the hairline.  hair=blonde, skin=grey."""
import bpy, os, sys, numpy as np
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
_a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
_tags=[x for x in _a if not x.endswith('.npy')]; _masks=[x for x in _a if x.endswith('.npy')]
TAG=("_"+_tags[0]) if _tags else ""
vz=np.load(os.path.join(D,"_genseams_viz.npz"))
co=vz['co'].astype(np.float64); tris=vz['tris']
th=(np.load(os.path.join(D,_masks[0]))[vz['tris_face']] if _masks else vz['tris_hair']).astype(bool)
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.scene.collection.objects.link(ob)
mg=bpy.data.materials.new("g"); mg.diffuse_color=(0.60,0.58,0.56,1)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.92,0.76,0.28,1)
me.materials.append(mg); me.materials.append(mb)
me.polygons.foreach_set("material_index",th.astype(np.int32)); me.update()
zmin,zmax=co[:,2].min(),co[:,2].max(); H=zmax-zmin
# face center ~ top of head; frame the forehead/upper face
fz=zmin+0.92*H; S=0.24*H
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=900; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
def shot(loc,tgt,nm):
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=S
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=Vector(loc); fwd=(Vector(tgt)-Vector(loc)).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
shot((0,-3*H,fz),(0,0,fz),f"_face_front{TAG}.png")
shot((0,-1.2*H,fz+1.6*H),(0,0,fz-0.04*H),f"_face_top{TAG}.png")   # look down at forehead/crown
# under-chin view: camera low and in front, looking UP at the jaw/chin underside
shot((0,-1.7*H,zmin+0.72*H),(0,-0.02*H,zmin+0.85*H),f"_face_chin{TAG}.png")
print(f"DONE_FACE{TAG}")
