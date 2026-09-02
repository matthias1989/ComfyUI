"""Clean hair(blonde)/skin(grey) render of the CURRENT _genseams_viz.npz mask,
front/back/side, head-framed, so I see the exact detected hair pattern and the holes
(no FBX shading). Rebuilt from the viz npz so it's always the latest gen_seams run."""
import bpy, os, sys, numpy as np
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
_a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
_tags=[x for x in _a if not x.endswith('.npy')]; _masks=[x for x in _a if x.endswith('.npy')]
TAG=("_"+_tags[0]) if _tags else ""   # pass A/B after `--`; optional .npy face-mask to override
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
# frame the HEAD: top of the mesh
zmin,zmax=co[:,2].min(),co[:,2].max(); xc=0.0; H=zmax-zmin
tgt=Vector((xc,0.0,zmin+0.86*H)); S=0.55*H
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=1300; sc.render.resolution_y=1400; sc.render.image_settings.file_format='PNG'
def shot(off,nm):
    loc=tgt+Vector(off)
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=S
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=loc; fwd=(tgt-loc).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
shot((0,-3.0*H,0.05*H),f"_head_front{TAG}.png")
shot((0, 3.0*H,0.05*H),f"_head_back{TAG}.png")
shot(( 2.3*H,-2.0*H,0.12*H),f"_head_qL{TAG}.png")   # 3/4 front, +X (her-right) side
shot((-2.3*H,-2.0*H,0.12*H),f"_head_qR{TAG}.png")   # 3/4 front, -X (her-left) side
shot(( 3.0*H,0,0.05*H),f"_head_side{TAG}.png")
print(f"DONE_HEAD{TAG} hair",int(th.sum()),"/",len(th))
