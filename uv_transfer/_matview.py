import bpy, numpy as np
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=D+r"\_work_00196.blend")
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data; nf=len(me.polygons)
hi=[i for i,m in enumerate(me.materials) if m and 'hair' in m.name.lower()]
hi=hi[0] if hi else -1
mi=np.empty(nf,dtype=np.int64); me.polygons.foreach_get('material_index',mi)
hair=(mi==hi)
print("00196 Hair-material faces:",int(hair.sum()),"/",nf)
lf=np.empty(len(me.loops),dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total): lf[p.loop_start+k]=p.index
GOLD=np.array([0.93,0.74,0.27]); SK=np.array([0.62,0.6,0.58])
fc=np.where(hair[lf][:,None],GOLD[None],SK[None])
rgba=np.concatenate([fc,np.ones((len(lf),1))],axis=1).astype(np.float32)
col=me.color_attributes.new(name="M",type='BYTE_COLOR',domain='CORNER'); col.data.foreach_set("color",rgba.ravel()); me.update()
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='VERTEX'
sc.render.resolution_x=850; sc.render.resolution_y=1000
def shot(loc,nm,tgt=Vector((0,0,0.40))):
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=0.42
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=tgt+Vector(loc); f=(tgt-cam.location).normalized(); cam.rotation_euler=f.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=D+"\\"+nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
shot((0,-3,0),"_m196_f.png"); shot((2.2,-2,0.3),"_m196_34.png")
print("done")
