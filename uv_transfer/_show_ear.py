"""Render the ear-carve result on the head: hair=blonde, skin=grey, CARVED faces
(_earcap_dbg.npy)=RED, from both side-3/4 angles so I can see if the carve took the
ear ONLY or also ate the temple strands in front of the ear."""
import bpy, os, numpy as np
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=os.path.join(D,"last_seams.blend"))
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data; mw=obj.matrix_world
co=np.array([list(mw @ v.co) for v in me.vertices])
ph=np.load(os.path.join(D,"last_seams_hairfaces.npy")).astype(bool)
try: ec=np.load(os.path.join(D,"_earcap_dbg.npy")).astype(bool)
except Exception: ec=np.zeros(len(ph),bool)
mi=np.where(ph,1,0).astype(np.int32); mi[ec]=2
me.materials.clear()
for nm,cl in [("skin",(0.62,0.60,0.58,1)),("hair",(0.90,0.74,0.26,1)),("ear",(0.97,0.05,0.05,1))]:
    m=bpy.data.materials.new(nm); m.diffuse_color=cl; me.materials.append(m)
me.polygons.foreach_set("material_index",mi); me.update()
try: obj.show_wire=True; obj.show_all_edges=True
except Exception: pass
zmin,zmax=co[:,2].min(),co[:,2].max(); xc=(co[:,0].min()+co[:,0].max())/2; yc=(co[:,1].min()+co[:,1].max())/2; H=zmax-zmin
tgt=Vector((xc,yc,zmin+0.82*H)); S=0.40*H
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=950; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
def shot(off,nm):
    loc=tgt+Vector(off)
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=S
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=loc; fwd=(tgt-loc).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
# her-left ear (+X side): camera off to +X, slightly front; and her-right ear (-X)
shot(( 2.6*H,-1.6*H,0.10*H),"_ear_L.png")   # +X 3/4 front -> her-left ear+temple
shot((-2.6*H,-1.6*H,0.10*H),"_ear_R.png")   # -X 3/4 front -> her-right ear+temple
shot((0,-3.0*H,0.10*H),"_ear_front.png")    # straight front, both temples
print("DONE_EAR earcap",int(ec.sum()))
