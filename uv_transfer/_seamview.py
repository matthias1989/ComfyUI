import bpy, numpy as np
from collections import defaultdict
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=D+r"\last_seams_B.blend")
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me = obj.data
nf=len(me.polygons); nv=len(me.vertices)
co=np.empty(nv*3); me.vertices.foreach_get('co',co); co=co.reshape(nv,3)
cen=np.zeros((nf,3))
for p in me.polygons: cen[p.index]=p.center[:]
seam=set()
for e in me.edges:
    if e.use_seam: seam.add((min(e.vertices),max(e.vertices)))
e2f=defaultdict(list)
for p in me.polygons:
    for ek in p.edge_keys: e2f[ek].append(p.index)
par=list(range(nf))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for ek,fs in e2f.items():
    if len(fs)==2 and ek not in seam:
        ra,rb=find(fs[0]),find(fs[1])
        if ra!=rb: par[ra]=rb
roots=np.array([find(i) for i in range(nf)])
hair=(roots==find(int(np.argmax(cen[:,2]))))
lf=np.empty(len(me.loops),dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total): lf[p.loop_start+k]=p.index
GOLD=np.array([0.93,0.74,0.27]); SK=np.array([0.62,0.6,0.58])
fc=np.where(hair[lf][:,None],GOLD[None],SK[None])
rgba=np.concatenate([fc,np.ones((len(lf),1))],axis=1).astype(np.float32)
col=me.color_attributes.new(name="SV",type='BYTE_COLOR',domain='CORNER')
col.data.foreach_set("color",rgba.ravel()); me.update()
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='VERTEX'
sc.render.resolution_x=1100; sc.render.resolution_y=1300
def shot(loc,nm,scale=0.22,tgt=Vector((0.10,0.02,0.36))):
    cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=scale
    cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
    cam.location=tgt+Vector(loc); f=(tgt-cam.location).normalized(); cam.rotation_euler=f.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=D+"\\"+nm; bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam,do_unlink=True)
shot((1.6,-1.6,0.3),"_sv_ear34.png")   # zoom: right ear, front-right 3/4
shot((3.0,0.2,0.1),"_sv_earside.png")  # zoom: right ear, side
print("done hair",int(hair.sum()))
