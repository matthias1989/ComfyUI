import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data; nf=len(me.polygons); out=os.environ['RS_OUT']
mats=[m.name if m else '' for m in me.materials]
hidx=[i for i,nm in enumerate(mats) if 'hair' in nm.lower()]
midx=np.empty(nf,dtype=int); me.polygons.foreach_get('material_index',midx)
hair=np.isin(midx,hidx)
e2f=defaultdict(list)
for p in me.polygons:
    for ek in p.edge_keys: e2f[ek].append(p.index)
bnd=np.zeros(nf,bool)
for fs in e2f.values():
    if len(fs)==2 and hair[fs[0]]!=hair[fs[1]]: bnd[fs[0]]=bnd[fs[1]]=True
mat=obj.matrix_world; cen=np.zeros((nf,3))
for p in me.polygons:
    c=mat@p.center; cen[p.index]=(c.x,c.y,c.z)
lf=np.empty(len(me.loops),dtype=np.int64)
for p in me.polygons:
    for k in range(p.loop_total): lf[p.loop_start+k]=p.index
G=np.array([0.82,0.64,0.20]);S=np.array([0.66,0.63,0.61]);R=np.array([0.95,0.06,0.06])
fcol=np.where(hair[:,None],G[None],S[None]); fcol=np.where(bnd[:,None],R[None],fcol)
rgba=np.concatenate([fcol[lf],np.ones((len(lf),1))],1).astype(np.float32)
for ca in list(me.color_attributes):
    if ca.name=='RS': me.color_attributes.remove(ca)
col=me.color_attributes.new(name='RS',type='BYTE_COLOR',domain='CORNER'); col.data.foreach_set('color',rgba.ravel()); me.color_attributes.active_color=col; me.update()
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='VERTEX'; sc.render.film_transparent=True
hc=cen[hair] if hair.any() else cen
ctr=Vector((hc[:,0].mean(),hc[:,1].mean(),hc[:,2].mean())); ext=max(np.ptp(hc[:,0]),np.ptp(hc[:,2]))*1.45
def shot(dv,nm):
    sc.render.resolution_x=1000; sc.render.resolution_y=1000
    cd=bpy.data.cameras.new('c');cd.type='ORTHO';cd.ortho_scale=ext;cd.clip_start=0.001;cd.clip_end=1000
    cam=bpy.data.objects.new('c',cd);sc.collection.objects.link(cam);sc.camera=cam
    cam.location=ctr+Vector(dv).normalized()*10.0; cam.rotation_euler=(ctr-cam.location).normalized().to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
shot((1,0,0),out+'_sideR.png'); shot((0.85,-0.85,0.12),out+'_q34.png'); shot((0,-1,0),out+'_front.png')
print('matrender',out,'hair',int(hair.sum()))
