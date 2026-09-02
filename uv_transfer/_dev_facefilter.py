"""Filter the photo-face region to FRONT-FACING faces only (drop side/back-wrapping grazing faces that
the 2D oval grabbed at the silhouette), then render. Run: blender last_seams_DEV.blend --background --python _dev_facefilter.py -- <ny_thresh>"""
import bpy, os, math, sys, numpy as np
UV=os.path.dirname(os.path.abspath(__file__))
NY=float(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else -0.25
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data; nv=len(me.vertices); nf=len(me.polygons)
co=np.empty(nv*3); me.vertices.foreach_get('co',co); co=co.reshape(nv,3)
mat=obj.matrix_world; R=np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T=np.array([mat[r][3] for r in range(3)])
co=co@R.T+T
zmn,zmx=float(co[:,2].min()),float(co[:,2].max()); xc=float((co[:,0].min()+co[:,0].max())/2); xsp=float(co[:,0].max()-co[:,0].min())
zs=0.91/(zmx-zmn); zo=-0.41-zmn*zs; xs=0.92/xsp
co[:,2]=co[:,2]*zs+zo; co[:,0]=(co[:,0]-xc)*xs; co[:,1]=co[:,1]*zs
lv=np.empty(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',lv)
ls=np.empty(nf,dtype=np.int32); lt=np.empty(nf,dtype=np.int32); me.polygons.foreach_get('loop_start',ls); me.polygons.foreach_get('loop_total',lt)
fn=np.zeros((nf,3)); fz=np.zeros(nf); fx=np.zeros(nf); fy=np.zeros(nf)
for f in range(nf):
    vp=co[lv[ls[f]:ls[f]+lt[f]]]; fz[f]=vp[:,2].mean(); fx[f]=vp[:,0].mean(); fy[f]=vp[:,1].mean()
    nvec=np.cross(vp[1]-vp[0],vp[2]-vp[0]); nl=np.linalg.norm(nvec)
    if nl>1e-12: fn[f]=nvec/nl
if fn[fz>np.percentile(fz,98),2].mean()<0: fn=-fn
face=np.load(os.path.join(UV,'_dev_faceregion.npy')).astype(bool)
front=fn[:,1]<NY
filt=face & front
# keep only the largest connected component (drop scattered front-facing specks the 2D oval grabbed)
from collections import defaultdict as _dd
_ef=_dd(list)
for _f in range(nf):
    _vs=lv[ls[_f]:ls[_f]+lt[_f]]
    for _j in range(len(_vs)):
        _a,_b=int(_vs[_j]),int(_vs[(_j+1)%len(_vs)]); _ef[(_a,_b) if _a<_b else (_b,_a)].append(_f)
_pairs=np.array([fcv for fcv in _ef.values() if len(fcv)==2],dtype=np.int64)
_par=np.arange(nf)
def _find(a):
    while _par[a]!=a: _par[a]=_par[_par[a]]; a=_par[a]
    return a
_both=filt[_pairs[:,0]]&filt[_pairs[:,1]]
for _a,_b in _pairs[_both].tolist():
    _ra,_rb=_find(_a),_find(_b)
    if _ra!=_rb: _par[_ra]=_rb
_lab=np.array([_find(i) for i in range(nf)])
_u,_c=np.unique(_lab[filt],return_counts=True); _root=_u[_c.argmax()]
filt = filt & (_lab==_root)
print(f"[filter] ny<{NY} + largest-component: {int(face.sum())} -> {int(filt.sum())} (dropped {int((face&~front).sum())} side/back + specks)")
np.save(os.path.join(UV,'_dev_faceregion_filt.npy'), filt)
# INTERIOR vertices only: ALL adjacent faces are face -> no boundary growth, hairline creases survive
zr=fz.max()-fz.min()
chin=float(fz[filt].min()) if filt.any() else fz.max()-0.12*zr      # ACTUAL chin = bottom of detected face
_nb=(np.abs(fx)<0.05)&(fz<chin)&(fz>chin-0.14*zr)
ymid=float(np.median(fy[_nb])) if _nb.any() else 0.0
throat=(np.abs(fx)<0.035)&(fz<chin)&(fz>chin-0.10*zr)&(fy<ymid)      # NECK right below chin: central + front-by-depth
mask=filt|throat
print('[filter] chin z=%.3f'%chin)
fvm=np.zeros(nv,bool)
for f in np.where(mask)[0]:
    for v in lv[ls[f]:ls[f]+lt[f]]: fvm[v]=True
np.save(os.path.join(UV,'_dev_facevert.npy'), fvm)
print('[filter] face %d + throat %d; throat x[%.2f,%.2f] z[%.2f,%.2f]'%(int(filt.sum()),int(throat.sum()),
      float(fx[throat].min()) if throat.any() else 0,float(fx[throat].max()) if throat.any() else 0,
      float(fz[throat].min()) if throat.any() else 0,float(fz[throat].max()) if throat.any() else 0))
print('[filter] saved facevert', int(fvm.sum()))
# render
me.materials.clear()
def m(n,c):
    mm=bpy.data.materials.new(n); mm.use_nodes=False; mm.diffuse_color=c; me.materials.append(mm)
m('skin',(0.80,0.76,0.72,1)); m('face',(0.15,0.85,0.90,1)); m('drop',(0.95,0.25,0.20,1))
midx=np.zeros(nf,dtype=np.int32); midx[face&~front]=2; midx[filt]=1
me.polygons.foreach_set('material_index',midx); me.update()
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=800; sc.render.resolution_y=900
if sc.world is None: sc.world=bpy.data.worlds.new('w')
sc.world.color=(0.12,0.12,0.14)
from mathutils import Vector
bb=[obj.matrix_world@Vector(c) for c in obj.bound_box]; mn=Vector((min(v.x for v in bb),min(v.y for v in bb),min(v.z for v in bb))); mx=Vector((max(v.x for v in bb),max(v.y for v in bb),max(v.z for v in bb)))
ctr=(mn+mx)*0.5; dim=mx-mn
cd=bpy.data.cameras.new('c'); cd.type='ORTHO'; cam=bpy.data.objects.new('c',cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(view,fn2):
    c=Vector((ctr.x,ctr.y,mn.z+0.88*dim.z)); cd.ortho_scale=0.17*dim.z*2; dd=max(dim.x,dim.y,dim.z)*3
    if view=='front': cam.location=c+Vector((0,-dd,0)); cam.rotation_euler=(math.radians(90),0,0)
    else: cam.location=c+Vector((dd,0,0)); cam.rotation_euler=(math.radians(90),0,math.radians(90))
    sc.render.filepath=fn2; bpy.ops.render.render(write_still=True)
shot('front',os.path.join(UV,'_dev_facefilt_front.png')); shot('right',os.path.join(UV,'_dev_facefilt_right.png'))
print('[filter] rendered')
