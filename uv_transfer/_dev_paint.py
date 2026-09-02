"""Paint debug layers onto last_seams_DEV.blend, visible in OBJECT mode (solid=Material):
 orange=all concave points, yellow=filtered/kept points, green=final seam. Saves the blend."""
import bpy, os, numpy as np
from collections import defaultdict
UV=os.path.dirname(os.path.abspath(__file__))
o=bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type=='MESH'); me=o.data
nv=len(me.vertices); nf=len(me.polygons)
cv_all=np.unique(np.load(UV+'/_gb_crease_all.npz')['edges'].reshape(-1))
con=np.zeros(nv,bool); con[cv_all[cv_all<nv]]=True
fil=np.load(UV+'/_gb_kept.npy').astype(bool)
lv=np.empty(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',lv)
ls=np.empty(nf,dtype=np.int32); lt=np.empty(nf,dtype=np.int32); me.polygons.foreach_get('loop_start',ls); me.polygons.foreach_get('loop_total',lt)
# seam edge -> faces
ef=defaultdict(list)
for f in range(nf):
    vs=lv[ls[f]:ls[f]+lt[f]]
    for j in range(len(vs)):
        a,b=int(vs[j]),int(vs[(j+1)%len(vs)]); ef[(a,b) if a<b else (b,a)].append(f)
seam_keys=set((min(e.vertices[0],e.vertices[1]),max(e.vertices[0],e.vertices[1])) for e in me.edges if e.use_seam)
conf=np.zeros(nf,bool); filf=np.zeros(nf,bool); seaf=np.zeros(nf,bool)
for f in range(nf):
    vs=lv[ls[f]:ls[f]+lt[f]]
    if any(con[v] for v in vs): conf[f]=True
    if any(fil[v] for v in vs): filf[f]=True
for k in seam_keys:
    for f in ef.get(k,[]): seaf[f]=True
me.materials.clear()
def m(n,c):
    x=bpy.data.materials.new(n); x.use_nodes=False; x.diffuse_color=c; me.materials.append(x)
m('skin',(0.80,0.78,0.74,1)); m('concave',(0.95,0.55,0.10,1)); m('filtered',(0.95,0.90,0.15,1))
midx=np.zeros(nf,dtype=np.int32); midx[conf]=1; midx[filf]=2   # orange concave, yellow kept; seam removed
me.polygons.foreach_set('material_index',midx); me.update()
print('[paint] concave faces %d, filtered %d, seam %d'%(int(conf.sum()),int(filf.sum()),int(seaf.sum())))
# make it show in OBJECT/solid mode = Material color
for sc in bpy.data.screens:
    for ar in sc.areas:
        if ar.type=='VIEW_3D':
            for sp in ar.spaces:
                if sp.type=='VIEW_3D':
                    sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'; sp.shading.light='FLAT'
bpy.ops.wm.save_as_mainfile(filepath=UV+'/last_seams_DEV.blend')
print('[paint] saved last_seams_DEV.blend')
