"""Paint ALL debug layers together on last_seams_DEV.blend (object-mode, solid=Material):
 blue=facemask, orange=all concave points, magenta=kept points. Priority: kept>concave>facemask>skin."""
import bpy, os, numpy as np
UV=os.path.dirname(os.path.abspath(__file__))
o=bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type=='MESH'); me=o.data
nv=len(me.vertices); nf=len(me.polygons)
fm=np.load(UV+'/_dev_facevert.npy').astype(bool)
cv_all=np.unique(np.load(UV+'/_gb_crease_all.npz')['edges'].reshape(-1))
con=np.zeros(nv,bool); con[cv_all[cv_all<nv]]=True
kept=np.load(UV+'/_gb_kept.npy').astype(bool)
lv=np.empty(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',lv)
ls=np.empty(nf,dtype=np.int32); lt=np.empty(nf,dtype=np.int32); me.polygons.foreach_get('loop_start',ls); me.polygons.foreach_get('loop_total',lt)
def faces_of(mask):
    out=np.zeros(nf,bool)
    for f in range(nf):
        for v in lv[ls[f]:ls[f]+lt[f]]:
            if mask[v]: out[f]=True; break
    return out
fmf=faces_of(fm); conf=faces_of(con); kepf=faces_of(kept)
me.materials.clear()
def m(n,c):
    x=bpy.data.materials.new(n); x.use_nodes=False; x.diffuse_color=c; me.materials.append(x)
m('skin',(0.78,0.76,0.72,1)); m('facemask',(0.10,0.30,0.95,1)); m('concave',(0.95,0.45,0.05,1)); m('kept',(0.95,0.05,0.85,1))
midx=np.zeros(nf,dtype=np.int32); midx[fmf]=1; midx[conf]=2; midx[kepf]=3   # kept>concave>facemask>skin
me.polygons.foreach_set('material_index',midx); me.update()
print('[all] facemask %d, concave %d, kept %d faces'%(int(fmf.sum()),int(conf.sum()),int(kepf.sum())))
for sc in bpy.data.screens:
    for ar in sc.areas:
        if ar.type=='VIEW_3D':
            for sp in ar.spaces:
                if sp.type=='VIEW_3D': sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'; sp.shading.light='FLAT'
bpy.ops.wm.save_as_mainfile(filepath=UV+'/last_seams_DEV.blend'); print('[all] saved')
