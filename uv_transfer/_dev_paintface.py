"""Paint ONLY the facemask (the P2_FACE region path3 uses) in BLUE on last_seams_DEV.blend, object-mode visible."""
import bpy, os, numpy as np
UV=os.path.dirname(os.path.abspath(__file__))
o=bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type=='MESH'); me=o.data
nv=len(me.vertices); nf=len(me.polygons)
fm=np.load(UV+'/_dev_facevert.npy').astype(bool)
lv=np.empty(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',lv)
ls=np.empty(nf,dtype=np.int32); lt=np.empty(nf,dtype=np.int32); me.polygons.foreach_get('loop_start',ls); me.polygons.foreach_get('loop_total',lt)
fmf=np.zeros(nf,bool)
for f in range(nf):
    for v in lv[ls[f]:ls[f]+lt[f]]:
        if fm[v]: fmf[f]=True; break
me.materials.clear()
def m(n,c):
    x=bpy.data.materials.new(n); x.use_nodes=False; x.diffuse_color=c; me.materials.append(x)
m('skin',(0.80,0.78,0.74,1)); m('facemask',(0.10,0.30,0.95,1))
midx=np.zeros(nf,dtype=np.int32); midx[fmf]=1
me.polygons.foreach_set('material_index',midx); me.update()
print('[face] facemask faces:', int(fmf.sum()),'verts:', int(fm.sum()))
for sc in bpy.data.screens:
    for ar in sc.areas:
        if ar.type=='VIEW_3D':
            for sp in ar.spaces:
                if sp.type=='VIEW_3D':
                    sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'; sp.shading.light='FLAT'
bpy.ops.wm.save_as_mainfile(filepath=UV+'/last_seams_DEV.blend')
print('[face] saved')
