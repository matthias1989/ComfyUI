import bpy, numpy as np
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me = obj.data; nf=len(me.polygons)
mats=[m.name if m else 'None' for m in me.materials]
print('  mesh:',obj.name,'faces:',nf,'materials:',mats)
if mats:
    midx=np.empty(nf,dtype=int); me.polygons.foreach_get('material_index',midx)
    for i,nm in enumerate(mats):
        print('    [%d] %-20s %d faces'%(i,nm,int((midx==i).sum())))
