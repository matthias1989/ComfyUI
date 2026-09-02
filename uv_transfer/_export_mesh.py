import bpy, numpy as np, os
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data; mat=obj.matrix_world
nv=len(me.vertices); nf=len(me.polygons)
co=np.empty(nv*3); me.vertices.foreach_get('co',co); co=co.reshape(nv,3)
# world transform
R=np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T=np.array([mat[r][3] for r in range(3)])
co=co@R.T+T
# faces as list of vertex-index tuples (quads/tris)
fv=[tuple(p.vertices) for p in me.polygons]
hair=np.load(os.environ['RS_NPY']).astype(bool)
np.savez(os.environ['RS_OUT'], co=co, hair=hair, fv=np.array(fv,dtype=object), nf=nf, nv=nv)
print('exported nv=%d nf=%d hairfaces=%d'%(nv,nf,int(hair.sum())))
