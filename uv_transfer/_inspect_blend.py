import bpy
for o in bpy.data.objects:
    if o.type!='MESH': continue
    me=o.data
    print('OBJ:', o.name, '| verts',len(me.vertices),'faces',len(me.polygons))
    print('  vertex_colors:', [l.name for l in me.vertex_colors] if hasattr(me,'vertex_colors') else 'n/a')
    print('  color_attrs  :', [(a.name,a.domain,a.data_type) for a in me.color_attributes] if hasattr(me,'color_attributes') else 'n/a')
    print('  attributes   :', [(a.name,a.domain,a.data_type) for a in me.attributes][:20])
    print('  seam edges   :', sum(1 for e in me.edges if e.use_seam))
    print('  materials    :', [m.name if m else None for m in me.materials])
    co=[v.co for v in me.vertices]
    import numpy as np; co=np.array([c[:] for c in co])
    print('  bbox min',np.round(co.min(0),3),'max',np.round(co.max(0),3))
