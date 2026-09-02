import bpy
o = bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type == 'MESH')
me = o.data
sides = {}
for p in me.polygons:
    sides[len(p.vertices)] = sides.get(len(p.vertices), 0) + 1
print('LASTSEAMS verts=%d polys=%d sides=%s' % (len(me.vertices), len(me.polygons), sides))
