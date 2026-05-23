import bpy, bmesh, json, sys

obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
if obj is None:
    print('JSON_OUT:' + json.dumps({'error': 'no mesh'}))
    sys.exit(0)

me  = obj.data
bm  = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()

mat = obj.matrix_world
wv  = [mat @ v.co for v in bm.verts]
xs  = [v.x for v in wv]
zs  = [v.z for v in wv]
z_min, z_max = min(zs), max(zs)
x_min, x_max = min(xs), max(xs)
z_range = max(z_max - z_min, 1e-6)
x_range = max(x_max - x_min, 1e-6)

bands  = ['head','neck_shoulder','torso','hip_leg','mid_leg','ankle']
br     = [(0.85,1.01),(0.70,0.85),(0.45,0.70),(0.30,0.45),(0.15,0.30),(0.0,0.15)]
bs     = {b: {'t':0,'L':0,'R':0,'C':0} for b in bands}

sdeg = {}
for e in bm.edges:
    if not e.seam:
        continue
    v0w = wv[e.verts[0].index]
    v1w = wv[e.verts[1].index]
    ec  = (v0w + v1w) * 0.5
    zn  = (ec.z - z_min) / z_range
    for i, (lo, hi) in enumerate(br):
        if lo <= zn < hi:
            b = bands[i]
            bs[b]['t'] += 1
            # X position relative to body center
            xc = (x_min + x_max) / 2.0
            if ec.x < xc - 0.03:
                bs[b]['L'] += 1
            elif ec.x > xc + 0.03:
                bs[b]['R'] += 1
            else:
                bs[b]['C'] += 1
            break
    for v in e.verts:
        sdeg[v.index] = sdeg.get(v.index, 0) + 1

open_ends = sum(1 for cnt in sdeg.values() if cnt == 1)

oe_band = {}
for vi, cnt in sdeg.items():
    if cnt == 1:
        vw = wv[vi]
        zn = (vw.z - z_min) / z_range
        for i, (lo, hi) in enumerate(br):
            if lo <= zn < hi:
                b = bands[i]
                oe_band[b] = oe_band.get(b, 0) + 1
                break

bm.free()

total_seams = sum(1 for e in me.edges if e.use_seam)
out = {
    'name':   obj.name,
    'verts':  len(me.vertices),
    'faces':  len(me.polygons),
    'seams':  total_seams,
    'open':   open_ends,
    'bs':     bs,
    'oe':     oe_band,
    'z_range': [round(z_min, 3), round(z_max, 3)],
    'x_center': round((x_min + x_max) / 2.0, 4),
}
print('JSON_OUT:' + json.dumps(out))
