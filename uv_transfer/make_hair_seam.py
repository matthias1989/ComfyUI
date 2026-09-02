"""Turn the hair-fill region (the GREEN border) into REAL seams on the character mesh:
  1. mark edge.use_seam on every edge of the region BOUNDARY (a fill face meets a non-fill face) -- the
     green border becomes an actual UV seam the unwrapper will cut along,
  2. split the mesh into two material slots: 0 = Skin, 1 = Hair (the fill faces),
  3. export the per-face hair flag (.npy) so the texture/FBX side can drive the Hair material,
  4. (optional) a green overlay + confirm shots.
Saves to a NEW blend (original untouched). Env: HS_FILL (_Xfill npz, 'faces'), HS_OUT (blend), HS_NPY
(per-face hair flag out), optional HS_RENDER (prefix)."""
import bpy, numpy as np, os
from collections import defaultdict
from mathutils import Vector

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
nf = len(me.polygons)
fill = np.zeros(nf, bool)
_ff = np.load(os.environ['HS_FILL'])['faces']
fill[_ff[_ff < nf]] = True
print('hair faces: %d / %d' % (int(fill.sum()), nf))

# 1) edge -> adjacent faces, then mark the region-boundary edges as REAL seams
e2f = defaultdict(list)
for p in me.polygons:
    vs = [int(v) for v in p.vertices]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(p.index)
ekey = {}
for e in me.edges:
    a, b = int(e.vertices[0]), int(e.vertices[1])
    ekey[(a, b) if a < b else (b, a)] = e.index
    e.use_seam = False
nmark = 0
for ek, fs in e2f.items():
    if len(fs) == 2 and (bool(fill[fs[0]]) != bool(fill[fs[1]])):
        ei = ekey.get(ek)
        if ei is not None:
            me.edges[ei].use_seam = True; nmark += 1
print('hair-boundary seam edges marked: %d' % nmark)

# 2) material slots: 0 = Skin, 1 = Hair
me.materials.clear()
skin = bpy.data.materials.get('Skin') or bpy.data.materials.new('Skin'); skin.use_nodes = False; skin.diffuse_color = (0.8, 0.62, 0.52, 1.0)
hair = bpy.data.materials.get('Hair') or bpy.data.materials.new('Hair'); hair.use_nodes = False; hair.diffuse_color = (0.85, 0.7, 0.25, 1.0)
me.materials.append(skin); me.materials.append(hair)
idx = fill.astype(np.int32)
me.polygons.foreach_set('material_index', idx)
me.update()
print('material slots: 0=Skin %d faces, 1=Hair %d faces' % (int((~fill).sum()), int(fill.sum())))

# 3) export the per-face hair flag for the texture/FBX side
if os.environ.get('HS_NPY'):
    np.save(os.environ['HS_NPY'], fill)
    print('saved hair-face flag ->', os.environ['HS_NPY'])

bpy.ops.wm.save_as_mainfile(filepath=os.environ['HS_OUT'])
print('saved', os.environ['HS_OUT'])

# 4) confirm: green overlay on the seam + shots
if os.environ.get('HS_RENDER'):
    lco = np.array([v.co[:] for v in me.vertices]); size = float((lco.max(0) - lco.min(0)).max())
    bverts = set()
    for ek, fs in e2f.items():
        if len(fs) == 2 and (bool(fill[fs[0]]) != bool(fill[fs[1]])):
            bverts.add(ek[0]); bverts.add(ek[1])
    eps = 0.004 * size; vmap = {}; nv = []; nfc = []
    for p in me.polygons:
        if not any(int(v) in bverts for v in p.vertices):
            continue
        fi = []
        for v in p.vertices:
            v = int(v)
            if v not in vmap:
                vmap[v] = len(nv); nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * eps)[:]))
            fi.append(vmap[v])
        nfc.append(fi)
    gm = bpy.data.meshes.new('SEAMVIS'); gm.from_pydata(nv, [], nfc)
    go = bpy.data.objects.new('SEAMVIS', gm); bpy.context.scene.collection.objects.link(go); go.matrix_world = obj.matrix_world.copy()
    gmat = bpy.data.materials.new('GMAT'); gmat.use_nodes = False; gmat.diffuse_color = (0.0, 1.0, 0.15, 1.0); gm.materials.append(gmat)
    cen = lco[np.array(sorted(bverts))]; zc = np.percentile(cen[:, 2], 40); cu = cen[cen[:, 2] > zc]
    lo = np.percentile(cu, 2, 0); hi = np.percentile(cu, 98, 0)
    ctr = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2)); ext = max(hi[0]-lo[0], hi[2]-lo[2]) * 1.5
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'MATERIAL'; sc.render.film_transparent = True
    def shot(dv, nm):
        sc.render.resolution_x = 900; sc.render.resolution_y = 900
        cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
        cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = ctr + Vector(dv).normalized() * 10.0
        cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
    shot((0, -1, 0), out := os.environ['HS_RENDER'] + '_front.png'); shot((1, 0, 0), os.environ['HS_RENDER'] + '_sideR.png')
    shot((1, 0.6, 0.1), os.environ['HS_RENDER'] + '_qback.png')
    print('rendered', os.environ['HS_RENDER'])
