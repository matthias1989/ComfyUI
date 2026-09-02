"""Bake a loop (_cloop-style npz: edges = vertex-index pairs) onto the character mesh as ACTUAL
UV seams (edge.use_seam) AND a bright green overlay tube (skin-modifier) so it's visible in 3D
when rotating. Saves to a NEW blend so the original is untouched. Env: MS_MESH (npz co), MS_LOOP
(npz with 'loop'), MS_OUT (blend path). Optional MS_RENDER (prefix) to also dump confirm shots."""
import bpy, numpy as np, os
from mathutils import Vector

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
loop = np.load(os.environ['MS_LOOP'])['loop']
co = np.load(os.environ['MS_MESH'], allow_pickle=True)['co']

# 1) ACTUAL UV seams on the character mesh (this is the real deliverable)
ekey = {}
for e in me.edges:
    a, b = int(e.vertices[0]), int(e.vertices[1])
    ekey[(a, b) if a < b else (b, a)] = e.index
    e.use_seam = False
nmark = 0
for u, w in loop:
    u, w = int(u), int(w)
    ei = ekey.get((u, w) if u < w else (w, u))
    if ei is not None:
        me.edges[ei].use_seam = True; nmark += 1
print('seam edges marked %d of %d loop edges (%.0f%% are real mesh edges)' % (nmark, len(loop), 100.0 * nmark / max(1, len(loop))))
# BRIDGE diagonals: the loop is traced on the TRIANGULATED mesh, so some loop edges are quad DIAGONALS
# (no direct quad edge) -> they can't be marked and leave GAPS. For each, mark the shortest 2-hop quad
# path (the two quad edges of the triangle's quad) so the seam is continuous.
from collections import defaultdict as _dd
_vadj = _dd(set)
for _e in me.edges:
    _a, _b = int(_e.vertices[0]), int(_e.vertices[1]); _vadj[_a].add(_b); _vadj[_b].add(_a)
_nbridge = 0
for u, w in loop:
    u, w = int(u), int(w)
    if ekey.get((u, w) if u < w else (w, u)) is not None:
        continue
    _common = _vadj[int(u)] & _vadj[int(w)]
    if _common:
        _mid = min(_common, key=lambda m: float((me.vertices[int(u)].co - me.vertices[m].co).length + (me.vertices[m].co - me.vertices[int(w)].co).length))
        for _a, _b in ((u, _mid), (_mid, w)):
            _ei = ekey.get((_a, _b) if _a < _b else (_b, _a))
            if _ei is not None:
                me.edges[_ei].use_seam = True
        _nbridge += 1
print('bridged %d diagonal loop edges via 2-hop quad paths -> continuous seam' % _nbridge)

# 2) SEPARATE green overlay object (character mesh untouched -> looks right in any shading mode).
# Copy the loop-adjacent faces into a new object, pushed slightly out along the normal so it sits
# proud of the surface without z-fighting.
lv = set(int(v) for v in np.unique(loop.ravel()).tolist())
lco = np.array([v.co[:] for v in me.vertices])
eps = 0.0018 * float((lco.max(0) - lco.min(0)).max())
vmap = {}; nverts = []; nfaces = []
for p in me.polygons:
    if not any(int(v) in lv for v in p.vertices):
        continue
    fi = []
    for v in p.vertices:
        v = int(v)
        if v not in vmap:
            vmap[v] = len(nverts)
            nverts.append(tuple((me.vertices[v].co + me.vertices[v].normal * eps)[:]))
        fi.append(vmap[v])
    nfaces.append(fi)
o = bpy.data.objects.get('HAIR_SEAM')
if o:
    bpy.data.objects.remove(o, do_unlink=True)
om = bpy.data.meshes.new('HAIR_SEAM'); om.from_pydata(nverts, [], nfaces)
oo = bpy.data.objects.new('HAIR_SEAM', om); bpy.context.scene.collection.objects.link(oo)
oo.matrix_world = obj.matrix_world.copy()
mat = bpy.data.materials.new('SEAMMAT'); mat.use_nodes = False; mat.diffuse_color = (0.0, 1.0, 0.15, 1.0)
om.materials.append(mat)
print('overlay faces %d' % len(nfaces))

bpy.ops.wm.save_as_mainfile(filepath=os.environ['MS_OUT'])
print('saved', os.environ['MS_OUT'])

if os.environ.get('MS_RENDER'):
    out = os.environ['MS_RENDER']
    cen = co[np.array(sorted(lv))]
    zc = np.percentile(cen[:, 2], 45); cu = cen[cen[:, 2] > zc]
    lo = np.percentile(cu, 2, axis=0); hi = np.percentile(cu, 98, axis=0)
    ctr = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2))
    ext = max(hi[0] - lo[0], hi[2] - lo[2]) * 1.5
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
    sc.render.film_transparent = True

    def shot(dv, nm):
        sc.render.resolution_x = 900; sc.render.resolution_y = 900
        cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext
        cd.clip_start = 0.001; cd.clip_end = 1000
        cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = ctr + Vector(dv).normalized() * 10.0
        cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
    shot((0, -1, 0), out + '_front.png'); shot((1, -0.6, 0.15), out + '_q34.png'); shot((1, 0, 0), out + '_sideR.png')
    shot((0, 0, 1), out + '_top.png'); shot((0, 1, 0), out + '_back.png'); shot((1, 0.6, 0.15), out + '_qback.png')
    print('rendered', out)
