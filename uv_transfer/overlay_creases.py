"""Diagnostic overlay: put the concave-crease field on the 3D character as separate colored objects
so it reads in any shading mode and the character stays untouched.
  RED  (small)  = RAW concave creases, NO filtering (env OV_RAW)
  BLUE (bigger) = creases AFTER face-clear + front-most filtering (env OV_FILT)
  GREEN         = the final loop (env OV_LOOP, optional)
Each = the mesh faces touching those verts, pushed out along the normal (stacked offsets so blue/green
sit above red). BLUE is dilated 1 ring so it reads 'bigger'. Restricted to the head (z_norm>OV_ZMIN)
to stay light. Env: OV_OUT (blend), OV_RENDER (prefix)."""
import bpy, numpy as np, os
from mathutils import Vector
from collections import defaultdict

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
co = np.array([v.co[:] for v in me.vertices])
zn = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min())
ZMIN = float(os.environ.get('OV_ZMIN', '0.80'))
size = float((co.max(0) - co.min(0)).max())

v2f = defaultdict(list)
for p in me.polygons:
    for v in p.vertices:
        v2f[int(v)].append(p.index)

def verts_of(key):
    pth = os.environ.get(key, '')
    if not pth or not os.path.exists(pth):
        return None
    a = np.load(pth)
    e = a['loop'] if 'loop' in a.files else a['edges']
    return set(int(x) for x in np.unique(e.ravel()).tolist())

def dilate(vset, rings):
    cur = set(vset)
    for _ in range(rings):
        add = set()
        for v in cur:
            for f in v2f[v]:
                for w in me.polygons[f].vertices:
                    add.add(int(w))
        cur |= add
    return cur

def make_overlay(name, vset, color, off_mult, rings, faces=None):
    if faces is None:
        vset = {v for v in vset if zn[v] > ZMIN}
        if rings:
            vset = {v for v in dilate(vset, rings) if zn[v] > ZMIN}
        faces = sorted({f for v in vset for f in v2f[v]})
    if not faces:
        return
    eps = off_mult * 0.0016 * size
    vmap = {}; nv = []; nf = []
    for f in faces:
        fi = []
        for v in me.polygons[f].vertices:
            v = int(v)
            if v not in vmap:
                vmap[v] = len(nv)
                nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * eps)[:]))
            fi.append(vmap[v])
        nf.append(fi)
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    m = bpy.data.meshes.new(name); m.from_pydata(nv, [], nf)
    o = bpy.data.objects.new(name, m); bpy.context.scene.collection.objects.link(o)
    o.matrix_world = obj.matrix_world.copy()
    mat = bpy.data.materials.new(name + 'MAT'); mat.use_nodes = False; mat.diffuse_color = color
    m.materials.append(mat)
    print('%s: %d faces' % (name, len(nf)))

# HAIR-COLOR fill (yellow), drawn UNDERNEATH so the red/blue/green overlap it. The enclosed faces are
# precomputed by _hairfill.py (scipy) and passed in OV_FILL (face indices).
_fp = os.environ.get('OV_FILL')
_fillbnd = None
if _fp and os.path.exists(_fp):
    _ff = [int(x) for x in np.load(_fp)['faces'] if int(x) < len(me.polygons)]
    make_overlay('HAIR_FILL', None, (1.0, 0.85, 0.05, 1.0), 0.4, 0, faces=_ff)   # yellow, under all
    print('HAIR_FILL: %d faces' % len(_ff))
    # GREEN = the EXACT boundary of the yellow region (an edge shared by one fill face + one non-fill
    # face). The seam line is then tight to the colour BY CONSTRUCTION -- they can't disagree.
    _fillset = set(_ff); _e2fb = defaultdict(list)
    for p in me.polygons:
        vs = [int(v) for v in p.vertices]; k = len(vs)
        for a in range(k):
            e = (min(vs[a], vs[(a + 1) % k]), max(vs[a], vs[(a + 1) % k])); _e2fb[e].append(p.index)
    _fillbnd = set()
    for e, fs in _e2fb.items():
        if len(fs) == 2 and ((fs[0] in _fillset) != (fs[1] in _fillset)):
            _fillbnd.add(e[0]); _fillbnd.add(e[1])
    print('HAIR_SEAM = fill boundary: %d verts' % len(_fillbnd))

raw = verts_of('OV_RAW'); filt = verts_of('OV_FILT'); loop = verts_of('OV_LOOP')
if _fillbnd is not None:
    loop = _fillbnd   # green follows the colour's own edge -> 100% tight, no independent loop to mismatch
if raw is not None:
    make_overlay('RAW_CREASE', raw, (0.95, 0.15, 0.10, 1.0), 1.0, 0)     # red, small, lowest
if filt is not None:
    make_overlay('FILT_CREASE', filt, (0.10, 0.35, 1.0, 1.0), 2.2, 1)    # blue, bigger (1-ring), above
if loop is not None:
    make_overlay('HAIR_SEAM', loop, (0.05, 1.0, 0.15, 1.0), 3.4, 0)      # green, top

bpy.ops.wm.save_as_mainfile(filepath=os.environ['OV_OUT'])
print('saved', os.environ['OV_OUT'])

if os.environ.get('OV_RENDER'):
    out = os.environ['OV_RENDER']
    _fr = loop if loop else (filt if filt else None)    # frame on loop, else filtered set, else head
    cen = co[sorted(_fr)] if _fr else co[zn > ZMIN]
    lo = np.percentile(cen, 2, axis=0); hi = np.percentile(cen, 98, axis=0)
    ctr = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2))
    ext = max(hi[0]-lo[0], hi[2]-lo[2]) * 1.5
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.render.film_transparent = True

    def shot(dv, nm):
        sc.render.resolution_x = 900; sc.render.resolution_y = 900
        cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = ext; cd.clip_start = 0.001; cd.clip_end = 1000
        cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = ctr + Vector(dv).normalized() * 10.0
        cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = nm; bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam, do_unlink=True)
    shot((1, 0, 0), out + '_sideR.png'); shot((0, -1, 0), out + '_front.png'); shot((1, -0.6, 0.15), out + '_q34.png')
    shot((1, 0.7, 0.1), out + '_qback.png')
    print('rendered', out)
