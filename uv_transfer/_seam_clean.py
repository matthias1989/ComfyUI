"""Clean up a baked seam: (1) DE-DOUBLE -- remove an edge if its endpoints still connect via a SHORT
alternative path (<= SC_MAXALT), i.e. it's a redundant parallel/chord edge; (2) PRUNE short spurs
(dead-end chains shorter than SC_MAXSPUR). Long straight runs (no short alt, not dead-ends) are kept.
Rebuilds the green HAIR_SEAM overlay to match. Saves in place. Env SC_MAXALT, SC_MAXSPUR."""
import bpy, numpy as np, os
from collections import defaultdict, deque
o = bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type == 'MESH')
me = o.data
co = np.array([v.co[:] for v in me.vertices])
adj = defaultdict(set)
for e in me.edges:
    if e.use_seam:
        adj[e.vertices[0]].add(e.vertices[1]); adj[e.vertices[1]].add(e.vertices[0])
n0 = sum(len(v) for v in adj.values()) // 2

# (1) DE-DOUBLE: drop edge if a short alternative path remains (redundant). Keeps essential edges.
MAXALT = int(os.environ.get('SC_MAXALT', '4'))
ndd = 0
for (u, w) in sorted((min(a, b), max(a, b)) for a in list(adj) for b in list(adj[a]) if a < b):
    if w not in adj[u]:
        continue
    adj[u].discard(w); adj[w].discard(u)
    dq = deque([(u, 0)]); seen = {u}; found = False
    while dq:
        cur, d = dq.popleft()
        if cur == w:
            found = True; break
        if d >= MAXALT:
            continue
        for x in adj[cur]:
            if x not in seen:
                seen.add(x); dq.append((x, d + 1))
    if found:
        ndd += 1            # redundant -> leave removed
    else:
        adj[u].add(w); adj[w].add(u)   # essential -> restore

# (2) PRUNE short spurs (dead-end chains up to MAXSPUR edges)
MAXSPUR = int(os.environ.get('SC_MAXSPUR', '12'))
nsp = 0; changed = True
while changed:
    changed = False
    for v in [x for x in list(adj) if len(adj[x]) == 1]:
        chain = [v]; cur = v; prev = None
        while True:
            nxt = [x for x in adj[cur] if x != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]; chain.append(cur)
            if len(adj[cur]) != 2:
                break
        if (len(chain) - 1) <= MAXSPUR and (not adj[chain[-1]] or len(adj[chain[-1]]) >= 3):
            for i in range(len(chain) - 1):
                adj[chain[i]].discard(chain[i + 1]); adj[chain[i + 1]].discard(chain[i])
            nsp += 1; changed = True

for e in me.edges:
    a, b = e.vertices[0], e.vertices[1]; e.use_seam = (b in adj[a])
n1 = sum(len(v) for v in adj.values()) // 2
print('[seam-clean] edges %d -> %d  (de-doubled %d, pruned %d spur-chains)' % (n0, n1, ndd, nsp))

# rebuild green overlay from the cleaned use_seam
_lv = set()
for e in me.edges:
    if e.use_seam:
        _lv.add(int(e.vertices[0])); _lv.add(int(e.vertices[1]))
_old = bpy.data.objects.get('HAIR_SEAM')
if _old:
    bpy.data.objects.remove(_old, do_unlink=True)
_eps = 0.0018 * float((co.max(0) - co.min(0)).max())
_vmap = {}; _nv = []; _nf = []
for p in me.polygons:
    if not any(int(v) in _lv for v in p.vertices):
        continue
    _fi = []
    for v in p.vertices:
        v = int(v)
        if v not in _vmap:
            _vmap[v] = len(_nv); _nv.append(tuple((me.vertices[v].co + me.vertices[v].normal * _eps)[:]))
        _fi.append(_vmap[v])
    _nf.append(_fi)
_om = bpy.data.meshes.new('HAIR_SEAM'); _om.from_pydata(_nv, [], _nf)
_oo = bpy.data.objects.new('HAIR_SEAM', _om); bpy.context.scene.collection.objects.link(_oo)
_oo.matrix_world = o.matrix_world.copy()
_mat = bpy.data.materials.new('SEAMMAT'); _mat.use_nodes = False; _mat.diffuse_color = (0.0, 1.0, 0.15, 1.0); _om.materials.append(_mat)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
