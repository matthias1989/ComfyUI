"""Measure the FACE region's share of the UV atlas in the currently-open blend.

Local harness for the face-chart work: gen_seams clears seams and re-unwraps from
scratch (gen_seams.py:1444), so it can be re-run on a saved blend and the UV result
measured without a full bake. Reports UV-area share and the equivalent texel count
at 4096, plus how many disjoint UV components the face lands in.
"""
import bpy, bmesh, numpy as np
from collections import deque

me = max((o for o in bpy.data.objects if o.type == 'MESH'),
         key=lambda o: len(o.data.polygons)).data
uvl = me.uv_layers.active
V = np.array([v.co for v in me.vertices], dtype=np.float64)

cen = np.zeros((len(me.polygons), 3)); nrm = np.zeros((len(me.polygons), 3))
for p in me.polygons:
    cen[p.index] = p.center; nrm[p.index] = p.normal

# gen_seams normalises Z=height, Y=depth. Face = top of head, facing -Y (front).
zmax = cen[:, 2].max()
sel = np.where((cen[:, 2] > zmax - 0.115) & (nrm[:, 1] < -0.45) & (np.abs(cen[:, 0]) < 0.08))[0]

def uvarea(fi):
    L = list(me.polygons[fi].loop_indices)
    P = [uvl.data[i].uv for i in L]; s = 0.0
    for k in range(1, len(P) - 1):
        a, b, c = P[0], P[k], P[k + 1]
        s += 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
    return s

fa = sum(uvarea(i) for i in sel)
ta = sum(uvarea(p.index) for p in me.polygons)

# UV-connected components restricted to the face selection
bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table(); bm.edges.ensure_lookup_table()
ul = bm.loops.layers.uv.active
S = set(int(i) for i in sel); adj = {i: set() for i in S}
for e in bm.edges:
    if len(e.link_faces) == 2:
        a, b = e.link_faces
        if a.index in S and b.index in S:
            vs = {e.verts[0].index, e.verts[1].index}
            ua = {l.vert.index: tuple(l[ul].uv) for l in a.loops if l.vert.index in vs}
            ub = {l.vert.index: tuple(l[ul].uv) for l in b.loops if l.vert.index in vs}
            if all(abs(ua[v][0]-ub[v][0]) < 1e-5 and abs(ua[v][1]-ub[v][1]) < 1e-5
                   for v in vs if v in ua and v in ub):
                adj[a.index].add(b.index); adj[b.index].add(a.index)
bm.free()
seen = set(); comps = []
for f in S:
    if f in seen: continue
    q = deque([f]); g = 0
    while q:
        c = q.popleft()
        if c in seen: continue
        seen.add(c); g += 1; q.extend(adj[c] - seen)
    comps.append(g)
comps.sort(reverse=True)
print("[FACETEST] face faces=%d" % len(sel))
print("[FACETEST] face UV area=%.6f  = %.2f%% of used UV  -> ~%d texels @4096"
      % (fa, 100.0 * fa / max(ta, 1e-12), int(fa * 4096 * 4096)))
print("[FACETEST] UV components=%d  largest=%d faces" % (len(comps), comps[0] if comps else 0))
