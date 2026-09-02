"""Pathfinding seam prototype (v2). For each hair-region boundary loop: subsample anchors and
reconnect them with Dijkstra shortest paths, BUT restrict the graph to a band of K rings around
the boundary so shortcuts hug the hairline (cut the notch) without slicing through solid hair.
Outputs the clean seam edges to _Bcleanseam.npz for a Blender 3D render."""
import numpy as np, scipy.sparse as sp, os, sys
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict, deque

N = int(os.environ.get('PS_N', '18'))      # anchor spacing along the loop
K = int(os.environ.get('PS_K', '10'))      # band half-width (rings) around the boundary
d = np.load('_Bmesh.npz', allow_pickle=True)
co = d['co']; hair = d['hair']; fv = d['fv']; nv = int(d['nv'])

e2f = defaultdict(list); vadj = defaultdict(set)
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]
        e2f[(u, w) if u < w else (w, u)].append(fi)
        vadj[u].add(w); vadj[w].add(u)

bnd = [e for e, fs in e2f.items() if len(fs) == 2 and hair[fs[0]] != hair[fs[1]]]
badj = defaultdict(list)
for (u, w) in bnd:
    badj[u].append(w); badj[w].append(u)

# band = verts within K rings of any boundary vertex
band = np.zeros(nv, bool); dq = deque()
for v in badj:
    band[v] = True; dq.append((v, 0))
while dq:
    v, dist = dq.popleft()
    if dist >= K:
        continue
    for w in vadj[v]:
        if not band[w]:
            band[w] = True; dq.append((w, dist + 1))
print('boundary verts=%d  band verts=%d' % (len(badj), int(band.sum())))

# walk boundary loops
visited = set()
def walk(start):
    loop = [start]; prev = None; cur = start; steps = 0
    while steps < 100000:
        steps += 1; nxt = None
        for x in badj[cur]:
            if x == prev:
                continue
            e = (cur, x) if cur < x else (x, cur)
            if e not in visited:
                nxt = x; break
        if nxt is None:
            break
        e = (cur, nxt) if cur < nxt else (nxt, cur); visited.add(e)
        if nxt == start:
            break
        loop.append(nxt); prev = cur; cur = nxt
    return loop
loops = []
for v in list(badj.keys()):
    for x in badj[v]:
        e = (v, x) if v < x else (x, v)
        if e not in visited:
            L = walk(v)
            if len(L) > 30:
                loops.append(L)
loops.sort(key=len, reverse=True)
print('loops kept (>30):', [len(L) for L in loops])

# SKIN vertices (touch >=1 skin face). Boundary verts touch both -> included.
skinv = np.zeros(nv, bool)
for fi, verts in enumerate(fv):
    if not hair[fi]:
        for x in verts:
            skinv[int(x)] = True
# graph restricted to band AND skin: chords may cross a concave notch (through its skin mouth)
# but cannot cut through a convex hair bulge (no interior-hair verts) -> fills notches, keeps bulges
rows = []; cols = []; wts = []
for (u, w) in e2f.keys():
    if band[u] and band[w] and skinv[u] and skinv[w]:
        Ln = float(np.linalg.norm(co[u] - co[w])); rows += [u, w]; cols += [w, u]; wts += [Ln, Ln]
G = sp.csr_matrix((wts, (rows, cols)), shape=(nv, nv))
print('skin verts in band=%d' % int((band & skinv).sum()))

clean_edges = []
for main in loops:
    anchors = main[::N]
    if anchors[-1] != main[0]:
        anchors.append(main[0])
    for a, b in zip(anchors[:-1], anchors[1:]):
        _, pred = dijkstra(G, indices=a, return_predecessors=True)
        if pred[b] < 0:
            clean_edges.append((a, b)); continue   # fallback: direct
        path = [b]
        while path[-1] != a and path[-1] >= 0:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        for i in range(len(path) - 1):
            clean_edges.append((path[i], path[i + 1]))
clean_edges = np.array(clean_edges, dtype=np.int64)
np.savez('_Bcleanseam.npz', edges=clean_edges)
print('clean seam edges=%d  (N=%d K=%d)  saved _Bcleanseam.npz' % (len(clean_edges), N, K))
