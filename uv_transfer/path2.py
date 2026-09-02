"""Step 2: connect crease arcs into ONE closed loop tracing the hairline.
- crease edges -> connected components (arcs); keep major (>= P2_MIN verts); endpoints = diameter
- ORDER the arcs along the hairline using the hair-region boundary loop as a 1-D backbone
  (t = index of nearest boundary-loop vertex) -- robust ordering, handles the front U-shape
- connect consecutive arcs with Dijkstra in a band around the hairline, crease edges CHEAP, so the
  path rides creases where they exist and bridges the gaps with short geodesics (cuts notches)
Out P2_OUT: {loop edges, anchors}."""
import numpy as np, scipy.sparse as sp, os
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from collections import defaultdict, deque

mesh = np.load(os.environ['P2_MESH'], allow_pickle=True)
co = mesh['co']; fv = mesh['fv']; hair = mesh['hair']; nv = int(mesh['nv'])
cre = np.load(os.environ['P2_CRE'])['edges']
MINV = int(os.environ.get('P2_MIN', '25')); P2K = int(os.environ.get('P2_K', '12'))
P2_BND = float(os.environ.get('P2_BND', '0.4'))   # cost of riding the hair-mask boundary (fallback hairline)

e2f = defaultdict(list); vadj = defaultdict(set)
for fi, verts in enumerate(fv):
    vs = [int(x) for x in verts]; k = len(vs)
    for a in range(k):
        u, w = vs[a], vs[(a + 1) % k]; e = (u, w) if u < w else (w, u)
        e2f[e].append(fi); vadj[u].add(w); vadj[w].add(u)

# ordered hair-region boundary loop (backbone for arc ordering)
badj = defaultdict(list); bnd_set = set()
for e, fs in e2f.items():
    if len(fs) == 2 and hair[fs[0]] != hair[fs[1]]:
        badj[e[0]].append(e[1]); badj[e[1]].append(e[0]); bnd_set.add(e)
visited = set()
def walk(start):
    loop = [start]; prev = None; cur = start; st = 0
    while st < 200000:
        st += 1; nxt = None
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
bloops = []
for v in list(badj.keys()):
    for x in badj[v]:
        e = (v, x) if v < x else (x, v)
        if e not in visited:
            L = walk(v)
            if len(L) > 30:
                bloops.append(L)
bloops.sort(key=len, reverse=True)
bigloop = bloops[0]
bl_tree = cKDTree(co[bigloop])
def tpos(v):
    return int(bl_tree.query(co[v])[1])

# band around the boundary
band = np.zeros(nv, bool); dq = deque()
for v in badj:
    band[v] = True; dq.append((v, 0))
while dq:
    v, dd = dq.popleft()
    if dd >= P2K:
        continue
    for w in vadj[v]:
        if not band[w]:
            band[w] = True; dq.append((w, dd + 1))

# crease arcs
creset = set((int(u), int(w)) if u < w else (int(w), int(u)) for u, w in cre)
crev = np.unique(cre); cidx = {int(v): i for i, v in enumerate(crev.tolist())}; ncv = len(crev)
rr = [cidx[int(u)] for u, w in cre]; cc = [cidx[int(w)] for u, w in cre]
Gc = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(ncv, ncv)); Gc = Gc + Gc.T
ncomp, lab = connected_components(Gc, directed=False)
comps = [crev[lab == ci] for ci in range(ncomp) if int((lab == ci).sum()) >= MINV]

# graph over ALL edges (so a bridge never fails) with tiered weights: ride creases (cheap), stay in
# the hairline band (normal), only stray off-zone as a last resort (expensive) -> no 3D jumps.
P2_FWD = float(os.environ.get('P2_FWD', '4.0'))   # how strongly the loop prefers the FRONT (most-forward) edge
ymin_m, ymax_m = float(co[:, 1].min()), float(co[:, 1].max())
rows = []; cols = []; wts = []
for (u, w), fs in e2f.items():
    L = float(np.linalg.norm(co[u] - co[w]))
    if (u, w) in creset:
        L *= 0.05
    elif (u, w) in bnd_set:
        L *= P2_BND          # hair-mask boundary = fallback hairline where creases are too faint (temple)
    elif band[u] and band[w]:
        L *= 1.0
    else:
        L *= 10.0
    yn = (0.5 * (co[u, 1] + co[w, 1]) - ymin_m) / (ymax_m - ymin_m + 1e-9)   # 0=front, 1=back (face=-Y)
    L *= (1.0 + P2_FWD * yn)   # prefer the front -> loop hugs the most-forward (most-left) hairline
    rows += [u, w]; cols += [w, u]; wts += [L, L]
G = sp.csr_matrix((wts, (rows, cols)), shape=(nv, nv))

def path(a, b):
    dist, pred = dijkstra(G, indices=a, return_predecessors=True)
    if a != b and pred[b] < 0:
        return [a, b]
    p = [b]
    while p[-1] != a and p[-1] >= 0:
        p.append(int(pred[p[-1]]))
    return p[::-1]
def endpoints(vs):
    rmask = np.zeros(nv, bool); rmask[vs] = True
    d0 = dijkstra(G, indices=int(vs[0])); e1 = int(np.argmax(np.where(rmask, d0, -1)))
    d1 = dijkstra(G, indices=e1); e2 = int(np.argmax(np.where(rmask, d1, -1)))
    return e1, e2

arcs = [endpoints(vs) for vs in comps]
arcs = [(a, b) if tpos(a) <= tpos(b) else (b, a) for (a, b) in arcs]   # orient by t
arcs.sort(key=lambda ab: tpos(ab[0]))                                  # order along hairline
anchors = [e for ar in arcs for e in ar]

loopverts = path(arcs[0][0], arcs[0][1])
for i in range(1, len(arcs)):
    loopverts += path(loopverts[-1], arcs[i][0])[1:]
    loopverts += path(arcs[i][0], arcs[i][1])[1:]
loopverts += path(loopverts[-1], arcs[0][0])[1:]
loopedges = [(loopverts[i], loopverts[i + 1]) for i in range(len(loopverts) - 1) if loopverts[i] != loopverts[i + 1]]
np.savez(os.environ['P2_OUT'], loop=np.array(loopedges, dtype=np.int64), anchors=np.array(anchors, dtype=np.int64))
print('arcs=%d bigloop=%d loop_edges=%d' % (len(arcs), len(bigloop), len(loopedges)))
