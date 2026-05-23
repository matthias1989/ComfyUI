"""Find disconnected components in L arm filtered verts."""
import bpy, numpy as np
from collections import defaultdict, deque

obj = bpy.data.objects.get('geometry_0')
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data

n_verts = len(me.vertices)
co_flat = np.empty(n_verts*3, dtype=np.float64)
me.vertices.foreach_get('co', co_flat)
co = co_flat.reshape(n_verts, 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
T = np.array([mat[r][3] for r in range(3)], dtype=np.float64)
co = co @ R.T + T

n_edges = len(me.edges)
ev_flat = np.empty(n_edges*2, dtype=np.int32)
me.edges.foreach_get('vertices', ev_flat)
EV = ev_flat.reshape(n_edges, 2)

# L arm verts (the working z-filtered set)
idx = np.where((co[:,0]>=-0.41)&(co[:,0]<=-0.065)&(co[:,2]>=0.24)&(co[:,2]<=0.38))[0]
idx_set = set(idx.tolist())
print(f"L arm idx: {len(idx)} verts")

vmask = np.zeros(n_verts, dtype=bool); vmask[idx] = True
keep = vmask[EV[:,0]] & vmask[EV[:,1]]
e_s = EV[keep]
adj = defaultdict(set)
for i0,i1 in zip(e_s[:,0].tolist(), e_s[:,1].tolist()):
    adj[i0].add(i1); adj[i1].add(i0)
print(f"  Edges in subgraph: {len(e_s)}")
print(f"  Verts with at least 1 edge: {len(adj)}")

# Find connected components
visited = set(); components = []
for v in idx.tolist():
    if v in visited: continue
    comp = set(); q = deque([v])
    while q:
        u = q.popleft()
        if u in visited: continue
        visited.add(u); comp.add(u)
        q.extend(adj[u] - visited)
    components.append(comp)

components.sort(key=len, reverse=True)
print(f"  Connected components: {len(components)}")
for i, c in enumerate(components[:8]):
    xv = co[list(c), 0]; zv = co[list(c), 2]
    print(f"  Comp {i}: {len(c)} verts, X=[{xv.min():.3f},{xv.max():.3f}], Z=[{zv.min():.3f},{zv.max():.3f}]")

# X histogram of ALL arm verts in range
print("\nX histogram (L arm, z=[0.24,0.38]):")
xv_all = co[idx, 0]
bins = np.arange(-0.42, -0.06, 0.02)
hist, edges = np.histogram(xv_all, bins=bins)
for h,lo in zip(hist, edges):
    bar = '#'*min(h//10,50)
    print(f"  X [{lo:.2f},{lo+0.02:.2f}]: {h:4d} {bar}")
