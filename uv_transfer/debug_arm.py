"""Debug L arm vert distribution to understand Z range and connectivity."""
import bpy, numpy as np

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

# L arm verts: x in [-0.41, -0.065], various Z ranges
for z_lo, z_hi in [(0.24, 0.38), (0.20, 0.42), (0.18, 0.45), (0.15, 0.50)]:
    idx = np.where((co[:,0]>=-0.41)&(co[:,0]<=-0.065)&(co[:,2]>=z_lo)&(co[:,2]<=z_hi))[0]
    if len(idx) == 0:
        print(f"Z=[{z_lo},{z_hi}]: 0 verts")
        continue
    # Check connectivity: is there a path from x≈-0.075 to x≈-0.40?
    from collections import defaultdict
    vmask = np.zeros(n_verts, dtype=bool); vmask[idx] = True
    keep = vmask[EV[:,0]] & vmask[EV[:,1]]
    e_s = EV[keep]
    adj = defaultdict(set)
    for i0, i1 in e_s[:,0:2].tolist():
        adj[i0].add(i1); adj[i1].add(i0)

    # Find shoulder candidates (x near -0.075)
    sv = idx[np.abs(co[idx,0]-(-0.075))<0.08]
    ev2 = idx[np.abs(co[idx,0]-(-0.40))<0.08]
    sv_in_adj = [i for i in sv if i in adj]
    ev_in_adj = [i for i in ev2 if i in adj]

    # BFS from best shoulder vert to check if wrist reachable
    if sv_in_adj and ev_in_adj:
        start = sorted(sv_in_adj, key=lambda i: co[i,1])[0]
        goal_set = set(ev_in_adj)
        visited = {start}; queue = [start]; found = False
        while queue and not found:
            nq = []
            for u in queue:
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        if v in goal_set: found = True; break
                        nq.append(v)
                if found: break
            queue = nq
        print(f"Z=[{z_lo:.2f},{z_hi:.2f}]: {len(idx)} verts, {len(e_s)} edges, "
              f"sv_adj={len(sv_in_adj)}, ev_adj={len(ev_in_adj)}, "
              f"path_exists={found}")
    else:
        print(f"Z=[{z_lo:.2f},{z_hi:.2f}]: {len(idx)} verts, sv_adj={len(sv_in_adj)}, ev_adj={len(ev_in_adj)} (no endpoints in adj)")

# Also show Z histogram of arm verts (x in [-0.41, -0.065])
arm_all = np.where((co[:,0]>=-0.41)&(co[:,0]<=-0.065))[0]
print(f"\nAll L arm verts (no Z filter): {len(arm_all)}")
if len(arm_all):
    zv = co[arm_all, 2]
    print(f"  Z range: [{zv.min():.3f}, {zv.max():.3f}]")
    # Histogram
    bins = np.arange(-0.5, 0.55, 0.05)
    hist, edges = np.histogram(zv, bins=bins)
    for i,(h,lo) in enumerate(zip(hist, edges)):
        if h > 0:
            print(f"  Z [{lo:.2f},{lo+0.05:.2f}]: {h}")
