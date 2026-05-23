"""Debug: show spatial extent of top UV islands to identify body parts."""
import bpy, bmesh, numpy as np
from collections import deque

obj = bpy.data.objects.get('geometry_0')
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data

# Get world-space vert positions
n_verts = len(me.vertices)
co_flat = np.empty(n_verts*3, dtype=np.float64)
me.vertices.foreach_get('co', co_flat)
co = co_flat.reshape(n_verts, 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
T = np.array([mat[r][3] for r in range(3)], dtype=np.float64)
co = co @ R.T + T

# Build UV island adjacency via BMesh
bm = bmesh.new(); bm.from_mesh(me)
bm.faces.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.verts.ensure_lookup_table()
uv_layer = bm.loops.layers.uv.active

# Build face adjacency (UV-connected)
fadj = {f.index: set() for f in bm.faces}
for e in bm.edges:
    if e.seam or len(e.link_faces) != 2: continue
    fa, fb = e.link_faces
    vs = {e.verts[0].index, e.verts[1].index}
    uva = {lp.vert.index: tuple(lp[uv_layer].uv) for lp in fa.loops if lp.vert.index in vs}
    uvb = {lp.vert.index: tuple(lp[uv_layer].uv) for lp in fb.loops if lp.vert.index in vs}
    same = all(abs(uva[vi][0]-uvb[vi][0])<1e-5 and abs(uva[vi][1]-uvb[vi][1])<1e-5
               for vi in vs if vi in uva and vi in uvb)
    if same:
        fadj[fa.index].add(fb.index); fadj[fb.index].add(fa.index)

# Flood fill to find islands
visited = set(); islands = []
for fi in range(len(bm.faces)):
    if fi in visited: continue
    stack = [fi]; faces = []
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur); faces.append(cur)
        stack.extend(fadj[cur] - visited)
    islands.append(faces)

islands.sort(key=len, reverse=True)
print(f"Total islands: {len(islands)}")

# For top 12, show spatial extent using face center positions
print("\nTop 12 islands (size, X range, Z range):")
for i, faces in enumerate(islands[:12]):
    xvals = []; zvals = []
    for fi in faces:
        f = bm.faces[fi]
        vs = [v.index for v in f.verts]
        cx = np.mean([co[v,0] for v in vs])
        cz = np.mean([co[v,2] for v in vs])
        xvals.append(cx); zvals.append(cz)
    xvals = np.array(xvals); zvals = np.array(zvals)
    print(f"  Island {i}: {len(faces)} faces, X=[{xvals.min():.3f},{xvals.max():.3f}], Z=[{zvals.min():.3f},{zvals.max():.3f}]")

bm.free()

# Also print seam edge count
seam_count = sum(1 for e in me.edges if e.use_seam)
print(f"\nSeam edges marked: {seam_count}")
