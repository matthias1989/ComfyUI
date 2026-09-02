"""Read the ACTUAL UV seams from last_seams_B.blend, classify each seam edge as hairline
(separates a hair face from a skin face) vs an anatomical CUT (both adjacent faces same class),
and report what runs through the temple/ear region. Tells us which seam the user's 1/3 is."""
import bpy, numpy as np, os
from collections import defaultdict
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=os.path.join(D, "last_seams_B.blend"))
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
n = len(me.vertices)
co = np.empty(n * 3); me.vertices.foreach_get('co', co); co = co.reshape(n, 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T = np.array([mat[r][3] for r in range(3)])
co = co @ R.T + T
zmn, zmx = co[:, 2].min(), co[:, 2].max(); xc = (co[:, 0].min() + co[:, 0].max()) / 2; xsp = co[:, 0].max() - co[:, 0].min()
zs = 0.91 / (zmx - zmn); co[:, 2] = co[:, 2] * zs + (-0.41 - zmn * zs); co[:, 0] = (co[:, 0] - xc) * (0.92 / xsp); co[:, 1] = co[:, 1] * zs
hair = np.load(os.path.join(D, "last_seams_hairfaces.npy")).astype(bool)
print("polys=%d hairfaces=%d" % (len(me.polygons), len(hair)))
e2f = defaultdict(list)
for p in me.polygons:
    for ek in p.edge_keys:
        e2f[ek].append(p.index)
cls_count = defaultdict(int); temple = defaultdict(list)
for e in me.edges:
    if not e.use_seam:
        continue
    v0, v1 = e.vertices; mid = (co[v0] + co[v1]) / 2
    fs = e2f.get(e.key, [])
    if len(fs) == 2:
        c = 'HAIRLINE' if hair[fs[0]] != hair[fs[1]] else ('in-hair' if hair[fs[0]] else 'in-skin/cut')
    else:
        c = 'border(%d)' % len(fs)
    cls_count[c] += 1
    if abs(mid[0]) > 0.03 and abs(mid[0]) < 0.20 and 0.22 < mid[2] < 0.48 and -0.12 < mid[1] < 0.16:
        temple[c].append((round(mid[0], 3), round(mid[1], 3), round(mid[2], 3)))
print("\nseam edge classes (whole mesh):", dict(cls_count))
print("\n--- seam edges in TEMPLE/EAR region (|x| .03-.20, z .22-.48) by class ---")
for c, pts in temple.items():
    ax = np.array([abs(p[0]) for p in pts]); zz = np.array([p[2] for p in pts]); yy = np.array([p[1] for p in pts])
    print("  %-14s n=%3d  |x|[%.3f-%.3f]  z[%.3f-%.3f]  y[%.3f-%.3f]" %
          (c, len(pts), ax.min(), ax.max(), zz.min(), zz.max(), yy.min(), yy.max()))
# FRONT-temple (y<0) HAIRLINE seam |x| profile vs z, BOTH sides -> verify both flatten
allh = np.array([(p[0], p[1], p[2]) for p in temple.get('HAIRLINE', [])])  # signed x, y, z
zb = np.linspace(0.38, 0.48, 6)
for sname, ssel in [("RIGHT", allh[:, 0] > 0.03), ("LEFT", allh[:, 0] < -0.03)]:
    sub = allh[ssel]; sub = sub[sub[:, 1] < 0.0]   # front of ear only
    print("\n--- %s FRONT temple HAIRLINE : n=%d ---" % (sname, len(sub)))
    for i in range(len(zb) - 1):
        m = (sub[:, 2] >= zb[i]) & (sub[:, 2] < zb[i + 1])
        if m.sum() > 0:
            a = np.abs(sub[m, 0])
            print("  z[%.3f-%.3f] n=%3d  |x| med %.3f max %.3f" % (zb[i], zb[i + 1], int(m.sum()), np.median(a), a.max()))
