"""Nail the local->normalized coordinate mapping so the temple fix lands exactly on the user's
clicked vertex (LOCAL 0.036, -0.029, 0.445). Print the world matrix, the gen_seams normalization
factors, the target's normalized coords, and the nearest hairline-seam vertex to it."""
import bpy, numpy as np, os
from collections import defaultdict
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
bpy.ops.wm.open_mainfile(filepath=os.path.join(D, "last_seams_B.blend"))
obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type == 'MESH')
me = obj.data
n = len(me.vertices)
loc = np.empty(n * 3); me.vertices.foreach_get('co', loc); loc = loc.reshape(n, 3)   # LOCAL
mat = obj.matrix_world
print("matrix_world:")
for r in range(4):
    print("   ", [round(mat[r][c], 4) for c in range(4)])
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T = np.array([mat[r][3] for r in range(3)])
co = loc @ R.T + T   # WORLD
zmn, zmx = co[:, 2].min(), co[:, 2].max(); xc = (co[:, 0].min() + co[:, 0].max()) / 2; xsp = co[:, 0].max() - co[:, 0].min()
zs = 0.91 / (zmx - zmn); zoff = -0.41 - zmn * zs; xsc = 0.92 / xsp
nrm = co.copy(); nrm[:, 2] = co[:, 2] * zs + zoff; nrm[:, 0] = (co[:, 0] - xc) * xsc; nrm[:, 1] = co[:, 1] * zs
print("norm factors: z_scale=%.4f z_off=%.4f  x_scale=%.4f x_ctr=%.4f" % (zs, zoff, xsc, xc))
tgt_local = np.array([0.035779, -0.028645, 0.44474])
di = np.linalg.norm(loc - tgt_local, axis=1); vi = int(di.argmin())
print("nearest mesh vert to target: idx=%d  local=%s  dist=%.5f" % (vi, np.round(loc[vi], 4).tolist(), di[vi]))
print("  -> WORLD      =", np.round(co[vi], 4).tolist())
print("  -> NORMALIZED =", np.round(nrm[vi], 4).tolist())
# nearest hairline-seam vertex to the target
hair = np.load(os.path.join(D, "last_seams_hairfaces.npy")).astype(bool)
e2f = defaultdict(list)
for p in me.polygons:
    for ek in p.edge_keys:
        e2f[ek].append(p.index)
seamverts = set()
for e in me.edges:
    if e.use_seam:
        fs = e2f.get(e.key, [])
        if len(fs) == 2 and hair[fs[0]] != hair[fs[1]]:
            seamverts.update(e.vertices)
sv = np.array(sorted(seamverts))
ds = np.linalg.norm(loc[sv] - tgt_local, axis=1); j = int(ds.argmin())
print("nearest HAIRLINE-seam vert to target: local=%s  normalized=%s  dist=%.5f" %
      (np.round(loc[sv[j]], 4).tolist(), np.round(nrm[sv[j]], 4).tolist(), ds[j]))
