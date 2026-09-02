"""Characterize the temple hairline boundary (issues 1/3) on the current charB viz.
Per height-bin, the lateral (x) spread of boundary vertices on each side -> tells whether the
seam wanders as a SOLID ear-ward bulge (max-x spikes at some z) or as jaggedness (wide spread)."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
f = "_genseams_viz.npz" if os.path.exists(os.path.join(D, "_genseams_viz.npz")) else "_viz_B80.npz"
print("using", f, "mtime", os.path.getmtime(os.path.join(D, f)))
vz = np.load(os.path.join(D, f))
co = vz['co'].astype(float); tris = vz['tris'].astype(int); H = vz['tris_hair'].astype(bool)
nv = co.shape[0]
vh = np.zeros(nv); vc = np.zeros(nv)
for k in range(3):
    np.add.at(vh, tris[:, k], H.astype(float)); np.add.at(vc, tris[:, k], 1.0)
frac = vh / np.maximum(vc, 1)
bnd = (frac > 0.2) & (frac < 0.8)
x, y, z = co[:, 0], co[:, 1], co[:, 2]
for side, sel in [("RIGHT", x > 0.04), ("LEFT", x < -0.04)]:
    reg = bnd & sel & (z > 0.26) & (z < 0.48)
    ax = np.abs(x)
    print("\n[%s temple] boundary verts=%d" % (side, int(reg.sum())))
    zb = np.linspace(0.26, 0.48, 12)
    for i in range(len(zb) - 1):
        m = reg & (z >= zb[i]) & (z < zb[i + 1])
        if m.sum() > 0:
            axs = ax[m]; ys = y[m]
            print("  z[%.2f-%.2f] n=%3d  |x|: min %.3f med %.3f max %.3f  spread %.3f  (y med %.3f)" %
                  (zb[i], zb[i + 1], int(m.sum()), axs.min(), np.median(axs), axs.max(),
                   axs.max() - axs.min(), np.median(ys)))
