"""Prototype the temple-hairline STRAIGHTEN for issues 1/3 on charB.
Per side: robustly fit the boundary's lateral position |x| as a line in height z (using per-z-bin
medians, so the ear-ward bulges don't bias it), then count the HAIR faces whose |x| pokes past
fit+margin in the temple band -- those are the ear-ward excursions a clip would remove."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(float); tris = vz['tris'].astype(int); H = vz['tris_hair'].astype(bool)
nv = co.shape[0]
vh = np.zeros(nv); vc = np.zeros(nv)
for k in range(3):
    np.add.at(vh, tris[:, k], H.astype(float)); np.add.at(vc, tris[:, k], 1.0)
frac = vh / np.maximum(vc, 1)
bnd_v = (frac > 0.2) & (frac < 0.8)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
cx, cy, cz = cen[:, 0], cen[:, 1], cen[:, 2]
vx, vy, vz_ = co[:, 0], co[:, 1], co[:, 2]
# temple band
Z0, Z1, YMIN, YMAX, XMAX = 0.26, 0.40, -0.05, 0.12, 0.135
for sname, sgn in [("RIGHT", 1.0), ("LEFT", -1.0)]:
    bvert = bnd_v & (sgn * vx > 0.03) & (np.abs(vx) < XMAX) & (vz_ > Z0) & (vz_ < Z1) & (vy > YMIN) & (vy < YMAX)
    zz = vz_[bvert]; xx = np.abs(vx[bvert])
    zb = np.linspace(Z0, Z1, 9); zc = []; xm = []
    for i in range(len(zb) - 1):
        m = (zz >= zb[i]) & (zz < zb[i + 1])
        if m.sum() >= 3:
            zc.append((zb[i] + zb[i + 1]) / 2); xm.append(np.median(xx[m]))
    if len(zc) < 2:
        print("[%s] too few bins" % sname); continue
    a, b = np.polyfit(zc, xm, 1)
    fit_med = a * np.array(zc) + b
    print("[%s] fit |x| = %.3f*z + %.3f   bin medians=%s" % (sname, a, b, ["%.3f" % v for v in xm]))
    # hair faces in band, excursion past fit+margin
    hb = H & (sgn * cx > 0.03) & (np.abs(cx) < XMAX) & (cz > Z0) & (cz < Z1) & (cy > YMIN) & (cy < YMAX)
    line = a * cz + b
    for mg in [0.003, 0.006, 0.010]:
        clip = hb & (np.abs(cx) > line + mg)
        print("    margin %.3f -> clip %4d hair faces (band has %d)" % (mg, int(clip.sum()), int(hb.sum())))
