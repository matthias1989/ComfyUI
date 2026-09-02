"""Temple straighten via SMOOTHED-PROFILE clip (not a line). Per side: median |x| per z-bin,
smooth that profile along z (keeps the curve, drops the wobble), then count hair faces poking
outward past trend+margin. Should clip only the ear-ward wobble, not the curved hairline."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(float); tris = vz['tris'].astype(int); H = vz['tris_hair'].astype(bool)
nv = co.shape[0]; vh = np.zeros(nv); vc = np.zeros(nv)
for k in range(3):
    np.add.at(vh, tris[:, k], H.astype(float)); np.add.at(vc, tris[:, k], 1.0)
frac = vh / np.maximum(vc, 1); bnd_v = (frac > 0.2) & (frac < 0.8)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
cx, cy, cz = cen[:, 0], cen[:, 1], cen[:, 2]; vx, vy, vz_ = co[:, 0], co[:, 1], co[:, 2]
Z0, Z1, YMIN, YMAX, XMAX = 0.26, 0.40, -0.05, 0.12, 0.135

def smooth_profile(zc, med, w=3):
    ok = ~np.isnan(med)
    med = np.interp(zc, zc[ok], med[ok])             # fill empty bins
    k = np.ones(w) / w
    return np.convolve(np.pad(med, w, 'edge'), k, 'same')[w:-w]

for sname, sgn in [("RIGHT", 1.0), ("LEFT", -1.0)]:
    bv = bnd_v & (sgn * vx > 0.03) & (np.abs(vx) < XMAX) & (vz_ > Z0) & (vz_ < Z1) & (vy > YMIN) & (vy < YMAX)
    zz = vz_[bv]; xx = np.abs(vx[bv])
    zb = np.linspace(Z0, Z1, 15); zc = 0.5 * (zb[:-1] + zb[1:])
    med = np.array([np.median(xx[(zz >= zb[i]) & (zz < zb[i + 1])]) if ((zz >= zb[i]) & (zz < zb[i + 1])).sum() >= 3 else np.nan
                    for i in range(len(zb) - 1)])
    sm = smooth_profile(zc, med, 3)
    hb = H & (sgn * cx > 0.03) & (np.abs(cx) < XMAX) & (cz > Z0) & (cz < 0.37) & (cy > YMIN) & (cy < YMAX)
    trend = np.interp(cz, zc, sm)
    print("[%s] raw med   =%s" % (sname, ["%.3f" % v for v in med]))
    print("       smoothed=%s" % ["%.3f" % v for v in sm])
    for mg in [0.003, 0.005, 0.008]:
        clip = hb & (np.abs(cx) > trend + mg)
        print("    margin %.3f -> clip %4d / %d band hair" % (mg, int(clip.sum()), int(hb.sum())))
