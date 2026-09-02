"""Which side of the temple seam is HAIR? Decides whether '+more hair' = toward or away from
the ear. Report hair-fraction vs lateral |x| in the temple band, and locate the ear."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co'].astype(float); tris = vz['tris'].astype(int); H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
cx, cy, cz = cen[:, 0], cen[:, 1], cen[:, 2]
ax = np.abs(cx)
# ear ~ the lateral-most surface around z 0.27-0.34, front-ish
ear = (ax > 0.10) & (cz > 0.26) & (cz < 0.35) & (cy > -0.06) & (cy < 0.10)
print("ear region: faces=%d  |x| max %.3f  mean(|x|,y,z)=(%.3f,%.3f,%.3f)" %
      (int(ear.sum()), ax[ear].max() if ear.any() else 0,
       ax[ear].mean() if ear.any() else 0, cy[ear].mean() if ear.any() else 0, cz[ear].mean() if ear.any() else 0))
for sname, sgn in [("RIGHT", 1.0), ("LEFT", -1.0)]:
    band = (sgn * cx > 0.0) & (cz > 0.27) & (cz < 0.40) & (cy > -0.05) & (cy < 0.12)
    print("\n[%s temple] hair-fraction vs lateral |x| (toward ear = larger |x|):" % sname)
    xb = np.linspace(0.02, 0.14, 13)
    for i in range(len(xb) - 1):
        m = band & (ax >= xb[i]) & (ax < xb[i + 1])
        if m.sum() > 0:
            print("   |x|[%.2f-%.2f] n=%4d  hair-frac %.2f" % (xb[i], xb[i + 1], int(m.sum()), H[m].mean()))
