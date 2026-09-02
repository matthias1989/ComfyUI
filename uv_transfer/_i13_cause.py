"""Characterize the temple-drift hair faces (z .28-.38, |x| .045-.07, y .02-.07) vs known real
hair (crown) and known skin (forehead): shell support, roughness, normals. Tells whether the
drift is real draping hair (can't recede cleanly) or a weak/false extension (cause-fixable)."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_genseams_viz.npz"))
co = vz['co']; tris = vz['tris']; H = vz['tris_hair'].astype(bool)
rough = vz['tris_rough'].astype(bool); shell = vz['tris_shell'].astype(bool); fn = vz['tris_fn']
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]; ax = np.abs(x)
drift  = H & (ax > 0.045) & (ax < 0.07) & (z > 0.28) & (z < 0.38) & (y > 0.02) & (y < 0.07)
medial = H & (ax > 0.0) & (ax < 0.045) & (z > 0.28) & (z < 0.38) & (y > 0.02) & (y < 0.07)  # hair just medial of the drift
refhair = H & (z > 0.43) & (ax < 0.05)                                   # crown = real hair
refskin = (~H) & (z > 0.40) & (z < 0.46) & (ax < 0.035) & (y < -0.02)    # forehead skin
for nm, m in [('DRIFT      ', drift), ('medial-hair', medial), ('ref HAIR(crown)', refhair), ('ref SKIN(brow)', refskin)]:
    if int(m.sum()) > 0:
        print("%-16s n=%5d  shell=%.2f  rough=%.2f  ny=%+.2f  nz=%+.2f  nx=%+.2f" %
              (nm, int(m.sum()), shell[m].mean(), rough[m].mean(), fn[m, 1].mean(), fn[m, 2].mean(), np.abs(fn[m, 0]).mean()))
