"""How many charB hair faces does the forward-facing carve remove at each setting? Goal: find
a MODEST, forehead-focused raise (center only), not the ~8k whole-front strip that ny<-0.40
over z>0.30,|x|<0.15 gave. Sweep ny threshold, half-width, and z-floor on the charB viz."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_viz_B80.npz"))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, z = cen[:, 0], cen[:, 2]; ny = fn[:, 1]
print("charB head-band hair (z>0.30, |x|<0.15): %d" % int((H & (z > 0.30) & (np.abs(x) < 0.15)).sum()))
for zmin in [0.30, 0.40, 0.43]:
    for XW in [0.15, 0.10, 0.07]:
        row = []
        for thr in [-0.40, -0.50, -0.55, -0.60, -0.65, -0.70, -0.78]:
            c = int((H & (ny < thr) & (np.abs(x) < XW) & (z > zmin)).sum())
            row.append("%5d" % c)
        print("zmin=%.2f XW=%.2f  ny<[-.40 -.50 -.55 -.60 -.65 -.70 -.78]: %s" % (zmin, XW, " ".join(row)))
