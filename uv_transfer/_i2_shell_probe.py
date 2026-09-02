"""Verify the issue-2 real plan on charB: would relaxing the _facefwd ny-threshold carve
reach the forehead hair, and does the ~_shell gate block it? Pure-numpy on the charB viz."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
vz = np.load(os.path.join(D, "_viz_B80.npz"))
print("keys:", list(vz.keys()))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; H = vz['tris_hair'].astype(bool)
shell = vz['tris_shell'].astype(bool) if 'tris_shell' in vz.files else None
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]; ny = fn[:, 1]

base = H & (np.abs(x) < 0.15) & (z > 0.30)              # the _facefwd spatial gate
fore = base & (z > 0.40) & (z < 0.49)                   # forehead sub-band == issue 2
print("hair in _facefwd gate: %d   of which forehead-band (z .40-.49): %d" % (int(base.sum()), int(fore.sum())))
print("shell key present:", shell is not None)
for thr in [-0.52, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20]:
    mf = fore & (ny < thr)
    if shell is not None:
        blk = int((mf & shell).sum()); car = int((mf & ~shell).sum())
        print("  ny<%5.2f  forehead hit %4d  | ~shell-carvable %4d  shell-BLOCKED %4d (%3d%%)"
              % (thr, int(mf.sum()), car, blk, 100 * blk / max(int(mf.sum()), 1)))
    else:
        print("  ny<%5.2f  forehead hit %4d  (no shell key)" % (thr, int(mf.sum())))
# how much of the forehead band is shell at all (the gate's reach ceiling):
if shell is not None:
    print("forehead-band hair that is shell: %d / %d (%d%%)"
          % (int((fore & shell).sum()), int(fore.sum()), 100 * int((fore & shell).sum()) / max(int(fore.sum()), 1)))
