"""Gold-guard check for the issue-2 carve on charA: would the shell-FREE forward-facing
forehead carve remove any of charA's approved gold hair faces? Pure-numpy on charA viz."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
gold = np.load(os.path.join(D, "_goldA_hairfaces.npy"))
print("gold:", gold.shape, gold.dtype, "true=", int(gold.astype(bool).sum()))
vz = np.load(os.path.join(D, "_charA_smooth_viz.npz"))
print("viz keys:", list(vz.files))
co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
fn = vz['tris_fn']; tf = vz['tris_face'].astype(np.int64)
shell = vz['tris_shell'].astype(bool) if 'tris_shell' in vz.files else None
cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
x, z = cen[:, 0], cen[:, 2]; ny = fn[:, 1]
nF = gold.shape[0]; gm = gold.astype(bool)
print("charA quad-faces=%d  tris=%d  tf range [%d,%d]" % (nF, len(tris), int(tf.min()), int(tf.max())))
for XW in [0.15, 0.12, 0.10]:
    print("--- |x| < %.2f ---" % XW)
    for thr in [-0.52, -0.45, -0.40, -0.35, -0.30]:
        ff = (ny < thr) & (np.abs(x) < XW) & (z > 0.30)        # _facefwd, NO shell gate
        cf = np.zeros(nF, bool); np.logical_or.at(cf, tf, ff)   # quad carved if any tri carved
        print("   ny<%5.2f  carved quad-faces %5d  | of which GOLD %5d" %
              (thr, int(cf.sum()), int((cf & gm).sum())))
