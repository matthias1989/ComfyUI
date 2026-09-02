"""Test the ROUGH discriminator for the issue-2 carve. Hypothesis: charB's smooth forehead
hair is ~rough (carvable) while charA's strandy framing gold is rough (protected). A carve
gated by (forward-facing & ~rough) should lift charB's hairline yet spare charA's gold."""
import numpy as np, os
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"

def load(f):
    vz = np.load(os.path.join(D, f))
    co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
    cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
    return (cen[:, 0], cen[:, 2], cen[:, 1], vz['tris_fn'][:, 1],
            vz['tris_hair'].astype(bool), vz['tris_rough'].astype(bool),
            vz['tris_shell'].astype(bool), vz['tris_face'].astype(np.int64))

# ---- charB: how much forehead hair does (forward & ~rough) carve? (want >> 0) ----
xB, zB, yB, nyB, HB, rghB, shB, tfB = load("_viz_B80.npz")
foreB = HB & (np.abs(xB) < 0.15) & (zB > 0.40) & (zB < 0.49)
print("charB forehead-band hair=%d  | ~rough=%d (%d%%)  rough=%d"
      % (int(foreB.sum()), int((foreB & ~rghB).sum()),
         100 * int((foreB & ~rghB).sum()) / max(int(foreB.sum()), 1), int((foreB & rghB).sum())))
for thr in [-0.45, -0.40, -0.35, -0.30]:
    c = foreB & (nyB < thr) & ~rghB
    print("   carve ny<%5.2f & ~rough : %4d forehead faces lifted" % (thr, int(c.sum())))

# ---- charA: does (forward & ~rough) carve hit gold? (want ~0) ----
gold = np.load(os.path.join(D, "_goldA_hairfaces.npy")).astype(bool); nF = gold.shape[0]
xA, zA, yA, nyA, HA, rghA, shA, tfA = load("_charA_smooth_viz.npz")
goldtri = gold[tfA]
print("\ncharA gold tris in forward forehead (ny<-0.40,|x|<.15,z>.30)=%d  | of those rough=%d (%d%%)"
      % (int((goldtri & (nyA < -0.40) & (np.abs(xA) < 0.15) & (zA > 0.30)).sum()),
         int((goldtri & (nyA < -0.40) & (np.abs(xA) < 0.15) & (zA > 0.30) & rghA).sum()),
         100 * int((goldtri & (nyA < -0.40) & (np.abs(xA) < 0.15) & (zA > 0.30) & rghA).sum())
         / max(int((goldtri & (nyA < -0.40) & (np.abs(xA) < 0.15) & (zA > 0.30)).sum()), 1)))
for thr in [-0.45, -0.40, -0.35, -0.30]:
    ff = (nyA < thr) & (np.abs(xA) < 0.15) & (zA > 0.30) & ~rghA   # carve, ~rough gate, NO shell
    cf = np.zeros(nF, bool); np.logical_or.at(cf, tfA, ff)
    print("   charA carve ny<%5.2f & ~rough : quad-faces %5d  | of which GOLD %5d" %
          (thr, int(cf.sum()), int((cf & gold).sum())))
