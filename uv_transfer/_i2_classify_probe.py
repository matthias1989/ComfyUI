"""De-risk option 3: does a GLOBAL strandiness score separate charA (strandy) from charB
(smooth)? Per-face geometry is identical; the hair SURFACE texture should differ at scale.
Compute, over the hair region of each mesh:
  - rough fraction (all hair, and head-band hair z>0.30)
  - local normal dispersion: mean over k-NN of (1 - dot(n_i,n_j)) -> high=strandy, low=smooth
If charA >> charB with a clear gap, a threshold classifier is viable."""
import numpy as np, os
try:
    from scipy.spatial import cKDTree
    HAVE_KD = True
except Exception as e:
    HAVE_KD = False; print("no scipy:", e)
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"

def score(f, label):
    vz = np.load(os.path.join(D, f))
    co = vz['co'].astype(np.float64); tris = vz['tris'].astype(np.int64)
    fn = vz['tris_fn'].astype(np.float64); H = vz['tris_hair'].astype(bool)
    rgh = vz['tris_rough'].astype(bool)
    cen = (co[tris[:, 0]] + co[tris[:, 1]] + co[tris[:, 2]]) / 3.0
    z = cen[:, 2]
    head = H & (z > 0.30)
    print("\n[%s]  %s" % (label, f))
    print("  hair faces=%d  head-band hair=%d" % (int(H.sum()), int(head.sum())))
    print("  rough frac: all-hair=%.3f   head-hair=%.3f" %
          (rgh[H].mean() if H.any() else 0, rgh[head].mean() if head.any() else 0))
    if not HAVE_KD:
        return
    # local normal dispersion over head-band hair
    idx = np.where(head)[0]
    C = cen[idx]; N = fn[idx]
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    tree = cKDTree(C)
    K = 9
    _, nn = tree.query(C, k=K, workers=-1)
    disp = np.zeros(len(idx))
    for j in range(1, K):
        disp += 1.0 - np.abs(np.einsum('ij,ij->i', N, N[nn[:, j]]))
    disp /= (K - 1)
    print("  normal-dispersion (head hair): mean=%.4f  median=%.4f  p75=%.4f  p90=%.4f" %
          (disp.mean(), np.median(disp), np.percentile(disp, 75), np.percentile(disp, 90)))

score("_charA_smooth_viz.npz", "charA = STRANDY (must NOT carve)")
score("_viz_B80.npz",         "charB = SMOOTH  (want to carve)")
