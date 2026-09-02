"""Face mask on the frozen gb mesh, CONSTANT-FREE.
The face region is the MediaPipe central-feature hull (anatomical landmarks, not tuned numbers).
The photo hull captures the camera-facing face AND the far side of the skull behind it; we keep the
camera-facing half via a DATA-DRIVEN depth split (median of the region's depth) -> both cheeks, drop the
back of the skull. No throat band, no normalization scales, no thresholds.
Out: _frozen/facevert.npy (per gb vertex)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "_frozen", "gb_mesh.npz"))
co = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); nv = len(co); nf = len(fv)
face = np.load(os.path.join(HERE, "_frozen", "faceregion.npy")).astype(bool)
fy = co[fv][:, :, 1].mean(1)                       # face-centroid depth (axis1; camera/front = lower y)
split = np.median(fy[face])                         # data-driven front/back divider for the hull region
filt = face.copy(); filt[face] = fy[face] < split  # keep the camera-facing half (both cheeks), drop far skull
fvm = np.zeros(nv, bool); fvm[fv[filt].reshape(-1)] = True
np.save(os.path.join(HERE, "_frozen", "facevert.npy"), fvm)
print('[fz-filter] face %d -> front-by-depth %d -> facevert %d  (no constants, no throat)' %
      (int(face.sum()), int(filt.sum()), int(fvm.sum())))
