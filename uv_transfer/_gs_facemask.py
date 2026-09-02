"""IN-BAKE face mask = exactly the offline _frozen_facemask + _frozen_facefilter, env-driven.
MediaPipe central-feature hull on the front PHOTO -> face-CENTROID-in-hull projection onto the path3 mesh
(density-independent) -> camera-facing half by a DATA-DRIVEN depth median (both cheeks, drop far skull).
No tuned constants. Env: GS_MESH (path3 mesh npz: co/fv), GS_PHOTO (front .png), GS_OUT (facevert .npy).
Run with python_embeded (MediaPipe). The bridge calls this after greenborder; postfix_loop deletes the
loop edges that touch this mask (the across-face crossing), leaving the forehead hairline arc intact."""
import os, sys, numpy as np, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
TR = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2"
sys.path.insert(0, TR)
from projection.landmark_warp import _get_landmarker
import mediapipe as mp
MESH = os.environ.get('GS_MESH', os.path.join(HERE, '_gb_mesh.npz'))
PHOTO = os.environ.get('GS_PHOTO', os.path.join(HERE, '_front_photo.png'))
OUT = os.environ.get('GS_OUT', os.path.join(HERE, '_gb_facevert.npy'))
IH = IW = 896

# ---- MediaPipe central-feature hull on the front photo ----
rgb = np.array(Image.open(PHOTO).convert("RGB"))
if rgb.shape[0] != IH or rgb.shape[1] != IW:
    rgb = cv2.resize(rgb, (IW, IH), interpolation=cv2.INTER_AREA)
sat = rgb.max(2).astype(int) - rgb.min(2).astype(int)
fg = sat > 20
rows = np.where(fg.any(1))[0]; cols = np.where(fg.any(0))[0]
ft, fb = int(rows[0]), int(rows[-1]); fh = fb - ft + 1
hb = ft + int(0.15 * fh)
ub = fg[ft: ft + max(8, int(0.09 * fh))]
hcols = np.where(ub.any(0))[0]; pad = int(0.04 * IW)
box = (max(0, int(hcols[0]) - pad), min(IW, int(hcols[-1]) + pad), ft, hb)
x0, x1, y0, y1 = box; sub = rgb[y0:y1, x0:x1]; up = 5
s2 = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up), interpolation=cv2.INTER_CUBIC)
res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(s2)))
if not res.face_landmarks:
    print("[gs-facemask] NO landmarks -> empty mask (postfix falls back to geometric)"); np.save(OUT, np.zeros(1, bool)); sys.exit(0)
H2, W2 = s2.shape[:2]
lm = np.array([[x0 + p.x * W2 / up, y0 + p.y * H2 / up] for p in res.face_landmarks[0]])
xl = float(lm[33, 0]); xr = float(lm[263, 0])                 # central features within the outer-eye-corner X span
selc = (lm[:, 0] >= xl) & (lm[:, 0] <= xr)
poly = cv2.convexHull(lm[selc].astype(np.int32))
mask = np.zeros((IH, IW), np.uint8); cv2.fillConvexPoly(mask, poly, 255)

# ---- project: face CENTROID in hull, aligned to the photo fg ----
d = np.load(MESH); co_all = d["co"].astype(np.float64); fv = d["fv"].astype(np.int64); nv = len(co_all); nf = len(fv)
v0 = co_all[:, [0, 2, 1]].copy()                              # [width,depth,height] -> [x,height,depth]
vt = v0.copy()
for _ in range(2):
    c = (vt.min(0) + vt.max(0)) * 0.5; vt = vt - c
    r = np.sqrt((vt**2).sum(1).max()); vt = vt * (1.15 / (r * 2 + 1e-9))
uc = vt[:, 0] / (1.15 / 2); vc = vt[:, 1] / (1.15 / 2)
fgc = np.argwhere(fg); iy0, iy1 = int(fgc[:, 0].min()), int(fgc[:, 0].max())
ich = float(iy1 - iy0 + 1); icx = float(fgc[:, 1].mean()); icy = float(iy0 + iy1) / 2.0
ppc = ich / float(vc.max() - vc.min() + 1e-9)
mcx = float((uc.min() + uc.max()) * 0.5); mcy = float((vc.min() + vc.max()) * 0.5)
xpix = icx + (uc - mcx) * ppc; ypix = icy - (vc - mcy) * ppc
cxp = xpix[fv].mean(1); cyp = ypix[fv].mean(1)
ix = np.round(cxp).astype(np.int64); iy = np.round(cyp).astype(np.int64)
valid = (ix >= 0) & (ix < IW) & (iy >= 0) & (iy < IH)
face = np.zeros(nf, bool); face[valid] = mask[iy[valid], ix[valid]] > 0

# ---- camera-facing half by DATA-DRIVEN depth median (both cheeks, drop far skull) ----
fy = co_all[fv][:, :, 1].mean(1)                             # face-centroid depth (axis1; camera/front = lower y)
split = np.median(fy[face]) if face.any() else 0.0
filt = face.copy(); filt[face] = fy[face] < split
fvm = np.zeros(nv, bool); fvm[fv[filt].reshape(-1)] = True
np.save(OUT, fvm)
print("[gs-facemask] %d landmarks -> face %d faces, front-by-depth %d -> facevert %d verts -> %s"
      % (len(lm), int(face.sum()), int(filt.sum()), int(fvm.sum()), os.path.basename(OUT)))
