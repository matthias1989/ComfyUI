"""SELF-CONTAINED facemask on the FROZEN bake mesh: MediaPipe on the real front photo (proven),
projected onto _frozen/gb_mesh.npz. Selection is by FACE-CENTROID-in-hull (density-independent) rather
than per-pixel rasterization, which undersamples dense (1M+ face) meshes and yields a scattered region.
Back-of-head faces that also fall inside the 2D hull are removed downstream by the front-facing filter.
Inputs: _frozen/gb_mesh.npz (co,fv) + _frozen/front.png.  Out: _frozen/faceregion.npy (per gb face).
Run with python_embeded. Meant to live in gen_seams (runs on the bake's own mesh)."""
import os, sys, numpy as np, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
TR = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2"
sys.path.insert(0, TR)
from projection.landmark_warp import _get_landmarker
import mediapipe as mp
PHOTO = os.path.join(HERE, "_frozen", "front.png")
IH = IW = 896

rgb = np.array(Image.open(PHOTO).convert("RGB"))
sat = rgb.max(2).astype(int) - rgb.min(2).astype(int)
fg = sat > 20
rows = np.where(fg.any(1))[0]; cols = np.where(fg.any(0))[0]
ft, fb = int(rows[0]), int(rows[-1]); fh = fb - ft + 1
hb = ft + int(0.15 * fh)
ub = fg[ft: ft + max(8, int(0.09 * fh))]
hcols = np.where(ub.any(0))[0]; pad = int(0.04 * IW)
box = (max(0, int(hcols[0]) - pad), min(IW, int(hcols[-1]) + pad), ft, hb)
print(f"[fz-face] head box={box}")
x0, x1, y0, y1 = box; sub = rgb[y0:y1, x0:x1]; up = 5
s2 = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up), interpolation=cv2.INTER_CUBIC)
res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(s2)))
if not res.face_landmarks:
    print("[fz-face] no landmarks"); sys.exit(1)
H2, W2 = s2.shape[:2]
lm = np.array([[x0 + p.x * W2 / up, y0 + p.y * H2 / up] for p in res.face_landmarks[0]])
print(f"[fz-face] {len(lm)} landmarks")
# CENTRAL features only: within the outer-eye-corner X span (33=L,263=R) -> sides never filtered
# central features = landmarks within the outer-eye-corner X span (33=L, 263=R). Anatomical bounds, no margin constant.
xl = float(lm[33, 0]); xr = float(lm[263, 0])
selc = (lm[:, 0] >= xl) & (lm[:, 0] <= xr)
poly = cv2.convexHull(lm[selc].astype(np.int32))
mask = np.zeros((IH, IW), np.uint8); cv2.fillConvexPoly(mask, poly, 255)
print('[fz-face] central-feature mask: %d landmarks in eye-span' % int(selc.sum()))

# align the FROZEN gb mesh (width->x, height->y) to the photo fg, then keep faces whose CENTROID
# projects inside the central hull. No raster -> density-independent. (Depth/turn ignored here; the
# back-of-head faces that also project into the hull are dropped by the front-facing filter step.)
d = np.load(os.path.join(HERE, "_frozen", "gb_mesh.npz"))
v0 = d["co"].astype(np.float64)[:, [0, 2, 1]].copy()   # [width,depth,height] -> [x,height,depth]
tris = d["fv"].astype(np.int64)
vt = v0.copy()
for _ in range(2):
    c = (vt.min(0) + vt.max(0)) * 0.5; vt = vt - c
    r = np.sqrt((vt**2).sum(1).max()); vt = vt * (1.15 / (r * 2 + 1e-9))
uc = vt[:, 0] / (1.15 / 2); vc = vt[:, 1] / (1.15 / 2)
fgc = np.argwhere(fg); iy0, iy1 = int(fgc[:, 0].min()), int(fgc[:, 0].max())
ich = float(iy1 - iy0 + 1); icx = float(fgc[:, 1].mean()); icy = float(iy0 + iy1) / 2.0
ppc = ich / float(vc.max() - vc.min() + 1e-9)
mcx = float((uc.min() + uc.max()) * 0.5); mcy = float((vc.min() + vc.max()) * 0.5)
xpix = icx + (uc - mcx) * ppc; ypix = icy - (vc - mcy) * ppc      # image space (y-down)
cxp = xpix[tris].mean(1); cyp = ypix[tris].mean(1)
ix = np.round(cxp).astype(np.int64); iy = np.round(cyp).astype(np.int64)
valid = (ix >= 0) & (ix < IW) & (iy >= 0) & (iy < IH)
sel = np.zeros(int(tris.shape[0]), bool)
sel[valid] = mask[iy[valid], ix[valid]] > 0
np.save(os.path.join(HERE, "_frozen", "faceregion.npy"), sel)
print(f"[fz-face] centroid-in-hull -> {int(sel.sum())} face faces (front+back; back removed by front filter)")
