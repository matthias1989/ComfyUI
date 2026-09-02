"""DEV facemask via MediaPipe on the REAL front photo (proven), mapped to the mesh through the same
front fg-alignment the texturing uses. Out: _dev_faceregion.npy. Run with python_embeded."""
import os, sys, numpy as np, torch, nvdiffrast.torch as dr, trimesh, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
TR = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2"
sys.path.insert(0, TR)
from projection.landmark_warp import _render_relief, _get_landmarker
import mediapipe as mp
PHOTO = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
IH = IW = 896

rgb = np.array(Image.open(PHOTO).convert("RGB"))                # photo (896)
# foreground = saturated (skin/character) vs gray gradient bg
sat = rgb.max(2).astype(int) - rgb.min(2).astype(int)
fg = sat > 20
rows = np.where(fg.any(1))[0]; cols = np.where(fg.any(0))[0]
ft, fb = int(rows[0]), int(rows[-1]); fh = fb - ft + 1
# head COLUMNS from the upper band only (crown/ears -> above the T-pose arms); ROWS crown->~chin.
hb = ft + int(0.15 * fh)
ub = fg[ft: ft + max(8, int(0.09 * fh))]                # upper head/ears -> no arms
hcols = np.where(ub.any(0))[0]; pad = int(0.04 * IW)
box = (max(0, int(hcols[0]) - pad), min(IW, int(hcols[-1]) + pad), ft, hb)
print(f"[photo-face] head box={box}")
# MediaPipe on the upscaled head crop
x0, x1, y0, y1 = box; sub = rgb[y0:y1, x0:x1]; up = 5
s2 = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up), interpolation=cv2.INTER_CUBIC)
res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(s2)))
if not res.face_landmarks:
    print("[photo-face] no landmarks"); sys.exit(1)
H2, W2 = s2.shape[:2]
lm = np.array([[x0 + p.x * W2 / up, y0 + p.y * H2 / up] for p in res.face_landmarks[0]])
print(f"[photo-face] {len(lm)} landmarks; face hull bbox x[{lm[:,0].min():.0f},{lm[:,0].max():.0f}] y[{lm[:,1].min():.0f},{lm[:,1].max():.0f}]")
# CENTRAL FEATURES only: landmarks within the outer-eye-corner X span (33=L,263=R). Covers
# eyes/brows/nose/mouth; excludes temple/jaw -> the side hairline is never filtered ("sharper centre").
import os as _os
xl=float(lm[33,0]); xr=float(lm[263,0]); mrg=float(_os.environ.get('GS_FACE_MRG','0.06'))*(xr-xl)
selc=(lm[:,0]>=xl-mrg)&(lm[:,0]<=xr+mrg)
poly=cv2.convexHull(lm[selc].astype(np.int32))
mask = np.zeros((IH, IW), np.uint8)
cv2.fillConvexPoly(mask, poly, 255)
print('[photo-face] central-feature mask: %d landmarks in eye-span'%int(selc.sum()))

# rasterize the mesh ALIGNED to the photo fg (same body-anchored ortho the texturing uses) -> fid
d = np.load("_gs_facemesh.npz"); v0 = d["verts"].astype(np.float32)[:, [0, 2, 1]].copy(); tris = d["tris"].astype(np.int64)
ctx = dr.RasterizeCudaContext(); f = torch.from_numpy(tris).int().cuda()
best = None
for sgn in (-1.0, 1.0):
    v = v0.copy(); v[:, 2] *= sgn
    vt = torch.from_numpy(v).float().cuda()
    for _ in range(2):
        c = (vt.min(0).values + vt.max(0).values) * 0.5; vt = vt - c
        r = torch.sqrt((vt**2).sum(-1).max()).clamp(min=1e-6); vt = vt * (1.15 / (r * 2))
    s = 1.15; uc = vt[:, 0] / (s / 2); vc = vt[:, 1] / (s / 2)
    co = np.argwhere(fg); iy0, iy1 = int(co[:, 0].min()), int(co[:, 0].max())
    ich = float(iy1 - iy0 + 1); icx = float(co[:, 1].mean()); icy = float(iy0 + iy1) / 2.0
    mcx = float((uc.min() + uc.max()) * 0.5); mcy = float((vc.min() + vc.max()) * 0.5)
    ppc = ich / float((vc.max() - vc.min()).clamp(min=1e-6))
    xpix = icx + (uc - mcx) * ppc; ypix = icy - (vc - mcy) * ppc
    xndc = (xpix + 0.5) / IW * 2 - 1; yndc = -(((ypix + 0.5) / IH) * 2 - 1)
    dd = vt[:, 2]; zndc = 1 - 2 * (dd - dd.min()) / (dd.max() - dd.min() + 1e-9)
    clip = torch.stack([xndc, yndc, zndc, torch.ones_like(xndc)], -1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, clip.contiguous(), f, resolution=[IH, IW])
    fid = (rast[0, :, :, 3].long() - 1).cpu().numpy()[::-1].copy()
    inside = (mask > 0) & (fid >= 0); ids = np.unique(fid[inside])
    sel = np.zeros(int(tris.shape[0]), bool); sel[ids[ids >= 0]] = True
    n = int(sel.sum()); print(f"[photo-face] sgn={sgn:+.0f} -> {n} face faces")
    if n > 200: best = sel; break
if best is None: print("[photo-face] mapping failed"); sys.exit(1)
np.save("_dev_faceregion.npy", best)
print(f"[photo-face] saved {int(best.sum())} face faces -> _dev_faceregion.npy")
