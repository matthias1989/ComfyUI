"""LOCAL read-only: visualize what MediaPipe sees for the face region — the relief it runs on,
the 478 landmarks (forehead ones highlighted), and the convex hull. Shows whether the region
stops at the eyes because MediaPipe under-detects the forehead, or the hull is fine."""
import os, sys, numpy as np, torch, cv2
from PIL import Image
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import _render_relief, _detect_478, _head_crop_box
import trimesh, nvdiffrast.torch as dr
from rembg import remove
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
mesh = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
v = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values+v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(mesh.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(mesh.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1, keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
box = _head_crop_box(fg); print("head_crop_box:", box)
relief = _render_relief(v, f, vn, 1.15, ctx, fg, IH, IW)
relief_np = relief if (isinstance(relief, np.ndarray) and relief.dtype == np.uint8) else (np.clip(np.asarray(relief), 0, 1)*255).astype(np.uint8)
if relief_np.ndim == 2: relief_np = np.stack([relief_np]*3, -1)
lm = _detect_478(relief, box)
vis = relief_np.copy()
if lm is None:
    print("[diag] landmark detection FAILED")
else:
    lm = lm.astype(np.int32)
    print(f"[diag] {len(lm)} landmarks; y-range=[{lm[:,1].min()},{lm[:,1].max()}]  head box y=[{box[1]},{box[3]}]")
    for p in lm: cv2.circle(vis, tuple(p), 1, (0, 255, 0), -1)
    hull = cv2.convexHull(lm); cv2.polylines(vis, [hull], True, (0, 200, 255), 2)
    for i in [10, 109, 338, 67, 297, 103, 332]:   # forehead/hairline landmarks
        if i < len(lm): cv2.circle(vis, tuple(lm[i]), 3, (255, 0, 0), -1)
    print(f"[diag] forehead landmark 10 at y={lm[10,1]}  (top of head box y={box[1]}); eyes ~ landmark 33 y={lm[33,1]}")
Image.fromarray(vis).save(os.path.join(HERE, "_faceregion_diag.png"))
print("saved _faceregion_diag.png  (green=478 lms, red=forehead lms, cyan=hull)")
