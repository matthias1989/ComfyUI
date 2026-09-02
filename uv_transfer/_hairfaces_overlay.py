"""LOCAL: overlay the pipeline's is_hair flag (orange) + MediaPipe core (red) on the real photo.
Shows whether is_hair leaves the FOREHEAD skin un-flagged (so 'front-facing & not-hair' can fill
the forehead and stop at the true hairline), or whether is_hair eats the forehead (then it's useless there)."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
RAW = os.environ.get("INPUT_IMG", r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png")
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
hair = np.load(os.path.join(HERE, "last_seams_hairfaces.npy")).astype(bool)
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
core = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
core = np.zeros(len(m.faces), bool) if core is None else np.asarray(core, bool)
print(f"hair={int(hair.sum())}  core={int(core.sum())}")
s = 1.15; uc = v[:,0]/(s/2); vc = v[:,1]/(s/2)
co = np.argwhere(fg); yy0, yy1 = int(co[:,0].min()), int(co[:,0].max())
ich = float(yy1-yy0+1); icx = float(co[:,1].mean()); icy = float(yy0+yy1)/2
mcx = float((uc.min()+uc.max())*0.5); mcy = float((vc.min()+vc.max())*0.5); ppc = ich/float((vc.max()-vc.min()).clamp(min=1e-6))
xpix = icx + (uc-mcx)*ppc; ypix = icy - (vc-mcy)*ppc
xndc = (xpix+0.5)/IW*2-1; yndc = -(((ypix+0.5)/IH)*2-1)
d = v[:,2]; dmin = d.min(); dsp = (d.max()-dmin).clamp(min=1e-6); zndc = 1-2*(d-dmin)/dsp
clip = torch.stack([xndc, yndc, zndc, torch.ones_like(xndc)], -1).unsqueeze(0)
rast, _ = dr.rasterize(ctx, clip, f, resolution=[IH, IW])
fid = (rast[0,:,:,3].long()-1).cpu().numpy()[::-1].copy()
disp = (front*255).clip(0,255).astype(np.uint8).copy()
valid = fid >= 0
hm = np.zeros((IH, IW), bool); hm[valid] = hair[fid[valid]]
cm = np.zeros((IH, IW), bool); cm[valid] = core[fid[valid]]
disp[hm] = (0.40*disp[hm] + 0.60*np.array([255, 150, 20])).astype(np.uint8)   # is_hair = orange
disp[cm] = (0.45*disp[cm] + 0.55*np.array([255, 40, 40])).astype(np.uint8)    # core   = red
Image.fromarray(disp).crop((int(IW*0.26), 0, int(IW*0.74), int(IH*0.38))).resize((680, 540), Image.LANCZOS).save(os.path.join(HERE, "_hairfaces_overlay.png"))
print("saved _hairfaces_overlay.png")
