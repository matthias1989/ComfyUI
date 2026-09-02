"""LOCAL prototype: grow the face chart from the MediaPipe core through faces whose PROJECTED
front-photo color stays near skin tone. Geometry can't separate blond face-framing locks from
skin (one smooth front-facing surface); color can. core=red, grown forehead=blue."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from collections import deque
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
TOL = float(os.environ.get("TOL", "0.14"))           # max color distance (0..~1.7) from skin median
FACE_THR = float(os.environ.get("FACE_THR", "0.05"))  # min front-facing dot (loose; color does the work)
RAW = os.environ.get("INPUT_IMG", r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png")
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
nF = len(m.faces)
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
core = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
core = np.zeros(nF, bool) if core is None else np.asarray(core, bool)
# rasterize mesh in photo alignment -> per-pixel face id, then sample per-face mean photo color
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
valid = fid >= 0
fcol = np.zeros((nF, 3), np.float64); fcnt = np.zeros(nF, np.float64)
np.add.at(fcol, fid[valid], front[valid]); np.add.at(fcnt, fid[valid], 1.0)
seen = fcnt > 0; fcol[seen] /= fcnt[seen, None]
fn = np.asarray(m.face_normals); cen = np.asarray(m.triangles_center)
fdir = fn[core].mean(0); fdir /= (np.linalg.norm(fdir)+1e-9); facing = fn @ fdir
chin_y = cen[core, 1].min()
skin = np.median(fcol[core & seen], axis=0)
cdist = np.linalg.norm(fcol - skin, axis=1)
accept = seen & (facing > FACE_THR) & (cen[:,1] >= chin_y) & (cdist < TOL)
print(f"skin median rgb = {np.round(skin,3)}   accept(before flood) = {int(accept.sum())}")
adj = [[] for _ in range(nF)]
for a, b in m.face_adjacency:
    adj[a].append(b); adj[b].append(a)
grown = core.copy(); dq = deque(np.where(core)[0].tolist())
while dq:
    cf = dq.popleft()
    for nb in adj[cf]:
        if not grown[nb] and accept[nb]:
            grown[nb] = True; dq.append(nb)
added = grown & ~core
print(f"core={int(core.sum())}  grown={int(grown.sum())}  added={int(added.sum())}  (TOL={TOL})")
disp = (front*255).clip(0,255).astype(np.uint8).copy()
rc = np.zeros((IH, IW), bool); rc[valid] = core[fid[valid]]
ad = np.zeros((IH, IW), bool); ad[valid] = added[fid[valid]]
disp[rc] = (0.45*disp[rc] + 0.55*np.array([255, 40, 40])).astype(np.uint8)
disp[ad] = (0.40*disp[ad] + 0.60*np.array([40, 90, 255])).astype(np.uint8)
Image.fromarray(disp).crop((int(IW*0.30), 0, int(IW*0.70), int(IH*0.34))).resize((640, 544), Image.LANCZOS).save(os.path.join(HERE, "_faceregion_color.png"))
print("saved _faceregion_color.png")
