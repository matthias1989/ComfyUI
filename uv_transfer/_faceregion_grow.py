"""LOCAL prototype: MediaPipe core (red) + geometric forehead-grow (blue) overlaid on the
real photo. Grow = flood-fill UP from the MediaPipe core through front-facing skin faces,
bounded at the chin (no neck creep) and stopped where the surface stops facing the camera
(crown / hair-behind). Lets us judge whether geometry can fill the forehead MediaPipe misses."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr, cv2
from collections import deque
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
FACE_THR = float(os.environ.get("FACE_THR", "0.20"))   # min front-facing dot to keep growing
RAW = os.environ.get("INPUT_IMG", r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png")
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
core = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
core = np.zeros(len(m.faces), bool) if core is None else np.asarray(core, bool)
# --- geometric forehead-grow on the mesh ---
fn = np.asarray(m.face_normals)                      # outward face normals
cen = np.asarray(m.triangles_center)                 # face centroids (height = cen[:,1])
UP_CAP = float(os.environ.get("UP_CAP", "0.55"))     # exclude up-pointing crown faces
LOWER  = float(os.environ.get("LOWER", "0.25"))       # how far below core-top the grow may start (frac of core height)
front_dir = fn[core].mean(0); front_dir /= (np.linalg.norm(front_dir)+1e-9)
facing = fn @ front_dir                               # high = same way the face front looks
proj = cen @ front_dir
ctop = cen[core, 1].max(); cbot = cen[core, 1].min(); chgt = ctop - cbot
xlo, xhi = cen[core, 0].min(), cen[core, 0].max()
ylow = ctop - LOWER*chgt                              # grow UP from the core top only -> never the side locks
accept = ((facing > FACE_THR) & (cen[:, 1] >= ylow)
          & (cen[:, 0] >= xlo) & (cen[:, 0] <= xhi)
          & (fn[:, 1] < UP_CAP))
# adjacency list
adj = [[] for _ in range(len(m.faces))]
for a, b in m.face_adjacency:
    adj[a].append(b); adj[b].append(a)
grown = core.copy(); dq = deque(np.where(core)[0].tolist())
while dq:
    cf = dq.popleft()
    for nb in adj[cf]:
        if not grown[nb] and accept[nb]:
            grown[nb] = True; dq.append(nb)
added = grown & ~core
print(f"core={int(core.sum())}  grown={int(grown.sum())}  added forehead={int(added.sum())}  (FACE_THR={FACE_THR})")
# --- rasterize mesh in photo alignment, overlay core(red) + added(blue) ---
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
eligible = accept & ~grown      # passed the gate but the flood-fill never reached it
disp = (front*255).clip(0,255).astype(np.uint8).copy()
valid = fid >= 0
if os.environ.get("FINAL"):     # combined chart as one region
    gm = np.zeros((IH, IW), bool); gm[valid] = grown[fid[valid]]
    disp[gm] = (0.45*disp[gm] + 0.55*np.array([255, 40, 40])).astype(np.uint8)
else:
    rc = np.zeros((IH, IW), bool); rc[valid] = core[fid[valid]]
    ad = np.zeros((IH, IW), bool); ad[valid] = added[fid[valid]]
    el = np.zeros((IH, IW), bool); el[valid] = eligible[fid[valid]]
    disp[el] = (0.45*disp[el] + 0.55*np.array([40, 220, 40])).astype(np.uint8)    # eligible-unreached = green
    disp[rc] = (0.45*disp[rc] + 0.55*np.array([255, 40, 40])).astype(np.uint8)    # core = red
    disp[ad] = (0.40*disp[ad] + 0.60*np.array([40, 90, 255])).astype(np.uint8)    # grown = blue
print(f"eligible-unreached={int(eligible.sum())}")
Image.fromarray(disp).crop((int(IW*0.30), 0, int(IW*0.70), int(IH*0.34))).resize((640, 544), Image.LANCZOS).save(os.path.join(HERE, "_faceregion_grow.png"))
print("saved _faceregion_grow.png")
