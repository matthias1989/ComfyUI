"""LOCAL: per-vertex LOCAL CONVEXITY = (vertex - local_neighbour_centroid) . normal, at the nipple
scale. Sharp bumps (nipples) read high+, broad bulges (belly/sternum) ~0, dimples (navel) negative.
Render heatmap on the mesh front: if the nipples are the brightest spots, this is the auto-detector."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from scipy.spatial import cKDTree
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
R = float(os.environ.get("R", "0.020"))     # neighbourhood radius (nipple scale, normalized)
m = trimesh.load(os.path.join(HERE, "_bodyquad.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c = (v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
V = v.numpy(); N = np.asarray(m.vertex_normals); F = np.asarray(m.faces)
conv = np.zeros(len(V), np.float32)
front = np.where(N[:,2] > 0.1)[0]                       # front-facing only (where it matters + faster)
tree = cKDTree(V[front])
nbrs = tree.query_ball_point(V[front], R, workers=-1)
for k, j in enumerate(front):
    nb = nbrs[k]
    if len(nb) < 4: continue
    cen = V[front[nb]].mean(0)
    conv[j] = float(np.dot(V[j]-cen, N[j]))             # +out (bump), -in (dimple)
print(f"convexity: p50={np.percentile(conv[front],50):.5f} p99={np.percentile(conv[front],99):.5f} max={conv[front].max():.5f}")
# render front heatmap
vt = torch.from_numpy(V).float().cuda(); ft = torch.from_numpy(F.astype(np.int32)).cuda(); ctx = dr.RasterizeCudaContext()
xx,yy,dd = vt[:,0],vt[:,1],vt[:,2]; Rres=1100; zz=1-2*(dd-dd.min())/(dd.max()-dd.min())
clip = torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rast,_ = dr.rasterize(ctx,clip,ft,resolution=[Rres,Rres])
cv = torch.from_numpy(conv).float().cuda()[:,None]
ci,_ = dr.interpolate(cv[None], rast, ft); ci = ci[0,:,:,0].cpu().numpy()[::-1].copy()
hit = (rast[0,:,:,3]>0).cpu().numpy()[::-1].copy()
hi = np.percentile(conv[front], 99.5)
t = np.clip(ci/max(hi,1e-6), 0, 1)
img = np.full((Rres,Rres,3), 25, np.uint8)
img[hit] = np.stack([ (40+215*t)[hit], (40+120*(1-np.abs(2*t-1)))[hit], (200*(1-t))[hit] ],-1).clip(0,255).astype(np.uint8)  # blue=flat -> red=sharp bump
full = Image.fromarray(img)
full.save(os.path.join(HERE,"_curv_front.png"))
full.crop((int(Rres*0.30),int(Rres*0.14),int(Rres*0.70),int(Rres*0.46))).resize((760,600),Image.LANCZOS).save(os.path.join(HERE,"_curv_chest.png"))
print(f"saved _curv_front.png + _curv_chest.png  (R={R}; red=sharp bump, blue=flat/dimple)")
