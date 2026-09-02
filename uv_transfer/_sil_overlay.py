"""LOCAL, theory-free: does the MESH silhouette match the PHOTO silhouette for a character?
Height-normalize + centroid-align both (the way the projector aligns), overlay:
  red = mesh only, green = photo only, yellow = both. Big red/green = mesh<->photo mismatch."""
import os, sys, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
MESH = os.environ["MESH"]; PHOTO = os.environ["PHOTO"]; OUT = os.environ.get("OUT", "_sil.png")
R = 700
# --- mesh front (x-y) silhouette ---
m = trimesh.load(MESH, force='mesh')
v = np.asarray(m.vertices).astype(np.float32)
# Y-up assumed (height=y, width=x, depth=z) — true for last_seams.obj and the rigged glb
vt = torch.from_numpy(v).cuda()
x = vt[:,0]; y = vt[:,1]
# normalize by HEIGHT, center on centroid-x and bbox-mid-y (projector's method)
hy = (y.max()-y.min())
cx = 0.5*(x.min()+x.max()); cym = 0.5*(y.min()+y.max())
nx = (x-cx)/hy; ny = (y-cym)/hy           # height -> 1.0 span, centered
clip = torch.stack([nx*1.6, ny*1.6, torch.zeros_like(nx), torch.ones_like(nx)],-1).unsqueeze(0).contiguous()
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
ctx = dr.RasterizeCudaContext()
rr,_ = dr.rasterize(ctx, clip, f, resolution=[R,R])
mesh_sil = (rr[0,:,:,3]>0).cpu().numpy()
mesh_sil = np.flipud(mesh_sil)            # clip y-up -> image y-down
# --- photo silhouette (rembg alpha) ---
inp = remove(Image.open(PHOTO).convert("RGB")); al = np.array(inp)[:,:,3] > 128
ys,xs = np.where(al)
y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
ph = (y1-y0); pcx = 0.5*(x0+x1); pcym = 0.5*(y0+y1)
# map photo into the SAME normalized canvas (height->1, centered), rasterize to RxR
pys,pxs = np.where(al)
nyp = (pys-pcym)/ph; nxp = (pxs-pcx)/ph
col = np.clip(((nxp*1.6)*0.5+0.5)*R,0,R-1).astype(int)
row = np.clip(((nyp*1.6)*0.5+0.5)*R,0,R-1).astype(int)
photo_sil = np.zeros((R,R),bool); photo_sil[row,col]=True
# fill small gaps in photo silhouette (scatter leaves holes)
import scipy.ndimage as ndi
photo_sil = ndi.binary_closing(photo_sil, iterations=3)
# --- overlay ---
img = np.full((R,R,3),30,np.uint8)
img[mesh_sil & ~photo_sil] = [230,30,30]   # mesh only
img[photo_sil & ~mesh_sil] = [30,200,30]   # photo only
img[mesh_sil & photo_sil]  = [220,220,40]  # both
Image.fromarray(img).save(os.path.join(HERE,OUT))
inter=(mesh_sil&photo_sil).sum(); union=(mesh_sil|photo_sil).sum()
print(f"{OUT}: IoU={inter/max(union,1):.2f}  mesh_only={((mesh_sil&~photo_sil).sum())} photo_only={((photo_sil&~mesh_sil).sum())}")
