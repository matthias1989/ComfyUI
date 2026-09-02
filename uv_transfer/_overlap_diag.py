"""LOCAL: find faces that are doubled-up (close to another face with opposing normal = overlapping
shell / z-fight) and render them RED on the body, front + side, so we can see the intruding layer."""
import os, sys, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from scipy.spatial import cKDTree
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.path.join(HERE, "_bodydiag.obj"), force='mesh')
V = np.asarray(m.vertices); F = np.asarray(m.faces)
tri = V[F]; cen = tri.mean(1); fn = np.asarray(m.face_normals)
e1 = tri[:,1]-tri[:,0]; e2 = tri[:,2]-tri[:,0]; area = 0.5*np.linalg.norm(np.cross(e1,e2),axis=1)
rad = float(np.median(area)**0.5 * 0.6)
tree = cKDTree(cen); pairs = tree.query_pairs(r=rad, output_type='ndarray')
dots = (fn[pairs[:,0]]*fn[pairs[:,1]]).sum(1)
opp = pairs[dots < -0.3]                       # close + opposing normal = doubled shell
doubled = np.zeros(len(F), bool); doubled[opp[:,0]] = True; doubled[opp[:,1]] = True
print(f"radius={rad:.5f}  doubled faces={int(doubled.sum())}  ({100*doubled.mean():.1f}% of mesh)")
# which side is the intruder: of each opposed pair, the back-facing one (fn.z<0) on the FRONT body
v = torch.from_numpy(V).float().cuda()
c0 = (v.min(0).values+v.max(0).values)*0.5; v = v-c0; r = torch.sqrt((v**2).sum(-1).max()); v = v/(r*2)
ft = torch.from_numpy(F.astype(np.int32)).cuda(); ctx = dr.RasterizeCudaContext()
xx,yy,dd = v[:,0],v[:,1],v[:,2]; R=1000
def render(name, X, Z):
    zz = 1-2*(Z-Z.min())/(Z.max()-Z.min())
    cam = torch.stack([X*2, yy*2, zz, torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
    rr,_ = dr.rasterize(ctx, cam, ft, resolution=[R,R]); fid=(rr[0,:,:,3].long()-1).cpu().numpy(); hit=fid>=0
    img=np.full((R,R,3),32,np.uint8); img[hit]=[110,110,120]
    dm=np.zeros((R,R),bool); val=hit; dm[val]=doubled[fid[val]]; img[dm]=[235,40,40]
    Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,f"_overlap_{name}.png"))
    print(f"saved _overlap_{name}.png")
render("front", xx, dd)
render("side", dd, -xx)   # 90deg: view from character's side
