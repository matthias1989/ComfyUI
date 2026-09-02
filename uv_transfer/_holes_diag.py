"""LOCAL: mark faces that touch a boundary (open) edge = holes in the surface, render RED on the
body front/side. If they cluster on the torso, the 'shattered' look is the mesh being full of holes."""
import os, sys, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.path.join(HERE, "_bodydiag.obj"), force='mesh')
V = np.asarray(m.vertices); F = np.asarray(m.faces)
# edges -> count; boundary = used by exactly 1 face
ef = F[:, [0,1,1,2,2,0]].reshape(-1,2); E = np.sort(ef, axis=1)
ue, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
bnd_edge = cnt == 1
face_of_edge = np.repeat(np.arange(len(F)), 3)
touch = np.zeros(len(F), bool)
touch[face_of_edge[bnd_edge[inv]]] = True
print(f"boundary edges={int(bnd_edge.sum())}  faces touching a hole={int(touch.sum())} ({100*touch.mean():.1f}%)")
v = torch.from_numpy(V).float().cuda()
c0=(v.min(0).values+v.max(0).values)*0.5; v=v-c0; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
ft=torch.from_numpy(F.astype(np.int32)).cuda(); ctx=dr.RasterizeCudaContext()
xx,yy,dd=v[:,0],v[:,1],v[:,2]; R=1000
def render(name,X,Z):
    zz=1-2*(Z-Z.min())/(Z.max()-Z.min())
    cam=torch.stack([X*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
    rr,_=dr.rasterize(ctx,cam,ft,resolution=[R,R]); fid=(rr[0,:,:,3].long()-1).cpu().numpy(); hit=fid>=0
    img=np.full((R,R,3),32,np.uint8); img[hit]=[110,110,120]
    hm=np.zeros((R,R),bool); hm[hit]=touch[fid[hit]]; img[hm]=[235,40,40]
    Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,f"_holes_{name}.png")); print(f"saved _holes_{name}.png")
render("front",xx,dd); render("side",dd,-xx)
