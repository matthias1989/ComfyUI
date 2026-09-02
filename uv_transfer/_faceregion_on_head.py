"""LOCAL read-only: compute the MediaPipe face region (mesh-only) on last_seams.obj and
render it on the 3D head (red = region) so I can see whether its boundary cuts the lips /
grabs the ear, and how much of the face it misses."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
v0 = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
v = v0.clone()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v - c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
s = 1.15; uc = v[:,0]/(s/2); vc = v[:,1]/(s/2); IH=IW=640
dd = v[:,2]; zz = 1-2*(dd-dd.min())/(dd.max()-dd.min())
clip = torch.stack([uc,vc,zz,torch.ones_like(uc)],-1).unsqueeze(0).contiguous()
rr,_ = dr.rasterize(ctx,clip,f,resolution=[IH,IW]); hit=(rr[0,:,:,3]>0).cpu().numpy()
front = np.zeros((IH,IW,3),np.float32); front[hit]=1.0
sel = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
sel = np.zeros(len(m.faces),bool) if sel is None else np.asarray(sel,bool)
print(f"MediaPipe region = {int(sel.sum())} faces")
# render head 3D (use original verts for the upright view)
v = v0.clone(); c0=(v.min(0).values+v.max(0).values)*0.5; v=v-c0; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz2=1-2*(d-d.min())/(d.max()-d.min()); R=1000
cam=torch.stack([xx*2,yy*2,zz2,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rr2,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fid=(rr2[0,:,:,3].long()-1).cpu().numpy(); hm=fid>=0
img=np.full((R,R,3),30,np.uint8); img[hm]=[120,120,120]
isreg=np.zeros((R,R),bool); val=hm&(fid<len(sel)); isreg[val]=sel[fid[val]]
img[isreg]=[220,40,40]
full=Image.fromarray(np.flipud(img).copy())
full.crop((int(R*0.36),int(R*0.02),int(R*0.64),int(R*0.22))).resize((720,560),Image.LANCZOS).save(os.path.join(HERE,"_faceregion_head.png"))
print("saved _faceregion_head.png")
