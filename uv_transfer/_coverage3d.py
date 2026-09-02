"""LOCAL: sample the baked atlas alpha at each face's UV, render painted(skin)/unpainted(red) on the
3D body front + head, to see the REAL coverage holes and whether body vs face is the uncovered one."""
import os, sys, glob, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
ATL = sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_*.png"), key=os.path.getmtime)[-1]
print("atlas:", os.path.basename(ATL))
a = np.asarray(Image.open(ATL).convert("RGBA")); AH, AW = a.shape[:2]
alpha = a[..., 3]
m = trimesh.load(os.path.join(HERE, "last_seams.glb"), force='mesh')
V = np.asarray(m.vertices); F = np.asarray(m.faces); uv = np.asarray(m.visual.uv)
# sample atlas alpha at each face's 3 UV verts; painted if any vert covered
fuv = uv[F]                                  # (nf,3,2)
painted = np.zeros(len(F), bool)
for k in range(3):
    px = np.clip((fuv[:, k, 0]*AW).astype(int), 0, AW-1)
    py = np.clip(((1-fuv[:, k, 1])*AH).astype(int), 0, AH-1)
    painted |= alpha[py, px] > 10
print(f"faces painted: {100*painted.mean():.1f}%")
# region split
c=(V.min(0)+V.max(0))*0.5; ext=(V.max(0)-V.min(0)).max(); Pc=((V[F].mean(1))-c)/ext
face_reg=(Pc[:,1]>0.30)&(Pc[:,2]>0.0)&(np.abs(Pc[:,0])<0.10)
body_reg=(Pc[:,1]<0.25)&(Pc[:,1]>-0.30)&(Pc[:,2]>0.0)
print(f"  FACE region painted: {100*painted[face_reg].mean():.1f}%  ({int(face_reg.sum())} faces)")
print(f"  BODY torso painted:  {100*painted[body_reg].mean():.1f}%  ({int(body_reg.sum())} faces)")
# render
v=torch.from_numpy(V).float().cuda(); c0=(v.min(0).values+v.max(0).values)*0.5; v=v-c0; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
ft=torch.from_numpy(F.astype(np.int32)).cuda(); ctx=dr.RasterizeCudaContext()
xx,yy,dd=v[:,0],v[:,1],v[:,2]; R=1000
zz=1-2*(dd-dd.min())/(dd.max()-dd.min())
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rr,_=dr.rasterize(ctx,cam,ft,resolution=[R,R]); fid=(rr[0,:,:,3].long()-1).cpu().numpy(); hit=fid>=0
img=np.full((R,R,3),28,np.uint8)
pm=np.zeros((R,R),bool); pm[hit]=painted[fid[hit]]
img[hit&~pm]=[235,40,40]            # unpainted = red
img[hit&pm]=[210,180,150]           # painted = skin
full=Image.fromarray(np.flipud(img).copy())
full.save(os.path.join(HERE,"_coverage3d_front.png"))
full.crop((int(R*0.36),int(R*0.02),int(R*0.64),int(R*0.22))).resize((720,560),Image.LANCZOS).save(os.path.join(HERE,"_coverage3d_head.png"))
print("saved _coverage3d_front.png + _coverage3d_head.png")
