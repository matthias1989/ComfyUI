"""Diagnose back-bleed: render the BACK of the mesh with the latest albedo, and measure
how many AWAY-facing faces are painted + whether they share UV texels with FRONT faces
(=> remesh UV overlap) vs being painted directly (=> projection mask bug)."""
import os,sys,glob,numpy as np
from PIL import Image
import torch,nvdiffrast.torch as dr,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); R=900; TS=2048
OBJ=os.path.join(HERE,"last_seams.obj")
ALB=sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00197_\run_*\character_posed_00197__albedo.png"))[-1]
print("albedo:",os.path.basename(os.path.dirname(ALB)))
alb=np.asarray(Image.open(ALB).convert("RGB")); AH,AW=alb.shape[:2]
m=trimesh.load(OBJ,force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()); v=v*(1.0/(r*2))
f=torch.from_numpy(np.asarray(m.faces)).int().cuda()
uvt=torch.from_numpy(np.asarray(m.visual.uv)).float().cuda(); ctx=dr.RasterizeCudaContext()
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
# BACK camera: look from -Z (mirror x so it reads naturally), back faces become frontmost
xx=-v[:,0]; yy=v[:,1]; d=-v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min())
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rast,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fm=(rast[0,:,:,3]>0).cpu().numpy()
uv=dr.interpolate(uvt.unsqueeze(0),rast,f)[0][0].cpu().numpy()
col=np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); rw=np.clip((1-uv[:,:,1])*AH,0,AH-1).astype(int)
out=alb[rw,col].copy(); out[~fm]=0
Image.fromarray(np.flipud(out).copy()).save(os.path.join(HERE,"_back_view.png"))
# per-face: away-facing (nz<-0.3) painted in albedo?
Fc=np.asarray(m.faces); UV=np.asarray(m.visual.uv); VN=np.asarray(m.vertex_normals)
cuv=UV[Fc].mean(1); fnz=VN[Fc].mean(1)[:,2]
px=np.clip((cuv[:,0]*AW).astype(int),0,AW-1); py=np.clip(((1-cuv[:,1])*AH).astype(int),0,AH-1)
painted=alb[py,px].sum(1)>24
back=fnz<-0.3; front=fnz>0.3
print(f"faces={len(Fc)}  away-facing(nz<-0.3)={int(back.sum())}  of those PAINTED={int((back&painted).sum())} ({100*(back&painted).mean()/max(1e-9,back.mean()):.0f}% of back)")
print(f"  front-facing painted={100*(front&painted).mean()/max(1e-9,front.mean()):.0f}%")
# UV overlap test: do back-painted faces land on the SAME atlas texels as front faces?
# rasterize front-only and back-only UV islands; overlap => shared texels => bleed via UV
def uvmask(sel):
    fsel=torch.from_numpy(Fc[sel].astype(np.int32)).cuda()
    uvc=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
    rr,_=dr.rasterize(ctx,uvc,fsel,resolution=[TS,TS]); return (rr[0,:,:,3]>0).cpu().numpy()
fmask=uvmask(front); bmask=uvmask(back)
ov=fmask&bmask
print(f"  UV texels used by FRONT={int(fmask.sum())}  by BACK={int(bmask.sum())}  OVERLAP(shared)={int(ov.sum())} "
      f"({100*ov.sum()/max(1,bmask.sum()):.0f}% of back texels shared with front)")
print("saved _back_view.png  (if it shows the front image, the back is bleeding)")
