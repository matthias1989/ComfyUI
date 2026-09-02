"""De-risk the coverage-based reclaim: build the atlas, then for each face sample the
atlas at its UV centroid (with v-flip) and classify painted(non-black)/unpainted.
Verify painted faces are FRONT-facing (so the v-flip + non-black test are correct).
Render front: painted=green, unpainted=red."""
import os,sys,importlib.util,numpy as np
from PIL import Image
import torch,torch.nn.functional as F,nvdiffrast.torch as dr,trimesh,cv2
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); TS=2048; ORTHO=1.15; DEPTH_EPS=0.02; FRONT_SKIP=0.35; R=900
RAW=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
from rembg import remove
inp=remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.0; ccy=(y0+y1)/2.0; size=int(max(x1-x0,y1-y0))
crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2)))
a2=np.array(crop).astype(np.float32)/255.0; rgbi=a2[:,:,:3]*a2[:,:,3:4]
spec=importlib.util.spec_from_file_location("fl","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-FlattenLight/__init__.py")
fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
photo=fl.FlattenLight().execute(torch.from_numpy(rgbi)[None],0.55,0.05,0.85,1.15,0.20,True)[0][0,...,:3].clamp(0,1).cpu().numpy()
IH,IW=photo.shape[:2]
mesh=trimesh.load(os.path.join(HERE,"_real_mesh.obj"),force='mesh')
V=np.asarray(mesh.vertices); Fc=np.asarray(mesh.faces); UV=np.asarray(mesh.visual.uv); VN=np.asarray(mesh.vertex_normals)
v=torch.from_numpy(V).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
f=torch.from_numpy(Fc).int().cuda()
vn=torch.from_numpy(VN.copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
uvt=torch.from_numpy(UV).float().cuda(); ctx=dr.RasterizeCudaContext()
uvclip=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
rast,_=dr.rasterize(ctx,uvclip,f,resolution=[TS,TS]); uvhit=rast[0,:,:,3]>0
tp=dr.interpolate(v.unsqueeze(0),rast,f)[0][0]; tn=dr.interpolate(vn.unsqueeze(0),rast,f)[0][0]; tn=tn/(tn.norm(dim=-1,keepdim=True)+1e-8)
s=ORTHO; uc=tp[:,:,0]/(s/2); vc=tp[:,:,1]/(s/2)
xx=v[:,0]/(s/2); yy=v[:,1]/(s/2); d=v[:,2]; dmn=d.min(); dsp=(d.max()-dmn).clamp(min=1e-6); zz=1-2*(d-dmn)/dsp
camclip=torch.stack([xx,yy,zz,torch.ones_like(xx)],-1).unsqueeze(0)
crast,_=dr.rasterize(ctx,camclip,f,resolution=[2048,2048]); chit=crast[0,:,:,3]>0
chimg=torch.from_numpy(cv2.dilate(chit.cpu().numpy().astype(np.uint8),np.ones((5,5),np.uint8),2)).float().cuda()[None,None]
cdep=dr.interpolate(v[:,2].contiguous()[None,:,None].contiguous(),crast,f)[0][0].permute(2,0,1)[None]
cocc=torch.where(chit[None,None],cdep,torch.full_like(cdep,1e9)); cocc=-F.max_pool2d(-cocc,3,1,1)
gocc=torch.stack([uc,vc],-1)[None]; shit=F.grid_sample(chimg,gocc,'bilinear','zeros',align_corners=False)[0,0]>0.05
sdep=F.grid_sample(cocc,gocc,'bilinear','border',align_corners=False)[0,0]; dm=tp[:,:,2]>=sdep-DEPTH_EPS
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=co[:,0].min(),co[:,0].max(); ich=float(fy1-fy0+1); icx=float(co[:,1].mean()); icy=float(fy0+fy1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6).item())
xp=icx+(uc-mcx)*ppc; yp=icy-(vc-mcy)*ppc; us=((xp+0.5)/IW)*2-1; vs=((yp+0.5)/IH)*2-1; inb=(us.abs()<=1)&(vs.abs()<=1)
g=torch.stack([us.clamp(-1,1),vs.clamp(-1,1)],-1)[None]
col=F.grid_sample(torch.from_numpy(photo).cuda().permute(2,0,1)[None],g,'bilinear','border',align_corners=False)[0].permute(1,2,0)
sfg=F.grid_sample(torch.from_numpy(fg.astype(np.float32)).cuda()[None,None],g,'bilinear','zeros',align_corners=False)[0,0]>0.5
cov=(uvhit&inb&shit&(dm|(tn[:,:,2]>FRONT_SKIP))&sfg&(col.mean(-1)>0.08)&(tn[:,:,2]>0.12)).cpu().numpy()
atlas=np.zeros((TS,TS,3),np.uint8); atlas[cov]=(col.cpu().numpy()[cov]*255).astype(np.uint8)
# ---- per-face coverage via UV centroid (v-flip), like the FBX export will do ----
cuv=UV[Fc].mean(1)                                   # (nface,2) centroid uv
px=np.clip((cuv[:,0]*TS).astype(int),0,TS-1)
fnz=VN[Fc].mean(1)[:,2]                               # per-face normal z (front>0)
for tag,pyf in (("v-flip (1-v)",np.clip(((1.0-cuv[:,1])*TS).astype(int),0,TS-1)),
                ("no-flip (v)", np.clip((cuv[:,1]*TS).astype(int),0,TS-1))):
    pn=atlas[pyf,px].sum(1)>24
    print(f"[{tag}] painted={int(pn.sum())}  painted frac front(nz>0.1)={100*(fnz[pn]>0.1).mean():.0f}%  unpainted frac back(nz<0)={100*(fnz[~pn]<0).mean():.0f}%")
py=np.clip((cuv[:,1]*TS).astype(int),0,TS-1); painted=atlas[py,px].sum(1)>24  # no-flip = correct (self-calibrated in export)
# render front: painted=green, unpainted-front=red
frast,_=dr.rasterize(ctx,camclip,f,resolution=[R,R]); fid=(frast[0,:,:,3].long()-1).cpu().numpy(); m=(frast[0,:,:,3]>0).cpu().numpy()
img=np.zeros((R,R,3),np.uint8)
valid=m&(fid>=0)
pf=painted[fid[valid]]
yy2,xx2=np.where(valid); img[yy2[pf],xx2[pf]]=(0,200,0); img[yy2[~pf],xx2[~pf]]=(200,0,0)
Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,"_coverage_check.png"))
print("saved _coverage_check.png (front view: GREEN=painted/skin, RED=unpainted) — GREEN should be the front body")
