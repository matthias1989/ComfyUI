"""Faithful reproduction of the bake's FRONT projection masks (INCLUDING occlusion
+ depth this time) to find what rejects most of the body. Per-mask pass-rate over
all front-facing texels + a covered-region image."""
import os, sys, math
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F, nvdiffrast.torch as dr, trimesh
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE=os.path.dirname(__file__); TS=2048; ORTHO=1.15; DEPTH_EPS=0.02
mesh=trimesh.load(os.path.join(HERE,"_real_mesh2.glb"),force='mesh')
uv=np.asarray(mesh.visual.uv)
v=torch.from_numpy(mesh.vertices).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
f=torch.from_numpy(mesh.faces).int().cuda()
vn=torch.from_numpy(np.asarray(mesh.vertex_normals)).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
uvt=torch.from_numpy(uv).float().cuda(); ctx=dr.RasterizeCudaContext()
uvclip=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
rast,_=dr.rasterize(ctx,uvclip,f,resolution=[TS,TS]); uvhit=rast[0,:,:,3]>0
tp=dr.interpolate(v.unsqueeze(0),rast,f)[0][0]; tn=dr.interpolate(vn.unsqueeze(0),rast,f)[0][0]; tn=tn/(tn.norm(dim=-1,keepdim=True)+1e-8)
# front camera: right=+X up=+Y look=-Z
s=ORTHO;
# occlusion rasterize (build_ortho_clip_verts)
xx=v[:,0]/(s/2); yy=v[:,1]/(s/2); d=v[:,2]; dmin=d.min(); dsp=(d.max()-dmin).clamp(min=1e-6); zz=1-2*(d-dmin)/dsp
camclip=torch.stack([xx,yy,zz,torch.ones_like(xx)],-1).unsqueeze(0)
OCC=2048
crast,_=dr.rasterize(ctx,camclip,f,resolution=[OCC,OCC]); chit=crast[0,:,:,3]>0
chit_np=chit.cpu().numpy().astype(np.uint8)
import cv2; chit_np=cv2.dilate(chit_np,np.ones((5,5),np.uint8),iterations=2)
chit_d=torch.from_numpy(chit_np).bool().cuda(); chit_img=chit_d.float()[None,None]
cdepth=dr.interpolate(v[:,2].contiguous().unsqueeze(0).unsqueeze(-1).contiguous(),crast,f)[0][0].permute(2,0,1).unsqueeze(0)
inf=torch.full_like(cdepth,1e9); cocc=torch.where(chit.unsqueeze(0).unsqueeze(0),cdepth,inf); cocc=-F.max_pool2d(-cocc,3,1,1)
# texel projection
uc=tp[:,:,0]/(s/2); vc=tp[:,:,1]/(s/2)
gocc=torch.stack([uc,vc],-1).unsqueeze(0)
shit=F.grid_sample(chit_img,gocc,'bilinear','zeros',align_corners=False)[0,0]>0.05
sdep=F.grid_sample(cocc,gocc,'bilinear','border',align_corners=False)[0,0]
tdep=tp[:,:,2]; depth_match=tdep>=sdep-DEPTH_EPS
# alignment
_imgname=sys.argv[1] if len(sys.argv)>1 else "_front_real_197.png"
photo=np.array(Image.open(os.path.join(HERE,_imgname)).convert("RGB")).astype(np.float32)/255
print(f"[using image: {_imgname}]")
IH,IW=photo.shape[:2]; fg=photo.sum(-1)>0.05; co=np.argwhere(fg)
y0,y1=co[:,0].min(),co[:,0].max(); ich=float(y1-y0+1); icx=float(co[:,1].mean()); icy=float(y0+y1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6))
xpix=icx+(uc-mcx)*ppc; ypix=icy-(vc-mcy)*ppc
us=((xpix+0.5)/IW)*2-1; vs=((ypix+0.5)/IH)*2-1
inb=(us.abs()<=1)&(vs.abs()<=1)
grid=torch.stack([us.clamp(-1,1),vs.clamp(-1,1)],-1).unsqueeze(0)
pt=torch.from_numpy(photo).cuda().permute(2,0,1).unsqueeze(0)
col=F.grid_sample(pt,grid,'bilinear','border',align_corners=False)[0].permute(1,2,0)
fgt=torch.from_numpy(fg.astype(np.float32)).cuda()[None,None]
sfg=F.grid_sample(fgt,grid,'bilinear','zeros',align_corners=False)[0,0]>0.5
bright=col.mean(-1)>0.08; ang=tn[:,:,2]>0.12
front=uvhit&(tn[:,:,2]>0.12)   # front-facing texels (should be coverable)
def pct(m):
    nf=int(front.sum()); return f"{int((front&m).sum())}/{nf} ({100*int((front&m).sum())/max(nf,1):.0f}%)"
print(f"front-facing texels (Nz>0.12): {int(front.sum())}")
print(f"  pass in_bounds   : {pct(inb)}")
print(f"  pass sampled_hit : {pct(shit)}")
print(f"  pass depth_match : {pct(depth_match)}")
print(f"  pass sampled_fg  : {pct(sfg)}")
print(f"  pass bright>0.08 : {pct(bright)}")
print(f"  pass good_angle  : {pct(ang)}")
vis=uvhit&inb&shit&depth_match&sfg&bright&ang
print(f"  ALL (covered)    : {pct(vis)}")
# save covered region as image (atlas)
atl=np.zeros((TS,TS,3),np.uint8); atl[vis.cpu().numpy()]=(col.cpu().numpy()[vis.cpu().numpy()]*255).astype(np.uint8)
Image.fromarray(atl).save(os.path.join(HERE,"_cov_atlas.png")); print("saved _cov_atlas.png")
