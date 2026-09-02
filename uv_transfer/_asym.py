"""Diagnose the LEFT/RIGHT asymmetry: for the breast band and the head band, how much
of each side is COVERED by the front projection, and WHERE each side samples in the
photo (to tell coverage-loss from mis-sampling)."""
import os,sys,numpy as np
from PIL import Image
import torch,torch.nn.functional as F,nvdiffrast.torch as dr,trimesh,cv2
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); TS=2048; ORTHO=1.15; DEPTH_EPS=0.02
mesh=trimesh.load(os.path.join(HERE,"last_seams.glb"),force='mesh')
v=torch.from_numpy(mesh.vertices).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
f=torch.from_numpy(mesh.faces).int().cuda()
vn=torch.from_numpy(np.asarray(mesh.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
uvt=torch.from_numpy(np.asarray(mesh.visual.uv)).float().cuda(); ctx=dr.RasterizeCudaContext()
uvclip=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
rast,_=dr.rasterize(ctx,uvclip,f,resolution=[TS,TS]); uvhit=rast[0,:,:,3]>0
tp=dr.interpolate(v.unsqueeze(0),rast,f)[0][0]; tn=dr.interpolate(vn.unsqueeze(0),rast,f)[0][0]; tn=tn/(tn.norm(dim=-1,keepdim=True)+1e-8)
s=ORTHO; uc=tp[:,:,0]/(s/2); vc=tp[:,:,1]/(s/2)
xx=v[:,0]/(s/2); yy=v[:,1]/(s/2); d=v[:,2]; dmin=d.min(); dsp=(d.max()-dmin).clamp(min=1e-6); zz=1-2*(d-dmin)/dsp
camclip=torch.stack([xx,yy,zz,torch.ones_like(xx)],-1).unsqueeze(0)
crast,_=dr.rasterize(ctx,camclip,f,resolution=[2048,2048]); chit=crast[0,:,:,3]>0
chit_img=torch.from_numpy(cv2.dilate(chit.cpu().numpy().astype(np.uint8),np.ones((5,5),np.uint8),2)).float().cuda()[None,None]
cdepth=dr.interpolate(v[:,2].contiguous()[None,:,None].contiguous(),crast,f)[0][0].permute(2,0,1)[None]
cocc=torch.where(chit[None,None],cdepth,torch.full_like(cdepth,1e9)); cocc=-F.max_pool2d(-cocc,3,1,1)
gocc=torch.stack([uc,vc],-1)[None]
shit=F.grid_sample(chit_img,gocc,'bilinear','zeros',align_corners=False)[0,0]>0.05
sdep=F.grid_sample(cocc,gocc,'bilinear','border',align_corners=False)[0,0]; depth_match=tp[:,:,2]>=sdep-DEPTH_EPS
photo=np.array(Image.open(os.path.join(HERE,"_rembg_premult.png")).convert("RGB")).astype(np.float32)/255
IH,IW=photo.shape[:2]; fg=photo.sum(-1)>0.05; co=np.argwhere(fg)
y0,y1=co[:,0].min(),co[:,0].max(); ich=float(y1-y0+1); icx=float(co[:,1].mean()); icy=float(y0+y1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6))
xpix=icx+(uc-mcx)*ppc; ypix=icy-(vc-mcy)*ppc; us=((xpix+0.5)/IW)*2-1; vs=((ypix+0.5)/IH)*2-1; inb=(us.abs()<=1)&(vs.abs()<=1)
grid=torch.stack([us.clamp(-1,1),vs.clamp(-1,1)],-1)[None]
col=F.grid_sample(torch.from_numpy(photo).cuda().permute(2,0,1)[None],grid,'bilinear','border',align_corners=False)[0].permute(1,2,0)
sfg=F.grid_sample(torch.from_numpy(fg.astype(np.float32)).cuda()[None,None],grid,'bilinear','zeros',align_corners=False)[0,0]>0.5
bright=col.mean(-1)>0.08; ang=tn[:,:,2]>0.12
covered=uvhit&inb&shit&depth_match&sfg&bright&ang
# to numpy
cov=covered.cpu().numpy(); hit=uvhit.cpu().numpy()
PX=tp[:,:,0].cpu().numpy(); PY=tp[:,:,1].cpu().numpy(); PZ=tp[:,:,2].cpu().numpy(); NZ=tn[:,:,2].cpu().numpy()
XPIX=xpix.cpu().numpy()
ymin=PY[hit].min(); ymax=PY[hit].max(); hn=(PY-ymin)/(ymax-ymin+1e-9)
front=hit&(NZ>0.12)
print(f"photo char center x={icx:.0f} (W={IW})")
def band(name, sel):
    L=sel&(PX>0.02); R=sel&(PX<-0.02)
    for tag,m in (("char-LEFT(+x)",L),("char-RIGHT(-x)",R)):
        nt=int(m.sum()); ncov=int((m&cov).sum())
        if nt==0: print(f"  {name} {tag}: none"); continue
        sx=XPIX[m&cov];
        print(f"  {name} {tag}: covered {ncov}/{nt} ({100*ncov/nt:.0f}%)  "
              f"sample_x mean={sx.mean():.0f}" if ncov>50 else f"  {name} {tag}: covered {ncov}/{nt} ({100*ncov/nt:.0f}%)")
print("HEAD band (hn>0.85, front-facing):"); band("head", front&(hn>0.85))
print("FACE band (hn 0.78-0.88, front-facing):"); band("face", front&(hn>0.78)&(hn<0.88))
print("BREAST band (hn 0.60-0.72, front, nz>0.3):"); band("breast", hit&(NZ>0.3)&(hn>0.60)&(hn<0.72))
# overall left/right front coverage
band("ALL front", front)
# per-mask breakdown for the breast region (why 0% on the right?)
M=dict(inb=inb.cpu().numpy(), shit=shit.cpu().numpy(), depth=depth_match.cpu().numpy(),
       sfg=sfg.cpu().numpy(), bright=bright.cpu().numpy(), ang=ang.cpu().numpy())
print("\n=== BREAST per-mask pass-rate (hit & nz>0.3 & hn0.60-0.72) ===")
for tag,xs in (("LEFT(+x)",PX>0.02),("RIGHT(-x)",PX<-0.02)):
    sel=hit&(NZ>0.3)&(hn>0.60)&(hn<0.72)&xs; nt=int(sel.sum())
    if nt==0: continue
    print(f" {tag} n={nt}: "+"  ".join(f"{k}={100*int((sel&m).sum())/nt:.0f}%" for k,m in M.items()))
print("nz>0.3 alone is the front-protruding subset; also try nz>0.5:")
for tag,xs in (("LEFT(+x)",PX>0.02),("RIGHT(-x)",PX<-0.02)):
    sel=hit&(NZ>0.5)&(hn>0.60)&(hn<0.74)&xs; nt=int(sel.sum())
    if nt==0: print(f" {tag} nz>0.5: none"); continue
    print(f" {tag} nz>0.5 n={nt}: "+"  ".join(f"{k}={100*int((sel&m).sum())/nt:.0f}%" for k,m in M.items()))
td=tp[:,:,2].cpu().numpy(); sd=sdep.cpu().numpy()
print("\n=== BREAST depth values (tex_depth vs sampled_depth, eps=0.02) ===")
for tag,xs in (("LEFT(+x)",PX>0.02),("RIGHT(-x)",PX<-0.02)):
    sel=hit&(NZ>0.3)&(hn>0.60)&(hn<0.72)&xs
    if sel.sum()==0: continue
    g=sd[sel]-td[sel]
    print(f" {tag}: tex_depth mean={td[sel].mean():.4f}  sampled_depth mean={sd[sel].mean():.4f}  "
          f"gap(samp-tex) mean={g.mean():.4f} max={g.max():.4f}  frac gap>eps={100*(g>0.02).mean():.0f}%")
# compare to a well-covered region (belly/torso center that passes)
sel=hit&(NZ>0.5)&(hn>0.45)&(hn<0.55)&(np.abs(PX)<0.05)
if sel.sum()>0:
    g=sd[sel]-td[sel]
    print(f" (ref) lower-torso center nz>0.5: gap mean={g.mean():.4f}  depth-pass={100*(g<=0.02).mean():.0f}%")
# where do the breast texels TRY to sample? (mean ypix vs photo)
sel=hit&(NZ>0.3)&(hn>0.60)&(hn<0.72)
print(f"\nbreast band sample y: mean={ypix.cpu().numpy()[sel].mean():.0f} (photo H={IH}, char y[{y0},{y1}])")
print(f"breast band mesh hn range maps to: this band should be chest/breast height")
