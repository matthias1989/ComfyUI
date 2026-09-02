"""LOCAL: the GREEN photo-nipple method (=_detect_photo_anchors on the FlattenLight delit image,
the one the user confirmed good) paired with the convexity mesh nipple. Green=photo, cyan=mesh."""
import os, sys, importlib.util, numpy as np, torch, trimesh, nvdiffrast.torch as dr, cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE=os.path.dirname(__file__); ORTHO=1.15
RAW=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
inp=remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.; ccy=(y0+y1)/2.; size=int(max(x1-x0,y1-y0))
crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2)))
a2=np.array(crop).astype(np.float32)/255.; rgbi=a2[:,:,:3]*a2[:,:,3:4]
# delit via FlattenLight (matches _align_diag's green)
spec=importlib.util.spec_from_file_location("fl","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-FlattenLight/__init__.py")
fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
photo=fl.FlattenLight().execute(torch.from_numpy(rgbi)[None],0.55,0.05,0.85,1.15,0.20,True)[0][0,...,:3].clamp(0,1).cpu().numpy()
IH,IW=photo.shape[:2]; fg=photo.sum(-1)>0.05
# GREEN photo nipple = the project's redness-blob detector
fa=importlib.util.spec_from_file_location("fa","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/feature_anchor_warp.py")
faw=importlib.util.module_from_spec(fa); fa.loader.exec_module(faw)
pa=faw._detect_photo_anchors(photo)
print("GREEN photo nipples:", {k:(round(pa[k][0]),round(pa[k][1])) if pa.get(k) else None for k in ('L','R')})
# CYAN mesh nipple = convexity per-half peak, in the same photo alignment
m=trimesh.load(os.path.join(HERE,"_bodyquad.obj"),force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
V=v.numpy(); Nn=np.asarray(m.vertex_normals); F=np.asarray(m.faces)
conv=np.zeros(len(V),np.float32); fr=np.where(Nn[:,2]>0.1)[0]; tr=cKDTree(V[fr]); nb=tr.query_ball_point(V[fr],0.02,workers=-1)
for k,j in enumerate(fr):
    if len(nb[k])>=4: conv[j]=np.dot(V[j]-V[fr[nb[k]]].mean(0),Nn[j])
conv=np.clip(conv,0,None)
co=np.argwhere(fg); icx=float(co[:,1].mean()); ich=float(co[:,0].max()-co[:,0].min()+1); icy=float(co[:,0].min()+co[:,0].max())/2
s=ORTHO; uc=V[:,0]/(s/2); vc=V[:,1]/(s/2); mcx=(uc.min()+uc.max())/2; mcy=(vc.min()+vc.max())/2; ppc=ich/max(vc.max()-vc.min(),1e-6)
xpix=icx+(uc-mcx)*ppc; ypix=icy-(vc-mcy)*ppc; xndc=(xpix+0.5)/IW*2-1; yndc=-(((ypix+0.5)/IH)*2-1); d=V[:,2]; zndc=1-2*(d-d.min())/(d.max()-d.min())
vt=torch.tensor(np.stack([xndc,yndc,zndc,np.ones_like(xndc)],1),dtype=torch.float32,device='cuda')[None].contiguous()
ft=torch.from_numpy(F.astype(np.int32)).cuda(); ctx=dr.RasterizeCudaContext()
rast,_=dr.rasterize(ctx,vt,ft,resolution=[IH,IW])
cimg,_=dr.interpolate(torch.from_numpy(conv).float().cuda()[:,None][None],rast,ft); cimg=cimg[0,:,:,0].cpu().numpy()[::-1].copy()
hit=(rast[0,:,:,3]>0).cpu().numpy()[::-1].copy(); cs=gaussian_filter(np.where(hit,cimg,0),3)
by0=int(icy-0.32*ich); by1=int(icy-0.05*ich); cols=np.arange(IW)[None,:]
def peak(side):
    msk=np.zeros((IH,IW),bool); msk[max(0,by0):by1]=True; msk&=hit
    msk&=(cols>icx+0.02*ich) if side=='L' else (cols<icx-0.02*ich)
    i=np.unravel_index(np.argmax(np.where(msk,cs,-9)),msk.shape); return (float(i[1]),float(i[0]))
mL,mR=peak('L'),peak('R')
print("CYAN mesh nipples:  ", {'L':(round(mL[0]),round(mL[1])),'R':(round(mR[0]),round(mR[1]))})
for nm,p,mm in (("L",pa.get('L'),mL),("R",pa.get('R'),mR)):
    if p: print(f"  {nm}: photo(green)-mesh(cyan) dx={p[0]-mm[0]:+.0f} dy={p[1]-mm[1]:+.0f}px ({100*(p[1]-mm[1])/ich:+.1f}% body-h)  [+dy=mesh too high]")
# FACE offset (uniform shift works only if face offset ~= nipple offset)
ma=faw._mesh_anchors(V, Nn, icx, icy, ppc, mcx, mcy, s)
if pa.get('face') and ma.get('face'):
    print(f"  FACE: photo-mesh dx={pa['face'][0]-ma['face'][0]:+.0f} dy={pa['face'][1]-ma['face'][1]:+.0f}px ({100*(pa['face'][1]-ma['face'][1])/ich:+.1f}% body-h)")
disp=(np.clip(photo,0,1)*255).astype(np.uint8).copy()
for k in ('L','R'):
    if pa.get(k): cv2.circle(disp,(int(pa[k][0]),int(pa[k][1])),9,(0,255,0),3)        # GREEN photo
for mm in (mL,mR): cv2.drawMarker(disp,(int(mm[0]),int(mm[1])),(0,230,255),cv2.MARKER_CROSS,30,3)  # CYAN mesh
Image.fromarray(disp).crop((int(IW*0.28),int(IH*0.12),int(IW*0.72),int(IH*0.46))).resize((780,560),Image.LANCZOS).save(os.path.join(HERE,"_green_nipple.png"))
print("saved _green_nipple.png (green=photo nipple, cyan=mesh nipple)")
