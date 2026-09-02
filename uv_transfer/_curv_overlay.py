"""LOCAL: render the local-convexity field in the PHOTO's alignment, overlay on the photo, and pick
the convexity peak in each half of the chest band = the mesh nipple (robust 2D extraction from the
clean heatmap). Shows mesh nipples (cyan) on the photo so we can judge alignment vs the real areolae."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr, cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
RC = float(os.environ.get("R", "0.020"))
inp = remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.; ccy=(y0+y1)/2.; size=int(max(x1-x0,y1-y0))
crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2))).convert("RGB")
front=np.asarray(crop).astype(np.float32)/255.; IH,IW=front.shape[:2]; fg=front.sum(-1)>0.05
m=trimesh.load(os.path.join(HERE,"_bodyquad.obj"),force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
V=v.numpy(); Nn=np.asarray(m.vertex_normals); F=np.asarray(m.faces)
conv=np.zeros(len(V),np.float32); fr=np.where(Nn[:,2]>0.1)[0]; tree=cKDTree(V[fr]); nb=tree.query_ball_point(V[fr],RC,workers=-1)
for k,j in enumerate(fr):
    if len(nb[k])>=4: conv[j]=np.dot(V[j]-V[fr[nb[k]]].mean(0),Nn[j])
conv=np.clip(conv,0,None)
# align + render conv in photo space
co=np.argwhere(fg); icx=float(co[:,1].mean()); ich=float(co[:,0].max()-co[:,0].min()+1); icy=float(co[:,0].min()+co[:,0].max())/2
s=1.15; uc=V[:,0]/(s/2); vc=V[:,1]/(s/2); mcx=(uc.min()+uc.max())/2; mcy=(vc.min()+vc.max())/2; ppc=ich/max(vc.max()-vc.min(),1e-6)
xpix=icx+(uc-mcx)*ppc; ypix=icy-(vc-mcy)*ppc
xndc=(xpix+0.5)/IW*2-1; yndc=-(((ypix+0.5)/IH)*2-1); d=V[:,2]; zndc=1-2*(d-d.min())/(d.max()-d.min())
vt=torch.tensor(np.stack([xndc,yndc,zndc,np.ones_like(xndc)],1),dtype=torch.float32,device='cuda')[None].contiguous()
ft=torch.from_numpy(F.astype(np.int32)).cuda(); ctx=dr.RasterizeCudaContext()
rast,_=dr.rasterize(ctx,vt,ft,resolution=[IH,IW])
cimg,_=dr.interpolate(torch.from_numpy(conv).float().cuda()[:,None][None],rast,ft)
cimg=cimg[0,:,:,0].cpu().numpy()[::-1].copy(); hit=(rast[0,:,:,3]>0).cpu().numpy()[::-1].copy()
cs=gaussian_filter(np.where(hit,cimg,0),3)
# nipple = conv peak per half in the chest band
by0=int(icy-0.32*ich); by1=int(icy-0.05*ich); cols=np.arange(IW)[None,:]
def peak(side):
    msk=np.zeros((IH,IW),bool); msk[max(0,by0):by1]=True; msk&=hit
    msk&=(cols>icx+0.02*ich) if side=='L' else (cols<icx-0.02*ich)
    i=np.unravel_index(np.argmax(np.where(msk,cs,-9)),msk.shape); return (float(i[1]),float(i[0]))
pL,pR=peak('L'),peak('R')
print(f"mesh nipples (conv peak): L={tuple(round(x) for x in pL)} R={tuple(round(x) for x in pR)}")
# PHOTO areola = color-anomaly peak (deviation from smooth skin) in a tight window around the cross
brt = front.mean(-1)
sm = np.stack([gaussian_filter(front[...,c], max(6.0,0.05*ich)) for c in range(3)], -1)
resid = np.abs(front - sm).sum(-1)
def areola(cross):
    cx,cy=int(cross[0]),int(cross[1]); w=int(0.055*ich)
    msk=np.zeros((IH,IW),bool); msk[max(0,cy-w):cy+w,max(0,cx-w):cx+w]=True; msk&=fg&(brt>0.40)
    if not msk.any(): return None
    i=np.unravel_index(np.argmax(np.where(msk,resid,-9)),msk.shape); return (float(i[1]),float(i[0]))
aL,aR=areola(pL),areola(pR)
print(f"photo areola (anomaly):   L={None if not aL else tuple(round(x) for x in aL)} R={None if not aR else tuple(round(x) for x in aR)}")
for nm,c,a in (("L",pL,aL),("R",pR,aR)):
    if a: print(f"  {nm}: areola-mesh dx={a[0]-c[0]:+.0f} dy={a[1]-c[1]:+.0f}px ({100*(a[1]-c[1])/ich:+.1f}% body-h)")
# overlay heatmap + markers
disp=(front*255).astype(np.uint8).copy()
t=np.clip(cs/max(np.percentile(cs[hit],99.5),1e-6),0,1)
heat=np.stack([(40+215*t),(60*(1-np.abs(2*t-1))),(200*(1-t))],-1).clip(0,255)
mm=(hit&(t>0.15))[...,None]*0.28
disp=(disp*(1-mm)+heat*mm).clip(0,255).astype(np.uint8)
for p in (pL,pR): cv2.drawMarker(disp,(int(p[0]),int(p[1])),(0,255,255),cv2.MARKER_CROSS,34,3)        # mesh nipple = cyan
for a in (aL,aR):
    if a: cv2.drawMarker(disp,(int(a[0]),int(a[1])),(255,40,40),cv2.MARKER_TILTED_CROSS,30,3)         # photo areola = red
Image.fromarray(disp).crop((int(IW*0.28),int(IH*0.12),int(IW*0.72),int(IH*0.46))).resize((780,560),Image.LANCZOS).save(os.path.join(HERE,"_curv_overlay.png"))
print("saved _curv_overlay.png (heat=convexity, cyan cross=detected mesh nipple)")
