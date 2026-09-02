"""LOCAL: reliable nipple detection. MESH nipple = local-convexity argmax per side (the sharp bump).
PHOTO nipple = areola (locally dark + saturated) in a TIGHT window around the projected mesh nipple.
Draw both + offset. This is the anchor pair for the warp."""
import os, sys, numpy as np, torch, trimesh, cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
RC = float(os.environ.get("R", "0.020")); WIN = float(os.environ.get("WIN", "0.045"))
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0,x0 = bb[:,0].min(),bb[:,1].min(); y1,x1 = bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.0; ccy=(y0+y1)/2.0; size=int(max(x1-x0,y1-y0))
crop = inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH,IW = front.shape[:2]; fg = front.sum(-1)>0.05
m = trimesh.load(os.path.join(HERE,"_bodyquad.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
V=v.numpy(); N=np.asarray(m.vertex_normals)
# alignment
co=np.argwhere(fg); icx=float(co[:,1].mean()); ich=float(co[:,0].max()-co[:,0].min()+1); icy=float(co[:,0].min()+co[:,0].max())/2
s=1.15; mcx=(V[:,0]/(s/2)).min()*0+ (V[:,0]/(s/2)); mcx=( (V[:,0]/(s/2)).min()+(V[:,0]/(s/2)).max())/2
mcy=((V[:,1]/(s/2)).min()+(V[:,1]/(s/2)).max())/2; ppc=ich/max((V[:,1]/(s/2)).max()-(V[:,1]/(s/2)).min(),1e-6)
def pr(px,py): return (icx+(px/(s/2)-mcx)*ppc, icy-(py/(s/2)-mcy)*ppc)
# MESH nipple = convexity argmax per side, upper torso front
conv=np.zeros(len(V)); fr=np.where(N[:,2]>0.1)[0]; tree=cKDTree(V[fr]); nb=tree.query_ball_point(V[fr],RC,workers=-1)
for k,j in enumerate(fr):
    if len(nb[k])>=4: conv[j]=np.dot(V[j]-V[fr[nb[k]]].mean(0), N[j])
hn=(V[:,1]-V[:,1].min())/(np.ptp(V[:,1])+1e-9)
reg=(hn>0.55)&(hn<0.80)&(V[:,2]>0)&(N[:,2]>0.2)
def mnip(xm):
    sel=reg&xm&(conv>0)
    if sel.sum()<10: return None
    ii=np.where(sel)[0]; w=conv[ii]**3              # cube emphasizes the sharp nipple peak over folds
    return pr(np.average(V[ii,0],weights=w), np.average(V[ii,1],weights=w))
mL=mnip(V[:,0]>0.02); mR=mnip(V[:,0]<-0.02)
# PHOTO areola: locally dark + saturated, in a tight window around the projected mesh nipple
brt=front.mean(-1); mx=front.max(-1); mn=front.min(-1); sat=(mx-mn)/(mx+1e-6)
darkl=gaussian_filter(brt,max(6.0,0.04*ich))-brt
score=darkl+0.5*gaussian_filter(sat,2)
def pnip(mp):
    if mp is None: return None
    cx,cy=int(mp[0]),int(mp[1]); w=int(WIN*ich)
    msk=np.zeros((IH,IW),bool); msk[max(0,cy-w):cy+w,max(0,cx-w):cx+w]=True; msk&=fg&(brt>0.40)
    if not msk.any(): return None
    i=np.unravel_index(np.argmax(np.where(msk,score,-9)),msk.shape); return (float(i[1]),float(i[0]))
pL,pR=pnip(mL),pnip(mR)
disp=(front*255).astype(np.uint8).copy()
def mk(p,c,st):
    if p: cv2.drawMarker(disp,(int(p[0]),int(p[1])),c,st,30,3)
mk(mL,(0,230,255),cv2.MARKER_TILTED_CROSS); mk(mR,(0,230,255),cv2.MARKER_TILTED_CROSS)
mk(pL,(255,40,40),cv2.MARKER_TILTED_CROSS); mk(pR,(255,40,40),cv2.MARKER_TILTED_CROSS)
print(f"mesh  nipples (cyan): L={None if not mL else tuple(round(x) for x in mL)} R={None if not mR else tuple(round(x) for x in mR)}")
print(f"photo nipples (red):  L={None if not pL else tuple(round(x) for x in pL)} R={None if not pR else tuple(round(x) for x in pR)}")
for nm,mp,pp in (("L",mL,pL),("R",mR,pR)):
    if mp and pp: print(f"  {nm}: photo-mesh dx={pp[0]-mp[0]:+.0f} dy={pp[1]-mp[1]:+.0f}px ({100*(pp[1]-mp[1])/ich:+.1f}% body-h)")
Image.fromarray(disp).crop((int(IW*0.28),int(IH*0.12),int(IW*0.72),int(IH*0.46))).resize((780,560),Image.LANCZOS).save(os.path.join(HERE,"_nipple_final.png"))
print("saved _nipple_final.png")
