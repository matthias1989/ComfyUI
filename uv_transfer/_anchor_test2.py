"""LOCAL: improved breast/nipple detection.
 MESH: local forward protrusion above a smoothed chest-wall baseline (isolates the breast bump,
       not the broad sternum that global max-Z grabbed).
 PHOTO: dark-AND-red areola score (areola is darker + redder than skin) in a lower band.
Draw markers + offsets; iterate until they sit on the real nipples."""
import os, sys, numpy as np, torch, trimesh, cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
PB_TOP = float(os.environ.get("PB_TOP", "0.30"))   # photo band top  = icy - PB_TOP*ich
PB_BOT = float(os.environ.get("PB_BOT", "0.12"))   # photo band bot  = icy - PB_BOT*ich
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
co = np.argwhere(fg); icx = float(co[:,1].mean()); ich = float(co[:,0].max()-co[:,0].min()+1); icy = float(co[:,0].min()+co[:,0].max())/2
m = trimesh.load(os.path.join(HERE, "_bodyquad.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c = (v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
vN = v.numpy(); nN = np.asarray(m.vertex_normals)
s = 1.15; uc = vN[:,0]/(s/2); vc = vN[:,1]/(s/2)
mcx = (uc.min()+uc.max())*0.5; mcy = (vc.min()+vc.max())*0.5; ppc = ich/max(vc.max()-vc.min(),1e-6)
def pr(px, py): return (icx + (px/(s/2)-mcx)*ppc, icy - (py/(s/2)-mcy)*ppc)
# ---- MESH: local protrusion ----
hn = (vN[:,1]-vN[:,1].min())/(np.ptp(vN[:,1])+1e-9)
cand = np.where((nN[:,2]>0.3) & (vN[:,2]>0) & (hn>0.45) & (hn<0.80) & (np.abs(vN[:,0])<0.35))[0]
xs, ys, zs = vN[cand,0], vN[cand,1], vN[cand,2]
G = 48
xi = ((xs-xs.min())/(np.ptp(xs)+1e-9)*(G-1)).astype(int); yi = ((ys-ys.min())/(np.ptp(ys)+1e-9)*(G-1)).astype(int)
zsum = np.zeros((G,G)); zcnt = np.zeros((G,G)); np.add.at(zsum,(yi,xi),zs); np.add.at(zcnt,(yi,xi),1)
zmean = np.where(zcnt>0, zsum/np.maximum(zcnt,1), np.nan); zmean[np.isnan(zmean)] = np.nanmedian(zmean)
base = gaussian_filter(zmean, sigma=4)        # smooth chest wall (breast bump removed)
protr = zs - base[yi, xi]
def mmound(xmask):
    sel = xmask & (protr > 0.20*protr.max())           # the breast bump region
    if sel.sum() < 10: return None
    ii = cand[sel]; w = protr[sel]                      # protrusion-weighted centroid of the mound
    return pr(np.average(vN[ii,0], weights=w), np.average(vN[ii,1], weights=w))
mL = mmound(xs > 0.02); mR = mmound(xs < -0.02)
# ---- PHOTO: areola = redness peak in a small window ON the breast (mesh apex = prior) ----
R_, G_, B_ = front[:,:,0], front[:,:,1], front[:,:,2]
redn = R_ - 0.5*(G_+B_); brt = (R_+G_+B_)/3.0
darkl = gaussian_filter(brt, max(8.0,0.05*ich)) - brt   # locally darker than surround (areola)
WIN = float(os.environ.get("WIN", "0.075"))             # search ± WIN*ich around the mesh mound
def pnip_near(apex):
    if apex is None: return None
    cx, cy = int(apex[0]), int(apex[1]); w = int(WIN*ich)
    msk = np.zeros((IH,IW),bool); msk[max(0,cy-w):cy+w, max(0,cx-w):cx+w] = True
    msk &= fg & (brt > 0.40)                             # lit breast only (excludes shadow/armpit)
    if not msk.any(): return None
    i = np.unravel_index(np.argmax(np.where(msk, darkl, -9)), msk.shape); return (float(i[1]), float(i[0]))
pL, pR = pnip_near(mL), pnip_near(mR)
disp = (front*255).astype(np.uint8).copy()
def mk(p,c,st):
    if p is not None: cv2.drawMarker(disp,(int(p[0]),int(p[1])),c,st,30,3)
mk(pL,(255,40,40),cv2.MARKER_TILTED_CROSS); mk(pR,(255,40,40),cv2.MARKER_TILTED_CROSS)
mk(mL,(0,230,255),cv2.MARKER_TILTED_CROSS); mk(mR,(0,230,255),cv2.MARKER_TILTED_CROSS)
print(f"photo nipples (red): L={None if pL is None else tuple(round(x) for x in pL)} R={None if pR is None else tuple(round(x) for x in pR)}")
print(f"mesh  tips   (cyan): L={None if mL is None else tuple(round(x) for x in mL)} R={None if mR is None else tuple(round(x) for x in mR)}")
for nm,p,mm in (("L",pL,mL),("R",pR,mR)):
    if p and mm: print(f"  {nm}: photo-mesh dy={p[1]-mm[1]:+.0f}px ({100*(p[1]-mm[1])/ich:+.1f}% body-h)  [+dy=mesh too HIGH]")
Image.fromarray(disp).crop((int(IW*0.26),int(IH*0.12),int(IW*0.74),int(IH*0.46))).resize((780,560),Image.LANCZOS).save(os.path.join(HERE,"_anchor_test2.png"))
print("saved _anchor_test2.png")
