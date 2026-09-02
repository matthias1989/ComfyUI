"""Measure face + nipple alignment error: project the mesh's face-centre + breast-tips
through the REAL projection alignment and compare to the photo's MediaPipe face + nipples.
Renders an overlay on the photo. Read-only; tells us if the fix is a uniform shift, a
face-region scale, or per-feature."""
import os,sys,importlib.util,numpy as np
from PIL import Image,ImageDraw
import torch,nvdiffrast.torch as dr,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); ORTHO=1.15
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
# mesh + projection-frame normalize (match the bake)
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
s=ORTHO; xx=v[:,0]/(s/2); yy=v[:,1]/(s/2)
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=co[:,0].min(),co[:,0].max()
ich=float(fy1-fy0+1); icx=float(co[:,1].mean()); icy=float(fy0+fy1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5)
ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6).item())
# anchors via the project's own detectors
fa=importlib.util.spec_from_file_location("fa","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/feature_anchor_warp.py")
faw=importlib.util.module_from_spec(fa); fa.loader.exec_module(faw)
pa=faw._detect_photo_anchors(photo)
ma=faw._mesh_anchors(v.cpu().numpy(), vn.cpu().numpy(), icx, icy, ppc, mcx, mcy, s)
print("PHOTO anchors :", {k:(round(pa[k][0],1),round(pa[k][1],1)) if isinstance(pa.get(k),tuple) else pa.get(k) for k in ('face','L','R')})
print("MESH  anchors :", {k:(round(ma[k][0],1),round(ma[k][1],1)) if ma.get(k) else None for k in ('face','L','R')})
for k in ('face','L','R'):
    if pa.get(k) and ma.get(k):
        print(f"  offset[{k}] = photo-mesh = ({pa[k][0]-ma[k][0]:+.1f}, {pa[k][1]-ma[k][1]:+.1f}) px   (ich={ich:.0f})")
# face SCALE: compare photo face height vs mesh face-region projected height
print(f"  photo face height(MediaPipe)~ {pa.get('fh','?')}")
# overlay
ov=(np.clip(photo,0,1)*255).astype(np.uint8).copy(); im=Image.fromarray(ov); dr2=ImageDraw.Draw(im)
def dot(p,c,r=6):
    if p: dr2.ellipse([p[0]-r,p[1]-r,p[0]+r,p[1]+r],outline=c,width=3)
for k,c in (('face',(0,255,0)),('L',(0,255,0)),('R',(0,255,0))): dot(pa.get(k),c)
for k,c in (('face',(255,0,0)),('L',(255,0,0)),('R',(255,0,0))): dot(ma.get(k),c)
# compare nipple detection on RAW (pre-delight) image — nipple colour intact there
pa_raw=faw._detect_photo_anchors(rgbi)
print("DELIT nipples :", {k:(round(pa[k][0],1),round(pa[k][1],1)) if pa.get(k) else None for k in ('L','R')})
print("RAW   nipples :", {k:(round(pa_raw[k][0],1),round(pa_raw[k][1],1)) if pa_raw.get(k) else None for k in ('L','R')})
for k in ('face','L','R'):
    if pa_raw.get(k): dot(pa_raw.get(k),(0,255,255))   # CYAN = raw-detected
im.save(os.path.join(HERE,"_align_overlay.png"))
print("saved _align_overlay.png  GREEN=delit-detected  CYAN=raw-detected  RED=mesh. Which sits on the real nipple?")
# NEW approach: fit per-axis AFFINE (scale+offset) mapping mesh-anchor px -> photo px
keys=[k for k in ('face','L','R') if pa.get(k) and ma.get(k)]
mx=np.array([ma[k][0] for k in keys]); my=np.array([ma[k][1] for k in keys])
pxs=np.array([pa[k][0] for k in keys]); pys=np.array([pa[k][1] for k in keys])
def fit(m,p):
    A=np.stack([m,np.ones_like(m)],1); sol,*_=np.linalg.lstsq(A,p,rcond=None); return float(sol[0]),float(sol[1])
ax,bx=fit(mx,pxs); ay,by=fit(my,pys)
print(f"\nAFFINE FIT  x: scale={ax:.3f} off={bx:+.1f}   y: scale={ay:.3f} off={by:+.1f}")
print("compare to OFFSET-ONLY (what failed): a single dy can't zero both face & nipples")
for k in keys:
    nx=ax*ma[k][0]+bx; ny=ay*ma[k][1]+by
    print(f"  post-AFFINE residual[{k}] = ({pa[k][0]-nx:+.1f}, {pa[k][1]-ny:+.1f}) px")
# similarity (Umeyama: scale+rotation+translation) — does rotation zero the nipples?
M=np.stack([mx,my],1); P=np.stack([pxs,pys],1)
mM=M.mean(0); mP=P.mean(0); Mc=M-mM; Pc=P-mP
C=(Pc.T@Mc)/len(M); U,S,Vt=np.linalg.svd(C); D=np.eye(2)
if np.linalg.det(U@Vt)<0: D[1,1]=-1
Rm=U@D@Vt; ss=np.trace(np.diag(S)@D)/ (Mc**2).sum()*len(M); t=mP-ss*(Rm@mM)
ang=np.degrees(np.arctan2(Rm[1,0],Rm[0,0]))
print(f"\nSIMILARITY  scale={ss:.3f} rotation={ang:+.1f}deg")
for i,k in enumerate(keys):
    q=ss*(Rm@M[i])+t
    print(f"  post-SIMILARITY residual[{k}] = ({P[i,0]-q[0]:+.1f}, {P[i,1]-q[1]:+.1f}) px")
# verify the ACTUAL function the projection will call (uniform median offset)
dx,dy=faw.compute_global_offset(v,vn,photo,icx,icy,ppc,mcx,mcy,s)
print(f"\ncompute_global_offset() -> dx={dx:.1f} dy={dy:.1f}  (uniform shift, no body distortion)")
for k in keys:
    print(f"  post-offset residual[{k}] = ({pa[k][0]-(ma[k][0]+dx):+.1f}, {pa[k][1]-(ma[k][1]+dy):+.1f}) px")
# CLEAR LARGE overlay: GREEN=photo target, RED=where mesh anchor lands after offset
ov2=(np.clip(photo,0,1)*255).astype(np.uint8).copy(); im2=Image.fromarray(ov2); d2=ImageDraw.Draw(im2)
def dot2(p,c,r=9):
    if p: d2.ellipse([p[0]-r,p[1]-r,p[0]+r,p[1]+r],outline=c,width=4)
for k in ('face','L','R'): dot2(pa.get(k),(0,255,0))
for k in ('face','L','R'):
    if ma.get(k): dot2((ma[k][0]+dx,ma[k][1]+dy),(255,40,40))
cx0=int(max(0,icx-230)); cx1=int(min(IW,icx+230))
crop2=im2.crop((cx0,0,cx1,300)).resize(((cx1-cx0)*2,600),Image.LANCZOS)
crop2.save(os.path.join(HERE,"_align_zoom.png"))
print("saved _align_zoom.png  GREEN=photo (target)  RED=mesh anchor AFTER offset (where it lands). Red on green = aligned.")
