"""Overlay the projected mesh (heuristic + offset) onto the photo: shows whether the body
SILHOUETTE stays aligned or the offset drags everything off, and whether features land."""
import os,sys,importlib.util,numpy as np
from PIL import Image,ImageDraw
import torch,trimesh
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
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
V=np.asarray(m.vertices); VN=np.asarray(m.vertex_normals)
v=V.copy().astype(np.float64)
for _ in range(2):
    c=(v.min(0)+v.max(0))*0.5; v=v-c; r=np.sqrt((v**2).sum(1).max()); v=v*(1.15/(r*2))
s=ORTHO; xx=v[:,0]/(s/2); yy=v[:,1]/(s/2)
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=co[:,0].min(),co[:,0].max()
ich=float(fy1-fy0+1); icx=float(co[:,1].mean()); icy=float(fy0+fy1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5)
ppc=ich/float(yy.max()-yy.min())
print(f"mesh y: min/max=({v[:,1].min():.3f},{v[:,1].max():.3f})  1/99 pct=({np.percentile(v[:,1],1):.3f},{np.percentile(v[:,1],99):.3f})")
print(f"-> bbox height={yy.max()-yy.min():.3f}  robust(1-99) height={(np.percentile(yy,99)-np.percentile(yy,1)):.3f}  (big gap = stray geometry inflating scale)")
fa=importlib.util.spec_from_file_location("fa","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/feature_anchor_warp.py")
faw=importlib.util.module_from_spec(fa); fa.loader.exec_module(faw)
dx,dy=faw.compute_global_offset(torch.from_numpy(v).float().cuda(),torch.from_numpy(VN.copy()).float().cuda(),photo,icx,icy,ppc,mcx,mcy,s)
print(f"offset dx={dx:.1f} dy={dy:.1f}")
# project FRONT-facing verts with heuristic+offset
front=VN[:,2]>0.2
xp=icx+(xx[front]-mcx)*ppc+dx; yp=icy-(yy[front]-mcy)*ppc+dy
ov=(np.clip(photo,0,1)*255).astype(np.uint8).copy()
X=np.clip(xp.astype(int),0,IW-1); Y=np.clip(yp.astype(int),0,IH-1)
ov[Y,X]=(0,150,255)                                   # projected mesh front verts = blue
im=Image.fromarray(ov); d=ImageDraw.Draw(im)
# photo silhouette top/bottom lines for reference
d.line([(0,fy0),(IW,fy0)],fill=(255,255,0),width=2); d.line([(0,fy1),(IW,fy1)],fill=(255,255,0),width=2)
print(f"photo body top/bottom y = {fy0}/{fy1}")
print(f"projected mesh front y range = {yp.min():.0f}/{yp.max():.0f}  (compare to photo {fy0}/{fy1})")
im.save(os.path.join(HERE,"_proj_check.png"))
print("saved _proj_check.png  BLUE=projected mesh (heuristic+offset)  YELLOW=photo body top/bottom. Aligned silhouette = blue fills between yellow lines.")
