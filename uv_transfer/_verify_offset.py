"""LOCAL verify: run the REAL pipeline functions (updated _mesh_anchors convexity + _detect_photo_anchors
on delit + compute_global_offset) and check post-offset residuals ~0 = the auto uniform shift aligns."""
import os, sys, importlib.util, numpy as np, torch, trimesh
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
spec=importlib.util.spec_from_file_location("fl","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-FlattenLight/__init__.py")
fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
photo=fl.FlattenLight().execute(torch.from_numpy(rgbi)[None],0.55,0.05,0.85,1.15,0.20,True)[0][0,...,:3].clamp(0,1).cpu().numpy()
IH,IW=photo.shape[:2]
m=trimesh.load(os.path.join(HERE,"_bodyquad.obj"),force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
s=ORTHO; xx=v[:,0]/(s/2); yy=v[:,1]/(s/2)
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); icx=float(co[:,1].mean()); ich=float(co[:,0].max()-co[:,0].min()+1); icy=float(co[:,0].min()+co[:,0].max())/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6).item())
fa=importlib.util.spec_from_file_location("fa","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/feature_anchor_warp.py")
faw=importlib.util.module_from_spec(fa); fa.loader.exec_module(faw)
pa=faw._detect_photo_anchors(photo)
ma=faw._mesh_anchors(v.cpu().numpy(), vn.cpu().numpy(), icx,icy,ppc,mcx,mcy,s)
print("PHOTO:",{k:(round(pa[k][0]),round(pa[k][1])) if pa.get(k) else None for k in('face','L','R')})
print("MESH :",{k:(round(ma[k][0]),round(ma[k][1])) if ma.get(k) else None for k in('face','L','R')})
for k in('face','L','R'):
    if pa.get(k) and ma.get(k): print(f"  offset[{k}] = ({pa[k][0]-ma[k][0]:+.0f},{pa[k][1]-ma[k][1]:+.0f})px")
dx,dy=faw.compute_global_offset(v,vn,photo,icx,icy,ppc,mcx,mcy,s)
print(f"\ncompute_global_offset -> dx={dx:.1f} dy={dy:.1f}")
for k in('face','L','R'):
    if pa.get(k) and ma.get(k): print(f"  POST-offset residual[{k}] = ({pa[k][0]-(ma[k][0]+dx):+.1f},{pa[k][1]-(ma[k][1]+dy):+.1f})px")
