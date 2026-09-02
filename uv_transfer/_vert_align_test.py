"""Test a PRINCIPLED vertical alignment: anchor the vertical map on the BODY (chin->feet),
not the ear-tip bounding box. Check if that aligns face + nipples vertically with NO offset."""
import os,sys,importlib.util,numpy as np
from PIL import Image
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
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
f=torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
lw=importlib.util.spec_from_file_location("lw","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/landmark_warp.py")
LW=importlib.util.module_from_spec(lw); lw.loader.exec_module(LW)
fa=importlib.util.spec_from_file_location("fa","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Trellis2/projection/feature_anchor_warp.py")
FA=importlib.util.module_from_spec(fa); fa.loader.exec_module(FA)
ctx=dr.RasterizeCudaContext()
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=int(co[:,0].min()),int(co[:,0].max())
icx=float(co[:,1].mean()); icy=float(fy0+fy1)/2; ich=float(fy1-fy0+1)
xx=v[:,0]/(ORTHO/2); yy=v[:,1]/(ORTHO/2)
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float(yy.max()-yy.min())
box=LW._head_crop_box(fg); relief=LW._render_relief(v,f,vn,ORTHO,ctx,fg,IH,IW)
mesh_lm=LW._detect_478(relief,box); img_lm=LW._detect_478((photo*255).astype(np.uint8),box)
pa=FA._detect_photo_anchors(photo); ma=FA._mesh_anchors(v.cpu().numpy(),vn.cpu().numpy(),icx,icy,ppc,mcx,mcy,ORTHO)
# vertical references (projected y, heuristic space)
m_chin=mesh_lm[:,1].max(); p_chin=img_lm[:,1].max()
m_eye=mesh_lm[[468,473],1].mean(); p_eye=img_lm[[468,473],1].mean()
m_feet=float(fy1); p_feet=float(fy1)   # heuristic maps mesh bbox bottom -> fy1 already
print(f"CURRENT (ear-tip bbox) vertical residuals (photo - mesh):")
print(f"  eyes  : {p_eye-m_eye:+.1f}px   nipL: {pa['L'][1]-ma['L'][1]:+.1f}   nipR: {pa['R'][1]-ma['R'][1]:+.1f}")
# PRINCIPLED: y' = a*y + b fit to (chin->chin, feet->feet) — body-anchored, ears excluded
A=np.array([[m_chin,1],[m_feet,1]]); B=np.array([p_chin,p_feet]); a,b=np.linalg.solve(A,B)
print(f"body-anchored vertical map: y' = {a:.3f}*y {b:+.1f}  (chin & feet, no ears, no pixel offset)")
def vy(y): return a*y+b
print(f"AFTER body-anchored vertical residuals:")
print(f"  eyes  : {p_eye-vy(m_eye):+.1f}px   nipL: {pa['L'][1]-vy(ma['L'][1]):+.1f}   nipR: {pa['R'][1]-vy(ma['R'][1]):+.1f}")
