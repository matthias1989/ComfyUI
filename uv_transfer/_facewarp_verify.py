"""Offline verification of the face-landmark warp on the CURRENT mesh, BEFORE any bake:
  - measure mean landmark offset mesh<->photo BEFORE and AFTER the warp (after should ~0)
  - render the warped face over the mesh relief to confirm eyes/nose/mouth land + NO smear."""
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
ctx=dr.RasterizeCudaContext()
fg=photo.sum(-1)>0.05; box=LW._head_crop_box(fg)
relief=LW._render_relief(v,f,vn,ORTHO,ctx,fg,IH,IW)
front_u8=(photo*255).clip(0,255).astype(np.uint8)
mesh_lm=LW._detect_478(relief,box); img_lm=LW._detect_478(front_u8,box)
if mesh_lm is None or img_lm is None:
    print(f"DETECTION FAILED mesh={mesh_lm is not None} img={img_lm is not None}"); sys.exit(1)
resid_before=float(np.linalg.norm(mesh_lm-img_lm,axis=1).mean())
out=LW.compute_face_landmark_warp(v,f,vn,photo,ORTHO,ctx)
warp,mask=out; wn=warp.cpu().numpy(); mn=mask.cpu().numpy()
_mr=np.where(mn.max(1)>0.1)[0]
print(f"WARP MASK reaches rows {_mr.min()}-{_mr.max()} of {IH}  (= {100*_mr.min()/IH:.0f}%-{100*_mr.max()/IH:.0f}% down the image; head only).")
print(f"  warp-mask pixels below the chest line (35% down) = {int((mn[int(0.35*IH):]>0.1).sum())}  -> body is untouchable by the warp")
sx=np.clip(wn[...,0],0,IW-1).astype(int); sy=np.clip(wn[...,1],0,IH-1).astype(int)
sampled=photo[sy,sx]
warped_lm=LW._detect_478((sampled*255).clip(0,255).astype(np.uint8),box)
resid_after=float(np.linalg.norm(mesh_lm-warped_lm,axis=1).mean()) if warped_lm is not None else -1
print(f"landmark offset mesh<->photo  BEFORE warp = {resid_before:.1f}px   AFTER warp = {resid_after:.1f}px  (want ~0)")
# visuals: relief with ORIGINAL photo vs WARPED photo blended in the face mask, zoomed
m3=mn[...,None]; relieff=relief.astype(np.float32)/255.0
def blend(src): return ((relieff*(1-m3)+src*m3)*255).astype(np.uint8)
bo=blend(photo); bw=blend(sampled)
# also full-image warped photo (to inspect smear on the whole face/body)
fullw=(photo*(1-m3)+sampled*m3*1.0); fullw=(fullw*255).astype(np.uint8)
cx0=int(max(0,box[0]-20)); cx1=int(min(IW,box[1]+20)); cy0=int(box[2]); cy1=int(min(IH,box[3]+int(0.10*IH)))
def zc(a): return Image.fromarray(a[cy0:cy1,cx0:cx1]).resize(((cx1-cx0)*2,(cy1-cy0)*2),Image.LANCZOS)
left=zc(bo); right=zc(bw); comp=Image.new("RGB",(left.width*2+10,left.height),(30,30,30))
comp.paste(left,(0,0)); comp.paste(right,(left.width+10,0))
comp.save(os.path.join(HERE,"_facewarp_compare.png"))
Image.fromarray(fullw).save(os.path.join(HERE,"_facewarp_full.png"))
print("saved _facewarp_compare.png  LEFT=relief+ORIGINAL photo (eyes off)  RIGHT=relief+WARPED photo (should align, no smear)")
print("saved _facewarp_full.png  (full warped photo — inspect for any smear outside the face)")
