"""Verify compute_face_region_faces: render the selected (MediaPipe face-region) faces in
GREEN on the front view. They should cover the FACE only (not the hairline, not the body)."""
import os,sys,importlib.util,numpy as np
from PIL import Image
import torch,nvdiffrast.torch as dr,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); ORTHO=1.15; R=900
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
sel=LW.compute_face_region_faces(v,f,vn,photo,ORTHO,ctx)
if sel is None: print("FACE REGION: None (detection failed)"); sys.exit(1)
print(f"face-region faces = {int(sel.sum())} of {len(sel)}")
# also compare to the OLD geometric guard
_fcen=np.asarray(m.vertices)[np.asarray(m.faces)].mean(1); _fnrm=np.asarray(m.vertex_normals)[np.asarray(m.faces)].mean(1)
# render front: green = MediaPipe face region
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min())
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rr,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fid=(rr[0,:,:,3].long()-1).cpu().numpy(); hit=(rr[0,:,:,3]>0).cpu().numpy()
# base shaded
nz=dr.interpolate(vn.unsqueeze(0),rr,f)[0][0,:,:,2].clamp(0,1).cpu().numpy()
img=np.zeros((R,R,3),np.uint8); vis=hit&(fid>=0)
yv,xv=np.where(vis); sh=(40+180*nz[vis]).astype(np.uint8)
img[yv,xv]=np.stack([sh,sh,sh],1)
fsel=sel[fid[vis]]
img[yv[fsel],xv[fsel]]=(40,220,40)
Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,"_faceregion.png"))
print("saved _faceregion.png  GREEN = MediaPipe face region (should be the face only)")
