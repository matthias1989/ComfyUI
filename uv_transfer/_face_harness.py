"""LOCAL full-projection harness: run the actual texturing on the saved UV'd mesh + front
photo, so I can see the FACE result + the [FaceWarp]/[FaceSkin]/[FeatureOffset] output and
iterate WITHOUT baking. Renders the face crop from the produced albedo."""
import os,sys,importlib.util,numpy as np
from PIL import Image
import trimesh,torch,nvdiffrast.torch as dr
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
RAW=os.environ.get("INPUT_IMG", r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00161_.png")
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection import texture_projection_multiview as tpm
# REMBG=1: rembg + square crop (clean black bg).  REMBG=0: raw image, gray bg, no crop
# (this is what the pipeline feeds -> exercises the corner-pixel fg-detection path).
REMBG=os.environ.get("REMBG","1")=="1"
if REMBG:
    from rembg import remove
    inp=remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
    bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
    ccx=(x0+x1)/2.0; ccy=(y0+y1)/2.0; size=int(max(x1-x0,y1-y0))
    crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2))).convert("RGB")
else:
    crop=Image.open(RAW).convert("RGB")
mesh=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
print(f"mesh: {len(mesh.vertices)} v {len(mesh.faces)} f  has_uv={hasattr(mesh.visual,'uv') and mesh.visual.uv is not None}")
FW=os.environ.get("FW","1")=="1"; FFO=os.environ.get("FFO","1")=="1"; TAG=os.environ.get("TAG","fwON")
print(f"=== RUN tag={TAG}  face_landmark_warp={FW}  face_front_only={FFO} ===")
out=tpm.texture_mesh_with_multiview(mesh,[crop],[0.0],[0.0],[1.0],
    texture_size=2048, ortho_scale=1.15, norm_size=1.15,
    face_front_only=FFO, face_landmark_warp=FW)
tri_obj,base_color,mr,nrm=out
base_color.save(os.path.join(HERE,f"_face_albedo_{TAG}.png"))
print(f"saved _face_albedo_{TAG}.png")
# render the face crop on the mesh (front camera)
v=torch.from_numpy(np.asarray(tri_obj.vertices)).float().cuda()
c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
f=torch.from_numpy(np.asarray(tri_obj.faces)).int().cuda()
uvt=torch.from_numpy(np.asarray(tri_obj.visual.uv)).float().cuda(); ctx=dr.RasterizeCudaContext()
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min())
alb=np.asarray(base_color.convert("RGB")); AH,AW=alb.shape[:2]
R=1000
def _render(name, sx, sz):
    cam=torch.stack([xx*2*sx,yy*2,zz*sz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
    rr,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fm=(rr[0,:,:,3]>0).cpu().numpy()
    uv=dr.interpolate(uvt.unsqueeze(0),rr,f)[0][0].cpu().numpy()
    col=np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); row=np.clip((1-uv[:,:,1])*AH,0,AH-1).astype(int)
    img=alb[row,col].copy(); img[~fm]=60
    Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,f"_render_{name}_{TAG}.png"))
    print(f"saved _render_{name}_{TAG}.png")
_render("front", 1.0, 1.0)
_render("back", -1.0, -1.0)
# report painted coverage vs the MODAL (base/unpainted) color, not vs white
alf=np.asarray(base_color.convert("RGB")).reshape(-1,3).astype(np.int32)
q=(alf//32); keys=q[:,0]*100+q[:,1]*10+q[:,2]
vals,cnts=np.unique(keys,return_counts=True); bk=vals[cnts.argmax()]
base=np.array([(bk//100)%10,(bk//10)%10,bk%10])*32+16
dist=np.abs(alf-base).sum(1); painted=dist>48
print(f"[atlas] base(unpainted)~{base.tolist()}  painted texels={100*painted.mean():.1f}%")
