"""LOCAL: apply the freshly-baked atlas to last_seams.glb through its real UVs and render the
textured mesh (front + head crop) so I see exactly what the bake looks like, no theory."""
import os, sys, glob, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
ATL = sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_*.png"), key=os.path.getmtime)[-1]
print("atlas:", os.path.basename(ATL))
atl = np.asarray(Image.open(ATL).convert("RGBA")).astype(np.float32)/255.0
AH, AW = atl.shape[:2]
tex = torch.from_numpy(atl[..., :3]).float().cuda().contiguous()
m = trimesh.load(os.path.join(HERE, "last_seams.glb"), force='mesh')
V = np.asarray(m.vertices); F = np.asarray(m.faces); UV = np.asarray(m.visual.uv)
v = torch.from_numpy(V).float().cuda()
c0=(v.min(0).values+v.max(0).values)*0.5; v=v-c0; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
ft = torch.from_numpy(F.astype(np.int32)).cuda()
uv = torch.from_numpy(UV).float().cuda().contiguous()
uvf = torch.from_numpy(F.astype(np.int32)).cuda()
ctx = dr.RasterizeCudaContext()
xx,yy,dd=v[:,0],v[:,1],v[:,2]; R=1100
zz=1-2*(dd-dd.min())/(dd.max()-dd.min())
clip=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rast,_=dr.rasterize(ctx,clip,ft,resolution=[R,R])
uvi,_=dr.interpolate(uv[None],rast,uvf)            # (1,R,R,2)
col=dr.texture(tex[None],uvi,filter_mode='linear')  # (1,R,R,3)
col=col[0].cpu().numpy()
hit=(rast[0,:,:,3]>0).cpu().numpy()
img=np.full((R,R,3),0.11,np.float32); img[hit]=col[hit]
full=Image.fromarray(np.flipud((img*255).clip(0,255).astype(np.uint8)).copy())
full.save(os.path.join(HERE,"_textured_front.png"))
full.crop((int(R*0.36),int(R*0.02),int(R*0.64),int(R*0.24))).resize((720,600),Image.LANCZOS).save(os.path.join(HERE,"_textured_head.png"))
print("saved _textured_front.png + _textured_head.png")
