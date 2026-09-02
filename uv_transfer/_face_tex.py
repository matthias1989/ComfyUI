"""LOCAL: render the actual bake's albedo on the mesh (manual UV gather, V-flip handled) and crop the
head, so I see the real smeared face and can diagnose it. Read-only."""
import os, sys, glob, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE=os.path.dirname(__file__)
ALB=sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\*\run_*\*__albedo.png"),key=os.path.getmtime)[-1]
print("albedo:",os.path.basename(os.path.dirname(ALB)),os.path.basename(ALB))
alb=np.asarray(Image.open(ALB).convert("RGB")).astype(np.float32)/255.; AH,AW=alb.shape[:2]
m=trimesh.load(os.path.join(HERE,"_facemesh.obj"),force='mesh')
V=np.asarray(m.vertices); F=np.asarray(m.faces); UV=np.asarray(m.visual.uv)
v=torch.from_numpy(V).float().cuda(); c0=(v.min(0).values+v.max(0).values)*0.5; v=v-c0; r=torch.sqrt((v**2).sum(-1).max()); v=v/(r*2)
ft=torch.from_numpy(F.astype(np.int32)).cuda(); uv=torch.from_numpy(UV).float().cuda().contiguous(); ctx=dr.RasterizeCudaContext()
xx,yy,dd=v[:,0],v[:,1],v[:,2]; R=1200; zz=1-2*(dd-dd.min())/(dd.max()-dd.min())
clip=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rast,_=dr.rasterize(ctx,clip,ft,resolution=[R,R])
uvi,_=dr.interpolate(uv[None],rast,ft); uvi=uvi[0].cpu().numpy(); hit=(rast[0,:,:,3]>0).cpu().numpy()
col=np.full((R,R,3),0.11,np.float32)
u=np.clip(uvi[...,0],0,1); vv=np.clip(uvi[...,1],0,1)
cc=np.clip((u*AW).astype(int),0,AW-1); rr=np.clip(((1-vv)*AH).astype(int),0,AH-1)   # V-flip: V up
col[hit]=alb[rr[hit],cc[hit]]
img=Image.fromarray(np.flipud((col*255).clip(0,255).astype(np.uint8)).copy())
img.save(os.path.join(HERE,"_face_tex_front.png"))
img.crop((int(R*0.36),int(R*0.02),int(R*0.64),int(R*0.26))).resize((720,620),Image.LANCZOS).save(os.path.join(HERE,"_face_tex_head.png"))
print("saved _face_tex_front.png + _face_tex_head.png")
