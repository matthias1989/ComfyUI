"""LOCAL read-only: faithful render of the current bake — skin=albedo, hair faces=cream
(the FBX hair material), unpainted non-hair holes left BLACK so we can see them. Front + head."""
import os, sys, glob, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__); ROOT = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
alb_path = glob.glob(sorted(glob.glob(ROOT + r"\output\characters\*\run_*"), key=os.path.getmtime)[-1] + r"\*__albedo.png")[0]
print("albedo:", os.path.basename(alb_path))
alb = np.asarray(Image.open(alb_path).convert("RGB")).astype(np.int32); AH, AW = alb.shape[:2]
hair = np.load(os.path.join(HERE, "last_seams_hairfaces.npy"))
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c = (v.min(0).values + v.max(0).values)*0.5; v = v-c; r = torch.sqrt((v**2).sum(-1).max()); v = v/(r*2)
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
uvt = torch.from_numpy(np.asarray(m.visual.uv)).float().cuda(); ctx = dr.RasterizeCudaContext()
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min()); R=1100
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rr,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fm=(rr[0,:,:,3]>0).cpu().numpy(); fid=(rr[0,:,:,3].long()-1).cpu().numpy()
uv=dr.interpolate(uvt.unsqueeze(0),rr,f)[0][0].cpu().numpy()
col=np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); row=np.clip((1-uv[:,:,1])*AH,0,AH-1).astype(int)
img=alb[row,col].copy().astype(np.uint8); img[~fm]=[55,55,55]
# hair faces -> cream (the FBX hair material)
ish=np.zeros((R,R),bool); val=fm&(fid<len(hair)); ish[val]=hair[fid[val]]
img[ish]=[224,210,178]
full=Image.fromarray(np.flipud(img).copy())
full.resize((660,660),Image.LANCZOS).save(os.path.join(HERE,"_result_front.png"))
full.crop((int(R*0.32),0,int(R*0.68),int(R*0.30))).resize((640,533),Image.LANCZOS).save(os.path.join(HERE,"_result_head.png"))
print("saved _result_front.png, _result_head.png")
