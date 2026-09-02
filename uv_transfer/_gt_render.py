"""Ground truth: render the REAL baked albedo on the REAL mesh, front view, both UV-v
conventions. Black-inside = unpainted gap (the user sees cream there if it's hair-flagged,
black otherwise). No projection re-run, no guessing - just the actual texture on the mesh."""
import os,sys,numpy as np
from PIL import Image
import torch,nvdiffrast.torch as dr,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); R=900
OBJ=os.path.join(HERE,"last_seams.obj")
ALB=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00197_\run_20260617_033543\character_posed_00197__albedo.png"
alb=np.asarray(Image.open(ALB).convert("RGB")); AH,AW=alb.shape[:2]
m=trimesh.load(OBJ,force='mesh')
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c=(v.min(0).values+v.max(0).values)*0.5; v=v-c; r=torch.sqrt((v**2).sum(-1).max()); v=v*(1.0/(r*2))
f=torch.from_numpy(np.asarray(m.faces)).int().cuda()
uvt=torch.from_numpy(np.asarray(m.visual.uv)).float().cuda(); ctx=dr.RasterizeCudaContext()
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min())
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
rast,_=dr.rasterize(ctx,cam,f,resolution=[R,R]); fm=(rast[0,:,:,3]>0).cpu().numpy()
uv=dr.interpolate(uvt.unsqueeze(0),rast,f)[0][0].cpu().numpy()      # per-pixel UV
albt=alb
for tag,row in (("noflip",(uv[:,:,1]*AH)),("vflip",((1-uv[:,:,1])*AH))):
    col=np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); rw=np.clip(row,0,AH-1).astype(int)
    out=albt[rw,col].copy(); out[~fm]=0
    black=(out.sum(2)<24)&fm; out[black]=(255,0,255)
    Image.fromarray(np.flipud(out).copy()).save(os.path.join(HERE,f"_gt_{tag}.png"))
    print(f"{tag}: unpainted-inside pixels={int(black.sum())}")
print("saved _gt_noflip.png and _gt_vflip.png  (magenta = unpainted; pick the one where the FACE is on the head)")
# camera-view normal z + painted, split L/R in the CHEST band -> is her-right turned away?
nz=dr.interpolate(vn.unsqueeze(0),rast,f)[0][0,:,:,2].cpu().numpy()
pos=dr.interpolate(v.unsqueeze(0),rast,f)[0][0].cpu().numpy()        # per-pixel 3D mesh pos
col=np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); rw=np.clip((1-uv[:,:,1])*AH,0,AH-1).astype(int)
paint=(albt[rw,col].sum(2)>=24)&fm
# upper-body gap pixels (exclude arms by |x| small), classify by mesh position+normal
hy=(pos[:,:,1]-pos[fm][:,1].min())/(np.ptp(pos[fm][:,1])+1e-9)       # 0=feet 1=head
torso=fm&(np.abs(pos[:,:,0])<0.12)&(hy>0.55)&(hy<0.80)              # central upper torso
gap=torso&(~paint)
print(f"central upper-torso: pixels={int(torso.sum())} unpainted={int(gap.sum())} ({100*gap.sum()/max(1,torso.sum()):.0f}%)")
if gap.sum()>20:
    print(f"  gap mean mesh-x={pos[gap][:,0].mean():+.3f} (sign=side; +x vs -x)  mean camera-nz={nz[gap].mean():.2f}  facing-front(nz>0.12)={100*(nz[gap]>0.12).mean():.0f}%")
    print(f"  painted mean mesh-x={pos[torso&paint][:,0].mean():+.3f}  (compare signs to see which side is the gap)")
    for s,sm in (("x<0",gap&(pos[:,:,0]<0)),("x>0",gap&(pos[:,:,0]>=0))):
        if sm.sum()>5: print(f"    gap {s}: n={int(sm.sum())} nz={nz[sm].mean():.2f} front%={100*(nz[sm]>0.12).mean():.0f}")
