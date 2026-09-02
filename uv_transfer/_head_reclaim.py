"""LOCAL read-only: on the head, show the HAIR flag (red) vs the MediaPipe face-reclaim
coverage (green); yellow=both. Reveals the reclaim covers the wrong/too-little region."""
import os, sys, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
hair = np.load(os.path.join(HERE, "last_seams_hairfaces.npy"))
fs = os.path.join(HERE, "last_seams_faceskin.npy")
face = np.load(fs) if os.path.exists(fs) else np.zeros(len(m.faces), bool)
print(f"faces={len(m.faces)} hair={int(hair.sum())} faceskin={int(face.sum())} overlap={int((hair&face).sum())}")
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c = (v.min(0).values + v.max(0).values)*0.5; v = v-c; r = torch.sqrt((v**2).sum(-1).max()); v = v/(r*2)
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
xx=v[:,0]; yy=v[:,1]; d=v[:,2]; zz=1-2*(d-d.min())/(d.max()-d.min()); R=1200
cam=torch.stack([xx*2,yy*2,zz,torch.ones_like(xx)],-1).unsqueeze(0).contiguous()
ctx=dr.RasterizeCudaContext(); rr,_=dr.rasterize(ctx,cam,f,resolution=[R,R])
fid=(rr[0,:,:,3].long()-1).cpu().numpy(); fm=fid>=0
img=np.full((R,R,3),50,np.uint8); img[fm]=[120,120,120]
H=np.zeros((R,R),bool); F=np.zeros((R,R),bool); val=fm&(fid<len(hair))
H[val]=hair[fid[val]]; F[val]=face[fid[val]]
img[H&~F]=[230,40,40]    # hair only
img[F&~H]=[40,210,40]    # face-reclaim only
img[H&F]=[230,230,40]    # both
im=Image.fromarray(np.flipud(img).copy()); im.crop((int(R*0.30),0,int(R*0.70),int(R*0.32))).resize((720,576),Image.LANCZOS).save(os.path.join(HERE,"_head_reclaim.png"))
print("saved _head_reclaim.png")
