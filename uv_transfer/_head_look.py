"""LOCAL: look at the current char's HEAD — (1) the baked albedo on the mesh, (2) which faces
gen_seams flagged as hair (red). Front view, cropped to the head. Auto-finds newest albedo."""
import os, sys, glob, time, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
ROOT = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
runs = sorted(glob.glob(ROOT + r"\output\characters\*\run_*"), key=os.path.getmtime, reverse=True)
newest = runs[0]; print("newest run:", os.path.relpath(newest, ROOT), time.ctime(os.path.getmtime(newest)))
alb_path = glob.glob(newest + r"\*__albedo.png")[0]
print("albedo:", os.path.basename(alb_path))
print("last_seams.obj mtime:", time.ctime(os.path.getmtime(os.path.join(HERE, "last_seams.obj"))))
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
alb = np.asarray(Image.open(alb_path).convert("RGB")); AH, AW = alb.shape[:2]
hair = np.load(os.path.join(HERE, "last_seams_hairfaces.npy"))
print(f"faces={len(m.faces)} hairfaces={int(hair.sum())} albedo={AW}x{AH}")
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c = (v.min(0).values + v.max(0).values) * 0.5; v = v - c; r = torch.sqrt((v**2).sum(-1).max()); v = v / (r*2)
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
uvt = torch.from_numpy(np.asarray(m.visual.uv)).float().cuda(); ctx = dr.RasterizeCudaContext()
xx = v[:,0]; yy = v[:,1]; d = v[:,2]; zz = 1 - 2*(d-d.min())/(d.max()-d.min())
R = 1200
cam = torch.stack([xx*2, yy*2, zz, torch.ones_like(xx)], -1).unsqueeze(0).contiguous()
rr,_ = dr.rasterize(ctx, cam, f, resolution=[R,R]); fm = (rr[0,:,:,3]>0).cpu().numpy()
fid = (rr[0,:,:,3].long()-1).cpu().numpy()
uv = dr.interpolate(uvt.unsqueeze(0), rr, f)[0][0].cpu().numpy()
col = np.clip(uv[:,:,0]*AW,0,AW-1).astype(int); row = np.clip((1-uv[:,:,1])*AH,0,AH-1).astype(int)
# (1) albedo on mesh
img = alb[row,col].copy(); img[~fm] = 50
# (2) hair-face overlay
hov = img.copy(); ish = np.zeros((R,R),bool); val = fm & (fid<len(hair)); ish[val] = hair[fid[val]]
hov[ish] = (0.5*hov[ish] + 0.5*np.array([230,30,30])).astype(np.uint8)
def headcrop(a):
    im = Image.fromarray(np.flipud(a).copy()); return im.crop((int(R*0.30),0,int(R*0.70),int(R*0.32))).resize((720,576),Image.LANCZOS)
headcrop(img).save(os.path.join(HERE,"_head_albedo.png"))
headcrop(hov).save(os.path.join(HERE,"_head_hair.png"))
print("saved _head_albedo.png, _head_hair.png")
