"""LOCAL: overlay the mesh relief (in the projection's OWN bbox-midpoint alignment) on the front
photo, so we can SEE where the mesh's features (nipple bumps, navel) land vs the photo's features.
Any vertical drift = the per-feature misalignment a global offset can't fix. Quad mesh: _bodyquad.obj."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
front = np.asarray(crop).astype(np.float32)/255.0; IH, IW = front.shape[:2]
fg = front.sum(-1) > 0.05
m = trimesh.load(os.path.join(HERE, "_bodyquad.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
# projection's bbox-midpoint alignment (same as texture_projection_multiview.py:588-602)
s = 1.15; uc = v[:,0]/(s/2); vc = v[:,1]/(s/2)
co = np.argwhere(fg); yy0, yy1 = int(co[:,0].min()), int(co[:,0].max())
ich = float(yy1-yy0+1); icx = float(co[:,1].mean()); icy = float(yy0+yy1)/2
mcx = float((uc.min()+uc.max())*0.5); mcy = float((vc.min()+vc.max())*0.5); ppc = ich/float((vc.max()-vc.min()).clamp(min=1e-6))
xpix = icx + (uc-mcx)*ppc; ypix = icy - (vc-mcy)*ppc
xndc = (xpix+0.5)/IW*2-1; yndc = -(((ypix+0.5)/IH)*2-1)
d = v[:,2]; dmin = d.min(); dsp = (d.max()-dmin).clamp(min=1e-6); zndc = 1-2*(d-dmin)/dsp
clip = torch.stack([xndc, yndc, zndc, torch.ones_like(xndc)], -1).unsqueeze(0).contiguous()
rast, _ = dr.rasterize(ctx, clip, f, resolution=[IH, IW])
nrm, _ = dr.interpolate(vn[None], rast, f)
nz = nrm[0, :, :, 2].clamp(0, 1).cpu().numpy()[::-1].copy()      # front-facing brightness (bumps darken at rims)
hit = (rast[0, :, :, 3] > 0).cpu().numpy()[::-1].copy()
shade = (0.30 + 0.70*nz)
disp = (front*255).astype(np.uint8).copy()
# overlay relief (cool tint) at 50% where the mesh is, so photo features show through
ov = np.zeros_like(disp, np.float32)
ov[..., 0] = shade*120; ov[..., 1] = shade*180; ov[..., 2] = shade*230
m3 = hit[..., None].astype(np.float32)*0.5
disp = (disp*(1-m3) + ov*m3).clip(0,255).astype(np.uint8)
out = Image.fromarray(disp)
out.save(os.path.join(HERE, "_body_align_full.png"))
# chest crop (nipples) + hip/navel crop
out.crop((int(IW*0.30), int(IH*0.18), int(IW*0.70), int(IH*0.45))).resize((760, 520), Image.LANCZOS).save(os.path.join(HERE, "_body_align_chest.png"))
print("saved _body_align_full.png + _body_align_chest.png  (cool overlay = mesh, photo shows through)")
