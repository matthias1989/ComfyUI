"""LOCAL: run the EXISTING feature_anchor_warp anchor detection (photo nipples via MediaPipe+redness,
mesh breast tips) on the quad mesh + front photo, draw where they land, and print the mesh->photo
offset. Tests whether the nipple landmark catches the breasts well enough to drive a warp."""
import os, sys, numpy as np, torch, trimesh, cv2
from PIL import Image
from rembg import remove
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.feature_anchor_warp import _detect_photo_anchors, _mesh_anchors
RAW = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
inp = remove(Image.open(RAW).convert("RGB")); arr = np.array(inp); al = arr[:, :, 3]
bb = np.argwhere(al > 0.8*255); y0, x0 = bb[:,0].min(), bb[:,1].min(); y1, x1 = bb[:,0].max(), bb[:,1].max()
ccx = (x0+x1)/2.0; ccy = (y0+y1)/2.0; size = int(max(x1-x0, y1-y0))
crop = inp.crop((int(ccx-size//2), int(ccy-size//2), int(ccx+size//2), int(ccy+size//2))).convert("RGB")
fimg = np.asarray(crop).astype(np.float32)/255.0; IH, IW = fimg.shape[:2]
fg = fimg.sum(-1) > 0.05
m = trimesh.load(os.path.join(HERE, "_bodyquad.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
vN = v.numpy(); nN = np.asarray(m.vertex_normals)
s = 1.15; uc = vN[:,0]/(s/2); vc = vN[:,1]/(s/2)
co = np.argwhere(fg); icx = float(co[:,1].mean()); ich = float(co[:,0].max()-co[:,0].min()+1); icy = float(co[:,0].min()+co[:,0].max())/2
mcx = (uc.min()+uc.max())*0.5; mcy = (vc.min()+vc.max())*0.5; ppc = ich/max(vc.max()-vc.min(),1e-6)
pa = _detect_photo_anchors(fimg)
ma = _mesh_anchors(vN, nN, icx, icy, ppc, mcx, mcy, s)
print("photo anchors:", {k:(None if pa is None or pa.get(k) is None else tuple(round(x,1) for x in pa[k])) for k in ('face','L','R')} if pa else None)
print("mesh  anchors:", {k:(None if ma.get(k) is None else tuple(round(x,1) for x in ma[k])) for k in ('face','L','R')})
disp = (fimg*255).astype(np.uint8).copy()
def mark(p, col, style):
    if p is not None: cv2.drawMarker(disp,(int(p[0]),int(p[1])),col,style,30,3)
if pa:
    for k in ('L','R'): mark(pa.get(k),(255,40,40),cv2.MARKER_TILTED_CROSS)   # photo nipple = red
    mark(pa.get('face'),(255,40,40),cv2.MARKER_CROSS)
for k in ('L','R'): mark(ma.get(k),(0,230,255),cv2.MARKER_TILTED_CROSS)        # mesh tip = cyan
mark(ma.get('face'),(0,230,255),cv2.MARKER_CROSS)
if pa:
    for k in ('L','R','face'):
        if pa.get(k) and ma.get(k):
            dx=pa[k][0]-ma[k][0]; dy=pa[k][1]-ma[k][1]
            print(f"  {k}: photo-mesh  dx={dx:+.0f} dy={dy:+.0f} px  ({100*dy/ich:+.1f}% body-h)  [+dy = mesh too HIGH]")
Image.fromarray(disp).crop((int(IW*0.28),int(IH*0.10),int(IW*0.72),int(IH*0.48))).resize((780,560),Image.LANCZOS).save(os.path.join(HERE,"_anchor_test.png"))
print("saved _anchor_test.png  (red=photo nipple/face, cyan=mesh tip/face)")
