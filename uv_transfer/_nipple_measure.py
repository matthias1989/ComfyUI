"""LOCAL: objectively measure nipple misalignment. Mesh nipple apex (most-forward vertex per breast)
projected into photo space vs the photo's areola (redness peak). dy>0 = photo areola is BELOW the
mesh apex => projection paints the nipple too HIGH on the mesh. Quad mesh: _bodyquad.obj."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr, cv2
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
V0 = np.asarray(m.vertices)
v = torch.from_numpy(V0).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v-c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
vN = v.cpu().numpy()                                   # normalized verts (Y up, Z front)
# projection alignment -> pixel coords per vertex
s = 1.15; uc = vN[:,0]/(s/2); vc = vN[:,1]/(s/2)
co = np.argwhere(fg); yy0, yy1 = int(co[:,0].min()), int(co[:,0].max())
ich = float(yy1-yy0+1); icx = float(co[:,1].mean()); icy = float(yy0+yy1)/2
mcx = (uc.min()+uc.max())*0.5; mcy = (vc.min()+vc.max())*0.5; ppc = ich/max(vc.max()-vc.min(),1e-6)
xpix = icx + (uc-mcx)*ppc; ypix = icy - (vc-mcy)*ppc
# --- mesh nipple apex: most-forward (max Z) vertex per breast, in the upper-chest band ---
yhi, ylo = vN[:,1].max(), vN[:,1].min(); H = yhi-ylo
chest = (vN[:,1] > ylo+0.62*H) & (vN[:,1] < ylo+0.80*H) & (vN[:,2] > 0)   # upper torso, front
apex = {}
for side, msk in (("L (char-right, viewer-left)", chest & (vN[:,0] < -0.02)),
                  ("R (char-left, viewer-right)", chest & (vN[:,0] >  0.02))):
    idx = np.where(msk)[0]
    if len(idx)==0: apex[side]=None; continue
    a = idx[np.argmax(vN[idx,2])]                    # max forward = nipple tip
    apex[side] = (xpix[a], ypix[a])
# --- photo areola: redness peak per side in the chest band ---
red = (front[:,:,0] - 0.5*(front[:,:,1]+front[:,:,2]))
red[~fg] = -1
cy0 = int(icy - 0.18*ich); cy1 = int(icy + 0.02*ich)        # chest vertical band
band = np.zeros_like(red, bool); band[cy0:cy1, :] = True
areola = {}
for side, xr in (("L (char-right, viewer-left)", (int(icx-0.22*ich), int(icx-0.02*ich))),
                 ("R (char-left, viewer-right)", (int(icx+0.02*ich), int(icx+0.22*ich)))):
    msk = band.copy(); msk[:, :xr[0]] = False; msk[:, xr[1]:] = False
    rm = np.where(msk, cv2.GaussianBlur(red,(0,0),5), -1)
    py, px = np.unravel_index(np.argmax(rm), rm.shape)
    areola[side] = (px, py)
disp = (front*255).astype(np.uint8).copy()
print("== nipple alignment (pixels; +dy = areola below mesh apex = painted too HIGH) ==")
for side in apex:
    a = apex[side]; ar = areola[side]
    if a is None: print(f"  {side}: no mesh apex found"); continue
    cv2.drawMarker(disp, (int(a[0]),int(a[1])), (0,230,255), cv2.MARKER_CROSS, 26, 3)   # mesh apex = cyan
    cv2.drawMarker(disp, (int(ar[0]),int(ar[1])), (255,40,40), cv2.MARKER_TILTED_CROSS, 22, 3)  # photo areola = red
    dx, dy = ar[0]-a[0], ar[1]-a[1]
    print(f"  {side}: mesh_apex=({a[0]:.0f},{a[1]:.0f})  photo_areola=({ar[0]},{ar[1]})  dx={dx:+.0f} dy={dy:+.0f} px  ({100*dy/ich:+.1f}% of body height)")
Image.fromarray(disp).crop((int(IW*0.30), int(IH*0.18), int(IW*0.70), int(IH*0.45))).resize((760,520), Image.LANCZOS).save(os.path.join(HERE, "_nipple_measure.png"))
print("saved _nipple_measure.png  (cyan=mesh apex, red=photo areola)")
