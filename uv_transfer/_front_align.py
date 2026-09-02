"""Compute the front-image -> mesh alignment from shared MediaPipe facial landmarks.

Runs in the ComfyUI embedded python (has mediapipe). Detects landmarks on:
  - the mesh head render (_ov_head_geom.png), produced by _front_overlay.py with a
    known orthographic camera -> converts landmark (x_norm,y_norm) to mesh (x,z).
  - the source image (_front_ref.png), cropped to the head so the face is big enough
    -> landmark (x_norm,y_norm) to full-image (col,row).
Fits the linear maps  row = a*z + b  and  col = c*x + d  and writes them to
_front_align.json for the Blender side (gen_seams / overlay) to consume.

Camera constants MUST match head_shot() in _front_overlay.py.
"""
import json, numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
MODEL = (r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
         r"\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models"
         r"\face_landmarker_v2_with_blendshapes.task")
HEAD_TGT_OFF = 0.105      # head_shot: tgt_z = zmax - HEAD_TGT_OFF
HEAD_S       = 0.30       # head_shot ortho_scale (vertical span; render is portrait)

# dense midline strip forehead->chin: robust slope, insensitive to any single noisy point
VERT = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0,
        11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HORZ = [33, 263, 133, 362, 234, 454, 130, 359, 162, 389]   # eye corners + cheeks + temples
def eye_y(lm): return (lm[468].y + lm[473].y) / 2

def detect(img):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8)))
    return res.face_landmarks[0] if res.face_landmarks else None

# ---- mesh render -> (x,z) per landmark, via the EXACT camera calibration that the
#      Blender side wrote (_head_cam.json) — no ortho_scale assumptions ----
import os
vz = np.load(os.path.join(D, '_genseams_viz.npz')); co = vz['co']
cal = json.load(open(os.path.join(D, '_head_cam.json')))
geom = np.array(Image.open(os.path.join(D, '_ov_head_geom.png')).convert('RGB'))
gh, gw = geom.shape[:2]
g = detect(geom)
if g is None:
    raise SystemExit("FAIL: no face on mesh render")
def mesh_xz(lm):
    z = cal["z1"] + (lm.y - cal["yn1"]) / (cal["yn2"] - cal["yn1"]) * (cal["z2"] - cal["z1"])
    x = cal["x1"] + (lm.x - cal["xn1"]) / (cal["xn2"] - cal["xn1"]) * (cal["x2"] - cal["x1"])
    return x, z

# ---- image cropped to head -> (col,row) per landmark ----
im = np.array(Image.open(os.path.join(D, '_front_ref.png')).convert('RGB'))
H, W, _ = im.shape
warm = im[:, :, 0].astype(float) - im[:, :, 2]
sub = warm > 0.10*255
rows = np.where(sub.sum(1) > 0.01*W)[0]; iy0, iy1 = rows.min(), rows.max(); Hb = iy1-iy0
hb = sub[iy0:iy0+int(0.22*Hb)]; cc = np.where(hb.sum(0) > 0)[0]; xc = (cc.min()+cc.max())//2
cy0, cy1 = max(0, iy0-int(0.03*Hb)), iy0+int(0.24*Hb)
cx0, cx1 = max(0, xc-int(0.16*Hb)), min(W, xc+int(0.16*Hb))
crop = im[cy0:cy1, cx0:cx1]; ch, cw = crop.shape[:2]
i = detect(crop)
if i is None:
    raise SystemExit("FAIL: no face on image head crop")
def img_cr(lm):
    return cx0 + lm.x*cw, cy0 + lm.y*ch          # full-image col,row

# ---- vertical fit: row = a*z + b ----
zs = np.array([mesh_xz(g[k])[1] for k in VERT])
rs = np.array([img_cr(i[k])[1] for k in VERT])
a, b = np.polyfit(zs, rs, 1); vres = rs - (a*zs+b)
# ---- piecewise upper segment (above the forehead) ----
# The mesh ears/crown are proportionally taller than the photo's, so the linear
# face fit extrapolates off the top of the image. Keep the exact face fit at/below
# the forehead landmark, and BEND the map above it so the mesh crown lands on the
# image head-top — the face/hairline stay pinned, only the crown/ear region shifts.
z_kink   = float(mesh_xz(g[10])[1])           # forehead-top landmark
row_kink = float(a*z_kink + b)
z_top    = float(co[:, 2].max())              # mesh crown / ear tip
row_top  = float(iy0)                         # image head top (warm)
a2 = (row_top - row_kink) / (z_top - z_kink)
b2 = row_kink - a2*z_kink
# ---- horizontal fit: col = c*x + d ----
xm = np.array([mesh_xz(g[k])[0] for k in HORZ]); cl = np.array([img_cr(i[k])[0] for k in HORZ])
c, d = np.polyfit(xm, cl, 1); hres = cl - (c*xm+d)

out = {"a": float(a), "b": float(b), "c": float(c), "d": float(d),
       "a2": float(a2), "b2": float(b2), "z_kink": z_kink,
       "z_face": float(np.mean(zs)),                # face-centre height (scale pivot)
       "v_maxres": float(np.abs(vres).max()), "h_maxres": float(np.abs(hres).max()),
       "img_w": int(W), "img_h": int(H)}
with open(os.path.join(D, '_front_align.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("vertical  row = %.1f*z + %.1f   max|res|=%.1f px" % (a, b, np.abs(vres).max()))
print("horizontal col = %.1f*x + %.1f  max|res|=%.1f px" % (c, d, np.abs(hres).max()))
print("wrote _front_align.json:", out)
