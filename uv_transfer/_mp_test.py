"""Anchor diagnosis: get shared facial landmarks on BOTH the source image
(cropped to the head so MediaPipe can see it) and the mesh head render, convert
each to its native vertical coordinate (image row / mesh z), and fit the linear
map row = a*z + b. Residuals tell us offset vs tilt; a,b give the correction."""
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = (r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
         r"\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models"
         r"\face_landmarker_v2_with_blendshapes.task")
IDS = {'forehead10': 10, 'eye': None, 'nose1': 1, 'mouth13': 13, 'chin152': 152}

def detect(img):
    h, w = img.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8)))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    out = {}
    for k, i in IDS.items():
        out[k] = (lm[468].y + lm[473].y) / 2 if k == 'eye' else lm[i].y
    return out, h, w

# ---- mesh render: full landmarks -> mesh z via head_shot camera ----
vz = np.load('_genseams_viz.npz'); zmax = vz['co'][:, 2].max()
TGT = zmax - 0.105; S = 0.30; RES = 1250                  # head_shot camera
def render_row_to_z(yn): return TGT + S/2 - (yn*RES/RES)*S * (RES/RES)  # yn normalised
geom = np.array(Image.open('_ov_head_geom.png').convert('RGB'))
mg = detect(geom); print('mesh render detect:', 'OK' if mg else 'FAIL')

# ---- source image: crop to head so the face is big enough ----
im = np.array(Image.open('_front_ref.png').convert('RGB'))
H, W, _ = im.shape
warm = im[:, :, 0].astype(float) - im[:, :, 2]
sub = warm > 0.10*255
rows = np.where(sub.sum(1) > 0.01*W)[0]; iy0, iy1 = rows.min(), rows.max()
Hb = iy1 - iy0
hb = sub[iy0:iy0+int(0.22*Hb)]
cc = np.where(hb.sum(0) > 0)[0]; xc = (cc.min()+cc.max())//2
cy0, cy1 = max(0, iy0-int(0.03*Hb)), iy0+int(0.24*Hb)
cx0, cx1 = max(0, xc-int(0.16*Hb)), min(W, xc+int(0.16*Hb))
crop = im[cy0:cy1, cx0:cx1]
print(f'head crop rows[{cy0},{cy1}] cols[{cx0},{cx1}]  size {crop.shape[1]}x{crop.shape[0]}')
mi = detect(crop); print('image crop detect:', 'OK' if mi else 'FAIL')

if mg and mi:
    mgp, _, _ = mg; mip, ch, cw = mi
    print('\n%-12s  mesh_z   img_row' % 'landmark')
    zs, rs = [], []
    for k in IDS:
        z = TGT + S/2 - mgp[k]*S                 # mesh render y_norm -> z
        r = cy0 + mip[k]*ch                       # crop y_norm -> full image row
        zs.append(z); rs.append(r)
        print('%-12s  %.3f   %6.0f' % (k, z, r))
    zs, rs = np.array(zs), np.array(rs)
    a, b = np.polyfit(zs, rs, 1)                  # row = a*z + b
    pred = a*zs + b
    print('\nfit row = %.1f*z + %.1f' % (a, b))
    print('residuals (img_row - fit):', np.round(rs-pred, 1), ' max|res|=%.1f px' % np.abs(rs-pred).max())
    # current full-bbox map for comparison
    a0 = (iy0-iy1)/(zmax-vz['co'][:,2].min()); b0 = iy1 - a0*vz['co'][:,2].min()
    print('current full-bbox map row = %.1f*z + %.1f' % (a0, b0))
    print('at eye z=%.3f : landmark-fit row=%.0f  vs full-bbox row=%.0f  (delta %+.0f px)'
          % (zs[1], a*zs[1]+b, a0*zs[1]+b0, (a*zs[1]+b)-(a0*zs[1]+b0)))
