"""SPIKE: semantic hair segmentation on the front image via the anime-face parser (shape/context,
not colour). HAIR class = blue (0,0,255). Crops the head, runs the parser, extracts the hair mask,
overlays it on the crop so we can judge hair-vs-skin separation. Embedded python (torch+cv2)."""
import sys, os, numpy as np
try:
    import truststore; truststore.inject_into_ssl(); print("truststore injected")
except Exception as e:
    print("truststore n/a:", e)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_controlnet_aux\src")
import torch
from custom_controlnet_aux.anime_face_segment import AnimeFaceSegmentor
from PIL import Image

D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
im = np.array(Image.open(os.path.join(D, "_front_ref.png")).convert("RGB")); H, W, _ = im.shape
# crop the head (subject bbox top portion) so the face/hair is big enough for the parser
warm = im[:, :, 0].astype(int) - im[:, :, 2]
sub = warm > 26; rows = np.where(sub.sum(1) > 0.01 * W)[0]; iy0, iy1 = int(rows.min()), int(rows.max())
Hb = iy1 - iy0
# head x-centre from the TOP of the subject (narrow head, before the arms widen it)
hb = sub[iy0:iy0 + int(0.20 * Hb)]; cc = np.where(hb.sum(0) > 0)[0]; xc = int((cc.min() + cc.max()) // 2)
# PORTRAIT crop of the head only: tall enough for hair+face, narrow enough to EXCLUDE the spread arms
y0, y1 = max(0, iy0 - int(0.03 * Hb)), iy0 + int(0.30 * Hb)
half = int(0.22 * Hb)
x0, x1 = max(0, xc - half), min(W, xc + half)
crop = np.ascontiguousarray(im[y0:y1, x0:x1])
print("crop", crop.shape, "head xc", xc)

seg = AnimeFaceSegmentor.from_pretrained().to("cuda" if torch.cuda.is_available() else "cpu")
out = seg(crop, detect_resolution=512, output_type="np", remove_background=True)
out = np.asarray(out)
print("seg out", out.shape)
segmap = out[:, :, :3]
# HAIR = blue (0,0,255)
hair = (segmap[:, :, 2] > 180) & (segmap[:, :, 0] < 70) & (segmap[:, :, 1] < 70)
face = (segmap[:, :, 1] > 180) & (segmap[:, :, 0] < 70) & (segmap[:, :, 2] < 70)   # FACE=green
skin = (segmap[:, :, 1] > 180) & (segmap[:, :, 2] > 180) & (segmap[:, :, 0] < 70)  # SKIN=cyan
print("hair px=%d  face px=%d  skin px=%d  / %d" % (int(hair.sum()), int(face.sum()), int(skin.sum()), hair.size))
Image.fromarray(segmap.astype(np.uint8)).save(os.path.join(D, "_seg_map.png"))
sh, sw = segmap.shape[:2]
crop_r = np.array(Image.fromarray(crop).resize((sw, sh)))
ov = crop_r.astype(float).copy()
ov[hair] = ov[hair] * 0.35 + np.array([255, 60, 60]) * 0.65   # tint hair red
Image.fromarray(ov.astype(np.uint8)).save(os.path.join(D, "_seg_hair_ov.png"))
print("saved _seg_map.png, _seg_hair_ov.png")
