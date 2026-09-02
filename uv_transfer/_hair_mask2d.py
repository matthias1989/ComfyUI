"""Clean 2D hair region from the front image (runs in ComfyUI embedded python: has cv2).

Colour can't separate temple-hair from shadowed cheek per-face — but in the image the hair
is ONE connected region and the cheek/brows/flecks are separate. So:
  1. classify hair pixels (Otsu on the red channel within the head -> per-character, no fixed
     threshold; hair is darker/less-red than skin),
  2. keep the single largest connected component (drops brows, lashes, shadow flecks),
  3. close interior holes (fills the 'white spots'), open away thin speckle.
Saves _hair_mask2d.png (255 = hair) for the Blender side to sample per face.
"""
import os, numpy as np, cv2
from PIL import Image

D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
im = np.array(Image.open(os.path.join(D, "_front_ref.png")).convert("RGB"))
H, W, _ = im.shape
R = im[:, :, 0]; B = im[:, :, 2]
warm = R.astype(int) - B.astype(int)
subject = warm > 26                                   # body silhouette (~0.10*255), excludes bg
rows = np.where(subject.sum(1) > 0.01 * W)[0]
iy0, iy1 = int(rows.min()), int(rows.max())
head = np.zeros_like(subject)
head[iy0:iy0 + int(0.28 * (iy1 - iy0))] = True        # top ~28% = head, for a clean Otsu

vals = R[subject & head].astype(np.uint8)
thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# hair only in the UPPER body (head + drape to ~shoulders); the dark legs are far below
# and would otherwise dominate the "largest blob".
upper = np.zeros_like(subject)
upper[iy0:iy0 + int(0.35 * (iy1 - iy0))] = True
hair = (subject & upper & (R < thr)).astype(np.uint8)

# bridge threshold fragmentation (hair colour ~= skin, so the hair breaks into pieces)
hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
# keep largest connected component (the real hair mass)
n, lab, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
if n > 1:
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hair = (lab == big).astype(np.uint8)
# fill interior holes (the white spots), then shave thin speckle
hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

Image.fromarray((hair * 255).astype(np.uint8)).save(os.path.join(D, "_hair_mask2d.png"))
print("otsu R thr=%d  hair px=%d / subject %d  (image %dx%d)"
      % (thr, int(hair.sum()), int(subject.sum()), W, H))
