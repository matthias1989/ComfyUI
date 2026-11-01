"""
FlattenLight — removes baked shadows from reference images before projection texturing.
Trellis2XAtlasUnwrap — xatlas UV unwrap (kept for reference).
Trellis2BlenderSmartUV — decimate + zone-based Smart UV Project via Blender 5.1 headless.

Zone-based UV strategy:
  The body is split into 6 zones by vertex position (head / torso / arm_pos / arm_neg /
  leg_pos / leg_neg).  Smart UV Project is run on each zone in isolation, which produces
  a small number of large, recognisable islands per body part rather than thousands of
  micro-fragments.  All islands are packed at the end.

  This avoids the LSCM-seam approach which requires clean quad edge-loops that are not
  present after edge-collapse decimation.
"""

import torch
import copy
import numpy as np
import os
import tempfile
import subprocess
import json
import struct
import datetime


BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"


# ── Eye detection + texture helpers ───────────────────────────────────────────

def _detect_eye_info_opencv(front_image_pil):
    """
    Fallback eye detector using OpenCV Haar cascades (always available — no
    extra installs).  Less precise than MediaPipe but good enough to locate
    iris centres and sample colour.

    Returns the same dict structure as the MediaPipe path, or None on failure.
    """
    img_np = np.array(front_image_pil.convert('RGB'))
    h, w   = img_np.shape[:2]
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    import os as _os
    _hc_dir = cv2.data.haarcascades

    face_cas = cv2.CascadeClassifier(_os.path.join(_hc_dir, 'haarcascade_frontalface_default.xml'))
    eye_cas  = cv2.CascadeClassifier(_os.path.join(_hc_dir, 'haarcascade_eye.xml'))

    # Detect face — pick largest hit
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(max(40, w // 8), max(40, h // 8)))
    if len(faces) == 0:
        print("[EyeDetect-CV] No face detected — using full-image eye scan")
        fx, fy, fw, fh = 0, 0, w, h
    else:
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    roi_gray = gray[fy:fy + fh, fx:fx + fw]
    eyes_cv  = eye_cas.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(max(10, fw // 10), max(10, fh // 10)))

    if len(eyes_cv) >= 2:
        # Convert to image-space centres, keep the two best
        centres = sorted(
            [(fx + ex + ew // 2, fy + ey + eh // 2, max(ew, eh) // 2)
             for (ex, ey, ew, eh) in eyes_cv],
            key=lambda c: c[0]            # sort left→right
        )[:2]
        print(f"[EyeDetect-CV] Haar: found {len(eyes_cv)} eye candidates, using 2")
    else:
        # Proportional heuristic — works reasonably for front-facing humanoids.
        # Eyes sit at ~40% from face top, 28% and 72% of face width.
        lx = fx + int(fw * 0.28);  rx = fx + int(fw * 0.72)
        cy = fy + int(fh * 0.40);  r  = max(4, int(fw * 0.07))
        centres = [(lx, cy, r), (rx, cy, r)]
        print(f"[EyeDetect-CV] Haar eye cascade found <2 eyes — using proportional fallback")

    def _iris_color(cx, cy, r_px):
        r_s = max(2, int(r_px * 0.55))
        patch = img_np[max(0, cy - r_s):min(h, cy + r_s),
                       max(0, cx - r_s):min(w, cx + r_s)].astype(float)
        if patch.size == 0:
            return (60, 40, 30)
        lum  = patch.mean(axis=2)
        mask = (lum > 35) & (lum < 190)
        if mask.sum() > 4:
            return tuple(int(v) for v in patch[mask].mean(axis=0).clip(0, 255))
        return (60, 40, 30)

    lx, ly, lr = centres[0]
    rx, ry, rr = centres[1]
    info = {
        'image_left_uv':      (lx / w, ly / h),
        'image_left_radius':  lr / w,
        'image_left_color':   _iris_color(lx, ly, lr),
        'image_right_uv':     (rx / w, ry / h),
        'image_right_radius': rr / w,
        'image_right_color':  _iris_color(rx, ry, rr),
    }
    print(f"[EyeDetect-CV] Left  eye UV={info['image_left_uv']}  color={info['image_left_color']}")
    print(f"[EyeDetect-CV] Right eye UV={info['image_right_uv']}  color={info['image_right_color']}")
    return info


def _mp_face_model():
    """Path to a MediaPipe Tasks face_landmarker bundle. Prefer an explicit env var, then any
    .task already shipped with another custom node -- LivePortraitKJ carries one -- so nothing has
    to be downloaded (this machine's Norton SSL interception makes that painful)."""
    import glob as _g
    _e = os.environ.get("FL_FACE_LANDMARKER")
    if _e and os.path.exists(_e):
        return _e
    _cn = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _pat in ("ComfyUI-LivePortraitKJ/media_pipe/mp_models/face_landmarker*.task",
                 "*/**/face_landmarker*.task"):
        for _m in _g.glob(os.path.join(_cn, _pat), recursive=True):
            return _m
    return None


def _head_crop_box(img_np, frac=0.22):
    """Square box around the head, from the foreground silhouette.

    The front photo is a full-body T-pose at 896px, so the face is only ~90px across and
    MediaPipe's detector finds NOTHING in it (measured: 0 faces). Cropping to the head makes
    the same image detect cleanly (478 landmarks, both irises)."""
    h, w = img_np.shape[:2]
    _bg = np.median(np.concatenate([img_np[:, :8].reshape(-1, 3),
                                    img_np[:, -8:].reshape(-1, 3)]), axis=0)
    _fg = (np.abs(img_np.astype(int) - _bg.astype(int)).sum(2) > 40)
    _rows = np.where(_fg.any(1))[0]
    if not len(_rows):
        return None
    _top = int(_rows[0])
    _band = _fg[_top:_top + int(0.25 * h)]
    _cols = np.where(_band.any(0))[0]
    if not len(_cols):
        return None
    _cx = int(0.5 * (_cols[0] + _cols[-1]))
    _side = int(frac * h)
    _x0 = max(0, _cx - _side // 2); _x1 = min(w, _x0 + _side)
    _y0 = max(0, _top - int(0.02 * h)); _y1 = min(h, _y0 + _side)
    return _x0, _y0, _x1, _y1


def _detect_eye_info_tasks(img_np):
    """MediaPipe >= 0.10.14 removed the `solutions` API entirely (0.10.35 here: mp.solutions is
    gone AND mediapipe.python is gone). Only the Tasks API remains, so use FaceLandmarker."""
    from PIL import Image as _I
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    _model = _mp_face_model()
    if not _model:
        print("[EyeDetect] Tasks API available but no face_landmarker .task model found "
              "(set FL_FACE_LANDMARKER)")
        return None
    _box = _head_crop_box(img_np)
    if _box is None:
        return None
    _x0, _y0, _x1, _y1 = _box
    h, w = img_np.shape[:2]
    _crop = _I.fromarray(img_np[_y0:_y1, _x0:_x1])
    _crop = _crop.resize((_crop.width * 2, _crop.height * 2), _I.LANCZOS)
    _opt = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=_model),
                                 running_mode=RunningMode.IMAGE, num_faces=1)
    with FaceLandmarker.create_from_options(_opt) as _lmk:
        _res = _lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(_crop)))
    if not _res.face_landmarks:
        print(f"[EyeDetect] Tasks: no face in the head crop {_x1-_x0}x{_y1-_y0}px")
        return None
    _L = _res.face_landmarks[0]
    if len(_L) < 478:
        print(f"[EyeDetect] Tasks: model returned {len(_L)} landmarks, no iris ring")
        return None
    _cw = _x1 - _x0; _ch = _y1 - _y0
    # Sample COLOUR from the upscaled crop, not the original: at 896px the iris is ~3px across,
    # which is far too few pixels to average a colour from. In the 2x crop it is ~13px.
    _ca = np.array(_crop)
    _lrc = (abs(_L[469].x - _L[471].x) + abs(_L[470].y - _L[472].y)) / 4.0
    _rrc = (abs(_L[474].x - _L[476].x) + abs(_L[475].y - _L[477].y)) / 4.0
    # crop-normalised -> whole-image normalised, so the rest of the pipeline is unchanged
    G = {i: ((_x0 + _L[i].x * _cw) / w, (_y0 + _L[i].y * _ch) / h) for i in range(468, 478)}
    # 469/471 are the iris ring's HORIZONTAL pair and 470/472 the VERTICAL pair, so the radius
    # comes from 469/471 in x and 470/472 in y. The original code paired them the other way round,
    # which differences two nearly-equal coordinates and yields a radius of ~0 (measured 0.0001 =
    # 0.09px). It never surfaced because MediaPipe never ran on this machine.
    l_r = (abs(G[469][0] - G[471][0]) + abs(G[470][1] - G[472][1])) / 4.0
    r_r = (abs(G[474][0] - G[476][0]) + abs(G[475][1] - G[477][1])) / 4.0
    info = {
        'image_left_uv':      (float(G[473][0]), float(G[473][1])),
        'image_left_radius':  float(r_r),
        'image_left_color':   _iris_color(_ca, _L[473].x, _L[473].y, _rrc),
        'image_right_uv':     (float(G[468][0]), float(G[468][1])),
        'image_right_radius': float(l_r),
        'image_right_color':  _iris_color(_ca, _L[468].x, _L[468].y, _lrc),
    }
    print(f"[EyeDetect] Tasks FaceLandmarker on {_cw}x{_ch} head crop: "
          f"left UV={info['image_left_uv']} r={info['image_left_radius']:.4f} "
          f"color={info['image_left_color']}", flush=True)
    print(f"[EyeDetect] Tasks: right UV={info['image_right_uv']} "
          f"r={info['image_right_radius']:.4f} color={info['image_right_color']}", flush=True)
    return info


def _iris_color(img_np, cx, cy, radius):
    """Median colour of the iris ANNULUS.

    A square patch at 0.55r is mostly pupil and lash shadow -- measured (77,66,50) on eyes that
    are visibly light grey-green. Take the ring between 0.35r and 0.90r instead, which is iris and
    nothing else, and use the MEDIAN so a stray artifact (this render has a red blob in one eye
    corner) cannot drag the result.
    """
    h, w = img_np.shape[:2]
    px, py = int(cx * w), int(cy * h)
    R = max(2, int(radius * w))
    y0, y1 = max(0, py - R), min(h, py + R + 1)
    x0, x1 = max(0, px - R), min(w, px + R + 1)
    patch = img_np[y0:y1, x0:x1].astype(float)
    if patch.size == 0:
        return (80, 55, 35)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / max(R, 1)
    lum = patch.mean(axis=2)
    mask = (rr >= 0.35) & (rr <= 0.90) & (lum > 35) & (lum < 220)
    if mask.sum() < 6:                       # tiny iris: fall back to the whole disc
        mask = (rr <= 1.0) & (lum > 35) & (lum < 220)
    if mask.sum() >= 3:
        return tuple(int(v) for v in np.median(patch[mask], axis=0).clip(0, 255))
    return (80, 55, 35)


def _detect_eye_info(front_image_pil):
    """
    Use MediaPipe FaceMesh (refine_landmarks=True) to locate both irises in the
    front-view image and sample their dominant colour.

    Returns a dict with normalised [0,1] UV positions (image convention, Y-down),
    iris radii as fraction of image width, and (R,G,B) colour tuples.
    Returns None if detection fails or MediaPipe is not installed.

    MediaPipe naming note:
      landmark 468 = subject's LEFT iris  → appears on the IMAGE RIGHT side
      landmark 473 = subject's RIGHT iris → appears on the IMAGE LEFT side
    We expose them as 'image_left_uv' / 'image_right_uv' to match world space.
    """
    img_np = np.array(front_image_pil.convert('RGB'))
    h, w = img_np.shape[:2]

    try:
        import mediapipe as mp

        # Locate FaceMesh class — API path changed across mediapipe versions:
        #   < 0.10.14 : mp.solutions.face_mesh.FaceMesh   (old public API)
        #   ≥ 0.10.14 : mp.solutions was removed from __init__.py but the
        #               underlying module is still importable directly.
        _FaceMesh = None
        try:
            _FaceMesh = mp.solutions.face_mesh.FaceMesh          # old API
        except AttributeError:
            pass
        if _FaceMesh is None:
            try:
                from mediapipe.python.solutions.face_mesh import FaceMesh as _FaceMesh
            except Exception:
                pass
        if _FaceMesh is None:
            # >= 0.10.14: legacy solutions gone -> Tasks API. And do NOT `return None` here as it
            # used to: that skipped the OpenCV fallback below entirely, so a missing legacy API
            # meant no eyes at all rather than degraded eyes.
            _ti = _detect_eye_info_tasks(img_np)
            if _ti is not None:
                return _ti
            print("[EyeDetect] Tasks API path failed — falling back to OpenCV Haar cascade")
            raise RuntimeError("no FaceMesh")

        with _FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(img_np)
            if not results.multi_face_landmarks:
                print("[EyeDetect] MediaPipe: no face detected in front image")
                return None
            lm = results.multi_face_landmarks[0].landmark

            # Iris radii from vertical + horizontal extent of the 5-point iris ring
            # axes were swapped here too -- see the Tasks path above
            l_r = (abs(lm[469].x - lm[471].x) + abs(lm[470].y - lm[472].y)) / 4.0
            r_r = (abs(lm[474].x - lm[476].x) + abs(lm[475].y - lm[477].y)) / 4.0

            def iris_color(cx, cy, radius):
                """Sample the iris zone, excluding dark pupil and bright specular."""
                px, py = int(cx * w), int(cy * h)
                r_px = max(2, int(radius * w * 0.55))  # inner 55% avoids limbal ring
                patch = img_np[
                    max(0, py - r_px):min(h, py + r_px),
                    max(0, px - r_px):min(w, px + r_px),
                ].astype(float)
                lum = patch.mean(axis=2)
                mask = (lum > 35) & (lum < 190)  # skip pupil (dark) and specular (bright)
                if mask.sum() > 8:
                    return tuple(int(v) for v in patch[mask].mean(axis=0).clip(0, 255))
                return (80, 55, 35)  # fallback dark brown

            info = {
                # subject's RIGHT iris → left side of image
                'image_left_uv':     (float(lm[473].x), float(lm[473].y)),
                'image_left_radius': float(r_r),
                'image_left_color':  iris_color(lm[473].x, lm[473].y, r_r),
                # subject's LEFT iris → right side of image
                'image_right_uv':     (float(lm[468].x), float(lm[468].y)),
                'image_right_radius': float(l_r),
                'image_right_color':  iris_color(lm[468].x, lm[468].y, l_r),
            }
            print(f"[EyeDetect] Left  iris UV={info['image_left_uv']}  "
                  f"r={info['image_left_radius']:.4f}  color={info['image_left_color']}")
            print(f"[EyeDetect] Right iris UV={info['image_right_uv']}  "
                  f"r={info['image_right_radius']:.4f}  color={info['image_right_color']}")
            return info

    except ImportError:
        print("[EyeDetect] MediaPipe not installed — falling back to OpenCV Haar cascade")
    except Exception as e:
        print(f"[EyeDetect] MediaPipe failed ({e}) — falling back to OpenCV Haar cascade")

    # ── OpenCV Haar cascade fallback (always available) ───────────────────────
    try:
        return _detect_eye_info_opencv(front_image_pil)
    except Exception as _cv_e:
        print(f"[EyeDetect] OpenCV fallback also failed: {_cv_e}")
        return None


def _generate_eye_texture(iris_color_rgb, size=512):
    """
    Generate a flat eye texture (PIL Image, RGB) with:
      - White sclera background
      - Coloured iris ring with subtle radial darkening toward center
      - Dark limbal ring at iris outer edge
      - Black pupil
      - Small off-center specular highlight
      - Light Gaussian blur for a natural look
    """
    from PIL import Image as PILImage, ImageDraw, ImageFilter

    img  = PILImage.new('RGB', (size, size), (248, 245, 240))  # off-white sclera
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r, g, b = int(iris_color_rgb[0]), int(iris_color_rgb[1]), int(iris_color_rgb[2])

    # --- Iris base (42 % of half-size) ---
    iris_r = int(size * 0.42)
    draw.ellipse([cx - iris_r, cy - iris_r, cx + iris_r, cy + iris_r], fill=(r, g, b))

    # --- Limbal ring (outer dark edge of iris, last 15 %) ---
    limbal_inner = int(iris_r * 0.87)
    dark = (max(0, r - 50), max(0, g - 50), max(0, b - 50))
    draw.ellipse([cx - iris_r,      cy - iris_r,      cx + iris_r,      cy + iris_r],      fill=dark)
    draw.ellipse([cx - limbal_inner, cy - limbal_inner, cx + limbal_inner, cy + limbal_inner], fill=(r, g, b))

    # --- Radial darkening: concentric rings toward center ---
    steps = 12
    for i in range(steps, 0, -1):
        frac   = i / steps                       # 1.0 at edge → 0 at center
        radius = int(limbal_inner * frac)
        shade  = 0.70 + 0.30 * frac             # 0.70 (dark center) → 1.0 (edge)
        rc = int(r * shade); gc = int(g * shade); bc = int(b * shade)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=(min(255, rc), min(255, gc), min(255, bc)))

    # --- Pupil (18 %) ---
    pupil_r = int(size * 0.18)
    draw.ellipse([cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r],
                 fill=(8, 8, 8))

    # --- Specular highlight (small white dot, upper-left of pupil) ---
    hl_r = int(size * 0.055)
    hl_x = cx - int(size * 0.09)
    hl_y = cy - int(size * 0.09)
    draw.ellipse([hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r],
                 fill=(255, 255, 255))

    # --- Soft blur for organic feel ---
    img = img.filter(ImageFilter.GaussianBlur(radius=max(1, size * 0.007)))
    return img


# ── Face-local delight helper (KAN-9) ─────────────────────────────────────────
# The global delight can't even out the small face's left/right shadow (the
# directional-light gradient there overlaps the face's own form-shading, and the
# face is tiny in a full-body shot). This normalizes the per-COLUMN brightness
# WITHIN the face hull only — directly cancelling the left-dim/right-bright ramp —
# while keeping vertical detail. Uses MediaPipe FaceLandmarker for the hull.

_FACE_LMK = None
_FACE_LMK_MODELS = [
    r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models\face_landmarker_v2_with_blendshapes.task",
]


def _get_face_lmk():
    global _FACE_LMK
    if _FACE_LMK is None:
        import os as _os
        from mediapipe.tasks import python as _mpp
        from mediapipe.tasks.python import vision as _vis
        _model = next((m for m in _FACE_LMK_MODELS if _os.path.exists(m)), None)
        if _model is None:
            raise FileNotFoundError("face_landmarker_v2_with_blendshapes.task not found")
        _FACE_LMK = _vis.FaceLandmarker.create_from_options(_vis.FaceLandmarkerOptions(
            base_options=_mpp.BaseOptions(model_asset_path=_model),
            num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))
    return _FACE_LMK


def _face_local_delight(rgb, fg):
    import cv2
    import mediapipe as mp
    H, W = rgb.shape[:2]
    rows = np.where(fg.any(axis=1))[0]
    if len(rows) < 8:
        return rgb
    ft, fb = int(rows[0]), int(rows[-1]); fh = fb - ft + 1
    cx0, cx1 = int(0.30 * W), int(0.70 * W)
    cy0, cy1 = ft, ft + int(0.25 * fh)
    UP = 3
    sub = rgb[cy0:cy1, cx0:cx1]
    if sub.size == 0:
        return rgb
    sub = cv2.resize(sub, (sub.shape[1] * UP, sub.shape[0] * UP), interpolation=cv2.INTER_LANCZOS4)
    res = _get_face_lmk().detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray((sub * 255).astype(np.uint8))))
    if not res.face_landmarks:
        print("[FlattenLight] face not detected — face-local delight skipped")
        return rgb
    H2, W2 = sub.shape[:2]
    pts = np.array([[cx0 + p.x * W2 / UP, cy0 + p.y * H2 / UP] for p in res.face_landmarks[0]])
    hull = cv2.convexHull(pts.astype(np.int32))
    fmask = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(fmask, hull, 255); fmask = fmask > 0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    fm = float(lum[fmask].mean())
    fw = max(8, int(pts[:, 0].max() - pts[:, 0].min()))
    colsum = (lum * fmask).sum(0); colcnt = fmask.sum(0).astype(np.float32)
    prof = np.where(colcnt > 3, colsum / np.maximum(colcnt, 1), fm)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), max(2.0, fw * 0.05)).ravel()
    gcol = np.clip(fm / np.clip(prof, 1e-3, None), 0.5, 2.2)
    corr = np.clip(rgb * gcol[None, :, None], 0.0, 1.0)
    soft = cv2.GaussianBlur(fmask.astype(np.float32), (0, 0), max(2.0, fw * 0.10))
    out = rgb * (1.0 - soft[..., None]) + corr * soft[..., None]
    print(f"[FlattenLight] face-local delight: evened L/R across face (width {fw}px)")
    return out.astype(np.float32)


# ── FlattenLight ──────────────────────────────────────────────────────────────

class FlattenLight:
    """
    Flattens baked lighting from SV3D frames before texture projection.

    The core operation is a gamma lift (pow < 1 brightens dark areas) to remove
    shadow baking.  The problem with a raw global gamma is that it treats dark
    eye irises, lips, and fine details the same as shadow penumbrae — brightening
    them to a pale gray.

    protect_dark_threshold (default 0.20):
        Pixels whose original luminance is BELOW this value are left completely
        unchanged (blend=0 = full original colour).  Pixels above 2× the threshold
        receive the full gamma+brightness+contrast correction (blend=1).  A smooth
        linear ramp in between prevents hard edges.

        Effect:
          • Dark irises / pupils (lum ≈ 0.05–0.20)  → fully protected, stay dark.
          • Shadow penumbrae on skin (lum ≈ 0.35–0.70) → fully corrected, lifted.
          • Mid-tone transitions (lum 0.20–0.40)       → partial correction, no jump.

        Increase toward 0.35 if light-coloured irises (blue/green, lum≈0.25) are
        still being washed out.  Decrease toward 0.10 to let more of the image
        be corrected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                  ("IMAGE",),
                "gamma":                  ("FLOAT", {"default": 0.55, "min": 0.1,  "max": 2.0, "step": 0.05, "display": "slider",
                                                     "tooltip": "Power curve applied to lift shadows (< 1 brightens, > 1 darkens). 0.55 is typical for SV3D."}),
                "brightness":             ("FLOAT", {"default": 0.05, "min": -0.5, "max": 0.5, "step": 0.01, "display": "slider"}),
                "contrast":               ("FLOAT", {"default": 0.85, "min": 0.0,  "max": 2.0, "step": 0.05, "display": "slider"}),
                "saturation":             ("FLOAT", {"default": 1.15, "min": 0.0,  "max": 3.0, "step": 0.05, "display": "slider"}),
                "protect_dark_threshold": ("FLOAT", {"default": 0.20, "min": 0.0,  "max": 0.6, "step": 0.01, "display": "slider",
                                                     "tooltip": "Luminance threshold below which pixels are NOT gamma-corrected. "
                                                                "Protects dark eye irises, pupils, and lips from being washed out. "
                                                                "Full correction resumes at 2× this value. "
                                                                "Raise toward 0.35 for light-coloured (blue/green) eyes."}),
                "enabled":                ("BOOLEAN", {"default": True,
                                                       "tooltip": "Uncheck to pass the image through completely unchanged. "
                                                                  "Disable for studio-lit SV3D frames that have no baked shadows."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/preprocessing"

    def execute(self, image, gamma, brightness, contrast, saturation, protect_dark_threshold=0.20, enabled=True):
        # PROPER DELIGHT (replaces the old gamma lift). Divides out the LOW-FREQUENCY
        # luminance — the baked directional lighting (lit cheek vs shadowed cheek, the
        # bright front of a limb vs its dim sides) — while keeping HIGH-FREQUENCY albedo
        # detail (eyes, lips, skin). A gamma lift only brightened darks globally; it
        # could not even out a directional gradient, which is why one face half stayed
        # dark and got flattened downstream. This removes the gradient at its source so
        # the albedo is evenly lit (what a texture should be).
        #   gamma / brightness / contrast / protect_dark_threshold: legacy, ignored.
        #   saturation: kept as an optional post-boost.
        # Automatic: the blur radius scales with the image; no per-image tuning.
        if not enabled:
            return (image,)
        import cv2
        dev = image.device
        frames = []
        for i in range(image.shape[0]):
            arr = image[i, ..., :3].clamp(0.0, 1.0).detach().cpu().numpy()
            H, W = arr.shape[:2]
            lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            fg = arr.sum(-1) > 0.04                       # PreProcessImage gives black bg
            if int(fg.sum()) < 100:
                frames.append(image[i]); continue
            m = float(lum[fg].mean())
            # low-frequency lighting (fill bg with fg-mean so the blur doesn't pull in bg)
            lf = np.where(fg, lum, m).astype(np.float32)
            sigma = max(8.0, 0.07 * max(H, W))
            low = cv2.GaussianBlur(lf, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gain = np.clip(m / np.clip(low, 1e-3, None), 0.6, 2.2)   # clamp = dark-feature guard
            rgb = np.clip(arr * gain[..., None], 0.0, 1.0)
            if saturation != 1.0:
                l2 = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
                rgb = np.clip(l2 + saturation * (rgb - l2), 0.0, 1.0)
            # FACE-LOCAL delight: the global pass can't reach the small face, so even the
            # left/right facial lighting directly (per-column normalise within the face
            # hull). Falls back silently to global-only if detection/mediapipe is missing.
            try:
                rgb = _face_local_delight(rgb, fg)
            except Exception as _fe:
                print(f"[FlattenLight] face-local delight skipped: {_fe!r}")
            rgb[~fg] = arr[~fg]                           # preserve original background
            t = torch.from_numpy(rgb).to(dev).float()
            if image.shape[-1] > 3:
                t = torch.cat([t, image[i, ..., 3:]], dim=-1)
            frames.append(t)
        return (torch.stack(frames, dim=0),)


# ── GLB binary UV extractor (reliable fallback) ───────────────────────────────

def _extract_uv_from_glb_binary(glb_path):
    try:
        with open(glb_path, 'rb') as f:
            data = f.read()
        if struct.unpack_from('<I', data, 0)[0] != 0x46546C67:
            return None
        jlen = struct.unpack_from('<I', data, 12)[0]
        if struct.unpack_from('<I', data, 16)[0] != 0x4E4F534A:
            return None
        gltf = json.loads(data[20:20 + jlen].decode('utf-8'))
        boff = 20 + jlen
        if boff + 8 > len(data):
            return None
        blen = struct.unpack_from('<I', data, boff)[0]
        bin_data = data[boff + 8: boff + 8 + blen]
        for mesh in gltf.get('meshes', []):
            for prim in mesh.get('primitives', []):
                idx = prim.get('attributes', {}).get('TEXCOORD_0')
                if idx is None:
                    continue
                acc = gltf['accessors'][idx]
                bv  = gltf['bufferViews'][acc['bufferView']]
                off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
                n   = acc['count']
                uv  = np.frombuffer(bin_data[off: off + n * 8], dtype=np.float32).reshape(-1, 2).copy()
                print(f"[BlenderSmartUV] Binary UV extracted: {uv.shape}")
                return uv
        return None
    except Exception as e:
        print(f"[BlenderSmartUV] Binary UV extraction failed: {e}")
        return None


# ── Trellis2BlenderSmartUV ────────────────────────────────────────────────────

_STAGE = {"t": None, "label": None}


def _stage(label=None):
    """Print how long the previous named stage took. Call with None to flush the last one.

    The bake is ~15 min across 7 Blender/Python subprocesses and nothing said where the time
    went, so every speed decision was guesswork. This makes the cost of each stage explicit.
    """
    import time as _t
    _now = _t.perf_counter()
    if _STAGE["t"] is not None:
        print(f"[stage] {_STAGE['label']} took {_now - _STAGE['t']:.1f}s", flush=True)
    _STAGE["t"] = None if label is None else _now
    _STAGE["label"] = label


class Trellis2BlenderSmartUV:
    # ── Re-run when one of the external Blender scripts changes ──────────────────
    # ComfyUI caches by node INPUTS, so an edit to gen_seams.py / pelvis_graft.py is
    # invisible to it: the cached output is reused and the fix silently never runs.
    # A restart is the only alternative, and it wipes the whole cache -- forcing the
    # ~7 min geometry chain and ~3 min Qwen pass to re-run for a one-line script edit.
    # Hashing the scripts' mtimes invalidates exactly this node and its downstream.
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import os as _o
        _uv = r"C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer"
        _hg = r"C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-HandGraft"
        _paths = [_o.path.join(_uv, f) for f in ("gen_seams.py", "_gs_facebridge.py",
                                                 "_gs_hairline_bridge.py", "export_glb.py")]
        _paths += [_o.path.join(_hg, f) for f in ("pelvis_graft.py", "genital_texture.py",
                                                  "hand_graft_post.py", "hand_replace.py")]
        # the donor ASSETS matter as much as the code: repairing the .npz without touching a .py
        # would otherwise serve a cached graft and look like the fix did nothing
        _paths += [_o.path.join(_hg, "assets", f) for f in ("donor_pelvis_v2.npz",
                                                            "genital_donor_zones.npy",
                                                            "donor_geo_R.npz", "donor_geo_L.npz")]
        try:
            return str([(_o.path.basename(p), _o.path.getmtime(p))
                        for p in _paths if _o.path.exists(p)])
        except Exception:
            return float("nan")

    """
    Blender 5.1 headless pipeline:
      1. Import GLB
      2. Join all objects → single mesh
      3. Blender Decimate (edge-collapse, preserves connectivity — replaces Meshlib)
      4. Merge by Distance  (welds coincident verts; increase if mesh is still fragmented)
      5. Diagnostic: count connected components before and after merge
      6. Zone-based Smart UV Project:
           head / torso / arm_pos / arm_neg / leg_pos / leg_neg
           Smart UV on each isolated zone → a few large islands per body part
      7. Pack all islands into [0,1] UV square
      8. Export GLB

    Wire: Trellis2MeshWithVoxelToTrimesh → this node → Trellis2MultiViewTexturing
    Remove Trellis2SimplifyMesh from the workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "default": 150000, "min": 10000, "max": 1000000, "step": 10000,
                    "tooltip": "Target face count after Blender Decimate (replaces Trellis2SimplifyMesh)."}),
                "uv_method": (["xatlas", "template_transfer", "smart_uv", "gen_seams"], {
                    "default": "xatlas",
                    "tooltip": "xatlas: best for disconnected meshes (recommended). "
                               "template_transfer: copies UV layout from a Rodin/reference mesh via Blender Data Transfer. "
                               "smart_uv: zone-based Blender Smart UV Project (legacy). "
                               "gen_seams: anatomical ring+Dijkstra seams via gen_seams.py (best for humanoids)."}),
                "angle_limit": ("FLOAT", {
                    "default": 80.0, "min": 1.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Smart UV angle limit (smart_uv mode only). Higher = fewer, larger islands."}),
                "island_margin": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 0.05, "step": 0.001,
                    "tooltip": "UV island packing margin. 0.005 is safe for 4096px."}),
                "merge_distance": ("FLOAT", {
                    "default": 0.0005, "min": 0.0, "max": 0.1, "step": 0.0001,
                    "tooltip": "Merge-by-Distance threshold (mesh units). "
                               "Welds near-coincident verts from upstream remesh. "
                               "Keep small (0.0005) — 0.01 destroys the mesh at this scale."}),
            },
            "optional": {
                "template_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a Rodin / Tripo3D FBX or OBJ used as UV template "
                               "(required when uv_method = template_transfer). "
                               "Supports FBX, OBJ, GLB/GLTF. Forward or back slashes both work."}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "obj_path")
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    def execute(self, trimesh, target_faces, uv_method, angle_limit, island_margin, merge_distance, template_path=""):
        import trimesh as trimesh_lib
        import trimesh.visual.texture as tex_vis

        # ── MESH SANITY: FAIL FAST ────────────────────────────────────────────────────
        # The generated mesh is occasionally degenerate, and every stage after this point is
        # expensive: pelvis graft (~3 min), gen_seams (~1 min), genital texturing, export, then
        # the whole projection. Twice today a broken mesh burned ~20 minutes before anything
        # noticed. Both were obvious the instant the mesh arrived:
        #   * flat pancake: extents X 1.150, Y 0.028, Z 0.016 -- smallest 2.4% of largest
        #   * blown-up:     wrists at -3.35e18, i.e. non-finite-scale coordinates
        # The test is a physical statement, not a tuned threshold: a humanoid body cannot have a
        # bounding-box extent smaller than 5% of its largest (a real one measures ~19%: depth 0.21
        # against height 1.09). Everything is relative to the mesh's own size, so it holds for any
        # character at any scale. TRELLIS_MESH_SANITY=0 disables.
        try:
            import numpy as _np_ms, os as _os_ms
            if _os_ms.environ.get("TRELLIS_MESH_SANITY", "1").strip() != "0":
                _mv = _np_ms.asarray(trimesh.vertices, dtype=_np_ms.float64)
                if _mv.size:
                    _finite = bool(_np_ms.isfinite(_mv).all())
                    _ext = _mv.max(axis=0) - _mv.min(axis=0) if _finite else None
                    _mx = float(_ext.max()) if _finite else 0.0
                    _mn = float(_ext.min()) if _finite else 0.0
                    _ratio = (_mn / _mx) if _mx > 1e-12 else 0.0
                    _bad = (not _finite) or _mx <= 1e-9 or _ratio < 0.05
                    print("[MeshSanity] extents "
                          + (str([round(float(e), 4) for e in _ext]) if _finite else "NON-FINITE")
                          + f"  min/max={_ratio:.4f}  faces={len(trimesh.faces):,}"
                          + ("  -- OK" if not _bad else "  -- DEGENERATE"))
                    if _bad:
                        raise RuntimeError(
                            "[MeshSanity] ABORTING: the generated mesh is degenerate"
                            + ("  (non-finite vertex coordinates)" if not _finite else
                               f"  (smallest bounding-box extent is {100*_ratio:.1f}% of the largest;"
                               f" a human body is ~19%). extents="
                               f"{[round(float(e), 4) for e in _ext]}")
                            + ". Stopped before the graft, gen_seams and projection rather than"
                            + " spending ~15 more minutes on it. Re-run to draw a new mesh, or set"
                            + " TRELLIS_MESH_SANITY=0 to force it through.")
        except RuntimeError:
            raise
        except Exception as _mse:
            print(f"[MeshSanity] check skipped: {_mse!r}")

        mesh = copy.deepcopy(trimesh)

        tmp_dir = tempfile.mkdtemp()
        in_glb  = os.path.join(tmp_dir, "mesh_in.glb")
        out_glb = os.path.join(tmp_dir, "mesh_out.glb")   # kept for xatlas path
        out_obj = os.path.join(tmp_dir, "mesh_out.obj")   # used for template_transfer / smart_uv

        scene = trimesh_lib.scene.Scene()
        scene.add_geometry(mesh)
        scene.export(in_glb)
        print(f"[BlenderSmartUV] Input: {len(mesh.vertices)} verts, {len(mesh.faces)} faces → target {target_faces}")

        # Normalise template path: forward slashes so the f-string stays clean
        template_path = (template_path or "").strip().replace("\\", "/")
        do_template_transfer = (uv_method == "template_transfer" and bool(template_path))
        do_smart_uv          = (uv_method == "smart_uv")
        do_gen_seams         = (uv_method == "gen_seams")
        do_blender_uv        = do_template_transfer or do_smart_uv

        # GREEN-BORDER high-res save REMOVED (2026-09-04): it existed only to feed the
        # hairline bridge (pass D1), which is retired now that we generate real back/side
        # views. Nothing reads _gs_hires.npz any more.

        if do_template_transfer:
            print(f"[BlenderSmartUV] Template transfer mode — source: {template_path}")

        blender_script = f"""
import bpy, sys, bmesh, collections, mathutils

# ── Clear scene ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import ────────────────────────────────────────────────────────────────────
bpy.ops.import_scene.gltf(filepath=r'{in_glb}')
mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objects:
    print("ERROR: no mesh found"); sys.exit(1)

bpy.ops.object.select_all(action='DESELECT')
for o in mesh_objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.active_object
print(f"[BlenderSmartUV] Imported: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")

# ── Remesh / Decimate ─────────────────────────────────────────────────────────
current_faces = len(obj.data.polygons)
import os as _os_tgt   # local: _os is not imported until much later in this script
# GS_TARGET_SCALE lets the mesh-prep retry loop ask for a slightly different density,
# so a retry produces a genuinely different mesh instead of repeating the same failure.
target = int(round({target_faces} * float(_os_tgt.environ.get("GS_TARGET_SCALE", "1.0"))))
print(f"[BlenderSmartUV] face target {{target}} (scale {{_os_tgt.environ.get('GS_TARGET_SCALE','1.0')}})")
if {do_gen_seams}:   # VOXEL REMESH sized straight to `target` -> smooth, EVEN QUAD body. This is the exact
    # pre-2026-06-17-18:12 behavior (last-good run = 157k QUADS, smooth). Turning it off on 06-17 (and my
    # later fine-voxel+decimate) produced a TRIANGLE mesh (raw decimated Dual-Contouring / collapse slivers)
    # = the blocky/streaky body. The direct target-sized voxel keeps quads = smooth. DO NOT decimate after
    # this (collapse -> sliver tris -> blocky) and DO NOT disable it (raw DC tris -> blocky). Keep the quads.
    import math as _math
    _bb = obj.bound_box
    _dims = [max(v[i] for v in _bb) - min(v[i] for v in _bb) for i in range(3)]
    _diag = _math.sqrt(sum(d * d for d in _dims))
    _vox = max(0.002, _diag * 0.006)
    print(f"[BlenderSmartUV] Voxel remesh start — diag={{_diag:.3f}}, initial voxel={{_vox:.5f}}, target={{target}}")
    # The old rule was  _vox *= (nf/target) ** 0.5,  i.e. a fixed-point iteration assuming face
    # count scales as voxel**-2. It does not: measured on a real bake, 0.00337 -> 77k and
    # 0.00239 -> 298k, which is voxel**-3.9. With the wrong exponent every step overshoots, so the
    # search ping-ponged between 70k and 320k for eight attempts, never tried the interval where
    # 150k actually lives, and then APPLIED THE LAST attempt (320k = 2.1x budget) instead of the
    # closest one. That mesh unwraps into broken island scales (one face owning 2.3% of the atlas)
    # and the bake is unusable. Now: bracket the target and bisect, measure the exponent from the
    # last two samples instead of assuming it, keep the BEST attempt, and refuse to continue if it
    # never converges.
    _samples = []
    _best = None          # (log-error, voxel, faces)
    _too_big = None       # a voxel that produced TOO FEW faces
    _too_small = None     # a voxel that produced TOO MANY faces
    _last_mod = None
    for _attempt in range(10):
        if _last_mod is not None:
            obj.modifiers.remove(_last_mod)
            _last_mod = None
        _mod = obj.modifiers.new(name='VoxRemesh', type='REMESH')
        _mod.mode = 'VOXEL'
        _mod.voxel_size = _vox
        _mod.adaptivity = 0.0
        _dg = bpy.context.evaluated_depsgraph_get()
        _ev = obj.evaluated_get(_dg)
        _nf = len(_ev.data.polygons)
        _samples.append((_vox, _nf))
        _err = abs(_math.log(max(_nf, 1) / float(target)))
        if _best is None or _err < _best[0]:
            _best = (_err, _vox, _nf)
        print(f"[BlenderSmartUV]   attempt {{_attempt}}: voxel={{_vox:.5f}} -> {{_nf}} faces")
        _last_mod = _mod
        if abs(_nf - target) < target * 0.10:
            break
        if _nf < target:
            if _too_big is None or _vox < _too_big:
                _too_big = _vox
        else:
            if _too_small is None or _vox > _too_small:
                _too_small = _vox
        if _too_big is not None and _too_small is not None:
            _nv = _math.sqrt(_too_big * _too_small)      # geometric bisection inside the bracket
        else:
            _p = 2.0                                     # measure the exponent, do not assume it
            if len(_samples) >= 2:
                (_v1, _n1), (_v2, _n2) = _samples[-2], _samples[-1]
                if _v1 > 0 and _v2 > 0 and _n1 > 0 and _n2 > 0 and abs(_v2 - _v1) > 1e-12:
                    try:
                        _p = abs(_math.log(float(_n2) / _n1) / _math.log(_v2 / _v1))
                    except Exception:
                        _p = 2.0
                _p = min(6.0, max(1.0, _p))
            _nv = _vox * (float(_nf) / target) ** (1.0 / _p)
        _nv = max(0.001, min(0.05, _nv))
        if abs(_nv - _vox) <= 1e-9:
            break
        _vox = _nv
    for _m in [_m2 for _m2 in obj.modifiers if _m2.name.startswith('VoxRemesh')]:
        obj.modifiers.remove(_m)
    if not (target * 0.55 <= _best[2] <= target * 1.60):
        raise RuntimeError(
            "[BlenderSmartUV] VOXEL REMESH DID NOT CONVERGE: best attempt {{}} faces vs target "
            "{{}}. Tried {{}}. Aborting HERE rather than unwrapping and texturing a mesh of the "
            "wrong density - that produces broken UV island scales and an unusable bake."
            .format(_best[2], target,
                    ", ".join("%.5f->%d" % (_v, _n) for _v, _n in _samples)))
    _mod = obj.modifiers.new(name='VoxRemesh', type='REMESH')
    _mod.mode = 'VOXEL'
    _mod.voxel_size = _best[1]
    _mod.adaptivity = 0.0
    print(f"[BlenderSmartUV] applying BEST voxel={{_best[1]:.5f}} (~{{_best[2]}} faces, target {{target}})")
    bpy.ops.object.modifier_apply(modifier='VoxRemesh')
    print(f"[BlenderSmartUV] After voxel remesh: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")
elif current_faces > target:
    ratio = max(0.001, target / current_faces)
    dec = obj.modifiers.new(name='Decimate', type='DECIMATE')
    dec.decimate_type = 'COLLAPSE'
    dec.ratio = ratio
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    print(f"[BlenderSmartUV] After decimate: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")

# ── Helper: count connected components via BFS ────────────────────────────────
def count_components(me):
    bm2 = bmesh.new()
    bm2.from_mesh(me)
    bm2.verts.ensure_lookup_table()
    visited = set()
    n = 0
    for start in bm2.verts:
        if start.index in visited:
            continue
        n += 1
        q = collections.deque([start])
        while q:
            v = q.popleft()
            if v.index in visited:
                continue
            visited.add(v.index)
            for e in v.link_edges:
                ov = e.other_vert(v)
                if ov.index not in visited:
                    q.append(ov)
    bm2.free()
    return n

comp_before = count_components(obj.data)
print(f"[BlenderSmartUV] Components before merge: {{comp_before}}")

# ── Merge by Distance ─────────────────────────────────────────────────────────
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
vb = len(obj.data.vertices)
bpy.ops.mesh.remove_doubles(threshold={merge_distance})
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
va = len(obj.data.vertices)
print(f"[BlenderSmartUV] Merge by distance (thr={merge_distance:.4f}): {{vb}} -> {{va}} verts (welded {{vb-va}})")
print("[BlenderSmartUV] Normals recalculated (outward-facing)")

comp_after = count_components(obj.data)
print(f"[BlenderSmartUV] Components after merge: {{comp_after}}")
if comp_after > 50:
    print(f"[BlenderSmartUV] WARNING: {{comp_after}} components remain - increase merge_distance!")
elif comp_after > 1:
    print(f"[BlenderSmartUV] Note: {{comp_after}} separate islands (e.g. eyes, teeth) - usually OK")

# ── Delete small disconnected fragments ───────────────────────────────────────
# Decimation creates tiny floating pieces (< 0.1% of faces) that break the
# V:E:F = 1:3:2 ratio (their boundary edges inflate edge count) and show as
# exploded colored splinters in material view.
# Keep any component with at least min_component_faces faces.
total_faces = len(obj.data.polygons)
min_component_faces = max(50, total_faces // 1000)   # 0.1% floor, never below 50
print(f"[BlenderSmartUV] Removing fragments < {{min_component_faces}} faces (0.1% of {{total_faces}})...")

bm_clean = bmesh.new()
bm_clean.from_mesh(obj.data)
bm_clean.faces.ensure_lookup_table()

visited_f = set()
components_f = []
for start_face in bm_clean.faces:
    if start_face.index in visited_f:
        continue
    comp = set()
    stack = [start_face]
    while stack:
        f = stack.pop()
        if f.index in visited_f:
            continue
        visited_f.add(f.index)
        comp.add(f.index)
        for edge in f.edges:
            for adj in edge.link_faces:
                if adj.index not in visited_f:
                    stack.append(adj)
    components_f.append(comp)

# KEEP ONLY THE LARGEST SHELL.
# The 0.1% floor above only catches decimation splinters. Multi-view TRELLIS can also
# hallucinate a whole DETACHED limb where the views disagree -- measured on the first
# 4-view bake: a second component of 8,254 verts (4.9% of the mesh), a thin object
# hanging between the legs ("the third leg"). It sits INSIDE the body's bounding box,
# so no spatial test finds it; only "is not the main shell" does. The character body is
# one closed shell, so anything else is an artefact.
components_f.sort(key=len, reverse=True)
faces_to_kill = {{fi for comp in components_f[1:] for fi in comp}}
for _c in components_f[1:]:
    _pct = 100.0 * len(_c) / max(total_faces, 1)
    print(f"[BlenderSmartUV] dropping detached component: {{len(_c)}} faces ({{_pct:.2f}}% of mesh)")
    if _pct > 1.0:
        print(f"[BlenderSmartUV] *** LARGE detached piece removed ({{_pct:.2f}}%) -- "
              f"confirm it was not real geometry ***")
if faces_to_kill:
    geom = [f for f in bm_clean.faces if f.index in faces_to_kill]
    bmesh.ops.delete(bm_clean, geom=geom, context='FACES')
    bm_clean.to_mesh(obj.data)
    print(f"[BlenderSmartUV] Removed {{len(faces_to_kill)}} fragment faces "
          f"({{len(components_f) - sum(1 for c in components_f if len(c) >= min_component_faces)}} components deleted)")
else:
    print("[BlenderSmartUV] No fragments to remove")
bm_clean.free()
obj.data.update()

# ── UV Template Transfer (transfer UV layout from reference mesh) ─────────────
mesh_data = obj.data
if {do_template_transfer}:
    template_path_str = r'{template_path}'
    print(f"[BlenderSmartUV] Loading UV template: {{template_path_str}}")
    ext = template_path_str.lower().rsplit('.', 1)[-1]
    pre_import_names = set(o.name for o in bpy.context.scene.objects)
    template_obj = None
    try:
        if ext == 'fbx':
            bpy.ops.import_scene.fbx(filepath=template_path_str, use_custom_normals=True)
        elif ext == 'obj':
            bpy.ops.wm.obj_import(filepath=template_path_str)
        elif ext in ('glb', 'gltf'):
            bpy.ops.import_scene.gltf(filepath=template_path_str)
        else:
            raise ValueError(f"Unknown extension '{{ext}}'")
        new_mesh_objs = [o for o in bpy.context.scene.objects
                         if o.name not in pre_import_names and o.type == 'MESH']
        if not new_mesh_objs:
            raise RuntimeError("No mesh objects found in template file")
        # Join all imported template objects into one
        bpy.ops.object.select_all(action='DESELECT')
        for to in new_mesh_objs:
            to.select_set(True)
        bpy.context.view_layer.objects.active = new_mesh_objs[0]
        if len(new_mesh_objs) > 1:
            bpy.ops.object.join()
        template_obj = bpy.context.active_object
        print(f"[BlenderSmartUV] Template imported: {{len(template_obj.data.vertices)}} verts, "
              f"{{len(template_obj.data.polygons)}} faces, "
              f"UV layers: {{[l.name for l in template_obj.data.uv_layers]}}")
    except Exception as e:
        print(f"[BlenderSmartUV] WARNING: Template import failed: {{e}}")
        template_obj = None

    if template_obj is not None and template_obj.data.uv_layers:
        # ── Align template bounding box to Trellis mesh ───────────────────
        def get_world_bounds(o):
            wv = [o.matrix_world @ v.co for v in o.data.vertices]
            return (mathutils.Vector((min(v.x for v in wv), min(v.y for v in wv), min(v.z for v in wv))),
                    mathutils.Vector((max(v.x for v in wv), max(v.y for v in wv), max(v.z for v in wv))))

        t_min, t_max = get_world_bounds(obj)
        r_min, r_max = get_world_bounds(template_obj)
        t_size   = t_max - t_min
        r_size   = r_max - r_min
        t_center = (t_min + t_max) / 2

        # Scale by height (Z) so the bodies overlap spatially
        scale = (t_size.z / r_size.z) if r_size.z > 0.0 else 1.0
        bpy.ops.object.select_all(action='DESELECT')
        template_obj.select_set(True)
        bpy.context.view_layer.objects.active = template_obj
        template_obj.scale = mathutils.Vector((scale, scale, scale))
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)

        # Re-center after scale
        r_min2, r_max2 = get_world_bounds(template_obj)
        r_center2 = (r_min2 + r_max2) / 2
        template_obj.location = template_obj.location + (t_center - r_center2)
        bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
        print(f"[BlenderSmartUV] Template aligned: scale={{scale:.4f}}, "
              f"center={{tuple(round(v,4) for v in t_center)}}")

        # ── Seam Transfer + Fresh UV Unwrap ──────────────────────────────────────
        # Instead of copying UV coordinates (which always scrambles at seam edges),
        # we transfer the SEAM TOPOLOGY from Rodin to Trellis:
        #   1. Build Trellis→Rodin vertex map (hemisphere-filtered KD-tree)
        #   2. Detect which Rodin edges are UV seams (UV discontinuity between adjacent faces)
        #   3. Mark the corresponding Trellis edges as seams
        #   4. Run Blender's UV Unwrap — produces clean islands, no seam-crossing possible

        def get_normalized_verts(mesh_obj):
            wv = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
            xs = [v.x for v in wv]; ys = [v.y for v in wv]; zs = [v.z for v in wv]
            xr = max(max(xs)-min(xs), 1e-6)
            yr = max(max(ys)-min(ys), 1e-6)
            zr = max(max(zs)-min(zs), 1e-6)
            xlo, ylo, zlo = min(xs), min(ys), min(zs)
            return [mathutils.Vector(((wv[i].x-xlo)/xr,
                                      (wv[i].y-ylo)/yr,
                                      (wv[i].z-zlo)/zr))
                    for i in range(len(wv))]

        def get_vert_normals(mesh_obj):
            ns = {{}}
            for poly in mesh_obj.data.polygons:
                wn = mesh_obj.matrix_world.to_3x3() @ poly.normal
                wn.normalize()
                for vi in poly.vertices:
                    if vi not in ns:
                        ns[vi] = mathutils.Vector((0,0,0))
                    ns[vi] += wn
            for vi in ns:
                ns[vi].normalize()
            return ns

        r_norm_v  = get_normalized_verts(template_obj)
        t_norm_v  = get_normalized_verts(obj)
        r_normals = get_vert_normals(template_obj)
        t_normals = get_vert_normals(obj)

        # ── 1. Build Trellis→Rodin vertex map (hemisphere-filtered) ──────────
        kd = mathutils.kdtree.KDTree(len(r_norm_v))
        for i, pos in enumerate(r_norm_v):
            kd.insert(pos, i)
        kd.balance()

        # K=100: wider search so same-hemisphere candidates are almost always found
        K = 100
        n_fallback = 0
        vert_to_rodin = {{}}   # t_vi → r_vi
        for vi, tp in enumerate(t_norm_v):
            candidates = kd.find_n(tp, K)
            t_n = t_normals.get(vi, mathutils.Vector((0,0,0)))
            same_hemi = [(r_vi, pd) for _, r_vi, pd in candidates
                         if t_n.dot(r_normals.get(r_vi, mathutils.Vector((0,0,0)))) >= 0.0]
            if same_hemi:
                best_r_vi = min(same_hemi, key=lambda x: x[1])[0]
            else:
                n_fallback += 1
                best_r_vi = min(candidates, key=lambda c: c[2])[1]
            vert_to_rodin[vi] = best_r_vi
        print(f"[BlenderSmartUV] Vertex matching: {{len(vert_to_rodin)}} verts, {{n_fallback}} fallbacks")

        # ── 2. Detect Rodin seam edges (UV discontinuity — exact) ────────────
        bm_r = bmesh.new()
        bm_r.from_mesh(template_obj.data)
        bm_r.edges.ensure_lookup_table()
        bm_r.verts.ensure_lookup_table()
        uv_layer_r    = bm_r.loops.layers.uv.active
        r_sharp_verts = set()
        r_seam_verts  = set()   # Rodin vertex indices adjacent to any seam edge
        r_seam_ec     = []      # (world_x, world_z) of each Rodin seam edge centroid
        r_seam_ec_3d  = []      # full world Vector3 of each Rodin seam edge centroid
        _r_mat_w      = template_obj.matrix_world
        n_r_seams     = 0
        for edge in bm_r.edges:
            v0i = edge.verts[0].index
            v1i = edge.verts[1].index
            is_seam = (len(edge.link_faces) < 2)
            if not is_seam and uv_layer_r:
                fa, fb = edge.link_faces[0], edge.link_faces[1]
                v_set = {{v0i, v1i}}
                uv_a = {{lp.vert.index: lp[uv_layer_r].uv.to_tuple(4)
                         for lp in fa.loops if lp.vert.index in v_set}}
                uv_b = {{lp.vert.index: lp[uv_layer_r].uv.to_tuple(4)
                         for lp in fb.loops if lp.vert.index in v_set}}
                for vi in v_set:
                    if vi in uv_a and vi in uv_b and uv_a[vi] != uv_b[vi]:
                        is_seam = True
                        break
            if is_seam:
                n_r_seams += 1
                r_seam_verts.add(v0i)
                r_seam_verts.add(v1i)
                _rv0w = _r_mat_w @ bm_r.verts[v0i].co
                _rv1w = _r_mat_w @ bm_r.verts[v1i].co
                r_seam_ec.append(((_rv0w.x+_rv1w.x)*0.5, (_rv0w.z+_rv1w.z)*0.5))
                r_seam_ec_3d.append((_rv0w + _rv1w) * 0.5)
            if not edge.smooth:
                r_sharp_verts.add(v0i)
                r_sharp_verts.add(v1i)
        bm_r.free()
        print(f"[BlenderSmartUV] Rodin seam edges: {{n_r_seams}}")
        print(f"[BlenderSmartUV] Rodin seam verts: {{len(r_seam_verts)}}")
        print(f"[BlenderSmartUV] Rodin sharp edge verts: {{len(r_sharp_verts)}}")

        # Build per-face UV lookup for Rodin seam vertices.
        # A seam vertex in Rodin has TWO UV values — one per adjacent island.
        # We store (world_face_normal, uv) pairs so we can later pick the UV
        # from the Rodin face whose normal best matches the current Trellis face.
        # This avoids the face-centroid UV sampling problem at seam boundaries.
        r_uv_data   = template_obj.data.uv_layers.active.data
        r_polys_sv  = template_obj.data.polygons
        r_loops_sv  = template_obj.data.loops
        r_nmat_sv   = template_obj.matrix_world.to_3x3()
        r_sv_uv     = {{}}   # r_vi -> [(world_normal, uv_Vector2)]
        for poly in r_polys_sv:
            wn = (r_nmat_sv @ poly.normal)
            if wn.length > 1e-6: wn.normalize()
            for li in poly.loop_indices:
                vi = r_loops_sv[li].vertex_index
                if vi not in r_seam_verts: continue
                uv = r_uv_data[li].uv.copy()
                if vi not in r_sv_uv: r_sv_uv[vi] = []
                r_sv_uv[vi].append((wn.copy(), uv))
        print(f"[BlenderSmartUV] Rodin seam-vert UV table: {{len(r_sv_uv)}} entries")

        # ── 3. Transfer Rodin UV to Trellis (barycentric — for seam detection) ─
        # Barycentric within nearest Rodin face gives smooth UV within islands.
        # Adjacent Trellis faces near a seam boundary that land on opposite sides
        # of the Rodin seam will have large UV jump → detected as seam in step 4.
        def _bary_uv_in_face(p_w, fvws, fuvs):
            # Barycentric UV at world-space point p_w inside a tri or quad face.
            def _tri_bary(ia, ib, ic):
                d0 = fvws[ib] - fvws[ia]
                d1 = fvws[ic] - fvws[ia]
                d2 = p_w      - fvws[ia]
                d00 = d0.dot(d0); d01 = d0.dot(d1); d11 = d1.dot(d1)
                d20 = d2.dot(d0); d21 = d2.dot(d1)
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-12:
                    return None
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                return (1.0 - v - w, v, w)
            n = len(fvws)
            if n == 3:
                bary = _tri_bary(0, 1, 2)
                if bary is None:
                    return (fuvs[0] + fuvs[1] + fuvs[2]) * (1.0 / 3)
                u, v, w = bary
                return u * fuvs[0] + v * fuvs[1] + w * fuvs[2]
            bary = _tri_bary(0, 1, 2)
            if bary is not None and bary[0] >= -0.01 and bary[1] >= -0.01 and bary[2] >= -0.01:
                u, v, w = bary
                return u * fuvs[0] + v * fuvs[1] + w * fuvs[2]
            bary = _tri_bary(0, 2, 3)
            if bary is None:
                return (fuvs[0] + fuvs[2] + fuvs[3]) * (1.0 / 3)
            u, v, w = bary
            return u * fuvs[0] + v * fuvs[2] + w * fuvs[3]

        r_polys_d    = template_obj.data.polygons
        r_uv_src     = template_obj.data.uv_layers.active.data
        r_loops_d_uv = template_obj.data.loops
        r_verts_d_uv = template_obj.data.vertices
        r_mat_uv     = template_obj.matrix_world
        r_vert_pos_w = [r_mat_uv @ v.co for v in r_verts_d_uv]
        r_face_kd_uv = mathutils.kdtree.KDTree(len(r_polys_d))
        r_face_centers = []   # world-space center per Rodin face (for Z-proximity filter)
        for _i, _poly in enumerate(r_polys_d):
            _cw = r_mat_uv @ _poly.center
            r_face_kd_uv.insert(_cw, _i)
            r_face_centers.append(_cw)
        r_face_kd_uv.balance()
        r_face_data = []
        for poly in r_polys_d:
            vis = list(poly.vertices)
            vws = [r_vert_pos_w[vi] for vi in vis]
            uvs = [r_uv_src[li].uv.copy() for li in poly.loop_indices]
            r_face_data.append((vis, vws, uvs))
        # Precompute Trellis z/x bounds for leg-region detection.
        # In the leg zone the two legs are spatially close, so a naive nearest-face
        # KD-tree query often assigns an inner-right-leg Trellis face to an outer-left-leg
        # Rodin face → smoothly interpolated UV → no UV jump → no seam detected.
        # Fix: restrict the KD-tree search to Rodin faces on the same X side as the
        # Trellis face being processed.
        t_wv_b    = [obj.matrix_world @ v.co for v in obj.data.vertices]
        t_z_min_b = min(v.z for v in t_wv_b)
        t_z_max_b = max(v.z for v in t_wv_b)
        t_z_rng_b = max(t_z_max_b - t_z_min_b, 1e-6)
        t_x_ctr_b = (min(v.x for v in t_wv_b) + max(v.x for v in t_wv_b)) / 2.0
        del t_wv_b
        # After alignment, Rodin's x-center matches Trellis's x-center.
        r_x_ctr          = t_x_ctr_b
        LEG_Z_THRESH     = 0.45   # normalised Z: below this → leg region
        HIP_Z_LO         = 0.30   # bottom of hip band
        N_CAND           = 100    # candidate pool for side-filtered search
        # Min component size for hip-zone seam cleanup (step 5c), X-zone aware.
        # Near center (|xn|<0.04): groin-blob noise → require 15 edges.
        # Outer leg (|xn|>=0.04): real outer-leg seam has short chains (5-9 edges on
        # denser Trellis topology vs Rodin's 52 right-hip edges) → keep at 5+.
        HIP_SEAM_MIN_CENTER = 15   # anti-groin-blob threshold
        HIP_SEAM_MIN_OUTER  = 5    # outer-leg threshold (recovers right-hip seam)

        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")
        t_uv_dst  = obj.data.uv_layers.active.data
        t_loops_d = obj.data.loops
        t_verts_d = obj.data.vertices
        t_mat_uv    = obj.matrix_world
        t_nmat_uv   = t_mat_uv.to_3x3()
        n_leg_fb    = 0
        n_sv_used   = 0

        # ── 3. Transfer Rodin UV to Trellis (face-centroid — for seam detection) ─
        # We use face-centroid KD-tree here intentionally: adjacent faces pointing
        # at different Rodin face centroids create UV discontinuities that the seam
        # detector in step 4 uses to find island boundaries.  This is a two-pass
        # design: face-centroid UV drives seam detection; BVH UV (step 5e) replaces
        # it afterwards for smooth, high-quality texture export.
        for poly in obj.data.polygons:
            fc_w  = t_mat_uv @ poly.center
            zn    = (fc_w.z - t_z_min_b) / t_z_rng_b
            t_wn  = t_nmat_uv @ poly.normal
            if t_wn.length > 1e-6: t_wn.normalize()
            if zn < LEG_Z_THRESH:
                t_side = 1 if fc_w.x >= t_x_ctr_b else -1
                cands  = r_face_kd_uv.find_n(fc_w, N_CAND)
                same   = [(pd, fi) for rco, fi, pd in cands
                          if (1 if rco.x >= r_x_ctr else -1) == t_side]
                if HIP_Z_LO <= zn < LEG_Z_THRESH and same:
                    z_tol_w = t_z_rng_b * 0.12
                    z_same  = [(pd, fi) for pd, fi in same
                               if abs(r_face_centers[fi].z - fc_w.z) < z_tol_w]
                    if z_same:
                        same = z_same
                if same:
                    r_fi = min(same, key=lambda x: x[0])[1]
                else:
                    n_leg_fb += 1
                    r_fi = min(cands, key=lambda x: x[2])[1]
            else:
                _, r_fi, _ = r_face_kd_uv.find(fc_w)
            _, r_vws, r_uvs = r_face_data[r_fi]
            for li in poly.loop_indices:
                t_vi = t_loops_d[li].vertex_index
                t_vw = t_mat_uv @ t_verts_d[t_vi].co
                r_vi = vert_to_rodin.get(t_vi, -1)
                if r_vi in r_sv_uv and zn < LEG_Z_THRESH:
                    best_uv = max(r_sv_uv[r_vi], key=lambda x: x[0].dot(t_wn))[1]
                    t_uv_dst[li].uv = best_uv
                    n_sv_used += 1
                else:
                    t_uv_dst[li].uv = _bary_uv_in_face(t_vw, r_vws, r_uvs)
        print(f"[BlenderSmartUV] UV transferred from Rodin (seam-vert={{n_sv_used}}, leg_fallbacks={{n_leg_fb}})")

        # ── 4. Detect seam edges from UV discontinuities ──────────────────────
        # Thresholds re-calibrated for BVH-based UV transfer.
        # With face-centroid UV, noisy intra-island UV jumps could exceed 0.35,
        # so high thresholds were needed to reject them.  With BVH, within-island
        # UV variation between adjacent faces is < 0.01 (smoothly interpolated),
        # so we can use much lower thresholds to catch real inter-island boundaries.
        # Rodin's head has many compact UV islands (face, hair sections, ears) that
        # may be only 0.05–0.15 UV units apart → need threshold << 0.35.
        # Torso arm-socket seams are larger jumps (~0.3–0.8) → 0.08 still catches
        # them while rejecting within-island noise.
        SEAM_UV_THRESH_HEAD  = 0.35
        SEAM_UV_THRESH_TORSO = 0.50
        SEAM_UV_THRESH_HIP   = 0.13
        SEAM_UV_THRESH_LEG   = 0.15
        SEAM_THRESH_SHARP    = 0.10
        HIP_Z_LO_SEAM        = 0.30
        NECK_Z_LO_SEAM       = 0.70
        LEG_Z_THRESH_SEAM    = 0.45
        bm_t  = bmesh.new()
        bm_t.from_mesh(obj.data)
        bm_t.verts.ensure_lookup_table()
        bm_t.edges.ensure_lookup_table()
        uv_layer_bm = bm_t.loops.layers.uv.active
        # Precompute world-space Z per vertex for threshold selection
        _t_mat4  = obj.matrix_world
        _t_vz    = {{v.index: (_t_mat4 @ v.co).z for v in bm_t.verts}}
        _t_vx    = {{v.index: (_t_mat4 @ v.co).x for v in bm_t.verts}}
        _t_vy    = {{v.index: (_t_mat4 @ v.co).y for v in bm_t.verts}}
        n_t_seams = 0
        n_t_sharp = 0
        n_leg_seams = 0
        for edge in bm_t.edges:
            t_vi0 = edge.verts[0].index
            t_vi1 = edge.verts[1].index
            is_seam = False
            if len(edge.link_faces) < 2:
                is_seam = True
            elif uv_layer_bm:
                e_zn = ((_t_vz[t_vi0] + _t_vz[t_vi1]) * 0.5 - t_z_min_b) / t_z_rng_b
                if   e_zn < HIP_Z_LO_SEAM:      thresh = SEAM_UV_THRESH_LEG
                elif e_zn < LEG_Z_THRESH_SEAM:  thresh = SEAM_UV_THRESH_HIP
                elif e_zn < NECK_Z_LO_SEAM:     thresh = SEAM_UV_THRESH_TORSO
                else:                            thresh = SEAM_UV_THRESH_HEAD
                fa, fb = edge.link_faces[0], edge.link_faces[1]
                v_set = {{t_vi0, t_vi1}}
                uv_a = {{lp.vert.index: lp[uv_layer_bm].uv.copy()
                         for lp in fa.loops if lp.vert.index in v_set}}
                uv_b = {{lp.vert.index: lp[uv_layer_bm].uv.copy()
                         for lp in fb.loops if lp.vert.index in v_set}}
                for vi in v_set:
                    if vi in uv_a and vi in uv_b:
                        if (uv_a[vi] - uv_b[vi]).length > thresh:
                            is_seam = True
                            if e_zn < LEG_Z_THRESH_SEAM:
                                n_leg_seams += 1
                            break
            if is_seam:
                edge.seam = True
                n_t_seams += 1
            r_vi0s = vert_to_rodin.get(t_vi0, -1)
            r_vi1s = vert_to_rodin.get(t_vi1, -1)
            if r_vi0s != r_vi1s and r_vi0s in r_sharp_verts and r_vi1s in r_sharp_verts:
                if (r_norm_v[r_vi0s] - r_norm_v[r_vi1s]).length < SEAM_THRESH_SHARP:
                    edge.smooth = False
                    n_t_sharp += 1
        print(f"[BlenderSmartUV] Trellis seams (thresh head={{SEAM_UV_THRESH_HEAD}} torso={{SEAM_UV_THRESH_TORSO}} hip={{SEAM_UV_THRESH_HIP}} leg={{SEAM_UV_THRESH_LEG}}): {{n_t_seams}} ({{n_t_seams/max(n_r_seams,1):.1%}} of Rodin), leg={{n_leg_seams}}")
        print(f"[BlenderSmartUV] Trellis sharp edges: {{n_t_sharp}}")

        # ── 5. Loop closure: fill 1-edge gaps in seam loops ──────────────────
        # A seam vertex with exactly 1 seam-edge neighbour is an open end.
        # If two open ends share a non-seam edge → mark it to close the 1-edge gap.
        # Repeat up to 20 passes until no more 1-edge gaps remain.
        gap_filled = True
        gap_passes = 0
        while gap_filled and gap_passes < 20:
            gap_filled = False
            gap_passes += 1
            seam_deg = {{}}
            for e in bm_t.edges:
                if e.seam:
                    for v in e.verts:
                        seam_deg[v.index] = seam_deg.get(v.index, 0) + 1
            open_ends = {{vi for vi, cnt in seam_deg.items() if cnt == 1}}
            if not open_ends:
                break
            for e in bm_t.edges:
                if not e.seam:
                    v0i, v1i = e.verts[0].index, e.verts[1].index
                    if v0i in open_ends and v1i in open_ends:
                        e.seam = True
                        n_t_seams += 1
                        gap_filled = True
        seam_deg2 = {{}}
        for e in bm_t.edges:
            if e.seam:
                for v in e.verts:
                    seam_deg2[v.index] = seam_deg2.get(v.index, 0) + 1
        open_end_ct = sum(1 for cnt in seam_deg2.values() if cnt == 1)
        print(f"[BlenderSmartUV] Loop closure ({{gap_passes}} passes): {{n_t_seams}} seams, {{open_end_ct}} open ends remain")

        # ── 5b. Seam thinning: trim only very short dangling tips (≤5 edges) ──
        # Each pass removes only the outermost tip edge from every dangling chain.
        # Capped at MAX_THIN_PASSES=5 so chains longer than 5 edges are preserved —
        # those long chains are real island boundaries with an unresolved gap and
        # must not be destroyed. Only tiny noise spurs (1-5 edges) get trimmed.
        # Closed loops (all verts degree ≥ 2) are never touched.
        MAX_THIN_PASSES = 5
        thin_passes  = 0
        thin_removed = True
        total_thinned = 0
        while thin_removed and thin_passes < MAX_THIN_PASSES:
            thin_removed = False
            thin_passes += 1
            sdeg = {{}}
            for e in bm_t.edges:
                if e.seam:
                    for v in e.verts:
                        sdeg[v.index] = sdeg.get(v.index, 0) + 1
            tips = {{vi for vi, cnt in sdeg.items() if cnt == 1}}
            if not tips:
                break
            for e in bm_t.edges:
                if e.seam and (e.verts[0].index in tips or e.verts[1].index in tips):
                    e.seam = False
                    n_t_seams -= 1
                    total_thinned += 1
                    thin_removed = True
        sdeg_final = {{}}
        for e in bm_t.edges:
            if e.seam:
                for v in e.verts:
                    sdeg_final[v.index] = sdeg_final.get(v.index, 0) + 1
        open_end_final = sum(1 for cnt in sdeg_final.values() if cnt == 1)
        print(f"[BlenderSmartUV] Seam thinning ({{thin_passes}} passes, {{total_thinned}} edges removed): {{n_t_seams}} seams, {{open_end_final}} open ends remain")

        # ── 5c. Hip-zone small-component removal (X-zone aware) ──────────────
        # UV seam detection in the hip zone can produce:
        #   a) Groin-blob: small closed loops (3-15 edges) near body center (|xn|<0.04)
        #      due to noisy cross-island UV at the crotch region.
        #   b) Real outer-leg seams: short chains (5-9 edges) on the outer hip
        #      silhouette (|xn|>=0.04). Right leg chains are small because the Rodin
        #      right-hip seam is geometrically narrow (52 edges at xn=0.075-0.131).
        # Strategy: use tight threshold for center (kills groin blob), loose for outer.
        _hip_lo_z = t_z_min_b + HIP_Z_LO_SEAM * t_z_rng_b
        _hip_hi_z = t_z_min_b + LEG_Z_THRESH_SEAM * t_z_rng_b
        _hip_seam_adj = {{}}
        for _e in bm_t.edges:
            if not _e.seam: continue
            _ez = (_t_vz[_e.verts[0].index] + _t_vz[_e.verts[1].index]) * 0.5
            if not (_hip_lo_z <= _ez < _hip_hi_z): continue
            _va, _vb = _e.verts[0].index, _e.verts[1].index
            _hip_seam_adj.setdefault(_va, []).append((_vb, _e.index))
            _hip_seam_adj.setdefault(_vb, []).append((_va, _e.index))
        _visited_hsc = set()
        _hip_components = []
        for _sv in list(_hip_seam_adj.keys()):
            if _sv in _visited_hsc: continue
            _comp_eids = set()
            _stack = [_sv]
            while _stack:
                _v = _stack.pop()
                if _v in _visited_hsc: continue
                _visited_hsc.add(_v)
                for _nb, _eid in _hip_seam_adj.get(_v, []):
                    _comp_eids.add(_eid)
                    if _nb not in _visited_hsc: _stack.append(_nb)
            _hip_components.append(_comp_eids)
        _hip_removed = 0
        _hip_kept = 0
        _n_center_removed = 0
        _n_outer_removed = 0
        bm_t.edges.ensure_lookup_table()
        for _comp in _hip_components:
            # Determine if this component is near body center or outer leg
            _comp_x_avg = sum(
                (_t_vx[bm_t.edges[_eid].verts[0].index] + _t_vx[bm_t.edges[_eid].verts[1].index]) * 0.5
                for _eid in _comp
            ) / max(len(_comp), 1)
            _xn = abs(_comp_x_avg - t_x_ctr_b)
            _min_chain = HIP_SEAM_MIN_OUTER if _xn >= 0.04 else HIP_SEAM_MIN_CENTER
            if len(_comp) < _min_chain:
                for _eid in _comp:
                    bm_t.edges[_eid].seam = False
                    n_t_seams -= 1
                    _hip_removed += 1
                if _xn >= 0.04:
                    _n_outer_removed += 1
                else:
                    _n_center_removed += 1
            else:
                _hip_kept += len(_comp)
        _n_kept_comps = len(_hip_components) - (_n_center_removed + _n_outer_removed)
        print(f"[BlenderSmartUV] Hip cleanup: removed {{_hip_removed}} edges "
              f"(center blobs={{_n_center_removed}}<{{HIP_SEAM_MIN_CENTER}}, "
              f"outer noise={{_n_outer_removed}}<{{HIP_SEAM_MIN_OUTER}}), "
              f"kept {{_hip_kept}} edges in {{_n_kept_comps}} components")

        # ── 5d. Hip zone forward seam projection ─────────────────────────────
        # UV-based detection gave L=~450 / R=~13 vs Rodin target L=~167 / R=~52.
        # Replacing with forward projection: for each Rodin hip seam edge, mark
        # the K nearest same-side Trellis edges. This bypasses the UV discontinuity
        # problem (right-hip seam is too narrow to show large UV deltas in Trellis).
        _fwd_hip_lo_z = t_z_min_b + HIP_Z_LO_SEAM * t_z_rng_b
        _fwd_hip_hi_z = t_z_min_b + LEG_Z_THRESH_SEAM * t_z_rng_b

        # Step 1: Unmark all current hip-zone seam marks (from UV detection + 5c)
        _fwd_cleared = 0
        bm_t.edges.ensure_lookup_table()
        for _e in bm_t.edges:
            if not _e.seam: continue
            _ez5 = (_t_vz[_e.verts[0].index] + _t_vz[_e.verts[1].index]) * 0.5
            if _fwd_hip_lo_z <= _ez5 < _fwd_hip_hi_z:
                _e.seam = False
                n_t_seams -= 1
                _fwd_cleared += 1

        # Step 2: Collect Rodin hip seam edge centroids with X-side
        _r_hip_secs_L = []  # Vector3 centroids on left side
        _r_hip_secs_R = []  # Vector3 centroids on right side
        for _sec in r_seam_ec_3d:
            _szn = (_sec.z - t_z_min_b) / t_z_rng_b
            if HIP_Z_LO_SEAM <= _szn < LEG_Z_THRESH_SEAM:
                if _sec.x >= r_x_ctr:
                    _r_hip_secs_R.append(_sec)
                else:
                    _r_hip_secs_L.append(_sec)

        # Step 3: Build per-side KD-trees of Trellis hip-zone edge centroids (3D)
        _t_hip_eL = []  # (eid, Vector3)
        _t_hip_eR = []
        for _e in bm_t.edges:
            _vi0, _vi1 = _e.verts[0].index, _e.verts[1].index
            _ez5 = (_t_vz[_vi0] + _t_vz[_vi1]) * 0.5
            _ezn = (_ez5 - t_z_min_b) / t_z_rng_b
            if not (HIP_Z_LO_SEAM <= _ezn < LEG_Z_THRESH_SEAM): continue
            _ex5 = (_t_vx[_vi0] + _t_vx[_vi1]) * 0.5
            _ey5 = (_t_vy[_vi0] + _t_vy[_vi1]) * 0.5
            _ec5 = mathutils.Vector((_ex5, _ey5, _ez5))
            if _ex5 >= r_x_ctr:
                _t_hip_eR.append((_e.index, _ec5))
            else:
                _t_hip_eL.append((_e.index, _ec5))

        def _build_kd(edata):
            kd = mathutils.kdtree.KDTree(max(len(edata), 1))
            for ii, (eid, ec) in enumerate(edata):
                kd.insert(ec, ii)
            kd.balance()
            return kd

        _kd_hip_L = _build_kd(_t_hip_eL)
        _kd_hip_R = _build_kd(_t_hip_eR)

        # Step 4: Forward project K=2 nearest Trellis edges per Rodin seam edge
        _HIP_FWD_K = 2
        _fwd_marked_L = set()
        _fwd_marked_R = set()
        for _sec in _r_hip_secs_L:
            for _, _ii, _ in _kd_hip_L.find_n(_sec, _HIP_FWD_K):
                _eid = _t_hip_eL[_ii][0]
                if _eid not in _fwd_marked_L:
                    _fwd_marked_L.add(_eid)
                    bm_t.edges[_eid].seam = True
                    n_t_seams += 1
        for _sec in _r_hip_secs_R:
            for _, _ii, _ in _kd_hip_R.find_n(_sec, _HIP_FWD_K):
                _eid = _t_hip_eR[_ii][0]
                if _eid not in _fwd_marked_R:
                    _fwd_marked_R.add(_eid)
                    bm_t.edges[_eid].seam = True
                    n_t_seams += 1
        print(f"[BlenderSmartUV] Hip fwd proj: cleared {{_fwd_cleared}} UV seams, "
              f"added {{len(_fwd_marked_L)+len(_fwd_marked_R)}} fwd seams "
              f"(L={{len(_fwd_marked_L)}} from {{len(_r_hip_secs_L)}} Rodin, "
              f"R={{len(_fwd_marked_R)}} from {{len(_r_hip_secs_R)}} Rodin, K={{_HIP_FWD_K}})")

        bm_t.to_mesh(obj.data)
        obj.data.update()   # flush bmesh seam flags to mesh

        # Snapshot seam indices NOW — before bm_t is freed and before
        # Smart UV Project (below) re-enters Edit Mode, which can reset
        # obj.data.edges[*].use_seam in Blender 5.x.
        _seam_edge_indices = [e.index for e in bm_t.edges if e.seam]
        print(f"[BlenderSmartUV] Seam snapshot: {{len(_seam_edge_indices)}} edges captured before UV unwrap")

        bm_t.free()

        # ── 5e. Smart UV Project — clean, non-overlapping UV ─────────────────
        # The face-centroid UV (step 3) and any vertex-level UV transfer produce
        # noisy, fragmented UV because the Trellis mesh is an isotropic remesh
        # with no topological alignment to Rodin's UV island boundaries.
        # Smart UV Project bypasses this by using face angles to determine UV
        # island cuts, producing a valid, well-packed UV layout that the
        # multi-view texturing node can use effectively.
        # The seam MARKS set in steps 4-5d are preserved (they indicate the
        # approximate Rodin-island boundaries on the Trellis surface).
        import bpy as _bpy5e
        _ctx5e = _bpy5e.context
        _ctx5e.view_layer.objects.active = obj
        _bpy5e.ops.object.mode_set(mode='EDIT')
        _bpy5e.ops.mesh.select_all(action='SELECT')
        _bpy5e.ops.uv.smart_project(
            angle_limit=66.0,
            margin_method='SCALED',
            island_margin=0.004,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False,
        )
        _bpy5e.ops.object.mode_set(mode='OBJECT')
        _uv5e_data = obj.data.uv_layers.active.data
        _us5e = [_uv5e_data[li].uv.x for li in range(len(_uv5e_data))]
        _vs5e = [_uv5e_data[li].uv.y for li in range(len(_uv5e_data))]
        _oor5e = sum(1 for u, v in zip(_us5e, _vs5e) if not (0 <= u <= 1 and 0 <= v <= 1))
        print(f"[BlenderSmartUV] Smart UV Project: UV range U=[{{min(_us5e):.4f}},{{max(_us5e):.4f}}] "
              f"V=[{{min(_vs5e):.4f}},{{max(_vs5e):.4f}}], out-of-range={{_oor5e}}")

        # ── Seam band analysis (printed every run for free) ───────────────────
        _mat2    = obj.matrix_world
        _wv2     = [_mat2 @ v.co for v in obj.data.vertices]
        _zs2     = [v.z for v in _wv2]
        _xs2     = [v.x for v in _wv2]
        _z2_min  = min(_zs2); _z2_max = max(_zs2)
        _z2_rng  = max(_z2_max - _z2_min, 1e-6)
        _x2_ctr  = (min(_xs2) + max(_xs2)) / 2.0
        _bands2  = [('head',0.85,1.01),('neck_shldr',0.70,0.85),
                    ('torso',0.45,0.70),('hip_leg',0.30,0.45),
                    ('mid_leg',0.15,0.30),('ankle',0.00,0.15)]
        _bs2     = {{b: [0,0,0,0] for b,_,_ in _bands2}}  # [total, L, R, C]
        _sdeg2   = {{}}
        for _e2 in obj.data.edges:
            if not _e2.use_seam: continue
            _ec2 = (_wv2[_e2.vertices[0]] + _wv2[_e2.vertices[1]]) * 0.5
            _zn2 = (_ec2.z - _z2_min) / _z2_rng
            for _bn, _lo, _hi in _bands2:
                if _lo <= _zn2 < _hi:
                    _bs2[_bn][0] += 1
                    if   _ec2.x < _x2_ctr - 0.03: _bs2[_bn][1] += 1
                    elif _ec2.x > _x2_ctr + 0.03: _bs2[_bn][2] += 1
                    else:                          _bs2[_bn][3] += 1
                    break
            for _vi2 in _e2.vertices:
                _sdeg2[_vi2] = _sdeg2.get(_vi2, 0) + 1
        _oe2     = {{b: 0 for b,_,_ in _bands2}}
        for _vi2, _cnt2 in _sdeg2.items():
            if _cnt2 != 1: continue
            _vw2 = _wv2[_vi2]; _zn2 = (_vw2.z - _z2_min) / _z2_rng
            for _bn, _lo, _hi in _bands2:
                if _lo <= _zn2 < _hi: _oe2[_bn] += 1; break
        _oe_tot2 = sum(1 for c in _sdeg2.values() if c == 1)
        # Rodin band distribution (for comparison) — uses same Z/X bounds as Trellis
        _r_bs2 = {{b: [0,0,0,0] for b,_,_ in _bands2}}
        for _rx, _rz in r_seam_ec:
            _rzn = (_rz - _z2_min) / _z2_rng
            for _rbn, _rlo, _rhi in _bands2:
                if _rlo <= _rzn < _rhi:
                    _r_bs2[_rbn][0] += 1
                    if   _rx < _x2_ctr - 0.03: _r_bs2[_rbn][1] += 1
                    elif _rx > _x2_ctr + 0.03: _r_bs2[_rbn][2] += 1
                    else:                       _r_bs2[_rbn][3] += 1
                    break
        print(f"[SeamBands] Trellis total={{n_t_seams}} open={{_oe_tot2}}  |  Rodin total={{n_r_seams}}")
        print(f"[SeamBands] {'band':<12} {'range':<12} {'T':>5}/{'R':>5} {'%':>5}  TL/TR  RL/RR  open")
        for _bn, _lo, _hi in _bands2:
            _t,_l,_r,_c   = _bs2[_bn]
            _rt,_rl,_rr,_ = _r_bs2[_bn]
            _pct = 100.0*_t/max(_rt,1)
            _oe_b = _oe2[_bn]
            print(f"[SeamBands]  {{_bn:<12}} ({{_lo:.2f}}-{{_hi:.2f}})  {{_t:4d}}/{{_rt:4d}} {{_pct:5.0f}}%  TL={{_l:3d}} TR={{_r:3d}}  RL={{_rl:3d}} RR={{_rr:3d}}  open={{_oe_b}}")
        del _wv2, _zs2, _xs2, _bs2, _sdeg2, _oe2, _r_bs2

        # ── Blend file save for seam visualisation ───────────────────────────
        # Re-stamp seam marks from the snapshot taken before Smart UV Project.
        # Smart UV Project (step 5e above) runs in Edit Mode, which can silently
        # clear use_seam flags on obj.data.edges in Blender 5.x when the Edit→Object
        # transition writes the BMesh back.  Re-stamping from the snapshot is the
        # reliable way to ensure seam marks survive into the saved .blend file.
        # Open in Edit Mode → Overlays → Show Seams to see cyan lines.
        _seam_restore_ct = 0
        for _sei in _seam_edge_indices:
            if _sei < len(obj.data.edges):
                obj.data.edges[_sei].use_seam = True
                _seam_restore_ct += 1
        obj.data.update()
        print(f"[BlenderSmartUV] Seam restore: {{_seam_restore_ct}} edges re-stamped after Smart UV Project")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        _blend_path = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/last_seams.blend'
        bpy.ops.wm.save_as_mainfile(filepath=_blend_path)
        n_seam_edges = sum(1 for e in obj.data.edges if e.use_seam)
        print(f"[BlenderSmartUV] Blend saved: {{_blend_path}} | mesh seam edges: {{n_seam_edges}}")

        # ── 6. UV already set by Smart UV Project in step 5e — skip re-unwrap ──
        # Smart UV Project ran in step 5e and produced a clean, non-overlapping
        # UV layout.  Running bpy.ops.uv.unwrap() here would destroy that layout
        # because the seam marks (from steps 4-5d) have open ends and don't form
        # closed island loops — unwrap collapses everything into one giant diagonal
        # strip.  Smart UV Project is self-contained and doesn't need seam marks.
        print("[BlenderSmartUV] UV unwrap skipped — UV already set by Smart UV Project (step 5e)")

        # Clean up template object
        bpy.ops.object.select_all(action='DESELECT')
        template_obj.select_set(True)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    elif template_obj is not None:
        print("[BlenderSmartUV] WARNING: Template has no UV layers — skipping transfer")
        bpy.ops.object.select_all(action='DESELECT')
        template_obj.select_set(True)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

# ── UV unwrap (smart_uv mode only — xatlas is done in Python after export) ───
if {do_smart_uv}:
    coords = [v.co for v in mesh_data.vertices]
    xs = [c.x for c in coords]; zs = [c.z for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    cx  = (min_x + max_x) / 2
    hw  = (max_x - min_x) / 2 + 1e-8
    h   = max_z - min_z + 1e-8

    HEAD_Z   = 0.82
    ARM_Z_LO = 0.55
    ARM_X    = 0.36
    LEG_Z_HI = 0.44

    zones = {{'head': [], 'torso': [],
              'arm_pos': [], 'arm_neg': [],
              'leg_pos': [], 'leg_neg': []}}

    for poly in mesh_data.polygons:
        c = mathutils.Vector((0.0, 0.0, 0.0))
        for vi in poly.vertices:
            c += mesh_data.vertices[vi].co
        c /= len(poly.vertices)
        rz  = (c.z - min_z) / h
        dx  = c.x - cx
        rx  = abs(dx) / hw
        if rz > HEAD_Z:
            zones['head'].append(poly.index)
        elif rx > ARM_X and rz > ARM_Z_LO:
            zones['arm_pos' if dx > 0 else 'arm_neg'].append(poly.index)
        elif rz < LEG_Z_HI:
            zones['leg_pos' if dx > 0 else 'leg_neg'].append(poly.index)
        else:
            zones['torso'].append(poly.index)

    for zn, idxs in zones.items():
        print(f"[BlenderSmartUV] Zone '{{zn}}': {{len(idxs)}} faces")

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh_data)
    bm.faces.ensure_lookup_table()

    for zone_name, face_indices in zones.items():
        if not face_indices:
            print(f"[BlenderSmartUV] Zone '{{zone_name}}' empty - skipping")
            continue
        for f in bm.faces:
            f.select = False
        bm.select_flush(False)
        for i in face_indices:
            if i < len(bm.faces):
                bm.faces[i].select = True
        bm.select_flush(True)
        bmesh.update_edit_mesh(mesh_data)
        bpy.ops.uv.smart_project(
            angle_limit={angle_limit},
            island_margin={island_margin},
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False,
        )
        print(f"[BlenderSmartUV] Zone '{{zone_name}}' done: {{len(face_indices)}}/{{len(mesh_data.polygons)}} faces")

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(margin={island_margin})
    print("[BlenderSmartUV] Islands packed")
    bpy.ops.object.mode_set(mode='OBJECT')
else:
    if {do_template_transfer}:
        print("[BlenderSmartUV] Smart UV skipped (template_transfer mode handled above)")
    else:
        print("[BlenderSmartUV] UV skipped in Blender (xatlas will unwrap in Python)")

uv_layer = mesh_data.uv_layers.active
print(f"[BlenderSmartUV] UV layer: {{uv_layer.name if uv_layer else 'NONE'}}")
print(f"[BlenderSmartUV] Final mesh: {{len(mesh_data.vertices)}} verts, {{len(mesh_data.polygons)}} faces")

# ── Shade smooth + respect sharp edges so normals encode hard/soft transitions ──
# This bakes the shading quality into the exported normals — sharp edges survive
# through trimesh as correct normals even though the marks themselves don't.
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.shade_smooth()
print("[BlenderSmartUV] Shade smooth applied (sharp edges respected in normals)")

# ── gen_seams mode: save blend file for gen_seams.py to process ──────────────
if {do_gen_seams}:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _gs_blend = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/last_seams.blend'
    bpy.ops.wm.save_as_mainfile(filepath=_gs_blend)
    print(f"[BlenderSmartUV] Saved blend for gen_seams.py: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")

# ── Export as OBJ (preserves UV + smoothing groups → sharp edges visible in Blender) ──
# Make sure the mesh object is active and selected before export
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

import os as _os
try:
    ret = bpy.ops.wm.obj_export(
        filepath=r'{out_obj}',
        export_uv=True,
        export_normals=True,
        export_smooth_groups=True,
        export_materials=False,
        export_selected_objects=False,
    )
    if 'FINISHED' not in ret:
        print(f"[BlenderSmartUV] ERROR: obj_export returned {{ret}} — trying without smooth groups")
        ret = bpy.ops.wm.obj_export(
            filepath=r'{out_obj}',
            export_uv=True,
            export_normals=False,
            export_selected_objects=False,
        )
        if 'FINISHED' not in ret:
            print(f"[BlenderSmartUV] ERROR: fallback obj_export also failed: {{ret}}")
            sys.exit(1)
        else:
            print("[BlenderSmartUV] WARNING: exported without smooth groups (sharp edges not in OBJ)")
except Exception as _e:
    print(f"[BlenderSmartUV] EXCEPTION during OBJ export: {{_e}}")
    sys.exit(1)

if not _os.path.exists(r'{out_obj}'):
    print("[BlenderSmartUV] ERROR: OBJ file not written despite FINISHED status")
    sys.exit(1)
print("[BlenderSmartUV] Export done:", r'{out_obj}')
"""

        # -- MESH-PREP RETRY -----------------------------------------------------
        # Everything above this point -- Qwen view generation and the Trellis mesh, ~7 minutes
        # of the run -- is already done by the time mesh prep can fail. Throwing that away and
        # making the user re-queue the whole prompt is pure waste, so retry the mesh-prep block
        # in place instead. Each attempt nudges the voxel face target: repeating a deterministic
        # step on identical input just reproduces the same failure, so the mesh has to actually
        # differ for a retry to mean anything.
        _MP_SCALES = (1.00, 0.93, 1.08, 0.86, 1.15)
        for _mp_try in range(len(_MP_SCALES)):
            os.environ["GS_TARGET_SCALE"] = str(_MP_SCALES[_mp_try])
            if _mp_try:
                print(f"[BlenderSmartUV] === MESH-PREP RETRY {_mp_try + 1}/{len(_MP_SCALES)} "
                      f"(face target x{_MP_SCALES[_mp_try]}) - reusing the generated images and "
                      f"the Trellis mesh; only mesh prep re-runs ===")
            try:
                script_path = os.path.join(tmp_dir, "uv_unwrap.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(blender_script)

                uv_layout_png = os.path.join(tmp_dir, "uv_layout.png")
                _stage("blender main (voxel remesh + export)")
                print("[BlenderSmartUV] Running Blender 5.1 headlessly...")
                result = subprocess.run(
                    # --python-exit-code: without it Blender exits 0 even when the script raises, so a
                    # hard failure inside the script is silently swallowed and the bake carries on.
                    [BLENDER_EXE, "--background", "--python-exit-code", "1", "--python", script_path],
                    capture_output=True, text=True, timeout=600,
                    encoding='utf-8', errors='replace'
                )
                for line in (result.stdout or "").splitlines():
                    if "[BlenderSmartUV]" in line or "[SeamBands]" in line or "ERROR" in line.upper() or "WARNING" in line.upper():
                        print(line)
                if result.returncode != 0:
                    print("[BlenderSmartUV] STDERR:", result.stderr[-3000:])
                    raise RuntimeError(f"Blender exited with code {result.returncode}")

                # ── gen_seams: run gen_seams.py then export_glb.py as extra subprocesses ─
                _UV_TRANSFER_DIR = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
                _LAST_SEAMS_BLEND = os.path.join(_UV_TRANSFER_DIR, "last_seams.blend")
                _LAST_SEAMS_GLB   = os.path.join(_UV_TRANSFER_DIR, "last_seams.glb")

                if do_gen_seams:
                    import sys as _sys
                    _GS = os.path.join(_UV_TRANSFER_DIR, "gen_seams.py")
                    _BRIDGE = os.path.join(_UV_TRANSFER_DIR, "_gs_facebridge.py")
                    # ── HAND RE-GRAFT (post-remesh) ──────────────────────────────────────
                    # ReplaceHandsWithDonor (pre-remesh) grafts the donor hands at 5x the body's
                    # resolution, and the target voxel remesh above -- being a UNIFORM-resolution
                    # operation -- then equalises them back down to body resolution. Measured on a
                    # real bake: leaving the graft node, hand edge 0.00111 vs body 0.00556 (0.20x);
                    # after the remesh, 0.00247 vs 0.00247 (1.00x). Of the donor's 7,432 triangles
                    # per hand, ~614 polygons survived. The pre-remesh graft is still needed (it is
                    # what makes the body watertight going INTO the remesh) -- so rather than move
                    # it, cut the voxelised hands off and bridge fresh full-res donor hands on here,
                    # in the same post-remesh slot the genital already uses for the same reason.
                    # Measured on _pregraft.blend: hands 2,399 -> 13,840 polys, edge ratio 1.04 ->
                    # 0.61, 0 boundary / 0 non-manifold edges, body quads untouched.
                    # MUST run BEFORE the pelvis graft: pelvis writes _gs_isgenital.npy as face
                    # indices into the mesh as it stands when IT finishes, so anything that adds
                    # faces afterwards would leave that flag pointing at the wrong faces.
                    # ON by default. It was briefly defaulted off so it would not ride along with the
                    # Trellis-texture change, and the consequence was shipping the voxel-flattened
                    # hands again (~614 polygons per hand out of the donor's 7,432) -- which is the
                    # whole defect this stage exists to fix. Disable with env HAND_GRAFT_POST=0.
                    _HGP_ENABLED = os.environ.get("HAND_GRAFT_POST", "0") == "1"
                    _HAND_GRAFT_POST = os.path.join(
                        r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
                        r"\custom_nodes\ComfyUI-HandGraft", "hand_graft_post.py")
                    if _HGP_ENABLED and os.path.exists(_HAND_GRAFT_POST):
                        _stage("hand re-graft")
                        print("[BlenderSmartUV] HAND RE-GRAFT: full-res donor hands onto the remeshed body...")
                        try:
                            import shutil as _sh
                            _sh.copyfile(_LAST_SEAMS_BLEND,
                                         os.path.join(_UV_TRANSFER_DIR, "_prehandgraft.blend"))
                        except Exception as _hge:
                            print(f"[BlenderSmartUV] pre-hand-graft snapshot failed ({_hge!r})")
                        # Fully headless: a wrist is a near-circular tube, so this needs no GPU
                        # sculpt and none of the windowed/off-screen dance the pelvis graft needs.
                        _hg_r = subprocess.run(
                            [BLENDER_EXE, "--background", "--python", _HAND_GRAFT_POST,
                             "--", "--in_blend", _LAST_SEAMS_BLEND],
                            capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=600)
                        if "done (pipeline)" in (_hg_r.stdout or ""):
                            for _l in (_hg_r.stdout or "").splitlines():
                                if "[hand_graft_post]" in _l:
                                    print("  " + _l)
                        else:
                            # Non-fatal, like the pelvis: the body still has its (coarse) remeshed
                            # hands, so the bake completes -- it just does not get the detail back.
                            print("[BlenderSmartUV] WARNING: HAND RE-GRAFT failed -- continuing with "
                                  "the remeshed (coarse) hands.")
                            print((_hg_r.stdout or "")[-1500:], (_hg_r.stderr or "")[-800:])
                    # ── PELVIS GRAFT (KAN-18) ────────────────────────────────────────────
                    # Graft the high-detail donor genital onto the ALREADY-REMESHED body in last_seams.blend, in
                    # place, BEFORE gen_seams UVs it. Trellis produces no genital, and the target voxel remesh would
                    # destroy genital detail -> so we graft AFTER the remesh (never re-remeshed) and let gen_seams UV
                    # the grafted mesh. LOCAL mode keeps the body's quads untouched and adds only the dense donor
                    # patch (~+25k faces, body stays lean -- no whole-body fuse/densify). The surround SMOOTH is the
                    # GPU sculpt (headless numpy Taubin welted/striated), which needs a VIEW_3D -> WINDOWED Blender
                    # launched OFF-SCREEN (-p 8000 8000) so it never shows + foreground-lock so it can't steal focus;
                    # intermittent GPU crash -> 3x retry. Non-fatal (warn + continue without the genital on failure).
                    # Toggle: env PELVIS_GRAFT=0 to disable.
                    _PELVIS_ENABLED = os.environ.get("PELVIS_GRAFT", "1") != "0"
                    _PELVIS_GRAFT = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-HandGraft\pelvis_graft.py"
                    if _PELVIS_ENABLED and os.path.exists(_PELVIS_GRAFT):
                        _stage("pelvis graft")
                        print("[BlenderSmartUV] PELVIS GRAFT: grafting genital onto the remeshed body (headless)...")
                        # Snapshot the PRE-graft blend. The graft edits last_seams.blend in place and
                        # gen_seams then overwrites it, so the input state is destroyed every bake and the
                        # graft can only be debugged by re-baking. With this, pelvis_graft.py --in_blend
                        # can be replayed against the exact same body in seconds.
                        try:
                            import shutil as _sh
                            _pre = os.path.join(_UV_TRANSFER_DIR, "_pregraft.blend")
                            _sh.copyfile(_LAST_SEAMS_BLEND, _pre)
                            print(f"[BlenderSmartUV] pre-graft snapshot -> {_pre}")
                        except Exception as _pge:
                            print(f"[BlenderSmartUV] pre-graft snapshot failed ({_pge!r})")
                        _pv_si = None; _pv_u = None; _pv_old = None
                        if os.name == "nt":
                            import ctypes as _ctypes
                            _pv_si = subprocess.STARTUPINFO(); _pv_si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; _pv_si.wShowWindow = 7
                            try:
                                _pv_u = _ctypes.windll.user32; _pv_buf = _ctypes.c_uint()
                                _pv_u.SystemParametersInfoW(0x2000, 0, _ctypes.byref(_pv_buf), 0); _pv_old = _pv_buf.value
                                _pv_u.SystemParametersInfoW(0x2001, 0, _ctypes.c_void_p(300000), 0)
                            except Exception:
                                _pv_u = None
                        _pv_cmd = [BLENDER_EXE, "--background", "--python", _PELVIS_GRAFT,
                                   "--", "--in_blend", _LAST_SEAMS_BLEND]
                        try:
                            _pv_ok = False; _pv_r = None
                            for _pv_try in range(3):
                                _pv_r = subprocess.run(_pv_cmd, capture_output=True, text=True, startupinfo=_pv_si,
                                                       encoding='utf-8', errors='replace', timeout=900)
                                if "done (pipeline)" in (_pv_r.stdout or ""):
                                    _pv_ok = True
                                    for _l in (_pv_r.stdout or "").splitlines():
                                        # [cutjoin] carries the rim-snap / spacing diagnostics -- the only
                                        # numbers that say WHY a graft came out crumpled. They were being
                                        # filtered out of every log, so the graft was undebuggable.
                                        if any(_k in _l for _k in ("[pelvis_graft]", "[cutjoin]", "[graft]")):
                                            print("  " + _l)
                                    break
                                print(f"[BlenderSmartUV] PELVIS GRAFT crash/retry {_pv_try + 1} (rc={_pv_r.returncode})")
                            if not _pv_ok:
                                print("[BlenderSmartUV] WARNING: PELVIS GRAFT failed after retries -- continuing WITHOUT the genital.")
                                print((_pv_r.stdout or "")[-1500:], (_pv_r.stderr or "")[-800:])
                        finally:
                            if _pv_u is not None and _pv_old is not None:
                                try: _pv_u.SystemParametersInfoW(0x2001, 0, _ctypes.c_void_p(int(_pv_old)), 0x02)
                                except Exception: pass
                        # GENITAL TEXTURING runs LATER -- after gen_seams D2, where the body UVMap exists (see below).
                    # ── FACE BRIDGE (KAN-9) ──────────────────────────────────────────────
                    # gen_seams runs in Blender (no MediaPipe). 3-step bridge gives it a robust
                    # per-character face region for the face UV chart: (A) gen_seams exports the
                    # mesh, (B) python_embeded runs MediaPipe, (C) gen_seams unwraps using it.
                    # ANY failure → gen_seams falls back to its geometric face region.
                    for _f in ("_gs_faceregion.npy", "_gs_facemesh.npz"):
                        try: os.remove(os.path.join(_UV_TRANSFER_DIR, _f))
                        except OSError: pass
                    _stage("face bridge A")
                    print("[BlenderSmartUV] FACE BRIDGE (A) export mesh for MediaPipe...")
                    subprocess.run(
                        [BLENDER_EXE, _LAST_SEAMS_BLEND, "--background", "--python-exit-code", "1",
                         "--python", _GS],
                        env=dict(os.environ, GS_FACE_EXPORT="1"),
                        capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
                    _stage("face bridge B")
                    print("[BlenderSmartUV] FACE BRIDGE (B) MediaPipe face detection...")
                    r_br = subprocess.run(
                        [_sys.executable, _BRIDGE],
                        capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
                    for line in (r_br.stdout or "").splitlines():
                        if "face-bridge" in line or "FaceRegion" in line:
                            print(f"  [face-bridge] {line}")

                    # ── passes C + D1 REMOVED (2026-09-04) ────────────────────────────────────────
                    # C (dump-only is_hair) and D1 (green-border hairline on the dense fixture mesh)
                    # existed ONLY to trace a hairline, which existed only to fence the hair off so a
                    # FRONT-ONLY photo could be interpolated into the sides/back without bleeding.
                    # We now generate real back/left/right views, so there is nothing to fence: the
                    # premise is gone. They cost 2 of the 5 Blender launches per bake and the hairline
                    # they produced was never reliable. D2 computes its own is_hair for the FBX hair
                    # material, so the material survives.
                    # Delete any _gs_ishair from an earlier bake: D2 OVERRIDES its hair mask with that
                    # file if it exists, and the mesh changes every bake, so a stale one is poison.
                    _ishair = os.path.join(_UV_TRANSFER_DIR, "_gs_ishair.npy")
                    try:
                        os.remove(_ishair)
                        print("[BlenderSmartUV] removed stale _gs_ishair.npy (hairline bridge retired)")
                    except OSError:
                        pass

                    # ── gen_seams pass D2: the SINGLE full pass (UV + seam + save). Overrides is_hair with the
                    # green border if _gs_ishair exists, else uses gen_seams' own. ─────────────────────────────
                    _stage("gen_seams D2")
                    print("[BlenderSmartUV] gen_seams (D2, single full pass) UV + seam + save...")
                    r_gs2 = subprocess.run(
                        [BLENDER_EXE, _LAST_SEAMS_BLEND, "--background", "--python-exit-code", "1",
                         "--python", _GS],
                        # GS_FACE_OVAL: cut the face out as its OWN UV chart. Without it the face
                        # merges into the 67k-face body island and ANGLE_BASED squashes the
                        # high-curvature head to nothing -- measured on bake 034459: 5,804 face
                        # faces got 631 texels total (a 25x25 patch), which is why the face never
                        # textured. gen_seams:1352 predicts exactly this ("squishes to a sliver
                        # ... no texels"). The oval costs one seam near the jaw/hairline; that is
                        # strictly better than having no face.
                        env=dict(os.environ, GS_USE_ISHAIR="1", GS_FACE_OVAL="1"),
                        capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace')
                    for line in (r_gs2.stdout or "").splitlines():
                        print(f"  [gen_seams-D2] {line}")
                    if r_gs2.returncode != 0:
                        print("[BlenderSmartUV] gen_seams (D2) STDERR:", (r_gs2.stderr or "")[-2000:])
                        raise RuntimeError(f"gen_seams.py (D2) failed (code {r_gs2.returncode})")

                    # ── GENITAL TEXTURING (KAN-18, approach B / hair-style) ──────────────────────
                    # Body UVMap now exists (gen_seams D2 just saved last_seams.blend). Bake the genital zone maps onto
                    # the body atlas UV -- NO mesh change, just 3 PNGs in uv_transfer/genital_tex. The FBXExport step
                    # then adds a 'Genital' material on slot 2 sampling them (like the Hair material). Non-fatal.
                    # ON by default (env GENITAL_TEX=0 to disable). Safe: separate slot-2 material, non-fatal,
                    # never touches the body material/UV.
                    _GT_OUT = os.path.join(_UV_TRANSFER_DIR, "genital_tex")
                    import shutil as _shutil
                    try: _shutil.rmtree(_GT_OUT)                 # clear stale maps so the FBX material only fires on fresh ones
                    except Exception: pass
                    if os.environ.get("GENITAL_TEX", "1") != "0":
                        _GT_PY = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-HandGraft/genital_texture.py'
                        _stage("genital texturing")
                        print("[BlenderSmartUV] GENITAL TEXTURING (headless): bake albedo/roughness/normal onto body UV...")
                        try:
                            _gt_r = subprocess.run([BLENDER_EXE, "--background", "--python", _GT_PY, "--",
                                                    _LAST_SEAMS_BLEND, _GT_OUT],
                                                   capture_output=True, text=True, timeout=600,
                                                   encoding='utf-8', errors='replace')
                            if "RESULT" in (_gt_r.stdout or ""):
                                for _l in (_gt_r.stdout or "").splitlines():
                                    if "RESULT" in _l:
                                        print("  [genital] " + _l)
                            else:
                                print("[BlenderSmartUV] WARNING: GENITAL TEXTURING failed -- continuing without it.")
                                print((_gt_r.stdout or "")[-1200:], (_gt_r.stderr or "")[-600:])
                        except Exception as _gte:
                            print(f"[BlenderSmartUV] GENITAL TEXTURING error (non-fatal): {_gte}")

                    _stage("export glb")
                    print("[BlenderSmartUV] Exporting last_seams.glb...")
                    r_ex = subprocess.run(
                        [BLENDER_EXE, _LAST_SEAMS_BLEND, "--background", "--python-exit-code", "1",
                         "--python",
                         os.path.join(_UV_TRANSFER_DIR, "export_glb.py")],
                        capture_output=True, text=True, timeout=120,
                        encoding='utf-8', errors='replace'
                    )
                    for line in (r_ex.stdout or "").splitlines():
                        print(f"  [export_glb] {line}")
                    if r_ex.returncode != 0:
                        raise RuntimeError(f"export_glb.py failed (code {r_ex.returncode})")

                    blender_out = _LAST_SEAMS_GLB
                else:
                    # OBJ for template_transfer/smart_uv (preserves UV + smoothing groups)
                    # GLB for xatlas (trimesh loads it cleanly for atlasing)
                    blender_out = out_obj if do_blender_uv else out_glb
                    if not os.path.exists(blender_out):
                        raise RuntimeError(f"Blender produced no output at {blender_out}")
                break
            except Exception as _mpe:
                print(f"[BlenderSmartUV] mesh-prep attempt {_mp_try + 1}/{len(_MP_SCALES)} "
                      f"FAILED: {_mpe}")
                if _mp_try == len(_MP_SCALES) - 1:
                    print("[BlenderSmartUV] all mesh-prep attempts failed - giving up.")
                    raise

        # ── Re-import ─────────────────────────────────────────────────────────
        s = trimesh_lib.load(blender_out, process=False)
        if hasattr(s, 'geometry') and len(s.geometry) > 0:
            geoms = list(s.geometry.values())
            loaded = geoms[0] if len(geoms) == 1 else trimesh_lib.util.concatenate(geoms)
        elif hasattr(s, 'vertices'):
            loaded = s
        else:
            raise RuntimeError("Could not extract mesh from Blender output")

        _stage()   # flush the final stage timing
        print(f"[BlenderSmartUV] Re-imported: {len(loaded.vertices)} verts, {len(loaded.faces)} faces | "
              f"visual={type(loaded.visual).__name__}")

        # ── Merge duplicate vertices (xatlas only, position-only) ────────────────
        # xatlas needs shared vertices to detect chart connectivity.
        # For template_transfer/smart_uv the (pos+UV) weld below handles it properly.
        if uv_method == "xatlas" and len(loaded.vertices) > len(loaded.faces) * 2:
            print(f"[BlenderSmartUV] Un-indexed mesh detected ({len(loaded.vertices)} verts / "
                  f"{len(loaded.faces)} faces) — merging for xatlas...")
            loaded = trimesh_lib.Trimesh(
                vertices=loaded.vertices,
                faces=loaded.faces,
                process=True)   # merges duplicate verts, required for correct xatlas charts
            print(f"[BlenderSmartUV] After merge: {len(loaded.vertices)} verts, {len(loaded.faces)} faces")

        # ── UV: xatlas (Python) | extract from Blender GLB (smart_uv / template_transfer) ──
        if uv_method == "xatlas":
            import xatlas
            print(f"[BlenderSmartUV] Running xatlas on {len(loaded.vertices)} verts, {len(loaded.faces)} faces...")
            vertices = np.asarray(loaded.vertices, dtype=np.float32)
            faces    = np.asarray(loaded.faces,    dtype=np.uint32)
            chart_opts = xatlas.ChartOptions()
            chart_opts.max_cost = 8.0
            chart_opts.normal_deviation_weight = 0.5
            pack_opts = xatlas.PackOptions()
            pack_opts.padding = 4
            pack_opts.bilinear = True
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
            vmapping, new_faces, new_uvs = atlas[0]
            print(f"[BlenderSmartUV] xatlas: {atlas.chart_count} charts, "
                  f"{len(vmapping)} verts, {len(new_faces)} faces")
            loaded = trimesh_lib.Trimesh(
                vertices=vertices[vmapping],
                faces=new_faces.astype(np.int64),
                process=False)
            loaded.visual = tex_vis.TextureVisuals(uv=new_uvs)
            print(f"[BlenderSmartUV] xatlas UV: U=[{new_uvs[:,0].min():.3f},{new_uvs[:,0].max():.3f}]  "
                  f"V=[{new_uvs[:,1].min():.3f},{new_uvs[:,1].max():.3f}]")
        else:
            uv = None
            if hasattr(loaded, 'visual') and hasattr(loaded.visual, 'uv'):
                c = loaded.visual.uv
                if c is not None and len(c) == len(loaded.vertices):
                    uv = c
                    print(f"[BlenderSmartUV] UV from visual.uv: {uv.shape}")
            if uv is None:
                print("[BlenderSmartUV] Falling back to binary GLB parse...")
                _fb = _LAST_SEAMS_GLB if do_gen_seams else out_glb
                uv = _extract_uv_from_glb_binary(_fb)
            if uv is not None:
                # ── V-flip for gen_seams GLB UV ───────────────────────────────
                # Blender's GLTF exporter flips V to conform to GLTF spec
                # (V_gltf = 1 - V_blender).  FBXExport uses last_seams.obj which
                # has V_obj = V_blender.  MultiViewTexturing bakes using the
                # trimesh UV, so we must flip V_gltf → V_blender here so the
                # baked texture and the OBJ/FBX UV are in the same space.
                if do_gen_seams:
                    uv = np.array(uv, dtype=np.float32)
                    uv[:, 1] = 1.0 - uv[:, 1]
                    print("[BlenderSmartUV] gen_seams: V-flipped GLTF UV → Blender/OBJ convention")
                # ── Fix seam-crossing faces ───────────────────────────────
                # A face straddles the UV seam when its U-span exceeds 0.5.
                # Fix: duplicate the minority-side vertex with a wrapped U so
                # the face stays on one side of the seam (no giant triangles).
                faces_arr = np.asarray(loaded.faces)
                verts_arr = np.asarray(loaded.vertices, dtype=np.float32)
                new_v = list(verts_arr)
                new_uv = list(uv)
                new_f  = []
                n_seam = 0
                for face in faces_arr:
                    fu = uv[face, 0]
                    if float(fu.max() - fu.min()) > 0.5:
                        u_med = float(np.median(fu))
                        nf = list(face)
                        for j in range(3):
                            vi = int(face[j])
                            uj = float(fu[j])
                            wrong = (u_med > 0.5 and uj < 0.5) or (u_med <= 0.5 and uj >= 0.5)
                            if wrong:
                                new_vi = len(new_v)
                                new_v.append(verts_arr[vi])
                                fixed = list(uv[vi])
                                fixed[0] = np.clip(uj + (1.0 if u_med > 0.5 else -1.0), 0.0, 1.0)
                                new_uv.append(fixed)
                                nf[j] = new_vi
                                n_seam += 1
                        new_f.append(nf)
                    else:
                        new_f.append(list(face))
                if n_seam > 0:
                    print(f"[BlenderSmartUV] Seam fix: {n_seam} vertex instances duplicated")
                    loaded = trimesh_lib.Trimesh(
                        vertices=np.array(new_v, dtype=np.float32),
                        faces=np.array(new_f, dtype=np.int64),
                        process=False)
                    uv = np.array(new_uv, dtype=np.float32)
                else:
                    print("[BlenderSmartUV] No seam-crossing faces found")

                # ── Weld by (position + UV) ───────────────────────────────────
                # trimesh OBJ loader creates one vertex per unique (pos, UV, normal)
                # triple, so interior vertices that share pos+UV but differ in smooth
                # normals all get split → 3N bloat (420K for a 140K-face mesh).
                # Merging by (pos+UV) collapses those while preserving seam splits
                # (seam vertices share position but have DIFFERENT UV → different key).
                _vw = np.asarray(loaded.vertices, dtype=np.float32)
                _uw = np.asarray(uv, dtype=np.float32)
                _fw = np.asarray(loaded.faces,    dtype=np.int64)
                _keys = np.round(np.hstack([_vw, _uw]), 5)
                _, _inv = np.unique(_keys, axis=0, return_inverse=True)
                _nf = _inv[_fw]
                _ok = (_nf[:, 0] != _nf[:, 1]) & (_nf[:, 1] != _nf[:, 2]) & (_nf[:, 0] != _nf[:, 2])
                _nf = _nf[_ok]
                _nu = int(_inv.max()) + 1
                _nv = np.zeros((_nu, 3), dtype=np.float32)
                _nuv = np.zeros((_nu, 2), dtype=np.float32)
                _nv[_inv]  = _vw
                _nuv[_inv] = _uw
                print(f"[BlenderSmartUV] Weld (pos+UV): {len(_vw)} → {_nu} verts, "
                      f"{len(_fw) - len(_nf)} degenerate faces removed")
                loaded = trimesh_lib.Trimesh(vertices=_nv, faces=_nf, process=False)
                uv = _nuv

                loaded.visual = tex_vis.TextureVisuals(uv=uv)
                print(f"[BlenderSmartUV] UV applied: {uv.shape}  "
                      f"U=[{uv[:,0].min():.3f},{uv[:,0].max():.3f}]  "
                      f"V=[{uv[:,1].min():.3f},{uv[:,1].max():.3f}]")
                # Draw UV layout as PNG using PIL (works in headless mode)
                try:
                    from PIL import Image as _PILImage, ImageDraw as _PILDraw
                    _sz = 1024
                    _img = _PILImage.new('RGB', (_sz, _sz), (20, 20, 20))
                    _draw = _PILDraw.Draw(_img)
                    _faces = np.asarray(loaded.faces)
                    _step = max(1, len(_faces) // 60000)  # subsample to keep it fast
                    for _f in _faces[::_step]:
                        _pts = [(int(uv[_v, 0] * (_sz-1)),
                                 int((1.0 - uv[_v, 1]) * (_sz-1))) for _v in _f]
                        _draw.polygon(_pts, outline=(200, 200, 200))
                    _uv_png = os.path.join(tmp_dir, "uv_layout.png")
                    _img.save(_uv_png)
                    # Also copy to a permanent location next to the template so it survives
                    _perm_png = os.path.join(os.path.dirname(template_path.replace("\\", "/")), "uv_layout_last.png") if do_template_transfer else None
                    if _perm_png:
                        import shutil as _shutil
                        _shutil.copy2(_uv_png, _perm_png)
                        print(f"[BlenderSmartUV] *** UV layout PNG: {_perm_png} ***")
                    else:
                        print(f"[BlenderSmartUV] *** UV layout PNG: {_uv_png} ***")
                except Exception as _e:
                    print(f"[BlenderSmartUV] UV layout draw skipped: {_e}")
            else:
                print("[BlenderSmartUV] WARNING: UV not extracted — MultiViewTexturing will use its own unwrap!")

        # Pass the OBJ/GLB path downstream so FBXExport can import it directly
        # (OBJ with smoothing groups preserves transferred sharp edges; GLB/trimesh strips them)
        if do_gen_seams:
            exported_obj = os.path.join(_UV_TRANSFER_DIR, "last_seams.obj")  # exported by export_glb.py
        else:
            exported_obj = out_obj if (do_blender_uv and os.path.exists(out_obj)) else ""
        return (loaded, exported_obj)


# ── Trellis2XAtlasUnwrap (reference) ─────────────────────────────────────────

class Trellis2XAtlasUnwrap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "max_cost":                ("FLOAT", {"default": 8.0, "min": 1.0, "max": 10000.0, "step": 0.5}),
                "normal_deviation_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0,    "step": 0.1}),
                "padding":                 ("INT",   {"default": 4,   "min": 0,   "max": 32,       "step": 1}),
                "normal_seam_weight":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0,    "step": 0.1}),
                "max_iterations":          ("INT",   {"default": 1,   "min": 1,   "max": 16,       "step": 1}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    def execute(self, trimesh, max_cost, normal_deviation_weight, padding,
                normal_seam_weight=0.0, max_iterations=1):
        import xatlas, trimesh as trimesh_lib
        mesh     = copy.deepcopy(trimesh)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces    = np.asarray(mesh.faces,    dtype=np.uint32)
        chart_opts = xatlas.ChartOptions()
        chart_opts.max_cost = max_cost
        chart_opts.normal_deviation_weight = normal_deviation_weight
        chart_opts.normal_seam_weight      = normal_seam_weight
        chart_opts.max_iterations          = max_iterations
        pack_opts = xatlas.PackOptions()
        pack_opts.padding = padding
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
        vmapping, new_faces, new_uvs = atlas[0]
        result = trimesh_lib.Trimesh(vertices=vertices[vmapping], faces=new_faces.astype(np.int64), process=False)
        result.visual = trimesh_lib.visual.TextureVisuals(uv=new_uvs)
        print(f"[XAtlasUnwrap] {atlas.chart_count} charts")
        return (result,)


# ── Trellis2FBXExport ─────────────────────────────────────────────────────────

class Trellis2FBXExport:
    """
    Final export node — place AFTER Trellis2MultiViewTexturing.
    Mirrors Rodin's output structure:
      - character.fbx       (mesh + hard edges + full PBR material)
      - character_albedo.png
      - character_metallic_roughness.png
      - character_normal.png
    Hard edges marked by angle threshold in Blender before export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {
                    "default": "C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/output/characters",
                    "tooltip": "Characters base directory. Files go to: output_path/character_name/run_TIMESTAMP/ (textures) and output_path/character_name/character.fbx"}),
                "sharp_angle": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Fallback: mark edges sharper than this angle as hard. Only used when no obj_path is connected."}),
            },
            "optional": {
                "base_color":         ("IMAGE",  {"tooltip": "Diffuse/albedo texture from MultiViewTexturing."}),
                "metallic_roughness": ("IMAGE",  {"tooltip": "Metallic-roughness texture from MultiViewTexturing."}),
                "normal_map":         ("IMAGE",  {"tooltip": "Normal map from MultiViewTexturing."}),
                "obj_path":           ("STRING", {"tooltip": "OBJ file from BlenderSmartUV — preserves topology-transferred sharp edges via smoothing groups."}),
                # Eye detection — connect the front-view SV3D image (or the original
                # reference image) to auto-detect iris positions + colours via MediaPipe.
                # Requires: pip install mediapipe
                "front_image":        ("IMAGE",  {"tooltip": "Front-view image used for automatic eye detection (MediaPipe). "
                                                             "Connect the same image used as the front view in MultiViewTexturing. "
                                                             "Leave disconnected to skip eye generation."}),
                "ortho_scale":        ("FLOAT",  {"default": 1.15, "min": 0.5, "max": 3.0, "step": 0.05,
                                                  "tooltip": "Must match the ortho_scale used in MultiViewTexturing — "
                                                             "used to convert 2-D iris pixel positions to 3-D mesh positions."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt":    "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("fbx_path", "eye_texture_left", "eye_texture_right")
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def execute(self, output_path, sharp_angle,
                base_color=None, metallic_roughness=None, normal_map=None,
                obj_path=None, front_image=None, ortho_scale=1.15,
                unique_id=None, prompt=None):
        import trimesh as trimesh_lib
        from PIL import Image as PILImage

        # ── Auto-detect character name from the Load Image node in the graph ───
        # Strategy: scan every node in the prompt for one whose class_type
        # contains "LoadImage" (or similar) and has an "image" string input.
        # This works regardless of how many hops front_image is away from us.
        char_name = ""
        if prompt is not None:
            try:
                # Collect all candidate Load Image nodes
                _load_candidates = []
                for _nid, _ndata in prompt.items():
                    _ct = _ndata.get("class_type", "")
                    _img = _ndata.get("inputs", {}).get("image", "")
                    if isinstance(_img, str) and _img and (
                        "load" in _ct.lower() or "image" in _ct.lower()
                    ):
                        _load_candidates.append((_nid, _ct, _img))
                        print(f"[FBXExport] candidate: node={_nid} type={_ct} image={_img!r}")

                if _load_candidates:
                    # Prefer the node directly feeding front_image if we can trace it
                    _found = None
                    if unique_id is not None:
                        _my_inputs = prompt.get(str(unique_id), {}).get("inputs", {})
                        _fi_link = _my_inputs.get("front_image")
                        if isinstance(_fi_link, list) and len(_fi_link) >= 1:
                            _src_id = str(_fi_link[0])
                            for _nid, _ct, _img in _load_candidates:
                                if _nid == _src_id:
                                    _found = _img
                                    print(f"[FBXExport] matched direct source node {_src_id}")
                                    break
                    # Fallback: just use the first Load Image node found
                    if not _found:
                        _found = _load_candidates[0][2]
                        print(f"[FBXExport] using first load candidate: {_found!r}")

                    char_name = os.path.splitext(os.path.basename(_found))[0].replace(" ", "_")
                    print(f"[FBXExport] character_name={char_name!r}")
            except Exception as _e:
                print(f"[FBXExport] auto character_name failed: {_e}")

        if not char_name:
            char_name = "character"
            print("[FBXExport] WARNING: could not detect image name — using 'character'")
        char_root = os.path.join(output_path.rstrip("/\\"), char_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = os.path.join(char_root, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Everything that made this mesh goes in the run folder with it, so a run can be judged (or
        # blamed) months later without hunting through output/3D.  The four views are what Trellis
        # was actually conditioned on; the Trellis GLB is its raw output before we touch anything.
        # The views are small and get copied; the GLB is ~120 MB and gets moved, since it was only
        # sitting in the character root so it could be eyeballed early enough to kill a bad run.
        try:
            import glob as _glob, shutil as _sh
            _front = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "uv_transfer", "_front_photo.png")
            _front = os.path.normpath(_front)
            if os.path.exists(_front):
                _sh.copy2(_front, os.path.join(run_dir, f"{char_name}__view_front.png"))
            for _tag in ("BACK", "LEFT", "RIGHT", "TOP"):
                _c = sorted(_glob.glob(os.path.join(char_root, f"QWEN_{_tag}_*.png")),
                            key=os.path.getmtime)
                if _c:
                    _sh.copy2(_c[-1], os.path.join(run_dir, f"{char_name}__view_{_tag.lower()}.png"))
            for _g in sorted(_glob.glob(os.path.join(char_root, "TRELLIS_*.glb")),
                             key=os.path.getmtime):
                _tag = "raw" if "TRELLIS_RAW" in os.path.basename(_g) else "textured"
                _dst = os.path.join(run_dir, f"{char_name}__trellis_{_tag}.glb")
                try:
                    _sh.move(_g, _dst)
                except Exception:
                    _sh.copy2(_g, _dst)
            print(f"[FBXExport] filed inputs + Trellis GLB into {run_dir}")
        except Exception as _ce:
            print(f"[FBXExport] could not file the run inputs ({_ce!r}) - continuing")

        # FBX lives inside the run folder alongside its textures so the .fbm
        # folder (created by Blender's FBX exporter) stays self-contained per run.
        fbx_path  = os.path.join(run_dir, f"{char_name}.fbx")
        base      = os.path.join(run_dir, char_name)   # texture filename prefix
        print(f"[FBXExport] Character root : {char_root}")
        print(f"[FBXExport] Run folder     : {run_dir}")
        print(f"[FBXExport] FBX            : {fbx_path}")

        tmp_dir = tempfile.mkdtemp()
        in_glb  = os.path.join(tmp_dir, "in.glb")

        if not obj_path or not os.path.exists(obj_path):
            raise RuntimeError(
                f"[FBXExport] obj_path is not connected or file does not exist: {obj_path!r}\n"
                "Connect the obj_path output from Trellis2BlenderSmartUV.")
        use_obj = True
        print(f"[FBXExport] Using OBJ from BlenderSmartUV (sharp edges preserved): {obj_path}")

        # ── Save texture maps as PNGs next to the FBX ─────────────────────────
        def save_image_tensor(tensor, path):
            if tensor is None:
                return False
            try:
                arr = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(arr).save(path)
                print(f"[FBXExport] Saved: {path}")
                return True
            except Exception as e:
                print(f"[FBXExport] Could not save {path}: {e}")
                return False

        albedo_png  = base + "_albedo.png"
        mr_png      = base + "_metallic_roughness.png"
        nrm_png     = base + "_normal.png"       # DirectX convention (UE5)
        nrm_gl_png  = base + "_normal_gl.png"    # OpenGL convention (Blender) — G channel flipped

        has_albedo = save_image_tensor(base_color,         albedo_png)
        has_mr     = save_image_tensor(metallic_roughness, mr_png)
        has_nrm    = save_image_tensor(normal_map,         nrm_png)

        # Generate the OpenGL version by flipping the G channel of the DX normal map.
        # Blender's Normal Map node expects OpenGL convention; saving it pre-flipped means
        # we can wire it with a single NormalMap node and no extra SeparateColor nodes.
        has_nrm_gl = False
        if has_nrm:
            try:
                import numpy as _np2
                _nrm_arr = _np2.array(PILImage.open(nrm_png))          # H×W×3 uint8
                _nrm_gl  = _nrm_arr.copy()
                _nrm_gl[:, :, 1] = 255 - _nrm_arr[:, :, 1]            # flip G
                PILImage.fromarray(_nrm_gl).save(nrm_gl_png)
                has_nrm_gl = True
                print(f"[FBXExport] Saved GL normal: {nrm_gl_png}")
            except Exception as _ngl_e:
                print(f"[FBXExport] Could not save GL normal: {_ngl_e}")

        # ── KAN-18: genital border skin-match ─────────────────────────────────────
        # The genital is a separate material (slot 2). To kill the hard/obvious border, composite the genital
        # albedo + roughness OVER the BODY maps using the genital feather alpha (baked by genital_texture). Both
        # share the body UVMap atlas layout, so this is pixel-aligned: where alpha<1 the genital map fades into
        # the EXACT body skin pixels the body material shows -> the material seam is invisible (no skin mismatch,
        # no jagged edge), with a soft ramp into the genital detail. Non-fatal.
        try:
            _gt_dir = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/genital_tex'
            _g_alpha = os.path.join(_gt_dir, 'genital_alpha.png')
            _g_alb   = os.path.join(_gt_dir, 'genital_albedo.png')
            if has_albedo and os.path.exists(_g_alpha) and os.path.exists(_g_alb):
                # INPAINT the body atlas's occluded crotch fill: the multiview texturing can't see between the
                # legs, so it leaves a flat 'band-aid' patch that doesn't match the surrounding skin. Inpaint it
                # (+ the MR map) from the surrounding skin using the genital feather alpha (dilated) as the mask,
                # so the crotch reads as continuous body -- fixes the patch AND the local seam/jump. The genital
                # detail is composited back on top below. Overwrites albedo_png so the BODY material gets it too.
                try:
                    import cv2 as _cv2, numpy as _npc
                    _bimg = PILImage.open(albedo_png).convert('RGB'); _S = _bimg.size
                    _amask = _npc.array(PILImage.open(_g_alpha).convert('L').resize(_S))
                    _msk = (_amask > 10).astype(_npc.uint8)
                    # COST + SANITY GUARD. TELEA inpaint is superlinear in radius and mask area; at
                    # radius 40 over a 4096 atlas it ran ~5 min per map. It is also only meaningful when
                    # the mask is a LOCAL crotch feather. Measured on a real bake the mask covered 38% of
                    # the atlas before dilation and 69% after -- a whole-body repaint, not a crotch fill.
                    _base = float(_msk.mean())
                    if _base > 0.12:
                        raise RuntimeError(f"genital alpha covers {_base*100:.0f}% of the atlas -- not a "
                                           f"local feather; refusing to inpaint the whole body")
                    # grow only until the mask stops being local (<= 2x its own area), never a fixed count
                    _it_max = max(8, int(_S[0] / 2048 * 16))
                    _grown = _msk
                    for _i in range(_it_max):
                        _try = _cv2.dilate(_grown, _npc.ones((9, 9), _npc.uint8), iterations=1)
                        if float(_try.mean()) > 2.0 * _base:
                            break
                        _grown = _try
                    _msk = _grown
                    _rad = max(6, min(12, int(_S[0] / 2048 * 8)))
                    # inpaint only the mask's bounding box -- cv2 processes the whole image otherwise
                    _ys, _xs = _npc.nonzero(_msk)
                    _m = _rad + 8
                    _x0, _x1 = max(0, int(_xs.min()) - _m), min(_S[0], int(_xs.max()) + _m)
                    _y0, _y1 = max(0, int(_ys.min()) - _m), min(_S[1], int(_ys.max()) + _m)
                    print(f"[FBXExport] crotch inpaint: mask {_base*100:.2f}% -> {_msk.mean()*100:.2f}%, "
                          f"radius {_rad}, tile {_x1-_x0}x{_y1-_y0} of {_S[0]}x{_S[1]}", flush=True)
                    def _inp_tile(_pil):
                        _arr = _npc.array(_pil)
                        _t = _cv2.inpaint(_arr[_y0:_y1, _x0:_x1].copy(), _msk[_y0:_y1, _x0:_x1],
                                          _rad, _cv2.INPAINT_TELEA)
                        _arr[_y0:_y1, _x0:_x1] = _t
                        return PILImage.fromarray(_arr)
                    _inp_tile(_bimg).save(albedo_png)
                    if has_mr:
                        _inp_tile(PILImage.open(mr_png).convert('RGB')).save(mr_png)
                    print("[FBXExport] Crotch occlusion-fill inpainted from surrounding skin")
                except Exception as _inp_e:
                    print(f"[FBXExport] Crotch inpaint skipped: {_inp_e}")
                _gA = PILImage.open(_g_alb).convert('RGB')
                _al = PILImage.open(_g_alpha).convert('L').resize(_gA.size)
                _bA = PILImage.open(albedo_png).convert('RGB').resize(_gA.size)   # now inpainted (real skin)
                # SKIN-MATCH the outer labia (majora): they should read as the surrounding skin, slightly darker
                # -- not a generic warm tone. Re-tint zones 1,2 to the LOCAL inpainted body skin x a darken
                # factor, so they blend with the per-character skin gradient. Inner labia/opening/anus (3-6) keep
                # their pigment. _OUTER_DARK = tunable (closer to 1.0 = matches skin, lower = darker).
                try:
                    import numpy as _npz
                    _zmp = os.path.join(_gt_dir, 'genital_zonemap.npy')
                    if os.path.exists(_zmp):
                        _zm = _npz.load(_zmp)
                        if _zm.shape[:2] != (_gA.size[1], _gA.size[0]):
                            _zm = _npz.array(PILImage.fromarray(_zm.astype('uint8')).resize(_gA.size, PILImage.NEAREST))
                        _gaa = _npz.array(_gA).astype(float); _baa = _npz.array(_bA).astype(float)
                        _OUTER_DARK = (0.95, 0.90)   # (zone1 outer edge, zone2 inner edge of the majora)
                        _gaa[_zm == 1] = _baa[_zm == 1] * _OUTER_DARK[0]
                        _gaa[_zm == 2] = _baa[_zm == 2] * _OUTER_DARK[1]
                        _gA = PILImage.fromarray(_npz.clip(_gaa, 0, 255).astype('uint8'))
                        print("[FBXExport] Outer labia skin-matched to body (x%.2f/%.2f)" % _OUTER_DARK)
                except Exception as _smt_e:
                    print(f"[FBXExport] Labia skin-match skipped: {_smt_e}")
                PILImage.composite(_gA, _bA, _al).save(_g_alb)
                _g_rou = os.path.join(_gt_dir, 'genital_roughness.png')
                if has_mr and os.path.exists(_g_rou):
                    _gR = PILImage.open(_g_rou).convert('RGB')
                    _bMg = PILImage.open(mr_png).convert('RGB').resize(_gA.size).split()[1]  # glTF roughness = G
                    PILImage.composite(_gR, PILImage.merge('RGB', (_bMg, _bMg, _bMg)), _al).save(_g_rou)
                print("[FBXExport] Genital border skin-matched to body via feather alpha")
        except Exception as _gsm_e:
            print(f"[FBXExport] Genital skin-match skipped: {_gsm_e}")


        # ── Eye detection + texture generation ────────────────────────────────────
        eye_info       = None
        eye_tex_l_pil  = None
        eye_tex_r_pil  = None
        eye_tex_l_path = ""
        eye_tex_r_path = ""

        if front_image is not None:
            try:
                # Convert ComfyUI IMAGE tensor → PIL
                arr_front = (front_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                front_pil = PILImage.fromarray(arr_front)
                eye_info = _detect_eye_info(front_pil)
            except Exception as _ede:
                print(f"[FBXExport] Eye detection error: {_ede}")

        if eye_info is not None:
            try:
                eye_tex_l_pil = _generate_eye_texture(eye_info['image_left_color'])
                eye_tex_r_pil = _generate_eye_texture(eye_info['image_right_color'])
                eye_tex_l_path = base + "_eye_left.png"
                eye_tex_r_path = base + "_eye_right.png"
                eye_tex_l_pil.save(eye_tex_l_path)
                eye_tex_r_pil.save(eye_tex_r_path)
                print(f"[FBXExport] Eye textures saved: {eye_tex_l_path}, {eye_tex_r_path}")
            except Exception as _ete:
                print(f"[FBXExport] Eye texture generation error: {_ete}")
                eye_info = None  # skip Blender eye step if texture failed

        # ── Serialise eye_info for Blender f-string ───────────────────────────
        # We embed it as a Python dict literal so the Blender script can use it directly.
        def _plain(o):
            """numpy scalars repr as `np.int64(74)` under NumPy 2, and the generated Blender
            script has no numpy import -> NameError, the script dies before export_scene.fbx,
            and (because Blender does not put script errors in its exit code) the wrapper
            happily printed "Done" with no FBX on disk. Coerce to plain Python first."""
            if isinstance(o, dict):
                return {k: _plain(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return type(o)(_plain(v) for v in o)
            if isinstance(o, np.generic):
                return o.item()
            return o
        _eye_info_repr  = repr(_plain(eye_info))   # None or a plain-Python dict literal
        _eye_tex_l_path = eye_tex_l_path
        _eye_tex_r_path = eye_tex_r_path
        _ortho_scale    = ortho_scale
        _faceregion_npy = os.path.join(
            r"C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer",
            "_gs_faceregion.npy")

        sharp_rad = sharp_angle * 3.14159265 / 180.0
        # Pass booleans + paths into the f-string cleanly
        _use_obj   = use_obj
        _src_path  = obj_path if use_obj else in_glb

        blender_script = f"""
import bpy, sys, bmesh, os

# ── Clear scene ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import mesh ───────────────────────────────────────────────────────────────
use_obj = {_use_obj}
if use_obj:
    # OBJ import: smoothing groups → sharp edges (topology-transferred, exact)
    bpy.ops.wm.obj_import(filepath=r'{_src_path}')
    print("[FBXExport] Imported OBJ — sharp edges from smoothing groups")
else:
    # GLB fallback: no sharp edge data, will mark by angle below
    bpy.ops.import_scene.gltf(filepath=r'{_src_path}')
    print("[FBXExport] Imported GLB — sharp edges will be angle-based")

mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objects:
    print("ERROR: no mesh"); sys.exit(1)
bpy.ops.object.select_all(action='DESELECT')
for o in mesh_objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.active_object
print(f"[FBXExport] Mesh: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")

# ── Sharp edges ───────────────────────────────────────────────────────────────
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.edges.ensure_lookup_table()
if use_obj:
    # Sharp edges already set by OBJ smoothing group import — just count them
    n_sharp = sum(1 for e in bm.edges if not e.smooth)
    print(f"[FBXExport] Sharp edges from OBJ smoothing groups: {{n_sharp}}")
else:
    # Angle-based fallback
    n_sharp = 0
    for edge in bm.edges:
        if len(edge.link_faces) < 2:
            edge.smooth = False; n_sharp += 1; continue
        if edge.link_faces[0].normal.angle(edge.link_faces[1].normal) > {sharp_rad:.6f}:
            edge.smooth = False; n_sharp += 1
    bm.to_mesh(obj.data)
    print(f"[FBXExport] Sharp edges by angle (>{sharp_angle}°): {{n_sharp}}")
bm.free()
bpy.ops.object.shade_smooth()
print(f"[FBXExport] Shade smooth applied")

# ── Build full PBR material ───────────────────────────────────────────────────
mat = bpy.data.materials.new(name="Character")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf  = nodes.get("Principled BSDF")

# --- Prevent silhouette glow / halo -------------------------------------------
# 1. Backface culling: back-faces at the mesh perimeter are not rendered, so they
#    cannot bleed colour around the silhouette edges.
mat.use_backface_culling = True

# 2. Subsurface scattering: Blender 4.x Principled BSDF has SSS on by default;
#    at thin silhouette edges it produces a characteristic skin-glow / rim effect.
#    Zero it out — we apply colour grading in the texture pipeline, not via SSS.
for _sss_name in ('Subsurface Weight', 'Subsurface'):   # 4.x / 3.x API names
    _sss = bsdf.inputs.get(_sss_name)
    if _sss is not None:
        _sss.default_value = 0.0
        break

# 3. Alpha-clip mode: the base-colour texture has alpha=255 inside the UV island
#    and alpha=0 in the seam-bleed zone.  Wiring the alpha here and using CLIP
#    mode gives a clean geometric cutoff at the UV island boundary — the bleed
#    zone pixels are discarded by the renderer so their colour never shows at the
#    mesh silhouette.  threshold=0.1 is permissive enough to avoid hard pixelation
#    while still clipping the fully-transparent (alpha=0) bleed pixels.
mat.blend_method    = 'CLIP'
mat.alpha_threshold = 0.1
# (alpha input wired below, after albedo_node is created)
# ------------------------------------------------------------------------------

def load_tex(path, label, colorspace='sRGB'):
    if not path or not os.path.exists(path):
        return None
    # If Blender already has this filename cached (e.g. from a previous run in the
    # same session), reload from disk so stale versions never show up in the viewport.
    basename = os.path.basename(path)
    existing = bpy.data.images.get(basename)
    if existing is not None:
        existing.filepath = path
        existing.reload()
        img = existing
    else:
        img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    node = nodes.new('ShaderNodeTexImage')
    node.image = img
    node.label = label
    return node

albedo_node = load_tex(r'{albedo_png}',  'Albedo',          'sRGB')
mr_node     = load_tex(r'{mr_png}',      'Metallic-Roughness', 'Non-Color')
# Use the pre-flipped OpenGL normal map so Blender's NormalMap node works without
# extra SeparateColor/Invert nodes (those break path_mode='COPY' texture bundling).
nrm_gl_path = r'{nrm_gl_png}'
nrm_dx_path = r'{nrm_png}'
nrm_path    = nrm_gl_path if os.path.exists(nrm_gl_path) else nrm_dx_path
nrm_node    = load_tex(nrm_path, 'Normal (OpenGL)', 'Non-Color')

if albedo_node:
    links.new(bsdf.inputs['Base Color'], albedo_node.outputs['Color'])
    # Wire texture alpha → BSDF Alpha so the alpha-clip mode can discard the
    # seam-bleed zone pixels (alpha=0 there) without affecting the face surface.
    links.new(bsdf.inputs['Alpha'], albedo_node.outputs['Alpha'])
    print("[FBXExport] Albedo + Alpha wired")

if mr_node:
    # glTF packing: G = Roughness, B = Metallic
    sep = nodes.new('ShaderNodeSeparateColor')
    links.new(sep.inputs['Color'], mr_node.outputs['Color'])
    links.new(bsdf.inputs['Roughness'], sep.outputs['Green'])
    links.new(bsdf.inputs['Metallic'],  sep.outputs['Blue'])
    print("[FBXExport] Metallic-roughness wired")

if nrm_node:
    # The _normal_gl.png is already in OpenGL convention (G flipped vs DX).
    # Plug straight into ShaderNodeNormalMap — no extra nodes needed.
    nrm_node_cvt = nodes.new('ShaderNodeNormalMap')
    links.new(nrm_node_cvt.inputs['Color'], nrm_node.outputs['Color'])
    links.new(bsdf.inputs['Normal'], nrm_node_cvt.outputs['Normal'])
    print(f"[FBXExport] Normal map wired (OpenGL pre-flipped): {{nrm_path}}")

# Clear ALL material slots, add ours, assign to every face.
# After join(), Blender merges slots from all sub-objects — faces may point
# to slot 1 or 2, so replacing slot 0 alone leaves most faces with no material.
obj.data.materials.clear()
obj.data.materials.append(mat)
for poly in obj.data.polygons:
    poly.material_index = 0
print(f"[FBXExport] Material assigned to all {{len(obj.data.polygons)}} faces")

# ── KAN-8: HAIR MATERIAL ──────────────────────────────────────────────────────
# Give the hair faces a separate BLONDE material so hair colour comes from the
# MATERIAL, not the collision-prone UV texture (the root-cause fix).  The flags are
# from gen_seams, computed on THIS same mesh (last_seams), so they map by face index.
# Wrapped so a mismatch/error can never break the export -- it just skips the slot.
try:
    import numpy as _nph
    _hf_path = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/last_seams_hairfaces.npy'
    # HAIR MATERIAL RETIRED (2026-09-04) -- hair colour now comes from the generated back and
    # side views like every other surface, so the hair needs no material of its own. Retiring
    # it also removes the coverage-reclaim below, which was the documented cause of a hair
    # side going missing (it dropped 10784 of 24367 painted hair faces on the last bake and
    # tripped its own tripwire). Flip _HAIR_MAT_ON to re-enable both.
    _HAIR_MAT_ON = False
    if _HAIR_MAT_ON and os.path.exists(_hf_path):
        _hflags = _nph.load(_hf_path)
        _npoly = len(obj.data.polygons)
        if len(_hflags) == _npoly:
            # COVERAGE-BASED RECLAIM (orientation-free, deterministic, per-character):
            # gen_seams over-flags front skin (face/chest) as hair, non-deterministically.
            # A hair-flagged face that the FRONT projection actually PAINTED (non-black in
            # the albedo atlas) is SKIN -> strip it from the hair material.  Un-painted
            # hair faces are the REAL hair (back/sides/scalp the front can't reach) -> keep.
            # The atlas v-orientation is SELF-CALIBRATED: of the two flips, the correct one
            # is the one whose painted faces' normals cluster on ONE hemisphere (the front).
            try:
                _aimg = albedo_node.image if albedo_node is not None else None
                if _aimg is None:
                    raise RuntimeError("no albedo image to test coverage")
                _aw = int(_aimg.size[0]); _ah = int(_aimg.size[1])
                _ap = _nph.empty(_aw * _ah * 4, _nph.float32); _aimg.pixels.foreach_get(_ap)
                _alb = _ap.reshape(_ah, _aw, 4)[:, :, :3]              # Blender pixels: row 0 = bottom
                _nl = len(obj.data.loops)
                _luv = _nph.empty(_nl * 2, _nph.float32)
                obj.data.uv_layers.active.data.foreach_get('uv', _luv); _luv = _luv.reshape(-1, 2)
                _lsr = _nph.empty(_npoly, _nph.int32); obj.data.polygons.foreach_get('loop_start', _lsr)
                _ltl = _nph.empty(_npoly, _nph.int32); obj.data.polygons.foreach_get('loop_total', _ltl)
                _cuv = _nph.add.reduceat(_luv, _lsr, axis=0) / _nph.maximum(_ltl, 1)[:, None]
                _fnr = _nph.empty(_npoly * 3, _nph.float32); obj.data.polygons.foreach_get('normal', _fnr); _fnr = _fnr.reshape(-1, 3)
                _pxc = _nph.clip((_cuv[:, 0] * _aw).astype(int), 0, _aw - 1)
                _best = None; _bestc = -1.0
                for _flip in (0, 1):
                    _vv = (1.0 - _cuv[:, 1]) if _flip else _cuv[:, 1]
                    _pyc = _nph.clip((_vv * _ah).astype(int), 0, _ah - 1)
                    _pn = _alb[_pyc, _pxc].sum(1) > 0.03                # non-black => front projection painted it
                    _conc = float(_nph.linalg.norm(_fnr[_pn].mean(0))) if _pn.any() else 0.0
                    if _conc > _bestc:
                        _bestc = _conc; _best = _pn
                # FRONT-FACING GATE (deterministic, axis-agnostic): only reclaim painted hair faces that
                # actually FACE THE FRONT CAMERA -- the front-projected skin / front-hair the user gets from
                # the image anyway. NEVER reclaim side/back-facing hair: that is the un-imaged region the user
                # FILLS, and removing it per-bake (by which faces the projection happened to paint) WAS the
                # "side missing" bug -- same character, R side dropped 4976->880 one bake, fine the next.
                # Verified on last_seams vs the real albedo: front-gate keeps both sides balanced (L4311/R4445)
                # where the old all-painted reclaim gutted R (L1407/R880). _fdir is the painted faces' mean
                # normal = the front-camera direction, so this works in any axis convention (OBJ vs blend).
                _fdir = _fnr[_best].mean(0); _fdir = _fdir / (_nph.linalg.norm(_fdir) + 1e-9)
                _facing = _fnr @ _fdir                              # +1 faces the front camera, 0 side, -1 back
                _reclaim = _best & _hflags & (_facing > 0.5)        # front-facing painted hair only
                _nb0 = int(_hflags.sum()); _hflags = _hflags & (~_reclaim)
                print(f"[FBXExport] coverage reclaim (front-facing gate): {{_nb0}} -> {{int(_hflags.sum())}} hair faces ({{int(_reclaim.sum())}} front-painted reclaimed, concentration={{_bestc:.2f}})")
                # TRIPWIRE: the reclaim must never gut the hair region. Removing >40% of it almost always means
                # a side was dropped (the recurring failure) -> SCREAM at the bake, don't discover it by eye.
                if int(_hflags.sum()) < 0.60 * max(1, _nb0):
                    print(f"[FBXExport] *** HAIR-RECLAIM TRIPWIRE: kept {{int(_hflags.sum())}}/{{_nb0}} (<60%) -- a side may be missing, investigate! ***")
            except Exception as _cre:
                import traceback as _tb
                print(f"[FBXExport] coverage reclaim SKIPPED: {{_cre!r}}"); _tb.print_exc()
            _hmat = bpy.data.materials.new(name="Hair")
            _hmat.use_nodes = True
            _hb = _hmat.node_tree.nodes.get("Principled BSDF")
            _hb.inputs['Base Color'].default_value = (228/255.0, 198/255.0, 120/255.0, 1.0)
            _hb.inputs['Roughness'].default_value = 0.65
            for _sn in ('Subsurface Weight', 'Subsurface'):
                _si = _hb.inputs.get(_sn)
                if _si is not None:
                    _si.default_value = 0.0
                    break
            # Give the hair the SAME surface form as the skin: reuse the normal +
            # roughness maps (new texture nodes in the hair node-tree pointing at the
            # SAME image datablocks).  Base Color stays the flat blonde above, so hair
            # colour is still material-driven (no UV-collision bleed) -- but the hair now
            # catches light with strand form instead of reading as a flat sheet.
            _hnt = _hmat.node_tree; _hnodes = _hnt.nodes; _hlinks = _hnt.links
            if nrm_node is not None and nrm_node.image is not None:
                _hn_tex = _hnodes.new('ShaderNodeTexImage')
                _hn_tex.image = nrm_node.image; _hn_tex.label = 'Normal (OpenGL)'
                _hn_tex.image.colorspace_settings.name = 'Non-Color'
                _hn_nm = _hnodes.new('ShaderNodeNormalMap')
                _hlinks.new(_hn_nm.inputs['Color'], _hn_tex.outputs['Color'])
                _hlinks.new(_hb.inputs['Normal'], _hn_nm.outputs['Normal'])
                print("[FBXExport] Hair material: normal map wired")
            if mr_node is not None and mr_node.image is not None:
                _hmr_tex = _hnodes.new('ShaderNodeTexImage')
                _hmr_tex.image = mr_node.image; _hmr_tex.label = 'Metallic-Roughness'
                _hmr_tex.image.colorspace_settings.name = 'Non-Color'
                _hsep = _hnodes.new('ShaderNodeSeparateColor')
                _hlinks.new(_hsep.inputs['Color'], _hmr_tex.outputs['Color'])
                _hlinks.new(_hb.inputs['Roughness'], _hsep.outputs['Green'])
                print("[FBXExport] Hair material: roughness map wired")
            obj.data.materials.append(_hmat)                       # slot 1 (Character = slot 0)
            _hslot = len(obj.data.materials) - 1
            _mi = _nph.where(_hflags, _hslot, 0).astype('int32')
            obj.data.polygons.foreach_set("material_index", _mi)
            obj.data.update()
            print(f"[FBXExport] Hair material -> {{int(_hflags.sum())}}/{{_npoly}} faces (slot {{_hslot}})")
        else:
            print(f"[FBXExport] Hair material SKIPPED: flags {{len(_hflags)}} != polys {{_npoly}}")
    else:
        print("[FBXExport] Hair material SKIPPED: no last_seams_hairfaces.npy")
except Exception as _hme:
    print(f"[FBXExport] Hair material SKIPPED ({{_hme!r}})")

# ── KAN-18: GENITAL MATERIAL ──────────────────────────────────────────────────
# Separate material (slot 2) on the genital island faces, sampling the genital maps that
# genital_texture.py baked onto the BODY UVMap. Mirrors the Hair material above. Non-fatal;
# silently skips if the flag/maps are absent (genital step disabled or failed).
try:
    import numpy as _npg
    _gf_path = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/_gs_genital_mat.npy'
    _gtex = r'C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/genital_tex'
    _galb = os.path.join(_gtex, 'genital_albedo.png')
    # OFF by default (2026-09-09). This overlay is currently WORSE than the body atlas it covers:
    # genital_albedo.png is baked at 2048 against a 4096 body atlas (so its edges read as blocky
    # steps), large parts of it are unpainted BLACK -- which is the black patch that shows above the
    # crotch -- and its painted zones do not line up with where the anatomy actually sits on the
    # genital island. The body atlas underneath is resampled from Trellis's own texture, is aligned
    # by construction and is full resolution, so covering it with this loses on every count.
    # Re-enable with env GENITAL_MATERIAL=1 once genital_texture.py bakes at the atlas resolution
    # and its zones register with the island.
    if os.environ.get("GENITAL_MATERIAL", "0") != "1":
        print("[FBXExport] Genital material DISABLED (GENITAL_MATERIAL=0) -- the body atlas covers "
              "that region; the 2048 overlay was blocky, partly black and misregistered.")
    elif os.path.exists(_gf_path) and os.path.exists(_galb):
        _gflags = _npg.load(_gf_path).astype(bool)
        _gnp = len(obj.data.polygons)
        if len(_gflags) == _gnp:
            _gmat = bpy.data.materials.new(name="Genital"); _gmat.use_nodes = True
            _gnt = _gmat.node_tree; _gbb = _gnt.nodes.get("Principled BSDF")
            _gat = _gnt.nodes.new('ShaderNodeTexImage'); _gat.image = bpy.data.images.load(_galb)
            _gnt.links.new(_gbb.inputs['Base Color'], _gat.outputs['Color'])
            _grp = os.path.join(_gtex, 'genital_roughness.png')
            if os.path.exists(_grp):
                _grt = _gnt.nodes.new('ShaderNodeTexImage'); _grt.image = bpy.data.images.load(_grp)
                _grt.image.colorspace_settings.name = 'Non-Color'
                _grs = _gnt.nodes.new('ShaderNodeSeparateColor'); _gnt.links.new(_grs.inputs['Color'], _grt.outputs['Color'])
                _gnt.links.new(_gbb.inputs['Roughness'], _grs.outputs['Red'])
            _gnrm = os.path.join(_gtex, 'genital_normal.png')
            if os.path.exists(_gnrm):
                _gtt = _gnt.nodes.new('ShaderNodeTexImage'); _gtt.image = bpy.data.images.load(_gnrm)
                _gtt.image.colorspace_settings.name = 'Non-Color'
                _gnm = _gnt.nodes.new('ShaderNodeNormalMap'); _gnt.links.new(_gnm.inputs['Color'], _gtt.outputs['Color'])
                _gnt.links.new(_gbb.inputs['Normal'], _gnm.outputs['Normal'])
            obj.data.materials.append(_gmat); _gslot = len(obj.data.materials) - 1
            _gmi = _npg.empty(_gnp, _npg.int32); obj.data.polygons.foreach_get('material_index', _gmi)
            _gmi[_gflags] = _gslot; obj.data.polygons.foreach_set('material_index', _gmi); obj.data.update()
            print(f"[FBXExport] Genital material -> {{int(_gflags.sum())}}/{{_gnp}} faces (slot {{_gslot}})")
        else:
            print(f"[FBXExport] Genital material SKIPPED: flags {{len(_gflags)}} != polys {{_gnp}}")
    else:
        print("[FBXExport] Genital material SKIPPED: no flag/maps (genital step off?)")
except Exception as _gme:
    print(f"[FBXExport] Genital material SKIPPED ({{_gme!r}})")

# ── Auto eye generation ────────────────────────────────────────────────────────
import mathutils

_eye_info       = {_eye_info_repr}
_eye_tex_l_path = r'{_eye_tex_l_path}'
_eye_tex_r_path = r'{_eye_tex_r_path}'
_ortho_scale    = {_ortho_scale}

_FACE_MASK = None
try:
    import numpy as _np_fm
    _fm_p = r"{_faceregion_npy}"
    if _fm_p and os.path.exists(_fm_p):
        _FACE_MASK = _np_fm.load(_fm_p)
        print("[EyeAdd] face-region mask loaded:", int(_FACE_MASK.sum()), "faces")
except Exception as _fm_e:
    print("[EyeAdd] face-region mask unavailable:", _fm_e)


def _uv_to_3d(uv_x, uv_y, mesh_obj, ortho_scale):
    # Image-UV -> a point on the mesh. Self-calibrating and frame-agnostic.
    # Two bugs made this miss 100% of the time (measured 0/2) once eye detection
    # finally started producing input:
    #  * SCALE: (0.5-uv_y)*ortho_scale = 0.5025 against a head topping out at 0.501,
    #    so the ray passed over the skull. ortho_scale is the projector clip scale,
    #    not this mesh extent. Use the mesh own bounding box: the character fills the
    #    frame (char_h 894 of 896px), so image fraction maps straight onto it.
    #  * SIDE: it cast from world +Y, which on this mesh is local -Z, the BACK of the
    #    head. Cast from both sides and keep the hit that lands on the face region.
    _bb = [mesh_obj.matrix_world @ mathutils.Vector(c) for c in mesh_obj.bound_box]
    _xs = [q.x for q in _bb]; _ys = [q.y for q in _bb]; _zs = [q.z for q in _bb]
    lo = [min(_xs), min(_ys), min(_zs)]
    hi = [max(_xs), max(_ys), max(_zs)]
    ext = [hi[i] - lo[i] for i in range(3)]
    UP  = max(range(3), key=lambda i: ext[i])
    DEP = min(range(3), key=lambda i: ext[i])
    LAT = 3 - UP - DEP
    base = [0.0, 0.0, 0.0]
    base[LAT] = lo[LAT] + uv_x * ext[LAT]
    base[UP]  = hi[UP]  - uv_y * ext[UP]
    mat_inv = mesh_obj.matrix_world.inverted()
    best = None
    for _sgn in (1, -1):
        o = list(base)
        o[DEP] = (hi[DEP] + 1.0) if _sgn > 0 else (lo[DEP] - 1.0)
        d = [0.0, 0.0, 0.0]
        d[DEP] = -1.0 if _sgn > 0 else 1.0
        _ro = mat_inv @ mathutils.Vector(o)
        _rd = (mat_inv.to_3x3() @ mathutils.Vector(d)); _rd.normalize()
        hit, loc, nrm, fidx = mesh_obj.ray_cast(_ro, _rd)
        if not hit:
            continue
        on_face = False
        if _FACE_MASK is not None and 0 <= fidx < len(_FACE_MASK):
            on_face = bool(_FACE_MASK[fidx])
        cand = (mesh_obj.matrix_world @ loc,
                (mesh_obj.matrix_world.to_3x3() @ nrm).normalized(), on_face)
        if best is None or (on_face and not best[2]):
            best = cand
    if best is None:
        return None, None
    if _FACE_MASK is not None and not best[2]:
        print("[EyeAdd] WARNING: ray hit the mesh but not the face region")
    return best[0], best[1]

def _add_eyeball(name, center, radius, iris_color_rgb, tex_path):
    # Add a UV-sphere eyeball using bmesh directly — works in --background mode
    # (bpy.ops.mesh.primitive_uv_sphere_add requires a viewport context).
    import bmesh as _bmesh
    _mesh = bpy.data.meshes.new(name + "_mesh")
    _bm   = _bmesh.new()
    _bmesh.ops.create_uvsphere(_bm, u_segments=32, v_segments=16, radius=radius)
    _bm.to_mesh(_mesh)
    _bm.free()
    eye_obj = bpy.data.objects.new(name, _mesh)
    eye_obj.location = center
    bpy.context.collection.objects.link(eye_obj)
    print(f"[EyeAdd] {{name}}: sphere created via bmesh at {{center}}")

    eye_mat = bpy.data.materials.new(name=name + "_mat")
    eye_mat.use_nodes = True
    enodes = eye_mat.node_tree.nodes
    elinks = eye_mat.node_tree.links
    ebsdf  = enodes.get("Principled BSDF")
    ebsdf.inputs['Roughness'].default_value = 0.08
    for _s in ('Specular IOR Level', 'Specular'):
        _si = ebsdf.inputs.get(_s)
        if _si: _si.default_value = 0.9; break
    for _sss_n in ('Subsurface Weight', 'Subsurface'):
        _sss_i = ebsdf.inputs.get(_sss_n)
        if _sss_i: _sss_i.default_value = 0.0; break

    if tex_path and os.path.exists(tex_path):
        tex_node = enodes.new('ShaderNodeTexImage')
        _img = bpy.data.images.load(tex_path)
        tex_node.image = _img
        elinks.new(ebsdf.inputs['Base Color'], tex_node.outputs['Color'])
        print(f"[EyeAdd] {{name}}: texture loaded from {{tex_path}}")
    else:
        r, g, b = iris_color_rgb
        ebsdf.inputs['Base Color'].default_value = (r/255, g/255, b/255, 1.0)
        print(f"[EyeAdd] {{name}}: using flat iris colour {{iris_color_rgb}}")

    eye_obj.data.materials.append(eye_mat)
    return eye_obj

if _eye_info:
    _eye_sides = [
        ('eye_L', 'image_left_uv',  'image_left_radius',  'image_left_color',  _eye_tex_l_path),
        ('eye_R', 'image_right_uv', 'image_right_radius', 'image_right_color', _eye_tex_r_path),
    ]
    for _ename, _uv_key, _r_key, _c_key, _tex in _eye_sides:
        _uv  = _eye_info.get(_uv_key)
        _col = _eye_info.get(_c_key, (80, 55, 35))
        _r   = _eye_info.get(_r_key, 0.025)
        if _uv is None:
            print(f"[EyeAdd] {{_ename}}: no UV data — skipping")
            continue
        _w_pos, _w_nrm = _uv_to_3d(_uv[0], _uv[1], obj, _ortho_scale)
        if _w_pos is None:
            print(f"[EyeAdd] {{_ename}}: raycast missed — skipping (check ortho_scale)")
            continue
        # Eyeball radius in world units from iris fraction-of-image-width
        _eye_r = _r * _ortho_scale * 1.3
        # Inset the sphere center slightly behind the face surface so it sits in the socket
        _eye_center = _w_pos - _w_nrm * (_eye_r * 0.4)
        _add_eyeball(_ename, _eye_center, _eye_r, _col, _tex)
        print(f"[EyeAdd] {{_ename}}: placed at {{_w_pos}}, r={{_eye_r:.4f}}")
    bpy.ops.object.select_all(action='DESELECT')
    print("[EyeAdd] Eye spheres added successfully")
else:
    print("[EyeAdd] No eye info — skipping eye generation")

# ── Export FBX ────────────────────────────────────────────────────────────────
# path_mode='COPY' copies texture files next to the FBX (no embed_textures —
# that flag is unreliable in Blender 5.1 and silently produces a bad file).
try:
    ret = bpy.ops.export_scene.fbx(
        filepath=r'{fbx_path}',
        use_mesh_modifiers=True,
        mesh_smooth_type='EDGE',
        use_tspace=True,
        path_mode='COPY',
    )
    if 'FINISHED' not in ret:
        print(f"[FBXExport] ERROR: export_scene.fbx returned {{ret}}")
        sys.exit(1)
except Exception as _fbx_e:
    print(f"[FBXExport] EXCEPTION during FBX export: {{_fbx_e}}")
    sys.exit(1)
import os as _os2
if not _os2.path.exists(r'{fbx_path}'):
    print("[FBXExport] ERROR: FBX file not created after export")
    sys.exit(1)
print("[FBXExport] *** FBX saved:", r'{fbx_path}', "***")
"""

        script_path = os.path.join(tmp_dir, "fbx_export.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(blender_script)

        print(f"[FBXExport] Running Blender → {fbx_path}")
        result = subprocess.run(
            # --python-exit-code: without it Blender exits 0 even when the script raises, so a
            # crash in the eye/material code looked like a successful export with no file.
            [BLENDER_EXE, "--background", "--python-exit-code", "1", "--python", script_path],
            capture_output=True, text=True, timeout=300,
            encoding='utf-8', errors='replace'
        )
        for line in (result.stdout or "").splitlines():
            if ("[FBXExport]" in line or "[EyeAdd]" in line
                    or "ERROR" in line.upper() or "WARNING" in line.upper()):
                print(line)
        if result.returncode != 0:
            print("[FBXExport] STDERR:", (result.stderr or "")[-3000:])
            raise RuntimeError(f"Blender FBX export failed with code {result.returncode}")
        if not os.path.exists(fbx_path):
            # The export is the whole point of this node; saying "Done" without the file on disk
            # is how a silent Blender-side crash reached the user as "there is no material".
            print("[FBXExport] STDERR:", (result.stderr or "")[-3000:])
            raise RuntimeError(f"[FBXExport] Blender reported success but {fbx_path} does not exist")

        print(f"[FBXExport] Done → {fbx_path}")
        print(f"[FBXExport]   albedo:            {albedo_png if has_albedo else 'not saved'}")
        print(f"[FBXExport]   metallic_roughness:{mr_png if has_mr else 'not saved'}")
        print(f"[FBXExport]   normal (DX/UE5):   {nrm_png if has_nrm else 'not saved'}")
        print(f"[FBXExport]   normal (GL/Blender):{nrm_gl_png if has_nrm_gl else 'not saved'}")
        if eye_tex_l_path:
            print(f"[FBXExport]   eye left:          {eye_tex_l_path}")
            print(f"[FBXExport]   eye right:         {eye_tex_r_path}")

        # Blender's FBX exporter (path_mode='COPY') only copies textures that are
        # wired directly into FBX-compatible material slots.  MR goes through a
        # SeparateColor node, so Blender silently omits it.  Copy MR and the GL
        # normal map manually so the .fbm folder is complete.
        import shutil
        fbm_dir = fbx_path.replace(".fbx", ".fbm")
        if os.path.isdir(fbm_dir):
            if has_mr:
                mr_dest = os.path.join(fbm_dir, os.path.basename(mr_png))
                shutil.copy2(mr_png, mr_dest)
                print(f"[FBXExport]   MR manually copied → {mr_dest}")
            if has_nrm_gl:
                nrm_gl_dest = os.path.join(fbm_dir, os.path.basename(nrm_gl_png))
                shutil.copy2(nrm_gl_png, nrm_gl_dest)
                print(f"[FBXExport]   GL normal manually copied → {nrm_gl_dest}")
            if eye_tex_l_path and os.path.exists(eye_tex_l_path):
                shutil.copy2(eye_tex_l_path, os.path.join(fbm_dir, os.path.basename(eye_tex_l_path)))
                shutil.copy2(eye_tex_r_path, os.path.join(fbm_dir, os.path.basename(eye_tex_r_path)))
                print(f"[FBXExport]   Eye textures copied → {fbm_dir}")
        else:
            print(f"[FBXExport]   WARNING: .fbm folder not found at {fbm_dir}, skipping manual copies")

        # Build ComfyUI IMAGE tensors for the eye textures so the user can
        # preview and compare them in the workflow.
        def pil_to_tensor(pil_img):
            if pil_img is None:
                # Return a 1×1 black placeholder so the output socket is always valid
                return torch.zeros(1, 1, 1, 3)
            arr = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)

        return (fbx_path, pil_to_tensor(eye_tex_l_pil), pil_to_tensor(eye_tex_r_pil))



class Trellis2RenderClayView:
    """Render an UNTEXTURED mesh from several azimuths, as a Canny/edge reference.

    Trellis2RenderViewFromMesh and Trellis2RenderMultiView both require a baseColorTexture /
    material; at the SmartUV stage the mesh has neither, so both raise. This renders raw
    geometry via clay_multiview.py, which drives the Trellis2 pack's own blender_render.py so
    normalisation (auto_center_and_scale/norm_size) and the ortho camera match the texturing
    stage exactly. All views come from ONE Blender session -- importing a 300k-face OBJ costs
    ~25 s, and doing it per view paid that three times over.

    Shading is the surface NORMAL emitted as colour, not lit clay: lit clay renders as a flat
    white silhouette whose Canny is just an outline, while normals give spine, shoulder blades,
    buttocks and hair grooves.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "obj_path":    ("STRING", {"tooltip": "OBJ from Trellis2BlenderSmartUV (obj_path output)."}),
                "azimuths":    ("STRING", {"default": "90,180,270",
                                           "tooltip": "0=front, 90=left, 180=back, 270=right."}),
                "elevation":   ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "ortho_scale": ("FLOAT", {"default": 1.15, "min": 0.1, "max": 5.0, "step": 0.01,
                                          "tooltip": "Must match the texturing node's ortho_scale."}),
                "norm_size":   ("FLOAT", {"default": 1.15, "min": 0.1, "max": 5.0, "step": 0.01,
                                          "tooltip": "Must match the texturing node's norm_size."}),
                "render_size": ("INT",   {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "shading":     (["normal", "clay"], {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_views",)
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    def execute(self, obj_path, azimuths, elevation, ortho_scale, norm_size, render_size, shading):
        import subprocess, tempfile, numpy as _np, torch as _t
        from PIL import Image as _Img

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clay_multiview.py")
        if not os.path.exists(script):
            raise RuntimeError("[RenderClayView] clay_multiview.py missing at " + script)
        if not obj_path or not os.path.exists(obj_path):
            raise RuntimeError("[RenderClayView] obj_path missing: " + repr(obj_path))

        az = [a.strip() for a in str(azimuths).split(",") if a.strip()]
        td = tempfile.mkdtemp(prefix="clayviews_")
        cmd = [BLENDER_EXE, "--background", "--python", script, "--",
               "--mesh", obj_path, "--outdir", td, "--azims", ",".join(az),
               "--elev", str(float(elevation)), "--scale", str(float(ortho_scale)),
               "--norm_size", str(float(norm_size)), "--resolution", str(int(render_size)),
               "--shading", shading]
        r = subprocess.run(cmd, capture_output=True, text=True)

        frames = []
        for a in az:
            f = os.path.join(td, "view_%d.png" % int(round(float(a))))
            if not os.path.exists(f):
                raise RuntimeError("[RenderClayView] missing view %s: %s%s"
                                   % (a, (r.stdout or "")[-1500:], (r.stderr or "")[-800:]))
            frames.append(_np.asarray(_Img.open(f).convert("RGB")).astype(_np.float32) / 255.0)
        print("[RenderClayView] %d views (%s) shading=%s from %s"
              % (len(frames), ",".join(az), shading, obj_path), flush=True)
        return (_t.from_numpy(_np.stack(frames, 0)),)


class Trellis2MeshGuard:
    """Pass `candidate` through, but fall back to `fallback` if the candidate is corrupt.

    Why: on 2026-09-09 a fresh geometry chain produced a mesh whose X axis was destroyed -- 67% of
    vertices sitting between -1e7 and -1.23e11 while Y and Z stayed perfectly normal, so the body's
    bounding box read [2.99e10, 0.997, 0.514]. The generator was NOT at fault: its own export from
    that same run measured extent [0.8704, 0.9999, 0.1977], ratio 0.198, zero out-of-range verts.
    Something in remesh -> CuMesh fill -> meshlib fill wrecked it, and the first thing to trip over
    it was find_wrist (wrists at -8.7e10), 11 minutes in.

    Hole filling is a nice-to-have. A mesh with a few unfilled holes is enormously better than a
    destroyed one, so when a stage corrupts the mesh this hands the previous stage's mesh onward and
    says loudly which stage did it -- instead of failing the bake. That also identifies the culprit
    on the first occurrence rather than costing a bisect.

    The test is RELATIVE: a stage that fills holes must not change the silhouette, so the candidate
    is judged against the fallback's own bounding box, not against any absolute size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidate": ("MESHWITHVOXEL", {"tooltip": "Output of the stage being guarded."}),
                "fallback":  ("MESHWITHVOXEL", {"tooltip": "That stage's INPUT -- used if it corrupted the mesh."}),
            },
            "optional": {
                "stage_name": ("STRING", {"default": "", "tooltip": "Named in the warning."}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    @staticmethod
    def _extent(m):
        try:
            v = m.vertices
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            v = np.asarray(v, np.float64)
            if v.size == 0:
                return None, None, "empty"
            finite = np.isfinite(v).all(1)
            if not finite.any():
                return None, None, "all vertices non-finite"
            ext = v[finite].max(0) - v[finite].min(0)
            return ext, int((~finite).sum()), None
        except Exception as e:
            return None, None, f"unreadable ({e!r})"

    @staticmethod
    def _shells(m):
        """Connected-component count and the largest component's share.

        Trellis models hair as SEPARATE shells floating over the body -- measured on the raw
        generator output, hair faces at the shoulder share ZERO vertices with the skin under them.
        Something in remesh -> hole fill welds them in: by the time the mesh reaches SmartUV, one
        component already holds 92% of the vertices and spans the whole body, and the shoulder goes
        from 0.8% hair-coloured to 17.9%. That weld is what makes the hair read as short, holed, and
        smeared yellow across the shoulders -- a 1mm shell cannot survive being unioned with a torso.
        Printing the shell count either side of every stage names the stage that does it, at the
        cost of one bake instead of a bisect."""
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            v = m.vertices; f = m.faces
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
            f = np.asarray(f, np.int64)
            n = len(v)
            if n == 0 or len(f) == 0:
                return None, None
            e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
            g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
            ncomp, lab = connected_components(g, directed=False)
            big = int(np.bincount(lab).max())
            return int(ncomp), 100.0 * big / max(n, 1)
        except Exception:
            return None, None

    def execute(self, candidate, fallback, stage_name=""):
        tag = stage_name or "stage"
        ce, cnf, cerr = self._extent(candidate)
        fe, fnf, ferr = self._extent(fallback)
        why = None
        if cerr:
            why = cerr
        elif cnf:
            why = f"{cnf:,} non-finite vertices"
        elif ce is not None:
            if float(ce.min()) <= 0 or float(ce.min() / ce.max()) < 0.02:
                why = (f"degenerate bounding box {np.round(ce, 4).tolist()} "
                       f"(thinnest axis {100 * ce.min() / ce.max():.2f}% of the longest; a body is ~19%)")
            elif fe is not None and float(ce.max()) > 3.0 * float(fe.max()):
                # a hole fill must not change the silhouette
                why = (f"bounding box grew from {np.round(fe, 4).tolist()} to "
                       f"{np.round(ce, 4).tolist()} -- {float(ce.max() / max(fe.max(), 1e-12)):.3g}x")
        if why is None:
            _cs, _cb = self._shells(candidate)
            _fs, _fb = self._shells(fallback)
            _sh = ""
            if _cs is not None and _fs is not None:
                # Reported as a plain sanity metric. NOT evidence about hair: Trellis already ships
                # the hair fused to the body (welding its verts by POSITION, hair and skin at the
                # shoulder share 261 vertices and sit in the same component; the raw mesh is 7
                # shells with the largest holding 93.8%). Our chain leaves that essentially
                # unchanged at 92.0%. An earlier reading of "separate shells" was an artifact of the
                # GLB duplicating vertices at UV seams -- index sharing is not connectivity there.
                _sh = f"  shells {_fs:,}->{_cs:,}  largest {_fb:.1f}%->{_cb:.1f}%"
                if _cb - _fb > 5.0:
                    _sh += "   <== this stage merged geometry that was separate on the way in"
            print(f"[MeshGuard] {tag}: OK  extent={np.round(ce, 4).tolist() if ce is not None else '?'}{_sh}",
                  flush=True)
            return (candidate,)
        # ABORT-ONLY by default. Substituting the previous stage's mesh is a real geometry change --
        # it silently ships a mesh with whatever holes this stage would have filled -- so it must not
        # be on while the pipeline is being judged against a known-good reference. Detecting the
        # corruption is the valuable half and costs nothing: this failure otherwise burns 20+ minutes
        # grinding CUDA and numpy over infinite coordinates before anything notices.
        # Set MESHGUARD_FALLBACK=1 to allow the substitution and let the bake continue.
        if os.environ.get("MESHGUARD_FALLBACK", "0") != "1":
            raise RuntimeError(
                f"[MeshGuard] {tag} CORRUPTED THE MESH: {why}. Stopping here rather than spending "
                f"the rest of the bake on it. {tag} is the stage to fix. "
                f"Set MESHGUARD_FALLBACK=1 to continue from this stage's INPUT instead "
                f"(you keep whatever holes {tag} would have filled).")

        # The fallback is only a fallback if it is itself sound. If the mesh was ALREADY broken on
        # the way in, this stage merely inherited it -- say so and stop, rather than passing a
        # corrupt mesh downstream under the banner of having recovered.
        _fbad = ferr
        if not _fbad and fnf:
            _fbad = f"{fnf:,} non-finite vertices"
        if not _fbad and fe is not None and (float(fe.min()) <= 0 or float(fe.min() / fe.max()) < 0.02):
            _fbad = (f"degenerate bounding box {np.round(fe, 4).tolist()} "
                     f"(thinnest axis {100 * fe.min() / fe.max():.2f}% of the longest)")
        if _fbad or fe is None:
            raise RuntimeError(
                f"[MeshGuard] {tag}: the mesh is corrupt ({why}) AND was already corrupt on the way "
                f"in ({_fbad}). So {tag} is not the culprit -- the damage happened UPSTREAM of it. "
                f"Stopped here rather than carrying a broken mesh through the rest of the bake.")
        print(f"*** [MeshGuard] {tag} CORRUPTED THE MESH: {why}.\n"
              f"*** Falling back to that stage's INPUT (extent {np.round(fe, 4).tolist()}) and "
              f"continuing. This bake keeps whatever holes {tag} would have filled.\n"
              f"*** {tag} is the stage to fix.", flush=True)
        return (fallback,)


class Trellis2ShapeGate:
    """Judge the generated shape IMMEDIATELY and abort a bad draw before the expensive stages.

    Trellis draws a fresh shape each run and some draws are simply bad -- most visibly a spurious
    extra limb between the legs. Measured on one such draw, that limb is not subtle: it is its own
    connected component holding 5.6% of the vertices, running the full length of the legs
    (y -0.500..-0.200) and only 0.08 wide. The eyes, which are legitimately separate, hold 0.1%
    each -- so "a detached component holding more than 1% of the mesh" separates them cleanly with
    no tuning.

    Placed right after the generator, this costs seconds and saves the ~25 minutes of remesh, hole
    fill, graft, UV unwrap, texture transfer and rigging that a doomed mesh would otherwise consume.
    That is the same fail-fast rule as the mesh-sanity and UV tripwires, moved to the earliest point
    where the defect is visible.

    NOTE it aborts rather than re-rolls: ComfyUI has no loop primitive, so re-queueing is the retry.
    Set the generator's seed control to 'randomize' and a re-queue draws a new shape.
    Toggle: env SHAPE_GATE=0 to pass everything through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mesh": ("MESHWITHVOXEL",)},
            "optional": {
                "max_stray_fraction": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001,
                                       "tooltip": "A detached component larger than this share of the "
                                                  "mesh is a spurious limb. Eyes are ~0.1%."}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    def execute(self, mesh, max_stray_fraction=0.01):
        if os.environ.get("SHAPE_GATE", "1") == "0":
            print("[ShapeGate] disabled (SHAPE_GATE=0)", flush=True)
            return (mesh,)
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            v = mesh.vertices; f = mesh.faces
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
            v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
        except Exception as e:
            print(f"[ShapeGate] could not read the mesh ({e!r}) -- passing through", flush=True)
            return (mesh,)

        fails = []
        finite = np.isfinite(v).all(1)
        if not finite.all():
            fails.append(f"{int((~finite).sum()):,} non-finite vertices")
        if finite.any():
            ext = v[finite].max(0) - v[finite].min(0)
            if ext.max() <= 0 or ext.min() / ext.max() < 0.02:
                fails.append(f"degenerate bounding box {np.round(ext, 4).tolist()}")

        # Weld by POSITION before counting components. The generator splits vertices at UV seams,
        # so index-based connectivity reports hundreds of thousands of phantom "shells" and would
        # make this test meaningless -- a trap that already cost one wrong diagnosis on this project.
        stray = []
        try:
            scale = float(np.ptp(v[finite], 0).max()) if finite.any() else 1.0
            q = np.round(v / max(scale * 1e-5, 1e-12)).astype(np.int64)
            _, inv = np.unique(q, axis=0, return_inverse=True)
            fw = inv[f]
            n = int(inv.max()) + 1
            e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]], 0)
            g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
            ncomp, lab = connected_components(g, directed=False)
            sizes = np.bincount(lab)
            main = int(sizes.argmax())
            frac = sizes / max(n, 1)
            for ci in np.argsort(sizes)[::-1]:
                if ci == main or frac[ci] <= max_stray_fraction:
                    continue
                p = v[np.isin(lab, [ci])[inv]] if False else v[(lab[inv] == ci)]
                ex = (p.max(0) - p.min(0)) / max(scale, 1e-12)
                stray.append((int(sizes[ci]), float(frac[ci]), np.round(ex, 3).tolist()))
            print(f"[ShapeGate] {n:,} welded verts, {ncomp:,} component(s), largest {100*frac[main]:.1f}%",
                  flush=True)
            for cnt, fr, ex in stray:
                print(f"[ShapeGate]   STRAY component: {cnt:,} verts ({100*fr:.1f}%) extent {ex}", flush=True)
            if stray:
                fails.append(f"{len(stray)} detached component(s) over {100*max_stray_fraction:.1f}% "
                             f"of the mesh (largest {100*stray[0][1]:.1f}%) -- a spurious limb")
        except Exception as e:
            print(f"[ShapeGate] component check skipped ({e!r})", flush=True)

        if fails:
            raise RuntimeError(
                "[ShapeGate] REJECTING THIS SHAPE: " + "; ".join(fails) +
                ". Stopped at the generator instead of spending ~25 more minutes remeshing, "
                "grafting, unwrapping and texturing a mesh that would have to be redone. "
                "Re-queue to draw a new shape (set the generator seed control to 'randomize'), "
                "or set SHAPE_GATE=0 to force this one through.")
        print("[ShapeGate] shape accepted", flush=True)
        return (mesh,)


class Trellis2TrellisTexture:
    """Base colour + metallic-roughness resampled from TRELLIS'S OWN texture onto the final UVs.

    Drop-in replacement for Trellis2MultiViewTexturing's base_color / metallic_roughness outputs.
    Trellis paints its mesh at generation time, so its texture is aligned to that mesh by
    construction -- there are no cameras and therefore no surfaces that no camera can see. The
    projection path left 9.7-12.8% of the body's surface area unpainted on real bakes (feet ~37%);
    this leaves 0.00%. The algorithm and the reasoning live in uv_transfer/trellis_transfer.py."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "obj_path": ("STRING", {"forceInput": True,
                             "tooltip": "obj_path from Trellis2BlenderSmartUV (the final UVs)."}),
                "source_trimesh": ("TRIMESH", {
                             "tooltip": "Trellis's own textured mesh -- the trimesh output of "
                                        "Trellis2UnWrapAndRasterizer fed from the mesh generator."}),
            },
            "optional": {
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("base_color", "metallic_roughness")
    FUNCTION = "execute"
    CATEGORY = "Trellis2Wrapper"

    # the algorithm lives in a separate file so it can be edited without a ComfyUI restart
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import os as _o
        _p = r"C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/trellis_transfer.py"
        try:
            return str(_o.path.getmtime(_p))
        except Exception:
            return float("nan")

    def execute(self, obj_path, source_trimesh, texture_size=4096):
        import importlib
        import sys as _sys
        import torch as _t
        _uv = r"C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer"
        if _uv not in _sys.path:
            _sys.path.insert(0, _uv)
        import trellis_transfer as _tt
        importlib.reload(_tt)            # pick up edits without restarting ComfyUI
        if not obj_path or not os.path.exists(obj_path):
            raise RuntimeError(f"[TrellisTexture] obj_path missing: {obj_path!r}. "
                               "Connect it to Trellis2BlenderSmartUV's obj_path output.")
        base, mr = _tt.transfer(obj_path, source_trimesh, size=int(texture_size), log=print)
        base_t = _t.from_numpy(base.astype(np.float32))[None, ...]
        mr_t = _t.from_numpy((mr if mr is not None else np.zeros_like(base)).astype(np.float32))[None, ...]
        print(f"[TrellisTexture] base_color {tuple(base_t.shape)}"
              f"{'' if mr is not None else '  (source had no MR texture -- emitting black)'}", flush=True)
        return (base_t, mr_t)


NODE_CLASS_MAPPINGS = {
    "FlattenLight":           FlattenLight,
    "Trellis2BlenderSmartUV": Trellis2BlenderSmartUV,
    "Trellis2XAtlasUnwrap":   Trellis2XAtlasUnwrap,
    "Trellis2FBXExport":      Trellis2FBXExport,
    "Trellis2RenderClayView": Trellis2RenderClayView,
    "Trellis2TrellisTexture": Trellis2TrellisTexture,
    "Trellis2MeshGuard":      Trellis2MeshGuard,
    "Trellis2ShapeGate":      Trellis2ShapeGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenLight":           "Flatten Light (shadow removal)",
    "Trellis2BlenderSmartUV": "Trellis2 - Blender Decimate + Zone UV",
    "Trellis2XAtlasUnwrap":   "Trellis2 - XAtlas UV Unwrap",
    "Trellis2FBXExport":      "Trellis2 - FBX Export (hard edges)",
    "Trellis2RenderClayView": "Trellis2 - Render Clay View (untextured, for Canny)",
    "Trellis2TrellisTexture": "Trellis2 - Trellis Own Texture -> final UVs",
    "Trellis2MeshGuard":      "Trellis2 - Mesh Guard (fall back if a stage corrupts it)",
    "Trellis2ShapeGate":      "Trellis2 - Shape Gate (reject a bad draw early)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
