"""ALIGNMENT PROOF for the front-image hairline idea.

Projects the front reference image (_front_ref.png, dropped by the SmartUV node
from the pipeline) onto the CURRENT gen_seams mesh (_genseams_viz.npz) and renders:

  _ov_sampled_* : front-facing faces tinted by the RAW sampled image colour.
                  If alignment is right the head reads like the photo (eyes, brows,
                  hairline in the correct spots). If it's mirrored/shifted, this shows it.
  _ov_class_*   : front-facing faces classified HAIR(blonde) vs SKIN(grey) by colour.
                  This is the money shot — does the colour boundary land on the hairline.

Everything that is NOT a sampled front face is dark grey (context only).

Tunables (correct by LOOKING, never assume):
  U_SIGN / V_SIGN  flip if the head looks mirrored / upside-down
  FRONTNESS        how head-on a face must be to be sampled (1=dead front)
  BG_THRESH        subject-vs-background separation for the image bbox
  HAIR_RULE        colour test for blonde hair vs skin

Run:  blender --background --python _front_overlay.py
(uses the mesh + image already on disk; no pipeline run needed)
"""
import bpy, os, numpy as np
from mathutils import Vector

D   = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
IMG = os.path.join(D, "_front_ref.png")
VZ  = os.path.join(D, "_genseams_viz.npz")

# ── tunables ────────────────────────────────────────────────────────────────
U_SIGN    = +1.0     # +1: mesh +x -> image right (per projection: image-right = world +x)
V_SIGN    = +1.0     # +1: mesh +z (height) -> image top
FRONTNESS   = 0.25   # face sampled only if it faces the camera at least this much
IMG_SCALE_V = 1.05   # >1 enlarges projected image vertically around the face anchor
IMG_SCALE_H = 1.05   # >1 enlarges projected image horizontally around the face anchor
SMOOTH_ITERS = 6     # mesh-surface smoothing passes on the image classification
WARM_THRESH = 0.10   # (r-b) above this = subject  (skin/hair ~0.2 vs grey bg ~0.05)
V_NUDGE     = 0      # manual vertical fine-tune, image px (+ = sample lower); residual offset knob
U_NUDGE     = 0      # manual horizontal fine-tune, image px (+ = sample righter)

def hair_rule(rgb):
    """rgb in 0..1, shape (...,3). True = blonde hair. Tune against the render."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # blonde: warm (r,g high) but b noticeably lower; skin is pinker (r>g) & brighter b.
    return (b < 0.82 * g) & (lum > 0.18) & (lum < 0.92)

# ── load mesh (gen_seams space: x=lateral, y=depth front<0, z=height) ─────────
vz   = np.load(VZ)
co   = vz['co'].astype(np.float64)
tris = vz['tris'].astype(np.int64)
v0, v1, v2 = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
cen = (v0 + v1 + v2) / 3.0
nrm = np.cross(v1 - v0, v2 - v0)
nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
frontness = -nrm[:, 1]                      # normal toward -y (front) => faces camera
front = frontness > FRONTNESS

# ── load image (Blender pixels are bottom-up, RGBA, 0..1) ─────────────────────
img = bpy.data.images.load(IMG)
W, H = img.size
px = np.array(img.pixels[:], dtype=np.float64).reshape(H, W, 4)[::-1]   # row0 = TOP
rgb_img = px[:, :, :3]

# subject bbox: warm=(r-b) separates skin/hair (~0.2) from grey bg (~0.05).
# Use row/column OCCUPANCY (a row/col counts only if enough warm pixels) so stray
# noise pixels can't widen the box — that full-frame blowup was the offset bug.
warm = rgb_img[:, :, 0] - rgb_img[:, :, 2]
subj = warm > WARM_THRESH
row_has = subj.sum(axis=1) > 0.01 * W
col_has = subj.sum(axis=0) > 0.01 * H
ys = np.where(row_has)[0]; xs = np.where(col_has)[0]
iy0, iy1 = int(ys.min()), int(ys.max())                               # iy0=top, iy1=bottom
ix0, ix1 = int(xs.min()), int(xs.max())
print(f"[overlay] image {W}x{H}  subject bbox x[{ix0},{ix1}] y[{iy0},{iy1}]  "
      f"(head top {100*iy0/H:.1f}%  body width {100*(ix1-ix0)/W:.1f}%)")

# ── map mesh (x,z) -> image (col,row).  Prefer the MediaPipe landmark fit
#    (_front_align.json: row=a*z+b, col=c*x+d) which anchors on the actual face;
#    fall back to the silhouette bbox if it isn't there. ─────────────────────────
import json
_alignf = os.path.join(D, "_front_align.json")
if os.path.exists(_alignf):
    A = json.load(open(_alignf))
    u = A["c"] * cen[:, 0] + A["d"]
    if "z_kink" in A:   # piecewise: face fit below the forehead, bent map above it
        v = np.where(cen[:, 2] <= A["z_kink"], A["a"] * cen[:, 2] + A["b"],
                     A["a2"] * cen[:, 2] + A["b2"])
    else:
        v = A["a"] * cen[:, 2] + A["b"]
    # scale the projected image around the face anchor (face stays put, periphery scales)
    z_anch = A.get("z_face", 0.43)
    row_anch = A["a"] * z_anch + A["b"]; col_anch = A["d"]
    u = col_anch + (u - col_anch) / IMG_SCALE_H
    v = row_anch + (v - row_anch) / IMG_SCALE_V
    print(f"[overlay] scale V={IMG_SCALE_V} H={IMG_SCALE_H} about z_face={z_anch:.3f}")
    print(f"[overlay] using landmark align: row={A['a']:.1f}*z+{A['b']:.1f} "
          f"(res {A['v_maxres']:.1f}px), col={A['c']:.1f}*x+{A['d']:.1f} (res {A['h_maxres']:.1f}px)"
          + (f"  +piecewise above z={A['z_kink']:.3f}" if 'z_kink' in A else ""))
else:
    mx0, mx1 = co[:, 0].min(), co[:, 0].max()
    mz0, mz1 = co[:, 2].min(), co[:, 2].max()
    if U_SIGN > 0:
        u = ix0 + (cen[:, 0] - mx0) / (mx1 - mx0) * (ix1 - ix0)
    else:
        u = ix1 - (cen[:, 0] - mx0) / (mx1 - mx0) * (ix1 - ix0)
    if V_SIGN > 0:
        v = iy1 + (cen[:, 2] - mz0) / (mz1 - mz0) * (iy0 - iy1)
    else:
        v = iy0 + (cen[:, 2] - mz0) / (mz1 - mz0) * (iy1 - iy0)
    print("[overlay] using bbox align (no _front_align.json)")
ui = np.clip(np.round(u + U_NUDGE).astype(int), 0, W - 1)
vi = np.clip(np.round(v + V_NUDGE).astype(int), 0, H - 1)
# Denoise AT THE SOURCE: average a small image patch per face (smooth colour, but the
# hairline boundary stays sharp because the patch follows the image, not the mesh).
PR = 2
_acc = np.zeros((len(ui), 3))
for _dy in range(-PR, PR + 1):
    for _dx in range(-PR, PR + 1):
        _acc += rgb_img[np.clip(vi + _dy, 0, H - 1), np.clip(ui + _dx, 0, W - 1)]
sampled = _acc / ((2 * PR + 1) ** 2)
# per-character hair/skin colour references from geometrically-DEFINITE regions
_dh = front & (((cen[:, 2] > 0.485) & (np.abs(cen[:, 0]) < 0.08)) |
               ((np.abs(cen[:, 0]) > 0.10) & (cen[:, 2] > 0.32) & (cen[:, 2] < 0.42)))
_ds = front & (cen[:, 2] > 0.40) & (cen[:, 2] < 0.45) & (np.abs(cen[:, 0]) < 0.04)
hair_ref = np.median(sampled[_dh], axis=0)
skin_ref = np.median(sampled[_ds], axis=0)
_dhd = np.linalg.norm(sampled - hair_ref, axis=1)
_dsd = np.linalg.norm(sampled - skin_ref, axis=1)
hairness_s = _dsd / (_dhd + _dsd + 1e-9)      # per-face, patch-smoothed (NO mesh diffusion)
is_hair = (hairness_s > 0.5) & front
print(f"[overlay] hair_ref={hair_ref.round(2)} skin_ref={skin_ref.round(2)}  hair {int(is_hair.sum())}")

# ── build mesh once, render with two colour sets ──────────────────────────────
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
me = bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob = bpy.data.objects.new("m", me); bpy.context.scene.collection.objects.link(ob)
col = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')

GREY = np.array([0.18, 0.18, 0.18])
def render_set(face_rgb, tag, light='FLAT'):
    # face_rgb: (n_tris,3) per-face colour; write to all 3 corners of each tri
    loop_rgb = np.repeat(face_rgb, 3, axis=0)
    rgba = np.concatenate([loop_rgb, np.ones((len(loop_rgb), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel())
    me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = light          # STUDIO reveals geometry under the colour
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1300; sc.render.resolution_y = 1400
    sc.render.image_settings.file_format = 'PNG'
    zmin, zmax = co[:, 2].min(), co[:, 2].max(); Hh = zmax - zmin
    tgt = Vector((0.0, 0.0, zmin + 0.86 * Hh)); S = 0.55 * Hh
    for off, nm in [((0, -3.0 * Hh, 0.05 * Hh), f"_ov_{tag}_front.png"),
                    (( 2.3 * Hh, -2.0 * Hh, 0.12 * Hh), f"_ov_{tag}_qL.png"),
                    ((-2.3 * Hh, -2.0 * Hh, 0.12 * Hh), f"_ov_{tag}_qR.png")]:
        loc = tgt + Vector(off)
        cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = S
        cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = loc; fwd = (tgt - loc).normalized()
        cam.rotation_euler = fwd.to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam, do_unlink=True)

# overlay A: raw sampled colour on front faces, grey elsewhere — STUDIO so geometry shows
face_rgb = np.where(front[:, None], sampled, GREY[None, :])
render_set(face_rgb, "sampled", light='STUDIO')
# overlay B: hair(blonde)/skin(grey) classification on front faces
BLOND = np.array([0.92, 0.76, 0.28]); SKIN = np.array([0.60, 0.58, 0.56])
face_rgb = np.where(is_hair[:, None], BLOND[None, :],
                    np.where(front[:, None], SKIN[None, :], GREY[None, :]))
render_set(face_rgb, "class")

# ── HEAD-ZOOM diagnostic: geometry-only vs projected, SAME camera, to expose offset ──
def head_shot(face_rgb, nm):
    loop_rgb = np.repeat(face_rgb, 3, axis=0)
    rgba = np.concatenate([loop_rgb, np.ones((len(loop_rgb), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel()); me.update()
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x = 1100; sc.render.resolution_y = 1250
    zmax = co[:, 2].max()
    tgt = Vector((0.0, 0.0, zmax - 0.105)); S = 0.30      # tight on the head
    loc = tgt + Vector((0.0, -3.0, 0.0))
    cd = bpy.data.cameras.new("c"); cd.type = 'ORTHO'; cd.ortho_scale = S
    cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = loc; fwd = (tgt - loc).normalized()
    cam.rotation_euler = fwd.to_track_quat('-Z', 'Y').to_euler()
    # EXACT render-pixel <-> mesh-(x,z) calibration from Blender's own camera projection
    # (no ortho_scale assumptions). _front_align.py reads this to convert MediaPipe
    # landmark (x_norm,y_norm) on this render straight to mesh (x,z).
    import bpy_extras, json as _json
    bpy.context.view_layer.update()
    def _yn(zw):  # top-down normalised row (0=top) for a world point at height zw on axis
        return 1.0 - bpy_extras.object_utils.world_to_camera_view(sc, cam, Vector((0.0, 0.0, zw))).y
    def _xn(xw):  # left-right normalised col (0=left) for a world point at lateral xw
        return bpy_extras.object_utils.world_to_camera_view(sc, cam, Vector((xw, 0.0, tgt.z))).x
    _json.dump({"z1": zmax, "yn1": _yn(zmax), "z2": zmax - 0.3, "yn2": _yn(zmax - 0.3),
                "x1": 0.0, "xn1": _xn(0.0), "x2": 0.1, "xn2": _xn(0.1)},
               open(os.path.join(D, "_head_cam.json"), "w"))
    sc.render.filepath = os.path.join(D, nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

GEOM = np.tile(np.array([0.72, 0.72, 0.74]), (len(tris), 1))      # plain grey, geometry only
head_shot(GEOM, "_ov_head_geom.png")
head_shot(np.where(front[:, None], sampled, GREY[None, :]), "_ov_head_proj_n0.png")
# coverage: RED = NOT front-facing (uncovered by the frontness filter, by design),
# so we can tell "bare because angled away" from "genuinely mismapped".
RED = np.array([0.62, 0.10, 0.10])
head_shot(np.where(front[:, None], sampled, RED[None, :]), "_ov_cover.png")
# head-zoom of the hair/skin classification over the geometry — the hairline money shot
head_shot(np.where(is_hair[:, None], BLOND[None, :],
                   np.where(front[:, None], SKIN[None, :], GREY[None, :])), "_ov_head_class.png")

# ── BLEND: start from geometry; use the front image to REMOVE the bleed it's sure is skin
#    (forehead + chin/neck, capped below the crown) and ADD the framing/temples it's sure
#    are hair AND connected to existing hair. Keep geometry where faces point up (crown). ──
geo_hair = vz['tris_hair'].astype(bool)
flat = tris.ravel()
hz = front & (np.abs(cen[:, 0]) < 0.16) & (cen[:, 2] > 0.30) & (cen[:, 2] < 0.55)
blend_hair = geo_hair.copy()
blend_hair[hz & geo_hair & (hairness_s < 0.40) & (cen[:, 2] < 0.475)] = False   # remove bleed
add_cand = hz & (~blend_hair) & (hairness_s > 0.62)                             # add framing/temples
for _ in range(25):
    vh = np.zeros(co.shape[0], bool)
    np.logical_or.at(vh, flat, np.repeat(blend_hair, 3))
    newly = add_cand & vh[tris].any(axis=1) & (~blend_hair)
    if not newly.any():
        break
    blend_hair |= newly
# pure image verdict (front head region) for side-by-side comparison
img_only = ((hairness_s > 0.5) & front & (np.abs(cen[:, 0]) < 0.16)
            & (cen[:, 2] > 0.30) & (cen[:, 2] < 0.55))
def hairviz(mask): return np.where(mask[:, None], BLOND[None, :],
                                   np.where(front[:, None], SKIN[None, :], GREY[None, :]))
head_shot(hairviz(geo_hair),   "_ov_geohair.png")     # geometry baseline
head_shot(hairviz(img_only),   "_ov_imageonly.png")   # pure image verdict (front)
head_shot(hairviz(blend_hair), "_ov_blend.png")       # geometry + image fixes
print("[overlay] geo %d  imageonly %d  blend %d  (remove<0.40 capped z<0.475, add>0.62 connected)"
      % (int(geo_hair.sum()), int(img_only.sum()), int(blend_hair.sum())))
print("[overlay] DONE  U_SIGN", U_SIGN, "V_SIGN", V_SIGN, "FRONTNESS", FRONTNESS,
      "bbox y[%d,%d] x[%d,%d]" % (iy0, iy1, ix0, ix1))
