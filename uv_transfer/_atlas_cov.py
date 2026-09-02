"""LOCAL: overlay the freshly-baked basecolor atlas coverage onto the UV island map.
Bright island = got paint; darkened island = empty. Tells us coverage-failure vs UV mismatch."""
import os, sys, numpy as np, trimesh, trimesh.graph
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
ATL = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_00461_.png"
a = Image.open(ATL); print("atlas mode", a.mode, "size", a.size)
arr = np.asarray(a.convert("RGBA")).astype(np.float32)
alpha = arr[..., 3]
rgb = arr[..., :3]
# painted = has alpha OR is not background (background appears white ~255 or transparent)
near_white = (rgb > 245).all(-1)
painted = (alpha > 10) & (~near_white)
print(f"atlas coverage: {100*painted.mean():.2f}% of atlas painted")
R = 1400
cov = np.asarray(Image.fromarray((painted*255).astype(np.uint8)).resize((R, R), Image.NEAREST)) > 127
# island map
m = trimesh.load(os.path.join(HERE, "last_seams.glb"), force='mesh')
uv = np.asarray(m.visual.uv); F = np.asarray(m.faces)
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)))
isl = np.zeros(len(F), np.int32)
for i, c in enumerate(cc): isl[c] = i
rng = np.random.RandomState(5); col = rng.randint(60, 235, (len(cc), 3))
from PIL import ImageDraw
img = Image.new("RGB", (R, R), (12, 12, 14)); dr = ImageDraw.Draw(img)
t = uv[F]
def px(p): return (float(p[0])*R, float((1-p[1])*R))
for fi in range(len(F)):
    c = tuple(int(x) for x in col[isl[fi]])
    dr.polygon([px(t[fi, 0]), px(t[fi, 1]), px(t[fi, 2])], fill=c)
base = np.asarray(img).astype(np.float32)
# darken where NOT covered by atlas paint
out = base.copy(); out[~cov] *= 0.18
# tint covered area edges bright
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(os.path.join(HERE, "_atlas_cov.png"))
print("saved _atlas_cov.png (bright island = painted, dark = empty)")
