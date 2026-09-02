"""LOCAL: render the actual UV unwrap (atlas layout) from last_seams.glb, colored by 3D island,
+ flag overlapping/flipped UV triangles. Tells us if the torso UVs are clean (so a stale texture is
the cause -> just re-project) or scrambled (unwrap itself broken -> fix gen_seams)."""
import os, sys, numpy as np, trimesh, trimesh.graph
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.path.join(HERE, "last_seams.glb"), force='mesh')
uv = np.asarray(m.visual.uv); F = np.asarray(m.faces)
print("uv range", uv.min(0).round(3), uv.max(0).round(3), " faces", len(F))
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)))
isl = np.zeros(len(F), np.int32)
for i,c in enumerate(cc): isl[c]=i
rng = np.random.RandomState(5); col = rng.randint(50,235,(len(cc),3))
t = uv[F]
sarea = 0.5*((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1]) - (t[:,2,0]-t[:,0,0])*(t[:,1,1]-t[:,0,1]))
print("flipped UV tris (signed area<0):", int((sarea<0).sum()), "/", len(F),
      "  degenerate(|a|<1e-12):", int((np.abs(sarea)<1e-12).sum()))
R=1400; img=Image.new("RGB",(R,R),(18,18,20)); dr=ImageDraw.Draw(img)
def px(p): return (float(p[0])*R, float((1-p[1])*R))
for fi in range(len(F)):
    c=tuple(int(x) for x in col[isl[fi]])
    if sarea[fi]<0: c=(240,30,30)
    dr.polygon([px(t[fi,0]),px(t[fi,1]),px(t[fi,2])], fill=c)
img.save(os.path.join(HERE,"_uvcheck.png")); print("saved _uvcheck.png")
