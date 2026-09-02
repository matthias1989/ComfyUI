"""LOCAL: render the UV island layout of last_seams.obj (each island a distinct color) and
report island count/sizes. A clean humanoid unwrap = ~10-30 islands; hundreds = shattered."""
import os, sys, time, numpy as np, trimesh, torch, nvdiffrast.torch as dr
from PIL import Image
import trimesh.graph
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
P = os.environ.get("OBJ_PATH", os.path.join(HERE, "last_seams.obj"))
print("last_seams.obj mtime:", time.ctime(os.path.getmtime(P)))
m = trimesh.load(P, force='mesh')
uv = np.asarray(m.visual.uv); f = np.asarray(m.faces)
print(f"verts={len(m.vertices)} faces={len(f)} uv_range u=[{uv[:,0].min():.2f},{uv[:,0].max():.2f}] v=[{uv[:,1].min():.2f},{uv[:,1].max():.2f}]")
# UV islands = connected components of faces sharing an edge (seams split verts -> separate)
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(f)))
sizes = sorted((len(c) for c in cc), reverse=True)
print(f"UV islands = {len(cc)}   top sizes={sizes[:12]}   (>50 faces: {sum(1 for s in sizes if s>50)})")
# color each island
island_of = np.zeros(len(f), np.int32)
for i, c in enumerate(cc): island_of[c] = i
rng = np.random.RandomState(1); colors = rng.randint(40, 255, (len(cc), 3)).astype(np.uint8)
# rasterize UV
clip = torch.from_numpy(np.concatenate([uv*2-1, np.zeros((len(uv),1)), np.ones((len(uv),1))],1).astype(np.float32)).cuda().unsqueeze(0).contiguous()
ft = torch.from_numpy(f.astype(np.int32)).cuda(); ctx = dr.RasterizeCudaContext()
R = 1024; rr,_ = dr.rasterize(ctx, clip, ft, resolution=[R,R])
fid = (rr[0,:,:,3].long()-1).cpu().numpy(); hit = fid>=0
img = np.full((R,R,3), 20, np.uint8)
isl = np.zeros((R,R), np.int32); isl[hit] = island_of[fid[hit]]
img[hit] = colors[isl[hit]]
Image.fromarray(np.flipud(img).copy()).save(os.path.join(HERE,"_uv_layout.png"))
print("saved _uv_layout.png")
