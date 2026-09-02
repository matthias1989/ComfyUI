"""LOCAL: render last_seams.obj's UV islands ON the 3D body (front), so the seam lines
(island boundaries) are visible where they actually land. Mislanded anatomical cuts show
as island boundaries cutting across the wrong places (e.g. mid-thigh, asymmetric)."""
import os, sys, numpy as np, trimesh, trimesh.graph, torch, nvdiffrast.torch as dr
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.environ.get("OBJ_PATH", os.path.join(HERE, "last_seams.obj")), force='mesh')
f = np.asarray(m.faces)
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(f)))
isl = np.zeros(len(f), np.int32)
for i, c in enumerate(cc):
    isl[c] = i
rng = np.random.RandomState(3); colors = rng.randint(45, 235, (len(cc), 3)).astype(np.uint8)
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
c0 = (v.min(0).values + v.max(0).values) * 0.5; v = v - c0; r = torch.sqrt((v**2).sum(-1).max()); v = v / (r*2)
ft = torch.from_numpy(f.astype(np.int32)).cuda(); ctx = dr.RasterizeCudaContext()
xx = v[:,0]; yy = v[:,1]; d = v[:,2]; zz = 1 - 2*(d-d.min())/(d.max()-d.min()); R = 1000
def render(name, sx, sz):
    cam = torch.stack([xx*2*sx, yy*2, zz*sz, torch.ones_like(xx)], -1).unsqueeze(0).contiguous()
    rr, _ = dr.rasterize(ctx, cam, ft, resolution=[R,R]); fid = (rr[0,:,:,3].long()-1).cpu().numpy(); hit = fid >= 0
    img = np.full((R,R,3), 30, np.uint8); img[hit] = colors[isl[fid[hit]]]
    full = Image.fromarray(np.flipud(img).copy())
    full.save(os.path.join(HERE, f"_seams3d_{name}.png"))
    if name == "front":
        full.crop((int(R*0.36), int(R*0.02), int(R*0.64), int(R*0.22))).resize((720,560), Image.LANCZOS).save(os.path.join(HERE, "_seams3d_head.png"))
    print(f"saved _seams3d_{name}.png")
render("front", 1.0, 1.0); render("back", -1.0, -1.0)
sizes = sorted((len(c) for c in cc), reverse=True)
print(f"islands={len(cc)} top={sizes[:10]}")
