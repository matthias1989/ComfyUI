"""Verify MediaPipe face detection works MESH-ONLY (no photo): render the mesh's own
silhouette as the 'front image', then call compute_face_region_faces. If it returns a
sensible face region (~hundreds of faces on the face), the gen_seams MediaPipe bridge is viable."""
import os, sys, numpy as np, torch, trimesh, nvdiffrast.torch as dr
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
m = trimesh.load(os.path.join(HERE, "last_seams.obj"), force='mesh')
v = torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c = (v.min(0).values + v.max(0).values)*0.5; v = v - c
    r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v*(1.15/(r*2))
f = torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn = torch.from_numpy(np.asarray(m.vertex_normals)).float().cuda(); vn = vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
ctx = dr.RasterizeCudaContext()
s = 1.15; uc = v[:,0]/(s/2); vc = v[:,1]/(s/2); IH = IW = 640
d = v[:,2]; zz = 1 - 2*(d-d.min())/(d.max()-d.min())
clip = torch.stack([uc, vc, zz, torch.ones_like(uc)], -1).unsqueeze(0).contiguous()
rr, _ = dr.rasterize(ctx, clip, f, resolution=[IH, IW]); hit = (rr[0,:,:,3] > 0).cpu().numpy()
for flip, lbl in ((False, "upright"), (True, "flipped")):
    sil = np.flipud(hit) if flip else hit
    front = np.zeros((IH, IW, 3), np.float32); front[sil] = 1.0
    try:
        sel = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
        print(f"[{lbl}] mesh-only face region -> {None if sel is None else int(sel.sum())} faces")
    except Exception as e:
        print(f"[{lbl}] FAILED: {e!r}")
