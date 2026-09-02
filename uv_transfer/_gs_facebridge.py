"""FACE BRIDGE compute step — runs in ComfyUI's python_embeded (has MediaPipe + nvdiffrast).
Loads the mesh gen_seams exported (_gs_facemesh.npz), detects the face region MESH-ONLY, and
saves the per-face mask (_gs_faceregion.npy) for gen_seams pass B. gen_seams exports Z-up
(height=col2), so we swap to Y-up and TRY BOTH depth signs (OBJ vs GLB import flip differently)
— using whichever orientation MediaPipe actually finds a face in. On any failure, writes nothing
so gen_seams falls back to its geometric face region."""
import os, sys, numpy as np, torch, nvdiffrast.torch as dr, trimesh
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
sys.path.insert(0, r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2")
from projection.landmark_warp import compute_face_region_faces
npz = os.path.join(HERE, "_gs_facemesh.npz")
out = os.path.join(HERE, "_gs_faceregion.npy")
if not os.path.exists(npz):
    print("[face-bridge] no _gs_facemesh.npz — skip (gen_seams uses geometric)"); sys.exit(0)
d = np.load(npz)
verts0 = d["verts"].astype(np.float32)[:, [0, 2, 1]].copy()   # Z-up -> Y-up (height=col1)
tris = d["tris"].astype(np.int64); n_faces = len(tris)
ctx = dr.RasterizeCudaContext(); f = torch.from_numpy(tris).int().cuda()
best = None
for sgn in (-1.0, 1.0):                                        # OBJ vs GLB import flip depth
    vv = verts0.copy(); vv[:, 2] *= sgn
    tm = trimesh.Trimesh(vertices=vv, faces=tris, process=False)
    v = torch.from_numpy(vv).float().cuda()
    for _ in range(2):
        c = (v.min(0).values + v.max(0).values) * 0.5; v = v - c
        r = torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v = v * (1.15 / (r * 2))
    vn = torch.from_numpy(np.asarray(tm.vertex_normals)).float().cuda(); vn = vn / (vn.norm(dim=-1, keepdim=True) + 1e-8)
    s = 1.15; uc = v[:, 0] / (s / 2); vc = v[:, 1] / (s / 2); IH = IW = 640
    dd = v[:, 2]; zz = 1 - 2 * (dd - dd.min()) / (dd.max() - dd.min())
    clip = torch.stack([uc, vc, zz, torch.ones_like(uc)], -1).unsqueeze(0).contiguous()
    rr, _ = dr.rasterize(ctx, clip, f, resolution=[IH, IW]); hit = (rr[0, :, :, 3] > 0).cpu().numpy()
    front = np.zeros((IH, IW, 3), np.float32); front[hit] = 1.0
    try:
        sel = compute_face_region_faces(v, f, vn, front, 1.15, ctx)
    except Exception as e:
        print(f"[face-bridge] sgn={sgn:+.0f} error {e!r}"); sel = None
    n = 0 if sel is None else int(np.asarray(sel, bool).sum())
    print(f"[face-bridge] depth sgn={sgn:+.0f} -> {n} face faces")
    if n > 100:
        best = np.asarray(sel, bool); break
if best is None:
    # FALLBACK: the synthetic relief is undetectable for this mesh -> detect on the rembg'd PHOTO instead
    # (a real photo MediaPipe reads reliably) and project it the same way. Reliable by construction; only
    # runs when the relief already FAILED, so it can never change a bake where the relief already works.
    try:
        from PIL import Image
        _pp = os.path.join(HERE, "_front_real.png")
        if os.path.exists(_pp):
            _ph = np.array(Image.open(_pp).convert("RGB")); _phf = _ph.astype(np.float32) / 255.0
            _zc = d["verts"].astype(np.float64)                          # Z-up: Y=depth (front=-Y)
            _e1 = _zc[tris[:, 1]] - _zc[tris[:, 0]]; _e2 = _zc[tris[:, 2]] - _zc[tris[:, 0]]
            _fn = np.cross(_e1, _e2); _fn /= (np.linalg.norm(_fn, axis=1, keepdims=True) + 1e-9)
            _ctr = _zc.mean(0); _fcc = _zc[tris].mean(1); _flp = ((_fcc - _ctr) * _fn).sum(1) < 0; _fn[_flp] = -_fn[_flp]
            _fwd = _fn[:, 1] < -0.2                                       # forward-facing test (to pick orientation)
            _pick = None; _pf = -1
            for _sgn in (-1.0, 1.0):
                _vv = verts0.copy(); _vv[:, 2] *= _sgn
                _tm = trimesh.Trimesh(vertices=_vv, faces=tris, process=False); _v = torch.from_numpy(_vv).float().cuda()
                for _ in range(2):
                    _c = (_v.min(0).values + _v.max(0).values) * 0.5; _v = _v - _c
                    _r = torch.sqrt((_v ** 2).sum(-1).max()).clamp(min=1e-6); _v = _v * (1.15 / (_r * 2))
                _vn = torch.from_numpy(np.asarray(_tm.vertex_normals)).float().cuda(); _vn = _vn / (_vn.norm(dim=-1, keepdim=True) + 1e-8)
                _sel = compute_face_region_faces(_v, f, _vn, _phf, 1.15, ctx, mp_image=_ph)
                if _sel is None:
                    continue
                _sel = np.asarray(_sel, bool); _nf = int((_sel & _fwd).sum())
                if int(_sel.sum()) > 100 and _nf > _pf:                   # pick the orientation with the face at the FRONT
                    _pick = _sel; _pf = _nf
            if _pick is not None:
                best = _pick
                print(f"[face-bridge] FALLBACK photo-detect -> {int(best.sum())} face faces (front-facing {_pf})")
    except Exception as _fe:
        import traceback; traceback.print_exc(); print(f"[face-bridge] FALLBACK error {_fe!r}")
if best is None:
    print("[face-bridge] no face found (relief + photo) — no mask; gen_seams uses geometric"); sys.exit(0)
np.save(out, best)
print(f"[face-bridge] compute: {int(best.sum())} face faces / {n_faces} -> {out}")
