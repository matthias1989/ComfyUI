"""Bake Trellis's OWN texture onto the final mesh's UV atlas.

Trellis generates colour and geometry together, so its texture is aligned to its own mesh by
construction -- no cameras, no projection, no parts of the body that no view can see. The pipeline
then rebuilds the body (decimate -> hand graft -> voxel remesh -> pelvis graft -> new UV unwrap),
which cannot carry UVs, so the colour has to be resampled onto the final surface. That resampling is
what this module does.

Why this replaces the multi-view projection for base colour: projection can only paint what 4-6
cameras can see, and measured on a real bake it left 9.7-12.8% of the body's SURFACE AREA unpainted
(feet ~37%). This transfer samples a mesh that covers the whole body, so there is no unseen-surface
failure mode at all -- measured 0.00% of texels without a source.

Two things that are easy to get wrong and cost real bakes:

  * ORIENTATION, NOT DIRECTION, when matching normals. The Trellis mesh's winding is inconsistent
    (trimesh: is_winding_consistent False) -- about 42% of its BACK and RIGHT faces have reversed
    triangles and therefore inverted normals. A signed dot test rejects those perfectly good faces
    and dumps whole regions into the unfiltered fallback, which is exactly why front/left came out
    clean and back/right did not. A flipped normal still describes the same surface.

  * MEDOID, NOT COMPONENT-WISE MEDIAN, for the outlier guard. At a hair/skin boundary the k
    neighbours span both, so their per-channel median is a blend that exists nowhere on the model --
    a flat grey-brown patch with staircase edges. The medoid is the actual candidate nearest the
    median, so the result is always a real colour off a real face.
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

K = 9


def load_target_obj(path):
    """Return V, VT, FV, FT, FG for an OBJ, fan-triangulating polygons and keeping, per triangle,
    the index of the polygon it came from (FG) so per-POLYGON flags line up."""
    V, VT, FV, FT, FG = [], [], [], [], []
    poly = 0
    for line in open(path, "r", errors="ignore"):
        if line.startswith("v "):
            p = line.split(); V.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("vt "):
            p = line.split(); VT.append((float(p[1]), float(p[2])))
        elif line.startswith("f "):
            vi, ti = [], []
            for t in line.split()[1:]:
                q = t.split("/")
                vi.append(int(q[0]) - 1)
                ti.append(int(q[1]) - 1 if len(q) > 1 and q[1] else -1)
            for k in range(1, len(vi) - 1):
                FV.append((vi[0], vi[k], vi[k + 1]))
                FT.append((ti[0], ti[k], ti[k + 1]))
                FG.append(poly)
            poly += 1
    return (np.array(V), np.array(VT), np.array(FV, np.int64),
            np.array(FT, np.int64), np.array(FG, np.int64))


def _unit(v):
    v = np.asarray(v, float)
    c = 0.5 * (v.min(0) + v.max(0))
    s = float((v.max(0) - v.min(0)).max())
    return (v - c) / max(s, 1e-9)


def transfer(obj_path, source, size=4096, log=print):
    """obj_path = the final mesh with UVs (gen_seams' OBJ). source = a trimesh with a baseColor
    texture and UVs (Trellis's own textured mesh), or a path to one.
    Returns (base_color HxWx3 float01, metallic_roughness HxWx3 float01 or None)."""
    t0 = time.time()

    def _log(m):
        log(f"[trellis-tex] {m}")

    import trimesh

    if isinstance(source, str):
        source = trimesh.load(source, process=False)
    if hasattr(source, "geometry") and len(getattr(source, "geometry", {})):
        g = list(source.geometry.values())
        source = g[0] if len(g) == 1 else trimesh.util.concatenate(g)

    V, VT, FV, FT, FG = load_target_obj(obj_path)
    if not (FT >= 0).all():
        raise RuntimeError("target OBJ has faces without UVs -- gen_seams did not unwrap it")
    _log(f"target: {len(V):,} verts  {len(FV):,} tris  {len(VT):,} uvs")

    mat = getattr(source.visual, "material", None)
    bct = getattr(mat, "baseColorTexture", None)
    if bct is None:
        raise RuntimeError("source mesh has no baseColorTexture -- nothing to transfer")
    img = np.asarray(bct.convert("RGB")).astype(np.float32) / 255.0
    suv = np.asarray(source.visual.uv)
    SV = np.asarray(source.vertices)
    SF = np.asarray(source.faces)
    H, W = img.shape[:2]

    # sample the source at each face's UV CENTROID -- inside the chart, never on its padded edge
    cuv = suv[SF].mean(axis=1)
    px = np.clip((cuv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    py = np.clip(((1 - cuv[:, 1]) * (H - 1)).astype(np.int32), 0, H - 1)
    scol = img[py, px]

    # Metallic-roughness rides the same lookup. The Trellis material is metallic 1.0 / roughness 1.0
    # with an MR TEXTURE carrying the real values; shipping base colour alone with metallic 0 renders
    # flat and washed out, which is what made the first transfers look lighter than the original.
    smr = None
    mrt = getattr(mat, "metallicRoughnessTexture", None)
    if mrt is not None:
        mri = np.asarray(mrt.convert("RGB")).astype(np.float32) / 255.0
        smr = mri[np.clip(((1 - cuv[:, 1]) * (mri.shape[0] - 1)).astype(np.int32), 0, mri.shape[0] - 1),
                  np.clip((cuv[:, 0] * (mri.shape[1] - 1)).astype(np.int32), 0, mri.shape[1] - 1)]
        _log(f"source has an MR texture {mri.shape} - transferring it too")

    spos = _unit(SV[SF].mean(axis=1))
    snrm = np.asarray(source.face_normals)
    tree = cKDTree(spos)
    _log(f"source: {len(SF):,} faces")

    # SPECKLE REJECTION. Trellis's texture carries isolated wrong-coloured texels, and the target is
    # ~19x sparser than the source, so every speck that gets picked is magnified into a visible dot.
    # The guard further down compares the chosen candidate against the k nearest CANDIDATES, which
    # fails exactly when it matters: a speck usually spans several of those k, so the median is
    # contaminated too and nothing is rejected. Judge each source face against its OWN neighbourhood
    # on the source mesh instead -- that separates the two populations cleanly (measured on a real
    # bake: faces behind visible dots deviate 0.0328 median / 0.4280 p90 from their neighbours,
    # against 0.0055 / 0.0204 for faces that look right).
    # Genuine fine detail survives because it is spatially COHERENT: a hair strand, a nipple or an
    # iris shares its colour with the faces around it, so its deviation stays low. Only a face that
    # disagrees with everything touching it is dropped. The threshold is a multiple of the source's
    # OWN median deviation, so it adapts to how noisy that particular generation is.
    _t0 = time.time()
    _, _knn = tree.query(spos, k=13, workers=-1)
    _smooth = np.median(scol[_knn], axis=1)
    _speck = np.linalg.norm(scol - _smooth, axis=1)
    _sthr = float(np.median(_speck)) * 8.0
    _log(f"speckle map: median deviation {np.median(_speck):.5f}, rejecting above {_sthr:.5f} "
         f"({100.0 * (_speck > _sthr).mean():.2f}% of source faces) [{time.time() - _t0:.1f}s]")

    # ---- rasterise the target UV triangles -> per-texel 3D position + face normal ----
    _log(f"rasterising target UVs into {size}x{size} ...")
    pos = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    mask = np.zeros((size, size), bool)
    e1 = V[FV[:, 1]] - V[FV[:, 0]]
    e2 = V[FV[:, 2]] - V[FV[:, 0]]
    fn = np.cross(e1, e2)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    uvp = VT[FT] * np.array([size - 1, size - 1])
    uvp[..., 1] = (size - 1) - uvp[..., 1]
    for i in range(len(FV)):
        t = uvp[i]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, size - 1); y1 = min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = np.stack([xs, ys], -1).astype(np.float64) - t[0]
        v0 = t[1] - t[0]; v1 = t[2] - t[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        a = (d[..., 0] * v1[1] - v1[0] * d[..., 1]) / den
        b = (v0[0] * d[..., 1] - d[..., 0] * v0[1]) / den
        ok = (a >= -1e-4) & (b >= -1e-4) & (a + b <= 1 + 1e-4)
        if not ok.any():
            continue
        w0 = (1 - a - b)[ok]; w1 = a[ok]; w2 = b[ok]
        p = (V[FV[i, 0]] * w0[:, None] + V[FV[i, 1]] * w1[:, None] + V[FV[i, 2]] * w2[:, None])
        yy = ys[ok]; xx = xs[ok]
        pos[yy, xx] = p
        nrm[yy, xx] = fn[i]
        mask[yy, xx] = True
    _log(f"rasterised {int(mask.sum()):,} texels ({100 * mask.mean():.1f}% of the atlas)")

    # ---- sample ----
    yy, xx = np.nonzero(mask)
    Vc = 0.5 * (V.min(0) + V.max(0))
    Vs = float((V.max(0) - V.min(0)).max())
    q = (pos[yy, xx] - Vc) / max(Vs, 1e-9)
    qn = nrm[yy, xx]
    out = np.zeros((size, size, 3), np.float32)
    outmr = np.zeros((size, size, 3), np.float32) if smr is not None else None
    sourced = np.zeros((size, size), bool)
    nfall = 0
    CH = 400000
    for s0 in range(0, len(q), CH):
        e = min(s0 + CH, len(q))
        d, j = tree.query(q[s0:e], k=K, workers=-1)
        ag = np.abs(np.einsum('ijk,ik->ij', snrm[j], qn[s0:e])) > 0.20   # orientation, not direction
        c = scol[j]
        w = ag.astype(np.float32)
        none = ~ag.any(1)
        nfall += int(none.sum())
        w[none] = 1.0
        cm = np.where(w[..., None] > 0, c, np.nan)
        with np.errstate(all='ignore'):
            med = np.nanmedian(cm, axis=1)
        idx0 = np.argmax(ag, axis=1)
        idx0[none] = 0
        near = c[np.arange(len(c)), idx0]
        dev = np.linalg.norm(near - med, axis=1)
        dm = np.linalg.norm(c - med[:, None, :], axis=2)
        dm = np.where(w > 0, dm, np.inf)
        medoid = c[np.arange(len(c)), np.argmin(dm, axis=1)]
        # nearest keeps genuine fine detail (hair strands); the medoid replaces it only where the
        # nearest disagrees strongly with its neighbours, which is the speckle case
        out[yy[s0:e], xx[s0:e]] = np.where((dev > 0.12)[:, None], medoid, near)
        sourced[yy[s0:e], xx[s0:e]] = True
        if outmr is not None:
            outmr[yy[s0:e], xx[s0:e]] = smr[j[np.arange(len(j)), idx0]]
    _log(f"no normal-agreeing face for {100.0 * nfall / max(len(q), 1):.2f}% of texels")

    # PAD island edges: the atlas is empty outside them, so a sampler filtering across an island
    # edge pulls in black and the seam reads as a dark line on the model.
    def _pad(buf, filled, passes=16):
        f = filled.copy()
        for _ in range(passes):
            grow = f.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sh = np.roll(f, (dy, dx), (0, 1))
                take = sh & ~grow
                if take.any():
                    buf[take] = np.roll(buf, (dy, dx), (0, 1))[take]
                    grow |= take
            if (grow == f).all():
                break
            f = grow
        return f

    filled = _pad(out, sourced)
    if outmr is not None:
        _pad(outmr, sourced)
    _log(f"padded to {100 * filled.mean():.1f}% (islands {100 * mask.mean():.1f}%) in {time.time() - t0:.1f}s")
    return np.clip(out, 0, 1), (np.clip(outmr, 0, 1) if outmr is not None else None)
