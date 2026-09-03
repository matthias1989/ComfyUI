"""KAN-18 genital texturing -- runs PER CHARACTER, AFTER gen_seams (the body UVMap + genital island exist).

Approach B-corrected / hair-style: bakes the genital zone maps (albedo / roughness / normal) onto the BODY
UVMap's genital island -- the SAME single UV the FBX uses. It does NOT add a material or a UV layer (that was
the bug: a 2nd material -> 2 GLB primitives -> the export round-trip mangled the body). The FBXExport step
then adds a separate 'Genital' material on slot 2 (on the _gs_isgenital faces), sampling these maps via the
body UVMap -- exactly how the Hair material is added on slot 1.

Pipeline: graft saves _gs_isgenital.npy (per-face genital flag, face-order stable through the whole pipeline).
gen_seams unwraps (genital = its own island in the body UVMap). THEN this runs: propagate vulva zones from the
donor + parametric anus, feather, rasterize the zone colours at the genital faces' BODY-UV coords -> 3 PNGs.

Zones: 0 skin, 1 outer_labia_outer, 2 outer_labia_inner, 3 inner_labia, 4 opening, 5 anus_ring, 6 anal_opening.
Runs in Blender (bpy). Entry: texture_genital(obj, out_dir).  Test: blender -b -P genital_texture.py -- body.blend out_dir
"""
import bpy, numpy as np, os, sys
from mathutils import Vector, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import pelvis_graft as pg
ASSET = os.path.join(HERE, "assets")
ISGEN = os.path.join(HERE, "..", "..", "uv_transfer", "_gs_isgenital.npy")   # per-face genital flag from the graft
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # Z-up <-> Y-up (graft convention)

# Outer labia (majora, zones 1-2) read as WARM SKIN -- barely pinker than the body, NOT rosy. Inner labia
# (minora, zone 3) are a distinctly saturated rosy pink so they clearly contrast the skin-toned outer. Opening
# (4) deeper rose. (Per the user's reference: majora ~= skin, minora = clear pink.)
ALB = {0: (0.80, 0.60, 0.52), 1: (0.82, 0.58, 0.52), 2: (0.83, 0.54, 0.48), 3: (0.85, 0.40, 0.45),
       4: (0.62, 0.24, 0.29), 5: (0.50, 0.28, 0.28), 6: (0.30, 0.13, 0.15)}
ROU = {0: 0.66, 1: 0.55, 2: 0.40, 3: 0.26, 4: 0.16, 5: 0.30, 6: 0.18}
ANUS_RING_R, ANUS_OPEN_R = 0.015, 0.009


def _save_png(rgb, path):
    res = rgb.shape[0]
    rgba = np.flipud(np.dstack([rgb, np.ones((res, res), np.float32)]))   # bpy pixels are bottom-up
    img = bpy.data.images.new(os.path.basename(path), res, res, alpha=True)
    img.pixels.foreach_set(rgba.ravel()); img.filepath_raw = path; img.file_format = 'PNG'; img.save()
    bpy.data.images.remove(img)


def _vnorm(me, nv):
    VN = np.zeros((nv, 3))
    for p in me.polygons:
        n = np.array(p.normal)
        for vi in p.vertices:
            VN[vi] += n
    return VN / (np.linalg.norm(VN, axis=1, keepdims=True) + 1e-9)


def _rasterize(tris_uv, tris_val, res, fill):
    C = tris_val.shape[2]
    img = np.zeros((res, res, C), np.float32); cov = np.zeros((res, res), bool)
    for t in range(len(tris_uv)):
        uv = tris_uv[t].reshape(3, 2); val = tris_val[t]
        px = uv[:, 0] * res; py = (1 - uv[:, 1]) * res
        x0 = max(int(np.floor(px.min())), 0); x1 = min(int(np.ceil(px.max())) + 1, res)
        y0 = max(int(np.floor(py.min())), 0); y1 = min(int(np.ceil(py.max())) + 1, res)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d00 = px[1] - px[0]; d01 = py[1] - py[0]; d10 = px[2] - px[0]; d11 = py[2] - py[0]
        den = d00 * d11 - d10 * d01
        if abs(den) < 1e-9:
            continue
        wx = xs - px[0]; wy = ys - py[0]
        v = (wx * d11 - d10 * wy) / den; w = (d00 * wy - wx * d01) / den; u = 1 - v - w
        ins = (u >= -0.001) & (v >= -0.001) & (w >= -0.001)
        yy, xx = np.where(ins); ay = y0 + yy; ax = x0 + xx
        uu = u[ins][:, None]; vv = v[ins][:, None]; ww = w[ins][:, None]
        img[ay, ax] = uu * val[0] + vv * val[1] + ww * val[2]; cov[ay, ax] = True
    for _ in range(6):                                  # edge-bleed dilation (no scipy)
        unc = ~cov; s = np.zeros_like(img); c = np.zeros((res, res))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            s += np.roll(np.roll(img, dy, 0), dx, 1) * np.roll(np.roll(cov, dy, 0), dx, 1)[..., None]
            c += np.roll(np.roll(cov, dy, 0), dx, 1)
        fillm = unc & (c > 0); img[fillm] = s[fillm] / c[fillm][..., None]; cov = cov | fillm
        if not unc.any():
            break
    img[~cov] = fill
    return img


def texture_genital(obj, out_dir, res=2048):
    out_dir = os.path.abspath(out_dir)
    me = obj.data; bpy.context.view_layer.objects.active = obj
    nv = len(me.vertices)
    V = np.empty(nv * 3); me.vertices.foreach_get("co", V); V = V.reshape(nv, 3)
    VN = _vnorm(me, nv)
    # genital faces = the graft's per-face flag (stable face order through the pipeline; SAME set the FBX uses)
    isgen = np.load(ISGEN).astype(bool)
    if len(isgen) != len(me.polygons):
        raise RuntimeError(f"_gs_isgenital len {len(isgen)} != polys {len(me.polygons)}")
    genvert = np.zeros(nv, bool)
    for fi in np.where(isgen)[0]:
        for vi in me.polygons[fi].vertices:
            genvert[vi] = True

    # 1) propagate VULVA zones from the canonical donor (nearest placed-donor vertex), genital verts only
    Vy = V @ M
    W, Fd, A = pg._place(Vy)
    dz = np.load(os.path.join(ASSET, "genital_donor_zones.npy"))
    kd = kdtree.KDTree(len(W))
    for i, w in enumerate(W):
        kd.insert(Vector(w.tolist()), i)
    kd.balance()
    zones = np.zeros(nv, int)
    for i in np.where(genvert)[0]:
        loc, idx, dist = kd.find(Vector(Vy[i].tolist()))
        if dist < 0.05:
            zones[i] = int(dz[idx])

    # 2) ANUS: anchor to the posterior fourchette + a RELATIVE perineal offset. The old back-percentile centroid
    #    drifted ~4 cm too far back into the gluteal cleft (measured on the target); this lands on the real
    #    perineal depression (validated vs the target concavity). Offsets are fractions of the vulva cleft length
    #    (scale-relative), midline-centred -- no magic world distances.
    vmask = np.isin(zones, [1, 2, 3, 4]) & genvert
    opmask = np.isin(zones, [3, 4]) & genvert
    gi = np.where(genvert)[0]
    if opmask.any() and vmask.any():
        fourch = V[np.where(opmask)[0][np.argmax(V[opmask][:, 1])]]   # back-most opening vertex (fourchette)
        vh = np.ptp(V[vmask][:, 2])                                   # vulva cleft length -> relative scale
        anus_t = np.array([0.0, fourch[1] + 0.35 * vh, fourch[2] + 0.12 * vh])
        backc = V[gi][np.argmin(np.linalg.norm(V[gi] - anus_t, axis=1))]
    else:
        by = np.percentile(V[genvert, 1], 78); backc = V[genvert & (V[:, 1] > by)].mean(0)
    danus = np.linalg.norm(V - backc, axis=1)
    zones[genvert & (danus < ANUS_RING_R)] = 5
    zones[genvert & (danus < ANUS_OPEN_R)] = 6

    # adjacency over genital verts (shared by the feather alpha, the material flag, and the colour feather)
    # Seeded on the genital verts, but CLOSED under traversal: the ring growth below deliberately
    # walks off the genital into the body (that is what makes the alpha ramp fade onto body skin),
    # so every vertex reachable as a neighbour must itself be a key. Keying only the genital verts
    # while storing body verts as values left the graph one-directional and the walk raised
    # KeyError on the first body vertex it stepped onto.
    nb = {i: set() for i in np.where(genvert)[0]}
    for p in me.polygons:
        vs = [v for v in p.vertices if v in nb]
        for a in vs:
            for b in p.vertices:
                if a != b:
                    nb[a].add(b)
                    nb.setdefault(b, set()).add(a)

    # 2b) feather alpha + material flag, from ring-distance to the FEATURE verts (zones 1-6).
    #     alpha: 1 on the features, ramping to 0 by RAMP rings -- the FBXExport composite then reads BODY skin
    #     where alpha<1, so the separate-material border fades into the exact body pixels (invisible seam).
    #     flag: features + FLAG_RINGS rings -- wide enough that its boundary sits in the alpha~0 band, so the
    #     material edge lands on body-matching skin and is invisible regardless of how jagged the cut is.
    feat = np.isin(zones, [1, 2, 3, 4, 5, 6]) & genvert
    RAMP, FLAG_RINGS = 4.0, 5
    ring = np.full(nv, 99, int); ring[feat] = 0
    cur = set(np.where(feat)[0])
    for r in range(1, FLAG_RINGS + 2):
        nxt = set()
        for i in cur:
            for j in nb[i]:
                if ring[j] > r:
                    ring[j] = r; nxt.add(j)
        cur = nxt
    alpha = np.clip(1.0 - ring / RAMP, 0.0, 1.0)
    fface = np.array([any(ring[v] <= FLAG_RINGS for v in p.vertices) for p in me.polygons])
    np.save(os.path.join(os.path.dirname(ISGEN), "_gs_genital_mat.npy"), fface)
    print("genital material faces:", int(fface.sum()), "/", len(me.polygons), "anus@", np.round(backc, 4))

    # 3) feather per-vertex albedo + roughness over the genital verts; re-assert the small anus zones after
    alb = np.array([ALB[int(z)] for z in zones]); rou = np.array([ROU[int(z)] for z in zones])
    for _ in range(4):
        na = alb.copy(); nr = rou.copy()
        for i in nb:
            ns = list(nb[i])
            if ns:
                na[i] = 0.5 * alb[i] + 0.5 * alb[ns].mean(0); nr[i] = 0.5 * rou[i] + 0.5 * rou[ns].mean()
        alb = na; rou = nr
    for z in (5, 6):
        m = zones == z
        if m.any():
            alb[m] = ALB[z]; rou[m] = ROU[z]

    # 4) bake onto the BODY UVMap (gen_seams' single atlas UV) -- the genital island within it
    body_uv = me.uv_layers.get("UVMap") or me.uv_layers.active
    if body_uv is None:
        raise RuntimeError("no UV map -- gen_seams must run before genital texturing")
    me.uv_layers.active = body_uv
    me.calc_tangents(uvmap=body_uv.name)
    Wz = W @ M.T
    dfc = Wz[Fd].mean(1)
    e1 = Wz[Fd[:, 1]] - Wz[Fd[:, 0]]; e2 = Wz[Fd[:, 2]] - Wz[Fd[:, 0]]
    dn = np.cross(e1, e2); dn /= (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9)
    dkd = kdtree.KDTree(len(dfc))
    for i, c in enumerate(dfc):
        dkd.insert(Vector(c.tolist()), i)
    dkd.balance()
    uvl = body_uv.data
    a_uv, a_alb, a_rou, a_nrm, a_alp, a_zn = [], [], [], [], [], []
    for fi in np.where(isgen)[0]:
        p = me.polygons[fi]; li = list(p.loop_indices)
        cu = [tuple(uvl[l].uv) for l in li]; cv = [me.loops[l].vertex_index for l in li]
        cn = []
        for l, vi in zip(li, cv):
            lp = me.loops[l]; N = VN[vi]; T = np.array(list(lp.tangent)); B = np.cross(N, T) * lp.bitangent_sign
            loc, idx, dist = dkd.find(Vector(V[vi].tolist())); hn = dn[idx]
            tn = np.array([hn @ T, hn @ B, hn @ N])
            if tn[2] < 0:
                tn = -tn
            cn.append(tn / (np.linalg.norm(tn) + 1e-9))
        for k in range(1, len(li) - 1):
            a_uv.append([cu[0][0], cu[0][1], cu[k][0], cu[k][1], cu[k + 1][0], cu[k + 1][1]])
            a_alb.append([alb[cv[0]], alb[cv[k]], alb[cv[k + 1]]])
            a_rou.append([[rou[cv[0]]], [rou[cv[k]]], [rou[cv[k + 1]]]])
            a_nrm.append([cn[0], cn[k], cn[k + 1]])
            a_alp.append([[alpha[cv[0]]], [alpha[cv[k]]], [alpha[cv[k + 1]]]])
            a_zn.append([[float(zones[cv[0]])], [float(zones[cv[k]])], [float(zones[cv[k + 1]])]])
    a_uv = np.array(a_uv)
    albimg = _rasterize(a_uv, np.array(a_alb), res, np.array(ALB[0]))
    rouimg = _rasterize(a_uv, np.array(a_rou), res, np.array([ROU[0]]))
    nrmimg = _rasterize(a_uv, np.array(a_nrm), res, np.array([0, 0, 1.0]))
    alpimg = _rasterize(a_uv, np.array(a_alp), res, np.array([0.0]))   # feather mask: 1 genital -> 0 body skin
    znimg = _rasterize(a_uv, np.array(a_zn), res, np.array([0.0]))[..., 0]   # per-pixel zone id (for skin-match re-tint)

    os.makedirs(out_dir, exist_ok=True)
    _save_png(np.clip(albimg, 0, 1), os.path.join(out_dir, "genital_albedo.png"))
    _save_png(np.repeat(np.clip(rouimg, 0, 1), 3, axis=2), os.path.join(out_dir, "genital_roughness.png"))
    _save_png(np.clip(nrmimg, -1, 1) * 0.5 + 0.5, os.path.join(out_dir, "genital_normal.png"))
    _save_png(np.repeat(np.clip(alpimg, 0, 1), 3, axis=2), os.path.join(out_dir, "genital_alpha.png"))
    np.save(os.path.join(out_dir, "genital_zonemap.npy"), np.rint(znimg).astype(np.int8))
    np.save(os.path.join(out_dir, "genital_zones.npy"), zones)
    return {"zones": {k: int((zones == k).sum()) for k in range(7)},
            "genital_faces": int(isgen.sum()), "out_dir": out_dir}


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    blend, outd = argv[0], argv[1]
    bpy.ops.wm.open_mainfile(filepath=blend)
    obj = max((x for x in bpy.data.objects if x.type == 'MESH'), key=lambda x: len(x.data.polygons))
    print("RESULT", texture_genital(obj, outd))   # NOTE: bakes maps only; does NOT modify/save the mesh
