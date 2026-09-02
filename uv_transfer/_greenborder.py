"""GREEN-BORDER hairline bridge (production entry point).

Computes the path3 crease-based hairline REGION on the high-res texturing/paint mesh -- the dense clean
mesh path3 was tuned on -- and returns a per-face hair flag. This replaces gen_seams' coarse 2-line
hairline as the texturing hair mask (and FBX Hair material).

Input  GB_IN  (npz): co = paint-mesh verts, Y-up texturing space (X=lateral, Y=height, Z=depth front=max)
                     fv = faces (tris), hair = ROUGH per-face hair region (gen_seams' is_hair transferred
                          to the paint faces -- coverage only; the crease FILTER defines the real hairline)
Output GB_OUT (npy): per-face bool hair region (same face order as fv)

Pipeline: gen-space remap -> arm/face cleanup -> de-speckle -> crease_curv -> path3 -> _hairfill.
The 3 chain stages are the SAME scripts used in research, run as subprocesses (they are env+npz driven)."""
import numpy as np, os, sys, subprocess
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _cc
from collections import defaultdict

UV = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _gen_normalize(co):
    co = co.copy()
    z_min, z_max = float(co[:, 2].min()), float(co[:, 2].max())
    x_ctr = float((co[:, 0].min() + co[:, 0].max()) / 2.0); x_span = float(co[:, 0].max() - co[:, 0].min())
    z_scale = 0.91 / (z_max - z_min); z_off = -0.41 - z_min * z_scale; x_scale = 0.92 / x_span
    co[:, 2] = co[:, 2] * z_scale + z_off; co[:, 0] = (co[:, 0] - x_ctr) * x_scale; co[:, 1] = co[:, 1] * z_scale
    return co


def compute(co_paint, fv, rough_hair):
    """co_paint: (Nv,3) paint mesh Y-up.  fv: (Nf,3) tris.  rough_hair: (Nf,) bool. -> (Nf,) bool region."""
    nf = len(fv)
    co = np.column_stack([co_paint[:, 0], -co_paint[:, 2], co_paint[:, 1]]).astype(np.float64)  # -> gen axes
    co = _gen_normalize(co)
    hair = rough_hair.astype(bool).copy()
    fc = co[fv].mean(1)
    znv = (co[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)
    fznv = (fc[:, 2] - co[:, 2].min()) / (co[:, 2].max() - co[:, 2].min() + 1e-9)
    hw = 0.5 * float(np.ptp(co[znv >= 0.858, 0])) if (znv >= 0.858).any() else 0.1

    if not os.environ.get('GB_NOCARVE'):
        # arm/shoulder: a drape can't be wider than the head below the neck floor (head-relative)
        hair &= ~((fznv < 0.858) & (np.abs(fc[:, 0]) > 1.5 * hw))
        # face-skin carve: drop front-facing central face skin (gen_seams' _facefwd) so the proximity guard
        # can't anchor on the bare throat/chin (which would pull the loop down the front neck)
        fn = np.cross(co[fv[:, 1]] - co[fv[:, 0]], co[fv[:, 2]] - co[fv[:, 0]])
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
        ctr = co.mean(0); flip = ((fc - ctr) * fn).sum(1) < 0; fn[flip] = -fn[flip]
        hair &= ~((fn[:, 1] < -0.52) & (fc[:, 2] > 0.30) & (np.abs(fc[:, 0]) < 0.15))

    # de-speckle: keep substantial hair components (crown + ponytail chunks), drop transfer specks
    e2f = defaultdict(list)
    for fi in range(nf):
        a, b, c = int(fv[fi, 0]), int(fv[fi, 1]), int(fv[fi, 2])
        for u, w in ((a, b), (b, c), (c, a)):
            e2f[(u, w) if u < w else (w, u)].append(fi)
    rr = []; cc = []
    for fs in e2f.values():
        if len(fs) == 2 and hair[fs[0]] and hair[fs[1]]:
            rr.append(fs[0]); cc.append(fs[1])
    A = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nf, nf)); A = A + A.T
    ncomp, lab = _cc(A, directed=False)
    sizes = np.bincount(lab[hair], minlength=ncomp) if hair.any() else np.zeros(ncomp)
    thr = max(50, int(0.005 * int(hair.sum())))
    hair = np.isin(lab, np.where(sizes >= thr)[0]) & hair

    # write the path3-format mesh and run the chain (crease_curv -> path3 -> _hairfill)
    mp = os.path.join(UV, '_gb_mesh.npz')
    np.savez(mp, co=co.astype(np.float32), fv=fv.astype(np.int64), hair=hair, nv=len(co))
    base = dict(os.environ)
    def run(script, extra):
        subprocess.run([PY, os.path.join(UV, script)], cwd=UV, check=True,
                       env={**base, **extra}, stdout=subprocess.DEVNULL)
    run('crease_curv_v2.py', {'CR_MESH': '_gb_mesh.npz', 'CR_OUT': '_gb_crease.npz'})   # v2: density-scalable MINC
    run('path3_v2.py', {'P2_MESH': '_gb_mesh.npz', 'P2_CRE': '_gb_crease.npz', 'P2_CRE_RAW': '_gb_crease.npz',
                        'P2_OUT': '_gb_loop.npz', 'P2_FILT_OUT': '_gb_filt.npz', 'P2_CLOSE': '0',
                        'P2_DRAPE_KEEPCRE': '1',   # KEEP below-floor drape concave creases (don't front-suppress them)
                        'P2_DRAPE_BRIDGE': '1'})   # BRIDGE the isolated drape creases so the loop rides DOWN to them (no gap / up-routing)
    try:   # REGRESSION TRIPWIRE: print hairline-follow coverage every bake; screams on a drop vs the baseline
        subprocess.run([PY, os.path.join(UV, '_hairline_follow_check.py')], cwd=UV, env=base, timeout=120)
    except Exception as _fce:
        print('[follow-check] skipped (%r)' % _fce)
    run('_hairfill.py', {'HF_MESH': '_gb_mesh.npz', 'HF_LOOP': '_gb_loop.npz', 'HF_OUT': '_gb_fill.npz',
                         'HF_POCKETS': '0'})
    fill = np.load(os.path.join(UV, '_gb_fill.npz'))['faces']
    out = np.zeros(nf, bool); out[fill[fill < nf]] = True
    return out


if __name__ == '__main__':   # standalone test: drive from GB_IN / GB_OUT, else the fixture
    if os.environ.get('GB_IN'):
        d = np.load(os.environ['GB_IN'])
        res = compute(d['co'].astype(np.float64), d['fv'].astype(np.int64), d['hair'].astype(bool))
        np.save(os.environ['GB_OUT'], res)
        print('[greenborder] %d/%d hair faces -> %s' % (int(res.sum()), len(res), os.environ['GB_OUT']))
    else:
        # fixture self-test: paint mesh = fixture norm_verts; rough mask = gen is_hair transferred
        PROJ = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
        fx = np.load(os.path.join(PROJ, 'hair_detection_fixture.npz'))
        nvv = fx['norm_verts'].astype(np.float64); nf_ = fx['norm_faces'].astype(np.int64)
        from scipy.spatial import cKDTree
        co_t = _gen_normalize(np.column_stack([nvv[:, 0], -nvv[:, 2], nvv[:, 1]]))
        gv = np.load(os.path.join(UV, '_genseams_viz.npz'))
        gco = gv['co'].astype(np.float64); gt = gv['tris'].astype(np.int64); gh = gv['tris_hair'].astype(bool)
        _, gi = cKDTree(gco[gt].mean(1)).query(co_t[nf_].mean(1), workers=-1)
        res = compute(nvv, nf_, gh[gi])
        print('[greenborder] fixture self-test: %d/%d hair faces' % (int(res.sum()), len(res)))
