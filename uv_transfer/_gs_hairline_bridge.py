"""Green-border -> gen_seams bridge. Computes the path3 hairline region on the HIGH-RES mesh (the dense
clean mesh path3 needs), then transfers it to gen_seams' OWN faces (_gs_cen.npy, me.polygons order) as
_gs_ishair.npy. gen_seams (GS_USE_ISHAIR=1) then overrides is_hair with it, so the SEAM follows the
green border. Also renders the resulting seam on the gen mesh for a sanity look.

High-res source: GS_HIRES npz (co=verts Y-up, fv=tris) if set, else the texturing fixture norm_verts.
Rough hair coverage: gen_seams' is_hair (_genseams_viz tris_hair) transferred to the high-res faces."""
import numpy as np, os, sys, importlib.util as ilu
from scipy.spatial import cKDTree

UV = os.path.dirname(os.path.abspath(__file__))
PROJ = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
spec = ilu.spec_from_file_location('_greenborder', os.path.join(UV, '_greenborder.py'))
gb = ilu.module_from_spec(spec); spec.loader.exec_module(gb)

# ---- high-res mesh (paint Y-up convention: X=lat, Y=height, Z=depth front=max) ----
if os.environ.get('GS_HIRES'):
    d = np.load(os.environ['GS_HIRES']); hv = d['co'].astype(np.float64); hf = d[('fv' if 'fv' in d.files else 'faces')].astype(np.int64)
else:
    # ROOT-CAUSE FIX (2026-06-20): compute path3 DIRECTLY on the production mesh (_genseams_viz) -- now a
    # dense ~157k quad-derived mesh path3 handles natively -- instead of the stale 149k TRIANGLE fixture
    # (hair_detection_fixture.npz, Jun 9). Source == target => the k-NN transfer below collapses to identity,
    # so there is NO cross-mesh resolution smear at the hairline. (GS_HIRES still overrides for explicit tests.)
    _vz0 = np.load(os.path.join(UV, '_genseams_viz.npz')); _gco0 = _vz0['co'].astype(np.float64)
    hv = np.column_stack([_gco0[:, 0], _gco0[:, 2], -_gco0[:, 1]])   # gen-space -> Y-up paint (compute() remaps it back)
    hf = _vz0['tris'].astype(np.int64)
co_hi_gen = gb._gen_normalize(np.column_stack([hv[:, 0], -hv[:, 2], hv[:, 1]]))   # high-res in gen space
hcen = co_hi_gen[hf].mean(1)

# ---- rough hair coverage from gen_seams' is_hair (transferred to high-res faces) ----
vz = np.load(os.path.join(UV, '_genseams_viz.npz'))
gco = vz['co'].astype(np.float64); gt = vz['tris'].astype(np.int64); gh = vz['tris_hair'].astype(bool)
_, gi = cKDTree(gco[gt].mean(1)).query(hcen, workers=-1)
rough = gh[gi]

# ---- GREEN BORDER on the high-res mesh ----
region = gb.compute(hv, hf, rough)            # per high-res face
print('[gs-bridge] green border on high-res: %d/%d hair faces' % (int(region.sum()), len(region)))

# ---- in-bake FACE MASK (MediaPipe on the front photo, == the offline _frozen mask) -> _gb_facevert.npy.
#      postfix_loop deletes the loop's across-face edges using it (geometric box is only the fallback). ----
import subprocess as _sp2
_photo = os.path.join(UV, '_front_photo.png'); _fvout = os.path.join(UV, '_gb_facevert.npy')
if os.path.exists(_fvout):
    os.remove(_fvout)                                   # fresh mask each bake; on failure -> geometric fallback
if os.path.exists(_photo):
    try:
        _sp2.run([sys.executable, os.path.join(UV, '_gs_facemask.py')], cwd=UV, check=True,
                 env={**os.environ, 'GS_MESH': os.path.join(UV, '_gb_mesh.npz'), 'GS_PHOTO': _photo, 'GS_OUT': _fvout})
    except Exception as _fme:
        print('[gs-bridge] facemask FAILED (%r) -> postfix uses geometric fallback' % _fme)
else:
    print('[gs-bridge] _front_photo.png missing -> postfix uses geometric face mask')

# ---- POST-FIX the hairline loop: DELETE path3's across-face edges using the face mask (the user-approved
#      fix; reroute-around-the-face was rejected). Saves _gb_loop_fix.npz; gen_seams marks it. ----
try:
    import importlib.util as _hpilu2
    _hps2 = _hpilu2.spec_from_file_location('_hairline_pipe', os.path.join(UV, '_hairline_pipe.py'))
    _hpm2 = _hpilu2.module_from_spec(_hps2); _hps2.loader.exec_module(_hpm2)
    _hpm2.postfix_loop(UV)
except Exception as _pfe:
    import traceback as _tb2; _tb2.print_exc()
    print('[gs-bridge] hairline post-fix FAILED (%r) -- gen_seams falls back to the raw loop' % _pfe)

# ---- transfer the region to gen_seams' faces ----
# The gen face centroids come straight from _genseams_viz (co + tris): the mesh is triangulated, so its
# tri order == me.polygons order == gen_seams' is_hair order. No separate centroid-export pass needed.
# k-NN MAJORITY (not nearest): the high-res mesh is ~2x denser, so nearest-face flip-flops at the
# resolution mismatch and shatters the region -> a noisy seam. A majority vote over the k nearest
# high-res faces transfers a smooth region.
gcen = gco[gt].mean(1)
_, hi = cKDTree(hcen).query(gcen, k=9, workers=-1)
ishair = region[hi].mean(1) > 0.5

# DE-SPECKLE on the gen mesh: keep substantial hair components (crown + ponytail), fill small non-hair
# holes, so the seam (is_hair boundary) is a clean curve instead of swiss-cheese.
import scipy.sparse as _sp
from scipy.sparse.csgraph import connected_components as _cc
from collections import defaultdict as _dd
_e2f = _dd(list)
for _fi in range(len(gt)):
    _a, _b, _c = int(gt[_fi, 0]), int(gt[_fi, 1]), int(gt[_fi, 2])
    for _u, _w in ((_a, _b), (_b, _c), (_c, _a)):
        _e2f[(_u, _w) if _u < _w else (_w, _u)].append(_fi)
def _comps(want_mask):
    rr = []; cc = []
    for fs in _e2f.values():
        if len(fs) == 2 and want_mask[fs[0]] and want_mask[fs[1]]:
            rr.append(fs[0]); cc.append(fs[1])
    A = _sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(gt), len(gt))); A = A + A.T
    return _cc(A, directed=False)
# keep hair components >= 1% of the hair (drop transfer specks; keep crown + ponytail chunks)
_n, _lab = _comps(ishair)
_sz = np.bincount(_lab[ishair], minlength=_n)
ishair = np.isin(_lab, np.where(_sz >= max(30, int(0.01 * int(ishair.sum()))))[0]) & ishair
# fill small non-hair HOLES (non-hair comps not touching the big body) -> solid hair, no pinholes
_nb, _labb = _comps(~ishair)
_szb = np.bincount(_labb[~ishair], minlength=_nb)
_bodyc = int(_szb.argmax())                       # the body is the largest non-hair component
_fill = (~ishair) & (_labb != _bodyc) & np.isin(_labb, np.where(_szb < max(50, int(0.02 * len(gt))))[0])
ishair = ishair | _fill
# SMOOTH the boundary: the cross-resolution transfer is ragged at the hairline (faces flip ~50/50),
# giving a noisy seam. Majority-vote over face neighbours (a few passes) -> a clean hairline curve.
_rr = []; _ccol = []
for fs in _e2f.values():
    if len(fs) == 2:
        _rr.append(fs[0]); _ccol.append(fs[1])
_Af = _sp.csr_matrix((np.ones(len(_rr)), (_rr, _ccol)), shape=(len(gt), len(gt))); _Af = _Af + _Af.T
_deg = np.maximum(np.asarray(_Af.sum(1)).ravel(), 1.0)
_m = ishair.astype(np.float64)
for _ in range(4):
    _nh = np.asarray(_Af @ _m).ravel() / _deg
    _m = np.where(_nh > 0.6, 1.0, np.where(_nh < 0.4, 0.0, _m))
ishair = _m > 0.5
# re-keep the largest hair component after smoothing (drop any speck the smooth detached)
_n2, _lab2 = _comps(ishair); _sz2 = np.bincount(_lab2[ishair], minlength=_n2)
ishair = np.isin(_lab2, np.where(_sz2 >= max(30, int(0.01 * int(ishair.sum()))))[0]) & ishair
# ---- aggregate the per-TRI region -> per-FACE (me.polygons order) for the GS_USE_ISHAIR override ----
# _genseams_viz fan-triangulates each mesh face (a quad -> 2 tris). gen_seams' override needs ONE flag per
# me.polygons face (n_faces = len(me.polygons)), NOT per tri -- on the QUAD production mesh these differ
# (314k tris vs 157k faces), so a per-tri save silently fails the override length check (gen_seams.py L1151)
# and gen_seams keeps its own hairline. Map tris back via tris_face + majority-vote. (All-tri mesh: 1:1 -> identity.)
_tface = vz['tris_face'].astype(np.int64)
_nfac = int(_tface.max()) + 1
_acc = np.zeros(_nfac); _cnt = np.zeros(_nfac)
np.add.at(_acc, _tface, ishair.astype(float)); np.add.at(_cnt, _tface, 1.0)
ishair_face = (_acc / np.maximum(_cnt, 1.0)) > 0.5
np.save(os.path.join(UV, '_gs_ishair.npy'), ishair_face)
print('[gs-bridge] -> %d/%d gen TRIS hair; aggregated to %d/%d FACES (%.1f%%) -> _gs_ishair.npy (me.polygons order)'
      % (int(ishair.sum()), len(ishair), int(ishair_face.sum()), _nfac, 100 * ishair_face.mean()))

# ---- sanity render: the seam (is_hair boundary) on the gen mesh, front + side ----
if int(os.environ.get('GB_RENDER', '1')):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from collections import defaultdict
    ish = ishair if len(ishair) == len(gt) else gh   # _gs_cen order == _genseams_viz tris order
    e2f = defaultdict(list)
    for fi in range(len(gt)):
        a, b, c = int(gt[fi, 0]), int(gt[fi, 1]), int(gt[fi, 2])
        for u, w in ((a, b), (b, c), (c, a)):
            e2f[(u, w) if u < w else (w, u)].append(fi)
    seam = [e for e, fs in e2f.items() if len(fs) == 2 and ish[fs[0]] != ish[fs[1]]]
    fc = gco[gt].mean(1)

    def panel(ax, a0, a1, t):
        sel = gco[:, 2] > 0.20
        ax.scatter(gco[sel, a0], gco[sel, a1], s=0.5, c='lightgray', lw=0)
        hs = ish & (fc[:, 2] > 0.20)
        ax.scatter(fc[hs, a0], fc[hs, a1], s=1.0, c='gold', lw=0)
        ax.add_collection(LineCollection([[(gco[u, a0], gco[u, a1]), (gco[w, a0], gco[w, a1])] for u, w in seam],
                                         colors='red', linewidths=1.3))
        ax.set_title(t); ax.set_aspect('equal')
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    panel(axs[0], 0, 2, 'FRONT (x,z)  red=SEAM (is_hair boundary)  gold=hair')
    panel(axs[1], 1, 2, 'RIGHT (y,z)  front=left(min y)')
    plt.tight_layout(); out = os.path.join(UV, '_gs_seam_look.png'); plt.savefig(out, dpi=95)
    print('[gs-bridge] seam render -> %s  (%d seam edges)' % (out, len(seam)))
