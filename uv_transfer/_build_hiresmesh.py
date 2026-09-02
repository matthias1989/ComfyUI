"""Build a path3-format mesh from the HIGH-RES texturing mesh (hair_detection_fixture.npz,
149k verts) in gen_seams' coordinate space, so the green-border chain (crease_curv -> path3 ->
_hairfill) runs on the DENSE clean mesh path3 was tuned on -- not the coarse decimated tri mesh.

Fixture norm_verts space:  X=lateral  Y=height  Z=depth(front=max)
gen_seams / path3 space :  x=lateral  y=depth(front=min)  z=height
  co_hi = [X, -Z, Y], then gen_seams' OWN normalization (deterministic, per-mesh bbox) so the
  high-res mesh lands in the SAME frame as the final mesh (_genseams_viz). Same char -> coincide.

Hair mask (DENSE, per-face on the high-res mesh) from the real detector, mapped via the mesh's own
UVs. Density matters: the proximity guard seeds from hair faces, so a sparse transferred mask
fragments it. path3's head/drape-decoupled binning absorbs the drape bloat; lateral-tight
front-suppression absorbs face bloat. Out: _hiresmesh.npz + _hires_cen.npy (centroids, gen space)."""
import numpy as np, os, sys
from scipy.spatial import cKDTree

D = os.path.dirname(__file__)
PROJ = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
FIX = os.path.join(PROJ, 'hair_detection_fixture.npz')

fx = np.load(FIX)
nvv = fx['norm_verts'].astype(np.float64); nf = fx['norm_faces'].astype(np.int64)

# ---- coordinates: remap to gen axis order, gen_seams normalization ----
co_hi = np.column_stack([nvv[:, 0], -nvv[:, 2], nvv[:, 1]])
def gen_normalize(co):
    co = co.copy()
    z_min, z_max = float(co[:, 2].min()), float(co[:, 2].max())
    x_ctr = float((co[:, 0].min() + co[:, 0].max()) / 2.0); x_span = float(co[:, 0].max() - co[:, 0].min())
    z_scale = 0.91 / (z_max - z_min); z_off = -0.41 - z_min * z_scale; x_scale = 0.92 / x_span
    co[:, 2] = co[:, 2] * z_scale + z_off; co[:, 0] = (co[:, 0] - x_ctr) * x_scale; co[:, 1] = co[:, 1] * z_scale
    return co
co_hi = gen_normalize(co_hi)

# ---- hair mask: transfer gen_seams' is_hair (tris_hair = crown+cap+drape, tight ~9.6%, no chest/arm
# bloat) to the high-res faces by nearest gen face. Use is_hair (NOT the raw shell, which is drape-only
# -> would leave the front hairline un-anchored). Its boundary is the "bad" hairline we are replacing,
# but its COVERAGE is exactly the rough region crease_curv(band)/path3(guard) need. ----
gv = np.load(os.path.join(D, '_genseams_viz.npz'))
gco = gv['co'].astype(np.float64); gtris = gv['tris'].astype(np.int64); ghair = gv['tris_hair'].astype(bool)
gcen = gco[gtris].mean(1); hcen = co_hi[nf].mean(1)
dist, gi = cKDTree(gcen).query(hcen, k=1, workers=-1)
hair_hi = ghair[gi]
print('hi faces=%d  hair(is_hair transfer)=%d (%.1f%%)  meanNN=%.4f  [gen is_hair=%.1f%%]'
      % (len(nf), int(hair_hi.sum()), 100*hair_hi.mean(), dist.mean(), 100*ghair.mean()))

# ARM/SHOULDER cleanup: a hair drape hangs from the head, so BELOW the neck floor it can't be wider
# than the head. Drop below-floor faces clearly off-head laterally (head-relative, not a fixed value).
_znv = (co_hi[:, 2] - co_hi[:, 2].min()) / (co_hi[:, 2].max() - co_hi[:, 2].min() + 1e-9)
_hw = 0.5 * float(np.ptp(co_hi[_znv >= 0.858, 0])) if (_znv >= 0.858).any() else 0.1
_fc = co_hi[nf].mean(1); _fznv = (_fc[:, 2] - co_hi[:, 2].min()) / (co_hi[:, 2].max() - co_hi[:, 2].min() + 1e-9)
_arm = (_fznv < 0.858) & (np.abs(_fc[:, 0]) > 1.5 * _hw)
_n0 = int(hair_hi.sum()); hair_hi &= ~_arm
print('arm/shoulder cleanup: head_halfwidth=%.3f  dropped %d off-head lateral faces -> hair=%d'
      % (_hw, _n0 - int(hair_hi.sum()), int(hair_hi.sum())))

# FACE-SKIN carve: gen is_hair over-flags the front face (forehead/throat/cheek skin). That makes
# path3's proximity guard anchor on bare THROAT/CHIN creases -> the loop dips down the front neck and
# the fill over-extends to the chest. Remove front-facing central face-skin (gen_seams' own _facefwd
# test, in gen space). Curved hair survives (its normal isn't purely -y), so the front hairline stays.
_fn = np.cross(co_hi[nf[:, 1]] - co_hi[nf[:, 0]], co_hi[nf[:, 2]] - co_hi[nf[:, 0]])
_fn /= (np.linalg.norm(_fn, axis=1, keepdims=True) + 1e-9)
_ctr = co_hi.mean(0); _flip = ((_fc - _ctr) * _fn).sum(1) < 0; _fn[_flip] = -_fn[_flip]   # orient outward
_facefwd = (_fn[:, 1] < -0.52) & (_fc[:, 2] > 0.30) & (np.abs(_fc[:, 0]) < 0.15)
_n1 = int(hair_hi.sum()); hair_hi &= ~_facefwd
print('face-skin carve: removed %d front-facing central faces -> hair=%d' % (_n1 - int(hair_hi.sum()), int(hair_hi.sum())))

# DE-SPECKLE: the nearest-face transfer leaves stray hair specks scattered across the body, which shatter
# the non-hair body into ~30k components (and a couple-iteration smooth doesn't remove them). Keep only
# the LARGEST connected hair component (the real crown+drape blob); every stray island reverts to body,
# so the body re-merges into ~one region -> _hairfill's full mask boundary and body-flood behave.
import scipy.sparse as _sp
from scipy.sparse.csgraph import connected_components as _cc
from collections import defaultdict as _dd
_e2f = _dd(list)
for _fi in range(len(nf)):
    _a, _b, _c = int(nf[_fi, 0]), int(nf[_fi, 1]), int(nf[_fi, 2])
    for _u, _w in ((_a, _b), (_b, _c), (_c, _a)):
        _e2f[(_u, _w) if _u < _w else (_w, _u)].append(_fi)
_rr = []; _cc2 = []
for _fs in _e2f.values():
    if len(_fs) == 2 and hair_hi[_fs[0]] and hair_hi[_fs[1]]:   # hair<->hair edges only
        _rr.append(_fs[0]); _cc2.append(_fs[1])
_Ah = _sp.csr_matrix((np.ones(len(_rr)), (_rr, _cc2)), shape=(len(nf), len(nf))); _Ah = _Ah + _Ah.T
_ncomp, _lab = _cc(_Ah, directed=False)
_sizes = np.bincount(_lab[hair_hi], minlength=_ncomp)
# keep every SUBSTANTIAL hair component (crown + the ponytail/drape, which the reconstruction often
# splits into chunks), drop only tiny transfer specks. Threshold is relative to the total hair.
_thr = max(50, int(0.005 * int(hair_hi.sum())))
_keep = np.where(_sizes >= _thr)[0]
_n2 = int(hair_hi.sum()); hair_hi = np.isin(_lab, _keep) & hair_hi
print('de-speckle: kept %d hair components >=%d faces -> %d/%d faces (dropped %d speck faces)'
      % (len(_keep), _thr, int(hair_hi.sum()), _n2, _n2 - int(hair_hi.sum())))

print('co_hi bbox=%s  (gen bbox=%s)' % ((co_hi.max(0)-co_hi.min(0)).round(3), (gco.max(0)-gco.min(0)).round(3)))
hv = np.zeros(len(co_hi), bool); hv[nf[hair_hi].ravel()] = True
print('hair verts: mean z=%.3f (all=%.3f)  mean y=%.3f (all=%.3f)  [hair=higher z, +y back]'
      % (co_hi[hv, 2].mean(), co_hi[:, 2].mean(), co_hi[hv, 1].mean(), co_hi[:, 1].mean()))

np.savez(os.path.join(D, '_hiresmesh.npz'),
         co=co_hi.astype(np.float32), fv=nf.astype(np.int64), hair=hair_hi.astype(bool), nv=len(co_hi))
np.save(os.path.join(D, '_hires_cen.npy'), co_hi[nf].mean(1).astype(np.float32))
print('saved _hiresmesh.npz + _hires_cen.npy')
