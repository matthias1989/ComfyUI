"""Calibrate the shell test: for KNOWN hair faces vs KNOWN body faces, cast a ray from the centroid
along -normal (inward) and measure the distance to the next surface. Hair (thin shell over body)
should hit close; solid body should hit far (its own far side). This tells us the thickness D that
separates them, and confirms normals point outward."""
import numpy as np
import pyvista as pv

m = np.load('_Amesh.npz', allow_pickle=True)
co = m['co']; fv = m['fv']
hair = m['hair'].astype(bool)
zmin, zmax = float(co[:, 2].min()), float(co[:, 2].max())

# build triangulated pyvista mesh + per-face centroid/normal (first-3-vert normal)
faces_flat = []
fn = np.zeros((len(fv), 3)); fc = np.zeros((len(fv), 3))
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]; faces_flat.append(len(vs)); faces_flat.extend(vs)
    pts = co[vs]; fc[fi] = pts.mean(0)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0]); ln = float(np.linalg.norm(n))
    fn[fi] = n / ln if ln > 0 else n
mesh = pv.PolyData(co, np.array(faces_flat)).triangulate()
fznv = (fc[:, 2] - zmin) / (zmax - zmin + 1e-9)
diag = float(np.linalg.norm(co.max(0) - co.min(0)))
eps = 0.0008 * diag

def behind_dist(idx):
    o = fc[idx] - eps * fn[idx]; d = -fn[idx]
    pts, _ = mesh.ray_trace(o, o + d * (0.2 * diag), first_point=True)
    return float(np.linalg.norm(pts - o)) if pts.size else 1e9

rng = np.random.default_rng(0)
for label, mask in [('HAIR (all)', hair), ('HAIR upper z>0.8', hair & (fznv > 0.8)),
                    ('BODY torso z0.3-0.6', (~hair) & (fznv > 0.3) & (fznv < 0.6) & (np.abs(fc[:, 0]) < 0.1)),
                    ('BODY neck z0.72-0.85', (~hair) & (fznv > 0.72) & (fznv < 0.85))]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(label, 'none'); continue
    samp = rng.choice(idx, size=min(400, len(idx)), replace=False)
    ds = np.array([behind_dist(int(i)) for i in samp])
    print('%-22s n=%4d  behind-dist p25 %.4f p50 %.4f p75 %.4f  (diag=%.3f)' % (
        label, len(samp), np.percentile(ds, 25), np.percentile(ds, 50), np.percentile(ds, 75), diag))
