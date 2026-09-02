"""LOCAL: can a distance-weld stitch the ~17 fragmented body chunks into ONE coherent
surface WITHOUT collapsing the shape? Quantize vertices to a grid (= weld within tol),
remap faces, drop degenerates, measure component collapse + shape preservation. No baking."""
import numpy as np, trimesh, os
from scipy.spatial import cKDTree
p = os.path.join(os.path.dirname(__file__), "last_seams.obj")
m0 = trimesh.load(p, force='mesh')
V0 = np.asarray(m0.vertices); F0 = np.asarray(m0.faces)
ext0 = m0.extents
diag = float(np.linalg.norm(ext0))

def comp_stats(V, F):
    mm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # drop degenerate faces (collapsed by weld)
    good = (F[:,0]!=F[:,1]) & (F[:,1]!=F[:,2]) & (F[:,0]!=F[:,2])
    mm = trimesh.Trimesh(vertices=V, faces=F[good], process=False)
    parts = mm.split(only_watertight=False)
    sizes = sorted((len(pp.faces) for pp in parts), reverse=True)
    big = len([s for s in sizes if s > 500])
    n = max(len(mm.faces),1)
    return len(parts), big, 100*sizes[0]/n if sizes else 0, np.round(mm.extents,3), int(good.sum())

print(f"baseline: extents={np.round(ext0,3).tolist()} diag={diag:.3f}")
c,b,lp,e,nf = comp_stats(V0, F0)
print(f"{'tol(abs)':>10} {'comps':>6} {'big>500':>7} {'largest%':>9} {'faces':>7}  extents")
print(f"{'0':>10} {c:6d} {b:7d} {lp:8.1f}% {nf:7d}  {e.tolist()}")

def weld(V, F, tol):
    # union close vertices via KDTree pairs, then relabel
    tree = cKDTree(V)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    parent = np.arange(len(V))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for a,b in pairs:
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    roots=np.array([find(i) for i in range(len(V))])
    uniq,inv=np.unique(roots,return_inverse=True)
    # new vertex = mean of merged group
    Vn=np.zeros((len(uniq),3)); cnt=np.zeros(len(uniq))
    np.add.at(Vn,inv,V); np.add.at(cnt,inv,1.0); Vn/=cnt[:,None]
    Fn=inv[F]
    return Vn,Fn

for tol in (1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 4e-3):
    Vn,Fn = weld(V0,F0,tol*1.0)  # tol is fraction-ish; mesh diag ~1.45 so abs ~ tol
    c,b,lp,e,nf = comp_stats(Vn,Fn)
    print(f"{tol:10.4f} {c:6d} {b:7d} {lp:8.1f}% {nf:7d}  {e.tolist()}")
