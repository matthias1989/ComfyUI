"""LOCAL harness: replicate gen_seams' SHELL hair over-flag, then sweep many mesh-clean
strategies and measure (over-flag %, components, shape preserved, faces). Goal: find a
clean that kills the doubled-geometry over-flag WITHOUT destroying the body. No baking."""
import os,sys,numpy as np,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
SRC=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')

def shell_overflag(mesh, sample=9000):
    """Replicate gen_seams shell test: cand = z>-0.15 & |x|<0.2; ray inward; hit another
    surface within 0.041 (dist>0.0021) with aligned normal (dot>0.30) -> hair. Return %."""
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces)
    if len(f)==0: return 0.0,0
    fn=mesh.face_normals; cen=v[f].mean(1)
    cand=np.where((cen[:,2]>-0.15)&(np.abs(cen[:,0])<0.20))[0]
    if len(cand)==0: return 0.0,0
    if len(cand)>sample: cand=np.random.RandomState(0).choice(cand,sample,replace=False)
    org=cen[cand]-0.0015*fn[cand]; dirs=-fn[cand]
    loc,ri,tri=mesh.ray.intersects_location(org,dirs,multiple_hits=False)
    hit=np.zeros(len(cand),bool)
    if len(ri):
        dist=np.linalg.norm(loc-org[ri],axis=1)
        algn=np.einsum('ij,ij->i',fn[cand[ri]],fn[tri])
        ok=(dist>0.0021)&(dist<0.041)&(algn>0.30)&(tri!=cand[ri])
        hit[ri[ok]]=True
    return 100.0*hit.mean(), len(cand)

def stats(mesh):
    of,_=shell_overflag(mesh)
    comps=len(mesh.split(only_watertight=False)) if len(mesh.faces) else 0
    ext=np.round(mesh.extents,3) if len(mesh.faces) else [0,0,0]
    return of,comps,len(mesh.faces),ext

of0,c0,n0,e0=stats(SRC)
print(f"{'STRATEGY':38s} {'overflag%':>9s} {'comps':>6s} {'faces':>7s}  extents (shape)")
print(f"{'0 baseline (no clean)':38s} {of0:9.0f} {c0:6d} {n0:7d}  {e0.tolist()}")

def comp_filter(mesh, min_faces):
    parts=mesh.split(only_watertight=False)
    keep=[p for p in parts if len(p.faces)>=min_faces]
    return trimesh.util.concatenate(keep) if keep else mesh

def interior_comp_removal(mesh):
    """Drop components whose centroid is INSIDE the union of the others (inner shells)."""
    parts=sorted(mesh.split(only_watertight=False), key=lambda p:-len(p.faces))
    keep=[parts[0]]
    big=parts[0]
    for p in parts[1:]:
        c=p.centroid
        inside=False
        try: inside=big.contains([c])[0]
        except Exception: inside=False
        if not inside: keep.append(p);
    return trimesh.util.concatenate(keep)

def weld(mesh, thr):
    m=mesh.copy(); m.merge_vertices(digits_vertex=thr); return m

def try_strategy(name, fn):
    try:
        m=fn(SRC.copy())
        of,c,n,e=stats(m)
        # shape preserved if extents within 15% of baseline
        ok="OK " if np.all(np.abs(np.array(e)-np.array(e0))<0.15*np.maximum(np.array(e0),1e-6)) else "SHAPE!"
        print(f"{name:38s} {of:9.0f} {c:6d} {n:7d}  {e.tolist()} {ok}")
        return m,of,e
    except Exception as ex:
        print(f"{name:38s}  ERROR {ex!r}")
        return None,999,None

try_strategy("1 remove comps <50 faces", lambda m: comp_filter(m,50))
try_strategy("2 remove comps <200 faces", lambda m: comp_filter(m,200))
try_strategy("3 remove comps <1000 faces", lambda m: comp_filter(m,1000))
try_strategy("4 interior-component removal", interior_comp_removal)
try_strategy("5 merge_vertices default", lambda m:(m.merge_vertices() or m))
try_strategy("6 remove duplicate faces", lambda m:(m.update_faces(m.unique_faces()) or m))
try_strategy("7 remove degenerate faces", lambda m:(m.update_faces(m.nondegenerate_faces()) or m))
try_strategy("8 fix_normals", lambda m:(m.fix_normals() or m))
try_strategy("9 comp<200 + interior removal", lambda m: interior_comp_removal(comp_filter(m,200)))
try_strategy("10 comp<200 + dup + degen", lambda m:(_c:=comp_filter(m,200), _c.update_faces(_c.unique_faces()), _c.update_faces(_c.nondegenerate_faces()), _c)[-1])
