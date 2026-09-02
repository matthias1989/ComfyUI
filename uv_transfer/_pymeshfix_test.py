"""Test pymeshfix as a gentle manifold pre-clean: does it remove the doubled/interior
geometry (the shell over-flag cause) while preserving the outer shape?"""
import os,sys,numpy as np
import trimesh,pymeshfix
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
V=np.asarray(m.vertices); F=np.asarray(m.faces)
print(f"BEFORE: {len(V)} verts {len(F)} faces  watertight={m.is_watertight}  components={len(m.split(only_watertight=False))}  bbox_size={np.round(m.extents,3)}")
# shell/doubled-geometry metric: per face, cast a ray INWARD; count faces that hit another
# surface within 0.041 (== the gen_seams shell test that over-flags on doubled geometry)
def doubled_frac(mesh, n=4000, d=0.041):
    f=np.asarray(mesh.faces); v=np.asarray(mesh.vertices); fn=mesh.face_normals
    cen=v[f].mean(1); idx=np.random.RandomState(0).choice(len(f),min(n,len(f)),replace=False)
    org=cen[idx]-0.0015*fn[idx]; dirs=-fn[idx]
    loc,ray_idx,tri=mesh.ray.intersects_location(org,dirs,multiple_hits=False)
    hit=np.zeros(len(idx),bool)
    if len(ray_idx):
        dist=np.linalg.norm(loc-org[ray_idx],axis=1)
        ok=(dist>0.0021)&(dist<d)
        hit[ray_idx[ok]]=True
    return 100.0*hit.mean()
print(f"  doubled-geometry (inward-ray hit <0.041) BEFORE = {doubled_frac(m):.0f}% of sampled faces")
vc,fc=pymeshfix.clean_from_arrays(V,F)
mc=trimesh.Trimesh(vc,fc,process=False)
print(f"AFTER : {len(vc)} verts {len(fc)} faces  watertight={mc.is_watertight}  components={len(mc.split(only_watertight=False))}  bbox_size={np.round(mc.extents,3)}")
print(f"  doubled-geometry AFTER = {doubled_frac(mc):.0f}%   (want much lower)")
print(f"  bbox match: before={np.round(m.bounds,3).tolist()}  after={np.round(mc.bounds,3).tolist()}")
mc.export(os.path.join(HERE,"_pymeshfix_out.obj"))
print("saved _pymeshfix_out.obj")
