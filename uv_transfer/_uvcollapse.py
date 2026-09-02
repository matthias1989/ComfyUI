"""LOCAL: per-island UV texel-density (UV_area / 3D_area). Collapsed islands (density << median)
get ~no atlas space -> shatter. Identify the face island + the big body islands and flag collapses."""
import os, sys, numpy as np, trimesh, trimesh.graph
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(__file__)
m = trimesh.load(os.path.join(HERE, "last_seams.glb"), force='mesh')
V = np.asarray(m.vertices); F = np.asarray(m.faces); uv = np.asarray(m.visual.uv)
# normalize 3D to bbox for region thresholds
c = (V.min(0)+V.max(0))*0.5; ext = (V.max(0)-V.min(0)).max(); P = (V-c)/ext  # ~[-0.5,0.5]
tri = V[F]; e1=tri[:,1]-tri[:,0]; e2=tri[:,2]-tri[:,0]; a3 = 0.5*np.linalg.norm(np.cross(e1,e2),axis=1)
t = uv[F]; auv = 0.5*np.abs((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,2,0]-t[:,0,0])*(t[:,1,1]-t[:,0,1]))
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)))
isl = np.zeros(len(F), np.int32)
for i,c2 in enumerate(cc): isl[c2]=i
nI=len(cc)
A3=np.zeros(nI); AUV=np.zeros(nI); SZ=np.zeros(nI,int)
np.add.at(A3,isl,a3); np.add.at(AUV,isl,auv); np.add.at(SZ,isl,1)
dens = AUV/np.maximum(A3,1e-12)
big = np.argsort(-A3)[:14]                     # biggest islands by 3D area
med = np.median(dens[A3>np.percentile(A3,80)]) # density of large islands
print(f"islands={nI}  median-density(large islands)={med:.3f}")
cen = tri.mean(1); Pc=(cen-c)/ext
# face island: upper-front-center faces
face_sel = (Pc[:,1]>0.28)&(Pc[:,1]<0.46)&(Pc[:,2]>0.05)&(np.abs(Pc[:,0])<0.07)
fi_isl = np.bincount(isl[face_sel], minlength=nI).argmax() if face_sel.any() else -1
print("\n  rank  island   faces     3Darea    UVarea   density   density/med   region")
def region(i):
    sel=isl==i; p=Pc[sel].mean(0)
    where = "FACE" if i==fi_isl else ("head/upper" if p[1]>0.25 else "torso" if p[1]>-0.05 else "lower")
    return f"y={p[1]:+.2f} z={p[2]:+.2f} x={p[0]:+.2f} {where}"
for r,i in enumerate(big):
    flag = "  <-- COLLAPSED" if dens[i] < 0.30*med else ""
    print(f"  {r:>4} {i:>7} {SZ[i]:>7} {A3[i]:>10.4f} {AUV[i]:>9.4f} {dens[i]:>9.3f} {dens[i]/med:>10.2f}   {region(i)}{flag}")
if fi_isl>=0:
    print(f"\nFACE island = {fi_isl}: faces={SZ[fi_isl]} density={dens[fi_isl]:.3f} (={dens[fi_isl]/med:.2f}x median)"
          + ("  <-- COLLAPSED" if dens[fi_isl]<0.30*med else "  ok"))
