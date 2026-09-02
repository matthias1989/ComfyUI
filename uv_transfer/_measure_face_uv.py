"""Measure the FACE region's UV area fraction in an OBJ (geometric region, order-independent).
Face = top-of-head band, central, front-facing. Reports UV% vs 3D% (collapsed if UV<<3D)."""
import sys, numpy as np, trimesh
P = sys.argv[1]
m = trimesh.load(P, force='mesh')
f = np.asarray(m.faces); uv = np.asarray(m.visual.uv); cen = np.asarray(m.vertices)[f].mean(1); fn = m.face_normals
H = cen[:,1]; X = cen[:,0]; Z = cen[:,2]
xc = 0.5*(X.min()+X.max())
ytop = H.min() + 0.74*(H.max()-H.min())          # top ~26% height = head
# front sign: head-central faces mostly point one way in Z
hc = (H > ytop) & (np.abs(X-xc) < 0.10*(X.max()-X.min()))
s = 1.0 if fn[hc,2].mean() > 0 else -1.0
face = (H > ytop) & (np.abs(X-xc) < 0.12*(X.max()-X.min())) & (s*fn[:,2] > 0.2)
t = uv[f]; uva = 0.5*np.abs((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,2,0]-t[:,0,0])*(t[:,1,1]-t[:,0,1]))
tot = uva.sum(); m3 = m.area_faces
print(f"{P.split(chr(92))[-1]}: faces={len(f)}  FACE-region={int(face.sum())} faces")
print(f"   FACE UV area = {100*uva[face].sum()/tot:.4f}% of used UV   |   FACE 3D area = {100*m3[face].sum()/m3.sum():.2f}% of body")
print(f"   -> UV/3D ratio = {(uva[face].sum()/tot)/(m3[face].sum()/m3.sum()+1e-12):.3f}  (1.0=proportional, <<1=collapsed)")
