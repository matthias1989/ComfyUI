"""LOCAL: replicate gen_seams ROUGH + GROW flood, compare FIXED vs ADAPTIVE rough threshold.
Confirms the grow flood (the actual over-flag) is contained by the adaptive threshold."""
import os,sys,numpy as np,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
f=np.asarray(m.faces); n=len(f); fn=m.face_normals; cen=np.asarray(m.vertices)[f].mean(1)
adj=m.face_adjacency
rij=1.0-np.sum(fn[adj[:,0]]*fn[adj[:,1]],axis=1)
rough=np.zeros(n); cnt=np.zeros(n)
np.add.at(rough,adj[:,0],rij); np.add.at(rough,adj[:,1],rij)
np.add.at(cnt,adj[:,0],1.0); np.add.at(cnt,adj[:,1],1.0); rough/=np.maximum(cnt,1)
for _ in range(3):
    acc=np.zeros(n); c2=np.zeros(n)
    np.add.at(acc,adj[:,0],rough[adj[:,1]]); np.add.at(acc,adj[:,1],rough[adj[:,0]])
    np.add.at(c2,adj[:,0],1.0); np.add.at(c2,adj[:,1],1.0); rough=0.5*rough+0.5*acc/np.maximum(c2,1)
# OBJ axes: Blender Z-up -> OBJ Y-up. height=Y=cen[:,1], lateral=X=cen[:,0], up-normal=fn[:,1]
hgt=cen[:,1]; lat=cen[:,0]; upn=fn[:,1]
zone=(np.abs(lat)<0.160)&(hgt>-0.10); up=(upn>0.25)&(hgt>0.36)
seed=(upn>0.25)&(hgt>0.40)&zone    # crown (high up-facing hair) approximating the shell seed
print(f"seed faces={int(seed.sum())}  zone faces={int(zone.sum())} ({100*zone.mean():.0f}%)")
P0,P1=adj[:,0],adj[:,1]
def grow(thr):
    is_rough=rough>thr
    growable=(is_rough|up)
    h=seed.copy()
    for _ in range(200):
        hn=np.zeros(n); np.add.at(hn,P0,h[P1].astype(float)); np.add.at(hn,P1,h[P0].astype(float))
        g=(~h)&growable&zone&(hn>=1.0)
        if not g.any(): break
        h|=g
    return h
med=float(np.median(rough[zone]))
print(f"median rough(zone)={med:.4f}  (clean mesh ~0.003; this is NOISY)")
print(f"{'rough threshold':34s} {'GROWN hair%':>11s}")
for name,thr in (("FIXED 0.009 (current)",0.009),
                 ("ADAPTIVE max(.009, 4*median)",max(0.009,4*med)),
                 ("ADAPTIVE max(.009, 5*median)",max(0.009,5*med)),
                 ("ADAPTIVE max(.009, 6*median)",max(0.009,6*med))):
    h=grow(thr); print(f"{name:34s} {100*h.sum()/n:10.0f}%  (thr={thr:.4f})")
