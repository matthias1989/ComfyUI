"""LOCAL: replicate gen_seams ROUGH metric + zone, and compare the FIXED threshold (0.009)
vs ADAPTIVE (percentile) — does adaptive cap the over-flag on a noisy mesh while leaving a
smooth mesh's ~strands intact? No baking."""
import os,sys,numpy as np,trimesh
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
m=trimesh.load(os.path.join(HERE,"last_seams.obj"),force='mesh')
f=np.asarray(m.faces); n_faces=len(f); fn=m.face_normals; cen=np.asarray(m.vertices)[f].mean(1)
adj=m.face_adjacency                                   # (E,2) adjacent face pairs
# _rough = per-face mean (1 - dot) over adjacent neighbours, + 3 light smoothing passes
rij=1.0-np.sum(fn[adj[:,0]]*fn[adj[:,1]],axis=1)
rough=np.zeros(n_faces); cnt=np.zeros(n_faces)
np.add.at(rough,adj[:,0],rij); np.add.at(rough,adj[:,1],rij)
np.add.at(cnt,adj[:,0],1.0); np.add.at(cnt,adj[:,1],1.0)
rough/=np.maximum(cnt,1)
for _ in range(3):
    acc=np.zeros(n_faces); c2=np.zeros(n_faces)
    np.add.at(acc,adj[:,0],rough[adj[:,1]]); np.add.at(acc,adj[:,1],rough[adj[:,0]])
    np.add.at(c2,adj[:,0],1.0); np.add.at(c2,adj[:,1],1.0)
    rough=0.5*rough+0.5*acc/np.maximum(c2,1)
zone=(np.abs(cen[:,0])<0.160)&(cen[:,2]>-0.10)
print(f"mesh: {n_faces} faces, zone={int(zone.sum())} ({100*zone.mean():.0f}%)")
print(f"rough stats in zone: median={np.median(rough[zone]):.4f} p85={np.percentile(rough[zone],85):.4f} p90={np.percentile(rough[zone],90):.4f} max={rough[zone].max():.4f}")
print(f"\n{'threshold':32s} {'is_rough&zone':>13s}  (proxy for the grow flood = over-flag)")
def report(name, thr):
    sel=(rough>thr)&zone
    print(f"{name:32s} {100*sel.sum()/n_faces:12.0f}%  thr={thr:.4f}")
report("FIXED 0.009 (current)", 0.009)
for p in (80,85,90,93):
    report(f"ADAPTIVE p{p} (zone)", float(np.percentile(rough[zone],p)))
report("ADAPTIVE max(0.009, p88)", max(0.009,float(np.percentile(rough[zone],88))))
