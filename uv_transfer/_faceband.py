import numpy as np, os
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
m=np.load(os.path.join(UV,os.environ.get('FB_MESH','_gb_mesh.npz'))); co=m['co'].astype(np.float64); fv=m['fv'].astype(np.int64); hair=m['hair'].astype(bool); nf=len(fv)
zn=(co[:,2]-co[:,2].min())/(co[:,2].max()-co[:,2].min()+1e-9)
lz=np.load(os.path.join(UV,os.environ.get('FB_LOOP','_gb_loop.npz'))); loop=lz['loop']; floor=float(lz['neck_floor']) if 'neck_floor' in lz.files else 0.0
e2f=defaultdict(list)
for fi in range(nf):
    a,b,c=int(fv[fi,0]),int(fv[fi,1]),int(fv[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): e2f[(min(u,w),max(u,w))].append(fi)
loopset=set((min(int(u),int(w)),max(int(u),int(w))) for u,w in loop)
wall=set()
for e,fs in e2f.items():
    if e in loopset:
        for f in fs: wall.add(f)
neckbar=set()
for (a,b),fs in e2f.items():
    if len(fs)==2 and ((zn[a]<floor)!=(zn[b]<floor)) and not (hair[fs[0]] and hair[fs[1]]): neckbar.add((a,b))
fcz=zn[fv].mean(1)
_hr=[];_hc=[]
for fs in e2f.values():
    if len(fs)==2 and hair[fs[0]] and hair[fs[1]]: _hr.append(fs[0]);_hc.append(fs[1])
Ah=sp.csr_matrix((np.ones(len(_hr)),(_hr,_hc)),shape=(nf,nf)); Ah=Ah+Ah.T
nch,lch=connected_components(Ah,directed=False)
crownc=int(np.bincount(lch[hair],minlength=nch).argmax()); cf=np.where(hair&(lch==crownc))[0]; seed=int(cf[np.argmax(fcz[cf])])
adj=defaultdict(list)
for e,fs in e2f.items():
    if len(fs)!=2 or e in neckbar or fs[0] in wall or fs[1] in wall: continue
    adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])
seen={seed}; st=[seed]
while st:
    f=st.pop()
    for g in adj[f]:
        if g not in seen: seen.add(g); st.append(g)
bottom=int(np.argmin(fcz))
print('CONTAINED? body-bottom reached = %s   inside=%d  wall=%d'%(bottom in seen, len(seen), len(wall)))
ishair=np.zeros(nf,bool); ishair[list(seen)]=True; ishair[list(wall)]=True
ishair[np.where(hair&(lch!=crownc))[0]]=True
green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']; gtr=cKDTree(green)
fb=[(u,w) for (u,w),fs in e2f.items() if len(fs)==2 and ishair[fs[0]]!=ishair[fs[1]]]
mids=np.array([(co[u]+co[w])*0.5 for u,w in fb]); mids=mids[mids[:,2]>0.30]
d,_=gtr.query(mids)
print('>>> FACE-BAND region boundary vs green: mean=%.4f med=%.4f p90=%.4f (fill=%d)  [_hairfill was 0.032, loop 0.021]'%(d.mean(),np.median(d),np.percentile(d,90),int(ishair.sum())))
