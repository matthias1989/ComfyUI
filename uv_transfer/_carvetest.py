import numpy as np, os
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as cc
from scipy.spatial import cKDTree
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
gv=np.load(os.path.join(UV,'_genseams_viz.npz')); co=gv['co'].astype(np.float64); gt=gv['tris'].astype(np.int64); gh=gv['tris_hair'].astype(bool); nf=len(gt)
green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']; gtr=cKDTree(green)
fc=co[gt].mean(1); yc=np.median(co[co[:,2]>0.40,1])
fn=np.cross(co[gt[:,1]]-co[gt[:,0]], co[gt[:,2]]-co[gt[:,0]]); fn/=(np.linalg.norm(fn,axis=1,keepdims=True)+1e-9)
ctr=co.mean(0); flip=((fc-ctr)*fn).sum(1)<0; fn[flip]=-fn[flip]
e2=defaultdict(list)
for fi in range(nf):
    a,b,c=int(gt[fi,0]),int(gt[fi,1]),int(gt[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): e2[(min(u,w),max(u,w))].append(fi)
pairs=[(fs[0],fs[1]) for fs in e2.values() if len(fs)==2]
def comps(mask):
    rr=[p[0] for p in pairs if mask[p[0]] and mask[p[1]]]; cc2=[p[1] for p in pairs if mask[p[0]] and mask[p[1]]]
    A=sp.csr_matrix((np.ones(len(rr)),(rr,cc2)),shape=(nf,nf)); A=A+A.T; return cc(A,directed=False)
def clean(ish):
    n,lab=comps(ish); sz=np.bincount(lab[ish],minlength=n); ish=np.isin(lab,np.where(sz>=max(30,int(0.01*ish.sum())))[0])&ish
    nb,labb=comps(~ish); szb=np.bincount(labb[~ish],minlength=nb); body=int(szb.argmax())
    return ish|((~ish)&(labb!=body)&np.isin(labb,np.where(szb<max(50,int(0.02*nf)))[0]))
base=clean(gh)
def metrics(ish,tag):
    seam=[(u,w) for (u,w),fs in e2.items() if len(fs)==2 and ish[fs[0]]!=ish[fs[1]]]
    mids=np.array([(co[u]+co[w])*0.5 for u,w in seam]); h=mids[:,2]>0.30
    d,_=gtr.query(mids[h]); far=(d>0.030).sum()
    print('%-12s far-from-green(>0.03)=%4d/%4d  mean=%.4f  hair=%d'%(tag,far,h.sum(),d.mean(),ish.sum()))
    return seam
metrics(base,'base')
best=None
for T in (-0.20,-0.35,-0.50):
    carve=base&(fn[:,1]<T)&(fc[:,2]>0.36)&(fc[:,1]<yc)
    ish=clean(base&~carve); metrics(ish,'carve%.2f'%T)
    if T==-0.35: best=ish
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
def panel(ax,ish,t):
    front=(co[:,1]<yc)&(co[:,2]>0.33); ax.scatter(co[front,0],co[front,2],s=0.4,c='lightgray',lw=0)
    hf=ish&(fc[:,1]<yc)&(fc[:,2]>0.33); ax.scatter(fc[hf,0],fc[hf,2],s=5,c='orange',lw=0)
    gm=green[:,1]<yc; ax.scatter(green[gm,0],green[gm,2],s=5,c='lime')
    seam=[(u,w) for (u,w),fs in e2.items() if len(fs)==2 and ish[fs[0]]!=ish[fs[1]]]
    segs=[[(co[u,0],co[u,2]),(co[w,0],co[w,2])] for u,w in seam if co[u,1]<yc and co[w,1]<yc and (co[u,2]>0.33 or co[w,2]>0.33)]
    ax.add_collection(LineCollection(segs,colors='red',linewidths=1.4)); ax.set_aspect('equal'); ax.set_title(t)
fig,axs=plt.subplots(1,2,figsize=(15,8)); panel(axs[0],base,'base (forehead blob)'); panel(axs[1],best,'carve -0.35')
plt.tight_layout(); plt.savefig(os.path.join(UV,'_carve.png'),dpi=95); print('saved')
