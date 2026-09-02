import numpy as np, os
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as cc
from scipy.spatial import cKDTree
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
gv=np.load(os.path.join(UV,'_genseams_viz.npz')); gco=gv['co'].astype(np.float64); gt=gv['tris'].astype(np.int64); gh=gv['tris_hair'].astype(bool); nf=len(gt)
green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']; gtr=cKDTree(green)
e2=defaultdict(list)
for fi in range(nf):
    a,b,c=int(gt[fi,0]),int(gt[fi,1]),int(gt[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): e2[(min(u,w),max(u,w))].append(fi)
pairs=[(fs[0],fs[1]) for fs in e2.values() if len(fs)==2]
r=[p[0] for p in pairs]; c=[p[1] for p in pairs]
Af=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(nf,nf)); Af=Af+Af.T; deg=np.maximum(np.asarray(Af.sum(1)).ravel(),1.0)
def comps(mask):
    rr=[p[0] for p in pairs if mask[p[0]] and mask[p[1]]]; cc2=[p[1] for p in pairs if mask[p[0]] and mask[p[1]]]
    A=sp.csr_matrix((np.ones(len(rr)),(rr,cc2)),shape=(nf,nf)); A=A+A.T; return cc(A,directed=False)
def clean(ish):
    n,lab=comps(ish); sz=np.bincount(lab[ish],minlength=n); ish=np.isin(lab,np.where(sz>=max(30,int(0.01*ish.sum())))[0])&ish
    nb,labb=comps(~ish); szb=np.bincount(labb[~ish],minlength=nb); body=int(szb.argmax())
    return ish|((~ish)&(labb!=body)&np.isin(labb,np.where(szb<max(50,int(0.02*nf)))[0]))
def smooth(ish,n):
    m=ish.astype(float)
    for _ in range(n):
        nh=np.asarray(Af@m).ravel()/deg; m=np.where(nh>0.6,1.0,np.where(nh<0.4,0.0,m))
    return m>0.5
yc=np.median(gco[gco[:,2]>0.40,1])
def stats(ish,tag):
    seam=[(u,w) for (u,w),fs in e2.items() if len(fs)==2 and ish[fs[0]]!=ish[fs[1]]]
    mids=np.array([(gco[u]+gco[w])*0.5 for u,w in seam]);
    f=(mids[:,2]>0.40)&(mids[:,1]<yc); h=mids[:,2]>0.30
    df,_=gtr.query(mids[f]); dh,_=gtr.query(mids[h])
    print('%-14s front mean=%.4f med=%.4f (n=%d) | whole mean=%.4f (edges=%d hair=%d)'%(tag,df.mean(),np.median(df),f.sum(),dh.mean(),len(seam),ish.sum()))
    return seam
base=clean(gh)
stats(gh,'raw tris_hair'); stats(base,'cleaned')
for N in (6,12,20): stats(smooth(base,N),'cleaned+sm%d'%N)
# render front: raw vs cleaned+sm12
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
def panel(ax,ish,t):
    front=(gco[:,1]<yc)&(gco[:,2]>0.30); ax.scatter(gco[front,0],gco[front,2],s=0.4,c='lightgray',lw=0)
    gm=green[:,1]<yc; ax.scatter(green[gm,0],green[gm,2],s=6,c='lime')
    seam=[(u,w) for (u,w),fs in e2.items() if len(fs)==2 and ish[fs[0]]!=ish[fs[1]]]
    segs=[[(gco[u,0],gco[u,2]),(gco[w,0],gco[w,2])] for u,w in seam if gco[u,1]<yc and gco[w,1]<yc and (gco[u,2]>0.30 or gco[w,2]>0.30)]
    ax.add_collection(LineCollection(segs,colors='red',linewidths=1.5)); ax.set_aspect('equal'); ax.set_title(t)
fig,axs=plt.subplots(1,2,figsize=(15,8)); panel(axs[0],base,'cleaned (raw rough)'); panel(axs[1],smooth(base,12),'cleaned + smooth12')
plt.tight_layout(); plt.savefig(os.path.join(UV,'_cleanrough.png'),dpi=95); print('saved')
