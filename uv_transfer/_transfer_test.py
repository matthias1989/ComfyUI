import numpy as np, os, importlib.util as ilu
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as cc
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
PROJ=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
spec=ilu.spec_from_file_location('_greenborder', os.path.join(UV,'_greenborder.py')); gb=ilu.module_from_spec(spec); spec.loader.exec_module(gb)

fx=np.load(os.path.join(PROJ,'hair_detection_fixture.npz')); hv=fx['norm_verts'].astype(np.float64); hf=fx['norm_faces'].astype(np.int64)
co_t=gb._gen_normalize(np.column_stack([hv[:,0],-hv[:,2],hv[:,1]])); hcen=co_t[hf].mean(1)
gv=np.load(os.path.join(UV,'_genseams_viz.npz')); gco=gv['co'].astype(np.float64); gt=gv['tris'].astype(np.int64); gh=gv['tris_hair'].astype(bool)
_,gi=cKDTree(gco[gt].mean(1)).query(hcen,workers=-1); rough=gh[gi]

rp=os.path.join(UV,'_fixregion.npy')
if os.path.exists(rp): region=np.load(rp)
else: region=gb.compute(hv,hf,rough); np.save(rp,region)
print('fixture region: %d/%d hair faces'%(region.sum(),len(region)))

green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']; gtree=cKDTree(green)

# --- green-border-code faithfulness: fixture region boundary vs B_diag green ---
hef=defaultdict(list)
for fi in range(len(hf)):
    a,b,c=int(hf[fi,0]),int(hf[fi,1]),int(hf[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): hef[(min(u,w),max(u,w))].append(fi)
fb=[k for k,fs in hef.items() if len(fs)==2 and region[fs[0]]!=region[fs[1]]]
fbm=np.array([(co_t[u]+co_t[w])*0.5 for u,w in fb]); fbm=fbm[fbm[:,2]>0.30]
d0,_=gtree.query(fbm)
print('FIXTURE region boundary vs B_diag green: mean=%.4f med=%.4f p90=%.4f  (code faithfulness floor)'%(d0.mean(),np.median(d0),np.percentile(d0,90)))

# --- coarse adjacency ---
gcen=gco[gt].mean(1)
e2f=defaultdict(list)
for fi in range(len(gt)):
    a,b,c=int(gt[fi,0]),int(gt[fi,1]),int(gt[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): e2f[(min(u,w),max(u,w))].append(fi)
e2 =[(k,fs) for k,fs in e2f.items() if len(fs)==2]
rr=[fs[0] for _,fs in e2]; ccol=[fs[1] for _,fs in e2]
Af=sp.csr_matrix((np.ones(len(rr)),(rr,ccol)),shape=(len(gt),len(gt))); Af=Af+Af.T
deg=np.maximum(np.asarray(Af.sum(1)).ravel(),1.0)
def comps(mask):
    r=[fs[0] for _,fs in e2 if mask[fs[0]] and mask[fs[1]]]; c=[fs[1] for _,fs in e2 if mask[fs[0]] and mask[fs[1]]]
    A=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(len(gt),len(gt))); A=A+A.T; return cc(A,directed=False)
def transfer(k,passes):
    _,hi=cKDTree(hcen).query(gcen,k=k,workers=-1)
    ish=(region[hi].mean(1)>0.5) if k>1 else region[hi]
    n,lab=comps(ish); sz=np.bincount(lab[ish],minlength=n); ish=np.isin(lab,np.where(sz>=max(30,int(0.01*ish.sum())))[0])&ish
    nb,labb=comps(~ish); szb=np.bincount(labb[~ish],minlength=nb); body=szb.argmax()
    ish=ish|((~ish)&(labb!=body)&np.isin(labb,np.where(szb<max(50,int(0.02*len(gt))))[0]))
    m=ish.astype(float)
    for _ in range(passes):
        nh=np.asarray(Af@m).ravel()/deg; m=np.where(nh>0.6,1.0,np.where(nh<0.4,0.0,m))
    ish=m>0.5
    n2,lab2=comps(ish); sz2=np.bincount(lab2[ish],minlength=n2); return np.isin(lab2,np.where(sz2>=max(30,int(0.01*ish.sum())))[0])&ish
def measure(ish):
    seam=[(u,w) for (u,w),fs in e2 if ish[fs[0]]!=ish[fs[1]]]
    mids=np.array([(gco[u]+gco[w])*0.5 for u,w in seam]); head=mids[:,2]>0.30
    d,_=gtree.query(mids[head]); return int(ish.sum()),int(head.sum()),d.mean(),np.median(d),np.percentile(d,90)
print()
print('%-14s %8s %8s %8s %8s %8s'%('variant','hair','headseam','mean','median','p90'))
for k in (1,9):
    for p in (0,1,2,4):
        ish=transfer(k,p); n,h,me,md,p9=measure(ish)
        print('k=%d smooth=%d  %8d %8d %8.4f %8.4f %8.4f'%(k,p,n,h,me,md,p9))
