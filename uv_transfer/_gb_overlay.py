import numpy as np, os, importlib.util as ilu
from scipy.spatial import cKDTree
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
spec=ilu.spec_from_file_location('_greenborder', os.path.join(UV,'_greenborder.py')); gb=ilu.module_from_spec(spec); spec.loader.exec_module(gb)
d=np.load(os.path.join(UV,'_gs_hires.npz')); hv=d['co'].astype(np.float64); hf=d['fv'].astype(np.int64)
co_t=gb._gen_normalize(np.column_stack([hv[:,0],-hv[:,2],hv[:,1]]))
gv=np.load(os.path.join(UV,'_genseams_viz.npz')); gco=gv['co'].astype(np.float64); gt=gv['tris'].astype(np.int64); gh=gv['tris_hair'].astype(bool)
_,gi=cKDTree(gco[gt].mean(1)).query(co_t[hf].mean(1),workers=-1); rough=gh[gi]
rp=os.path.join(UV,'_gshires_region.npy')
if os.path.exists(rp): region=np.load(rp)
else: region=gb.compute(hv,hf,rough); np.save(rp,region)
hef=defaultdict(list)
for fi in range(len(hf)):
    a,b,c=int(hf[fi,0]),int(hf[fi,1]),int(hf[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): hef[(min(u,w),max(u,w))].append(fi)
fb=[(u,w) for (u,w),fs in hef.items() if len(fs)==2 and region[fs[0]]!=region[fs[1]]]
green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sel=co_t[:,2]>0.25
fig,axs=plt.subplots(1,2,figsize=(14,8))
for ax,a in zip(axs,[0,1]):
    ax.scatter(co_t[sel,a],co_t[sel,2],s=0.3,c='lightgray',lw=0)
    ax.scatter(green[:,a],green[:,2],s=5,c='lime')
    segs=[[(co_t[u,a],co_t[u,2]),(co_t[w,a],co_t[w,2])] for u,w in fb if co_t[u,2]>0.25 or co_t[w,2]>0.25]
    ax.add_collection(LineCollection(segs,colors='red',linewidths=1.1))
    ax.set_aspect('equal'); ax.set_title(['FRONT (x,z)','SIDE (y,z)'][a]+'  green=B_diag target  red=pipeline border')
plt.tight_layout(); plt.savefig(os.path.join(UV,'_gb_overlay.png'),dpi=95); print('saved; fb edges=%d'%len(fb))
