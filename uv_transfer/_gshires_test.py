import numpy as np, os, importlib.util as ilu
from scipy.spatial import cKDTree
from collections import defaultdict
UV=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
spec=ilu.spec_from_file_location('_greenborder', os.path.join(UV,'_greenborder.py')); gb=ilu.module_from_spec(spec); spec.loader.exec_module(gb)

d=np.load(os.path.join(UV,'_gs_hires.npz'))
print('gs_hires keys:', d.files)
hv=d['co'].astype(np.float64); hf=d[('fv' if 'fv' in d.files else 'faces')].astype(np.int64)
print('gs_hires: %d verts  %d faces'%(len(hv),len(hf)))
co_t=gb._gen_normalize(np.column_stack([hv[:,0],-hv[:,2],hv[:,1]]))
gv=np.load(os.path.join(UV,'_genseams_viz.npz')); gco=gv['co'].astype(np.float64); gt=gv['tris'].astype(np.int64); gh=gv['tris_hair'].astype(bool)
_,gi=cKDTree(gco[gt].mean(1)).query(co_t[hf].mean(1),workers=-1); rough=gh[gi]

region=gb.compute(hv,hf,rough)
print('gs_hires region: %d/%d hair faces (%.1f%%)'%(region.sum(),len(region),100*region.mean()))

green=np.load(os.path.join(UV,'_gb_extract.npz'))['HAIR_SEAM_co']; gtree=cKDTree(green)
hef=defaultdict(list)
for fi in range(len(hf)):
    a,b,c=int(hf[fi,0]),int(hf[fi,1]),int(hf[fi,2])
    for u,w in ((a,b),(b,c),(c,a)): hef[(min(u,w),max(u,w))].append(fi)
fb=[k for k,fs in hef.items() if len(fs)==2 and region[fs[0]]!=region[fs[1]]]
fbm=np.array([(co_t[u]+co_t[w])*0.5 for u,w in fb]); fbm=fbm[fbm[:,2]>0.30]
dd,_=gtree.query(fbm)
print('>>> gs_hires (CURRENT char) green border vs B_diag green: mean=%.4f med=%.4f p90=%.4f'%(dd.mean(),np.median(dd),np.percentile(dd,90)))
print('    (compare: FIXTURE green border vs B_diag = 0.0338)')
