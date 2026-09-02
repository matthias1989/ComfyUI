import bpy, numpy as np
from collections import defaultdict
o=bpy.data.objects.get('geometry_0'); me=o.data; nf=len(me.polygons)
co=np.array([(o.matrix_world @ v.co)[:] for v in me.vertices], dtype=np.float64)
ish=np.load(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams_hairfaces.npy')
e2f=defaultdict(list)
for p in me.polygons:
    vs=[int(v) for v in p.vertices]
    for a in range(len(vs)):
        k=(min(vs[a],vs[(a+1)%len(vs)]),max(vs[a],vs[(a+1)%len(vs)])); e2f[k].append(p.index)
se=[]; hairline=[]
for e in me.edges:
    if not e.use_seam: continue
    k=(min(e.vertices[0],e.vertices[1]),max(e.vertices[0],e.vertices[1])); fs=e2f.get(k,[])
    se.append((e.vertices[0],e.vertices[1])); hairline.append(len(fs)==2 and len(ish)==nf and (bool(ish[fs[0]])!=bool(ish[fs[1]])))
np.savez(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_ls_extract.npz', co=co, seam_edges=np.array(se), hairline=np.array(hairline))
print('  seam_edges=%d  hairline=%d'%(len(se),int(np.sum(hairline))))
