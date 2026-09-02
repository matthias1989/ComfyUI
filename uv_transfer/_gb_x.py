import bpy, numpy as np
out={}
for name in ['HAIR_SEAM','HAIR_FILL','RAW_CREASE','geometry_0']:
    o=bpy.data.objects.get(name)
    if o is None: continue
    me=o.data
    co=np.array([(o.matrix_world @ v.co)[:] for v in me.vertices], dtype=np.float64)
    out[name+'_co']=co
    col=None
    if me.materials and me.materials[0]:
        try: col=np.array(me.materials[0].diffuse_color[:])
        except: col=None
    print('  %-12s verts=%6d  matcolor=%s  zrange=[%.3f,%.3f]'%(name,len(co),(np.round(col,2) if col is not None else None),co[:,2].min(),co[:,2].max()))
np.savez(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_gb_extract.npz', **out)
