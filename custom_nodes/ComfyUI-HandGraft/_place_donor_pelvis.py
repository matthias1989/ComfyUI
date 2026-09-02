"""KAN-18: auto-placement of the pelvic donor patch onto a target body.
Reads _pelvis_place_fixed.npz (L=fixed orient+proportions, A_ref, ref_width, Gc) fitted from the
user's hand-aligned _pelvis_placement_test1.blend. Per character: find leg-junction + pelvis width
geometrically -> derive anchor + size ratio -> world = (Vd-Gc)@(L.T*ratio)+anchor. Validated:
reproduces the user placement (anchor identical, ratio 0.99). NEXT: drop-overlap + voxel fuse."""
import numpy as np, bpy
OUT=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output"
ASSET=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-HandGraft\assets\donor_pelvis.npz"
f=np.load(OUT+r"\_pelvis_place_fixed.npz"); L=f["L"]; A_ref=f["A"]; ref_w=float(f["ref_width"]); Gc=f["Gc"]
dd=np.load(ASSET); Vd=dd["V"].astype(float); Fd=dd["F"].astype(int)
def leg_junction_Y(V):
    ymin,ymax=V[:,1].min(),V[:,1].max(); step=(ymax-ymin)/60
    for i in range(60):
        y=ymin+step*i; band=V[(V[:,1]>=y)&(V[:,1]<y+step)]
        if len(band)==0: continue
        if (np.abs(band[:,0])<0.04).sum()>20: return y
    return (ymin+ymax)/2
def pelvis_width(V,lj):
    band=V[(V[:,1]>lj-0.05)&(V[:,1]<lj+0.06)]
    return band[:,0].max()-band[:,0].min()
Vtref=np.load(OUT+r"\handgraft_out.npz")["V"].astype(float)
ref_lj=leg_junction_Y(Vtref); ref_h=Vtref[:,1].max()-Vtref[:,1].min()
dY=A_ref[1]-ref_lj
def auto_place(V):
    lj=leg_junction_Y(V); h=V[:,1].max()-V[:,1].min()
    ay=lj+dY*(h/ref_h)
    near=V[(np.abs(V[:,0])<0.04)&(np.abs(V[:,1]-ay)<0.03)]
    az=near[:,2].min()
    seat=A_ref[2]-(Vtref[(np.abs(Vtref[:,0])<0.04)&(np.abs(Vtref[:,1]-A_ref[1])<0.03)][:,2].min())  # ref seat offset
    A=np.array([0.0, ay, az+seat])
    ratio=pelvis_width(V,lj)/ref_w
    world=(Vd-Gc)@(L.T*ratio)+A
    return world,A,ratio
W,A,ratio=auto_place(Vtref)
print(f"AUTO anchor = {np.round(A,4)}   (user A_ref = {np.round(A_ref,4)})")
print(f"AUTO ratio  = {ratio:.4f}   (should be ~1.0 on the reference body)")
print(f"AUTO patch world bbox X[{W[:,0].min():.3f},{W[:,0].max():.3f}] Y[{W[:,1].min():.3f},{W[:,1].max():.3f}] Z[{W[:,2].min():.3f},{W[:,2].max():.3f}]")
# save blend: body + auto-placed patch
Ft=np.load(OUT+r"\handgraft_out.npz")["F"].astype(int)
for o in list(bpy.context.scene.objects): bpy.data.objects.remove(o,do_unlink=True)
def mk(n,V,F,c):
    me=bpy.data.meshes.new(n); me.from_pydata(V.tolist(),[],[list(map(int,x)) for x in F]); me.update()
    o=bpy.data.objects.new(n,me); bpy.context.scene.collection.objects.link(o); o.color=c
    m=bpy.data.materials.new(n); m.diffuse_color=c; me.materials.append(m); return o
mk("target_body",Vtref,Ft,(0.55,0.55,0.58,1)); mk("pelvis_auto",W,Fd,(0.3,0.8,0.4,1))
bpy.context.scene.display.shading.color_type='OBJECT'
bpy.ops.wm.save_as_mainfile(filepath=OUT+r"\_pelvis_autoplace_test.blend")
print("saved _pelvis_autoplace_test.blend")
