"""KAN-18: build the pelvic/genital donor patch (donor_pelvis.npz) from a donor character.
Source: Female Elf 5 Body_LP_Ref. Sphere/ellipsoid select around the genital (X-tight=no thighs,
Y-deep=reaches anus). Re-run to re-cut; tweak G / AX,AY,AZ. Renders to output/_pelvis_*.png."""
import bpy, bmesh, numpy as np
from mathutils import Vector
from collections import defaultdict
BLEND=r"C:\Blender Projects\Midgard\Characters\Female Elfs\Female Elf 5\Female_Elf5_HighPoly.blend"
OUT=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output"
ASSET=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-HandGraft\assets\donor_pelvis.npz"
bpy.ops.wm.open_mainfile(filepath=BLEND)
body=bpy.data.objects.get("Body_LP_Ref"); mw=body.matrix_world
bpy.context.view_layer.objects.active=body; bpy.ops.object.mode_set(mode='OBJECT')
for o in bpy.context.scene.objects: o.select_set(False)
body.select_set(True)
zc=-0.94
cand=[mw@v.co for v in body.data.vertices if abs((mw@v.co).x)<0.05 and abs((mw@v.co).z-zc)<0.04]
fronty=min(c.y for c in cand)
G=Vector((0.0, fronty+0.07, zc))     # shifted back so the patch spans vulva -> perineum -> anus
print("center G =",[round(x,3) for x in G], "fronty",round(fronty,3), flush=True)
AX,AY,AZ = 0.050, 0.140, 0.090       # X tight (no thighs), Y deeper (reaches anus), Z a touch taller
for poly in body.data.polygons:
    d=(mw@poly.center)-G; poly.select = (d.x/AX)**2+(d.y/AY)**2+(d.z/AZ)**2 < 1.0
print("faces selected =",sum(1 for p in body.data.polygons if p.select), flush=True)
pre=set(bpy.context.scene.objects)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
patch=[o for o in bpy.context.scene.objects if o not in pre][0]; patch.name="PelvisDonor"
for o in bpy.context.scene.objects: o.select_set(False)
bpy.context.view_layer.objects.active=patch; patch.select_set(True)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.mesh.select_mode(type="VERT"); bpy.ops.mesh.select_all(action='DESELECT'); bpy.ops.mesh.select_non_manifold(); bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=2)
bpy.ops.object.mode_set(mode='OBJECT')
bm=bmesh.new(); bm.from_mesh(patch.data)
bnd=[e for e in bm.edges if len(e.link_faces)==1]; adj=defaultdict(list)
for e in bnd: adj[e.verts[0].index].append(e.verts[1].index); adj[e.verts[1].index].append(e.verts[0].index)
seen=set(); loops=[]
for vi in list(adj):
    if vi in seen: continue
    c=0; st=[vi]
    while st:
        x=st.pop()
        if x in seen: continue
        seen.add(x); c+=1; st.extend(adj[x])
    loops.append(c)
bm.free(); loops.sort(reverse=True)
print(f"PelvisDonor: verts={len(patch.data.vertices)} faces={len(patch.data.polygons)} boundary_loops={len(loops)} sizes={loops[:4]}", flush=True)
pw=patch.matrix_world
V=np.array([list(pw@v.co) for v in patch.data.vertices], np.float32)
F=np.array([list(p.vertices) for p in patch.data.polygons], np.int64)
np.savez(ASSET, V=V, F=F, center=np.array(list(G),np.float32)); print("saved asset", flush=True)
for o in list(bpy.context.scene.objects):
    if o!=patch: bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT+r"\_pelvis_donor.blend")
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.show_cavity=True
sc.render.resolution_x=sc.render.resolution_y=820
co=[v.co for v in patch.data.vertices]; mn=Vector((min(c[i] for c in co) for i in range(3))); mx=Vector((max(c[i] for c in co) for i in range(3)))
ctr=(mn+mx)/2; H=max(mx-mn)
cd=bpy.data.cameras.new("C"); cam=bpy.data.objects.new("C",cd); sc.collection.objects.link(cam); sc.camera=cam; cd.type='ORTHO'
def shot(d,p):
    cd.ortho_scale=H*1.3; cam.location=ctr+Vector(d).normalized()*max(H,0.2)*2
    cam.rotation_euler=(ctr-cam.location).to_track_quat('-Z','Y').to_euler(); sc.render.filepath=p; bpy.ops.render.render(write_still=True)
shot((0,0,-1), OUT+r"\_pelvis_bottom.png"); shot((0,-0.5,-1), OUT+r"\_pelvis_belowfront.png")
print("rendered", flush=True)
