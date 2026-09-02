"""KAN-18: build the pelvic/genital donor patch v2 (donor_pelvis_v2.npz) from Female Elf 5 Body_LP_Ref.
Wider than v1 (inner-thigh margin) + a clean Taubin-smoothed boundary loop. v2.1 (2026-06-21):
extended the cut BACK+DOWN (AY/AZ + center) so it captures the anus->gluteal-cleft transition behind
the anus -- otherwise the perineum is a flat plateau and the cleft never forms (measured 4.4cm anus->
buttock gap). Re-run to re-cut; the graft recipe (pelvis_graft.py) is unchanged and flows it through.
Renders to output/_pelvis2_*.png. Source frame: X=width, Y=front<->back (vulva..anus..buttock), Z=vertical."""
import bpy, bmesh, numpy as np
from mathutils import Vector
from collections import defaultdict
BLEND = r"C:\Blender Projects\Midgard\Characters\Female Elfs\Female Elf 5\Female_Elf5_HighPoly.blend"
OUT = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output"
ASSET = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-HandGraft\assets\donor_pelvis_v2.npz"

# ---- cut parameters ----
# ORIGINAL v2 values that produced the user-approved _pg_test14. (A back-only extension to reach
# the gluteal cleft was tried -- G_YOFF=0.09/AY=0.185 -- but it distorted the placement ["pushed in"],
# so reverted. Revisit the anus->cleft gap LATER via a different approach, not by enlarging this cut.)
ZC = -0.94
G_YOFF = 0.07
AX, AY, AZ = 0.072, 0.165, 0.108

bpy.ops.wm.open_mainfile(filepath=BLEND)
body = bpy.data.objects.get("Body_LP_Ref"); mw = body.matrix_world
bpy.context.view_layer.objects.active = body; bpy.ops.object.mode_set(mode='OBJECT')
for o in bpy.context.scene.objects: o.select_set(False)
body.select_set(True)
cand = [mw @ v.co for v in body.data.vertices if abs((mw @ v.co).x) < 0.05 and abs((mw @ v.co).z - ZC) < 0.04]
fronty = min(c.y for c in cand)
G = Vector((0.0, fronty + G_YOFF, ZC))
print("center G =", [round(x, 3) for x in G], "fronty", round(fronty, 3), flush=True)
for poly in body.data.polygons:
    d = (mw @ poly.center) - G
    poly.select = (d.x / AX) ** 2 + (d.y / AY) ** 2 + (d.z / AZ) ** 2 < 1.0
print("faces selected =", sum(1 for p in body.data.polygons if p.select), flush=True)
pre = set(bpy.context.scene.objects)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
patch = [o for o in bpy.context.scene.objects if o not in pre][0]; patch.name = "PelvisDonorV2"
for o in bpy.context.scene.objects: o.select_set(False)
bpy.context.view_layer.objects.active = patch; patch.select_set(True)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.object.mode_set(mode='OBJECT')
pw = patch.matrix_world
V = np.array([list(pw @ v.co) for v in patch.data.vertices], np.float64)
F = np.array([list(p.vertices) for p in patch.data.polygons], np.int64)
# ---- ordered boundary loops ----
ec = defaultdict(int)
for t in F:
    for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]: ec[(min(a, b), max(a, b))] += 1
bnd = [k for k, c in ec.items() if c == 1]
adj = defaultdict(list)
for a, b in bnd: adj[a].append(b); adj[b].append(a)
visited = set(); loops = []
for s in list(adj):
    if s in visited or len(adj[s]) != 2: continue
    loop = [s]; visited.add(s); prev = -1; cur = s; ok = True
    while True:
        ns = adj[cur]
        if len(ns) != 2: ok = False; break
        nxt = ns[0] if ns[0] != prev else ns[1]
        if nxt == s: break
        if nxt in visited: ok = False; break
        loop.append(nxt); visited.add(nxt); prev = cur; cur = nxt
    if ok and len(loop) > 6: loops.append(loop)
loops.sort(key=len, reverse=True)
print(f"boundary loops: {[len(l) for l in loops[:5]]}", flush=True)
# ---- Taubin-smooth the main boundary loop (clean, even, no shrink) ----
main = loops[0]; idx = np.array(main)
P = V[idx].copy()
for it in range(24):
    lam = 0.5 if it % 2 == 0 else -0.53
    P = P + lam * (((np.roll(P, 1, 0) + np.roll(P, -1, 0)) * 0.5) - P)
V[idx] = P
# light relax of the 1-ring just inside the boundary so faces don't stretch
bset = set(main); ring = set()
for t in F:
    if any(v in bset for v in t):
        for v in t:
            if v not in bset: ring.add(int(v))
if ring:
    nbr = defaultdict(list)
    for t in F:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]: nbr[a].append(b); nbr[b].append(a)
    ring = np.array(sorted(ring))
    for _ in range(6):
        newp = V[ring].copy()
        for j, vi in enumerate(ring):
            newp[j] = V[nbr[vi]].mean(0) * 0.4 + V[vi] * 0.6
        V[ring] = newp
np.savez(ASSET, V=V.astype(np.float32), F=F, center=np.array(list(G), np.float32))
print(f"saved donor_pelvis_v2: verts={len(V)} faces={len(F)} mainloop={len(main)}", flush=True)
# rebuild + render
for v, co in zip(patch.data.vertices, V): v.co = Vector(co.tolist())
for o in list(bpy.context.scene.objects):
    if o != patch: bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\_pelvis_donor_v2.blend")
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'; sc.display.shading.show_cavity = True
sc.render.resolution_x = sc.render.resolution_y = 900
co = V; mn = co.min(0); mx = co.max(0); ctr = Vector(((mn + mx) / 2).tolist()); H = float((mx - mn).max())
cd = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cd); sc.collection.objects.link(cam); sc.camera = cam; cd.type = 'ORTHO'
def shot(d, p):
    cd.ortho_scale = H * 1.25; cam.location = ctr + Vector(d).normalized() * max(H, 0.2) * 2
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = p; bpy.ops.render.render(write_still=True)
shot((0, 0, -1), OUT + r"\_pelvis2_bottom.png"); shot((0, -0.5, -1), OUT + r"\_pelvis2_belowfront.png")
print("rendered", flush=True)
