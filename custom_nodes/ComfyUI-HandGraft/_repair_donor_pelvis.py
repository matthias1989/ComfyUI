"""Repair the pelvis donor asset IN PLACE-COMPATIBLE form: move vertices only.

`genital_donor_zones.npy` is indexed by donor vertex, and pelvis_graft.py asserts
len(zones) == len(W), so the repair may never add, remove or reorder a vertex. The
only permitted edit is moving the ones that are broken.

Measured on donor_pelvis_v2.npz (2026-09-05): 92 self-intersecting face pairs
involving just 112 vertices, all inside a 12 x 1.5 x 6 mm patch on the midline
where the inner labia (zone 3) pass through the surrounding skin (zone 0). Those
92 become ~705 once the graft's smoothing passes amplify them, and they are the
mess left on the graft's sides.

Run:  blender --background --python _repair_donor_pelvis.py -- --in <npz> --out <npz>
"""
import bpy, bmesh, sys, math
import numpy as np
from mathutils.bvhtree import BVHTree


def _arg(flag, default=None):
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return a[a.index(flag) + 1] if flag in a else default


def _bm(V, F):
    me = bpy.data.meshes.new("t")
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(map(int, f)) for f in F])
    me.update()
    bm = bmesh.new(); bm.from_mesh(me)
    bm.faces.ensure_lookup_table(); bm.verts.ensure_lookup_table(); bm.verts.index_update()
    bpy.data.meshes.remove(me)
    return bm


def _pairs(bm):
    bv = BVHTree.FromBMesh(bm)
    out = []
    for x, y in bv.overlap(bv):
        if x >= y:
            continue
        fa, fb = bm.faces[x], bm.faces[y]
        if {v.index for v in fa.verts} & {v.index for v in fb.verts}:
            continue          # faces that merely touch are not intersections
        out.append((fa, fb))
    return out


def _creases(bm):
    a = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            try:
                a.append(math.degrees(e.calc_face_angle()))
            except Exception:
                pass
    return np.array(a)


def main():
    src = _arg("--in"); dst = _arg("--out")
    iters = int(_arg("--iters", "120")); w = float(_arg("--w", "0.4"))
    capk = float(_arg("--cap", "2.0"))      # max displacement, x median edge length
    ring = int(_arg("--ring", "1"))

    d = np.load(src)
    V0 = d["V"].astype(float); F = d["F"].astype(int)
    extra = {k: d[k] for k in d.keys() if k not in ("V", "F")}
    V = V0.copy()

    bm = _bm(V, F)
    nb = {}
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    med_edge = float(np.median([e.calc_length() for e in bm.edges]))
    cap = capk * med_edge
    c0 = _creases(bm)
    p = _pairs(bm)
    print(f"[repair] start: {len(p)} self-intersecting pairs; median edge {med_edge:.5f}; "
          f"displacement cap {cap:.5f}", flush=True)
    bm.free()

    it = 0
    for it in range(1, iters + 1):
        bm = _bm(V, F)
        p = _pairs(bm)
        if not p:
            bm.free()
            break
        hit = set()
        for fa, fb in p:
            for f in (fa, fb):
                for v in f.verts:
                    hit.add(v.index)
        for _ in range(ring):
            hit |= {j for i in list(hit) for j in nb.get(i, ())}
        bm.free()
        Q = V.copy()
        for i in hit:
            k = nb.get(i)
            if k:
                Q[i] = (1.0 - w) * V[i] + w * V[list(k)].mean(0)
        # never let the repair drift a vertex further than the cap from where it started
        dv = Q - V0
        n = np.linalg.norm(dv, axis=1)
        over = n > cap
        if over.any():
            Q[over] = V0[over] + dv[over] * (cap / n[over])[:, None]
        V = Q
        if it % 10 == 0 or it == 1:
            print(f"[repair] iter {it:3d}: {len(p)} pairs, {len(hit)} verts moving", flush=True)

    bm = _bm(V, F)
    p = _pairs(bm); c1 = _creases(bm); bm.free()
    disp = np.linalg.norm(V - V0, axis=1)
    print(f"[repair] done after {it} iters: pairs {len(p)}", flush=True)
    print(f"[repair] moved {int((disp > 1e-9).sum())} verts; displacement med "
          f"{np.median(disp[disp > 1e-9]) if (disp > 1e-9).any() else 0:.5f} max {disp.max():.5f} "
          f"(cap {cap:.5f}, median edge {med_edge:.5f})", flush=True)
    print(f"[repair] dihedral >60deg {int((c0 > 60).sum())} -> {int((c1 > 60).sum())}; "
          f">90deg {int((c0 > 90).sum())} -> {int((c1 > 90).sum())}; "
          f"p99 {np.percentile(c0, 99):.2f} -> {np.percentile(c1, 99):.2f}", flush=True)
    if dst:
        np.savez(dst, V=V, F=F, **extra)
        print(f"[repair] wrote {dst}", flush=True)


main()
