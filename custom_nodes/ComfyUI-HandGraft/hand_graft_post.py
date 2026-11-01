"""POST-remesh hand graft: re-attach the FULL-RES donor hands to the already-remeshed body.

Why this exists (measured, 2026-09-08). ReplaceHandsWithDonor runs PRE-remesh and does its job
perfectly: it decimates the body to ~135k and grafts the untouched donor hands, so leaving that node
the hands are 5x finer than the body (hand edge 0.00111 vs body 0.00556). The target voxel remesh in
Trellis2BlenderSmartUV is then a UNIFORM-resolution operation -- it resamples everything onto one grid
and equalises it, so the hands come out at exactly body resolution (0.00247 vs 0.00247, ratio 1.00).
The donor supplies 7,432 triangles per hand; ~614 polygons per hand survive. The graft is undone.

The pelvis already solved this. Its comment states the rule: "the target voxel remesh would destroy
genital detail -> so we graft AFTER the remesh (never re-remeshed) and let gen_seams UV the grafted
mesh." Proof that it works, from one bake, same mesh, the two grafts differing only in which side of
the remesh they sit on:  GENITAL (post-remesh) edge 0.00116 = 0.45x body;  HANDS (pre-remesh) 0.00247
= 0.97x body.

So this runs in the pelvis's slot -- on last_seams.blend, after the voxel remesh, before gen_seams --
and simply cuts the voxelised hands off at the wrist and bridges fresh full-res donor hands on. The
pre-remesh graft stays exactly as it is: it is what makes the body watertight going into the remesh,
and its hands are what this stage cuts away.

Runs BEFORE pelvis_graft.py: pelvis writes _gs_isgenital.npy as face indices into the mesh as it
stands when IT finishes, so anything that adds faces must happen first or those indices go stale.

Fully headless (no GPU sculpt -- unlike the pelvis, a wrist is a near-circular tube). The stitch is
hand_replace.bridge_rings, the same angular two-pointer merge the pre-remesh graft uses: watertight
and twist-proof even when the stump and cuff rings have very different vertex counts.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import hand_replace as hr                                       # noqa: E402

_DONOR_R = os.path.join(HERE, "assets", "donor_geo_R.npz")
_DONOR_L = os.path.join(HERE, "assets", "donor_geo_L.npz")


def _argval(flag, default=None):
    a = sys.argv
    if "--" in a:
        a = a[a.index("--") + 1:]
    return a[a.index(flag) + 1] if flag in a else default


def _boundary_loops_mixed(polys):
    """hr.boundary_loops for a MIXED quad/tri face list. The remeshed body is quads (that is what
    makes it smooth -- triangles read as blocky), so it must not be triangulated just to find the
    wrist stump rings."""
    from collections import defaultdict
    ec = defaultdict(int)
    for f in polys:
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            ec[(a, b) if a < b else (b, a)] += 1
    adj = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1:
            adj[a].append(b)
            adj[b].append(a)
    seen, loops = set(), []
    for start in list(adj):
        if start in seen:
            continue
        loop, cur, prev = [], start, None
        while cur is not None and cur not in seen:
            loop.append(cur)
            seen.add(cur)
            nxts = [n for n in adj[cur] if n != prev]
            prev, cur = cur, (nxts[0] if nxts else None)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def graft_hands_post(V, F, donorR, donorL, seam=0.5, oversize=1.0):
    """V = body verts (Y-up), F = body faces AS-IS (mixed quads/tris -- kept). Returns (V2, F2, info).

    Same algorithm as hr.replace_hands, but face-LIST based so the body's quads survive: replace_hands
    concatenates face ARRAYS, which cannot hold quads and triangles in one buffer."""
    V = np.asarray(V, np.float64)
    arm, up, depth = hr.find_arm_axis(V)
    A = V[:, arm]
    wxR, OR, rR = hr.find_wrist(V, arm, -1)
    wxL, OL, rL = hr.find_wrist(V, arm, +1)

    # Same overshoot clamp as replace_hands: find_wrist slices a fixed depth from each fingertip, so on
    # the shorter arm it overshoots into the torso. A mis-detect ALWAYS overshoots -> clamp to the closer.
    full = float(A.max() - A.min())
    tipR = float(A.min())
    tipL = float(A.max())
    dR = abs(wxR - tipR)
    dL = abs(wxL - tipL)
    dmin = min(dR, dL)
    o2 = [i for i in range(3) if i != arm]

    def _wrist_at(wx):
        mm = (A >= wx - full * 0.025) & (A < wx + full * 0.025)
        sub = V[mm]
        O = sub.mean(0) if int(mm.sum()) else V.mean(0)
        rr = float(np.median(np.sqrt(((sub[:, o2] - O[o2]) ** 2).sum(1)))) if int(mm.sum()) else full * 0.03
        return O, rr

    if dR > 1.6 * dmin:
        wxR = tipR + dmin
        OR, rR = _wrist_at(wxR)
        print(f"[hand_graft_post] R-wrist overshoot clamped -> {wxR:.4f}", flush=True)
    if dL > 1.6 * dmin:
        wxL = tipL - dmin
        OL, rL = _wrist_at(wxL)
        print(f"[hand_graft_post] L-wrist overshoot clamped -> {wxL:.4f}", flush=True)

    keep = (A >= wxR) & (A <= wxL)                      # body = everything between the wrists
    newidx = -np.ones(len(V), np.int64)
    newidx[keep] = np.arange(int(keep.sum()))
    Vb = V[keep]
    Fb = [[int(newidx[v]) for v in f] for f in F if all(keep[v] for v in f)]
    print(f"[hand_graft_post] cut at wrists {wxR:.4f}/{wxL:.4f} (arm axis {arm}): "
          f"body {len(V)}v/{len(F)}f -> {len(Vb)}v/{len(Fb)}f", flush=True)
    stump_loops = _boundary_loops_mixed(Fb)
    span = A.max() - A.min()

    def _rad(lp):
        sub = Vb[lp][:, o2]
        return float(np.median(np.sqrt(((sub - sub.mean(0)) ** 2).sum(1))))

    Vparts = [Vb]
    Fout = list(Fb)
    off = len(Vb)
    bridges, cap_loops = [], []
    ngrafted = 0
    for dz, sign, Ob, rb, wx, side in ((donorR, -1, OR, rR, wxR, "R"), (donorL, +1, OL, rL, wxL, "L")):
        here = [lp for lp in stump_loops if abs(Vb[lp].mean(0)[arm] - wx) < 0.06 * span]
        if not here:
            print(f"[hand_graft_post] WARNING: no stump ring at the {side} wrist -- that hand is left as-is",
                  flush=True)
            continue
        outer = max(here, key=_rad)                     # OUTER rim = bridge target; rest are inner holes
        rim_c = Vb[outer].mean(0)
        dv = np.asarray(dz['verts'], np.float64)
        dfc = np.asarray(dz['faces'], np.int64)
        Od = np.asarray(dz['cuff_O'], np.float64)
        ad = np.asarray(dz['axis'], np.float64)
        rd = float(dz['radius'])
        a_out = np.zeros(3)
        a_out[arm] = sign
        s = (rb / rd) * oversize
        target = rim_c + a_out * (seam * rb)
        P = hr._apply_R(hr.rodrigues(ad, a_out), dv - Od) * s + target
        cuff = hr.boundary_loops(dfc)
        cuff_g = [off + i for i in max(cuff, key=len)] if cuff else []
        bridges.append((list(outer), cuff_g, a_out.copy(), rim_c.copy()))
        cap_loops += [list(lp) for lp in here if lp is not outer]
        Vparts.append(P)
        Fout += [[int(a) + off, int(b) + off, int(c) + off] for a, b, c in dfc]
        off += len(P)
        ngrafted += 1
        print(f"[hand_graft_post] {side}: stump ring {len(outer)}v (r={rb:.5f}) <- donor cuff "
              f"{len(cuff_g)}v (r={rd:.5f}, scale {s:.4f}), {len(dfc)} donor faces", flush=True)

    V2 = np.concatenate(Vparts, 0)
    for lp in cap_loops:                                # seal inner rings / tiny cut-holes
        c, cf = hr._cap_loop(V2, lp, len(V2))
        V2 = np.concatenate([V2, c], 0)
        Fout += [[int(x) for x in f] for f in cf]
    for st, cu, ax, ce in bridges:
        if len(st) >= 3 and len(cu) >= 3:
            bf = hr.bridge_rings(V2, st, cu, ax, ce)
            Fout += [[int(x) for x in f] for f in bf]
    return V2, Fout, dict(arm=arm, wxR=wxR, wxL=wxL, ngrafted=ngrafted)


if __name__ == "__main__":
    import traceback
    in_blend = _argval("--in_blend")
    seam = float(_argval("--seam", "0.5"))
    try:
        if not in_blend:
            raise SystemExit("hand_graft_post.py needs --in_blend <last_seams.blend>")
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        with bpy.data.libraries.load(in_blend) as (_df, _dt):
            _dt.objects = list(_df.objects)
        src = None
        for _ob in _dt.objects:
            if _ob is None:
                continue
            bpy.context.scene.collection.objects.link(_ob)
            if _ob.type == 'MESH' and (src is None or len(_ob.data.polygons) > len(src.data.polygons)):
                src = _ob
        if src is None:
            raise RuntimeError("no mesh object in " + in_blend)
        name = src.name
        mw = src.matrix_world
        R = np.array([[mw[r][c] for c in range(3)] for r in range(3)], float)
        T = np.array([mw[r][3] for r in range(3)], float)
        nv = len(src.data.vertices)
        _co = np.empty(nv * 3, np.float64)
        src.data.vertices.foreach_get('co', _co)
        Vw = _co.reshape(nv, 3) @ R.T + T
        Fw = [list(p.vertices) for p in src.data.polygons]      # KEEP quads

        # The pipeline body comes out of the OBJ import Z-UP; the donor cuffs were authored in the
        # Y-up trimesh frame. rodrigues() only aligns the cuff AXIS to the arm -- the roll about that
        # axis is whatever the minimal rotation gives, so a frame mismatch lands the palm rotated.
        # Rotate into Y-up for the graft and back afterwards, exactly as pelvis_graft.py does.
        _zup = float(np.ptp(Vw[:, 2])) > float(np.ptp(Vw[:, 1]))
        if _zup:
            Vw = Vw @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])    # -90 about X
        V, F, info = graft_hands_post(Vw, Fw, np.load(_DONOR_R), np.load(_DONOR_L), seam=seam)
        if info["ngrafted"] == 0:
            raise RuntimeError("no wrist stump ring found on either side -- nothing grafted")
        if _zup:
            V = V @ np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])      # +90 about X (inverse)

        me = bpy.data.meshes.new(name + "_hg")
        me.from_pydata([tuple(p) for p in V.tolist()], [], F)
        me.validate(verbose=False)
        for p in me.polygons:
            p.use_smooth = True
        me.update()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        for x in list(bpy.data.objects):
            if x is not o:
                bpy.data.objects.remove(x, do_unlink=True)
        for _lib in list(bpy.data.libraries):
            bpy.data.libraries.remove(_lib)
        o.name = name
        o.matrix_world = Matrix.Identity(4)
        bpy.ops.wm.save_as_mainfile(filepath=in_blend)
        print(f"[hand_graft_post] done (pipeline): {len(V)} verts, {len(F)} faces "
              f"({info['ngrafted']}/2 hands grafted)", flush=True)
    except Exception:
        traceback.print_exc()
        raise
