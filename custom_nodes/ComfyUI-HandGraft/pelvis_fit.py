"""Fit the genital donor to THIS body -- the per-character step the shipped donor never had.

donor_pelvis_T5D10.npz is the base donor (T5R) already fitted to character_posed_00003_'s cage by the offline harness
(scratchpad refit_old: _landscape/_grid/_apron/_fit8). pelvis_graft._place only re-anchors it (crotch height, leg-gap
midline, pelvis-width scale), so every other body inherited 0003's seat. On character_posed_00197_ the mons stands
23 mm further forward relative to the crotch, and the graft landed 17-25 mm INSIDE her belly over a 4 cm band above the
vulva -- the "cut off part of the belly" -- with 3,896 low-poly verts off the master (0003: ~440), so the colour fill
darkened the whole region (skin under the features 0.53x reference against 0.75-0.84x).

This is that harness consolidated, run per character:
  1. scale the LIPS only to a 9 cm cleft (geodesic falloff to the rim), the fourchette held on the crotch line;
  2. rigid seat (up / forward, axis tilt locked) of the outer-lip edge (zone 1) on the smoothed crotch-skin sheet;
  3. back hinge seating the anus core on the skin;
  4. apron deform: the free skin apron moves along its own smoothed normal onto the skin, the detail never moves.
Output: the fitted donor re-expressed in the base donor's ASSET space (the inverse of this body's placement affine),
so pelvis_graft._place on the same cage reproduces the fitted pose exactly.

Numerics are the harness's, operation for operation: on 0003's inputs this reproduces the shipped T5D10 bit-for-bit.
Two things differ, neither touching the result: A and MM come from this cage (the harness had 0003's constants), and
the body crop is sized from the donor itself (the harness sized it from two of 0003's old fit poses).

    python pelvis_fit.py <in.npz> <out_donor.npz>
      in.npz: cage_V, cage_F (tris), W (base donor as placed on this cage), A (anchor), MM, base (base donor path)
Settings (pelvis_config.PIPELINE_ENV): PELVIS_FIT_BACK_MM 8.8, PELVIS_FIT_TILT0 -10, PELVIS_FIT_FWD0 8,
FIT8_AXIS_TILT 27, FIT8_BEND_Z1 1, FIT8_APRON_BAND 0.35, FIT8_UP_CM 1, PELVIS_FIT_CLEFT_CM 9.0.
Needs scipy + trimesh (ComfyUI's python, not Blender's).
"""
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags as _diags
from scipy.sparse.csgraph import connected_components, dijkstra as _dij
from scipy.sparse.linalg import spsolve as _spsolve

_T0 = time.time()


def say(m):
    print("[pfit] %s   (+%.0fs)" % (m, time.time() - _T0), flush=True)


# ---- body + rim helpers (harness _collide.py) ------------------------------------------------------------------------
def order_loop(F):
    Es = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(Es, axis=0, return_counts=True); rimE = u[c == 1]
    nb = {}
    for a, b in rimE: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    loops = []; seen = set()
    for s0 in nb:
        if s0 in seen: continue
        lp = [s0]; seen.add(s0); prev, cur = None, s0
        while True:
            nxt = [x for x in nb[cur] if x != prev]
            if not nxt or nxt[0] == s0: break
            prev, cur = cur, nxt[0]
            if cur in seen: break
            lp.append(cur); seen.add(cur)
        loops.append(np.asarray(lp))
    return u, rimE, max(loops, key=len), len(loops)


class Body:
    """The cage cropped to a box around the donor: face centres, normals, face adjacency."""
    def __init__(self, Vc, Fc, lo, hi):
        cen = Vc[Fc].mean(1)
        sel = np.nonzero(((cen >= lo) & (cen <= hi)).all(1))[0]
        self.fidx = sel; F = Fc[sel]
        used, inv = np.unique(F.ravel(), return_inverse=True)
        self.vidx = used; self.V = Vc[used]; self.F = inv.reshape(-1, 3)
        T = self.V[self.F]; self.c = T.mean(1)
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); self.n = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30)
        self.r = np.linalg.norm(T - self.c[:, None], axis=2).max(1)
        self.kdc = cKDTree(self.c)
        nf = len(self.F)
        E = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]]); E.sort(1)
        fi = np.r_[np.arange(nf), np.arange(nf), np.arange(nf)]
        key = E[:, 0].astype(np.int64) * (len(self.V) + 1) + E[:, 1]
        o = np.argsort(key, kind="stable"); key = key[o]; fi = fi[o]
        same = key[1:] == key[:-1]
        self.fa = fi[:-1][same]; self.fb = fi[1:][same]

    def crossings(self, P, E):
        """donor edges crossing body faces: (edge, face)"""
        p, q = P[E[:, 0]], P[E[:, 1]]
        m = 0.5 * (p + q); h = 0.5 * np.linalg.norm(q - p, axis=1)
        lists = self.kdc.query_ball_point(m, h + self.r.max())
        ei = np.repeat(np.arange(len(E)), [len(l) for l in lists]); fi = np.fromiter((j for l in lists for j in l), int, len(ei))
        if not len(ei): return np.zeros(0, int), np.zeros(0, int)
        T = self.V[self.F[fi]]; o = p[ei]; d = q[ei] - o
        e1 = T[:, 1] - T[:, 0]; e2 = T[:, 2] - T[:, 0]
        pv = np.cross(d, e2); det = (e1 * pv).sum(1); ok = np.abs(det) > 1e-20
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - T[:, 0]; u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1); v = (d * qv).sum(1) * inv; t = (e2 * qv).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        return ei[hit], fi[hit]


def crotch_bottom(Vc, A):
    """The body's crotch, measured: the lowest point of its underside ON ITS OWN MIDLINE (A[0], which _place takes
    from the leg gap), searched within two of _crotch_y's bins of A[1].

    _crotch_y reads the crotch off 60 fixed height bins and returns the lower EDGE of the bin it lands in, so its
    error is anything up to one bin (1/60 of the height, 2.8 cm at 170 cm) and differs per body. Measured
    2026-09-18 against this point: 0003 2.5 mm above it, 197 5 mm below, female_elf_00004_ 12.5 mm below,
    female_elf_00005_ ~9 mm below -- and the fit held the fourchette a fixed 1 cm above that estimate, so 00004's
    lips hung below her crotch. Below the crotch the midline is the empty leg gap (8-20 mm wide on all four bodies);
    the first surface above it is the perineum. Returns None when the window's floor is reached (legs closing the gap
    all the way down) -- the caller then keeps A[1]."""
    H = float(np.ptp(Vc[:, 1]))
    b = H / 60.0
    m = (np.abs(Vc[:, 0] - A[0]) < 0.003 * H) & (Vc[:, 1] > A[1] - 2 * b) & (Vc[:, 1] < A[1] + 2 * b)
    if not m.any():
        return None
    y = float(Vc[m, 1].min())
    return None if y < A[1] - 2 * b + 0.002 * H else y


def fit(Vc, Fc, W, d, Z, aw, A, MM, back_mm=8.8, tilt0=-10.0, fwd0=8.0, axis_tilt="27", bend_z1=True,
        apron_band=0.35, up_cm=1.0, cleft_cm=9.0, crop=None, cleft_ref_cm=0.0, crotch_up_cm=None,
        anus_seat="rotate", lips_floor=False, scale=1.0, perineum_mm=0.0, anus_seat_rim=False, canal_inward=False):
    """Returns the fitted donor in the base donor's asset space (same dtype as d["V"]) and a dict of measurements.
    `crop` = (lo, hi) of the body box; default: the donor's box grown by BACK + one donor extent.
    `cleft_ref_cm` > 0 makes the cleft RELATIVE to this pelvis: cleft_cm is the length approved on the reference body
    (character_posed_00003_), cleft_ref_cm the placed donor's cleft there; here the target is cleft_cm x (this body's
    placed cleft / cleft_ref_cm). _place already scales the donor by pelvis width, so the lips keep 0003's proportion
    instead of being stretched to one absolute size -- on 197's narrower pelvis 9 cm took x1.32 against 0003's x1.16
    and the lips hung into the leg gap. 0 = the absolute cleft_cm (the harness's rule)."""
    BACK = back_mm / MM
    F = d["F"].astype(np.int64)
    n = len(W)
    if crop is None:
        grow = BACK + float(np.ptp(W, 0).max())      # the fit moves the donor by tens of mm; one donor extent covers it
        crop = (W.min(0) - grow, W.max(0) + grow)
    B = Body(Vc, Fc, crop[0], crop[1])
    E, rimE, loop, nl = order_loop(F); rim = loop
    nf = len(B.F)

    def sheets(P):
        lo = P.min(0) - BACK; hi = P.max(0) + BACK
        if (lo < crop[0]).any() or (hi > crop[1]).any():
            raise RuntimeError("the donor left the body crop -- the crop would change the result")
        inbox = ((B.c >= lo) & (B.c <= hi)).all(1)
        keep = inbox[B.fa] & inbox[B.fb]
        _, lab = connected_components(coo_matrix((np.ones(keep.sum()), (B.fa[keep], B.fb[keep])), shape=(nf, nf)), directed=False)
        cand = np.nonzero(inbox)[0]; fA = cand[np.argmin(np.linalg.norm(B.c[cand] - A, axis=1))]
        return inbox & (lab == lab[fA]), inbox

    C0 = (W[rim] - A) * MM
    regions = (("front", C0[:, 2] > 40), ("sides", (C0[:, 2] >= -20) & (C0[:, 2] <= 40)), ("back", C0[:, 2] < -20))

    def rimfit(P):
        an, _ = sheets(P)
        sub = trimesh.Trimesh(B.V, B.F[an], process=False)
        cp, dist, _ = trimesh.proximity.closest_point(sub, P[rim])
        return dist * MM

    # ---- harness _grid.py: rim band, crossing counts --------------------------------------------------------------
    Ed = E; wl = np.linalg.norm(W[Ed[:, 0]] - W[Ed[:, 1]], axis=1)
    Gd = coo_matrix((np.r_[wl, wl], (np.r_[Ed[:, 0], Ed[:, 1]], np.r_[Ed[:, 1], Ed[:, 0]])), shape=(n, n)).tocsr()
    rsp = np.median(np.linalg.norm(np.roll(W[loop], -1, 0) - W[loop], axis=1))
    gd = _dij(Gd, directed=False, indices=loop, min_only=True)
    band = gd <= 2 * rsp
    say("rim spacing %.2f mm; rim band %d verts, lips %d, anus %d" % (rsp * MM, band.sum(), np.isin(Z, [1, 2, 3, 4]).sum(), (aw > 0.5).sum()))

    def ev(P):
        an, inbox = sheets(P)
        ei, fi = B.crossings(P, E)
        oth = ~an[fi]
        vb = band[E[ei, 0]] | band[E[ei, 1]]
        return dict(interior=int((oth & ~vb).sum()), own=int((~oth).sum()))

    # ---- harness _apron.py: graph, weights, smoothed normals / skin, signed gap -------------------------------------
    Eu = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    Lg = coo_matrix((np.ones(2 * len(Eu)), (np.r_[Eu[:, 0], Eu[:, 1]], np.r_[Eu[:, 1], Eu[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(Lg.sum(1)).ravel()
    Lap = (_diags(deg) - Lg).tocsr()
    outw_ref = A - Vc.mean(0)
    lab = np.isin(Z, [1, 2, 3, 4])
    acore = aw >= 0.5
    detail = (np.isin(Z, [2, 3, 4]) | acore) if bend_z1 else (lab | acore)

    def weights(P):
        le = np.linalg.norm(P[Eu[:, 0]] - P[Eu[:, 1]], axis=1)
        G = coo_matrix((np.r_[le, le], (np.r_[Eu[:, 0], Eu[:, 1]], np.r_[Eu[:, 1], Eu[:, 0]])), shape=(n, n)).tocsr()
        dd = _dij(G, directed=False, indices=np.nonzero(detail)[0], min_only=True)
        dr = _dij(G, directed=False, indices=rim, min_only=True)
        t = np.clip(dd / np.maximum(dd + dr, 1e-12), 0, 1)
        if apron_band > 0:       # the outermost band follows the body fully: the donor arrives with the body's own slope
            t = np.clip(t / max(1.0 - apron_band, 1e-6), 0, 1)
        w = t * t * (3 - 2 * t)
        w[detail] = 0.0
        return w, float(np.median(dd[rim])), float(np.median(le))

    def smooth_normals(P, iters):
        Q = P.copy()
        for _ in range(iters):
            Q = 0.5 * Q + 0.5 * (Lg @ Q) / np.maximum(deg, 1)[:, None]
        fn = np.cross(Q[F[:, 1]] - Q[F[:, 0]], Q[F[:, 2]] - Q[F[:, 0]])
        if (fn.sum(0) @ outw_ref) < 0:
            fn = -fn
        vn = np.zeros_like(P)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30)

    def smooth_skin(P, iters):
        an, _ = sheets(P)
        Fs = B.F[an]
        used, inv = np.unique(Fs.ravel(), return_inverse=True)
        Vs = B.V[used].copy()
        Fl = inv.reshape(-1, 3)
        m = len(Vs)
        Es = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
        ue, ce = np.unique(Es, axis=0, return_counts=True)
        bnd = np.zeros(m, bool)
        bnd[ue[ce == 1].ravel()] = True
        As = coo_matrix((np.ones(2 * len(ue)), (np.r_[ue[:, 0], ue[:, 1]], np.r_[ue[:, 1], ue[:, 0]])), shape=(m, m)).tocsr()
        ds = np.maximum(np.asarray(As.sum(1)).ravel(), 1)
        for _ in range(iters):
            Q = 0.5 * Vs + 0.5 * (As @ Vs) / ds[:, None]
            Q[bnd] = Vs[bnd]
            Vs = Q
        return trimesh.Trimesh(Vs, Fl, process=False)

    def signed_gap(P, N, skin, gate):
        t = np.full(n, np.nan)
        for sgn in (1.0, -1.0):
            loc, ri, _ = skin.ray.intersects_location(P - sgn * N * 1e-6, sgn * N, multiple_hits=True)
            for k in np.unique(ri):
                tt = ((loc[ri == k] - P[k]) * N[k]).sum(1)
                tt = tt[np.argmin(np.abs(tt))]
                if abs(tt) <= gate and (np.isnan(t[k]) or abs(tt) < abs(t[k])):
                    t[k] = tt                  # + = the skin is OUTSIDE the vertex (donor sunk), - = donor floats
        return t

    # ---- harness _fit8.py ---------------------------------------------------------------------------------------------
    cm = lambda x: x * MM / 10
    inner = np.isin(Z, [3, 4])
    z1 = Z == 1

    def landmarks(P):
        c0 = P[lab].mean(0)
        _, _, vt = np.linalg.svd(P[lab] - c0, full_matrices=False)
        ax = vt[0] if vt[0][2] >= 0 else -vt[0]
        uu = (P - c0) @ ax
        w4 = aw ** 4
        return dict(ant=P[lab][np.argmax(uu[lab])], four=P[inner][np.argmin(uu[inner])],
                    anus=(P * w4[:, None]).sum(0) / w4.sum(), low=P[lab][np.argmin(P[lab][:, 1])],
                    tilt=float(np.degrees(np.arctan2(ax[1], ax[2]))))

    # 0. the WHOLE donor bigger (user 2026-09-19: "use a bigger scale and then play with the movement, that way the anus
    # will go more back"): a longer perineum puts the anus further back; the seat below then moves the donor into place.
    # About the fourchette, before the lips scale; the placement W_place stays for the way back to asset space
    W_place = W
    if scale != 1.0:
        _f0 = landmarks(W)["four"]
        W = _f0 + float(scale) * (W - _f0)
        say("whole donor scaled x%.3f about the fourchette" % scale)

    # 1. lips-only scale to the cleft length (geodesic falloff to the rim)
    Lw = np.linalg.norm(W[Eu[:, 0]] - W[Eu[:, 1]], axis=1)
    Gw = coo_matrix((np.r_[Lw, Lw], (np.r_[Eu[:, 0], Eu[:, 1]], np.r_[Eu[:, 1], Eu[:, 0]])), shape=(n, n)).tocsr()
    d_lab = _dij(Gw, directed=False, indices=np.nonzero(lab)[0], min_only=True)
    d_rim = _dij(Gw, directed=False, indices=rim, min_only=True)
    tl_ = np.clip(d_lab / np.maximum(d_lab + d_rim, 1e-12), 0, 1)
    w_lips = 1 - tl_ * tl_ * (3 - 2 * tl_)
    cl = W[lab].mean(0)

    def cleft_len(P):
        L = landmarks(P)
        return cm(np.linalg.norm(L["ant"] - L["four"]))

    if cleft_ref_cm > 0:
        placed = cleft_len(W)
        say("cleft relative to this pelvis: placed %.2f cm vs %.2f cm on the reference -> target %.2f cm (reference %.1f)"
            % (placed, cleft_ref_cm, cleft_cm * placed / cleft_ref_cm, cleft_cm))
        cleft_cm = cleft_cm * placed / cleft_ref_cm
    lo_s, hi_s = 1.0, 1.5
    for _ in range(30):
        ms = 0.5 * (lo_s + hi_s)
        if cleft_len(cl + (1 + (ms - 1) * w_lips)[:, None] * (W - cl)) < cleft_cm:
            lo_s = ms
        else:
            hi_s = ms
    Wl = cl + (1 + (hi_s - 1) * w_lips)[:, None] * (W - cl)
    f_idx = int(np.nonzero(inner)[0][np.argmin(((Wl[inner] - Wl[lab].mean(0)) @ (landmarks(Wl)["ant"] - landmarks(Wl)["four"])))])
    cc = Wl.mean(0)
    say("lips scale %.3f -> cleft %.1f cm" % (hi_s, cleft_len(Wl)))

    def rigid(tilt, up, fwd, base=Wl):
        ca, sa = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
        R = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        return (base - cc) @ R.T + cc + np.array([0, up, fwd]) / MM

    def skin_residuals(P, idx, skin):
        N = smooth_normals(P, 30)
        q, dist, tri = trimesh.proximity.closest_point(skin, P[idx])
        m = skin.face_normals[tri]
        m = m * np.sign((m * N[idx]).sum(1))[:, None]
        return ((P[idx] - q) * m).sum(1), m                 # + = floats above the skin, - = sunk

    def huber_w(r, delta):
        a = np.abs(r)
        return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))

    # 2. rigid seat of the outer-lip edge, fourchette on the crotch line
    tilt, fwd = tilt0, fwd0
    if axis_tilt:
        lo_t, hi_t = -80.0, 40.0
        for _ in range(40):
            mt = 0.5 * (lo_t + hi_t)
            if landmarks(rigid(mt, 0, 0))["tilt"] > float(axis_tilt):
                lo_t = mt
            else:
                hi_t = mt
        tilt = 0.5 * (lo_t + hi_t)
        say("axis tilt locked at %.0f deg -> rigid tilt %+.1f" % (float(axis_tilt), tilt))
    UPOFF = float(up_cm) * 10 / MM
    TGT = A[1] + UPOFF                                      # fourchette height: up_cm above the detector's crotch
    cb = crotch_bottom(Vc, A)
    if cb is None:
        say("crotch bottom: not measurable on the midline (legs closed) -- fourchette %.2f cm above the detector" % up_cm)
    else:
        say("crotch bottom (lowest midline surface) %+.2f cm from the detector; the detector rule puts the fourchette "
            "%.12f cm above it" % (cm(cb - A[1]), cm(TGT - cb)))
        if crotch_up_cm is not None:
            TGT = cb + float(crotch_up_cm) * 10 / MM
            say("fourchette held %.3f cm above the MEASURED crotch (%+.2f cm vs the detector rule)"
                % (crotch_up_cm, cm(TGT - (A[1] + UPOFF))))
    if lips_floor and cb is not None:
        # LIPS NEVER BELOW THE CROTCH (user 2026-09-18, female_elf_00004_'s lips hung into her leg gap): with the axis
        # tilt locked the seat only moves the donor up / forward, so the lowest labia sits a fixed D below the
        # fourchette. Hold the fourchette at least -D above the MEASURED crotch; where the rule above already clears it,
        # nothing changes (0003: 0.40 cm clear -> byte-identical). A fixed offset from the measured crotch lifted
        # flat-floored bodies 1-2 cm and the anus hinge then folded the perineum into a nub (197, 00005); this raises
        # only as far as needed: 197 +0.56, 00004 +1.71, 00005 +1.26 cm, hinges -30..-32 deg (0003 -29.5).
        _Pt = rigid(tilt, 0, fwd)
        D = float(_Pt[lab][:, 1].min() - _Pt[f_idx, 1])
        need = cb - D
        if need > TGT:
            say("lips would hang %.2f cm below the crotch -> fourchette raised %.2f cm (lowest labia on the crotch)"
                % (cm(need - TGT), cm(need - TGT)))
            TGT = need
        else:
            say("lowest labia %.2f cm above the crotch -- placement unchanged" % cm(TGT - need))
    idx1 = np.nonzero(z1)[0]
    delta = 2.0 / MM

    def seat_from(tilt, fwd):
        up = (TGT - rigid(tilt, 0, fwd)[f_idx, 1]) * MM
        for it in range(20):
            P = rigid(tilt, up, fwd)
            if it % 5 == 0:
                skin = smooth_skin(P, 20)
            r, m = skin_residuals(P, idx1, skin)
            q = P[idx1] - (cc + np.array([0, up, fwd]) / MM)
            J = np.c_[(m * np.c_[np.zeros(len(q)), -q[:, 2], q[:, 1]]).sum(1) * np.pi / 180, m[:, 1] / MM, m[:, 2] / MM]
            wv = huber_w(r, delta)
            qf = P[f_idx] - (cc + np.array([0, up, fwd]) / MM)
            Jc = np.array([[-qf[2] * np.pi / 180, 1 / MM, 0.0]])
            rc = np.array([P[f_idx, 1] - TGT])
            wc = 1e3 * len(idx1)
            H = (J * wv[:, None]).T @ J + wc * Jc.T @ Jc
            g = (J * wv[:, None]).T @ r + wc * Jc.T @ rc
            if axis_tilt:
                step = np.r_[0.0, np.linalg.lstsq(H[1:, 1:] + 1e-9 * np.eye(2), -g[1:], rcond=None)[0]]
            else:
                step = np.linalg.lstsq(H + 1e-9 * np.eye(3), -g, rcond=None)[0]
            step = np.clip(step, [-3, -3, -3], [3, 3, 3])
            tilt += step[0]; up += step[1]; fwd += step[2]
            if np.abs(step).max() < 0.02:
                break
        return tilt, up, fwd, it

    # THE SEAT STARTS ON HER FRONT SKIN (2026-09-25). It is a local search (20 steps of at most 3 mm) that seats the
    # outer-lip edge on the NEAREST skin, and it used to start where the placement left the donor -- 8 mm in front of
    # an anchor whose depth under her skin is the reference body's, not hers. That start was 6-10 cm behind her front
    # and the search went wherever the nearest skin was: female_elf_00006_ (anchor 77% of her pelvis depth back, the
    # others 49-57%) was seated 50.8 mm BACKWARD onto her buttocks ("no pelvis graft"); 00011 (anchor 3.7 cm lower
    # than the reference) ran out of steps 1.7 mm forward with the lip edge sunk 28.6 mm inside her ("the pelvis graft
    # completely collapses"); and 00005, 00008, 00010, 197 used all 20 steps too, still moving. So the search starts
    # from a measurement: the donor's front-most lip-edge vertex on her midline front skin at its own height (the
    # midline and height band as crotch_bottom measures them), and refines from there.
    idx_front = idx1[np.argmax(rigid(tilt, 0, 0)[idx1, 2])]
    P0 = rigid(tilt, 0, 0)
    Hb = float(np.ptp(Vc[:, 1]))
    colm = (np.abs(Vc[:, 0] - A[0]) < 0.003 * Hb) & (np.abs(Vc[:, 1] - P0[idx_front, 1]) < Hb / 60.0)
    if colm.sum() >= 8:
        fstart = (float(np.percentile(Vc[colm, 2], 98)) - P0[idx_front, 2]) * MM
        say("seat starts with the front lip edge on her front skin: fwd %+.1f mm (was %+.1f from the placement)" % (fstart, fwd))
    else:
        fstart = fwd
        say("seat: her midline front not measurable at the lip edge's height -- starting from the placement (fwd %+.1f)" % fwd)
    tilt, up, fwd, it = seat_from(tilt, fstart)
    # Behind the crotch anchor is never the vulva: every good seat is in front of it
    if fwd < 0:
        raise RuntimeError("the seat went behind the crotch anchor (fwd %+.1f mm) -- no front seat on this body" % fwd)
    P = rigid(tilt, up, fwd)
    skin = smooth_skin(P, 20)
    r, _ = skin_residuals(P, idx1, skin)
    seat = dict(tilt=float(tilt), up=float(up), fwd=float(fwd), edge_gap_med=float(np.median(r) * MM),
                edge_gap_p90=float(np.percentile(np.abs(r), 90) * MM))
    say("rigid seat: tilt %+.1f deg, up %+.1f mm, fwd %+.1f mm (%d it) | outer-lip edge gap med %+.1f mm, |gap| p90 %.1f mm | fourchette %+.2f cm"
        % (tilt, up, fwd, it + 1, seat["edge_gap_med"], seat["edge_gap_p90"], cm(P[f_idx, 1] - A[1])))

    # 3. back hinge seating the anus core
    c0 = W.mean(0)
    _, _, Vt0 = np.linalg.svd(W - c0, full_matrices=False)
    uax = (W - c0) @ Vt0[0] * MM
    if np.median(uax[aw > 0.5]) > np.median(uax[lab]):
        uax = -uax
    u_post = uax[inner].min()
    u_ring = uax[(aw > 0) & (aw < 0.99)].max()
    t2 = np.clip((u_post - uax) / max(u_post - u_ring, 1.0), 0, 1)
    bw = t2 * t2 * (3 - 2 * t2)
    bw[lab & (uax > u_post - 1e-9)] = 0.0
    if perineum_mm:
        # the perineum LONGER (user 2026-09-19: the anus "should start very close to the end of the buttocks, but right now
        # it seems to be much deeper"): the anus ring and everything behind it move back along the fourchette -> anus
        # line, the perineum stretches between (the hinge's weights: 0 at the back of the inner lips, 1 from the anus
        # ring on). Sizes stay -- the whole-donor scale would grow the anus and the lips with it. The hinge below then
        # seats the anus on the skin there.
        _L = landmarks(P)
        _dv = _L["anus"] - _L["four"]; _dv = _dv / np.linalg.norm(_dv)
        P = P + (bw * (perineum_mm / MM))[:, None] * _dv
        say("perineum %+.1f mm longer: the anus moved back along the fourchette -> anus line" % perineum_mm)
    Pr = P.copy()
    hinge = Pr[inner & (np.abs(uax - u_post) < 2.0)].mean(0)

    def hinged(th):
        qq = Pr - hinge
        ca, sa = np.cos(np.radians(th)), np.sin(np.radians(th))
        Rt = np.c_[Pr[:, 0], hinge[1] + ca * qq[:, 1] - sa * qq[:, 2], hinge[2] + sa * qq[:, 1] + ca * qq[:, 2]]
        return Pr + bw[:, None] * (Rt - Pr)

    # anus_seat_rim: seat the RIM of the opening on the skin (the soft band of the anus label, 0.3..0.99), not the whole
    # core -- the core includes the canal, and pulling all of it onto the skin laid the canal flat along it (79-99 deg to
    # the skin normal after the fit on 00004 / 00005; the bulb then came through the cleft floor as a dark spot)
    ia = np.nonzero((aw > 0.3) & (aw < 0.99))[0] if anus_seat_rim else np.nonzero(acore)[0]
    th = 0.0
    for it in range(20):
        Ph = hinged(th)
        r, m = skin_residuals(Ph, ia, skin)
        qh = Ph[ia] - hinge
        J = ((m * np.c_[np.zeros(len(qh)), -qh[:, 2], qh[:, 1]]).sum(1) * bw[ia] * np.pi / 180)[:, None]
        wv = huber_w(r, delta)
        stp = float(np.linalg.lstsq((J * wv[:, None]).T @ J + 1e-9, -(J * wv[:, None]).T @ r, rcond=None)[0][0])
        stp = float(np.clip(stp, -5, 5))
        th += stp
        if abs(stp) < 0.05:
            break
    if anus_seat == "translate":
        # Seat the anus by MOVING it onto the skin along the skin's own normal instead of swinging it about the lips.
        # With the fourchette held on the measured crotch, the rotation needed 40-43 deg on flat-floored bodies (0003:
        # 31) and folded the perineum into a protruding tube (female_elf_00005_, 197). Same weights as the hinge (0 at
        # the back of the inner lips, 1 from the anus ring on): the perineum stretches or shortens between the two.
        th = 0.0
        _, m0 = skin_residuals(Pr, ia, skin)
        nd = m0.mean(0); nd /= np.linalg.norm(nd) + 1e-30
        t = 0.0
        for it in range(30):
            Ph = Pr + (bw * t)[:, None] * nd
            r, _ = skin_residuals(Ph, ia, skin)
            stp = -float(np.median(r))
            t += stp
            if abs(stp) * MM < 0.02:
                break
        P = Pr + (bw * t)[:, None] * nd
        say("anus seated by translation: %+.1f mm along the skin normal (%d it)" % (t * MM, it + 1))
    else:
        P = hinged(th)
    if canal_inward:
        # THE CANAL INTO THE BODY (user 2026-09-19, the black spot behind female_elf_00005_'s anus): the anus lands on the
        # cleft floor, whose skin faces down and back, while the donor's canal points up in its own frame -- after the
        # seat it ran along the skin (79-99 deg to the outward normal) and half of its cup (57-66 of 122 vertices) lay
        # OUTSIDE the body, painted with the anus core colour. Turned until it points up into the pelvis -- at the centre
        # of the body's cross-section 2-3 donor widths above the opening (hip level, one closed section; the centre of the
        # surface right around the anus lies OUTSIDE, in the cleft, and turned the canal straight out).
        # The turn BENDS the tube from its mouth: pivot = the centre of the label's outer edge (anus label 0.3, where the
        # tube meets the skin and the weight is 0), weight = the label ramp 0.3 -> 0.99. The first version pivoted on the
        # centre of the soft band, halfway down the tube: its two ends swung past each other and the band crossed itself
        # (50 / 81 crossing face pairs on 00004 / 00005, shredded by the retopo into shards inside the opening); about
        # the mouth: 0 crossings, the whole cup inside the body.
        _lab = aw > 0.3; _rim = _lab & (aw < 0.99); _core = aw >= 0.99
        _mouth = np.zeros(n, bool)
        _x = _lab[Eu[:, 0]] != _lab[Eu[:, 1]]
        _mouth[Eu[_x].ravel()] = True
        _mouth &= _lab
        _wid = float(np.ptp(P[:, 0]))
        _m0 = P[_mouth].mean(0) if _mouth.any() else None
        _sl = (Vc[:, 1] > (P[_rim].mean(0)[1] + 2 * _wid)) & (Vc[:, 1] < (P[_rim].mean(0)[1] + 3 * _wid)) if _m0 is not None else None
        if _mouth.any() and _core.any() and _sl.sum() > 20:
            _a = P[_core].mean(0) - _m0; _a /= np.linalg.norm(_a)
            _pc = Vc[_sl].mean(0)
            _t = _pc - _m0; _t /= np.linalg.norm(_t)
            _ang = float(np.arccos(np.clip(_a @ _t, -1, 1)))
            _k = np.cross(_a, _t)
            if np.linalg.norm(_k) > 1e-9 and _ang > 1e-3:
                _k /= np.linalg.norm(_k)
                _u = np.clip((aw - 0.3) / 0.69, 0, 1); _wv = _u * _u * (3 - 2 * _u)
                _th = (_wv * _ang)[:, None]
                _v = P - _m0
                _rot = _v * np.cos(_th) + np.cross(_k, _v) * np.sin(_th) + np.outer(_v @ _k, _k) * (1 - np.cos(_th))
                P = _m0 + np.where((_wv > 0)[:, None], _rot, _v)
                say("canal bent %.0f deg from its mouth to point up into the pelvis (towards the body's section %.0f mm above the opening)"
                    % (np.degrees(_ang), (_pc[1] - _m0[1]) * MM))
    r, _ = skin_residuals(P, ia, skin)
    L = landmarks(P)
    seat.update(hinge=float(th), anus_gap_med=float(np.median(r) * MM))
    say("hinge %+.1f deg | anus core gap med %+.1f mm | anus %+.1f cm, fourchette %+.2f, lowest labia %+.1f, top of cleft %+.1f cm, axis tilt %.0f deg"
        % (th, seat["anus_gap_med"], cm(L["anus"][1] - A[1]), cm(L["four"][1] - A[1]), cm(L["low"][1] - A[1]), cm(L["ant"][1] - A[1]), L["tilt"]))

    # 4. apron deform: wide gate, detail = lips + anus core only
    wA, apron_w, med_e = weights(P)
    Nn = smooth_normals(P, 30)
    skin = smooth_skin(P, 20)
    gate = float(np.ptp(W[:, 0]))                           # one donor width
    t0 = signed_gap(P, Nn, skin, gate)
    hit = np.isfinite(t0)
    rr = np.where(hit, wA * t0, 0.0)
    cw = np.where(hit, 1.0, 1e-6)
    lam = (apron_w / (4 * med_e)) ** 2
    free = ~detail
    fi = np.nonzero(free)[0]
    s = np.zeros(n)
    s[fi] = _spsolve((_diags(cw[fi]) + lam * Lap[fi][:, fi]).tocsc(), cw[fi] * rr[fi])
    P1 = P + s[:, None] * Nn
    le0 = np.linalg.norm(P[Eu[:, 0]] - P[Eu[:, 1]], axis=1)
    le1 = np.linalg.norm(P1[Eu[:, 0]] - P1[Eu[:, 1]], axis=1)
    t1 = signed_gap(P1, smooth_normals(P1, 30), smooth_skin(P1, 20), gate)
    ap = free & np.isfinite(t0)
    rimm = np.zeros(n, bool)
    rimm[rim] = True

    def gs(tv, mm_):
        v = tv[mm_ & np.isfinite(tv)] * MM
        return "med %+.1f |p90| %.1f" % (np.median(v), np.percentile(np.abs(v), 90)) if len(v) else "-"

    e0, e1 = ev(P), ev(P1)
    ratio = le1 / le0
    seat.update(apron_moved_max=float(np.abs(s).max() * MM), stretched_2x=int((ratio > 2.0).sum()),
                leg_crossings=int(e1["interior"]))
    say("apron (width %.1f mm, gate %.0f mm, lambda %.1f): gap (+ sunk) rim %s -> %s | apron %s -> %s | moved max %.1f mm, detail moved %.3f | edge ratio %.2f..%.2f | leg-sheet crossings interior %d->%d, own %d->%d"
        % (apron_w * MM, gate * MM, lam, gs(t0, rimm), gs(t1, rimm), gs(t0, ap & ~rimm), gs(t1, ap & ~rimm), np.abs(s).max() * MM,
           np.abs(s[detail]).max() * MM, ratio.min(), ratio.max(), e0["interior"], e1["interior"], e0["own"], e1["own"]))

    # back into the base donor's asset space: the inverse of this body's placement affine
    Va = d["V"].astype(float)
    Minv = np.linalg.lstsq(np.c_[W_place, np.ones(n)], Va, rcond=None)[0]
    return (np.c_[P1, np.ones(n)] @ Minv).astype(d["V"].dtype), seat


def main(argv):
    src, out = argv[0], argv[1]
    z = np.load(src, allow_pickle=False)
    base = str(z["base"])
    d = np.load(base)
    Z = np.load(str(z["zones"])); aw = np.load(str(z["anus_w"]))
    if not (len(Z) == len(aw) == len(z["W"]) == len(d["V"])):
        raise SystemExit("[pfit] labels do not match the donor: zones %d, anus %d, placed %d, base %d"
                         % (len(Z), len(aw), len(z["W"]), len(d["V"])))
    env = os.environ.get
    V, seat = fit(z["cage_V"].astype(float), z["cage_F"].astype(np.int64)[:, :3], z["W"].astype(float), d, Z, aw,
                  z["A"].astype(float), float(z["MM"]),
                  back_mm=float(env("PELVIS_FIT_BACK_MM", "8.8")), tilt0=float(env("PELVIS_FIT_TILT0", "-10")),
                  fwd0=float(env("PELVIS_FIT_FWD0", "8")), axis_tilt=env("FIT8_AXIS_TILT", "27"),
                  bend_z1=env("FIT8_BEND_Z1", "1") == "1", apron_band=float(env("FIT8_APRON_BAND", "0.35")),
                  up_cm=float(env("FIT8_UP_CM", "1")), cleft_cm=float(env("PELVIS_FIT_CLEFT_CM", "9.0")),
                  cleft_ref_cm=float(env("PELVIS_FIT_CLEFT_REF_CM", "0")),
                  crotch_up_cm=float(env("PELVIS_FIT_CROTCH_UP_CM")) if env("PELVIS_FIT_CROTCH_UP_CM", "") else None,
                  anus_seat=env("PELVIS_FIT_ANUS_SEAT", "rotate"),
                  lips_floor=env("PELVIS_FIT_LIPS_FLOOR", "0") == "1",
                  scale=float(env("PELVIS_FIT_SCALE", "1") or "1"),
                  perineum_mm=float(env("PELVIS_FIT_PERINEUM_MM", "0") or "0"),
                  anus_seat_rim=env("PELVIS_FIT_ANUS_SEAT_RIM", "0") == "1",
                  canal_inward=env("PELVIS_FIT_CANAL_INWARD", "0") == "1")
    o = {k: d[k] for k in d.files}
    o["V"] = V
    np.savez(out, **o)
    say("wrote %s (seat up %+.1f mm, fwd %+.1f mm, hinge %+.1f deg)" % (os.path.basename(out), seat["up"], seat["fwd"], seat["hinge"]))


if __name__ == "__main__":
    main(sys.argv[1:])
