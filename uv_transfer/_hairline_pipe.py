"""Pipeline hairline seam (KAN-9) -- the v8 chain, living IN the pipeline so what's perfected here is
exactly what a bake produces (no more offline copies that never get used).

  postfix_loop()   -- BRIDGE side (python_embeded; has scipy): reroute the path3 loop's face-crossings
                      AROUND the face -> _gb_loop_fix.npz. Turns the band-across-the-eyes into a hairline.
  mark_loop_seam() -- gen_seams side (Blender; numpy/bpy): mark _gb_loop_fix as use_seam + bridge quad
                      diagonals, un-mark the front-neck band, prune spurs / de-double. Replaces the old
                      is_hair-boundary seam. Touches ONLY the hairline loop edges (body seams untouched).

Both read the per-run files the bridge/gen_seams already write, so it's source-agnostic and dynamic."""
import numpy as np, os
from collections import defaultdict, deque


def _face_mask(co):
    """Geometric FACE region on the path3 mesh (z-up: X=lateral, Y=depth front=-Y, Z=height): the
    front-central head BELOW the hairline (eyes/nose/mouth). Routing excludes it so the loop is forced
    AROUND the face instead of across it. Head-relative (z-norm + head bbox), no absolute magic values."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    zn = (z - z.min()) / (z.max() - z.min())
    hd = zn > 0.84
    xc = x[hd].mean(); xsp = x[hd].max() - x[hd].min(); ym = np.median(y[hd])
    # CENTRAL only: the across-face crossing spans the face centre; the side hairline runs at the temples
    # (larger |x|). A wide mask catches the temples and fragments the seam -> keep it central. Head-relative.
    return (zn > 0.86) & (zn < 0.93) & (y < ym) & (np.abs(x - xc) < float(os.environ.get('GS_FACEMASK_XFRAC', '0.20')) * xsp)


def _bridge_face_gap(co, fv, faceb, edges, full_edges):
    """The face-delete can SPLIT the loop when the only left<->right bridge ran across the forehead THROUGH
    the mask (per-bake: happens when path3's forehead arc dips into the face instead of sitting above it).
    Reconnect the cut ends along the shortest mesh path that AVOIDS the face mask -> the gap closes over the
    brow at hairline height, never back through the face. Only fires when the delete actually disconnected
    the loop; fully geometric/dynamic (shortest path can't balloon over the crown)."""
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import dijkstra as _dij
    from collections import defaultdict as _dd, deque as _dq
    edges = [tuple(map(int, e)) for e in edges]
    if len(edges) < 2:
        return edges, 0

    def _comps(E):
        adj = _dd(set)
        for u, w in E:
            adj[u].add(w); adj[w].add(u)
        seen = set(); cs = []
        for s in list(adj):
            if s in seen:
                continue
            q = _dq([s]); seen.add(s); c = [s]
            while q:
                v = q.popleft()
                for x in adj[v]:
                    if x not in seen:
                        seen.add(x); q.append(x); c.append(x)
            cs.append(c)
        return cs, adj

    cs, _ = _comps(edges)
    if len(cs) <= 1:
        return edges, 0
    # mesh graph with FACE verts forbidden, so ANY bridge is forced to route around the face
    nv = len(co); rows = []; cols = []; wts = []; seen = set()
    for tri in fv:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, w in ((a, b), (b, c), (c, a)):
            k = (u, w) if u < w else (w, u)
            if k in seen:
                continue
            seen.add(k)
            if faceb[u] or faceb[w]:
                continue
            d = float(((co[u] - co[w]) ** 2).sum() ** 0.5); rows += [u, w]; cols += [w, u]; wts += [d, d]
    G = _sp.csr_matrix((wts, (rows, cols)), shape=(nv, nv))
    full = set(tuple(sorted(map(int, e))) for e in full_edges)
    kept = set(tuple(sorted(e)) for e in edges)
    delv = set()
    for u, w in (full - kept):
        delv.add(u); delv.add(w)
    nbridge = 0
    for _ in range(8):
        cs, adj = _comps(edges)
        if len(cs) <= 1:
            break
        deg1 = [v for v in adj if len(adj[v]) == 1 and not faceb[v]]
        ends = [v for v in deg1 if v in delv] or deg1     # prefer the cut ends; else any open end
        if len(ends) < 2:
            break
        cid = {}
        for i, c in enumerate(cs):
            for v in c:
                cid[v] = i
        dist, pred = _dij(G, indices=ends, return_predecessors=True, min_only=False)
        best = None
        for si, s in enumerate(ends):
            for t in ends:
                if cid.get(s) == cid.get(t):
                    continue
                dd = dist[si, t]
                if dd < 1e18 and (best is None or dd < best[0]):
                    best = (dd, si, s, t)
        if best is None:
            break
        _, si, s, t = best
        cur = t; path = [t]
        while cur != s and cur >= 0:
            cur = pred[si, cur]; path.append(cur)
        if cur < 0:
            break
        for i in range(len(path) - 1):
            edges.append((path[i], path[i + 1]))
        nbridge += len(path) - 1
    return edges, nbridge


def _back_fold_edges(co, fv, loop, hairf):
    """2-POINT back seam (the user's recipe): place 2 anchor points on the BACK (one per side) at height
    GS_BACKSEAM_Z (znv -- the one knob), pathfind a line between them across the back (back-only graph so it
    can't cut through the front), then pathfind each anchor out to the side seams (the loop) to close the hair
    island. Marked SEPARATELY after the loop, so it can never erase the real seams. No surface-hunting."""
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import dijkstra as _dij
    loop = np.asarray(loop)
    if not len(loop):
        return np.zeros((0, 2), np.int64)
    nv = len(co); znv = (co[:, 2] - co[:, 2].min()) / (np.ptp(co[:, 2]) + 1e-9)
    ymed = float(np.median(co[:, 1])); xc = 0.5 * (float(co[:, 0].min()) + float(co[:, 0].max()))
    ZT_target = float(os.environ.get('GS_BACKSEAM_Z', '0.67'))            # PREFERRED height; ADAPTS below
    # Anchors = the two SIDE SEAMS themselves (loop verts on the BACK), one per side. Using the loop -- not
    # the hair -- guarantees a LEFT and a RIGHT anchor (the hair can be gappy/asymmetric per bake, which
    # gave a half-seam). ADAPTIVE height: pick the band CLOSEST to the preferred height where the loop has a
    # vert on BOTH sides of centre, so the line can actually cross. Dynamic -> robust to every bake.
    lvb = np.unique(loop); lvb = lvb[co[lvb, 1] > ymed]                   # the loop's BACK part = drape side seams
    if len(lvb) < 2:
        return np.zeros((0, 2), np.int64)
    lz = znv[lvb]; lx = co[lvb, 0] - xc; _best = None
    for _zc in np.arange(0.40, 0.96, 0.01):
        _m = np.abs(lz - _zc) < 0.04
        if int(_m.sum()) < 2:
            continue
        if lx[_m].min() < 0.0 and lx[_m].max() > 0.0:                    # side seam present LEFT and RIGHT
            _d = abs(_zc - ZT_target)
            if _best is None or _d < _best[0]:
                _best = (_d, float(_zc))
    if _best is None:
        return np.zeros((0, 2), np.int64)
    ZT = _best[1]; _m = np.abs(lz - ZT) < 0.04; _bl = lvb[_m]
    pL = int(_bl[int(np.argmin(co[_bl, 0]))]); pR = int(_bl[int(np.argmax(co[_bl, 0]))])   # left & right side seam
    print('[hairline-pipe] back-seam adaptive height: target znv %.2f -> %.2f (L x/2=%+.3f R x/2=%+.3f)'
          % (ZT_target, ZT, co[pL, 0] - xc, co[pR, 0] - xc))
    E = np.unique(np.sort(np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], 0), 1), axis=0)
    Lw = np.linalg.norm(co[E[:, 0]] - co[E[:, 1]], axis=1)
    bm = (co[E[:, 0], 1] > ymed) & (co[E[:, 1], 1] > ymed)                 # back-only graph -> can't cut through the front
    Es = E[bm]; Ls = Lw[bm]
    rr = np.concatenate([Es[:, 0], Es[:, 1]]); cc = np.concatenate([Es[:, 1], Es[:, 0]]); ww = np.concatenate([Ls, Ls])
    Gb = _sp.csr_matrix((ww, (rr, cc)), shape=(nv, nv))
    d, pr = _dij(Gb, indices=[pL], return_predecessors=True)
    if not np.isfinite(d[0, pR]):
        return np.zeros((0, 2), np.int64)
    cur = pR; path = [cur]
    while cur != pL and cur >= 0:
        cur = int(pr[0, cur]); path.append(cur)
    out = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return np.array(out, np.int64) if out else np.zeros((0, 2), np.int64)


def postfix_loop(uv_dir, loop_name='_gb_loop.npz', mesh_name='_gb_mesh.npz', out_name='_gb_loop_fix.npz'):
    """v2 (user-approved): DELETE the loop's face-crossing edges (drop any edge touching the face mask).
    The path3_v2 loop is already the clean OPEN hairline down both sides; the only bad part is where a
    connector shortcuts ACROSS the face -- deleting it leaves the forehead hairline arc intact. (The earlier
    reroute-AROUND-the-face approach ballooned over the crown / tangled the temple and was rejected.)"""
    gm = np.load(os.path.join(uv_dir, mesh_name)); co = gm['co'].astype(np.float64)
    fv = gm['fv'].astype(np.int64) if 'fv' in gm.files else None
    dl = np.load(os.path.join(uv_dir, loop_name)); loopE = dl['loop']
    # prefer the MediaPipe face mask (_gs_facemask.py -> identical to the offline-approved result); the
    # geometric box is the fallback only (too coarse: it catches the temples and fragments the seam).
    _fvp = os.path.join(uv_dir, '_gb_facevert.npy')
    if os.path.exists(_fvp):
        _fm = np.load(_fvp).astype(bool)
        faceb = _fm if len(_fm) == len(co) else _face_mask(co)
        print('[hairline-pipe] postfix using MediaPipe face mask (%d verts)' % int(faceb.sum()))
    else:
        faceb = _face_mask(co); print('[hairline-pipe] postfix using GEOMETRIC face mask (MediaPipe missing)')
    if len(loopE):
        keep = np.array([not (faceb[int(u)] or faceb[int(w)]) for u, w in loopE], dtype=bool)
        ne = loopE[keep]; ndel = int((~keep).sum())
    else:
        ne = loopE; ndel = 0
    # CLOSE the gap the delete left: if it split the loop (the forehead bridge ran through the mask), rejoin
    # the cut ends along the shortest FACE-AVOIDING mesh path -> 1 connected loop, over the brow, off the face.
    nbridge = 0
    if fv is not None and len(ne):
        ne_list, nbridge = _bridge_face_gap(co, fv, faceb, ne, loopE)
        ne = np.array(ne_list, dtype=np.int64) if len(ne_list) else np.zeros((0, 2), np.int64)
    # CLOSE the FOREHEAD GAP: deleting the across-face connector can leave TWO open ends up at the forehead
    # (where the straight connector was) that the face-avoiding bridge won't join -- they're already one
    # component (linked the long way over the crown), so it skips them, leaving the hairline open across the
    # front. Join them with a hairline ARC routed along the brow: forbid only the face BELOW their level, so
    # the path runs JUST ABOVE the face, not across it. Relative (loop znv midpoint), no magic height.
    if fv is not None and len(ne) >= 2:
        import scipy.sparse as _sp2
        from scipy.sparse.csgraph import dijkstra as _dij2
        _adjF = defaultdict(set)
        for _u, _w in ne:
            _adjF[int(_u)].add(int(_w)); _adjF[int(_w)].add(int(_u))
        _zc = (co[:, 2] - co[:, 2].min()) / (np.ptp(co[:, 2]) + 1e-9)
        _lvv = np.unique(ne); _mid = 0.5 * (float(_zc[_lvv].min()) + float(_zc[_lvv].max()))
        _fe2 = sorted([v for v in _adjF if len(_adjF[v]) == 1 and _zc[v] > _mid], key=lambda v: -_zc[v])
        if len(_fe2) >= 2:
            _e1, _e2 = int(_fe2[0]), int(_fe2[1]); _lvl = min(float(_zc[_e1]), float(_zc[_e2]))
            _forbid = faceb & (_zc < _lvl - 0.005)                # brow level: face below the ends is off-limits
            _r = []; _c = []; _wt = []; _sn = set()
            for _t in fv:
                _a, _b, _cc = int(_t[0]), int(_t[1]), int(_t[2])
                for _u, _v in ((_a, _b), (_b, _cc), (_cc, _a)):
                    _k = (_u, _v) if _u < _v else (_v, _u)
                    if _k in _sn:
                        continue
                    _sn.add(_k)
                    if _forbid[_u] or _forbid[_v]:
                        continue
                    _d = float(((co[_u] - co[_v]) ** 2).sum() ** 0.5)
                    _r += [_u, _v]; _c += [_v, _u]; _wt += [_d, _d]
            _G2 = _sp2.csr_matrix((_wt, (_r, _c)), shape=(len(co), len(co)))
            _dist, _pred = _dij2(_G2, indices=[_e1], return_predecessors=True)
            if np.isfinite(_dist[0, _e2]):
                _p = [_e2]; _cur = _e2
                while _cur != _e1 and _cur >= 0:
                    _cur = int(_pred[0, _cur]); _p.append(_cur)
                _arc = [(_p[i], _p[i + 1]) for i in range(len(_p) - 1)]
                if _arc:
                    ne = np.vstack([ne, np.array(_arc, np.int64)])
                    print('[hairline-pipe] closed forehead gap: +%d arc edges along the brow' % len(_arc))
    # BACK-FOLD: compute the concave under-hair fold here but save it SEPARATELY (gen_seams marks it AFTER
    # mark_loop_seam, so its de-double/prune can never erase the real seams). Not added to the loop.
    _foldp = os.path.join(uv_dir, '_gb_fold.npz')
    try:
        _flp = os.path.join(uv_dir, '_gb_fill.npz')
        if fv is not None and len(ne) and os.path.exists(_flp):
            _ff = np.load(_flp)['faces']; _hf = np.zeros(len(fv), bool); _hf[_ff[_ff < len(fv)]] = True
            _fe = _back_fold_edges(co, fv, ne, _hf)
            np.savez(_foldp, edges=_fe)
            print('[hairline-pipe] back-seam (2-point): %d edges -> _gb_fold.npz (marked separately, after the loop)' % len(_fe))
        elif os.path.exists(_foldp):
            os.remove(_foldp)
    except Exception as _bfe:
        print('[hairline-pipe] back-seam skipped (%r)' % _bfe)
    np.savez(os.path.join(uv_dir, out_name),
             loop=np.array(ne, dtype=np.int64) if len(ne) else np.zeros((0, 2), np.int64),
             anchors=dl['anchors'], neck_floor=dl['neck_floor'])
    print('[hairline-pipe] postfix: DELETED %d face-crossing edges, BRIDGED %d to reclose -> %s (%d edges, face mask %d verts)'
          % (ndel, nbridge, out_name, len(ne), int(faceb.sum())))
    return len(ne)


def mark_loop_seam(me, eidx, uv_dir):
    """Mark the post-fixed loop as use_seam (+ bridge quad-diagonals), un-mark the front-neck band, then
    prune short spurs / de-double -- all restricted to the hairline loop (body seams untouched).
    me = bpy mesh, eidx = {(vmin,vmax): edge_index}. Returns the final hairline seam-key set."""
    lp = os.path.join(uv_dir, '_gb_loop_fix.npz')
    if not os.path.exists(lp):
        lp = os.path.join(uv_dir, '_gb_loop.npz')   # fallback: raw loop (face-crossing NOT removed)
        print('[hairline-pipe] WARNING: _gb_loop_fix missing, using raw loop (face may be crossed)')
    d = np.load(lp); loopE = d['loop']
    neck_floor = float(d['neck_floor']) if 'neck_floor' in d.files else 0.858
    co = np.array([v.co[:] for v in me.vertices])
    vadj = defaultdict(set)
    for e in me.edges:
        vadj[int(e.vertices[0])].add(int(e.vertices[1])); vadj[int(e.vertices[1])].add(int(e.vertices[0]))
    loopset = set(); nmark = 0; nbridge = 0
    for u, w in loopE:
        u, w = int(u), int(w); k = (u, w) if u < w else (w, u)
        ei = eidx.get(k)
        if ei is not None:
            me.edges[ei].use_seam = True; loopset.add(k); nmark += 1
        else:
            common = vadj[u] & vadj[w]
            if common:
                mid = min(common, key=lambda m: float((me.vertices[u].co - me.vertices[m].co).length
                                                       + (me.vertices[m].co - me.vertices[w].co).length))
                for a, b in ((u, mid), (mid, w)):
                    kk = (a, b) if a < b else (b, a); ee = eidx.get(kk)
                    if ee is not None:
                        me.edges[ee].use_seam = True; loopset.add(kk)
                nbridge += 1
    # ── un-mark the FRONT-NECK band (neck_floor-relative, per-character) so the neck is its own island ──
    zmn, zmx = co[:, 2].min(), co[:, 2].max(); zn = (co[:, 2] - zmn) / (zmx - zmn)
    xc = float((co[:, 0].min() + co[:, 0].max()) / 2.0); ym = float((co[:, 1].min() + co[:, 1].max()) / 2.0)
    zlo, zhi, xw = neck_floor - 0.14, neck_floor + 0.05, 0.03
    nband = 0
    for k in list(loopset):
        a, b = k
        mx = abs((co[a, 0] + co[b, 0]) / 2.0 - xc); mz = (zn[a] + zn[b]) / 2.0; my = (co[a, 1] + co[b, 1]) / 2.0
        if mx < xw and zlo < mz < zhi and my < ym:
            ei = eidx.get(k)
            if ei is not None:
                me.edges[ei].use_seam = False; loopset.discard(k); nband += 1
    # ── clean: de-double (drop edge if a short alt path remains) + prune short spurs -- loop-only ──
    adj = defaultdict(set)
    for (a, b) in loopset:
        adj[a].add(b); adj[b].add(a)
    ndd = 0
    for (u, w) in sorted(loopset):
        if w not in adj[u]:
            continue
        adj[u].discard(w); adj[w].discard(u)
        dq = deque([(u, 0)]); seen = {u}; found = False
        while dq:
            cur, dd = dq.popleft()
            if cur == w:
                found = True; break
            if dd >= 4:
                continue
            for x in adj[cur]:
                if x not in seen:
                    seen.add(x); dq.append((x, dd + 1))
        if found:
            ndd += 1
        else:
            adj[u].add(w); adj[w].add(u)
    nsp = 0; changed = True
    while changed:
        changed = False
        for v in [x for x in list(adj) if len(adj[x]) == 1]:
            chain = [v]; cur = v; prev = None
            while True:
                nxt = [x for x in adj[cur] if x != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]; chain.append(cur)
                if len(adj[cur]) != 2:
                    break
            if (len(chain) - 1) <= 12 and (not adj[chain[-1]] or len(adj[chain[-1]]) >= 3):
                for i in range(len(chain) - 1):
                    adj[chain[i]].discard(chain[i + 1]); adj[chain[i + 1]].discard(chain[i])
                nsp += 1; changed = True
    cleaned = set()
    for a in adj:
        for b in adj[a]:
            cleaned.add((a, b) if a < b else (b, a))
    for k in list(loopset):
        if k not in cleaned:
            ei = eidx.get(k)
            if ei is not None:
                me.edges[ei].use_seam = False
    print('[hairline-pipe] seam: marked %d + bridged %d, un-marked %d neck-band, de-doubled %d, '
          'pruned %d spurs -> %d hairline edges' % (nmark, nbridge, nband, ndd, nsp, len(cleaned)))
    return cleaned
