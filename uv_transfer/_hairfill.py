"""Compute the HAIR fill = faces enclosed by (the seam loop) + (the hair-mask OUTER boundary), flooded from
the crown. Robust: the loop closes gaps where the mask under-detects; the mask boundary closes gaps where the
loop doesn't enclose. Internal mask holes are EXCLUDED (only the body-bordering boundary is a barrier), so the
flood fills them. Env: HF_MESH (_Xmesh), HF_LOOP (_Xloop), HF_OUT (_Xfill)."""
import numpy as np, scipy.sparse as sp, os
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

m = np.load(os.environ['HF_MESH'], allow_pickle=True)
co = m['co']; fv = m['fv']; nf = len(fv); hair = m['hair'].astype(bool)
z = co[:, 2]; zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
le = np.load(os.environ['HF_LOOP'])['loop']

e2f = defaultdict(list)
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]; k = len(vs)
    for a in range(k):
        e = (min(vs[a], vs[(a + 1) % k]), max(vs[a], vs[(a + 1) % k])); e2f[e].append(fi)

# largest non-hair face component = the body/skin
rr = []; cc = []
for (a, b), fs in e2f.items():
    if len(fs) == 2 and (not hair[fs[0]]) and (not hair[fs[1]]):
        rr.append(fs[0]); cc.append(fs[1])
G = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nf, nf)); G = G + G.T
_, lab = connected_components(G, directed=False)
bodyc = int(np.bincount(lab[~hair]).argmax())

loop_b = set((int(min(u, w)), int(max(u, w))) for u, w in le)
# NECK BARRIER: the head/body split is the neck floor. Only HAIR (the drape) may cross it; the bare
# body must not. The front neck (below the chin) has no hair, so neither the loop nor the mask boundary
# closes it -> a crown-flood escapes down the front of the neck to the shoulders. Block every edge that
# crosses the neck floor whose two faces are NOT both hair (i.e. body crossing the neck). The drape
# (hair-to-hair across the floor) stays open. Principled: head/body boundary, hair-gated.
_lz = np.load(os.environ['HF_LOOP'])
neckbar = set()
if 'neck_floor' in _lz.files:
    _floor = float(_lz['neck_floor'])
    for (a, b), fs in e2f.items():
        if len(fs) == 2 and ((zn[a] < _floor) != (zn[b] < _floor)) and not (hair[fs[0]] and hair[fs[1]]):
            neckbar.add((a, b))
    print('neck barrier: floor_znv=%.3f  %d body edges blocked at the neck' % (_floor, len(neckbar)))
# FULL hair-skin boundary (every hair<->non-hair edge). The OLD code barriered only the LARGEST body
# component, but the (nearest-face) transferred mask is holey -> the body shatters into ~30k components,
# so most of the real hairline/drape/nape boundary was NOT barriered and the crown-flood escaped to the
# chest. Barriering the FULL boundary caps the flood inside the hair; internal holes are refilled by the
# pocket pass below (which floods the body WITHOUT the neck barrier, so the face is reached, not filled).
maskbnd = set()
for (a, b), fs in e2f.items():
    if len(fs) == 2 and (bool(hair[fs[0]]) != bool(hair[fs[1]])):
        maskbnd.add((a, b))

fcz = np.array([zn[[int(a) for a in f]].mean() for f in fv])
# HAIR COMPONENTS (used for the seed AND for adding disconnected ponytail pieces at the end).
_hr = []; _hc = []
for fs in e2f.values():
    if len(fs) == 2 and hair[fs[0]] and hair[fs[1]]:
        _hr.append(fs[0]); _hc.append(fs[1])
_Ahair = sp.csr_matrix((np.ones(len(_hr)), (_hr, _hc)), shape=(nf, nf)); _Ahair = _Ahair + _Ahair.T
nch, lch = connected_components(_Ahair, directed=False)
# SEED = topmost face of the LARGEST hair component (the crown blob). NOT the topmost hair face overall:
# that can be a small top speck / ear bit in its own tiny component, which would make the crown-flood
# tiny and the real crown get added (un-refined) as a "disconnected piece", bypassing the loop.
if hair.any():
    _crownc = int(np.bincount(lch[hair], minlength=nch).argmax())
    _cf = np.where(hair & (lch == _crownc))[0]
    seed = int(_cf[np.argmax(fcz[_cf])])
else:
    _crownc = -1; seed = int(np.argmax(fcz))

def adjof(barrier):
    adj = defaultdict(list)
    for e, fs in e2f.items():
        if e in barrier or len(fs) != 2:
            continue
        adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])
    return adj

def flood(adj, seeds):
    seen = set(seeds); st = list(seeds)
    while st:
        f = st.pop()
        for g in adj[f]:
            if g not in seen:
                seen.add(g); st.append(g)
    return seen

# CROWN-FLOOD bounded by the loop + the FULL hair-mask boundary + the neck barrier. With the full mask
# boundary the flood is physically confined to the hair (no chest/shoulder leak). Going DOWN from the
# crown it stops at whichever is higher -- the loop (mask over-detected the front) or the mask edge (mask
# under-detected); under-detection is then recovered by the DILATE-to-loop pass below.
bar = loop_b | maskbnd | neckbar; adj = adjof(bar)
filled = flood(adj, [seed]); mode = 'loop+maskbnd+neck'
n0 = len(filled)
# ENCLOSED POCKETS: the holey transferred mask leaves little non-hair islands inside the hair. They are
# walled off from BOTH the crown-flood and the body. Flood from the body (lowest face) -- WITHOUT the
# neck barrier, so the body-flood reaches the face/forehead skin through the neck and does NOT mistake it
# for a pocket. Any face neither flood reaches is an enclosed hole on the hair side -> fill it.
npock = 0
if int(os.environ.get('HF_POCKETS', '1')):
    body_seed = int(np.argmin(fcz))
    outside = flood(adjof(loop_b | maskbnd), [body_seed])   # NO neck barrier -> reaches the face via the neck
    pockets = [f for f in range(nf) if f not in filled and f not in outside]
    for f in pockets:
        filled.add(f)
    npock = len(pockets)
# CLOSE THE WHITE BAND: the flood stops at the mask boundary, which sits INSIDE the green line (the mask
# under-detects). Dilate the fill outward UP TO the green loop -- blocked by loop edges only (so it crosses
# the mask boundary into the white band and reaches the green, but can't cross the green). Bounded ring
# count, so any leak through a non-enclosing stretch of loop is limited.
DIL = int(os.environ.get('HF_DILATE', '8'))
if DIL > 0:
    adjL = defaultdict(list)
    for e, fs in e2f.items():
        if e in loop_b or e in neckbar or len(fs) != 2:
            continue
        adjL[fs[0]].append(fs[1]); adjL[fs[1]].append(fs[0])
    n1 = len(filled); cur = set(filled)
    for _ in range(DIL):
        nxt = set()
        for f in cur:
            for g in adjL[f]:
                if g not in filled:
                    filled.add(g); nxt.add(g)
        cur = nxt
        if not cur:
            break
# DISCONNECTED HAIR PIECES: the reconstruction often splits a long ponytail/drape from the crown by a
# gap, so the crown-flood can't reach it. Add every hair component OTHER than the crown's. ALSO re-add
# crown-component hair the loop cut off from the flood that sits at the BACK (the drape continuation,
# y>median): per-bake the path3 loop can stop short of the drape bottom, which otherwise drops the whole
# lower back drape ("the side completely missing"). The over-detected FOREHEAD (front, y<median) stays
# excluded -- the loop correctly cuts it off there. Depth-gated, no magic height.
_fm = np.zeros(nf, bool); _fm[list(filled)] = True
_ymed = float(np.median(co[:, 1])); _fyc = co[fv].mean(1)[:, 1]
_discon = np.where(hair & ~_fm & ((lch != _crownc) | (_fyc > _ymed)))[0]
for f in _discon:
    filled.add(int(f))
print('disconnected + back-drape hair added: %d faces' % len(_discon))
np.savez(os.environ['HF_OUT'], faces=np.array(sorted(filled), dtype=np.int64))
print('hairfill: %d faces [%s]  flood=%d +pockets=%d +dilate=%d  (of %d, loop=%d, mask-bnd=%d)' % (
    len(filled), mode, n0, npock, len(filled) - n0 - npock, nf, len(le), len(maskbnd)))
