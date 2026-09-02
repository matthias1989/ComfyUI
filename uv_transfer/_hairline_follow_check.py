"""HAIRLINE-FOLLOW REGRESSION TRIPWIRE.

Measures how much of the hair-drape hairline the path3 loop actually rides (total + per side), so a
regression is caught the MOMENT it happens instead of weeks later. The absolute % is not a quality score
(the rough-hair reference is noisy) -- it is a DELTA detector: baseline it on a bake you've confirmed
good, then every bake compares and screams if coverage drops.

  Set baseline (run right after a GOOD bake):   FOLLOW_SETBASE=1 python _hairline_follow_check.py
  Check (every bake, or wired after path3):                       python _hairline_follow_check.py

Reads the per-bake files path3 already writes (_gb_mesh.npz, _gb_loop.npz). No symmetry assumption: L and
R are reported independently so a genuinely lopsided mesh is fine -- only a DROP vs its own baseline flags.
"""
import numpy as np, collections, os, json
from scipy.spatial import cKDTree

UV = os.path.dirname(os.path.abspath(__file__))
DROP = float(os.environ.get('FOLLOW_DROP', '8'))   # percentage-points drop that counts as a regression


def measure():
    gm = np.load(os.path.join(UV, '_gb_mesh.npz')); co = gm['co'].astype(float)
    fv = gm['fv'].astype(np.int64); hair = gm['hair'].astype(bool)
    loop = np.load(os.path.join(UV, '_gb_loop.npz'))['loop']
    znv = (co[:, 2] - co[:, 2].min()) / (np.ptp(co[:, 2]) + 1e-9)
    xc = 0.5 * (co[:, 0].min() + co[:, 0].max()); hw = 0.5 * (co[:, 0].max() - co[:, 0].min())
    e2f = collections.defaultdict(list)
    for fi in range(len(fv)):
        a, b, c = fv[fi]
        for u, w in ((a, b), (b, c), (c, a)):
            e2f[(min(u, w), max(u, w))].append(fi)
    bnd = set()
    for (u, w), fs in e2f.items():
        if len(fs) == 2 and (hair[fs[0]] != hair[fs[1]]):
            bnd.add(u); bnd.add(w)
    bnd = np.array(sorted(bnd))
    # the DRAPE hairline: head band below the crown, within head width (drop the elf ear tips at |x|>0.16)
    sel = bnd[(znv[bnd] > 0.68) & (znv[bnd] < 0.88) & (np.abs((co[bnd, 0] - xc) / hw) < 0.16)]
    el = np.median([np.linalg.norm(co[u] - co[w]) for (u, w) in list(e2f)[:5000]])
    T = cKDTree(co[np.unique(loop)])
    def pc(v):
        if len(v) == 0:
            return 0.0
        d, _ = T.query(co[v]); return round(100 * float((d < 3 * el).mean()), 1)
    L = sel[(co[sel, 0] - xc) < 0]; R = sel[(co[sel, 0] - xc) > 0]
    return {'total': pc(sel), 'left': pc(L), 'right': pc(R), 'n': int(len(sel))}


if __name__ == '__main__':
    m = measure()
    base_p = os.path.join(UV, '_follow_baseline.json')
    if os.environ.get('FOLLOW_SETBASE'):
        json.dump(m, open(base_p, 'w'))
        print('[follow-check] BASELINE SET -> total %(total)s%% (L %(left)s%% R %(right)s%%) of %(n)s verts' % m)
    else:
        print('[follow-check] hairline follow: total %(total)s%% (L %(left)s%% R %(right)s%%) of %(n)s drape verts' % m)
        if os.path.exists(base_p):
            b = json.load(open(base_p)); hit = False
            for k in ('total', 'left', 'right'):
                if m[k] < b[k] - DROP:
                    print('  *** REGRESSION: %s follow %.0f%% << baseline %.0f%% (dropped %.0f pts) ***' % (k, m[k], b[k], b[k] - m[k]))
                    hit = True
            if not hit:
                print('  [follow-check] OK -- no regression vs baseline (total base %.0f%%)' % b['total'])
        else:
            print('  [follow-check] no baseline yet -- run FOLLOW_SETBASE=1 after a bake you confirm GOOD')
