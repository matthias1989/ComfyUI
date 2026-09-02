"""Diagnose WHY the crown-flood over-extends: loop closure (open ends?), non-hair body fragmentation
(maskbnd uses only the largest body component), and crown/chest flood separation."""
import numpy as np, os
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

D = os.path.dirname(__file__)
m = np.load(os.path.join(D, '_hiresmesh.npz'), allow_pickle=True)
co = m['co']; fv = m['fv']; nf = len(fv); hair = m['hair'].astype(bool)
loop = np.load(os.path.join(D, '_hiresloop.npz'))['loop']

# loop topology: vertex degree
deg = defaultdict(int)
for u, w in loop:
    deg[int(u)] += 1; deg[int(w)] += 1
d1 = [v for v, d in deg.items() if d == 1]; d3 = [v for v, d in deg.items() if d >= 3]
print('loop: %d edges, %d verts; degree-1 (open ends)=%d  degree>=3 (branches)=%d' % (len(loop), len(deg), len(d1), len(d3)))

# face adjacency
e2f = defaultdict(list)
for fi, f in enumerate(fv):
    vs = [int(a) for a in f]
    for a in range(len(vs)):
        e2f[(min(vs[a], vs[(a+1) % len(vs)]), max(vs[a], vs[(a+1) % len(vs)]))].append(fi)

# non-hair body components
rr = []; cc = []
for (a, b), fs in e2f.items():
    if len(fs) == 2 and (not hair[fs[0]]) and (not hair[fs[1]]):
        rr.append(fs[0]); cc.append(fs[1])
G = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nf, nf)); G = G + G.T
ncc, lab = connected_components(G, directed=False)
sizes = np.bincount(lab[~hair])
top = np.sort(sizes)[::-1][:6]
print('non-hair components: %d total; top sizes=%s  (maskbnd uses ONLY the largest)' % (ncc, top.tolist()))

# flood from crown, loop-alone barrier; does it reach the chest?
loop_b = set((int(min(u, w)), int(max(u, w))) for u, w in loop)
adj = defaultdict(list)
for e, fs in e2f.items():
    if e in loop_b or len(fs) != 2:
        continue
    adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])
fcz = co[fv].mean(1)[:, 2]
crown = int(np.where(hair)[0][np.argmax(fcz[hair])])
def flood(s):
    seen = {s}; st = [s]
    while st:
        f = st.pop()
        for g in adj[f]:
            if g not in seen: seen.add(g); st.append(g)
    return seen
cf = flood(crown)
cfz = fcz[np.fromiter(cf, int, len(cf))]
print('crown-flood (loop-alone): %d faces  z=[%.3f..%.3f]  (crown z=%.3f)' % (len(cf), cfz.min(), cfz.max(), fcz[crown]))
print('  -> reaches below neck floor (z<0.37): %d faces' % int((cfz < 0.37).sum()))
