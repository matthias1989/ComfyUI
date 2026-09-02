"""Quick 2D look at the high-res green-border result (debug only). Loop drawn as LINES. Zoom to head."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

D = os.path.dirname(__file__)
m = np.load(os.path.join(D, '_hiresmesh.npz'), allow_pickle=True)
co = m['co']; fv = m['fv']; hair = m['hair'].astype(bool)
loop = np.load(os.path.join(D, '_hiresloop.npz'))['loop']
fill = np.load(os.path.join(D, '_hiresfill.npz'))['faces']
fillm = np.zeros(len(fv), bool); fillm[fill[fill < len(fv)]] = True
fcen = co[fv].mean(1)

def panel(ax, a0, a1, title):
    sel = co[:, 2] > 0.20
    ax.scatter(co[sel, a0], co[sel, a1], s=0.5, c='lightgray', lw=0)
    fs = fillm & (fcen[:, 2] > 0.20)
    ax.scatter(fcen[fs, a0], fcen[fs, a1], s=1.2, c='gold', lw=0)
    hs = hair & (fcen[:, 2] > 0.20)
    ax.scatter(fcen[hs, a0], fcen[hs, a1], s=0.7, c='#9cf', lw=0, alpha=0.5)
    segs = [[(co[u, a0], co[u, a1]), (co[w, a0], co[w, a1])] for u, w in loop]
    ax.add_collection(LineCollection(segs, colors='green', linewidths=1.0))
    ax.set_title(title); ax.set_aspect('equal')

fig, axs = plt.subplots(1, 2, figsize=(13, 8))
panel(axs[0], 0, 2, 'FRONT (x,z)  green=loop LINES  gold=fill  blue=hairmask')
panel(axs[1], 1, 2, 'RIGHT (y,z)  front=left(min y)')
plt.tight_layout(); out = os.path.join(D, '_hires_look.png'); plt.savefig(out, dpi=95)
print('fill=%d -> %s' % (int(fillm.sum()), out))
