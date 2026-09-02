"""Map gb-order debug data (facemask/concave/kept) onto the DEV blend vertices by POSITION
(normalize both, nearest-neighbour) so the painted result is correct regardless of vertex reordering.
Run with python_embeded (scipy). Needs _dev_co.npy (exported from the DEV blend)."""
import numpy as np, os
from scipy.spatial import cKDTree
UV = os.path.dirname(os.path.abspath(__file__))
gco = np.load(UV+'/_gb_mesh.npz')['co']
dco = np.load(UV+'/_dev_co.npy')
def nz(a):
    mn = a.min(0); return (a-mn)/(a.max(0)-mn+1e-9)
d, i = cKDTree(nz(gco)).query(nz(dco))                 # DEV vertex -> nearest gb vertex
print('DEV<-gb map: meandist %.5f max %.5f  nDEV=%d ngb=%d' % (float(d.mean()), float(d.max()), len(dco), len(gco)))
fm   = np.load(UV+'/_dev_facevert_gb.npy').astype(bool)
kept = np.load(UV+'/_gb_kept.npy').astype(bool)
ed   = np.load(UV+'/_gb_crease_all.npz')['edges']
con  = np.zeros(len(gco), bool); con[np.unique(ed.reshape(-1))] = True
np.save(UV+'/_dev_fm.npy',   fm[i])
np.save(UV+'/_dev_con.npy',  con[i])
np.save(UV+'/_dev_kept.npy', kept[i])
print('mapped onto DEV: facemask %d, concave %d, kept %d' % (int(fm[i].sum()), int(con[i].sum()), int(kept[i].sum())))
