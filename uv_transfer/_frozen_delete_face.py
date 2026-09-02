"""FACE FIX (the simple one that works): after the seam is drawn, delete any seam edge that touches the
face mask. The part that dipped INTO the face goes; the forehead hairline arc (which sits above the mask)
stays. Run AFTER path3_v2 writes _frozen/loop.npz, BEFORE painting.
Inputs: _frozen/loop.npz + _frozen/facevert.npy.  Overwrites _frozen/loop.npz."""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "_frozen", "loop.npz")); loop = d['loop']; anc = d['anchors']; nf = d['neck_floor']
fm = np.load(os.path.join(HERE, "_frozen", "facevert.npy")).astype(bool)
keep = ~(fm[loop[:, 0]] | fm[loop[:, 1]])          # drop any edge with a vertex inside the face mask
np.savez(os.path.join(HERE, "_frozen", "loop.npz"), loop=loop[keep], anchors=anc, neck_floor=nf)
print('[face-delete] on-face edges removed: %d -> %d edges' % (len(loop), int(keep.sum())))
