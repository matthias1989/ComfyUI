"""POST-FIX the path3 loop for the face: instead of DELETING the edges that cross the MediaPipe face
region (which leaves a gap where the loop crossed the face), RECONNECT each crossing's boundary verts
via a path that routes AROUND the face (Dijkstra on the mesh graph with face verts excluded). This
removes the nose crossing AND keeps the loop closed, forcing the seam down the sides instead of across
the face. Env: PF_LOOP (in), PF_FACE (per-vert face npy), PF_MESH (_gb_mesh), PF_OUT."""
import numpy as np, scipy.sparse as sp, os
from scipy.sparse.csgraph import dijkstra

UV = os.path.dirname(os.path.abspath(__file__))
m = np.load(os.environ.get('PF_MESH', os.path.join(UV, '_gb_mesh.npz')))
co = m['co'].astype(np.float64); fv = m['fv']; nv = int(m['nv'])
faceb = np.load(os.environ.get('PF_FACE', os.path.join(UV, '_faceverts2.npy'))).astype(bool)
d = np.load(os.environ.get('PF_LOOP', os.path.join(UV, '_test_loop.npz')))
loop = d['loop']

# ordered cyclic loopverts
verts = [int(loop[0][0])] + [int(e[1]) for e in loop]
if verts[-1] == verts[0]:
    verts = verts[:-1]
N = len(verts)

# around-face graph: mesh edges with face verts EXCLUDED so paths cannot cross the face
seen = set(); rows = []; cols = []; wts = []
for fi in range(len(fv)):
    a, b, c = int(fv[fi][0]), int(fv[fi][1]), int(fv[fi][2])
    for u, w in ((a, b), (b, c), (c, a)):
        k = (u, w) if u < w else (w, u)
        if k in seen:
            continue
        seen.add(k)
        if faceb[u] or faceb[w]:
            continue
        L = float(np.linalg.norm(co[u] - co[w])); rows += [u, w]; cols += [w, u]; wts += [L, L]
G = sp.csr_matrix((wts, (rows, cols)), shape=(nv, nv))

inface = np.array([faceb[v] for v in verts])
if inface.all():
    raise SystemExit('[postfix] entire loop in face?!')
st = int(np.where(~inface)[0][0]); verts = verts[st:] + verts[:st]
inface = np.array([faceb[v] for v in verts])

def around(a, b):
    dist, pred = dijkstra(G, indices=a, return_predecessors=True)
    if a != b and pred[b] < 0:
        return [a, b]
    p = [b]
    while p[-1] != a and p[-1] >= 0:
        p.append(int(pred[p[-1]]))
    return p[::-1]

newseq = []; i = 0; nrec = 0
while i < N:
    if not inface[i]:
        newseq.append(verts[i]); i += 1
    else:
        j = i
        while j < N and inface[j]:
            j += 1
        if j < N:
            if newseq:
                newseq += around(newseq[-1], verts[j])[1:]; nrec += 1
            else:
                newseq.append(verts[j])
            i = j + 1
        else:
            i = j   # run to the end -> handled by the close

ne = [(newseq[k], newseq[k + 1]) for k in range(len(newseq) - 1) if newseq[k] != newseq[k + 1]]
cp = around(newseq[-1], newseq[0]); nrec += 1
ne += [(cp[k], cp[k + 1]) for k in range(len(cp) - 1) if cp[k] != cp[k + 1]]
np.savez(os.environ.get('PF_OUT', os.path.join(UV, '_test_loop_fix.npz')),
         loop=np.array(ne, dtype=np.int64), anchors=d['anchors'], neck_floor=d['neck_floor'])
print('[postfix] reconnected %d face-crossings around the face -> %d loop edges' % (nrec, len(ne)))
