"""HER BODY'S OWN MIRROR PLANE round the crotch, for the pelvis graft (2026-10-16, the user: "why is it not symmetric?").

The graft assumed her mirror plane is x = const: _place shifts the donor to the leg gap but never turns it, and every later step
(DONOR_X_SCALE about x = 0, the cleft strip |X| < CLEFT_W, the anal canal's centre line in build_canal.py) is built in that frame.
On female_elf_00011_ her skin round the vulva mirrors best about a plane tilted ~3.4 deg off it (2.3 deg up, 2.6 deg front; the
mirror gap rms 1.14 mm about it against 2.42 mm about x = const) -- the donor's stitched edge sat up to 4.9 mm uneven, a crease
beside her anus on one side only.

Fitted on her cage under and round where the donor goes (the donor is a strip along her midline, lower belly -> crotch -> buttock
cleft; nothing on the cage is ours yet, so nothing is left out): the plane whose mirror image of her skin lands closest to her skin (gap per point trimmed at 15 x her median edge, so a one-sided feature cannot drag it),
over a few reaches from the placed donor (in its own half-width), the reach she mirrors best at. The same fit as the vulva
generator's (vulva_new/mirror_plane.py), validated there on 00011's low poly.
   python pelvis_mirror.py <in.npz: V (cage), W (the donor as placed), MM> <out.json>"""
import sys, json, numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

REACHES = (1.0, 1.25, 1.5, 2.0, 3.0)


def _fit(X, x0):
    kd = cKDTree(X); cap = 15 * np.median(kd.query(X, k=2)[0][:, 1])

    def cost(p):
        n = np.array([1.0, p[0], p[1]]); n /= np.linalg.norm(n); dd = X @ n - p[2]; Xm = X - 2 * dd[:, None] * n
        return np.mean(np.minimum(kd.query(Xm)[0], cap) ** 2)
    r = minimize(cost, [0.0, 0.0, x0], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 2000})
    n = np.array([1.0, *r.x[:2]]); s = np.linalg.norm(n)
    return n / s, r.x[2] / s, float(np.sqrt(r.fun)), float(np.sqrt(cost([0.0, 0.0, x0])))


def main(src, out):
    z = np.load(src); V = z["V"].astype(float); W = z["W"].astype(float); MM = float(z["MM"])
    cen = W.mean(0); rf = 0.5 * float(np.ptp(W[:, 0]))                   # the donor's half-width across her
    d = cKDTree(W).query(V)[0]
    best = None
    for k in REACHES:
        sel = d < k * rf
        if sel.sum() < 500:
            continue
        n, c, g, g0 = _fit(V[sel], float(cen[0]))
        print("[mirror] reach %.2f x the donor's half-width (%.0f mm, %d skin verts): tilted %.2f deg up / %.2f deg front, mirror gap rms %.2f mm "
              "(about x = const %.2f)" % (k, k * rf * MM, int(sel.sum()), np.degrees(np.arctan2(n[1], n[0])),
                                          np.degrees(np.arctan2(n[2], n[0])), g * MM, g0 * MM), flush=True)
        if best is None or g < best[2]:
            best = (n, c, g, g0, k)
    if best is None:
        raise SystemExit("[mirror] no skin round the donor to fit on")
    n, c, g, g0, k = best
    tilt = float(np.degrees(np.arccos(np.clip(n[0], -1, 1))))
    json.dump(dict(n=n.tolist(), c=float(c), cen=cen.tolist(), reach=k, reach_mm=k * rf * MM, rms_mm=g * MM, rms_xconst_mm=g0 * MM,
                   tilt_deg=tilt, frame="cage / low poly (x across, y up, z front)",
                   note="her mirror plane: n . p = c"), open(out, "w"), indent=1)
    print("[mirror] her mirror plane: reach %.2f x the donor's half-width, %.2f deg off x = const; mirror gap rms %.2f mm (about x = const %.2f)"
          % (k, tilt, g * MM, g0 * MM), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
