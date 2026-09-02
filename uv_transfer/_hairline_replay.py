"""LOCAL hairline iteration harness -- replay the REAL chain without a full bake.

Runs the actual pipeline scripts (crease_curv_v2 -> path3_v2 -> _hairline_pipe.postfix) on the LAST bake's
captured `_gb_mesh.npz`, then the follow-check, then renders the result. This is how a hairline-stage change
(crease / path3 / postfix) gets tested in seconds instead of a ~10-min bake.

WHY THIS IS TRUSTWORTHY (and the historical "worked locally, broke in pipeline" was not):
  - It uses the bake's REAL intermediate mesh, not a frozen snapshot.
  - It runs the REAL scripts the pipeline runs, in the same order, same env flags.
  - It HARD-GUARDS the stale-file trap: every captured aux input must match THIS mesh, or it HALTS. A
    mismatch is exactly what silently ruined local runs before (a different bake's facemask/hair).

A BAKE IS STILL NEEDED ONLY when an UPSTREAM stage changed (Trellis mesh gen, the rough-hair region, or the
facemask) -- then `_gb_mesh.npz`/`_gb_facevert.npy` are genuinely stale and must be regenerated. Run:
  python _hairline_replay.py
"""
import numpy as np, os, sys, subprocess, importlib.util, time, json
UV = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(UV, path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    mp = os.path.join(UV, '_gb_mesh.npz')
    if not os.path.exists(mp):
        print('[replay] no _gb_mesh.npz -- bake once first to capture it.'); return 1
    gm = np.load(mp, allow_pickle=True); nv = int(gm['nv']); nf = len(gm['fv'])
    age = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(mp)))
    print('[replay] mesh: %d verts, %d faces  (captured %s)' % (nv, nf, age))

    # ---- STALE-FILE GUARD: the historical killer. Aux inputs the postfix/back-seam read MUST match this mesh ----
    stale = []
    fvp = os.path.join(UV, '_gb_facevert.npy')
    if os.path.exists(fvp):
        fl = len(np.load(fvp))
        if fl != nv:
            stale.append('_gb_facevert.npy (face mask) len %d != mesh nv %d' % (fl, nv))
    flp = os.path.join(UV, '_gb_fill.npz')
    if os.path.exists(flp):
        ff = np.load(flp)['faces']
        if ff.size and int(ff.max()) >= nf:
            stale.append('_gb_fill.npz (hair faces) max idx %d >= nf %d' % (int(ff.max()), nf))
    if stale:
        print('[replay] *** STALE CAPTURES -- BAKE NEEDED. A local result here would LIE (this is the trap): ***')
        for s in stale:
            print('     ' + s)
        return 2
    print('[replay] aux captures aligned with the mesh -> local replay is trustworthy')

    # ---- run the REAL chain (same scripts + flags as _greenborder.compute) ----
    def run(script, extra):
        subprocess.run([PY, os.path.join(UV, script)], cwd=UV, check=True,
                       env={**os.environ, **extra}, stdout=subprocess.DEVNULL)
    run('crease_curv_v2.py', {'CR_MESH': '_gb_mesh.npz', 'CR_OUT': '_gb_crease.npz'})
    run('path3_v2.py', {'P2_MESH': '_gb_mesh.npz', 'P2_CRE': '_gb_crease.npz', 'P2_CRE_RAW': '_gb_crease.npz',
                        'P2_OUT': '_gb_loop.npz', 'P2_FILT_OUT': '_gb_filt.npz', 'P2_CLOSE': '0',
                        'P2_DRAPE_KEEPCRE': '1', 'P2_DRAPE_BRIDGE': '1'})
    _imp('_hp', '_hairline_pipe.py').postfix_loop(UV)

    # ---- follow-check + baseline compare ----
    m = _imp('_fc', '_hairline_follow_check.py').measure()
    print('[replay] >>> hairline follow: total %(total)s%% (L %(left)s%% R %(right)s%%) of %(n)s verts <<<' % m)
    bp = os.path.join(UV, '_follow_baseline.json')
    if os.path.exists(bp):
        b = json.load(open(bp)); hit = False
        for k in ('total', 'left', 'right'):
            if m[k] < b[k] - 8:
                print('     *** REGRESSION: %s %.0f%% << baseline %.0f%% ***' % (k, m[k], b[k])); hit = True
        if not hit:
            print('     OK vs baseline (total base %.0f%%)' % b['total'])
    else:
        print('     (no baseline yet -- FOLLOW_SETBASE=1 python _hairline_follow_check.py after a GOOD bake)')

    # ---- render the final (post-fixed) loop, front + side ----
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        co = gm['co'].astype(float); fv = gm['fv'].astype(np.int64); hair = gm['hair'].astype(bool)
        loop = np.load(os.path.join(UV, '_gb_loop_fix.npz'))['loop']
        fold = np.load(os.path.join(UV, '_gb_fold.npz'))['edges'] if os.path.exists(os.path.join(UV, '_gb_fold.npz')) else np.zeros((0, 2), int)
        znv = (co[:, 2] - co[:, 2].min()) / (np.ptp(co[:, 2]) + 1e-9)
        xc = 0.5 * (co[:, 0].min() + co[:, 0].max()); hw = 0.5 * (co[:, 0].max() - co[:, 0].min())
        hv = np.zeros(len(co), bool); hv[np.unique(fv[hair])] = True
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        for ax, ix, tt in [(axs[0], 0, 'FRONT'), (axs[1], 1, 'SIDE (front=low y)')]:
            sel = znv > 0.55
            ax.scatter((co[sel, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1), znv[sel], s=2, c='0.9')
            sel = hv & (znv > 0.55)
            ax.scatter((co[sel, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1), znv[sel], s=8, c='orange', alpha=0.4)
            for u, w in loop:
                ax.plot([(co[u, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1), (co[w, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1)], [znv[u], znv[w]], c='red', lw=1.4)
            for u, w in fold:
                ax.plot([(co[u, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1), (co[w, ix] - (xc if ix == 0 else 0)) / (hw if ix == 0 else 1)], [znv[u], znv[w]], c='blue', lw=1.8)
            ax.set_title(tt)
            if ix == 0:
                ax.set_xlim(-0.4, 0.4)
        axs[0].set_title('FRONT  follow=%s%% (L%s R%s)  orange=hair red=hairline blue=back-seam' % (m['total'], m['left'], m['right']))
        plt.tight_layout(); plt.savefig(os.path.join(UV, '_replay_result.png'), dpi=95)
        print('[replay] rendered -> _replay_result.png')
    except Exception as e:
        print('[replay] render skipped (%r)' % e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
