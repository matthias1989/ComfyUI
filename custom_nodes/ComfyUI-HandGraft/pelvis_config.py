"""Pipeline recipe of the pelvis graft -- wired 2026-09-15 after the user reviewed the R11 renders/blend.

pelvis_graft.py keeps its historical defaults: the offline parity harness replays those byte-identically.
The PIPELINE runs this recipe instead. Every consumer calls apply() BEFORE importing pelvis_graft
(ComfyUI-Retopo/scripts/graft_stage.py, genital_texture.py), so the graft and the genital texture place the
SAME donor with the SAME zones. A PELVIS_* variable already set in the calling process wins (setdefault),
so a harness can still override a single flag. fingerprint() goes into cache keys: changing the recipe or
the donor bytes invalidates cached grafts.

R11 = the user's TRIM4 donor, isotropically remeshed, fitted to avoid the leg surface that runs through the
Trellis crotch, plus a back hinge (scratchpad _fit5b.py / _fit5_variants.py). NOTE: the donor asset is
FITTED to character_posed_00003_'s cage. Another character needs the fit re-run (not yet in the node).

R22 (wired 2026-09-16, user: "yes do both") replaces R11's donor: the user's TRIM5 (lips re-heighted) carried onto the
same remesh (T5R, same vertex order), lips scaled to a 9 cm cleft with the fourchette on the body's crotch line,
axis tilt held at 27 deg, the outer-lip edge strip bending with the skin (scratchpad _t5transfer.py, _fit8.py).
Same recipe flags as R11; faces, zones and anus weights are byte-identical to the T4RK files, only V differs.
Also FITTED to character_posed_00003_'s cage. Previous donor files (T4RK) stay in assets/.

R42 (wired 2026-09-16, user: 'yes please go ahead'): the TRIM5 donor refit 1 cm higher (fourchette +1.0 cm above the
crotch line) with the outer 35% of the skin apron lying on the body (FIT8_APRON_BAND), plus the join smoothing below.
The leg skin that ran through the vestibule is pushed out by ComfyUI-Retopo/scripts/vestibule_clear.py right after
the graft (it needs scipy, so it runs with ComfyUI's python, not inside Blender).
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

PIPELINE_ENV = {
    "PELVIS_DONOR": os.path.join(ASSETS, "donor_pelvis_T5SD.npz"),          # T5S fitted to 0003 = the fallback (was T5D10)
    "PELVIS_ZONES": os.path.join(ASSETS, "genital_donor_zones_T5S.npy"),     # (was genital_donor_zones_T5D10.npy)
    # base recipe (scratchpad _matrix.sh)
    "PELVIS_CUT_BOUND": "1", "PELVIS_CUT_REACH": "0.55", "PELVIS_CUT_GUARD": "0.3", "PELVIS_CUT_OPEN": "2",
    "PELVIS_CREASE_MAX": "1.0", "PELVIS_HOLE_OUTLIER": "5.0", "PELVIS_HOLE_SNAP": "1.0",
    "PELVIS_BRIDGE_FAN": "1.2", "PELVIS_WELD_MAX": "2", "PELVIS_CUT_ISLANDS": "1", "PELVIS_SEAM_MAX": "2",
    "PELVIS_SEAM_GEO": "1",
    # R11 row
    "PELVIS_CUT_RULE": "cover", "PELVIS_RIM_KEEP_SPACING": "1", "PELVIS_DONOR_FROZEN": "1",
    "PELVIS_HOLE_REBUILD_NONMONO": "1", "PELVIS_HOLE_MAP": "monotone", "PELVIS_HOLE_SPREAD": "8.98",
    "PELVIS_BRUSH_SMOOTH": "1",
    # R42 join smoothing (user picked seam_60it_20mm_inner_only, 2026-09-16): 60 iterations, reach 0.5411 of the donor
    # width (= 20 mm on T5D10), only the inner lips + opening protected
    "PELVIS_BRUSH_ITERS": "60", "PELVIS_BRUSH_R_FRAC": "0.5411", "PELVIS_BRUSH_PROTECT_ZONES": "3,4",
    # A + B (user OK 2026-09-17, after the leg crease on the first outer-skin bake): the cover window is measured over the
    # donor's rim band (0.35 x the apron width), not the vaginal canal -- -25..+27 mm -> -13..+16 mm on the true skin;
    # and each hole vertex spreads its drag over 2x its own displacement along the skin. Offline: strip folds 42 -> 19.
    "PELVIS_WINDOW_RIM_BAND": "0.35", "PELVIS_HOLE_SPREAD_DISP": "2",
    # PER-CHARACTER FIT (user OK 2026-09-18, after character_posed_00197_'s graft sank 17-25 mm into her belly): the
    # donor above is T5R fitted ONCE to 0003's cage; graft_stage.py now fits the base T5R to every cage itself
    # (pelvis_fit.py, the offline harness consolidated -- bit-identical to T5D10 on 0003's inputs) with the recipe
    # that built T5D10 (_fit8.py 8.8 -10 8, FIT_BASE=T5R, tilt 27, bend Z1, apron band 0.35, +1 cm, 9 cm cleft).
    # PELVIS_DONOR stays the fallback: if the fit cannot run, the graft uses it and says so.
    "PELVIS_FIT": "1",
    "PELVIS_FIT_BASE": os.path.join(ASSETS, "donor_pelvis_T5S.npz"),         # recut rim (was T5R)
    "PELVIS_FIT_BACK_MM": "8.8", "PELVIS_FIT_TILT0": "-10", "PELVIS_FIT_FWD0": "8",
    "FIT8_AXIS_TILT": "27", "FIT8_BEND_Z1": "1", "FIT8_APRON_BAND": "0.35", "FIT8_UP_CM": "1",
    "PELVIS_FIT_CLEFT_CM": "9.0",
    # RELATIVE cleft (user 2026-09-18, after 197's lips hung into the leg gap): 9 cm was approved on
    # character_posed_00003_, whose placed donor measures 7.730734 cm before the lips scale (her 2026-09-18 cage).
    # Every body keeps that proportion: target = 9.0 x its placed cleft / 7.730734 -> 0003 9.0 cm, 197 7.94 cm.
    "PELVIS_FIT_CLEFT_REF_CM": "7.730734",
    # CUT FLOOD SEED (user 2026-09-18 "fix all issues", after female_elf_00004_'s graft crashed): the cut region was
    # flooded from the covered face nearest the anchor, which on 00004 sat in a 2-face scrap next to the cut; the open
    # step then erased it and the empty cut crashed. 'largest' floods from the largest covered piece instead.
    # Offline: 0003 and 197 grafts byte-identical to 'anchor'; 00004 grafts its full 1,517-face cut.
    "PELVIS_FLOOD_SEED": "largest",
    # RIM RECUT T5S (user 2026-09-18 "1 yes", after the jagged stitch ring / "sticker" outline on 197): the base donor's
    # outer rim was a hand-trim zigzag 1.68x the length of a smooth loop, and the graft stitches the body to exactly that
    # edge. T5S = T5R cut along a smoothed rim curve (scratchpad attach/_recut_rim.py: window 15 mm, 1 edge inside, marching
    # triangles; lips and anus core untouched) -> rim 327 -> 280 verts, length ratio 1.61 -> 1.09. Offline: 197 ring + nick
    # gone, 0003 / 00004 look the same; vestibule clear + genital colours run on its 4,215-vertex sidecar. The four files
    # above (base, fallback, zones, anus weights) are T5S; the T5R / T5D10 files stay in assets/.
    # LIPS NEVER BELOW THE CROTCH (user 2026-09-18 "yes, switch it on after the bake"): pelvis_fit measures the crotch
    # (lowest midline surface -- _crotch_y's 60 fixed bins put it 0.5-2 cm low) and raises the fourchette only as far as
    # the lowest labia needs to sit on it. 0003 untouched (byte-identical), 197 +0.56, 00004 +1.71, 00005 +1.26 cm.
    "PELVIS_FIT_LIPS_FLOOR": "1",
    # ANUS PROTECT (user 2026-09-19 "anus not deep enough", "yes to all"): the donor's anus is a real canal (~13 mm long,
    # ~9 mm wide), but the join's 60-pass brush smooth did not count it as detail and squeezed its mouth 8.8 -> 5.8 mm
    # (a 4 mm crescent). The anus core (anus weight >= 0.5) now joins the protected detail: it leaves the join untouched
    # (0.00 mm moved on 0003 / 00004 / 00005), no crease at the freeze edge. Off = byte-identical.
    "PELVIS_PROTECT_ANUS": "1",
    # GRAFT SIZE BY STATURE (user 2026-09-19: "scaling to each pelvis makes sense, you just need to adjust your math",
    # "use a bigger scale ... the anus will go more back"): _place sized the donor by hip width, which made it 10-12% smaller
    # on every narrower-hipped body (197 / 00004 / 00005: 0.88-0.90 vs 0003 1.02) -- shorter perineum, anus too far forward.
    # Now body height / PELVIS_PLACE_REF_H, the reference = 0003's height / her width ratio (she keeps her size): anus
    # 23.8-24.7 -> 28.3 mm behind the fourchette (0003: 28.5), vulva 46-47 -> 53 mm (0003: 53), cleft 9.0 cm on all.
    "PELVIS_PLACE_SCALE_BY": "height", "PELVIS_PLACE_REF_H": "0.9823804050936058",
    # ANUS FURTHER BACK + CANAL INTO THE BODY (user 2026-09-19: "it should start very close to the end of the buttocks,
    # but right now it seems to be much deeper", picked "perineum +15 mm" from rendered variants; and the black spot
    # behind female_elf_00005_'s anus). The perineum is 15 mm longer (mm of a 1.7 m body -- the cage's own height, so
    # relative); sizes unchanged (the whole-donor scale would also grow the vulva: a 11.2 cm cleft at x1.25). The canal
    # is bent to point up into the pelvis: seated on the cleft floor it ran ALONG the skin (79-99 deg to the outward
    # normal) and half its cup (57-66 of 122 vertices, ray parity on the watertight cage) lay OUTSIDE the body, painted
    # with the anus core colour = the black spot. The bend pivots on the tube's MOUTH (the first version pivoted halfway
    # down the tube and folded the soft band into itself: 50 / 81 crossing face pairs -> shards inside the opening).
    # Dry runs 2026-09-19: cup outside 0 on 00004 / 00005, graft crossings at the anus 0 / 0 (low poly 20 / 0).
    "PELVIS_FIT_PERINEUM_MM": "15", "PELVIS_FIT_CANAL_INWARD": "1",
}
ANUS_WEIGHTS = os.path.join(ASSETS, "genital_donor_anus_w_T5S.npy")     # donor-intrinsic anus label (soft), per donor vertex (was _T5D10)
PIPELINE_ENV["PELVIS_ANUS_W"] = ANUS_WEIGHTS   # the graft copies it into the donor sidecar for the low poly's colours
# HER MIRROR PLANE (user OK 2026-10-17, 00011's uneven back): the graft runs in her own mirror frame (graft_stage.py +
# pelvis_mirror.py: the plane fitted on her skin round the donor -> <run>/mirror_plane.json); anal_region.py builds the cleft
# about it (the carve and the area it rebuilds on her mirror-averaged skin) and build_canal.py lays the canal in it.
PIPELINE_ENV["PELVIS_MIRROR"] = "1"


def apply():
    for k, v in PIPELINE_ENV.items():
        os.environ.setdefault(k, v)


def fingerprint():
    h = hashlib.md5()
    for k in sorted(PIPELINE_ENV):
        h.update(("%s=%s;" % (k, os.environ.get(k, PIPELINE_ENV[k]))).encode())
    for k in ("PELVIS_DONOR", "PELVIS_ZONES", "PELVIS_FIT_BASE"):
        with open(os.environ.get(k, PIPELINE_ENV[k]), "rb") as f:
            h.update(f.read())
    return h.hexdigest()
