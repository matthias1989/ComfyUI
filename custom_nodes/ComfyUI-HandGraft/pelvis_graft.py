"""KAN-18 pelvic/genital donor graft -- the consolidated, validated recipe (B40).

Runs in headless Blender. Grafts the high-res donor genital (donor_pelvis_v2.npz) onto a
SMOOTH remeshed body and returns a clean, border-free, uniform-density result.

The seam saga (B1-B40) distilled to 3 stages, each fixing a distinct failure:
  1. BUILD  - insert donor + flush the outer margin flush to the body (no height step) +
              bridge + beautify (kill slivers) + HC-Laplacian wide smooth (flatten the
              seam without a moat, protecting the central vulva detail).
  2. FUSE   - voxel remesh @ FUSE_VOX. This makes the whole region UNIFORM density, which
              ELIMINATES the dense-donor/coarse-body density boundary. That boundary is the
              root of the persistent "border" (a vertex-normal artifact that survives every
              smoothing/normal trick) -- only fusing removes it.
  3. SEAM   - Taubin (anti-shrink) smooth of the fusion-seam ring. This now WORKS headless
              precisely because step 2 made the density uniform (the same smooth failed on the
              un-fused mesh because the density boundary fought it). Vulva + far body protected.

Detail intentionally ends up slightly soft (voxel) -> the fine micro-detail is restored by a
separate genital normal-map bake from the high-poly donor (Body_LP_Ref). See project memory.

Usage (standalone test):
  blender --background --python pelvis_graft.py -- --body handgraft_out.npz --voxel_body --out out.npz --blend out.blend
Pipeline:
  call graft_pelvis(body_V, body_F, voxel_body=False)  # body already the gen_seams remesh
"""
import os, sys, numpy as np
import time as _time

# Every line in this module is stamped with elapsed seconds. Without it the only way to find
# out which stage costs what is to watch the log and guess, which is how a pass that tested the
# whole 2.18M-face body 17 times to look at the crotch survived unnoticed.
_T0 = _time.time()
_builtin_print = print


def print(*a, **k):
    _builtin_print("[%7.1fs]" % (_time.time() - _T0), *a, **k)

import bpy, bmesh
from collections import defaultdict
from mathutils import Vector, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
A_DONOR = os.environ.get("PELVIS_DONOR", "") or os.path.join(HERE, "assets", "donor_pelvis_v2.npz")   # env override for testing a trimmed asset without touching the shipping one
A_ZONES = os.environ.get("PELVIS_ZONES", "") or os.path.join(HERE, "assets", "genital_donor_zones.npy")  # per-donor-vert label: 0 skin, 1-2 outer labia, 3 inner labia, 4 opening
A_PLACE = os.path.join(HERE, "assets", "pelvis_place_fixed.npz")

# ---- recipe parameters (validated B38/B39/B40) ----
GS         = float(os.environ.get("PELVIS_GS", "1.10"))  # genital scale (1.0=anatomical, 1.10=+10%)
# LATERAL-only multiplier on top of GS. The graft reads too WIDE but not too big, so narrowing
# it in x keeps its height, depth and every bit of its detail -- unlike lowering GS (shrinks
# everything) or trimming faces (removes detail the cut was about to expose).
GS_X       = float(os.environ.get("PELVIS_GS_X", "0.9"))
# Placement nudge (mesh Y=up/height, Z=depth; +Z is FRONT -- the face region's normals are +Z).
# The anchor A itself comes from pelvis_place_fixed.npz and is a CONSTANT from a reference body:
# only the donor's SCALE adapts to this character (band width / ref_width). So on a body whose
# pelvis sits at a different height or depth the graft lands off, which is what these nudge.
PLACE_DY   = float(os.environ.get("PELVIS_PLACE_DY", "0.0"))
PLACE_DZ   = float(os.environ.get("PELVIS_PLACE_DZ", "0.0"))
BODY_VOX   = 0.0028    # body voxel (matches gen_seams target density) -- only if voxel_body
# Rim snap clamp, as a multiple of the rim's OWN median snap distance (relative, not absolute).
# Unclamped, closest_point_on_mesh drags outlier rim verts onto whatever body surface happens to
# be nearest -- across the gap, onto the far labia or the thigh -- by up to 12x the rim spacing.
RIM_SNAP_CLAMP = float(os.environ.get("PELVIS_RIM_CLAMP", "0.25"))   # swept 1.0-3.0 on 00003; see commit
# Body-side hole loop relaxation. The rim is now evenly spaced, but the hole the flood-cut leaves
# in the voxel body is as jagged as the voxel grid; the stitch inherits that from the other side.
HOLE_RELAX_ITERS = int(os.environ.get("PELVIS_HOLE_RELAX_ITERS", "60"))
HOLE_RELAX       = float(os.environ.get("PELVIS_HOLE_RELAX", "0.5"))
# Pull the cut hole boundary onto the donor rim outline. Flood-cutting along face boundaries
# traces a staircase, so the hole perimeter came out 1.75x the rim it is stitched to (0.6529 vs
# 0.3740); that surplus length becomes skewed triangles. Moving the EXISTING hole verts onto the
# rim outline shortens the staircase without adding a single vertex anywhere.
# "auto" (default) solves for the blend factor that makes the hole perimeter match the
# rim, per character, instead of a value tuned to one body. Measured on 00003: the raw
# staircase is 1.60x the rim; a fixed 0.75 lands at 1.03 and 1.0 OVERSHOOTS to 0.90
# (chord-cutting shortens it past the rim), so the right factor is not a constant.
HOLE_SNAP = os.environ.get("PELVIS_HOLE_SNAP", "auto")
# Fraction of a hole vertex's own snap displacement over which that displacement is spread into the
# surrounding body. 0 = old behaviour (one ring of body faces absorbs the whole move -> slivers).
HOLE_SPREAD = float(os.environ.get("PELVIS_HOLE_SPREAD", "0.0"))
# WINDOW_RIM_BAND (2026-09-17, default 0 = off). The cover cut's window along the donor normal was sized by the
# farthest / deepest donor vertex of ALL -- and that is the vaginal canal, inside the body by design. On the hollow cage
# its inner layers sat next to the canal and held the window at -10..+10 mm; on the outer-skin cage it is -25..+27 mm
# and the cut takes thigh skin the donor surface never covers. With a fraction here, the standoff and buried depth are
# measured only over the donor's rim band: donor verts within that fraction of the apron width of the rim (geodesic
# along the donor; apron width = median rim-to-labelled-zone distance) -- the part the fit lays ON the skin.
WINDOW_RIM_BAND = float(os.environ.get("PELVIS_WINDOW_RIM_BAND", "0"))
# HOLE_SPREAD_DISP (2026-09-17, default 0 = off). The hole spread gives every hole vertex the same radius (HOLE_SPREAD x
# the rim spacing, ~9.6 mm), but where the donor edge meets the thigh a hole vertex moves up to 24 mm: a 24 mm drag over
# 9.6 mm folds the skin -- the pleat on the inner left thigh (user 2026-09-17). With a factor k here each hole vertex
# spreads over max(that radius, k x ITS OWN displacement), measured ALONG THE BODY SURFACE (edge Dijkstra, so it cannot
# jump the leg gap). Only the vertices that move far get a wide reach; with k = 2 the dragged skin tilts < ~45 deg.
HOLE_SPREAD_DISP = float(os.environ.get("PELVIS_HOLE_SPREAD_DISP", "0"))
# Hole rebuild correspondence: "monotone" = nearest-point projection made monotone around the
# ring (minimal per-vertex travel); "arc" = the old uniform arc-fraction resample.
HOLE_MAP = os.environ.get("PELVIS_HOLE_MAP", "arc")
# Bridge apportionment: "length" = hole vertices shared out in proportion to rim-edge length
# (fan-free by construction); "prox" = counts from the nearest-vertex map, total corrected.
BRIDGE_ALLOC = os.environ.get("PELVIS_BRIDGE_ALLOC", "local")
# Match the donor RIM's spacing to the body HOLE's spacing before bridging. The IDEA is sound:
# with 217 rim verts against 555 hole verts the mean advance is 2.6, so almost every rim edge
# has to fan two or three hole vertices, and that fan IS the sheet of long triangles across the
# mons. Splitting the rim's own long edges would make the advance ~1 and the bridge a strip.
#
# OFF, because the implementation failed twice on 2026-09-12 and neither failure is a detail:
#   1. bmesh.ops.subdivide_edges REALLOCATES. Every BMVert reference held across it dies with
#      "BMesh data of type BMVert has been removed" -- R, and H, and anything else already
#      captured. Working in indices fixes R; H dies two lines later unless it is re-fetched too.
#   2. Rebuilding the rim ring by walking `is_boundary` from a seed returned 1 vertex, so the
#      rim edges are not simple bmesh boundaries at that point in _cutjoin. Whatever the donor
#      rim is there, it is not "the edges with one face", and that has to be understood before
#      any of this is worth retrying.
# Anyone re-enabling this: re-fetch BOTH loops by index after the op, and find out what the rim
# actually is topologically first.
RIM_MATCH_PASSES = int(os.environ.get("PELVIS_RIM_MATCH", "0"))

# Fan limit as a MULTIPLE OF THE MEAN advance (nH/nR), not an absolute count: no rim edge
# may swallow more than this share of the hole, so a pole cannot form at any mesh density.
BRIDGE_FAN = float(os.environ.get("PELVIS_BRIDGE_FAN", "4.0"))
# How far past the donor the cut may reach, as a MULTIPLE OF THE FOOTPRINT'S OWN MEDIAN
# distance-to-donor. 0 = off (cut the whole planar shadow, thighs included).
# x the donor's OWN worst standoff from THIS body, measured per character in _graft_cutjoin.
# Swept on 00003 against the numbers the bridge actually has to span (hole/rim perimeter, worst
# rim->hole gap, self-intersections left after the relax):
# Swept on 00003 against what the bridge has to span and what it leaves behind. "slivers" =
# faces near the graft whose longest edge is over 5x / 8x the local median -- those ARE the
# spike, and they come from the hole boundary being dragged to meet the rim, so they track the
# post-snap gap:
#   0.40   hole 0.76x rim -- too small, the parity solver gives up (factor 0.000), gap med 11.3x
#   0.55   hole 1.82x -> 0.99, gap med 0.1x max 0.2x, 1376 self-int, slivers  47 / 2, worst 31mm
#   0.70   hole 1.63x -> 1.04, gap med 0.7x max 7.0x, 1141 self-int, slivers  53 / 7, worst 43mm
#   1.00   hole 1.69x -> 1.00, gap med 2.0x max 11.5x, 1394 self-int
#   shipping (footprint-median rule):                              slivers  75 / 17, worst 35mm
# 0.55 gives a bridge that spans essentially nothing (0.1x the rim spacing) and leaves the
# fewest stretched faces; that is the number the user sees. Past it the cut takes inner thigh
# the donor cannot cover and the bridge webs the gap.
CUT_RULE = os.environ.get("PELVIS_CUT_RULE", "footprint")   # "donor" = bound by the donor's own standoff (2026-09-13, unapproved)
CUT_REACH = float(os.environ.get("PELVIS_CUT_REACH", "3.0"))
# "donor" = cut every body face the donor actually covers (3D proximity to the donor
# surface); "shadow" = the old planar-projection footprint, which misses the body under
# the donor's rear and leaves 40% of the rim with nothing to stitch to.
CUT_MODE = os.environ.get("PELVIS_CUT_MODE", "shadow")
# The flood must still open a hole comparable to the unbounded one. Bounding the CANDIDATES
# can starve it until the body stays CLOSED over the donor and the graft is invisible --
# this happened on a real bake 2026-09-05 while the frozen snapshot looked fine. If a bound
# costs more than this fraction of the unbounded flood, it is discarded and we cut plain.
CUT_GUARD = float(os.environ.get("PELVIS_CUT_GUARD", "0.5"))   # see CUT_REACH: 0.55 floods 0.35x
# Seam smooth band = this multiple of the measured p95 rim->hole gap (0 = fixed band).
SEAM_BAND_FOLLOW = float(os.environ.get("PELVIS_SEAM_BAND_FOLLOW", "1.0"))
# Crease relaxation. The standard is the BODY'S OWN smoothness: any seam edge whose dihedral
# exceeds CREASE_K x the p95 dihedral of the surrounding body gets relaxed. No absolute angle.
CREASE_ITERS = int(os.environ.get("PELVIS_CREASE_ITERS", "40"))
# Drift cap for the crease relax, x the vertex's own edge length; 0 = off. Swept on 00003
# against faces stretched past 5x / 8x the local edge and the worst edge, alongside how many
# creased seam edges the relax still clears:
#   off   47 / 2 / 30.5 mm   13392 -> 8006 creased
#   1.0   33 / 2 / 28.1 mm   13392 -> 9763
#   2.0   44 / 8 / 26.3 mm   13392 -> 8320
#   4.0   38 / 8 / 26.7 mm   13392 -> 7803
#   relax off entirely  28 / 0 / 22.2 mm   13392 (none cleared)
# 1.0 is the best sliver result that still clears most of the creasing. Turning the relax off
# altogether wins on slivers but leaves every creased seam edge, which is the other half of
# what reads as a seam.
CREASE_MAX   = float(os.environ.get("PELVIS_CREASE_MAX", "0"))
CREASE_K     = float(os.environ.get("PELVIS_CREASE_K", "1.5"))
# ...but never below the seam's own top percentile: the donor is a detailed asset whose real
# anatomy sits far above the body's smoothness, so a body-referenced threshold alone flags
# everything (9184 edges, no convergence). Attack the worst tail instead -- that IS the facets.
CREASE_PCT   = float(os.environ.get("PELVIS_CREASE_PCT", "95"))
CREASE_W     = float(os.environ.get("PELVIS_CREASE_W", "0.5"))
# SELF-INTERSECTION relax. The donor's sides pass THROUGH the thigh: geometry inside geometry,
# which no surface smoothing can see (it is not a crease -- both sheets are locally flat).
# Find the actually-intersecting face pairs with a BVH and relax only those, until none are
# left. Labeled labia/opening detail stays exempt, as everywhere else.
XSECT_ITERS = int(os.environ.get("PELVIS_XSECT_ITERS", "16"))
XSECT_W     = float(os.environ.get("PELVIS_XSECT_W", "0.6"))
XSECT_MAX   = float(os.environ.get("PELVIS_XSECT_MAX", "0"))   # drift cap, x the vertex's own edge length; 0 = off
XSECT_MARGIN= float(os.environ.get("PELVIS_XSECT_MARGIN", "0.35"))  # x local edge length
# The self-intersections sit INSIDE the labeled labia folds, which are protected by the hard
# rule. Set 0 to let the intersection pass touch them too (the only way to reach them).
XSECT_PROTECT = int(os.environ.get("PELVIS_XSECT_PROTECT", "1"))
# LIFT the donor out of the body. After the cut we know exactly which body SURVIVES, so a
# donor vertex still inside it is a genuine penetration -- the graft's sides passing through
# the thigh. Project those onto the surviving surface. Verts the cut exposed are untouched,
# which is what my earlier "35% buried" probe got wrong by testing against the UNCUT body.
XOUT      = int(os.environ.get("PELVIS_XOUT", "0"))   # OFF: nearest-point inside/outside is
# unreliable in the crotch concavity (normals flip) -- it turned 705 intersections into 1293.
XOUT_EPS  = float(os.environ.get("PELVIS_XOUT_EPS", "0.0004"))
CREASE_RING  = int(os.environ.get("PELVIS_CREASE_RING", "0"))   # measured: dilating made it WORSE (375 -> 417)
# Morphological CLOSE on the cut region. The flood leaves one-face-wide peninsulas of uncut
# body poking into the opening; the bridge then wraps them into a thin blade sticking out of
# the seam, which no amount of smoothing removes because it is topology, not curvature.
# Any face with >=2 of its edges already on the flood is surrounded and gets absorbed.
RIM_SEAT_ITERS = int(os.environ.get("PELVIS_RIM_SEAT", "0"))   # project+respace passes; 0 = off
RIM_RING = os.environ.get("PELVIS_RIM_RING", "0") == "1"   # cut at the rim curve, not its shadow
RIM_RING_W = float(os.environ.get("PELVIS_RIM_RING_W", "1.5"))   # band half-width, x the body edge
RIM_RING_BALL = float(os.environ.get("PELVIS_RIM_RING_BALL", "4.0"))  # search ball margin past the rim radius, x edge
CUT_ISLANDS = os.environ.get("PELVIS_CUT_ISLANDS", "0") == "1"  # absorb uncut islands inside the cut
# COVER_RIM_FOOT (2026-09-14). The cover cut keeps every body face whose nearest donor point lies within one
# rim spacing of the rim, so the hole edge stays outside the donor. That assumes the donor FLOATS above the
# body. Once the donor is fitted INTO the skin, the strip it keeps lies under the donor edge and passes
# through it -- the groove across the mons top on the fitted TRIM4 copy (46 of 63 crossing faces were
# never covered). Keep such a face only when it is really beside the donor: its offset from the nearest
# donor point has a sideways part (clamped to the rim) larger than half the body edge; a perpendicular
# foot means the face lies under the donor.
COVER_RIM_FOOT = os.environ.get("PELVIS_COVER_RIM_FOOT", "0") == "1"
# CUT_JOIN_PIECES (2026-09-14). The flood grows only the covered piece at the anchor. On the fitted copy
# the covered skin came out in 32 pieces (2211, 501, 399, 326, 243, 152, ...) and 173 of 199 body faces the
# donor passed through sat in the pieces the flood dropped. Cut every covered piece the donor actually
# passes through (tested by overlap), and join it to the hole through a one-ring corridor of faces near
# the donor, so the cut stays ONE hole for the bridge.
CUT_JOIN_PIECES = os.environ.get("PELVIS_CUT_JOIN_PIECES", "0") == "1"
HOLE_FILL = os.environ.get("PELVIS_HOLE_FILL", "0") == "1"   # fill leftover seam openings in place
WELD_MAX = float(os.environ.get("PELVIS_WELD_MAX", "999"))  # x rim spacing; 999 keeps the old 0.02 behaviour
BRIDGE_MAXEDGE = float(os.environ.get("PELVIS_BRIDGE_MAXEDGE", "0"))  # x rim spacing; 0 = off
HOLE_OUTLIER = float(os.environ.get("PELVIS_HOLE_OUTLIER", "0"))  # x rim spacing; 0 = off
CUT_OPEN  = int(os.environ.get("PELVIS_CUT_OPEN", "0"))   # erode+regrow passes: kills thin cut tongues
CUT_CLOSE = int(os.environ.get("PELVIS_CUT_CLOSE", "3"))
# Where the cut flood starts. "anchor" (historical): the candidate face nearest the anchor. On female_elf_00004_
# (2026-09-18) the cover cut came out in pieces [1517, 175, 126, 2, 2, 2] and the anchor's nearest face sat in a
# 2-face scrap -> the flood cut 2 faces, the open step eroded them to 0 and the graft crashed. "largest": start in
# the LARGEST connected piece of the candidates (the region the donor actually replaces), nearest-to-anchor inside it.
# Where the nearest face already lies in the largest piece (00003, 00197) the result is identical.
FLOOD_SEED = os.environ.get("PELVIS_FLOOD_SEED", "anchor")
# TRIM the donor to the width that actually SHOWS. Measured: 35% of the donor sits buried
# inside the thighs, and those verts are 5x further from the midline than the visible ones
# (|x| med 0.0130 vs 0.0026). Keep no wider than TRIM_K x the p95 |x| of the verts that are
# NOT buried -- the graft sets its own width from the body it is going into. 0 = off.
# COLLAR TRIM (2026-09-13). The donor carries a big skin apron -- 4128 of its 7012 verts, 136 mm
# of pelvis front to back -- that was authored on ANOTHER body. On this one it does not touch:
# median standoff 6.6 mm, up to 23 mm, and it runs straight OVER the notch between the legs, so
# the shadow it casts lands on the two thighs as separate patches with nothing under the middle.
# That sheet across the notch is the "wall", and no cut rule can seat it: measured, cutting only
# what the donor covers splits into 24 components (largest 38%) and the graft tears.
# The body already HAS thighs and a mons. The one thing it lacks is the vulva. So keep the
# labelled detail plus a collar of skin measured ALONG THE DONOR SURFACE and drop the rest --
# a geodesic ball around a connected set stays connected, so the patch comes out compact with a
# single clean rim (measured: 1 boundary loop at every collar size, against 29 for a standoff
# threshold). The width is a fraction of the DETAIL'S OWN radius, so it scales with the donor.
COLLAR = float(os.environ.get("PELVIS_COLLAR", "0"))   # 0 = off (ship the whole apron)
NOTCH_TRIM = os.environ.get("PELVIS_NOTCH_TRIM", "0") == "1"   # drop donor skin with no body under it
RIM_RELAX = int(os.environ.get("PELVIS_RIM_RELAX", "40"))        # boundary smoothing passes after the notch trim
RIM_RELAX_RINGS = int(os.environ.get("PELVIS_RIM_RELAX_RINGS", "30"))  # how far the move is carried inward
NOTCH_SMOOTH_ITERS = int(os.environ.get("PELVIS_NOTCH_SMOOTH", "12"))  # diffusion passes on the airborne field
TRIM_K   = float(os.environ.get("PELVIS_TRIM_K", "0"))
# ...but never remove more than this share of the donor, however the measurement comes out.
TRIM_MAX = float(os.environ.get("PELVIS_TRIM_MAX", "0.5"))
# Graded transition ring. Donor core sits at 0.33x the body edge length and the step to 1.00x is
# crossed in ONE shell (0.51x -> 0.90x). Refine the BODY only, in a narrow annulus around the
# anchor, so the density walks down instead of falling off a cliff. Donor core and the lean body
# are both untouched -- this only widens the transition band.
GRADE_R1   = float(os.environ.get("PELVIS_GRADE_R1", "0.045"))   # inner band outer radius (2 cuts)
GRADE_R2   = float(os.environ.get("PELVIS_GRADE_R2", "0.070"))   # mid band outer radius  (1 cut)
GRADE_CUTS = int(os.environ.get("PELVIS_GRADE_CUTS", "0"))       # 0 = off
DONOR_X_SCALE = 0.80   # narrow the placed donor in X (width) about the midline; <1 pulls the seam off the thighs
MORPH_R    = 0.05      # snap: full weight within MORPH_R of the anchor
MORPH_FALL = 0.022     # ... soft falloff to 0 over this width (blends the genital into the body)
SNAP_SUBDIV = 2        # (legacy snap) subdivide the genital region this many cuts
SNAP_CLAMP  = 0.012    # (legacy snap) clamp distance
SNAP_SMOOTH_ITERS = 40 # (legacy snap) ring iterations
SEAM_MAX = float(os.environ.get("PELVIS_SEAM_MAX", "0"))   # seam-smooth drift cap, x the local edge; 0 = off
# The seam-smooth zone is selected by projecting every vertex into ONE plane and measuring its
# distance to the rim's 2D shadow (`poly`). That shadow runs straight down the inner thighs, so
# the "9 mm band" actually caught 45,917 verts -- 6.5x the area a real 9 mm band around a 530 mm
# ring can cover -- and 120 Laplacian iterations flattened all of it. The oval creases either
# side of the vulva ARE that shadow's outline. SEAM_GEO measures the band ALONG THE SURFACE from
# the donor/body seam instead, and ramps the weight to zero at the band edge so there is no
# on/off boundary left to crease.
SEAM_GEO = os.environ.get("PELVIS_SEAM_GEO", "0") == "1"
# Band width, as a MULTIPLE of the shipped 0.009. That number was tuned as a distance to the
# planar shadow, which is not the same quantity as a distance along the surface, so it has to be
# re-swept once the band is geodesic. 1.0 = unchanged.
SEAM_BAND_MUL = float(os.environ.get("PELVIS_SEAM_BAND_MUL", "1"))
# The cut's reach test is a EUCLIDEAN distance from a body face to the nearest donor vertex, and
# at the crotch the two thighs are ~2 mm apart -- so a face on the FAR wall of the leg gap, whose
# skin faces the other way, measures "within reach" and gets cut. That is what opens the two
# lens-shaped holes either side of the vulva: the membrane in front is removed and the far wall
# of the gap is exposed, edge and all. A donor can only replace skin that faces the same way it
# does, so require the body face normal to agree with the donor's normal where it is nearest.
# A sign test -- nothing to tune.
CUT_FACING = os.environ.get("PELVIS_CUT_FACING", "0")   # "1" = face normals, "smooth" = vertex-averaged
# The donor is placed by a similarity transform, so its RIM cannot lie on a body of a different
# shape: measured on 00003 it floats a median 6 mm and up to 21 mm off the skin. The graft copes
# by dragging the body's cut edge up onto the floating rim, which lifts a collar of skin -- and
# the patch still reads as a raised island. RIM_CONFORM does it the other way round: seat the rim
# ON the body and carry that correction INTO the donor by diffusion, held at exactly zero on the
# labelled labia/opening. The rim lands on the skin, the apron bends to follow, the detail does
# not move at all. Needs PELVIS_RIM_SEAT to have produced the seated curve.
RIM_CONFORM = os.environ.get("PELVIS_RIM_CONFORM", "0") == "1"
# RIM_CONFORM only carries the RIM's correction inward, so the middle of the apron keeps whatever
# standoff the rigid placement gave it -- measured, the patch still stands proud along its left
# edge and reads as a pasted island. APRON_CONFORM gives every zone-0 apron vertex its OWN target
# on the body and scales the move by the same harmonic weight (1 on the rim, 0 on the labelled
# labia/opening), so the apron settles onto the skin while the detail does not move at all.
# The target comes from a RAY along the donor's own normal, not from nearest-point: in the leg
# gap nearest-point teleports to the other thigh (see the barrier-ring failure above).
APRON_CONFORM = os.environ.get("PELVIS_APRON_CONFORM", "0") == "1"
# The rim re-space exists to undo the COLLAPSE nearest-point projection causes. It re-spaced to
# UNIFORM arc length instead, which also "corrects" the donor's own spacing: a hand-trimmed rim is
# 0.5..6.7 mm between vertices, so rim vertices were slid along the rim by up to 19 mm and every
# triangle hanging off them turned inside out -- measured, 128 folded faces on the rim before the
# bridge exists, which is the crumpled mons. KEEP_SPACING places each vertex back at its ORIGINAL
# arc-length fraction along the snapped curve: it moves across the rim by the snap, never along it.
RIM_KEEP_SPACING = os.environ.get("PELVIS_RIM_KEEP_SPACING", "0") == "1"
# The seam smooth and the crease relax both move a vertex toward the MEAN of its neighbours, which
# slides it ACROSS the surface as well as smoothing it. On a donor made of a few large thin
# triangles (a hand-trimmed mons) that slide turns triangles over: measured on TRIM4, the mons went
# 0.5% folded -> 20.6% in the seam smooth and -> 56.6% in the crease relax. NORMAL_ONLY keeps only
# the component of each move along the vertex normal: bumps and creases still flatten, but a
# vertex cannot travel over its neighbours, so a triangle cannot flip -- and it cannot migrate
# tens of mm along the band either, which is what the drift caps were bolted on for.
SMOOTH_NORMAL_ONLY = os.environ.get("PELVIS_SMOOTH_NORMAL_ONLY", "0") == "1"
# MEASURED WORSE, leave off: normal-only took the TRIM4 mons to 69% folded in the seam smooth (the
# plain move gave 20.6%). The slide was never the cause -- the donor interior being smoothed at all is.
# SEAM_DONOR_RIM_ONLY: the seam smooth blends what the graft BUILT (the bridge) into what it CUT
# (the body's hole edge). It was also smoothing the donor itself, because its band is a fixed 9e-3
# sized for the old donor -- on a 2 cm donor that band covers the whole mons, and 120 passes destroy
# a surface that was authored smooth. With this on, the only donor vertices it moves are the ones
# ON the seam (a donor/body edge); every other donor vertex keeps exactly the shape it was given.
SEAM_DONOR_RIM_ONLY = os.environ.get("PELVIS_SEAM_DONOR_RIM_ONLY", "0") == "1"
# Not enough on its own: with only the rim moving, the mons still went 0.5% -> 19.3% folded in the
# seam smooth, 44% after the crease relax, 46.9% after the self-intersection relax. A hand-trimmed
# donor has LARGE triangles hanging off the rim (spacing up to 6.7 mm), so moving a rim vertex a few
# mm while its inner neighbour stays put flips them. DONOR_FROZEN: once placed, no pass moves a
# donor vertex -- seam smooth, mons de-bump, crease relax and self-intersection relax all adapt the
# BODY side (hole edge + bridge) to the donor instead of the other way round.
DONOR_FROZEN = os.environ.get("PELVIS_DONOR_FROZEN", "0") == "1"
# The bridge walks the rim one edge at a time and gives each edge a COUNT of hole vertices. Every
# edge that gets zero (42 of 145 on TRIM4) builds its triangle onto the same hole vertex as the edge
# before it, so one body vertex fans across a run of rim vertices: 320 spike faces at the bridge,
# longest 80 mm, on a seam whose loops sit 0.3x a rim spacing apart. The later passes only hid them
# by dragging the donor rim -- which is what crumpled the mons. ZIPPER stitches the two loops the
# standard way: from the aligned start, repeatedly add whichever triangle has the SHORTER new edge
# (advance along the rim, or along the hole) until both loops are consumed. No counts, no caps.
BRIDGE_ZIPPER = os.environ.get("PELVIS_BRIDGE_ZIPPER", "0") == "1"
# MEASURED, did not help on its own: the zipper stacked 84 triangles on one vertex, because the loops
# it was given were already out of order (see HOLE_REBUILD_NONMONO).
# The hole->rim snap projects each hole vertex to its nearest point on the rim. That is not
# one-to-one: on TRIM4, 73 snapped hole verts collapsed onto a neighbour, 56 of 145 rim verts got no
# hole vertex at all, and the hole's order ran BACKWARD along the rim 16 times -- the bridge then
# fans across the empty stretches and folds where the order reverses (~300 spike faces). The
# in-order rebuild already exists but only fires when the projected hole is LONGER than the rim;
# this one came out shorter (0.85x) and slipped through. NONMONO fires the rebuild on the thing that
# actually breaks the bridge: the projected hole running backward along the rim more than once.
HOLE_REBUILD_NONMONO = os.environ.get("PELVIS_HOLE_REBUILD_NONMONO", "0") == "1"
# Guarding the (unlabelled) anus from the apron conform sounded right and measures WORSE: freezing
# 1,556 verts solid in the middle of a 42 mm conform is the same on/off boundary that creates a
# crease everywhere else in this file. Low-poly worst edge 13.0 -> 23.8 mm, verts over 10 mm off
# the cage 389 -> 588, for a barely visible improvement at the dimple. Off unless asked for.
APRON_ANUS = os.environ.get("PELVIS_APRON_ANUS", "0") == "1"
# This cage's inner thigh is visibly torn -- measured, 43% of the faces there flip the sign of the
# test between neighbours, because a single triangle's normal on a spiky surface is noise. The
# vertex-averaged normal is the same quantity measured over the 1-ring, which is what "which way
# does this skin face" actually means at the scale of the graft.
GRAFT_SEAM_BAND  = 0.009  # GRAFT seam smooth: smooth verts within this of the donor rim line (both sides)
GRAFT_SEAM_ITERS = int(os.environ.get("PELVIS_GRAFT_SEAM_ITERS", "120"))    # ... iterations (inner detail beyond the band is protected, so this can be high)
_DONOR_MESH_IDX = None   # donor vertex -> merged-mesh index, recorded by _graft_cutjoin for the donor sidecar
GRAFT_PROTECT_R  = 0.003  # NEVER smooth verts within this of a LABELED detail vert (outer/inner labia + opening,
                          # donor zones 1-4). Distance-to-rim alone fails near the anus where the rim hugs the
                          # labia -> the outer labia got eaten. Labeled detail = 100% protected; smooth only outside it.
MONS_R     = 0.020        # gentle pubic-mound de-bump: radius from the vulva centre. The anus sits ~0.025-0.03 back
MONS_ITERS = int(os.environ.get("PELVIS_GRAFT_MONS_ITERS", "12"))           # so it stays untouched; labia stay protected. Light de-bump, not a flatten.
MONS_W     = 0.30
FLUSH_HI   = 0.032     # flush the outer margin: weight 1 for dmin<FLUSH_LO ...
FLUSH_W    = 0.014     #   ... ramping to 0 by FLUSH_HI (dmin = dist from donor rim)
# A-fix (depth-stretch walls): the stretch lifts the back perineum off the body surface -> a height step
# (wall) the smoother can't remove. So also SHRINKWRAP the stretched back perineum onto the body so it
# follows the curve. Ramp 0 at the vulva (front, protected) -> BACK_STRENGTH at the back; keep the anus dimple.
BACK_STRENGTH = 0.0             # A-fix DISABLED (didn't kill the stretch walls). Option B = gentler stretch.
BACK_Z0, BACK_W = 0.008, 0.045   # back-region ramp: starts 0.008 behind vulva, full by +0.045 (body +Z)
BACK_ANUS_KEEP = 0.012           # don't shrinkwrap within this radius of the anus (keep the dimple)
CUT        = 1.08      # cut the body hole just BEYOND the donor rim (no overlap)
BRIDGE_CUTS= 2
BEAUTY_R   = 0.040     # seam band (from rim) to triangulate+beautify
# HC-Laplacian wide smooth (flatten seam, protect central vulva)
HC_REG, HC_PX, HC_PR = 0.084, 0.016, 0.044
HC_ALPHA, HC_BETA, HC_ITERS = 0.1, 0.6, 45
FUSE_VOX   = 0.0012    # fine fuse -> uniform density (kills the density border)
# seam-smooth of the fused crease: tangential RELAX (even spacing) + Taubin (flatten)
# smooth the WHOLE surround (mons + inner thighs + perineum) -- not just the seam ring -- to match
# B40; the surround carries the original body-voxel waviness the fine fuse preserved.
# GPU-sculpt mask. Protect the whole DONOR interior (<INNER; rim/seam is at dA~0.074) so the strong
# smooth works OUTWARD on the body/walls instead of dishing the donor. SOFT edges (wide falloffs) so no
# step/wall forms where smooth meets protected.
# pg14 mask (proven) + small anus guard. The walls are fixed at the SOURCE by the back-flush (A-fix),
# not by the smoother, so the mask stays at pg14's values.
SEAM_INNER, SEAM_FAR = 0.028, 0.130   # (legacy wide-surround band -- standalone/non-local only)
INNER_FALL, FAR_FALL = 0.008, 0.025
# LOCAL/pipeline: smooth ONLY a thin band at the donor->body SEAM (dist-from-rim), NOT the wide surround --
# the pipeline body has real detail (buttocks cleft, mons) a wide smooth destroys. We only blend the graft line.
SEAM_BAND      = float(os.environ.get("PELVIS_SEAM_BAND", "0.030"))
SEAM_BAND_FALL = float(os.environ.get("PELVIS_SEAM_FALL", "0.024"))
# (was fixed 0.030/0.024)   # front/legs: feathered band -> blends the seam without a band-edge crease
SEAM_BELLY_CUT = 0.5                  # belly side (above the anchor, +Y) smoothed less
# BACK: tight band + tight falloff -> smooth ONLY the immediate seam line, leave the buttocks raw
SEAM_BACK_NARROW = 0.024             # COMBINE (v7 back + v13 front): tight back band ~0.006 -> minimal back
                                     #   smoothing = preserves the BUTT SHAPE (user preferred v7's back). Front
                                     #   unaffected (front-only belly-cut keeps v13's front). (v13 used 0.008/wide.)
SEAM_BACK_FALL = 0.005              # back falloff tight (v7) -> butt stays raw
# SIDES: the lateral seam (groin / inner-thigh junction) only ever got the plain front/legs
# band -- back is narrowed, belly is halved, sides had no term of their own. Widen the band
# and its falloff toward the flanks so the graft blends further out there. 0 = previous.
SEAM_SIDE_WIDEN = float(os.environ.get("PELVIS_SEAM_SIDE", "0.0"))
SEAM_SIDE_X     = 0.020   # lateral distance over which the widening ramps in
CLEFT_W = float(os.environ.get("PELVIS_CLEFT_W", "0.013"))                       # PROTECT the gluteal cleft: a midline strip (|X|<CLEFT_W) on the back
                                      #   stays unsmoothed so the natural cleft groove is kept (like the brush)
ANUS_R, ANUS_FALL = 0.016, 0.016      # small guard: keep the anus dimple from the smoother
RELAX_ITERS, SEAM_ITERS = 60, 200   # v13 locked (280 added no real smoothing gain)
SURR_ITERS = 130                      # LOCAL/headless surround polish: Taubin lambda/mu pairs
SURR_LAM, SURR_MU = 0.5, -0.53        #   (volume-preserving -> no crater/shrink)
SURR_RELAX = 0                        #   tangential RELAX pre-pass OFF: with stale normals it INFLATED the
                                      #   surround into a raised welt around the genital (much worse). Don't.
# LOCAL border fix: grade the donor's RIM density down toward the body's so the dense-donor/coarse-body
# density jump (-> the vertex-normal "border") disappears. Protect the detailed center; decimate the rim.
GRADE_RATIO = 1.0                     # decimate the rim to this fraction (1.0 = OFF: collapse-decimating the
                                      #   rim made JAGGED slivers = a worse seam. Border -> handled by the
                                      #   genital normal-map bake instead, which overrides geometry normals.)
GRADE_KEEP, GRADE_RAMP = 0.045, 0.025 # protect dA<KEEP (vulva/anus); ramp to full decimate by KEEP+RAMP
# Genital flag margin as a MULTIPLE OF THE DONOR'S OWN MEDIAN EDGE, so it scales with the asset
# and with the placement scale instead of being a length in body units.
GENITAL_MARGIN_K = float(os.environ.get("PELVIS_GENITAL_MARGIN_K", "6.0"))
GENITAL_HUG = 0.013                   # TIGHT genital flag: face within this of the donor SURFACE = genital
                                      #   (donor + ~1 border ring) -> UV island/material hugs only the genitals,
                                      #   surrounding skin stays part of the body. (replaces the big GENITAL_R sphere)
GENITAL_R = 0.085                     # LOCAL: faces within this dA of the anchor = the genital region ->
                                      #   saved as a per-face flag so gen_seams cuts it into its own UV
                                      #   island (separate texturing; body front-image can't see it)
FINAL_SMOOTH_FAC, FINAL_SMOOTH_ITERS = 0.5, 0   # global pass off



_TRACE = os.environ.get("PELVIS_TRACE", "") == "1"
_trace_state = {}


def _trace(tag, bm, A):
    """Report the worst displacement of any vertex BELOW the graft since the last checkpoint.

    The remaining spike is a fan around a single body vertex dragged 8.45 mm, 19 mm below the
    donor's lowest point. Ablating the relax passes one at a time never removed it, so instead
    of guessing which stage is responsible, have each stage say so."""
    if not _TRACE:
        return
    try:
        co = np.array([v.co[:] for v in bm.verts], float)
    except Exception:
        return
    y0 = A[1] - 0.020
    prev = _trace_state.get("co")
    if prev is not None and len(prev) == len(co):
        d = np.linalg.norm(co - prev, axis=1) * (co[:, 1] < y0)
        if d.max() > 1e-9:
            k = int(np.argmax(d))
            print("[trace] %-26s %5d vert(s) below the graft moved >0.5mm; worst %6.2f mm at %s"
                  % (tag, int((d > 0.0005).sum()), d.max() / 1.0016 * 1700, np.round(co[k], 4)),
                  flush=True)
        else:
            print("[trace] %-26s nothing below the graft moved" % tag, flush=True)
    else:
        print("[trace] %-26s baseline (%d verts)" % (tag, len(co)), flush=True)
    _trace_state["co"] = co


def _leg_junction_Y(V):
    ymin, ymax = V[:, 1].min(), V[:, 1].max(); st = (ymax - ymin) / 60
    for i in range(60):
        b = V[(V[:, 1] >= ymin + st * i) & (V[:, 1] < ymin + st * (i + 1))]
        if len(b) and (np.abs(b[:, 0]) < 0.04).sum() > 20:
            return ymin + st * i
    return (ymin + ymax) / 2


def _midline_front_z(V, y, half=0.012, xw=0.030):
    """Front-most Z of the body on the midline at height y (+Z is front)."""
    b = V[(np.abs(V[:, 1] - y) < half) & (np.abs(V[:, 0]) < xw)]
    return None if len(b) < 8 else float(np.percentile(b[:, 2], 98))


def _height(V):
    """Her standing height (head to feet). PELVIS_BODY_H when graft_stage.py has turned the cage into her mirror frame: turned,
    the feet and hands shift a little in y and the extent reads ~0.2 % taller -- her height is the same in either frame."""
    h = os.environ.get("PELVIS_BODY_H", "")
    return float(h) if h else float(np.ptp(V[:, 1]))


def _crotch_y(V, F):
    """Body-derived crotch height, from surface ORIENTATION rather than occupancy.

    The anchor in pelvis_place_fixed.npz is a CONSTANT measured on a reference body -- only the
    donor SCALE adapts to this character -- so on a body whose pelvis sits higher or lower the
    graft lands off. Measured on character_posed_00003_: the fixed anchor is 6-9 cm too high.

    Occupancy cannot find the crotch. After the voxel remesh the touching inner thighs fill the
    midline too, which is exactly why _leg_junction_Y returns the THIGH CONTACT (28-38% of
    height) instead. Orientation separates them with no useful threshold ambiguity: the inner
    thigh surfaces face each other SIDEWAYS (|nx| 0.93-0.97, |ny|/|nx| 0.02-0.17) while the
    perineum faces DOWN (|nx| 0.67, ratio 0.76). That is a ~38x gap, so any cut inside it lands
    on the same band. Returns None if the signature is absent -- caller keeps the fixed anchor.
    """
    tris = []
    for f in F:
        f = list(f)
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    T = np.asarray(tris, int)
    if len(T) < 100:
        return None
    c = V[T].mean(1)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max()); H = _height(V)
    if H <= 0:
        return None
    mid = np.abs(c[:, 0]) < 0.012
    NB = 60
    lat = np.full(NB, np.nan)
    for i in range(NB):
        m = mid & (c[:, 1] >= ymin + H * i / NB) & (c[:, 1] < ymin + H * (i + 1) / NB)
        if int(m.sum()) >= 8:
            lat[i] = float(np.abs(n[m, 0]).mean())
    lo, hi = int(0.30 * NB), int(0.60 * NB)
    thigh = [i for i in range(lo, hi) if not np.isnan(lat[i]) and lat[i] > 0.90]
    if not thigh:
        return None
    ref = float(np.median([lat[i] for i in thigh]))
    for i in range(max(thigh) + 1, hi):
        if not np.isnan(lat[i]) and lat[i] < 0.75 * ref:
            return ymin + H * i / NB
    return None


def _crotch_x(V, y, hint=0.0):
    """The body's own midline AT THE CROTCH, from the gap between the legs.

    A[0] comes from pelvis_place_fixed.npz and is a constant measured on the reference body, so
    the donor lands at x ~ 0 whatever the character does. Bodies are not symmetric: on
    character_posed_00003_ the nipple midline reads -0.005 and the glute midline +0.017, and the
    crotch itself sits at +0.0070 -- so the donor came out ~7 mm off centre, 14% of its own
    width, which is plainly visible from the front.

    Below the crotch the two legs are separate, so a thin horizontal slice of the surface has
    one wide gap in x and the middle of that gap IS the midline. Sampled over several heights
    and taken as the median, because one slice can catch a fold. Measured on the 2026-09-12
    cage the bands read +0.0076 +0.0072 +0.0070 +0.0067 +0.0064 -- stable to under a millimetre.

    Everything is a fraction of the body's own height, so it carries to any character. Returns
    None when the legs touch all the way down (no gap to measure) or the bands disagree, and the
    caller then keeps the reference X rather than guessing.
    """
    H = _height(V)
    if H <= 0:
        return None
    out = []
    for frac in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040):
        m = np.abs(V[:, 1] - (y - frac * H)) < 0.0015 * H
        if m.sum() < 200:
            continue
        x = np.sort(V[m, 0])
        x = x[np.abs(x - hint) < 0.07 * H]
        if len(x) < 20:
            continue
        g = np.diff(x)
        k = int(np.argmax(g))
        if g[k] < 0.002 * H:                 # legs touching at this height -- nothing to measure
            continue
        out.append(0.5 * (x[k] + x[k + 1]))
    if len(out) < 3:
        print("[place] leg gap: only %d usable band(s) -- keeping the reference X" % len(out),
              flush=True)
        return None
    a = np.asarray(out)
    if float(np.ptp(a)) > 0.02 * H:
        print("[place] leg gap: bands disagree by %.4f -- keeping the reference X"
              % float(np.ptp(a)), flush=True)
        return None
    return float(np.median(a))


def _genital_flag(cents, W, Fd=None, W_placed=None):
    """Which faces belong to the GENITAL texture island.

    Proximity to the donor SURFACE is the wrong test: 4,128 of the donor's 7,012 vertices are
    labelled zone 0 = SKIN (the apron that blends it into the body), and they run 7.3 cm below
    the anchor and 4.9 cm out. Flagging everything near the donor therefore painted the genital
    texture down both inner thighs -- the "colours leaking to the legs".

    genital_donor_zones.npy labels the donor's own vertices: 0 skin, 1-2 outer labia, 3 inner
    labia, 4 opening. Zones 1-4 span 1.8 cm below the anchor and 3.0 cm wide -- an actual vulva.
    Measured on the 2026-09-12 graft, flagging against those instead:

        whole donor, 0.013      49,678 faces   9.5 cm below the anchor, 6.9 cm wide
        zones 1-4 + margin      11,583 faces   2.8 cm below the anchor, 3.6 cm wide

    The margin is a multiple of the donor's own median edge, so it is one border ring whatever
    the asset's resolution. Falls back to the old whole-donor test if the label file is missing,
    and says so.
    """
    import numpy as _np
    pts, why = None, ""
    try:
        Z = _np.load(A_ZONES)
        # _graft_cutjoin may hand back a W that no longer matches the donor's own vertex list
        # (the rim snap rebuilds it), so try the post-join copy first and the placed one second.
        for src, tag in ((W, "post-join"), (W_placed, "as placed")):
            if src is None or len(Z) != len(src):
                continue
            core = _np.where(_np.isin(Z, [1, 2, 3, 4]))[0]
            if len(core) >= 100:
                pts = src[core]
                why = "%d labelled genital verts (zones 1-4) of %d, %s" % (len(core), len(src), tag)
                break
        if pts is None:
            print("[pelvis_graft] genital zones: %d labels but donor is %d vert(s) -- cannot match"
                  % (len(Z), len(W)), flush=True)
    except Exception as e:
        print("[pelvis_graft] genital zones unreadable (%r)" % (e,), flush=True)
    if pts is None:
        pts = W
        why = "NO zone labels -- falling back to the whole donor, the flag will include its skin apron"
    m = GENITAL_HUG
    if Fd is not None and len(Fd) and int(_np.max(Fd)) < len(W):
        de = _np.concatenate([_np.linalg.norm(W[Fd[:, i]] - W[Fd[:, (i + 1) % 3]], axis=1)
                              for i in range(3)])
        m = GENITAL_MARGIN_K * float(_np.median(de))
    t = kdtree.KDTree(len(pts))
    for i in range(len(pts)):
        t.insert(Vector(pts[i].tolist()), i)
    t.balance()
    flag = _np.array([t.find(Vector(c.tolist()))[2] < m for c in cents])
    print("[pelvis_graft] genital flag: %s; margin %.5f -> %d face(s)"
          % (why, m, int(flag.sum())), flush=True)
    return flag


def _place(Vt, Ft=None):
    """Auto-place the donor on this body: anchor at the leg junction, scale to pelvis width."""
    f = np.load(A_PLACE); L = f["L"]; A = f["A"].copy(); ref_w = float(f["ref_width"]); Gc = f["Gc"]
    # OFF by default. The detector is right -- it puts the anchor at the true crotch, 6.5cm
    # lower, matching the user's estimate -- but the donor was AUTHORED for the higher anchor,
    # so at the crotch it no longer conforms to the body: measured rim->hole median gap went
    # 0.6x -> 7.8x rim spacing and the hole/rim perimeter 1.00 -> 1.47 even with the lateral
    # gate. Moving the anchor needs the donor's fit to adapt with it. Until that exists this
    # would ship a worse join than the (wrongly placed but well-fitted) fixed anchor.
    if Ft is not None and os.environ.get("PELVIS_AUTO_ANCHOR", "1") == "1":
        _cy = _crotch_y(Vt, Ft)
        if _cy is not None:
            _h = _height(Vt)
            print(f"[place] anchor Y {A[1]:+.4f} -> {_cy:+.4f} (body crotch, "
                  f"{100.0 * (_cy - Vt[:, 1].min()) / max(_h, 1e-9):.1f}% of height; "
                  f"moved {(_cy - A[1]) / max(_h, 1e-9) * 170.0:+.1f} cm at 170cm)", flush=True)
            # Follow the body's FRONT SURFACE down as well. A[2] is meaningless in absolute
            # terms -- the donor extends forward from the anchor to reach the skin -- so what
            # must be preserved is the anchor's depth RELATIVE to the local surface. The midline
            # front is +0.055 at the old anchor but +0.034 at the crotch (the pelvis curves
            # under), so moving Y alone buries the donor ~0.02 inside the body and only a sliver
            # stays visible.
            _z_old = _midline_front_z(Vt, A[1])
            _z_new = _midline_front_z(Vt, _cy)
            if _z_old is not None and _z_new is not None:
                print(f"[place] anchor Z {A[2]:+.4f} -> {A[2] + (_z_new - _z_old):+.4f} "
                      f"(midline front {_z_old:+.4f} -> {_z_new:+.4f}, keeping depth under skin)",
                      flush=True)
                A[2] = A[2] + (_z_new - _z_old)
            A[1] = _cy
            # ... and sideways. Y and Z were already measured off this body; X was still the
            # reference constant, which is what put the donor off the midline.
            _cx = _crotch_x(Vt, A[1], hint=A[0])
            # HER MIRROR PLANE (2026-10-16): graft_stage.py turns the cage so her own mirror plane (pelvis_mirror.py, fitted on
            # her skin round the donor) is x = PELVIS_MIDLINE_X. The leg gap's middle is only a proxy for it -- on 00011 it sat
            # 1.7 mm to one side, the donor's stitched edges uneven on the back -- so the donor's midline goes on her plane.
            if os.environ.get("PELVIS_MIDLINE_X", "") != "":
                _mx = float(os.environ["PELVIS_MIDLINE_X"])
                print(f"[place] anchor X on her mirror plane {_mx:+.4f} (the leg gap's middle "
                      f"{_cx if _cx is not None else float('nan'):+.4f})", flush=True)
                _cx = _mx
            if _cx is not None:
                print(f"[place] anchor X {A[0]:+.4f} -> {_cx:+.4f} (body midline at the leg "
                      f"gap; moved {(_cx - A[0]) / max(_h, 1e-9) * 170.0:+.1f} cm at 170cm)",
                      flush=True)
                A[0] = _cx
        else:
            print("[place] crotch signature not found -- keeping the fixed anchor", flush=True)
    A[1] += PLACE_DY; A[2] += PLACE_DZ   # manual nudge on top (default 0/0)
    dd = np.load(A_DONOR); Vd = dd["V"].astype(float); Fd = dd["F"].astype(int)
    # Scale the donor by the body width WHERE THE DONOR ACTUALLY SITS -- the anchor -- not at a
    # separately detected "leg junction".
    # _leg_junction_Y returns the FIRST y (scanning up from the feet) with >20 verts near the
    # midline, which is where the inner THIGHS touch, not the crotch. On a legs-together pose
    # that is far below the pelvis: on character_posed_00003_ it landed at 28% of body height,
    # so the band was measured across the thighs (X width 0.2700) and scaled the donor to
    # ratio 1.1819 -- 18% oversize. An oversized donor rim cannot match the hole cut in the
    # body, and the stitch fans out into long spikes at the vulva.
    # ref_w is itself calibrated at the anchor: on this body the anchor band measures 0.2286
    # against ref_width 0.2284, i.e. ratio 1.0005 -- the donor was authored for this size.
    band = Vt[(Vt[:, 1] > A[1] - 0.05) & (Vt[:, 1] < A[1] + 0.06)]
    ratio = np.ptp(band[:, 0]) / ref_w
    # PELVIS_PLACE_SCALE_BY=height (2026-09-19, user: "scaling to each pelvis makes sense, you just need to adjust your
    # math"): hip WIDTH made the graft 10-12% smaller on every narrower-hipped body (197 / 00004 / 00005: 0.88-0.90 vs
    # 0003's 1.02) -- a shorter perineum, the anus too far forward -- and the band it is read in hangs off the crotch
    # detector's per-body error. The size follows STATURE instead: ratio = body height / PELVIS_PLACE_REF_H, the
    # reference being 0003's height / her width ratio, so she keeps her exact size.
    _hb = _height(Vt)
    print(f"[place] scale: hip-width ratio {ratio!r}, body height {_hb!r}", flush=True)
    if os.environ.get("PELVIS_PLACE_SCALE_BY", "width") == "height" and os.environ.get("PELVIS_PLACE_REF_H", ""):
        _rh = _hb / float(os.environ["PELVIS_PLACE_REF_H"])
        print(f"[place] scale by HEIGHT: {_rh:.4f} (hip width would give {ratio:.4f})", flush=True)
        ratio = _rh
    _Vl = (Vd - Gc) * GS
    if GS_X != 1.0:
        _Vl = _Vl * np.array([GS_X, 1.0, 1.0])
    W = _Vl @ (L.T * ratio) + A
    return W, Fd, A


def _collar_trim(W, Fd, frac):
    """Keep the labelled detail plus `frac` x its own radius of skin around it, measured along
    the donor surface. Returns the kept faces; W and the zone labels are left alone so every
    index-keyed thing downstream (genital_donor_zones.npy, W_placed) still lines up."""
    import heapq
    from collections import defaultdict, deque
    try:
        Z = np.load(A_ZONES)
    except Exception as e:
        print(f"[collar] zone labels unreadable ({e!r}) -- shipping the whole donor", flush=True)
        return Fd
    if len(Z) != len(W):
        print(f"[collar] zone labels are {len(Z)} for {len(W)} donor verts -- STALE, skipped",
              flush=True)
        return Fd
    core = np.isin(Z, [1, 2, 3, 4])
    if not core.any():
        return Fd
    nb = defaultdict(set)
    for f in Fd:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    gd = np.full(len(W), np.inf)
    pq = [(0.0, int(i)) for i in np.where(core)[0]]
    for _, i in pq:
        gd[i] = 0.0
    heapq.heapify(pq)
    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur > gd[u]:
            continue
        for v in nb[u]:
            nd = dcur + float(np.linalg.norm(W[u] - W[v]))
            if nd < gd[v]:
                gd[v] = nd; heapq.heappush(pq, (nd, v))
    R = float(np.linalg.norm(W[core] - W[core].mean(0), axis=1).max())
    keepv = core | (gd <= frac * R)
    fadj = defaultdict(list)
    for fi, f in enumerate(Fd):
        for e in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
            fadj[(min(e), max(e))].append(fi)
    inside = set(fi for fi, f in enumerate(Fd) if keepv[f[0]] and keepv[f[1]] and keepv[f[2]])
    # the component holding the most labelled detail -- not the biggest, the RIGHT one
    seen = set(); best = None; bestc = -1
    for s in inside:
        if s in seen:
            continue
        comp = {s}; dq = deque([s]); seen.add(s)
        while dq:
            x = dq.popleft(); f = Fd[x]
            for e in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
                for g in fadj[(min(e), max(e))]:
                    if g in inside and g not in seen:
                        seen.add(g); comp.add(g); dq.append(g)
        nc = int(sum(1 for x in comp if core[Fd[x]].any()))
        if nc > bestc:
            bestc, best = nc, comp
    if best is None or len(best) < 0.1 * len(Fd):
        print(f"[collar] trim would leave only {0 if best is None else len(best)} of {len(Fd)} "
              f"faces -- refusing, shipping the whole donor", flush=True)
        return Fd
    out = np.asarray([Fd[i] for i in sorted(best)], int)   # ndarray: _genital_flag slices it
    kv = len(set(int(x) for f in out for x in f))
    print(f"[collar] donor trimmed to the detail + {frac:.2f}x its radius "
          f"({frac * R:.5f} along the surface): {len(out)} of {len(Fd)} faces, {kv} of {len(W)} "
          f"verts, {bestc} of {int(core[Fd].any(1).sum())} detail faces kept", flush=True)
    return out


def _notch_trim(ob, W, Fd):
    """Drop donor skin that has no body underneath it.

    The donor's lower skin runs down into the notch between the legs, where the body is not a
    surface at all -- it is a gap with a thigh wall either side. Those rim vertices have nothing
    to stitch to, so the bridge pairs them with whatever is nearest around the loop (the far
    thigh) and fans a sheet of long triangles straight across the gap. That sheet is the spike,
    and it is the same shape as the "walls" this whole exercise started from.

    Inventing a hole boundary in mid-air is not possible, so do the other thing: the donor may
    only keep skin that has body under it. Looked for along each face's OWN normal, both ways,
    within the donor's own worst standoff -- no chosen distance. The labelled detail is exempt:
    the vulva is a recess, the body has nothing under it by definition, and that is the entire
    reason the graft exists."""
    from collections import defaultdict, deque
    Fd = np.asarray(Fd, int)
    try:
        Z = np.load(A_ZONES)
        core = np.isin(Z, [1, 2, 3, 4]) if len(Z) == len(W) else np.zeros(len(W), bool)
    except Exception:
        core = np.zeros(len(W), bool)
    reach = 0.0
    for i in range(len(W)):
        ok, cp, nn, ix = ob.closest_point_on_mesh(Vector(W[i].tolist()))
        if ok:
            reach = max(reach, float((Vector(W[i].tolist()) - cp).length))
    if reach <= 0.0:
        return Fd
    # Ask the question per VERTEX and smooth the answer, then take a level set. Deciding face
    # by face gives a fringe rather than a line -- measured, the rim's perimeter went
    # 0.2384 -> 0.4875, the hole came out SHORTER than the rim so the parity solver gave up
    # (factor 0.000), and the bridge spanned 194 mm further down the leg. A level set of a
    # smoothed field is a smooth curve by construction, and 0.5 is not a tuned threshold: it is
    # the halfway point of a 0/1 indicator.
    Vn = np.zeros_like(W)
    _fn = np.cross(W[Fd[:, 1]] - W[Fd[:, 0]], W[Fd[:, 2]] - W[Fd[:, 0]])
    for _k in range(3):
        np.add.at(Vn, Fd[:, _k], _fn)
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
    used = np.zeros(len(W), bool); used[Fd.ravel()] = True
    air = np.zeros(len(W))
    nair = 0
    for i in np.where(used & ~core)[0]:
        o = Vector(W[i].tolist()); n = Vector(Vn[i].tolist())
        if not (ob.ray_cast(o, -n, distance=reach)[0] or ob.ray_cast(o, n, distance=reach)[0]):
            air[i] = 1.0; nair += 1
    nb = defaultdict(set)
    for f in Fd:
        x, y, z = int(f[0]), int(f[1]), int(f[2])
        nb[x].update((y, z)); nb[y].update((x, z)); nb[z].update((x, y))
    order = np.where(used)[0]
    flat = np.concatenate([np.fromiter(nb[int(i)], int, len(nb[int(i)])) for i in order])
    ptr = np.cumsum([0] + [len(nb[int(i)]) for i in order])[:-1]
    cnt = np.array([len(nb[int(i)]) for i in order], float)
    for _ in range(NOTCH_SMOOTH_ITERS):
        air[order] = np.add.reduceat(air[flat], ptr) / np.maximum(cnt, 1.0)
        air[core] = 0.0                      # the vulva is a recess; it never counts as airborne
    keep = air[Fd].mean(1) < 0.5
    keep |= core[Fd].any(1)
    idx = np.where(keep)[0]
    fadj = defaultdict(list)
    for k in idx:
        f = Fd[k]
        for e in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
            fadj[(min(e), max(e))].append(int(k))
    inside = set(int(k) for k in idx)
    seen = set(); best = None; bestc = -1
    for s0 in inside:
        if s0 in seen:
            continue
        comp = {s0}; dq = deque([s0]); seen.add(s0)
        while dq:
            x = dq.popleft(); f = Fd[x]
            for e in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
                for g in fadj[(min(e), max(e))]:
                    if g in inside and g not in seen:
                        seen.add(g); comp.add(g); dq.append(g)
        nc = int(sum(1 for x in comp if core[Fd[x]].any()))
        if nc > bestc:
            bestc, best = nc, comp
    out = np.asarray([Fd[i] for i in sorted(best)], int)
    if len(out) < 0.25 * len(Fd):
        print(f"[notch] trim would leave only {len(out)} of {len(Fd)} faces -- refusing", flush=True)
        return Fd
    print(f"[notch] {nair} donor vert(s) see no body within {reach:.5f} along their own normal "
          f"(skin hanging over the leg gap); after {NOTCH_SMOOTH_ITERS} diffusion passes the 0.5 "
          f"level set keeps {len(out)} of {len(Fd)} faces, {bestc} of "
          f"{int(core[Fd].any(1).sum())} detail faces", flush=True)
    return out


def _rim_relax(W, Fd, iters, rings):
    """Straighten the donor's boundary after the notch trim, and let the skin behind it follow.

    Trimming the skin that hangs over the leg gap has to cut along the notch edge, and at 1.5 mm
    mesh resolution that edge is a zigzag: measured, the rim's perimeter went 0.2384 -> 0.3300
    for a SMALLER patch. A wiggly rim cannot be matched to the body's hole -- the parity solver
    gave up and the bridge webbed on the thigh instead. Diffusing the trim field harder only
    trades it back (80 passes trims 4 faces of 10455, i.e. nothing).

    So trim hard, then relax the boundary it leaves. This is donor SKIN: its outline carries no
    anatomy, and the rim gets snapped onto the body a few lines later anyway. The labelled detail
    is pinned, and the displacement is diffused inward over `rings` so the patch is carried
    rather than torn at the edge."""
    from collections import defaultdict
    Fd = np.asarray(Fd, int)
    try:
        Z = np.load(A_ZONES)
        core = np.isin(Z, [1, 2, 3, 4]) if len(Z) == len(W) else np.zeros(len(W), bool)
    except Exception:
        core = np.zeros(len(W), bool)
    ec = defaultdict(int)
    for f in Fd:
        for e in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
            ec[(min(e), max(e))] += 1
    adj = defaultdict(list)
    for (a, b), n in ec.items():
        if n == 1:
            adj[a].append(b); adj[b].append(a)
    if not adj:
        return W
    seen = set(); loops = []
    for s0 in list(adj):
        if s0 in seen:
            continue
        lp = [s0]; seen.add(s0); prev = None; cur = s0
        while True:
            nx = [w for w in adj[cur] if w != prev and w not in seen]
            if not nx:
                break
            n2 = nx[0]; lp.append(n2); seen.add(n2); prev = cur; cur = n2
        if len(lp) > 6:
            loops.append(lp)
    if not loops:
        return W
    loops.sort(key=len, reverse=True)
    loop = np.asarray(loops[0], int)
    P0 = W[loop].copy()
    P = P0.copy()
    for _ in range(iters):
        P = P + 0.5 * (0.5 * (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0)) - P)
    def _per(Q):
        return float(np.linalg.norm(Q - np.roll(Q, -1, axis=0), axis=1).sum())
    D = np.zeros_like(W)
    D[loop] = P - P0
    nb = defaultdict(set)
    for f in Fd:
        x, y, z = int(f[0]), int(f[1]), int(f[2])
        nb[x].update((y, z)); nb[y].update((x, z)); nb[z].update((x, y))
    used = np.zeros(len(W), bool); used[Fd.ravel()] = True
    order = np.where(used)[0]
    flat = np.concatenate([np.fromiter(nb[int(i)], int, len(nb[int(i)])) for i in order])
    ptr = np.cumsum([0] + [len(nb[int(i)]) for i in order])[:-1]
    cnt = np.array([len(nb[int(i)]) for i in order], float)[:, None]
    pin = np.zeros(len(W), bool); pin[loop] = True
    for _ in range(rings):
        Dn = np.add.reduceat(D[flat], ptr, axis=0) / cnt
        D[order] = Dn
        D[loop] = P - P0
        D[core] = 0.0
    W = W + D
    print(f"[rimrelax] boundary {len(loop)} verts: perimeter {_per(P0):.4f} -> {_per(P):.4f} "
          f"({iters} passes); carried {rings} rings into the patch, labelled detail pinned; "
          f"skin moved med {np.median(np.linalg.norm(D[used & ~core], axis=1)):.5f} max "
          f"{np.linalg.norm(D, axis=1).max():.5f}", flush=True)
    return W



def _rim_band_mask(W, Fd, loop, frac):
    """Donor verts within `frac` x the apron width of the rim, geodesic along the donor's own edges. Apron width = the
    median over rim verts of the geodesic distance to the nearest labelled vert (zones 1-4). None if the zones do not
    match the donor."""
    import heapq as _hq9
    try:
        _zn9 = np.load(A_ZONES)
    except Exception:
        return None
    W = np.asarray(W, float)
    if len(_zn9) != len(W):
        return None
    _adj9 = [[] for _ in range(len(W))]
    for _f9 in Fd:
        _a9, _b9, _c9 = int(_f9[0]), int(_f9[1]), int(_f9[2])
        for _p9, _q9 in ((_a9, _b9), (_b9, _c9), (_c9, _a9)):
            _adj9[_p9].append(_q9); _adj9[_q9].append(_p9)

    def _geo9(seeds):
        _d9 = np.full(len(W), np.inf)
        _pq9 = []
        for _s9 in seeds:
            _d9[_s9] = 0.0; _pq9.append((0.0, int(_s9)))
        _hq9.heapify(_pq9)
        while _pq9:
            _dd9, _u9 = _hq9.heappop(_pq9)
            if _dd9 > _d9[_u9]:
                continue
            for _w9 in _adj9[_u9]:
                _nd9 = _dd9 + float(np.linalg.norm(W[_u9] - W[_w9]))
                if _nd9 < _d9[_w9]:
                    _d9[_w9] = _nd9; _hq9.heappush(_pq9, (_nd9, _w9))
        return _d9
    _lab9 = np.nonzero(_zn9 >= 1)[0]
    if not len(_lab9):
        return None
    _drim9 = _geo9([int(li) for li in loop])
    _dlab9 = _geo9(_lab9)
    _apw9 = float(np.median(_dlab9[[int(li) for li in loop]]))
    if not np.isfinite(_apw9) or _apw9 <= 0.0:
        return None
    return _drim9 <= frac * _apw9


def _stage_dump_np(tag, W, Fd):
    """Same as _stage_dump, for the donor while it is still a bare array (before it joins the body)."""
    _dir = os.environ.get("PELVIS_DUMP_STAGES", "")
    if not _dir:
        return
    try:
        os.makedirs(_dir, exist_ok=True)
        _nl = chr(10)
        _path = os.path.join(_dir, tag + ".obj")
        with open(_path, "w") as _fh:
            _fh.write("".join("v %.7f %.7f %.7f" % tuple(v) + _nl for v in np.asarray(W, float)))
            _fh.write("".join("f " + " ".join(str(int(j) + 1) for j in f) + _nl for f in Fd))
        np.save(os.path.join(_dir, tag + "_tag.npy"), np.ones(len(W), bool))
        print(f"[stage] {tag}: donor only, {len(W)} verts, {len(Fd)} faces -> {_path}", flush=True)
    except Exception as _e:
        print(f"[stage] {tag} dump failed ({_e!r})", flush=True)


def _stage_dump(tag, bm, A):
    """Write the mesh near the graft AS IT IS at this stage, plus the donor tag, so each stage's
    effect on the donor can be measured instead of inferred from ablations (switching one pass
    off changes what every later pass sees)."""
    _dir = os.environ.get("PELVIS_DUMP_STAGES", "")
    if not _dir:
        return
    try:
        os.makedirs(_dir, exist_ok=True)
        bm.verts.ensure_lookup_table()
        _lay = bm.verts.layers.int.get("pg_donor")
        _co = np.array([v.co[:] for v in bm.verts], float)
        _near = np.where(np.linalg.norm(_co - np.asarray(A, float), axis=1) < 0.07)[0]
        _vs = [bm.verts[int(i)] for i in _near]
        _vm = {v: k for k, v in enumerate(_vs)}
        _fs = set()
        for v in _vs:
            for f in v.link_faces:
                _fs.add(f)
        _rows = [[_vm[v] for v in f.verts] for f in _fs if all(v in _vm for v in f.verts)]
        _path = os.path.join(_dir, tag + ".obj")
        _nl = chr(10)
        with open(_path, "w") as _fh:
            _fh.write("".join("v %.7f %.7f %.7f" % tuple(_co[int(i)]) + _nl for i in _near))
            _fh.write("".join("f " + " ".join(str(j + 1) for j in r) + _nl for r in _rows))
        _tg = np.array([(v[_lay] > 0) if _lay is not None else False for v in _vs], bool)
        np.save(os.path.join(_dir, tag + "_tag.npy"), _tg)
        print(f"[stage] {tag}: {len(_vs)} verts, {len(_rows)} faces, {int(_tg.sum())} donor -> {_path}", flush=True)
    except Exception as _e:
        print(f"[stage] {tag} dump failed ({_e!r})", flush=True)


def _dtrace(tag, bm, lay, Wref):
    """How far has any DONOR vertex drifted from where it was placed, at this point?

    Donor vertices were turning up 200 mm down the thigh and ablating the relax passes never
    stopped it, so ask each stage directly. Uses the attribute layer, which survives the
    reindexing that invalidates index lists and BMVert references."""
    if os.environ.get("PELVIS_DTRACE", "") != "1":
        return
    try:
        _kd = kdtree.KDTree(len(Wref))
        for _i in range(len(Wref)):
            _kd.insert(Vector(Wref[_i].tolist()), _i)
        _kd.balance()
        _co = np.array([v.co[:] for v in bm.verts if v[lay] > 0], float)
        if not len(_co):
            print(f"[dtrace] {tag:<28} no donor verts", flush=True); return
        _d = np.array([_kd.find(Vector(c.tolist()))[2] for c in _co])
        print(f"[dtrace] {tag:<28} {len(_co)} donor verts | drift p99 "
              f"{np.percentile(_d, 99) / 1.0016 * 1700:6.2f} MAX {_d.max() / 1.0016 * 1700:7.2f} mm "
              f"| {int((_d > 0.006).sum())} over 10mm", flush=True)
    except Exception as _e:
        print(f"[dtrace] {tag:<28} failed ({_e!r})", flush=True)


def _new_obj(V, F, name="m"):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], [list(map(int, f)) for f in F]); me.update()
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    return o, me


def _voxel(o, vox):
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    m = o.modifiers.new("vr", 'REMESH'); m.mode = 'VOXEL'; m.voxel_size = vox; m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="vr")


def _np_verts(me):
    return np.array([v.co for v in me.vertices])


def _adj(F2, mask, pad=0.01, dA=None, lim=None):
    a = defaultdict(set)
    for fl in F2:
        if (dA is None) or (dA[list(fl)].min() < lim + pad):
            n = len(fl)
            for k in range(n):
                x, y = fl[k], fl[(k + 1) % n]; a[x].add(y); a[y].add(x)
    return a


def _grade_ring(ob, A, r1, r2, cuts):
    """Locally refine the body around the graft anchor so the donor->body density step is walked
    down over two bands instead of one shell. Inner band gets `cuts` cuts, mid band gets one fewer.
    Adds geometry ONLY inside r2 of the anchor; the rest of the body is untouched."""
    A = np.asarray(A, float)
    for lo, hi, nc in ((0.0, r1, cuts), (r1, r2, max(cuts - 1, 1))):
        if nc <= 0: continue
        bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table()
        es = set()
        for f in bm.faces:
            d = float(np.linalg.norm(np.array(f.calc_center_median()) - A))
            if lo <= d < hi:
                for e in f.edges: es.add(e)
        n0 = len(bm.faces)
        if es:
            bmesh.ops.subdivide_edges(bm, edges=list(es), cuts=nc, use_grid_fill=True)
        print(f"[grade] band {lo:.3f}-{hi:.3f}: {len(es)} edges, {nc} cuts -> faces {n0} -> {len(bm.faces)}", flush=True)
        bm.to_mesh(ob.data); bm.free()

def _graft_cutjoin(ob, W, Fd, A):
    """Cut-and-JOIN graft (KAN-18, 2026-06-24, HEADLESS). Flood-cut the body under the donor (this OPENS the
    vagina/anus recesses instead of the body meshing a wall behind them), stitch the donor OUTER rim to the
    hole by angle (proportional -> cannot twist), weld leftover branch edges, then smooth ONLY a 7mm ring at
    the seam (vulva + surround never touched). Body MUST be watertight at the crotch (gen_seams remesh /
    voxel_body) -- raw meshes are open there and leave gaps. Modifies ob.data in place; returns (ob, Wp)."""
    # ===== GRAFT donor geometry + seam-RING smooth (2026-06-25, user-directed) =====
    # Snapping re-sampled the donor with the body's coarse verts -> jagged interior (spikes by the vagina) and
    # the smooth ate the labia. Instead GRAFT the donor's OWN geometry (= the user's sculpt = clean detail):
    # flood-cut the body under the donor footprint, stitch the donor rim to the hole, then smooth ONLY the
    # seam RING (verts within SEAM_BAND of the rim line, BOTH sides). The inner detail (labia/vagina/anus,
    # farther than SEAM_BAND from the rim) is 100% PROTECTED. Everything below `return ob, W` is DEAD code.
    from collections import defaultdict, deque
    A = np.asarray(A, float); W = np.asarray(W, float); me = ob.data
    Vb = _np_verts(me)
    fn = np.zeros(3)
    for f in Fd:
        a, b, c = W[int(f[0])], W[int(f[1])], W[int(f[2])]; fn = fn + np.cross(b - a, c - a)
    outw = fn / (np.linalg.norm(fn) + 1e-12)
    if outw @ (A - Vb.mean(0)) < 0: outw = -outw
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    # How much of the donor is BURIED in the body? A donor vertex whose nearest body point has the
    # body surface facing away from it is inside the flesh -- that is the part of the graft that is
    # "too wide" and just overlaps the inner thigh instead of showing.
    _ins = np.zeros(len(W), bool); _dep = np.zeros(len(W)); _dall = np.zeros(len(W))
    for _i in range(len(W)):
        _ok, _cp, _nn, _ix = ob.closest_point_on_mesh(Vector(W[_i].tolist()))
        if not _ok: continue
        _d = W[_i] - np.array(_cp.to_tuple())
        _dall[_i] = float(np.linalg.norm(_d))
        if float(np.dot(_d, np.array(_nn.to_tuple()))) < 0.0:
            _ins[_i] = True; _dep[_i] = _dall[_i]
    _stand = float(_dall.max())     # how far the donor ever gets from the skin it replaces
    if _ins.any():
        _ax = np.abs(W[:, 0] - A[0])
        print(f"[cutjoin] donor overlap probe: {int(_ins.sum())}/{len(W)} verts "
              f"({100.0*_ins.mean():.1f}%) inside the body; depth med {np.median(_dep[_ins]):.5f} "
              f"max {_dep.max():.5f}; |x-A| inside med {np.median(_ax[_ins]):.5f} vs outside med "
              f"{np.median(_ax[~_ins]):.5f}; donor half-width {_ax.max():.5f}", flush=True)
        _qi = np.percentile(_ax[_ins], [50,75,90,95,99]); _qo = np.percentile(_ax[~_ins], [50,75,90,95,99])
        print(f"[cutjoin]   |x| buried  p50/75/90/95/99 = " + "/".join(f"{q:.5f}" for q in _qi), flush=True)
        print(f"[cutjoin]   |x| visible p50/75/90/95/99 = " + "/".join(f"{q:.5f}" for q in _qo), flush=True)
        if TRIM_K > 0.0 and (~_ins).any():
            _xk = TRIM_K * float(np.percentile(_ax[~_ins], 95))
            _keep = [f for f in Fd if min(_ax[int(f[0])], _ax[int(f[1])], _ax[int(f[2])]) <= _xk]
            _cut = len(Fd) - len(_keep)
            if _cut and _cut <= TRIM_MAX * len(Fd):
                print(f"[cutjoin] donor trim: keep |x-A| <= {_xk:.5f} ({TRIM_K:.2f}x visible p95); "
                      f"dropped {_cut} of {len(Fd)} donor faces "
                      f"(half-width {_ax.max():.5f} -> {_xk:.5f})", flush=True)
                Fd = _keep
                # the rim, the footprint and the seam all derive from Fd, so they follow automatically
                fn = np.zeros(3)
                for f in Fd:
                    a, b, c = W[int(f[0])], W[int(f[1])], W[int(f[2])]; fn = fn + np.cross(b - a, c - a)
                outw = fn / (np.linalg.norm(fn) + 1e-12)
                if outw @ (A - Vb.mean(0)) < 0: outw = -outw
                _t = np.array([1.0, 0, 0])
                if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
                u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
            elif _cut:
                print(f"[cutjoin] donor trim SKIPPED: would drop {_cut} of {len(Fd)} faces "
                      f"({100.0*_cut/len(Fd):.0f}% > {100.0*TRIM_MAX:.0f}% cap)", flush=True)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adj[a].append(b); adj[b].append(a)
    seen = set(); loops = []
    for s in list(adj):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; cur = s
        while True:
            nx = [w for w in adj[cur] if w != prev and w not in seen]
            if not nx: break
            n = nx[0]; lp.append(n); seen.add(n); prev = cur; cur = n
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); loop = loops[0]
    _stage_dump_np("0a_placed", W, Fd)
    _pre = np.array([W[li] for li in loop])
    _cand = []
    for li in loop:
        o = Vector(W[li].tolist()); ok, cp, nn, ix = ob.closest_point_on_mesh(o)
        _cand.append(np.array(cp.to_tuple()) if ok else np.array(W[li], float))
    _cand = np.array(_cand)
    _disp = np.linalg.norm(_cand - _pre, axis=1)
    _lim = RIM_SNAP_CLAMP * float(np.median(_disp))
    _clamped = 0
    for _k, li in enumerate(loop):
        _v = _cand[_k] - _pre[_k]; _n = _disp[_k]
        if _lim > 0 and _n > _lim:
            _v = _v * (_lim / _n); _clamped += 1
        W[li] = _pre[_k] + _v
    print(f"[cutjoin] rim snap: median {np.median(_disp):.5f} max {_disp.max():.5f}; "
          f"clamp {RIM_SNAP_CLAMP}x median = {_lim:.5f} -> clamped {_clamped}/{len(loop)}", flush=True)
    # Nearest-point projection is NOT injective: neighbouring rim verts land on the same body
    # point (measured min spacing 0.00000 after snap, vs 0.00025 before) while others stretch,
    # taking max/median spacing from 3.2 to 5.2. Those degenerate and stretched segments become
    # the sliver fan above the vulva when the rim is stitched to the hole.
    # Re-space the rim UNIFORMLY along its own snapped polyline: the loop keeps exactly the shape
    # the snap gave it (so it still lies on the body), only the vertex distribution is evened out.
    _P = np.array([W[li] for li in loop])
    _d = np.linalg.norm(np.roll(_P, -1, axis=0) - _P, axis=1)
    _cum = np.concatenate([[0.0], np.cumsum(_d)]); _tot = _cum[-1]
    if _tot > 1e-9:
        _tg = np.arange(len(loop)) * (_tot / len(loop))
        if RIM_KEEP_SPACING:
            _dpre = np.linalg.norm(np.roll(_pre, -1, axis=0) - _pre, axis=1)
            _cpre = np.concatenate([[0.0], np.cumsum(_dpre)])
            if _cpre[-1] > 1e-12:
                _tg = _cpre[:-1] / _cpre[-1] * _tot
                print(f"[cutjoin] rim re-space: keeping the donor's own spacing (each vertex at its "
                      f"original arc-length fraction), not uniform", flush=True)
        _new = np.empty_like(_P)
        for _k, _t in enumerate(_tg):
            _i = min(int(np.searchsorted(_cum, _t, side='right')) - 1, len(_P) - 1)
            _seg = _d[_i] if _d[_i] > 1e-12 else 1.0
            _f = (_t - _cum[_i]) / _seg
            _new[_k] = _P[_i] * (1.0 - _f) + _P[(_i + 1) % len(_P)] * _f
        for _k, li in enumerate(loop):
            W[li] = _new[_k]
    _post = np.array([W[li] for li in loop])
    _stage_dump_np("0b_rim_snapped_respaced", W, Fd)
    def _seg(P):
        d = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        return d.min(), np.median(d), d.max()
    _b = _seg(_pre); _a = _seg(_post)
    _mv = np.linalg.norm(_post - _pre, axis=1)
    print(f"[cutjoin] rim spacing BEFORE snap min/med/max = {_b[0]:.5f}/{_b[1]:.5f}/{_b[2]:.5f}", flush=True)
    print(f"[cutjoin] rim spacing AFTER  snap min/med/max = {_a[0]:.5f}/{_a[1]:.5f}/{_a[2]:.5f}", flush=True)
    print(f"[cutjoin] snap moved verts med {np.median(_mv):.5f} max {_mv.max():.5f}; "
          f"spacing ratio max/med {_b[2]/max(_b[1],1e-9):.1f} -> {_a[2]/max(_a[1],1e-9):.1f}", flush=True)
    # SEAT the rim: project and re-space until it is BOTH on the surface and evenly spaced.
    # One pass of each is not enough and the two fight: the projection puts every vertex on the
    # body, then the re-space walks a straight chord between them and lifts the curve back off a
    # curved surface -- measured 9.6 mm, which is 6 body faces. That is why marking "the faces
    # the rim crosses" produced a ring full of gaps both times it was tried; the ring was built
    # from a curve that is not on the mesh. Alternating the two converges: each projection has
    # less to correct because the spacing is already even, and each re-space moves less because
    # the vertices are already spread. Reported per iteration, so a mesh where it does NOT
    # converge says so instead of silently handing a floating curve to the cut.
    if RIM_SEAT_ITERS > 0:
        def _rimoff(P):
            _d = []
            for _q in P:
                _ok2, _cp2, _n2, _i2 = ob.closest_point_on_mesh(Vector(_q.tolist()))
                _d.append(float((Vector(_q.tolist()) - _cp2).length) if _ok2 else 0.0)
            return np.asarray(_d)
        _P = np.array([W[li] for li in loop], float)
        _o0 = _rimoff(_P)
        for _it in range(RIM_SEAT_ITERS):
            for _k in range(len(_P)):                       # project
                _ok2, _cp2, _n2, _i2 = ob.closest_point_on_mesh(Vector(_P[_k].tolist()))
                if _ok2:
                    _P[_k] = np.array(_cp2.to_tuple())
            _d2 = np.linalg.norm(np.roll(_P, -1, axis=0) - _P, axis=1)   # re-space
            _cu = np.concatenate([[0.0], np.cumsum(_d2)]); _tot2 = _cu[-1]
            if _tot2 <= 1e-9:
                break
            _tg2 = np.arange(len(_P)) * (_tot2 / len(_P))
            _nw = np.empty_like(_P)
            for _k, _t2 in enumerate(_tg2):
                _i3 = min(int(np.searchsorted(_cu, _t2, side='right')) - 1, len(_P) - 1)
                _sg = _d2[_i3] if _d2[_i3] > 1e-12 else 1.0
                _f4 = (_t2 - _cu[_i3]) / _sg
                _nw[_k] = _P[_i3] * (1.0 - _f4) + _P[(_i3 + 1) % len(_P)] * _f4
            _P = _nw
            _oN = _rimoff(_P)
            _sp2 = np.linalg.norm(np.roll(_P, -1, axis=0) - _P, axis=1)
            print(f"[cutjoin] rim seat {_it + 1}/{RIM_SEAT_ITERS}: off the surface med "
                  f"{np.median(_oN):.5f} max {_oN.max():.5f}; spacing min/med/max "
                  f"{_sp2.min():.5f}/{np.median(_sp2):.5f}/{_sp2.max():.5f}", flush=True)
        _oF = _rimoff(_P)
        if RIM_CONFORM or APRON_CONFORM:
            _dlt = np.zeros_like(W)
            for _k, li in enumerate(loop):
                _dlt[li] = _P[_k] - W[li]
            _fixed = np.zeros(len(W), bool)
            _fixed[np.asarray(loop, int)] = True
            _held = np.zeros(len(W), bool)
            try:
                _zc = np.load(A_ZONES)
                if len(_zc) == len(W):
                    _held = np.isin(_zc, [1, 2, 3, 4])       # labia + opening: correction is 0 there
            except Exception:
                pass
            _held &= ~_fixed
            _nbd = [[] for _ in range(len(W))]
            for _f5 in Fd:
                _a5, _b5, _c5 = int(_f5[0]), int(_f5[1]), int(_f5[2])
                _nbd[_a5].extend((_b5, _c5)); _nbd[_b5].extend((_a5, _c5)); _nbd[_c5].extend((_a5, _b5))
            _free = ~(_fixed | _held)
            _fl5 = []; _of5 = [0]
            _fi5 = np.where(_free)[0]
            for _i5 in _fi5:
                _ns5 = sorted(set(_nbd[_i5]))
                _fl5.extend(_ns5); _of5.append(len(_fl5))
            _fl5 = np.asarray(_fl5 + [0], np.int64)
            _cn5 = np.diff(np.asarray(_of5, np.int64))
            _sg5 = np.asarray(_of5[:-1], np.int64)
            _ok5 = _cn5 > 0
            for _ in range(400):                              # harmonic: mean of the neighbours
                _sm5 = np.add.reduceat(_dlt[_fl5], _sg5, axis=0)
                _dlt[_fi5] = np.where(_ok5[:, None], _sm5 / np.maximum(_cn5, 1)[:, None], _dlt[_fi5])
            if APRON_CONFORM:
                # the same harmonic solve, on a SCALAR: 1 on the rim, 0 on the labelled detail
                _wf = np.zeros(len(W)); _wf[np.asarray(loop, int)] = 1.0
                for _ in range(400):
                    _sw6 = np.add.reduceat(_wf[_fl5], _sg5)
                    _wf[_fi5] = np.where(_ok5, _sw6 / np.maximum(_cn5, 1), _wf[_fi5])
                # The zone labels cover the labia and the opening; the ANUS is plain zone 0, so
                # without this the conform pulls the dimple flat onto the body -- visible as a
                # crumpled perineum. Reuse the graft's own anus anchor and its own ANUS_R/ANUS_FALL
                # guard rather than inventing a second definition of where the anus is.
                _midA = W[np.abs(W[:, 0] - A[0]) < 0.02] if APRON_ANUS else W[:0]
                if len(_midA):
                    _bkA = _midA[_midA[:, 2] >= np.percentile(_midA[:, 2], 72)]
                    _anA = np.array([A[0], float(_bkA[:, 1].mean()), float(_bkA[:, 2].mean())])
                    _dA6 = np.linalg.norm(W - _anA, axis=1)
                    _gA = np.clip((_dA6 - ANUS_R) / max(ANUS_FALL, 1e-9), 0.0, 1.0)
                    _nA = int((_gA < 1.0).sum())
                    _wf = _wf * _gA
                    print(f"[cutjoin] apron conform: anus guard at {np.round(_anA, 4)} holds "
                          f"{_nA} vert(s) back ({int((_gA <= 0.0).sum())} of them completely) so "
                          f"the dimple is not pulled flat", flush=True)
                # donor vertex normals (consistent winding, oriented away from the body centre)
                _dnA = np.zeros((len(W), 3))
                for _f6 in Fd:
                    _j0, _j1, _j2 = int(_f6[0]), int(_f6[1]), int(_f6[2])
                    _n6 = np.cross(W[_j1] - W[_j0], W[_j2] - W[_j0])
                    _dnA[_j0] += _n6; _dnA[_j1] += _n6; _dnA[_j2] += _n6
                _l6 = np.linalg.norm(_dnA, axis=1); _l6[_l6 < 1e-15] = 1.0
                _dnA = _dnA / _l6[:, None]
                if float(_dnA.sum(0) @ outw) < 0.0:
                    _dnA = -_dnA
                _rayL = 2.0 * float(_o0.max())        # measured on THIS donor and THIS body
                _hit = 0; _miss = 0
                for _i6 in _fi5.tolist():
                    if _wf[_i6] <= 1e-6:
                        continue
                    _o6 = Vector(W[_i6].tolist()); _d6 = Vector(_dnA[_i6].tolist())
                    _best = None
                    for _sgn in (-1.0, 1.0):
                        _r6 = ob.ray_cast(_o6, _d6 * _sgn, distance=_rayL)
                        if _r6[0]:                       # (hit, location, normal, index)
                            _t6 = np.array(_r6[1].to_tuple())
                            if _best is None or np.linalg.norm(_t6 - W[_i6]) < np.linalg.norm(_best - W[_i6]):
                                _best = _t6
                    if _best is None:
                        _miss += 1
                        continue
                    _hit += 1
                    _dlt[_i6] = _wf[_i6] * (_best - W[_i6])
                print(f"[cutjoin] apron conform: {_hit} apron vert(s) given their own target on the "
                      f"body along the donor normal within {_rayL / 1.0016 * 1700:.0f} mm "
                      f"({_miss} found no body and were left where the rim's diffusion put them); "
                      f"the move is scaled by a harmonic weight that is 1 on the rim and 0 on the "
                      f"labelled detail", flush=True)
            _mv5 = np.linalg.norm(_dlt, axis=1)
            W = W + _dlt
            print(f"[cutjoin] rim conform{' + apron' if APRON_CONFORM else ''}: seated the rim ON the body (moved med "
                  f"{np.median(_mv5[_fixed]) / 1.0016 * 1700:.1f} mm) and diffused that correction "
                  f"through {int(_free.sum())} apron vert(s) (med "
                  f"{np.median(_mv5[_free]) / 1.0016 * 1700:.1f} mm, max "
                  f"{_mv5[_free].max() / 1.0016 * 1700:.1f} mm); {int(_held.sum())} labelled "
                  f"labia/opening vert(s) held at exactly 0", flush=True)
        else:
            for _k, li in enumerate(loop):
                W[li] = _P[_k]
        print(f"[cutjoin] rim seated: off the surface med {np.median(_o0):.5f} -> "
              f"{np.median(_oF):.5f}, max {_o0.max():.5f} -> {_oF.max():.5f}", flush=True)
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in loop])
    def _inp(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    print("[t] prologue: bmb built, starting footprint", flush=True)
    bmb = bmesh.new(); bmb.from_mesh(me); bmb.faces.ensure_lookup_table()
    # Tag donor vertices with a mesh ATTRIBUTE, not an index list: bmesh ops below (weld_verts
    # among them) reindex, and indices captured then point at unrelated vertices. The layer is
    # created HERE, before anything holds a BMVert reference -- layers.int.new() reallocates and
    # invalidates every reference across it (it emptied the hole-boundary set when I created it
    # later, and the hole walk came back with nothing).
    _dlay = bmb.verts.layers.int.get("pg_donor") or bmb.verts.layers.int.new("pg_donor")
    cand = set()
    # Reject INNER-THIGH faces. The footprint test is a projection, so once the anchor sits at
    # the real crotch the inner thighs -- which are laterally offset, not under the donor --
    # project inside the rim polygon too and the flood cuts them away: the hole went from 1.60x
    # the rim to 3.04x (nH 268 -> 520). They are separable by orientation, the same signature
    # _crotch_y uses: inner-thigh surfaces face each other sideways (|nx| 0.93-0.97) while the
    # perineum/mons faces down and front (|nx| ~0.4-0.67). Gate on it, relative to nothing --
    # a face whose normal is that lateral is thigh, not graft bed.
    # Tied to AUTO_ANCHOR: the leak only happens once the anchor sits at the real crotch,
    # and enabling the anchor WITHOUT this gate is the worst of both (hole 3.04x the rim,
    # nH 520 -- the flood eats the crotch and the graft disappears into a stretched bridge).
    # Independently switchable so the donor-extent bound can be tested on the SHIPPING
    # anchor too, where the footprint is unbounded (2311 faces vs 808).
    # OFF. Measured: with the anchor at the crotch these three made the graft WORSE, not better.
    # Plain cut, anchor moved:            gap med 0.5x  max 18.7x   (shipping is 0.4x / 22.0x)
    # + lateral gate / rim-cut / bounds:  gap med 8.6x  max 26.2x   -- donor buried, back collapsed
    # The flood takes the connected component containing the seed, so bounding the CANDIDATES
    # was a no-op on the hole (four separate changes produced byte-identical geometry); all they
    # did was starve the flood until the body stayed closed over the donor. Do not re-enable
    # without re-measuring the post-snap gap, which is the number the bridge actually spans.
    _rimcut_bound = os.environ.get("PELVIS_CUT_BOUND", "0") == "1"
    _LAT = float(os.environ.get("PELVIS_CUT_LATERAL", "2.0"))   # >1 = off (see above)
    # DONOR-EXTENT bound. The footprint is a planar shadow, so at the true crotch it also lands
    # on the BACK of the body -- the anus and the cleft between the buttocks -- and the flood
    # ate them (visible as a collapsed backside). The donor only ever replaces what it covers,
    # so nothing outside its own placed bounding box may be cut. No margin, no constant: the
    # donor's extent IS the bound.
    _dlo = W.min(0); _dhi = W.max(0)
    _nlat = 0; _next = 0

    # ---- ONE local index set, built once, used by every pass below ------------------------
    # Every stage of this function used to walk all 2.18M faces calling calc_center_median()
    # and throw away everything past ~0.12 of the anchor. Profiled: the footprint scan alone
    # was 167 s of a 608 s graft, and the same shape of loop appears in the mons de-bump, both
    # relax gathers and the flood seed.
    #
    # Vertices come out as one numpy array (a single pass, ~1 s), the near ones are selected
    # vectorised, and the faces are reached through THEM -- so the face loop is proportional to
    # the graft, not to the character. Radius is generous enough to cover every consumer's own
    # cut-off; each still applies its own.
    print("[t] prologue: gates start, building local set", flush=True)
    _LOCR = 0.16
    _Vall = np.array([v.co[:] for v in bmb.verts], float)
    bmb.verts.ensure_lookup_table()
    _nearv = np.where(np.linalg.norm(_Vall - A, axis=1) < _LOCR)[0]
    _localf = set()
    for _i in _nearv:
        for _lf in bmb.verts[int(_i)].link_faces:
            _localf.add(_lf.index)
    bmb.faces.ensure_lookup_table()
    _LOCAL = [bmb.faces[i] for i in sorted(_localf)]
    _LOCC = {f.index: np.array(f.calc_center_median()) for f in _LOCAL}
    print(f"[cutjoin] local set: {len(_nearv)} vert(s), {len(_LOCAL)} face(s) within {_LOCR} "
          f"of the anchor (body has {len(bmb.verts)} / {len(bmb.faces)})", flush=True)

    def _flood(_c, _seed=None):
        if not _c:
            return set()
        def _cen(_i):
            _q = _LOCC.get(_i)
            return _q if _q is not None else np.array(bmb.faces[_i].calc_center_median())
        _pool = _c
        if _seed is None and FLOOD_SEED == "largest":
            _seen7 = set(); _big7 = set()
            for _s7 in _c:
                if _s7 in _seen7:
                    continue
                _c7 = {_s7}; _dq7 = deque([_s7]); _seen7.add(_s7)
                while _dq7:
                    _x7 = _dq7.popleft()
                    for _e7 in bmb.faces[_x7].edges:
                        for _g7 in _e7.link_faces:
                            if _g7.index in _c and _g7.index not in _seen7:
                                _seen7.add(_g7.index); _c7.add(_g7.index); _dq7.append(_g7.index)
                if len(_c7) > len(_big7):
                    _big7 = _c7
            _pool = _big7
        _sd = bmb.faces[_seed if _seed is not None
                        else min(_pool, key=lambda _i: float(np.linalg.norm(_cen(_i) - A)))]
        _fl = set([_sd.index]); _dq = deque([_sd])
        while _dq:
            f = _dq.popleft()
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index in _c and nf.index not in _fl:
                        _fl.add(nf.index); _dq.append(nf)
        return _fl

    # VECTORISED footprint. The per-face point-in-polygon walked the rim's ~217-point outline
    # in Python for every face that passed the cheap gates -- tens of millions of iterations,
    # and 161 s of a 341 s graft. Shrinking the face set did nothing because the cost is per
    # SURVIVING face, not per scanned face. Gates first as arrays, then one even-odd test over
    # all the survivors at once.
    _lf = np.array([f.index for f in _LOCAL])
    _lc = np.array([_LOCC[i] for i in _lf])
    _lnx = np.array([abs(float(f.normal.x)) for f in _LOCAL])
    _keepm = np.linalg.norm(_lc - A, axis=1) <= 0.12
    if _rimcut_bound:
        # ...and in Y. Measured on 00003: with the extent bound off, the cut reached 84 mm
        # below the anchor while the donor stopped at 22 mm -- 62 mm of thigh removed that the
        # donor was never going to cover, which the bridge then had to span. The donor's own
        # placed extent IS the bound; no margin, no constant.
        _bad = ((_lc[:, 2] < _dlo[2]) | (_lc[:, 0] < _dlo[0]) | (_lc[:, 0] > _dhi[0])
                | (_lc[:, 1] < _dlo[1]) | (_lc[:, 1] > _dhi[1]))
        _next = int((_keepm & _bad).sum())
        _keepm &= ~_bad
    _latm = _lnx > _LAT
    _nlat = int((_keepm & _latm).sum())
    _keepm &= ~_latm
    _sel = np.where(_keepm)[0]
    if len(_sel):
        _dd = _lc[_sel] - A
        _px = _dd @ u
        _py = _dd @ v
        _Pp = np.asarray(poly, float)
        _xi, _yi = _Pp[:, 0], _Pp[:, 1]
        _xj, _yj = np.roll(_xi, 1), np.roll(_yi, 1)
        _cross = ((_yi[None, :] > _py[:, None]) != (_yj[None, :] > _py[:, None]))
        _xint = ((_xj - _xi)[None, :] * (_py[:, None] - _yi[None, :])
                 / (_yj - _yi + 1e-12)[None, :] + _xi[None, :])
        _inside = ((_cross & (_px[:, None] < _xint)).sum(1) % 2).astype(bool)
        for _k in _sel[_inside]:
            cand.add(int(_lf[_k]))
    print(f"[cutjoin] footprint gates: {len(_LOCAL)} local -> {len(_sel)} past the gates -> "
          f"{len(cand)} inside the rim outline", flush=True)

    if _rimcut_bound:
        print(f"[cutjoin] donor-extent bound: {_next} faces rejected outside "
              f"X {_dlo[0]:+.4f}..{_dhi[0]:+.4f} or behind Z {_dlo[2]:+.4f} "
              f"(front left open -- the donor sits under the skin it replaces)", flush=True)
    print(f"[cutjoin] footprint: {len(cand)} body faces ({_nlat} lateral/inner-thigh faces rejected)", flush=True)
    # How FAR from the donor does the footprint reach? The footprint is a planar shadow, so on a
    # saddle like the crotch it keeps going up the inner thighs long after the donor surface has
    # curved away -- and every such face is cut, leaving the bridge to web across the gap in big
    # flat panels. Distance from each candidate face to the nearest donor vertex says how much of
    # the cut the donor can actually cover.
    try:
        _ckd = kdtree.KDTree(len(W))
        for _i in range(len(W)): _ckd.insert(Vector(W[_i].tolist()), _i)
        _ckd.balance()
        _cand_plain = set(cand)          # unbounded footprint, kept as the fallback
        _cl = list(cand)
        _cq = [(_LOCC[_fi] if _fi in _LOCC
                else np.array(bmb.faces[_fi].calc_center_median())) for _fi in _cl]
        _cfind = [_ckd.find(Vector(np.asarray(_q).tolist())) for _q in _cq]
        _cd = np.array([_r[2] for _r in _cfind])
        _cdi = np.array([int(_r[1]) for _r in _cfind])
        _qs = np.percentile(_cd, [50, 75, 90, 95, 99])
        print(f"[cutjoin] footprint dist-to-donor: med {_qs[0]:.5f} p75 {_qs[1]:.5f} p90 {_qs[2]:.5f} "
              f"p95 {_qs[3]:.5f} p99 {_qs[4]:.5f} max {_cd.max():.5f}", flush=True)
        if CUT_FACING and CUT_FACING != "0":
            _dn = np.zeros((len(W), 3))
            for _f3 in Fd:
                _i0, _i1, _i2 = int(_f3[0]), int(_f3[1]), int(_f3[2])
                _nn3 = np.cross(W[_i1] - W[_i0], W[_i2] - W[_i0])
                _dn[_i0] += _nn3; _dn[_i1] += _nn3; _dn[_i2] += _nn3
            _l3 = np.linalg.norm(_dn, axis=1); _l3[_l3 < 1e-15] = 1.0
            _dn = _dn / _l3[:, None]
            if float(_dn.sum(0) @ outw) < 0.0:
                _dn = -_dn
            if CUT_FACING == "smooth":
                bmb.normal_update()
                _bn = {}
                for f in _LOCAL:
                    _acc = np.zeros(3)
                    for _v in f.verts:
                        _acc += np.array([float(_v.normal.x), float(_v.normal.y), float(_v.normal.z)])
                    _l4 = float(np.linalg.norm(_acc))
                    _bn[f.index] = _acc / _l4 if _l4 > 1e-15 else np.array(
                        [float(f.normal.x), float(f.normal.y), float(f.normal.z)])
            else:
                _bn = {f.index: np.array([float(f.normal.x), float(f.normal.y), float(f.normal.z)])
                       for f in _LOCAL}
            _face_ok = np.array([float(np.dot(_bn[_fi], _dn[_cdi[_k]])) > 0.0
                                 if _fi in _bn else True for _k, _fi in enumerate(_cl)])
            _away = int((~_face_ok).sum())
            if _away and _away < len(_cl):
                print(f"[cutjoin] cut facing ({CUT_FACING}): {_away} of {len(_cl)} footprint faces have skin "
                      f"pointing AWAY from the donor (the far wall of the leg gap, reached only "
                      f"because the gap is thinner than the reach) -> dropped", flush=True)
                _keep_i = [_k for _k, _ok in enumerate(_face_ok) if _ok]
                _cl = [int(_cl[_k]) for _k in _keep_i]
                _cd = _cd[_keep_i]
                _cdi = _cdi[_keep_i]
                cand = set(_cl)
            else:
                print(f"[cutjoin] cut facing: {_away} of {len(_cl)} faces point away -- "
                      f"nothing dropped", flush=True)
        if CUT_REACH > 0.0:
            # Cut only what the donor can actually cover.
            # The bound used to be a multiple of the footprint's OWN median distance-to-donor,
            # which is circular: the footprint is a planar shadow, and at the crotch the inner
            # thighs run almost parallel to the projection, so the shadow's own median was 0.027
            # (46 mm at 170 cm) and 3x of it let the cut reach 0.081 -- 137 mm, on a donor that
            # is nowhere more than 14 mm off the skin. It dropped 17% of the footprint and the
            # flood then took the inner thigh either side of the crotch; the bridge webbed that
            # gap and left the hard oval crease on each thigh the user has been calling a wall.
            # Bound it by the DONOR instead: the donor can only replace skin it can reach, and
            # its own worst standoff from this body is exactly that reach. Still measured per
            # character, but measured on the thing doing the covering.
            _thr = (CUT_REACH * _stand) if CUT_RULE == "donor" else (CUT_REACH * float(_qs[0]))
            print(f"[cutjoin] cut bound: donor is at most {_stand:.5f} off the body "
                  f"({_stand / max(float(np.ptp(Vb[:, 1])), 1e-9) * 1700:.1f} mm at 170cm) -> "
                  f"reach {CUT_REACH:.2f}x = {_thr:.5f}  (the old footprint-median rule gave "
                  f"{CUT_REACH * float(_qs[0]):.5f})", flush=True)
            if CUT_MODE == "donor":
                # Take the footprint straight from the donor: every body face within reach of the
                # donor SURFACE, regardless of where it projects. The planar shadow both over- and
                # under-cuts -- it runs 9.6cm up the inner thighs where the donor has curved away,
                # and it skips the body beneath the donor's rear, leaving 88 of 217 rim vertices
                # (40%, both sides) with no body to stitch to for the bridge to web across.
                _new = set()
                for _f in bmb.faces:
                    _cc = np.array(_f.calc_center_median())
                    if np.linalg.norm(_cc - A) > 0.12:
                        continue
                    if _ckd.find(Vector(_cc.tolist()))[2] <= _thr:
                        _new.add(int(_f.index))
                if _new:
                    print(f"[cutjoin] cut mode DONOR (reach {CUT_REACH:.1f}x med = {_thr:.5f}): "
                          f"{len(_new)} faces within reach of the donor surface "
                          f"(planar shadow had {len(_cl)}; {len(_new - cand)} the shadow missed, "
                          f"{len(cand - _new)} it over-cut)", flush=True)
                    cand = _new
            else:
                _drop = int((_cd > _thr).sum())
                if _drop and _drop < len(_cl):
                    cand = set(int(_f) for _f, _d in zip(_cl, _cd) if _d <= _thr)
                    print(f"[cutjoin] cut reach {CUT_REACH:.2f}x = {_thr:.5f}: dropped {_drop} of "
                          f"{len(_cl)} footprint faces the donor cannot reach (inner thigh) -> "
                          f"{len(cand)} kept", flush=True)
    except Exception as _pe:
        print(f"[cutjoin] footprint dist probe failed: {_pe}", flush=True)
    # ---- COVER CUT (PELVIS_CUT_RULE=cover) ----------------------------------------------------
    # Cut exactly the skin the donor covers, judged ALONG THE DONOR'S OWN NORMAL. The facing test
    # removed the lens holes by barely cutting at all -- measured, it left the body's whole front
    # crotch and mons under the donor, and the conform then pressed the donor onto that skin: two
    # surfaces in the same place, the crumpled mons. Every earlier rule failed for one of three
    # reasons, and each has a measured answer here instead of a constant:
    #   * planar shadow / Euclidean reach jump the leg gap   -> signed distance along the donor
    #     normal at the NEAREST donor point, window [-stand, +buried depth], both measured;
    #   * the far wall of the gap is within reach            -> the straight line from the donor
    #     to the face must not pass through any other body surface first;
    #   * faces beside the donor are nearest its rim edge    -> the nearest donor point must lie
    #     more than one rim spacing inside the rim.
    # The body's own normals are never used: on this cage 43% of them flip between neighbours.
    if CUT_RULE == "cover":
        try:
            from mathutils.bvhtree import BVHTree as _BVc
            _dbvh = _BVc.FromPolygons([tuple(x) for x in W.tolist()],
                                      [list(map(int, f)) for f in Fd], all_triangles=True)
            _fsum = np.zeros(3)
            for _f7 in Fd:
                _fsum += np.cross(W[int(_f7[1])] - W[int(_f7[0])], W[int(_f7[2])] - W[int(_f7[0])])
            _nsgn = 1.0 if float(_fsum @ outw) >= 0.0 else -1.0
            _rimP7 = np.array([W[li] for li in loop], float)
            _rsp7 = float(np.median(np.linalg.norm(np.roll(_rimP7, -1, 0) - _rimP7, axis=1)))
            _rkd = kdtree.KDTree(len(_rimP7))
            for _k7, _q7 in enumerate(_rimP7):
                _rkd.insert(Vector(_q7.tolist()), _k7)
            _rkd.balance()
            _smp = _LOCAL[::max(1, len(_LOCAL) // 3000)]
            _ledge = float(np.median([(_e7.verts[0].co - _e7.verts[1].co).length
                                      for _f7 in _smp for _e7 in _f7.edges]))
            _back = _stand                                   # donor floats at most this far off
            _front = float(_dep.max()) + _ledge              # ... and is buried at most this deep
            if WINDOW_RIM_BAND > 0.0:
                _band9 = _rim_band_mask(W, Fd, loop, WINDOW_RIM_BAND)
                if _band9 is not None and _band9.any():
                    print(f"[cutjoin] cover window over the rim band ({int(_band9.sum())} of {len(W)} donor verts "
                          f"within {WINDOW_RIM_BAND:.2f}x the apron width of the rim): stand "
                          f"{_back / 1.0016 * 1700:.1f} -> {float(_dall[_band9].max()) / 1.0016 * 1700:.1f} mm, buried "
                          f"{float(_dep.max()) / 1.0016 * 1700:.1f} -> {float(_dep[_band9].max()) / 1.0016 * 1700:.1f} mm",
                          flush=True)
                    _back = float(_dall[_band9].max())
                    _front = float(_dep[_band9].max()) + _ledge
                else:
                    print("[cutjoin] cover window over the rim band: zones do not match the donor -- all verts kept",
                          flush=True)
            _lo7 = W.min(0) - _back; _hi7 = W.max(0) + _back
            _box = [f for f in _LOCAL if np.all(_LOCC[f.index] >= _lo7) and np.all(_LOCC[f.index] <= _hi7)]
            _vm7 = {}; _vl7 = []; _pl7 = []
            for f in _box:
                _row = []
                for _v7 in f.verts:
                    _j7 = _vm7.get(_v7.index)
                    if _j7 is None:
                        _j7 = len(_vl7); _vm7[_v7.index] = _j7; _vl7.append(tuple(_v7.co))
                    _row.append(_j7)
                _pl7.append(_row)
            _bbvh = _BVc.FromPolygons(_vl7, _pl7, all_triangles=False)
            _cov = set(); _n_win = _n_rim = _n_blk = 0; _n_foot = 0
            for _k7, f in enumerate(_box):
                _c7 = Vector(_LOCC[f.index].tolist())
                _loc, _nor, _di, _dd = _dbvh.find_nearest(_c7)
                if _loc is None or _dd > _back + _front:
                    continue
                _sd7 = float((_c7 - _loc).dot(_nor)) * _nsgn
                if _sd7 < -_back or _sd7 > _front:
                    _n_win += 1; continue
                if _rkd.find(_loc)[2] <= _rsp7:
                    if COVER_RIM_FOOT:
                        _v8 = _c7 - _loc
                        _vt8 = _v8 - _nor * float(_v8.dot(_nor))
                        if _vt8.length > 0.5 * _ledge:
                            _n_rim += 1; continue
                        _n_foot += 1
                    else:
                        _n_rim += 1; continue
                _vec = _c7 - _loc; _ln7 = _vec.length
                if _ln7 > 2.0 * _ledge:
                    _hit = _bbvh.ray_cast(_loc, _vec / _ln7, _ln7 - _ledge)
                    if _hit[0] is not None and _hit[2] != _k7:
                        _n_blk += 1; continue
                _cov.add(f.index)
            # connected pieces, for the log -- the flood keeps the one at the anchor
            _pcs = []; _sn7 = set()
            for _s7 in _cov:
                if _s7 in _sn7: continue
                _cc7 = {_s7}; _dq7 = deque([_s7]); _sn7.add(_s7)
                while _dq7:
                    _x7 = _dq7.popleft()
                    for _e7 in bmb.faces[_x7].edges:
                        for _g7 in _e7.link_faces:
                            if _g7.index in _cov and _g7.index not in _sn7:
                                _sn7.add(_g7.index); _cc7.add(_g7.index); _dq7.append(_g7.index)
                _pcs.append(len(_cc7))
            _pcs.sort(reverse=True)
            _dcv = os.environ.get("PELVIS_DUMP_COVER", "")
            if _dcv:
                # raw cover set, one id per connected piece (largest = 1), plus why the rest of the
                # box was rejected -- so the pieces the flood throws away can be SEEN
                _pm = np.zeros(len(bmb.faces), np.int16)
                _sn8 = set(); _pieces8 = []
                for _s8 in _cov:
                    if _s8 in _sn8: continue
                    _cc8 = [_s8]; _dq8 = deque([_s8]); _sn8.add(_s8)
                    while _dq8:
                        _x8 = _dq8.popleft()
                        for _e8 in bmb.faces[_x8].edges:
                            for _g8 in _e8.link_faces:
                                if _g8.index in _cov and _g8.index not in _sn8:
                                    _sn8.add(_g8.index); _cc8.append(_g8.index); _dq8.append(_g8.index)
                    _pieces8.append(_cc8)
                _pieces8.sort(key=len, reverse=True)
                for _k8, _cc8 in enumerate(_pieces8):
                    _pm[_cc8] = min(_k8 + 1, 32000)
                np.save(_dcv, _pm)
                print(f"[cutjoin] cover pieces -> {_dcv}", flush=True)
            print(f"[cutjoin] cover cut: {len(_cov)} of {len(_box)} faces near the donor are covered "
                  f"(window {-_back / 1.0016 * 1700:.1f}..+{_front / 1.0016 * 1700:.1f} mm along the "
                  f"donor normal); rejected {_n_win} outside the window, {_n_rim} beside the rim, "
                  f"{_n_blk} behind another surface; pieces {_pcs[:6]}", flush=True)
            if COVER_RIM_FOOT:
                print(f"[cutjoin] cover rim-foot: {_n_foot} face(s) within a rim spacing of the rim but lying "
                      f"UNDER the donor edge were covered instead of kept", flush=True)
            if _cov:
                cand = _cov
                _cand_plain = set(_cov)       # nothing to guard against: this IS the footprint
        except Exception as _ce:
            import traceback as _tb7; _tb7.print_exc()
            print(f"[cutjoin] cover cut FAILED ({_ce!r}) -- keeping the previous footprint", flush=True)
    print("[t] prologue: footprint done, flood next", flush=True)
    if not cand:
        bmb.free(); print("[cutjoin] WARN: no body faces under donor footprint", flush=True); return ob, W
    # RIM-BOUNDED CUT. `poly` is a PLANAR projection of the rim: fine where the pelvis faces
    # forward, wrong at the true crotch where the surface wraps under, because the rim's shadow
    # covers far more surface than the rim encloses (hole came out 1.75x the rim, and the
    # hole->rim snap maxes out at factor 1.0 still leaving 1.71 -- it cannot recover that).
    # The rim is already snapped ONTO the body, so it is a closed curve on the surface: mark the
    # body faces nearest each rim vertex as a blocking ring and let the flood stop there. The
    # polygon stays as an outer safety bound so a gap in the ring cannot leak into the legs.
    _rimcut = os.environ.get("PELVIS_RIM_CUT", "0") == "1"   # off (see above)
    block = set()
    if _rimcut:
        # local faces only -- this feeds a KD-tree that is queried with donor rim points, all
        # of which are inside the local set by construction
        _fcidx = [f.index for f in _LOCAL]
        _fc = np.array([_LOCC[i] for i in _fcidx])
        try:
            from scipy.spatial import cKDTree as _KD
            _t = _KD(_fc)
            for li in loop:
                for _j in _t.query(W[li], k=3)[1]:
                    block.add(_fcidx[int(_j)])       # local row -> global face index
        except Exception:
            for li in loop:
                block.add(_fcidx[int(np.argmin(np.linalg.norm(_fc - W[li], axis=1)))])
        cand -= block
        print(f"[cutjoin] rim-bounded cut: {len(block)} blocker faces on the rim curve; "
              f"footprint {len(cand)} after removing them", flush=True)
    if not cand:
        bmb.free(); print("[cutjoin] WARN: rim ring consumed the footprint", flush=True); return ob, W
    # ---- CUT AT THE RIM (2026-09-13) ------------------------------------------------------
    # `poly` is a planar SHADOW of the rim, so the hole it opens is nowhere near where the rim
    # lands. Measured: the hole boundary then has to be dragged a MEDIAN 18.5 mm to meet the rim,
    # and every body face hanging off it stretches by that much -- the worst become 40 mm slivers,
    # which is the spike. The rim is a closed curve ON the body once RIM_SEAT_ITERS has seated it
    # (median 0.017 mm off the surface), so cut at the curve itself.
    #
    # The barrier is a DISTANCE FIELD, not a set of "faces the curve passes through". Two earlier
    # attempts marked nearest faces and both leaked, for two separate reasons: nearest-face
    # marking leaves vertex-only-connected gaps that an edge flood walks straight through, and
    # projecting sample points with closest_point_on_mesh teleports between the two thighs in the
    # leg gap. A level set around a curve that is already on the surface has neither problem --
    # every face within `w` is in, so there is nothing to walk through, and no projection is
    # called at all. Sample step and `w` are the body's own edge length.
    _ring_ok = False
    if RIM_RING:
        try:
            _rimP = np.array([W[li] for li in loop], float)
            _rr = float(np.linalg.norm(_rimP - A, axis=1).max())
            _sfa = _LOCAL[::max(1, len(_LOCAL) // 2000)]
            _sed = np.array([np.array(_e.verts[0].co) - np.array(_e.verts[1].co)
                             for _f in _sfa for _e in _f.edges], float)
            _medge = float(np.median(np.linalg.norm(_sed, axis=1)))
            _w = RIM_RING_W * _medge
            _lcA = np.linalg.norm(_lc - A, axis=1)
            _ball = _rr + RIM_RING_BALL * _medge
            _sel2 = np.where(_lcA <= _ball)[0]
            _bidx = _lf[_sel2]; _bc = _lc[_sel2]
            # dense samples along the seated curve; consecutive rim verts are ~1 edge apart and
            # both on the surface, so the chord between them never leaves it either
            # Band width FOLLOWS the curve. The seat leaves a median 0.017 mm but a handful of
            # vertices at 8 mm -- the ones whose chord crosses the notch between the legs. A flat
            # 1.5-edge band misses the surface under those and the flood walks out through the
            # gap (measured: 425 of 2445 shell faces reached). Widen the band exactly where the
            # curve is off the surface, by exactly how far it is off, so the level set stays
            # closed without fattening the ring everywhere.
            _roff = np.empty(len(_rimP))
            for _k in range(len(_rimP)):
                _ok3, _cp3, _n3, _i3 = ob.closest_point_on_mesh(Vector(_rimP[_k].tolist()))
                _roff[_k] = float((Vector(_rimP[_k].tolist()) - _cp3).length) if _ok3 else 0.0
            _S = []; _SR = []
            for _k in range(len(_rimP)):
                _a0 = _rimP[_k]; _b0 = _rimP[(_k + 1) % len(_rimP)]
                _oa, _ob2 = _roff[_k], _roff[(_k + 1) % len(_rimP)]
                _n5 = max(1, int(np.ceil(float(np.linalg.norm(_b0 - _a0)) / max(0.5 * _medge, 1e-9))))
                for _j in range(_n5):
                    _f5 = _j / float(_n5)
                    _S.append(_a0 + (_b0 - _a0) * _f5)
                    _SR.append(max(_w, (_oa * (1.0 - _f5) + _ob2 * _f5) + _medge))
            _S = np.asarray(_S); _SR = np.asarray(_SR)
            _fkd = kdtree.KDTree(len(_bc))
            for _k in range(len(_bc)):
                _fkd.insert(Vector(_bc[_k].tolist()), _k)
            _fkd.balance()
            # Mark the chord AND its projection. Where the curve crosses the notch between the
            # legs, consecutive samples project to opposite thighs, so the chord between them
            # runs through thin air and a band around it covers no faces at all -- that is the
            # gap the flood was walking out of, visible as a ring that is only a short arc.
            # Marking both closes it on both sides. Over-marking there is harmless: the barrier
            # is cut away with the hole anyway.
            barrier = set(); _jump = 0
            for _k in range(len(_S)):
                _r5 = float(_SR[_k])
                for _hit in _fkd.find_range(Vector(_S[_k].tolist()), _r5):
                    barrier.add(int(_bidx[_hit[1]]))
                _ok5, _cp5, _n5b, _i5 = ob.closest_point_on_mesh(Vector(_S[_k].tolist()))
                if not _ok5:
                    continue
                if float((Vector(_S[_k].tolist()) - _cp5).length) > _w:
                    _jump += 1
                for _hit in _fkd.find_range(_cp5, max(_w, _medge)):
                    barrier.add(int(_bidx[_hit[1]]))
            print(f"[cutjoin] rim ring: {_jump} of {len(_S)} samples sit further than the band "
                  f"from the surface (the notch crossings); their projections are marked too",
                  flush=True)
            print(f"[cutjoin] rim ring: curve sits {np.median(_roff):.5f} off the surface "
                  f"(max {_roff.max():.5f}); band {_w:.5f} widened to {_SR.max():.5f} where it "
                  f"lifts", flush=True)
            _cs = set(int(i) for i in _bidx) - barrier
            _ballset = set(int(i) for i in _bidx)
            _shell = set()
            for _gi in _ballset:
                for _e in bmb.faces[_gi].edges:
                    if any(_g.index not in _ballset for _g in _e.link_faces):
                        _shell.add(_gi); break
            # Do not guess a seed. Split what the band leaves into components and take the one
            # that cannot reach the edge of the search ball -- that IS the enclosed region, and
            # the same test says whether the band closed at all. Seeding by proximity kept
            # picking a face just OUTSIDE the ring (the vulva's nearest non-band face is on a
            # thigh wall), so the flood filled the outside and every barrier I tried reported an
            # identical 425 shell faces reached. Four different barriers, one unchanged number.
            _seen5 = set(); _inside = []; _outside = 0
            for _s5 in _cs:
                if _s5 in _seen5:
                    continue
                _c5 = {_s5}; _dq5 = deque([_s5]); _seen5.add(_s5)
                while _dq5:
                    _x5 = _dq5.popleft()
                    for _e5 in bmb.faces[_x5].edges:
                        for _g5 in _e5.link_faces:
                            if _g5.index in _cs and _g5.index not in _seen5:
                                _seen5.add(_g5.index); _c5.add(_g5.index); _dq5.append(_g5.index)
                if _c5 & _shell:
                    _outside += len(_c5)
                else:
                    _inside.append(_c5)
            _inside.sort(key=len, reverse=True)
            _fl = _inside[0] if _inside else set()
            _leak = 0 if _fl else 1
            print(f"[cutjoin] rim ring: the band splits the ball into {len(_inside)} enclosed "
                  f"region(s) {[len(_c) for _c in _inside[:4]]} and {_outside} face(s) that reach "
                  f"the ball edge" + ("" if _fl else " -- NOTHING is enclosed, the band leaks"),
                  flush=True)
            if _fl:
                _dprox = np.array([float(np.linalg.norm(
                    (_LOCC[int(_i)] if int(_i) in _LOCC
                     else np.array(bmb.faces[int(_i)].calc_center_median())) - A)) for _i in _fl])
                print(f"[cutjoin] rim ring: enclosed region {len(_fl)} face(s), "
                      f"{np.median(_dprox):.5f} med / {_dprox.max():.5f} max from the anchor "
                      f"(rim radius {_rr:.5f})", flush=True)
            _dr = os.environ.get("PELVIS_DUMP_RING", "")
            if _dr:
                _m3 = np.zeros(len(bmb.faces), np.uint8)
                _m3[sorted(_fl)] = 1
                _m3[sorted(barrier)] = 2
                np.save(_dr, _m3)
                print(f"[cutjoin] ring map -> {_dr}", flush=True)
            if _fl and not _leak:
                cand = _cs | barrier
                flood = _fl | barrier
                _ring_ok = True
        except Exception as _re:
            print(f"[cutjoin] rim ring failed ({_re!r}) -- using the planar shadow", flush=True)
    if not _ring_ok:
        flood = _flood(cand)
    # GUARD. Any bound on the candidates can leave the flood unable to break through, so the body
    # closes over the donor and the graft is simply not there. Measure it instead of hoping: flood
    # the unbounded footprint too and keep the bound only if it still opens a comparable hole.
    _plain = locals().get("_cand_plain")
    if (not _ring_ok) and _plain and cand != _plain:
        _fl_plain = _flood(_plain)
        _keep = len(flood) >= CUT_GUARD * max(len(_fl_plain), 1)
        print(f"[cutjoin] flood guard: bounded cut floods {len(flood)} faces vs {len(_fl_plain)} "
              f"unbounded ({len(flood)/max(len(_fl_plain),1):.2f}x, need >={CUT_GUARD:.2f}) -> "
              + ("bound kept" if _keep else "*** BOUND DISCARDED, cutting plain (it would have "
                                            "buried the graft) ***"), flush=True)
        if not _keep:
            cand = _plain; flood = _fl_plain
    if not flood:
        bmb.free(); print("[cutjoin] WARN: flood empty -- nothing cut, graft would be buried",
                          flush=True); return ob, W
    # OPEN the cut before closing it. The flood follows the footprint into thin tongues down
    # the leg gap, and a tongue makes the hole loop wander: measured, the hole came out 1.69x
    # the rim's perimeter with the same enclosed area, and after the proportional hole->rim
    # snap the worst vertex still sat 11.5 rim-spacings away. The bridge has to web that, and
    # a 29 mm bridge edge is the spike on the render. Erode, keep the body of the cut, dilate
    # back: tongues narrower than the erosion vanish, the bulk is restored unchanged.
    if CUT_OPEN > 0 and flood:
        _o0 = len(flood)
        for _ in range(CUT_OPEN):
            _rm = set()
            for _fi in flood:
                _t = sum(1 for _e in bmb.faces[_fi].edges
                         for _g in _e.link_faces if _g.index != _fi and _g.index in flood)
                if _t <= 1:
                    _rm.add(_fi)
            if not _rm:
                break
            flood -= _rm
        if flood:
            _seen = set(); _bigc = set()
            for _s in flood:
                if _s in _seen:
                    continue
                _c = {_s}; _dq = deque([_s]); _seen.add(_s)
                while _dq:
                    _x = _dq.popleft()
                    for _e in bmb.faces[_x].edges:
                        for _g in _e.link_faces:
                            if _g.index in flood and _g.index not in _seen:
                                _seen.add(_g.index); _c.add(_g.index); _dq.append(_g.index)
                if len(_c) > len(_bigc):
                    _bigc = _c
            flood = _bigc
        for _ in range(CUT_OPEN):
            _add = set()
            for _fi in list(flood):
                for _e in bmb.faces[_fi].edges:
                    for _g in _e.link_faces:
                        if _g.index not in flood and _g.index in cand:
                            _add.add(_g.index)
            if not _add:
                break
            flood |= _add
        print(f"[cutjoin] cut open: {CUT_OPEN} erode + regrow trimmed the tongues "
              f"({_o0} -> {len(flood)} faces)", flush=True)
        if not flood:
            bmb.free(); print("[cutjoin] WARN: the open step eroded the whole cut -- nothing cut, graft would be "
                              "buried", flush=True); return ob, W
    if CUT_CLOSE > 0 and flood:
        _n0 = len(flood)
        for _ in range(CUT_CLOSE):
            _add = set()
            for _f in bmb.faces:
                if _f.index in flood: continue
                _c = np.array(_f.calc_center_median())
                if np.linalg.norm(_c - A) > 0.12: continue
                _t = sum(1 for _e in _f.edges
                         for _g in _e.link_faces if _g is not _f and _g.index in flood)
                # Surrounded means all but ONE side, so the test must scale with the face's own
                # edge count. The body is QUADS: a flat ">=2 of its edges" absorbs any ordinary
                # boundary quad, eating an extra ring wherever the cut runs with the quad grid --
                # measured as the right side getting worse (x>0 creases 150 -> 174) while the left
                # improved. A triangle needs 2, a quad needs 3.
                if _t >= len(_f.edges) - 1: _add.add(_f.index)
            if not _add: break
            flood |= _add
        print("[t] cut close: finished", flush=True)
        print(f"[cutjoin] cut close: absorbed {len(flood)-_n0} one-face peninsulas "
              f"({_n0} -> {len(flood)} faces, {CUT_CLOSE} passes)", flush=True)
    # ABSORB islands. The bridge stitches exactly ONE boundary loop, so any other loop the cut
    # leaves stays open -- and the weld then drags those stray vertices onto the rim, which is
    # what builds the fan in the leg gap. Measured here: the cut's boundary had 2 loops (302 and
    # 125 verts) and the 125 was an island of uncut body sitting inside the cut region.
    # A face the cut completely surrounds is inside the area the donor replaces, so cut it too.
    if CUT_JOIN_PIECES and CUT_RULE == "cover" and flood and cand:
        from mathutils.bvhtree import BVHTree as _BVj
        _dbj = locals().get("_dbvh"); _boxj = locals().get("_box")
        if _dbj is None or not _boxj:
            print("[cutjoin] join pieces: no cover data -- skipped", flush=True)
        else:
            _boxset = set(f.index for f in _boxj)
            _leftj = set(cand) - flood
            _pcsj = []; _snj = set()
            for _s in _leftj:
                if _s in _snj:
                    continue
                _ccj = {_s}; _dqj = deque([_s]); _snj.add(_s)
                while _dqj:
                    _xj = _dqj.popleft()
                    for _ej in bmb.faces[_xj].edges:
                        for _gj in _ej.link_faces:
                            if _gj.index in _leftj and _gj.index not in _snj:
                                _snj.add(_gj.index); _ccj.add(_gj.index); _dqj.append(_gj.index)
                _pcsj.append(_ccj)
            _addj = 0; _npj = 0; _skipj = 0; _nocj = 0; _corrj = 0
            for _ccj in sorted(_pcsj, key=len, reverse=True):
                _vmj = {}; _vlj = []; _plj = []
                for _fj in _ccj:
                    _rowj = []
                    for _vj in bmb.faces[_fj].verts:
                        _ij = _vmj.get(_vj.index)
                        if _ij is None:
                            _ij = len(_vlj); _vmj[_vj.index] = _ij; _vlj.append(tuple(_vj.co))
                        _rowj.append(_ij)
                    _plj.append(_rowj)
                if not _dbj.overlap(_BVj.FromPolygons(_vlj, _plj, all_triangles=False)):
                    _skipj += 1
                    continue
                _prevj = {}; _dqj = deque()
                for _fj in _ccj:
                    _prevj[_fj] = -1; _dqj.append(_fj)
                _hitj = None
                while _dqj and _hitj is None:
                    _xj = _dqj.popleft()
                    for _ej in bmb.faces[_xj].edges:
                        for _gj in _ej.link_faces:
                            _gi = _gj.index
                            if _gi in _prevj:
                                continue
                            _prevj[_gi] = _xj
                            if _gi in flood:
                                _hitj = _gi; break
                            if _gi in _boxset:
                                _dqj.append(_gi)
                        if _hitj is not None:
                            break
                if _hitj is None:
                    _nocj += 1
                    _bxin = sum(1 for _fj in _ccj if _fj in _boxset)
                    _nbr = set()
                    for _fj in _ccj:
                        for _ej in bmb.faces[_fj].edges:
                            for _gj in _ej.link_faces:
                                if _gj.index not in _ccj:
                                    _nbr.add(_gj.index)
                    print(f"[cutjoin] join pieces DIAG: piece of {len(_ccj)} faces ({_bxin} in the box) touches the "
                          f"donor but no corridor: search visited {len(_prevj)} faces; its {len(_nbr)} edge-neighbours: "
                          f"{sum(1 for _g in _nbr if _g in _boxset)} in the box, {sum(1 for _g in _nbr if _g in flood)} "
                          f"in the cut, {sum(1 for _g in _nbr if _g in cand)} covered; box {len(_boxset)}, cut {len(flood)}",
                          flush=True)
                    _djp = os.environ.get("PELVIS_DUMP_JOIN", "")
                    if _djp and not os.path.exists(_djp):
                        np.savez(_djp, flood=np.array(sorted(flood), np.int64), box=np.array(sorted(_boxset), np.int64),
                                 piece=np.array(sorted(_ccj), np.int64), visited=np.array(sorted(_prevj.keys()), np.int64),
                                 cand=np.array(sorted(cand), np.int64))
                        print(f"[cutjoin] join pieces DIAG: sets dumped -> {_djp}", flush=True)
                    continue
                _pathj = []; _xj = _prevj[_hitj]
                while _xj != -1 and _xj not in _ccj:
                    _pathj.append(_xj); _xj = _prevj[_xj]
                _ringj = set(_pathj)
                for _fj in _pathj:
                    for _ej in bmb.faces[_fj].edges:
                        for _gj in _ej.link_faces:
                            if _gj.index in _boxset:
                                _ringj.add(_gj.index)
                _newj = (_ccj | _ringj) - flood
                flood |= _newj; _addj += len(_newj); _npj += 1; _corrj += len(_ringj)
            print(f"[cutjoin] join pieces: {_npj} more covered piece(s) the donor passes through joined to "
                  f"the cut ({_addj} faces incl. {_corrj} corridor faces); {_skipj} piece(s) the donor does "
                  f"not touch left alone, {_nocj} with no corridor", flush=True)
    if CUT_ISLANDS and flood:
        _localset = set(int(f.index) for f in _LOCAL)
        _rest = _localset - flood
        _outer = set()
        for _fi in _rest:
            for _e in bmb.faces[_fi].edges:
                if any(_g.index not in _localset for _g in _e.link_faces):
                    _outer.add(_fi); break
        _seen3 = set(_outer); _dq3 = deque(_outer)
        while _dq3:
            _x3 = _dq3.popleft()
            for _e3 in bmb.faces[_x3].edges:
                for _g3 in _e3.link_faces:
                    if _g3.index in _rest and _g3.index not in _seen3:
                        _seen3.add(_g3.index); _dq3.append(_g3.index)
        _isl = _rest - _seen3
        if _isl:
            flood |= _isl
            print(f"[cutjoin] cut islands: absorbed {len(_isl)} face(s) the cut completely "
                  f"surrounded ({len(flood) - len(_isl)} -> {len(flood)}); each island would "
                  f"otherwise leave a boundary loop the bridge never stitches", flush=True)
        else:
            print("[cutjoin] cut islands: none (the cut leaves a single boundary)", flush=True)
    _dc = os.environ.get("PELVIS_DUMP_CUT", "")
    if _dc:
        # exactly which body faces the graft removes -- the only honest way to see the cut
        # outline, since the export triangulates and face indices no longer line up after
        _m = np.zeros(len(bmb.faces), bool); _m[sorted(flood)] = True
        np.save(_dc, _m)
        print(f"[cutjoin] cut map: {int(_m.sum())} face(s) -> {_dc}", flush=True)
    hb = set()
    for fi in flood:
        for e in bmb.faces[fi].edges:
            if any(lf.index not in flood for lf in e.link_faces): hb.add(e.verts[0]); hb.add(e.verts[1])
    bmesh.ops.delete(bmb, geom=[bmb.faces[fi] for fi in flood], context='FACES')
    if XOUT:
        from mathutils.bvhtree import BVHTree
        _bvb = BVHTree.FromBMesh(bmb)          # the body that SURVIVES the cut, donor not added yet
        _prot = np.zeros(len(W), bool)
        try:
            _zn0 = np.load(A_ZONES)
            if len(_zn0) == len(W):
                _prot = np.isin(_zn0, [1, 2, 3, 4])   # labia + opening: never moved
        except Exception:
            pass
        _lift = 0; _dmax = 0.0
        for _i in range(len(W)):
            if _prot[_i]:
                continue
            _loc, _nor, _fi, _dd = _bvb.find_nearest(Vector(W[_i].tolist()))
            if _loc is None:
                continue
            _lo = np.array(_loc.to_tuple()); _nn = np.array(_nor.to_tuple())
            if float(np.dot(W[_i] - _lo, _nn)) < 0.0:     # inside the surviving body
                _dmax = max(_dmax, float(np.linalg.norm(W[_i] - _lo)))
                W[_i] = _lo + _nn * XOUT_EPS
                _lift += 1
        print(f"[cutjoin] donor lift-out: {_lift}/{len(W)} donor verts were INSIDE the surviving body "
              f"(deepest {_dmax:.5f}) -> projected onto its surface "
              f"({int(_prot.sum())} labeled detail verts exempt)", flush=True)
    _trace("after donor insert", bmb, A)
    _Wplaced = np.asarray(W, float).copy()
    dvm = [bmb.verts.new(W[i].tolist()) for i in range(len(W))]; bmb.verts.ensure_lookup_table()
    _dtrace("after donor insert", bmb, _dlay, _Wplaced)
    for _dvx in dvm:
        _dvx[_dlay] = 1
    donset = set(id(x) for x in dvm)
    for f in Fd:
        try: bmb.faces.new([dvm[int(f[0])], dvm[int(f[1])], dvm[int(f[2])]])
        except Exception: pass
    _loose = [x for x in dvm if x.is_valid and not x.link_faces]
    if _loose:
        bmesh.ops.delete(bmb, geom=_loose, context='VERTS')
        bmb.verts.ensure_lookup_table()
        donset = set(id(x) for x in dvm if x.is_valid)
        print(f"[cutjoin] donor trim: removed {len(_loose)} now-unused donor verts", flush=True)
    bmb.edges.ensure_lookup_table()
    he = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hb and e.verts[1] in hb]
    hadj = defaultdict(list)
    for e in he: hadj[e.verts[0]].append(e.verts[1]); hadj[e.verts[1]].append(e.verts[0])
    hs = he[0].verts[0]; H = [hs]; prev = None; cur = hs
    while True:
        nx = [w for w in hadj[cur] if w is not prev]
        if not nx: break
        if len(nx) == 1: n = nx[0]
        else:
            cc = np.array(cur.co); din = (cc - np.array(prev.co)) if prev is not None else np.array([1.0, 0, 0])
            din = din / (np.linalg.norm(din) + 1e-9)
            n = min(nx, key=lambda w: -float(np.dot(din, (np.array(w.co) - cc) / (np.linalg.norm(np.array(w.co) - cc) + 1e-9))))
        if n is hs or n in H: break
        H.append(n); prev = cur; cur = n
    R = [dvm[li] for li in loop]
    def _ang(x): p = np.array(x.co) - A; return float(np.arctan2(p @ v, p @ u))
    def _sar(L):
        p = np.array([[(np.array(x.co) - A) @ u, (np.array(x.co) - A) @ v] for x in L])
        return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    if _sar(R) * _sar(H) < 0: H = H[::-1]
    a0 = _ang(R[0]); j0 = int(np.argmin([abs(((_ang(h) - a0 + np.pi) % (2 * np.pi)) - np.pi) for h in H])); H = H[j0:] + H[:j0]
    _HP = np.array([[h.co[0], h.co[1], h.co[2]] for h in H], float)
    def _sp(P):
        dd = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        return dd.min(), np.median(dd), dd.max()
    _h0 = _sp(_HP)
    for _ in range(HOLE_RELAX_ITERS):
        _HP = _HP + HOLE_RELAX * (0.5 * (np.roll(_HP, 1, axis=0) + np.roll(_HP, -1, axis=0)) - _HP)
    if HOLE_RELAX_ITERS:
        for _k, _h in enumerate(H):
            _h.co = Vector(_HP[_k].tolist())
    _h1 = _sp(_HP)
    print(f"[cutjoin] hole loop spacing min/med/max {_h0[0]:.5f}/{_h0[1]:.5f}/{_h0[2]:.5f}"
          f" -> {_h1[0]:.5f}/{_h1[1]:.5f}/{_h1[2]:.5f}  (relax {HOLE_RELAX_ITERS}x{HOLE_RELAX})", flush=True)
    _trace("after hole relax+snap", bmb, A)
    _dtrace("after hole relax+snap", bmb, _dlay, _Wplaced)
    _stage_dump("1_hole_snap", bmb, A)
    _RP = np.array([[x.co[0], x.co[1], x.co[2]] for x in R], float)
    _HQ = np.array([[x.co[0], x.co[1], x.co[2]] for x in H], float)
    _per = lambda P: float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum())
    _pr, _ph = _per(_RP), _per(_HQ)
    _gap = np.linalg.norm(_RP[:, None, :] - _HQ[None, :, :], axis=2).min(1)
    print(f"[cutjoin] perimeter rim {_pr:.4f}  hole {_ph:.4f}  ratio {_ph/max(_pr,1e-9):.2f}", flush=True)
    if _ph < 0.5 * _pr:
        # The hole the cut opened is far smaller than the donor's own rim, so the body cannot make
        # way for the donor and it ends up under the skin -- the "graft disappeared" failure. Say so
        # in the bake log; it is otherwise only visible by looking at the finished character.
        print(f"*** [cutjoin] TRIPWIRE: hole perimeter {_ph:.4f} is only "
              f"{_ph/max(_pr,1e-9):.2f}x the donor rim {_pr:.4f} -- THE GRAFT WILL BE BURIED ***",
              flush=True)
    print(f"[cutjoin] rim->hole gap  min {_gap.min():.5f}  med {np.median(_gap):.5f}  max {_gap.max():.5f}"
          f"  (rim spacing {_pr/len(R):.5f})", flush=True)
    print(f"[cutjoin] gap / rim-spacing  med {np.median(_gap)/(_pr/len(R)):.1f}x  max {_gap.max()/(_pr/len(R)):.1f}x", flush=True)
    # WHERE is the rim far from the hole? Those rim vertices have no body to stitch to and the
    # bridge webs a flat panel across them, which reads as a hole in the render.
    _sp0 = _pr / max(len(R), 1)
    _far = np.where(_gap > 5.0 * _sp0)[0]
    if len(_far):
        _fp = _RP[_far]
        print(f"[cutjoin] rim far from hole: {len(_far)}/{len(R)} verts > 5x spacing; "
              f"x<0 {int((_fp[:,0]<0).sum())} x>0 {int((_fp[:,0]>0).sum())}; "
              f"worst at ({_RP[int(np.argmax(_gap))][0]:+.5f},{_RP[int(np.argmax(_gap))][1]:+.5f},"
              f"{_RP[int(np.argmax(_gap))][2]:+.5f}) gap {_gap.max():.5f}", flush=True)
    # The seam smooth reaches GRAFT_SEAM_BAND from the rim line, but the BRIDGE is as wide as the
    # rim->hole gap -- measured up to 0.036 against a 0.009 band, so most bridge faces were never
    # smoothed at all and kept their raw zigzag (554 edges over 60deg after grafting, against 5 on
    # the body before it). Let the band follow the bridge it has to cover.
    _seam_band = GRAFT_SEAM_BAND * SEAM_BAND_MUL
    if HOLE_SNAP != "0.0" and HOLE_SNAP != 0.0:
        _A2 = _RP; _B2 = np.roll(_RP, -1, axis=0); _AB = _B2 - _A2
        _den = np.maximum((_AB * _AB).sum(1), 1e-18)
        _newH = np.empty_like(_HQ)
        for _k, _p in enumerate(_HQ):
            _t = np.clip((((_p - _A2) * _AB).sum(1)) / _den, 0.0, 1.0)
            _pr2 = _A2 + _t[:, None] * _AB
            _newH[_k] = _pr2[int(np.argmin(np.linalg.norm(_pr2 - _p, axis=1)))]
        # If projection cannot reach parity, REBUILD the hole boundary as the rim curve itself,
        # resampled to nH points in the same order. Per-vertex nearest projection scatters the
        # verts out of sequence -- measured: perimeter 0.7972 -> 0.8125, it got LONGER -- leaving
        # 2.21x the rim's length with no rim to pair to, which is the excess the bridge gathers
        # into spikes. Resampling in order gives parity by construction and the two loops become
        # the same curve, so any sane pairing works.
        _hs = HOLE_SNAP
        _proj_ratio = _per(_newH) / max(_pr, 1e-9)
        _back_steps = 0
        if HOLE_REBUILD_NONMONO:
            _rpc = np.vstack([_RP, _RP[:1]]); _sgc = np.linalg.norm(np.diff(_rpc, axis=0), axis=1)
            _cmc = np.concatenate([[0.0], np.cumsum(_sgc)]); _Lc = float(_cmc[-1])
            _Ac = _rpc[:-1]; _ABc = np.diff(_rpc, axis=0); _dnc = np.maximum((_ABc * _ABc).sum(1), 1e-18)
            _tpar = np.empty(len(_newH))
            for _q, _pq in enumerate(_newH):
                _tt = np.clip((((_pq - _Ac) * _ABc).sum(1)) / _dnc, 0.0, 1.0)
                _pj = _Ac + _tt[:, None] * _ABc
                _ii = int(np.argmin(np.linalg.norm(_pj - _pq, axis=1)))
                _tpar[_q] = _cmc[_ii] + _tt[_ii] * _sgc[_ii]
            _dtp = (np.diff(np.concatenate([_tpar, _tpar[:1]])) + 0.5 * _Lc) % _Lc - 0.5 * _Lc
            _back_steps = int((_dtp < 0).sum())
            print(f"[cutjoin] hole snap: the projected hole runs backward along the rim "
                  f"{_back_steps} time(s) -> {'REBUILT in order' if _back_steps > 1 else 'kept'}", flush=True)
        if _proj_ratio > 1.15 or (HOLE_REBUILD_NONMONO and _back_steps > 1):
            _rp2 = np.vstack([_RP, _RP[:1]])
            _seg2 = np.linalg.norm(np.diff(_rp2, axis=0), axis=1)
            _cum2 = np.concatenate([[0.0], np.cumsum(_seg2)])
            _L2 = float(_cum2[-1])
            _nH2 = len(_HQ)
            _pos = lambda _tt: (lambda _ii, _tm: _rp2[_ii] * (1.0 - (_tm - _cum2[_ii]) / max(_seg2[_ii], 1e-12))
                                + _rp2[_ii + 1] * ((_tm - _cum2[_ii]) / max(_seg2[_ii], 1e-12)))(
                                min(int(np.searchsorted(_cum2, _tt % _L2, side='right')) - 1, len(_RP) - 1),
                                _tt % _L2)
            if HOLE_MAP == "monotone":
                # Uniform arc-fraction resampling assumes the two loops accumulate length the same
                # way. They do not: the hole traces near the rim (gap med 1x its spacing) but is
                # 2.17x as LONG because it staircases along body edges and makes one wide excursion.
                # Fraction 0.5 of a wiggly loop is nowhere near fraction 0.5 of a smooth one, so the
                # resample teleported single verts up to 0.144 -- 4x the true gap -- shredding the
                # body faces hanging off them into slivers, worst where the excursion is (one side).
                # Instead map every hole vertex to its NEAREST point on the rim (travel = the real
                # gap), then make those arc positions monotone so the loop still winds once and the
                # bridge stays fannable. Verts that only existed to trace a wiggle collapse together,
                # which is exactly what flattening the wiggle should do.
                _Astt = _rp2[:-1]; _ABs = np.diff(_rp2, axis=0)
                _dns = np.maximum((_ABs * _ABs).sum(1), 1e-18)
                _tk = np.empty(_nH2)
                for _q in range(_nH2):
                    _tt = np.clip((((_HQ[_q] - _Astt) * _ABs).sum(1)) / _dns, 0.0, 1.0)
                    _pj = _Astt + _tt[:, None] * _ABs
                    _ii = int(np.argmin(np.linalg.norm(_pj - _HQ[_q], axis=1)))
                    _tk[_q] = _cum2[_ii] + _tt[_ii] * _seg2[_ii]
                _u = np.empty(_nH2); _u[0] = _tk[0]
                for _q in range(1, _nH2):
                    _dq = (_tk[_q] - _tk[_q - 1]) % _L2
                    _u[_q] = _u[_q - 1] + (0.0 if _dq > 0.5 * _L2 else _dq)   # backward pick -> hold
                # strictly increasing (held verts would coincide -> degenerate faces), then rescale
                # so the sequence advances exactly one lap over nH steps
                _u = _u + np.arange(_nH2) * (_L2 / (10.0 * _nH2))
                _span = _u[-1] - _u[0]
                if _span > 1e-9:
                    _u = _u[0] + (_u - _u[0]) * (_L2 * (_nH2 - 1) / _nH2) / _span
                _newH = np.array([_pos(_t) for _t in _u])
                _trav = np.linalg.norm(_newH - _HQ, axis=1)
                print(f"[cutjoin] hole rebuilt on the rim curve, MONOTONE projection "
                      f"(arc-fraction would leave {_proj_ratio:.2f}x); {_nH2} verts, "
                      f"travel med {np.median(_trav):.5f} max {_trav.max():.5f}", flush=True)
            else:
                _t2 = np.linspace(0.0, _L2, _nH2, endpoint=False)
                _off = _cum2[int(np.argmin(np.linalg.norm(_RP - _HQ[0], axis=1)))]
                _newH = np.array([_pos(_tv + _off) for _tv in _t2])
                _trav = np.linalg.norm(_newH - _HQ, axis=1)
                print(f"[cutjoin] hole rebuilt on the rim curve (projection would leave "
                      f"{_proj_ratio:.2f}x); resampled {_nH2} verts, "
                      f"travel med {np.median(_trav):.5f} max {_trav.max():.5f}", flush=True)
            _hs = "1.0"
        if str(_hs).lower() == "auto":
            # Bisect the blend factor so the hole perimeter lands on the rim's. Perimeter is
            # monotonically decreasing in s over [0,1], so 24 steps are ample and exact.
            _lo, _hi = 0.0, 1.0
            for _ in range(24):
                _mid = 0.5 * (_lo + _hi)
                if _per(_HQ + _mid * (_newH - _HQ)) > _pr:
                    _lo = _mid
                else:
                    _hi = _mid
            _s = 0.5 * (_lo + _hi)
        else:
            _s = float(_hs)
        _HQ0 = _HQ.copy()
        _HQ = _HQ + _s * (_newH - _HQ)
        for _k, _h in enumerate(H):
            _h.co = Vector(_HQ[_k].tolist())
        # SPREAD the hole displacement outward instead of letting ONE ring of body faces absorb it.
        # The rebuild teleports each hole vertex onto the rim curve; where the flood cut ran wide
        # (measured gap max 0.034 = 20x the rim spacing, and strongly one-sided) the body faces just
        # outside the hole keep their outer vertex while the inner one jumps ~3cm -- that is the
        # sliver ring, and it is why the pinching is on one side only. Relative rule, no fixed
        # radius: each hole vertex drags its neighbourhood over a radius proportional to ITS OWN
        # displacement, so the side where the cut already met the rim (gap ~1x spacing) is left
        # untouched and only the wide side relaxes.
        _dsp  = _HQ - _HQ0
        _dmag = np.linalg.norm(_dsp, axis=1)
        if HOLE_SPREAD > 0.0 and float(_dmag.max()) > 1e-9:
            # Let the body move WITH the hole instead of the first ring taking all the strain.
            # The snap contracts the hole loop hard (measured: perimeter 0.4338 -> 0.2362, 45%),
            # and every body face hanging off it stretches by that much -- 47 faces past 5x the
            # local edge, the worst at 30 mm. Carry each hole vertex's displacement into the
            # skin around it with a falloff, so the same contraction is absorbed over several
            # rings and no single face has to hold it.
            #
            # The radius is a multiple of the BODY's own edge spacing, not of the displacement.
            # Tying it to the displacement (the original form) made it 2 x 18.5 mm = 37 mm and
            # dragged 180,000 verts by up to 33 mm, which reshapes the crotch instead of
            # relieving the seam. How far the strain should reach is a property of the mesh's
            # resolution, not of how badly the loop happened to move.
            _hset = set(id(x) for x in H)
            _allco = np.array([v.co[:] for v in bmb.verts], float)
            _keepv = np.linalg.norm(_allco - A, axis=1) < 0.15
            _bv = [x for x in bmb.verts
                   if _keepv[x.index] and x.is_valid
                   and id(x) not in donset and id(x) not in _hset]
            if _bv:
                _bp  = np.array([list(x.co) for x in _bv], float)
                _Rad = np.full(len(_dmag), HOLE_SPREAD * (_pr / max(len(R), 1)))
                _geo_ok = False
                if HOLE_SPREAD_DISP > 0.0:
                    import heapq as _hq8
                    _Rad = np.maximum(_Rad, HOLE_SPREAD_DISP * _dmag)
                    _bidx = {id(x): _k8 for _k8, x in enumerate(_bv)}
                    _geo_ok = True
                _num = np.zeros_like(_bp); _den = np.zeros(len(_bp)); _mxw = np.zeros(len(_bp))
                for _i in np.where(_Rad > 1e-6)[0]:
                    if _geo_ok:
                        # geodesic from this hole vertex over body edges, out to its own radius; the hole vertices
                        # themselves are passed through (they border the skin) but never receive the offset
                        _dd = np.full(len(_bp), np.inf)
                        _h0 = H[_i]
                        _pq8 = [(0.0, id(_h0), _h0)]
                        _seen8 = {id(_h0): 0.0}
                        while _pq8:
                            _d8, _k8, _x8 = _hq8.heappop(_pq8)
                            if _d8 > _seen8.get(_k8, np.inf):
                                continue
                            _j8 = _bidx.get(_k8)
                            if _j8 is not None:
                                _dd[_j8] = _d8
                            for _e8 in _x8.link_edges:
                                _y8 = _e8.other_vert(_x8)
                                if id(_y8) not in _bidx and id(_y8) not in _hset:
                                    continue
                                _n8 = _d8 + _e8.calc_length()
                                if _n8 < _Rad[_i] and _n8 < _seen8.get(id(_y8), np.inf):
                                    _seen8[id(_y8)] = _n8
                                    _hq8.heappush(_pq8, (_n8, id(_y8), _y8))
                    else:
                        _dd = np.linalg.norm(_bp - _HQ0[_i], axis=1)
                    _w  = np.clip(1.0 - _dd / _Rad[_i], 0.0, 1.0) ** 2
                    _m  = _w > 0.0
                    if not _m.any():
                        continue
                    _num[_m] += _w[_m, None] * _dsp[_i]
                    _den[_m] += _w[_m]
                    _mxw[_m]  = np.maximum(_mxw[_m], _w[_m])
                _mv = _den > 0.0
                if _mv.any():
                    _off = np.zeros_like(_bp)
                    _off[_mv] = (_num[_mv] / _den[_mv, None]) * _mxw[_mv, None]
                    for _j in np.where(_mv)[0]:
                        _bv[_j].co = Vector((_bp[_j] + _off[_j]).tolist())
                    print(f"[cutjoin] hole spread: relaxed {int(_mv.sum())} body verts "
                          f"(hole disp med {np.median(_dmag):.5f} max {float(_dmag.max()):.5f}; "
                          f"radius {HOLE_SPREAD:.1f}x the rim spacing = {_Rad[0]:.5f}; max body "
                          f"move {float(np.linalg.norm(_off, axis=1).max()):.5f})", flush=True)
                    if HOLE_SPREAD_DISP > 0.0:
                        print(f"[cutjoin] hole spread by displacement: radius = max(rim-spacing radius, "
                              f"{HOLE_SPREAD_DISP:.1f}x own displacement), geodesic; radius med "
                              f"{float(np.median(_Rad)) / 1.0016 * 1700:.1f} max {float(_Rad.max()) / 1.0016 * 1700:.1f} mm; "
                              f"{int((_Rad > HOLE_SPREAD * (_pr / max(len(R), 1)) + 1e-9).sum())} of {len(_Rad)} hole verts "
                              f"reach further than before", flush=True)
        # POST-snap gap is what the bridge actually spans. Perimeter parity says nothing about
        # whether the two loops COINCIDE: on the crotch anchor the hole came out shorter than the
        # rim yet sat 8.6x its own spacing away, so the parity solver returned factor 0.000 and
        # left the bridge stretching across the whole offset.
        _g2 = np.linalg.norm(_RP[:, None, :] - _HQ[None, :, :], axis=2).min(1)
        if SEAM_BAND_FOLLOW > 0.0:
            _bw = float(np.percentile(_g2, 95))
            _seam_band = max(GRAFT_SEAM_BAND * SEAM_BAND_MUL, SEAM_BAND_FOLLOW * _bw)
            print(f"[cutjoin] seam band follows the bridge: p95 gap {_bw:.5f} x{SEAM_BAND_FOLLOW:.1f} "
                  f"-> band {_seam_band:.5f} (was {GRAFT_SEAM_BAND:.5f})", flush=True)
        print(f"[cutjoin] hole->rim snap {_hs} (factor {_s:.3f}): perimeter {_ph:.4f} -> "
              f"{_per(_HQ):.4f} (rim {_pr:.4f}, ratio {_per(_HQ)/max(_pr,1e-9):.2f}); "
              f"gap med {np.median(_g2)/(_pr/len(R)):.1f}x max {_g2.max()/(_pr/len(R)):.1f}x", flush=True)
    # OUTLIER hole vertices. The snap above is one global factor over the whole loop, so a
    # handful of vertices can stay far out however good the average gets -- and each one becomes
    # a bridge triangle drawn all the way across. Measured on 00003: two triangles with a 42.7 mm
    # edge hanging into the leg gap, out of 887550 near the graft. That pair IS the spike.
    # Pull only those onto the rim POLYLINE (segments, not just its vertices, so the landing
    # point is wherever the rim actually passes). The bar is a multiple of the rim's own
    # spacing, which is the same yardstick every other gap in this function is reported in.
    if HOLE_OUTLIER > 0 and len(H) and len(_RP) > 1:
        _sp = _pr / len(R)
        _As, _Bs = _RP, np.roll(_RP, -1, axis=0)
        _AB = _Bs - _As
        _dn = np.maximum((_AB ** 2).sum(1), 1e-18)
        _HN = np.array([[x.co[0], x.co[1], x.co[2]] for x in H], float)
        _t = np.clip(((_HN[:, None, :] - _As[None, :, :]) * _AB[None, :, :]).sum(2) / _dn[None, :],
                     0.0, 1.0)
        _proj = _As[None, :, :] + _t[:, :, None] * _AB[None, :, :]
        _dseg = np.linalg.norm(_proj - _HN[:, None, :], axis=2)
        _best = np.argmin(_dseg, axis=1)
        _dmin = _dseg[np.arange(len(_HN)), _best]
        _bad = np.where(_dmin > HOLE_OUTLIER * _sp)[0]
        if len(_bad):
            for _k in _bad:
                H[int(_k)].co = Vector(_proj[int(_k), int(_best[int(_k)])].tolist())
            print(f"[cutjoin] hole outliers: {len(_bad)} of {len(_HN)} hole vert(s) sat over "
                  f"{HOLE_OUTLIER:.0f}x the rim spacing ({HOLE_OUTLIER * _sp:.5f}) from the rim "
                  f"line -- worst {_dmin.max():.5f} ({_dmin.max() / _sp:.1f}x) -- pulled onto it; "
                  f"they are what the bridge was spanning as slivers", flush=True)
        else:
            print(f"[cutjoin] hole outliers: none over {HOLE_OUTLIER:.0f}x the rim spacing "
                  f"(worst {_dmin.max() / _sp:.1f}x)", flush=True)
    # ALIGN the two loops before stitching. The walk below pairs rim vertex i/nR with hole
    # vertex j/nH -- purely by index fraction -- which is only correct if both loops start at
    # corresponding points and wind the same way. Nothing guaranteed either. When they are out
    # of phase (or wound oppositely) vertex 0 of the rim gets stitched to a hole vertex on the
    # far side of the ring: long thin triangles drag across the opening on one side and the
    # opposite side is left unclosed -- the spikes-and-hole failure.
    # ---- match the rim's spacing to the hole's, so the bridge is a strip not a fan ----------
    # bmesh.ops.subdivide_edges REALLOCATES, so every BMVert reference held across it dies with
    # "BMesh data of type BMVert has been removed". Work in indices, re-fetch after the op, and
    # rebuild the ring by walking the donor's boundary. Wrapped: on any failure the original rim
    # is restored and the old fan-and-cap bridge runs, so this can never cost a graft.
    if RIM_MATCH_PASSES > 0 and len(H) > len(R):
        _n0, _ridx_keep = len(R), [v.index for v in R]
        try:
            _Hc0 = np.array([list(_h.co) for _h in H], float)
            _hsp = float(np.median(np.linalg.norm(np.roll(_Hc0, -1, axis=0) - _Hc0, axis=1)))
            _hset = set(v.index for v in H)
            for _pass in range(RIM_MATCH_PASSES):
                _Rc0 = np.array([list(_r.co) for _r in R], float)
                _rl = np.linalg.norm(np.roll(_Rc0, -1, axis=0) - _Rc0, axis=1)
                _cut = []
                for _i in range(len(R)):
                    if _rl[_i] > 1.5 * _hsp:
                        _e = bmb.edges.get((R[_i], R[(_i + 1) % len(R)]))
                        if _e is not None:
                            _cut.append(_e)
                if not _cut:
                    break
                _seed = R[0].index
                bmesh.ops.subdivide_edges(bmb, edges=_cut, cuts=1, use_grid_fill=False)
                bmb.verts.ensure_lookup_table(); bmb.edges.ensure_lookup_table()
                # walk the donor's boundary loop from the seed, skipping the body's hole loop
                _cur = bmb.verts[_seed]
                _ring, _prev = [_cur], None
                while True:
                    _nx = None
                    for _e in _cur.link_edges:
                        if not _e.is_boundary:
                            continue
                        _o = _e.other_vert(_cur)
                        if _o is _prev or _o.index in _hset:
                            continue
                        _nx = _o
                        break
                    if _nx is None or _nx is _ring[0]:
                        break
                    _ring.append(_nx); _prev, _cur = _cur, _nx
                    if len(_ring) > 8 * _n0:
                        raise RuntimeError("rim walk did not close (%d verts)" % len(_ring))
                if len(_ring) <= len(R):
                    raise RuntimeError("rim walk shrank: %d -> %d" % (len(R), len(_ring)))
                R = _ring
            print("[cutjoin] rim match: %d -> %d rim verts against %d hole verts "
                  "(hole spacing %.5f; advance %.2f -> %.2f)"
                  % (_n0, len(R), len(H), _hsp, len(H) / max(_n0, 1), len(H) / max(len(R), 1)),
                  flush=True)
        except Exception as _rm:
            bmb.verts.ensure_lookup_table()
            R = [bmb.verts[_i] for _i in _ridx_keep]
            print("[cutjoin] rim match ABANDONED (%r) -- original %d-vert rim restored, "
                  "bridging with the fan-and-cap path" % (_rm, len(R)), flush=True)

    _Rc = np.array([list(_r.co) for _r in R], float)
    _Hc = np.array([list(_h.co) for _h in H], float)

    def _areavec(P):
        Q = P - P.mean(0)
        return np.cross(Q, np.roll(Q, -1, axis=0)).sum(0)

    _stage_dump("1b_snapped", bmb, A)
    _rev = float(np.dot(_areavec(_Rc), _areavec(_Hc))) < 0.0
    if _rev:
        H = H[::-1]; _Hc = _Hc[::-1]
    _k0 = int(np.argmin(np.linalg.norm(_Hc - _Rc[0], axis=1)))
    if _k0:
        H = H[_k0:] + H[:_k0]; _Hc = np.roll(_Hc, -_k0, axis=0)
    _pair = np.linalg.norm(_Hc[(np.arange(len(_Rc)) * len(_Hc) // max(len(_Rc), 1)) % len(_Hc)] - _Rc, axis=1)
    nR = len(R); nH = len(H)
    print(f"[cutjoin] bridge align: winding {'REVERSED' if _rev else 'ok'}, "
          f"phase roll {_k0}/{len(_Hc)}; index-paired rim->hole dist med {np.median(_pair):.5f} "
          f"max {_pair.max():.5f}", flush=True)
    print(f"[cutjoin] rim verts nR={nR}  hole verts nH={nH}  ratio={nR/max(nH,1):.2f}", flush=True)
    # PAIR BY PROXIMITY, not by index fraction. The loops differ in both length (hole 2.21x the
    # rim here) and shape, so "30% along the rim" and "30% along the hole" are 22 rim-spacings
    # apart -- the walk then drags long triangles across the opening on one side and runs out on
    # the other (spikes + hole). Map each rim vertex to its NEAREST hole vertex, unwrap that map
    # so it advances monotonically around the ring, then fan the hole vertices between
    # consecutive rim vertices. Handles unequal counts and dissimilar shapes by construction.
    _mn = np.argmin(np.linalg.norm(_Rc[:, None, :] - _Hc[None, :, :], axis=2), axis=1)
    # The raw nearest map is not monotonic (a few rim verts pick a hole vert slightly behind
    # their neighbour), so its diffs do not sum to nH. Unwrap it into a non-decreasing walk that
    # advances at most one full lap: each step takes the forward distance to that rim vertex's
    # own nearest hole vertex, and any step that would overshoot the lap is held at zero. The
    # last edge absorbs the remainder, so H is traversed exactly once by construction.
    # Raw per-edge advance straight from the proximity map (no running budget). A nearest vertex
    # slightly BEHIND the previous one comes back from the modulo as an almost-full lap, which is
    # not a real advance.
    _raw = np.empty(nR, dtype=int)
    for _i in range(nR):
        _d = (int(_mn[(_i + 1) % nR]) - int(_mn[_i])) % nH
        _raw[_i] = 0 if _d > nH // 2 else _d
    # APPORTION the shortfall instead of dumping it on the last edge. The old walk carried a
    # running budget and gave the final edge whatever was left over -- measured 180 of 217 edges
    # advancing zero and the remainder, 106 hole vertices, fanned onto ONE rim vertex. That is the
    # star-shaped pinch at the top of the graft: a valence-106 pole. Spread the difference across
    # the rim edges in proportion to their LENGTH (a long rim edge should span more hole vertices
    # than a short one), by largest remainder, so the total is nH by construction and no single
    # edge can collect a fan.
    _rlen = np.linalg.norm(np.roll(_Rc, -1, axis=0) - _Rc, axis=1)
    _diff = int(nH) - int(_raw.sum())
    if BRIDGE_ALLOC == "local":
        # Proximity counts, total corrected proportionally (which preserves the correspondence and
        # keeps the graft clean), then ONLY the fans are relieved -- and locally. Spreading the
        # excess across the whole ring by length share touches all 217 edges to cure a defect at
        # one of them, and that blanket move is what tore the graft open. An edge over the fan
        # limit hands its surplus to its NEAREST neighbours with room, walking outward, so nothing
        # beyond the pole's immediate neighbourhood is disturbed.
        _sc = _raw * (float(nH) / max(float(_raw.sum()), 1.0))
        _raw = np.floor(_sc).astype(int)
        _rem = int(nH) - int(_raw.sum())
        if _rem > 0:
            _raw[np.argsort(-(_sc - _raw))[:_rem]] += 1
        _cap = max(2, int(np.ceil(BRIDGE_FAN * nH / max(nR, 1))))
        _relieved = 0
        for _j in np.argsort(-_raw):
            if _raw[_j] <= _cap:
                break
            _ex = int(_raw[_j]) - _cap; _raw[_j] = _cap; _relieved += _ex
            _r = 1
            while _ex > 0 and _r < nR:
                for _sgn in (-1, 1):
                    _t = int((_j + _sgn * _r) % nR)
                    _room = _cap - int(_raw[_t])
                    if _room > 0:
                        _put = min(_room, _ex); _raw[_t] += _put; _ex -= _put
                        if _ex == 0:
                            break
                _r += 1
            if _ex > 0:
                _raw[_j] += _ex
        print(f"[cutjoin] bridge: fan cap {_cap} (= {BRIDGE_FAN:.1f}x mean {nH/max(nR,1):.2f}); "
              f"{_relieved} surplus hole verts handed to NEAREST edges with room "
              f"(max per edge now {int(_raw.max())})", flush=True)
    elif BRIDGE_ALLOC == "capped":
        # Proximity counts keep the pairing locally correct, but where the cut still runs wide of
        # the rim they pile up: measured 96 hole vertices onto ONE rim edge, a valence-100 pole --
        # the star-shaped pinch. Pure length apportionment cannot fan (max valence 11) but throws
        # away the correspondence and scatters slivers down the wide side (45). So: keep proximity,
        # forbid any edge from taking more than BRIDGE_FAN x the mean advance, and hand the excess
        # to the edges that still have room, in proportion to their rim length.
        _cap = max(2, int(np.ceil(BRIDGE_FAN * nH / max(nR, 1))))
        _over = int(np.maximum(_raw - _cap, 0).sum())
        _raw = np.minimum(_raw, _cap)
        _need = int(nH) - int(_raw.sum())
        _wgt = _rlen if _rlen.sum() > 0 else np.ones(nR)
        while _need > 0:
            _room = _cap - _raw
            _elig = _room > 0
            if not _elig.any():
                _cap += 1
                continue
            _w = np.where(_elig, _wgt, 0.0)
            _q = _need * _w / max(_w.sum(), 1e-12)
            _add = np.minimum(np.floor(_q).astype(int), _room)
            if _add.sum() == 0:                      # remainder smaller than one per edge
                _ord = np.argsort(-(_q * _elig))
                for _j in _ord[:_need]:
                    if _room[_j] > 0:
                        _raw[_j] += 1; _need -= 1
                continue
            _raw += _add; _need -= int(_add.sum())
        print(f"[cutjoin] bridge: fan cap {_cap} (= {BRIDGE_FAN:.1f}x mean {nH/max(nR,1):.2f}); "
              f"{_over} over-cap hole verts redistributed by rim-edge length", flush=True)
    elif BRIDGE_ALLOC == "length":
        # Purely geometric apportionment: an edge spans the share of hole vertices matching its
        # share of the rim's length. The loops now coincide (gap med 2x the rim spacing), so this
        # IS the correspondence -- and being proportional it cannot produce a fan, whereas the
        # proximity counts are lumpy wherever the cut still runs wide.
        _wgt = _rlen if _rlen.sum() > 0 else np.ones(nR)
        _q = nH * _wgt / _wgt.sum()
        _raw = np.floor(_q).astype(int)
        _rem = int(nH) - int(_raw.sum())
        if _rem > 0:
            _raw[np.argsort(-(_q - _raw))[:_rem]] += 1
    elif _diff != 0:
        _wgt = _rlen if _rlen.sum() > 0 else np.ones(nR)
        if _diff > 0:
            _q = _diff * _wgt / _wgt.sum()
            _add = np.floor(_q).astype(int)
            _rem = _diff - int(_add.sum())
            if _rem > 0:
                _add[np.argsort(-(_q - _add))[:_rem]] += 1
            _raw = _raw + _add
        else:
            _sc = _raw * (float(nH) / max(float(_raw.sum()), 1.0))
            _raw = np.floor(_sc).astype(int)
            _rem = int(nH) - int(_raw.sum())
            if _rem > 0:
                _raw[np.argsort(-(_sc - _raw))[:_rem]] += 1
    _adv = _raw
    _held = int((_adv == 0).sum())
    print(f"[cutjoin] bridge: apportioned {int(_adv.sum())}/{nH} hole verts over {nR} rim edges "
          f"({_held} edges take none; max per edge {int(_adv.max())}, "
          f"shortfall {_diff:+d} spread by rim-edge length)", flush=True)
    _pd = np.linalg.norm(_Hc[_mn] - _Rc, axis=1)
    print(f"[cutjoin] bridge: proximity-paired rim->hole dist med {np.median(_pd):.5f} "
          f"max {_pd.max():.5f}  (was med {np.median(_pair):.5f})", flush=True)
    _k = int(_mn[0])
    # Every bridge face was created under a bare try/except pass: a face that fails to build is a
    # HOLE in the seam and nothing said so. Count them by kind so a gap in the bridge is visible in
    # the log instead of only in the render.
    _fok = _fdegen = _fdup = _ferr = _flong = 0
    # REFUSE to span open space. The bridge's instruction is "connect the rim to the hole", and
    # where the rim hangs over the gap between the legs the nearest body is the FAR THIGH, so it
    # draws a triangle straight across -- the fin. Nothing should exist in that gap, so cap the
    # edge length: a legitimate bridge edge is about one rim spacing, the fin's are 7-25x it.
    # A refused face leaves the seam open there, which is correct -- there is nothing to seam to
    # -- and fill_holes closes it along the surface afterwards instead of across the void.
    _rimsp = _pr / max(nR, 1)
    _maxe = BRIDGE_MAXEDGE * _rimsp
    def _mkface(vs):
        nonlocal _fok, _fdegen, _fdup, _ferr, _flong
        if len(set(id(x) for x in vs)) < len(vs):
            _fdegen += 1; return
        if BRIDGE_MAXEDGE > 0:
            _p = [np.array(x.co) for x in vs]
            if max(float(np.linalg.norm(_p[i] - _p[(i + 1) % len(_p)])) for i in range(len(_p))) > _maxe:
                _flong += 1; return
        try:
            bmb.faces.new(vs); _fok += 1
        except ValueError:
            _fdup += 1
        except Exception:
            _ferr += 1
    if BRIDGE_ZIPPER:
        _ri, _hi = 0, 0                      # steps taken on each loop
        _h0 = _k
        _val = {}
        while _ri < nR or _hi < nH:
            _rc = R[_ri % nR]; _hc = H[(_h0 + _hi) % nH]
            _rn = R[(_ri + 1) % nR]; _hn = H[(_h0 + _hi + 1) % nH]
            if _ri >= nR:
                _adv_rim = False
            elif _hi >= nH:
                _adv_rim = True
            else:
                _adv_rim = (_rn.co - _hc.co).length <= (_rc.co - _hn.co).length
            if _adv_rim:
                _mkface([_rc, _rn, _hc]); _ri += 1
                _val[id(_hc)] = _val.get(id(_hc), 0) + 1
            else:
                _mkface([_hc, _rc, _hn]); _hi += 1
                _val[id(_rc)] = _val.get(id(_rc), 0) + 1
        print(f"[cutjoin] bridge ZIPPER: {nR} rim + {nH} hole steps, shortest new edge each time; "
              f"most triangles fanned onto one vertex: {max(_val.values()) if _val else 0}", flush=True)
    else:
      for i in range(nR):
        _a = R[i]; _b = R[(i + 1) % nR]
        _mkface([_a, _b, H[_k % nH]])
        for _ in range(int(_adv[i])):
            _k2 = (_k + 1) % nH
            _mkface([H[_k % nH], _b, H[_k2]])
            _k = _k2
    if BRIDGE_MAXEDGE > 0:
        print(f"[cutjoin] bridge: refused {_flong} face(s) with an edge over "
              f"{BRIDGE_MAXEDGE:.1f}x the rim spacing ({_maxe:.5f}) -- those are the ones that "
              f"would have spanned the gap between the legs", flush=True)
    _dtrace("after bridge", bmb, _dlay, _Wplaced)
    _fbad = _fdegen + _fdup + _ferr
    _trace("after bridge", bmb, A)
    _stage_dump("2_bridge", bmb, A)
    print(f"[cutjoin] bridge faces: {_fok} built"
          + (f"  *** {_fbad} FAILED = holes in the seam "
             f"(degenerate {_fdegen}, duplicate {_fdup}, other {_ferr}) ***" if _fbad else "  (no failures)"),
          flush=True)
    bmb.edges.ensure_lookup_table()
    rimco = np.array([W[li] for li in loop]); rimv = [dvm[li] for li in loop]; rimset = set(rimv)
    opv = set()
    for e in bmb.edges:
        if len(e.link_faces) == 1:
            for vv in e.verts:
                if vv.is_valid and vv not in rimset and id(vv) not in donset and np.linalg.norm(np.array(vv.co) - A) < 0.12: opv.add(vv)
    # Weld leftover open body vertices onto the rim -- but only ones that are actually AT the
    # rim. The old bound was a flat 0.02 (20 mm at 170 cm), which is not "closing a seam": every
    # face touching such a vertex gets dragged up to 20 mm onto one rim vertex, and since many
    # strays pick the same nearest rim vertex they all converge there. That is the fan of long
    # triangles in the leg gap. Bound it by the rim's OWN spacing instead, so a weld only ever
    # closes a gap of a face or two.
    _wsp = _pr / max(len(loop), 1)
    _wmax = WELD_MAX * _wsp
    tm = {}
    _wfar = 0
    for vv in opv:
        p = np.array(vv.co); k = int(np.argmin(((rimco - p) ** 2).sum(1)))
        _wd = float(np.linalg.norm(rimco[k] - p))
        if _wd < _wmax:
            tm[vv] = rimv[k]
        elif _wd < 0.02:
            _wfar += 1
    print(f"[cutjoin] rim weld: {len(tm)} stray open vert(s) welded within {WELD_MAX:.1f}x the "
          f"rim spacing ({_wmax:.5f}); {_wfar} further out left alone (the old rule welded those "
          f"too, up to 0.02, and each one dragged its faces onto a single rim vertex)", flush=True)
    if tm: bmesh.ops.weld_verts(bmb, targetmap=tm)
    # Whatever the weld did not close, FILL where it is. Dragging a stray vertex onto a rim
    # vertex 20 mm away closes the hole by moving geometry across the leg gap; filling closes it
    # without moving anything. Restricted to openings near the anchor so a hole elsewhere in the
    # body (there are pre-existing ones) is left exactly as it was.
    if HOLE_FILL:
        bmb.edges.ensure_lookup_table()
        _open = [e for e in bmb.edges if len(e.link_faces) == 1
                 and np.linalg.norm(0.5 * (np.array(e.verts[0].co) + np.array(e.verts[1].co)) - A) < 0.12]
        if _open:
            _nf0 = len(bmb.faces)
            try:
                bmesh.ops.holes_fill(bmb, edges=_open, sides=0)
                bmb.edges.ensure_lookup_table()
                _left = sum(1 for e in bmb.edges if len(e.link_faces) == 1
                            and np.linalg.norm(0.5 * (np.array(e.verts[0].co)
                                                      + np.array(e.verts[1].co)) - A) < 0.12)
                print(f"[cutjoin] hole fill: {len(_open)} open edge(s) near the graft -> "
                      f"{len(bmb.faces) - _nf0} face(s) added, {_left} still open", flush=True)
            except Exception as _he:
                print(f"[cutjoin] hole fill failed ({_he!r})", flush=True)
    _dtrace("after weld/fill", bmb, _dlay, _Wplaced)
    _stage_dump("3_weld", bmb, A)
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    # Where each donor vertex lands in the merged mesh. The self-intersection relax and the brush smooth below move
    # donor verts AFTER W is final, so the donor sidecar reads their final positions through this map (a vertex a
    # weld consumed is -1 and keeps W). to_mesh writes vertices in index order and nothing below reorders them.
    try:
        bmb.verts.index_update()
        globals()['_DONOR_MESH_IDX'] = np.array([x.index if x.is_valid else -1 for x in dvm], np.int64)
    except Exception:
        globals()['_DONOR_MESH_IDX'] = None
    bmb.to_mesh(me); bmb.free(); me.update()
    bm2 = bmesh.new(); bm2.from_mesh(me); bm2.verts.ensure_lookup_table()
    allv = list(bm2.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    # PROTECT labeled detail (outer/inner labia + opening = donor zones 1-4): KD-tree of those donor verts'
    # placed positions so they are NEVER smoothed, even where the rim hugs them (near the anus). User spec:
    # inner parts 100% protected, smoothing happens only OUTWARD of them.
    _detail_kd = None
    try:
        _zn = np.load(A_ZONES)
        if len(_zn) == len(W):
            # PELVIS_BRUSH_PROTECT_ZONES (default 1,2,3,4): the zones kept crisp by the brush smooth. 2026-09-16:
            # the user's own smoothing protected only the inner lips + opening and smoothed the outer lips
            # and the seam -- with all four frozen no number of iterations can reach the fins there.
            _pz = [int(x) for x in os.environ.get('PELVIS_BRUSH_PROTECT_ZONES', '1,2,3,4').split(',') if x.strip()]
            _dsel = np.isin(_zn, _pz)
            # PELVIS_PROTECT_ANUS=1 (2026-09-19, user: "anus not deep enough"): the anus core (donor anus weight >= 0.5)
            # is detail too -- the fit already treats it so. Unprotected, the 60-pass brush smooth squeezed the anal
            # canal's mouth from 8.8 to 5.8 mm (female_elf_00005_) and it read as a shallow 4 mm crescent.
            if os.environ.get("PELVIS_PROTECT_ANUS", "0") == "1":
                _awf = os.environ.get("PELVIS_ANUS_W", "")
                _awd = np.load(_awf) if _awf and os.path.exists(_awf) else None
                if _awd is not None and len(_awd) == len(W):
                    _dsel = _dsel | (_awd >= 0.5)
                    print(f"[cutjoin] anus protect: {int((_awd >= 0.5).sum())} anus-core verts join the protected detail",
                          flush=True)
                else:
                    print("[cutjoin] WARN: anus protect asked for but no matching anus weights -- not protected", flush=True)
            _dp = W[_dsel]
            if len(_dp):
                _detail_kd = kdtree.KDTree(len(_dp))
                for _i in range(len(_dp)): _detail_kd.insert(Vector(_dp[_i].tolist()), _i)
                _detail_kd.balance()
    except Exception as _ze:
        print(f"[cutjoin] WARN: detail-zone protect off ({_ze})", flush=True)
    # GEODESIC band. `poly` is a planar SHADOW of the rim (see SEAM_GEO), so `d2r` below is the
    # distance to that shadow, not to the seam. Measure it along the surface instead: seed on the
    # donor/body boundary edges (the seam itself, read off the pg_donor tag) and Dijkstra outward
    # by edge length. The gap between the thighs is not an edge, so the band cannot jump it.
    _geod = None; _dmask_g = None
    if SEAM_GEO:
        import heapq as _hq
        _dvv = np.zeros(len(allv), np.int32)
        _atg = me.attributes.get("pg_donor")
        if _atg is not None and len(_atg.data) == len(allv):
            _atg.data.foreach_get("value", _dvv)
        _evv = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", _evv)
        _evv = _evv.reshape(-1, 2)
        _dmask = _dvv > 0; _dmask_g = _dmask
        _bnd = _dmask[_evv[:, 0]] != _dmask[_evv[:, 1]]
        _seed = np.unique(_evv[_bnd].ravel()) if _bnd.any() else np.zeros(0, np.int32)
        _ball = dA2 < 0.1
        _sel = _ball[_evv[:, 0]] & _ball[_evv[:, 1]]
        _e2 = _evv[_sel]
        _elen = np.linalg.norm(co[_e2[:, 0]] - co[_e2[:, 1]], axis=1)
        _nb = [[] for _ in range(len(allv))]
        for _k in range(len(_e2)):
            _a2, _b2 = int(_e2[_k, 0]), int(_e2[_k, 1]); _L2 = float(_elen[_k])
            _nb[_a2].append((_b2, _L2)); _nb[_b2].append((_a2, _L2))
        _geod = np.full(len(allv), np.inf)
        _pq = []
        for _sv in _seed.tolist():
            if _ball[_sv]:
                _geod[_sv] = 0.0; _hq.heappush(_pq, (0.0, int(_sv)))
        while _pq:
            _dc, _cv = _hq.heappop(_pq)
            if _dc > _geod[_cv] or _dc > _seam_band: continue
            for _nv, _w2 in _nb[_cv]:
                _nd = _dc + _w2
                if _nd < _geod[_nv] and _nd <= _seam_band:
                    _geod[_nv] = _nd; _hq.heappush(_pq, (_nd, _nv))
        _reach = int(np.isfinite(_geod).sum())
        _bi = np.where(_ball)[0]
        _uv = np.stack([(co[_bi] - A) @ u, (co[_bi] - A) @ v], 1)
        _sh = np.concatenate([np.sqrt(((_uv[_c:_c + 20000, None, :] - poly[None, :, :]) ** 2)
                                      .sum(2)).min(1)
                              for _c in range(0, len(_uv), 20000)]) if len(_uv) else np.zeros(0)
        print(f"[cutjoin] seam band: GEODESIC {_seam_band:.5f} "
              f"({_seam_band / 1.0016 * 1700:.1f} mm) measured along the surface from "
              f"{len(_seed)} seam vert(s) -> {_reach} vert(s); the planar shadow it replaces "
              f"reached {int((_sh < _seam_band).sum())}", flush=True)
    zone = set(); _prot = 0; _sw = {}
    _seedset = set(int(x) for x in _seed.tolist()) if (_geod is not None and SEAM_DONOR_RIM_ONLY) else None
    _skipd = 0
    for i in range(len(allv)):
        if dA2[i] > 0.1: continue
        if _seedset is not None and _dmask_g is not None and _dmask_g[i] and i not in _seedset:
            _skipd += 1
            continue
        if DONOR_FROZEN and _dmask_g is not None and _dmask_g[i]:
            _skipd += 1
            continue
        if _geod is not None:
            d2r = float(_geod[i])
        else:
            dd = co[i] - A; d2r = float(np.min(np.sqrt(((poly - np.array([dd @ u, dd @ v])) ** 2).sum(1))))
        if d2r < _seam_band:
            # FEATHERED protection. A hard on/off boundary froze one vertex solid while its direct
            # neighbour took the full 0.5-Laplacian x GRAFT_SEAM_ITERS treatment, so faces spanning
            # that line stretched into slivers -- measured aspect up to 148.8, while the donor asset's
            # own worst is 15.5, i.e. the mesh is clean and the discontinuity created them.
            # Inner detail stays 100% protected (w = 0 within GRAFT_PROTECT_R); the weight ramps to
            # full smoothing over one further protect-radius, so only the OUTER edge moves.
            _w = 1.0
            if _detail_kd is not None:
                _dd = _detail_kd.find(Vector(co[i].tolist()))[2]
                _w = float(np.clip((_dd - GRAFT_PROTECT_R) / max(GRAFT_PROTECT_R, 1e-9), 0.0, 1.0))
                if _w <= 0.0:
                    _prot += 1; continue
            if _geod is not None:
                # taper to nothing at the band edge. A hard on/off boundary IS a crease: the
                # vertex just inside took 120 full Laplacian passes and its neighbour took none.
                _w *= float(np.clip(1.0 - d2r / max(_seam_band, 1e-12), 0.0, 1.0))
                if _w <= 0.0:
                    continue
            zone.add(i); _sw[i] = _w
    print(f"[cutjoin] detail-zone protect: spared {_prot} labeled labia/opening verts from the seam smooth", flush=True)
    if DONOR_FROZEN and _dmask_g is not None:
        print(f"[cutjoin] donor frozen: seam smooth skips all {int(_dmask_g.sum())} donor vert(s)", flush=True)
    if _seedset is not None:
        print(f"[cutjoin] seam smooth: donor interior left alone -- {_skipd} donor vert(s) off the seam not smoothed", flush=True)
    _dz = os.environ.get("PELVIS_DUMP_ZONE", "")
    if _dz:
        _zw = np.zeros(len(allv), np.float32)
        for _i2, _w2 in _sw.items(): _zw[_i2] = _w2
        np.save(_dz, _zw)
        print(f"[cutjoin] seam-zone weights: {int((_zw > 0).sum())} vert(s) -> {_dz}", flush=True)
    # 49 s of a 188 s graft lived here, in two places that had nothing to do with the smoothing:
    # a 3.27M-edge sweep to find the neighbours of 46,718 zone vertices, and 120 iterations of a
    # per-vertex Python np.mean (5.6M calls). Neighbours now come from the zone vertices' own
    # edges, and the Laplacian is one flat gather per iteration.
    _zl = sorted(zone)
    _zpos = {v: k for k, v in enumerate(_zl)}
    _flat, _off, _w = [], [0], []
    for i in _zl:
        _ns = {idx[e.other_vert(allv[i])] for e in allv[i].link_edges}
        _flat.extend(_ns)
        _off.append(len(_flat))
        _w.append(_sw.get(i, 1.0))
    # a trailing dummy so an empty segment's reduceat index can never run off the end; those
    # rows are discarded by _has anyway
    _flat = np.asarray(_flat + [0], np.int64)
    _cnt = np.diff(np.asarray(_off, np.int64))
    _seg = np.asarray(_off[:-1], np.int64)
    _zi = np.asarray(_zl, np.int64)
    _wv = np.asarray(_w, float)[:, None]
    _has = _cnt > 0
    if SMOOTH_NORMAL_ONLY:
        _nz = np.zeros((len(_zi), 3))
        for _k9, _i9 in enumerate(_zl):
            _acc9 = Vector((0.0, 0.0, 0.0))
            for _f9 in allv[_i9].link_faces:
                _f9.normal_update()
                _acc9 += _f9.normal * _f9.calc_area()
            if _acc9.length > 1e-20:
                _nz[_k9] = _acc9.normalized().to_tuple()
        print(f"[cutjoin] seam smooth: moves restricted to the vertex normal ({len(_zl)} verts)", flush=True)
    P = co.copy()
    for _ in range(GRAFT_SEAM_ITERS):
        _sum = np.add.reduceat(P[_flat], _seg, axis=0)
        _mean = np.where(_has[:, None], _sum / np.maximum(_cnt, 1)[:, None], P[_zi])
        _lap = 0.5 * P[_zi] + 0.5 * _mean
        _stp = _wv * (_lap - P[_zi])
        if SMOOTH_NORMAL_ONLY:
            _stp = np.einsum('ij,ij->i', _stp, _nz)[:, None] * _nz
        P[_zi] = np.where(_has[:, None], P[_zi] + _stp, P[_zi])
    # Bound the drift, as the crease relax is bounded. 120 unbounded Laplacian iterations do not
    # just smooth a seam -- they let a vertex migrate: traced, donor vertices ended up as much as
    # 74 mm from where they were placed, tens of mm below the donor, and the spike in the leg gap
    # was a fan hanging off them. Budget is a multiple of each vertex's OWN edge length.
    if SEAM_MAX > 0:
        _es = [e for e in bm2.edges[:20000]]
        _medS = float(np.median([ (np.array(e.verts[0].co) - np.array(e.verts[1].co))
                                  for e in _es ] and
                                [np.linalg.norm(np.array(e.verts[0].co) - np.array(e.verts[1].co))
                                 for e in _es])) if _es else 0.0015
        _budS = np.full(len(_zi), SEAM_MAX * _medS)
        _dS = P[_zi] - co[_zi]
        _nS = np.linalg.norm(_dS, axis=1)
        _ovS = (_nS > _budS) & (_budS > 0)
        if _ovS.any():
            _scS = np.ones(len(_zi))
            _scS[_ovS] = _budS[_ovS] / np.maximum(_nS[_ovS], 1e-12)
            P[_zi] = co[_zi] + _dS * _scS[:, None]
        print(f"[cutjoin] seam smooth: drift cap {SEAM_MAX:.1f}x the local edge held back "
              f"{int(_ovS.sum())} of {len(_zi)} vert(s); worst move {_nS.max() / 1.0016 * 1700:.1f}"
              f" -> {np.linalg.norm(P[_zi] - co[_zi], axis=1).max() / 1.0016 * 1700:.1f} mm",
              flush=True)
    for i in _zl: allv[i].co = Vector(P[i].tolist())
    _stage_dump("4_seam_smooth", bm2, A)
    # gentle MONS de-bump (KAN-18): the pubic mound just above the vulva reads bumpy. Lightly smooth the FRONT
    # skin (donor zone 0) within MONS_R of the vulva centre. Labia (zones 1-4) stay protected; the anus/perineum
    # are further back than MONS_R, untouched. Light (MONS_ITERS x MONS_W) so it de-bumps without flattening.
    if _detail_kd is not None:
        print("[t] mons: start", flush=True)
        _co2 = np.array([list(x.co) for x in allv])
        _vc = W[np.isin(_zn, [3, 4])].mean(0)            # vulva centre (inner labia + opening)
        _skin = W[_zn == 0]
        print("[t] mons: co2 materialised, building skin KD", flush=True)
        _skd = kdtree.KDTree(len(_skin))
        for _i in range(len(_skin)): _skd.insert(Vector(_skin[_i].tolist()), _i)
        _skd.balance()
        # The vulva-radius gate rejects all but a few thousand vertices, so apply it VECTORISED
        # instead of walking 1.09M in Python with a norm each; and reach the neighbours through
        # the mons vertices themselves rather than scanning all 3.27M edges. 52 s of a 608 s
        # graft was in these two loops, and neither was doing anything the local set cannot.
        print("[t] mons: skin KD built, gating", flush=True)
        mons = set()
        _mcand = np.where(np.linalg.norm(_co2 - _vc, axis=1) <= MONS_R)[0]
        for i in _mcand:
            p = _co2[i]
            if DONOR_FROZEN and _dmask_g is not None and _dmask_g[int(i)]: continue
            if _detail_kd.find(Vector(p.tolist()))[2] < GRAFT_PROTECT_R: continue      # protect labia
            if _skd.find(Vector(p.tolist()))[2] < 0.004: mons.add(int(i))              # donor skin (zone 0), not body
        print("[t] mons: candidates chosen, building neighbours", flush=True)
        mnb = {i: set() for i in mons}
        for i in mons:
            for e in allv[i].link_edges:
                mnb[i].add(idx[e.other_vert(allv[i])])
        print("[t] mons: neighbours built, iterating", flush=True)
        Pm = _co2.copy()
        for _ in range(MONS_ITERS):
            nPm = Pm.copy()
            for i in mons:
                if mnb[i]: nPm[i] = (1 - MONS_W) * Pm[i] + MONS_W * np.mean([Pm[k] for k in mnb[i]], axis=0)
            Pm = nPm
        for i in mons: allv[i].co = Vector(Pm[i].tolist())
        print(f"[cutjoin] mons de-bump: gently smoothed {len(mons)} pubic-mound skin verts", flush=True)
    _stage_dump("5_mons", bm2, A)
    # CREASE RELAX. The seam smooth works on a band around the rim; it does not know whether the
    # result is actually smooth. Measured against the pre-graft body (5 edges over 60deg) the graft
    # left 479 -- hard facets the user has to sculpt out by hand. Fix it by the only standard that
    # means anything here: the BODY'S OWN smoothness. Take the p95 dihedral of the surrounding body
    # (dA 0.07-0.12, untouched by the graft), and relax every seam edge that exceeds CREASE_K x it.
    # No absolute angle, so it holds on any mesh density. Labeled labia/opening detail is exempt.
    if CREASE_ITERS > 0:
        import math as _m
        bm2.normal_update()
        _nb = {}
        for _e in bm2.edges:
            _a = idx[_e.verts[0]]; _b = idx[_e.verts[1]]
            _nb.setdefault(_a, set()).add(_b); _nb.setdefault(_b, set()).add(_a)
        # Gather the edges this pass can possibly care about ONCE. _angs() used to walk all
        # ~3.2M edges of the body, measure each one's distance to the anchor and throw away
        # everything past 0.12 -- and it is called again at the end of each of the 40
        # iterations, so that ran ~130M times. Vertices move by a fraction of an edge here, so
        # a 25% margin on the cut-off is ample.
        # "Near the anchor" at 0.12 is 1.25M of the body's 3.27M edges -- 38% of the character,
        # because the crotch sits in the middle of it. Re-measuring that 40 times was 260 s of
        # a 608 s graft. Two things fix it, and neither changes the result:
        #
        #   * the SEAM band (d < 0.07) is the only part the loop reads; the 0.07-0.12 band is
        #     the reference percentile and is measured exactly once;
        #   * only edges touching a vertex that MOVED can have changed angle, and the pass moves
        #     a few thousand, so the rescan is incremental instead of total.
        print("[t] crease: start edge gather", flush=True)
        # Still one sweep of all 3.27M edges, but it was building three numpy arrays per edge
        # (two np.array(co) and a norm) to answer "is this near the anchor" -- ~10M allocations.
        # Plain float arithmetic against squared radii gives the identical partition.
        _CR, _CSEAM = 0.12, 0.07
        _ax, _ay, _az = float(A[0]), float(A[1]), float(A[2])
        _cr2, _cs2 = _CR * _CR, _CSEAM * _CSEAM
        _seam_e, _ref_e = [], []
        # A 0.07 BALL around the anchor is 119 mm -- both whole inner thighs, the perineum and the
        # bottom of the buttocks, 270k edges. It is the same mistake as the seam smooth's planar
        # shadow: a cheap proxy for "near the seam" that is nothing of the kind, so the pass
        # relaxes real anatomy (the inguinal fold) against a threshold set by the graft. With
        # SEAM_GEO the region the graft actually touched is known exactly: the donor patch itself,
        # plus the geodesic blend band around its seam. Nothing chosen, nothing absolute.
        _gtouch = None
        if _geod is not None:
            _gtouch = np.isfinite(_geod)
            if _dmask_g is not None:
                _gtouch = _gtouch | _dmask_g
        for _e in bm2.edges:
            if len(_e.link_faces) != 2:
                continue
            _v0 = _e.verts[0].co; _v1 = _e.verts[1].co
            _dx = 0.5 * (_v0.x + _v1.x) - _ax
            _dy = 0.5 * (_v0.y + _v1.y) - _ay
            _dz = 0.5 * (_v0.z + _v1.z) - _az
            _q = _dx * _dx + _dy * _dy + _dz * _dz
            if _gtouch is not None:
                if _q > _cr2:
                    continue
                if _gtouch[idx[_e.verts[0]]] or _gtouch[idx[_e.verts[1]]]:
                    _seam_e.append((_e, _q ** 0.5))
                elif _q >= _cs2:
                    _ref_e.append(_e)
            elif _q < _cs2:
                _seam_e.append((_e, _q ** 0.5))
            elif _q <= _cr2:
                _ref_e.append(_e)
        print("[t] crease: edges gathered, indexing by vert", flush=True)
        _e_of_v = {}
        for _e, _d in _seam_e:
            for _v in _e.verts:
                _e_of_v.setdefault(idx[_v], []).append((_e, _d))
        print("[cutjoin] crease relax: %d seam edge(s) + %d reference edge(s) of %d in the body"
              % (len(_seam_e), len(_ref_e), len(bm2.edges)))

        def _ang1(_e):
            try:
                return _m.degrees(_e.calc_face_angle())
            except Exception:
                return None

        print("[t] crease: building live seam state", flush=True)
        # the reference band, measured once
        _ref = []
        for _e in _ref_e:
            _a = _ang1(_e)
            if _a is not None:
                _ref.append(_a)

        _cur = {}                       # edge -> (angle, dist, ia, ib), the live seam state
        for _e, _d in _seam_e:
            _a = _ang1(_e)
            if _a is not None:
                _cur[_e] = (_a, _d, idx[_e.verts[0]], idx[_e.verts[1]])

        def _angs():
            return list(_cur.values())

        def _rescan(_moved):
            """Only edges touching a vertex that moved can have changed."""
            _seen = set()
            for _i in _moved:
                for _e, _d in _e_of_v.get(_i, ()):
                    if id(_e) in _seen:
                        continue
                    _seen.add(id(_e))
                    _a = _ang1(_e)
                    if _a is not None:
                        _cur[_e] = (_a, _d, idx[_e.verts[0]], idx[_e.verts[1]])
            return len(_seen)

        _o = _angs()
        # _ref was measured above from the reference band
        if _ref and _o:
            _seamang = [x[0] for x in _o if x[1] < 0.07]
            _thr = max(CREASE_K * float(np.percentile(_ref, 95)),
                       float(np.percentile(_seamang, CREASE_PCT)) if _seamang else 0.0)
            # FEATHERED protection, not binary. A hard on/off froze 302 of the 374 crease vertices
            # solid -- a facet 1mm from labeled detail could not move at all, which is why the pass
            # did nothing. Weight by distance to the labeled detail instead: 0 ON the labia/opening
            # (the hard rule holds, they are never touched) rising to full at GRAFT_PROTECT_R, so
            # only the outer edges relax.
            # Only the vertices this pass can move need a protection weight. Querying the
            # KD-tree for all 1.09M body vertices to use a few thousand is the same waste in
            # miniature; everything far from the graft keeps weight 1.0 and never moves anyway.
            _wt = np.ones(len(allv))
            if _detail_kd is not None:
                _cv = set(_e_of_v.keys())
                for _i in _cv:
                    _dd = _detail_kd.find(Vector(list(allv[_i].co)))[2]
                    _wt[_i] = float(np.clip(_dd / max(GRAFT_PROTECT_R, 1e-9), 0.0, 1.0))
            _free = _wt > 0.0
            if DONOR_FROZEN and _dmask_g is not None and len(_dmask_g) == len(_free):
                _free = _free & ~_dmask_g
                print(f"[cutjoin] donor frozen: crease relax may not move {int(_dmask_g.sum())} donor vert(s)", flush=True)
            _n0 = sum(1 for x in _o if x[0] > _thr and x[1] < 0.07)
            _hv = set(); _pv = set()
            for _an, _d, _ia, _ib in _o:
                if _an > _thr and _d < 0.07:
                    for _i in (_ia, _ib):
                        (_hv if _free[_i] else _pv).add(_i)
            print(f"[cutjoin] crease relax: {_n0} hard seam edges -> {len(_hv)} movable verts, "
                  f"{len(_pv)} fully frozen on labeled detail (feather r={GRAFT_PROTECT_R})", flush=True)
            _P = np.array([list(x.co) for x in allv])
            # Bound how far the relax may MOVE a vertex. Flattening a fold is a local operation;
            # 40 unbounded Laplacian steps are not, and where the seam meets the coarse body the
            # neighbour mean drags a vertex a long way. Ablated: with the crease relax off, faces
            # stretched past 8x the local edge went 2 -> 0 and the worst edge 30.5 -> 22.2 mm, so
            # this pass -- not the cut and not the bridge -- is what produced the spike. Budget
            # is a multiple of each vertex's OWN edge length, so it scales with the mesh.
            _P0 = _P.copy()
            _ledge = np.zeros(len(allv))
            for _i in range(len(allv)):
                _k0 = _nb.get(_i)
                if _k0:
                    _ledge[_i] = float(np.mean(np.linalg.norm(_P[list(_k0)] - _P[_i], axis=1)))
            _budget = CREASE_MAX * _ledge
            _hitcap = 0
            _nrm9 = {}
            for _it in range(CREASE_ITERS):
                _hard = set()
                for _an, _d, _ia, _ib in _o:
                    if _an > _thr and _d < 0.07:
                        if _free[_ia]: _hard.add(_ia)
                        if _free[_ib]: _hard.add(_ib)
                if not _hard: break
                # Relaxing ONLY the vertices ON a hard edge barely converges (measured 1608 -> 1476
                # over 40 iterations): their neighbours are creased too and pull them straight back.
                # Dilate the set by CREASE_RING rings so the fold has somewhere to flatten into.
                for _r in range(CREASE_RING):
                    _hard |= {_j for _i in list(_hard) for _j in _nb.get(_i, ()) if _free[_j]}
                _Q = _P.copy()
                for _i in _hard:
                    _k = _nb.get(_i)
                    if _k:
                        _w = CREASE_W * _wt[_i]
                        _Q[_i] = (1 - _w) * _P[_i] + _w * _P[list(_k)].mean(0)
                        if SMOOTH_NORMAL_ONLY:
                            _n9 = _nrm9.get(_i)
                            if _n9 is None:
                                _acc9 = Vector((0.0, 0.0, 0.0))
                                for _f9 in allv[_i].link_faces:
                                    _acc9 += _f9.normal * _f9.calc_area()
                                _n9 = np.array(_acc9.normalized().to_tuple()) if _acc9.length > 1e-20 else np.zeros(3)
                                _nrm9[_i] = _n9
                            _Q[_i] = _P[_i] + float(np.dot(_Q[_i] - _P[_i], _n9)) * _n9
                if CREASE_MAX > 0:
                    _hl = np.fromiter(_hard, int, len(_hard))
                    _dv2 = _Q[_hl] - _P0[_hl]
                    _dn2 = np.linalg.norm(_dv2, axis=1)
                    _ov = _dn2 > _budget[_hl]
                    if _ov.any():
                        _hitcap = max(_hitcap, int(_ov.sum()))
                        _sc2 = np.ones(len(_hl))
                        _sc2[_ov] = _budget[_hl][_ov] / np.maximum(_dn2[_ov], 1e-12)
                        _Q[_hl] = _P0[_hl] + _dv2 * _sc2[:, None]
                _P = _Q
                for _i in _hard: allv[_i].co = Vector(_P[_i].tolist())
                # calc_face_angle needs current face normals, but only the faces around the
                # moved vertices changed -- updating those beats a 1.09M-vertex normal_update.
                _touched = set()
                for _i in _hard:
                    for _f in allv[_i].link_faces:
                        _touched.add(_f)
                for _f in _touched:
                    _f.normal_update()
                _rescan(_hard)
                _o = _angs()
            _n1 = sum(1 for x in _o if x[0] > _thr and x[1] < 0.07)
            print(f"[cutjoin] crease relax: body p95 {np.percentile(_ref,95):.2f}deg, seam p{CREASE_PCT:.0f} "
                  f"{np.percentile(_seamang, CREASE_PCT):.2f}deg -> threshold {_thr:.2f}deg; "
                  f"seam edges over it {_n0} -> {_n1} ({CREASE_ITERS} iters, labeled detail "
                  f"exempt); drift cap {CREASE_MAX:.1f}x the local edge held back up to "
                  f"{_hitcap} vert(s) in an iteration", flush=True)
    _trace("after crease relax", bmb, A)
    _stage_dump("6_crease", bm2, A)
    if XSECT_ITERS > 0:
        from mathutils.bvhtree import BVHTree
        bm2.faces.ensure_lookup_table(); bm2.verts.ensure_lookup_table(); bm2.verts.index_update()
        print("[t] xsect: start weights", flush=True)
        _wt2 = np.ones(len(allv))
        if _detail_kd is not None and XSECT_PROTECT:
            _wax, _way, _waz = float(A[0]), float(A[1]), float(A[2])
            _w2 = 0.126 ** 2
            _xv = set()
            for _f in bm2.faces:
                _cc = _f.calc_center_median()
                _dx = _cc.x - _wax; _dy = _cc.y - _way; _dz = _cc.z - _waz
                if _dx * _dx + _dy * _dy + _dz * _dz < _w2:
                    for _v in _f.verts:
                        _xv.add(_v.index)
            for _i in _xv:
                _dd = _detail_kd.find(Vector(list(allv[_i].co)))[2]
                _wt2[_i] = float(np.clip(_dd / max(GRAFT_PROTECT_R, 1e-9), 0.0, 1.0))
        if DONOR_FROZEN and _dmask_g is not None and len(_dmask_g) == len(_wt2):
            _wt2[_dmask_g] = 0.0
            print(f"[cutjoin] donor frozen: self-intersection relax weight 0 on {int(_dmask_g.sum())} donor vert(s)", flush=True)
        # THE COST OF THIS PASS WAS ALL WASTE. It built a BVH over the WHOLE body -- 2.18M faces
        # -- ran overlap() against itself, and then threw away every pair further than 0.12 from
        # the anchor. And _xpairs() is called again at the end of each iteration, so that ran 17
        # times: ~25 minutes, essentially all of the graft's runtime, to examine the crotch.
        #
        # The faces beyond the cut-off can never contribute a kept pair, so they are excluded
        # from the tree instead of from the results. The set is chosen ONCE with a margin over
        # the 0.12 cut-off, which is safe because this pass only ever moves vertices by a
        # fraction of an edge (plain Laplacian at XSECT_W, on verts near the graft).
        print("[t] xsect: weights done, gathering local faces", flush=True)
        _XR = 0.12
        _ax2, _ay2, _az2 = float(A[0]), float(A[1]), float(A[2])
        # 1.25x the cut-off is 1.95x the volume of 1.05x, and this pass only moves vertices by
        # a fraction of an edge, so the smaller margin is still ample.
        _xr2 = (_XR * 1.05) ** 2
        _loc = []
        for _f in bm2.faces:
            _cc = _f.calc_center_median()
            _dx = _cc.x - _ax2; _dy = _cc.y - _ay2; _dz = _cc.z - _az2
            if _dx * _dx + _dy * _dy + _dz * _dz < _xr2:
                _loc.append(_f)
        _lface = [f.index for f in _loc]
        _lvert, _lmap = [], {}
        for _f in _loc:
            for _v in _f.verts:
                if _v.index not in _lmap:
                    _lmap[_v.index] = len(_lvert); _lvert.append(_v)
        _lpoly = [[_lmap[_v.index] for _v in _f.verts] for _f in _loc]
        print("[cutjoin] self-intersection: testing %d face(s) near the anchor, not the whole "
              "%d-face body" % (len(_loc), len(bm2.faces)), flush=True)

        def _xpairs():
            if not _lpoly:
                return []
            _bv = BVHTree.FromPolygons([list(v.co) for v in _lvert], _lpoly, all_triangles=False)
            _o = []
            for _a, _b in _bv.overlap(_bv):
                if _a >= _b:
                    continue
                _fa = bm2.faces[_lface[_a]]; _fb = bm2.faces[_lface[_b]]
                # faces that touch are not intersections
                if {_v.index for _v in _fa.verts} & {_v.index for _v in _fb.verts}:
                    continue
                _c = 0.5 * (np.array(_fa.calc_center_median()) + np.array(_fb.calc_center_median()))
                if np.linalg.norm(_c - A) > _XR:
                    continue
                _o.append((_fa, _fb))
            return _o
        print("[t] xsect: local faces gathered, building neighbours", flush=True)
        _nb2 = {}
        for _e in bm2.edges:
            _a = idx[_e.verts[0]]; _b = idx[_e.verts[1]]
            _nb2.setdefault(_a, set()).add(_b); _nb2.setdefault(_b, set()).add(_a)
        print("[t] xsect: neighbours built, first overlap", flush=True)
        _px = _xpairs(); _p0 = len(_px)
        # Bound how far this relax may MOVE a vertex, the same way the crease relax is bounded.
        # Untethered it relocates geometry rather than untangling it: traced, one vertex 19 mm
        # below the donor was pushed 7.5 mm and every face of the remaining spike fanned off it.
        # Budget is a multiple of the vertex's OWN edge length, so it scales with the mesh.
        _P20 = np.array([list(x.co) for x in allv])
        _led2 = np.zeros(len(allv))
        for _i2, _k2 in _nb2.items():
            if _k2:
                _led2[_i2] = float(np.mean(np.linalg.norm(_P20[list(_k2)] - _P20[_i2], axis=1)))
        _bud2 = XSECT_MAX * _led2
        _capped2 = 0
        for _it in range(XSECT_ITERS):
            if not _px:
                break
            _hit = set()
            for _fa, _fb in _px:
                for _f in (_fa, _fb):
                    for _v in _f.verts:
                        _i = idx[_v]
                        if _wt2[_i] > 0.0:
                            _hit.add(_i)
            if not _hit:
                break
            _P2 = np.array([list(x.co) for x in allv])
            _Q2 = _P2.copy()
            # DIRECTED separation was tried and is DISABLED: pushing each face off the other's
            # plane feeds back on itself (measured 705 -> 35954 intersections in 16 iterations).
            # Plain Laplacian on the intersecting verts is weak (705 -> 666) but never harmful.
            for _i in _hit:
                _k = _nb2.get(_i)
                if _k:
                    _w = XSECT_W * _wt2[_i]
                    _Q2[_i] = (1 - _w) * _P2[_i] + _w * _P2[list(_k)].mean(0)
            if XSECT_MAX > 0:
                _hl2 = np.fromiter(_hit, int, len(_hit))
                _dv3 = _Q2[_hl2] - _P20[_hl2]
                _dn3 = np.linalg.norm(_dv3, axis=1)
                _ov3 = (_dn3 > _bud2[_hl2]) & (_bud2[_hl2] > 0)
                if _ov3.any():
                    _capped2 = max(_capped2, int(_ov3.sum()))
                    _sc3 = np.ones(len(_hl2))
                    _sc3[_ov3] = _bud2[_hl2][_ov3] / np.maximum(_dn3[_ov3], 1e-12)
                    _Q2[_hl2] = _P20[_hl2] + _dv3 * _sc3[:, None]
            for _i in _hit:
                allv[_i].co = Vector(_Q2[_i].tolist())
            bm2.normal_update()
            _px = _xpairs()
        print(f"[cutjoin] self-intersection relax: {_p0} intersecting face pairs near the graft "
              f"-> {len(_px)} ({_it + 1 if _p0 else 0} iters, labeled detail exempt); drift cap "
              f"{XSECT_MAX:.1f}x the local edge held back up to {_capped2} vert(s)", flush=True)
    _stage_dump("7_xsect", bm2, A)
    # BRUSH SMOOTH (PELVIS_BRUSH_SMOOTH=1, default off). The user hand-smoothed the finished TRIM4
    # graft with Blender's sculpt Smooth brush (GRAFT_T5_smoothed.blend) and that is the target.
    # Measured on it: 5,309 verts moved, mostly SIDEWAYS (normal share of the motion 23%) -- the
    # brush evens out the bridge fans / stretched body strips, it does not flatten -- and 1,943
    # of the 3,590 body verts it moved sit across the ~2 mm leg gap: a brush reaches by STRAIGHT-
    # LINE distance, so the geodesic seam band above (which cannot cross the gap) never touched
    # them. This is that brush: uniform Laplacian over MESH neighbours (nothing is pulled across
    # the gap, only smoothed on its own surface), strength fading smoothly from the seam to zero
    # at one donor width, labelled labia/opening frozen within half GRAFT_PROTECT_R and feathered
    # to full strength at GRAFT_PROTECT_R. Donor verts are allowed to move (frozen donor measured
    # 69 body slivers vs 26). Offline on TRIM4 vs the user's T5: body slivers <5deg 26 (T5 25),
    # body edges >4x 17 (27), seam dihedrals >60deg 0 (5), self-intersections 497 (622),
    # labelled zones 1-4 moved 0.
    if os.environ.get("PELVIS_BRUSH_SMOOTH", "0") == "1":
        from mathutils import kdtree as _kdm
        _bit = int(os.environ.get("PELVIS_BRUSH_ITERS", "30"))
        bm2.verts.ensure_lookup_table(); bm2.verts.index_update()
        _Pb = np.array([list(x.co) for x in allv])
        _db = _dmask_g if (_dmask_g is not None and len(_dmask_g) == len(allv)) else None
        if _db is None:
            _atb = me.attributes.get("pg_donor")
            _tmpb = np.zeros(len(allv), np.int32)
            if _atb is not None and len(_atb.data) == len(allv):
                _atb.data.foreach_get("value", _tmpb)
            _db = _tmpb > 0
        _evb = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", _evb)
        _evb = _evb.reshape(-1, 2)
        _seamv = np.unique(_evb[_db[_evb[:, 0]] != _db[_evb[:, 1]]].ravel())
        # PELVIS_BRUSH_R_FRAC (default 1): the reach as a fraction of the donor's own width (relative, per donor)
        _Rb = float(os.environ.get('PELVIS_BRUSH_R', '0')) or float(np.ptp(np.asarray(W)[:, 0])) * float(os.environ.get('PELVIS_BRUSH_R_FRAC', '1'))
        if len(_seamv) and _Rb > 0:
            _kb = _kdm.KDTree(len(_seamv))
            for _k, _s in enumerate(_seamv.tolist()):
                _kb.insert(Vector(_Pb[_s].tolist()), _k)
            _kb.balance()
            _reachA = float(np.linalg.norm(_Pb[_seamv] - A, axis=1).max()) + _Rb
            _cand = np.nonzero(np.linalg.norm(_Pb - A, axis=1) < _reachA)[0]
            _wb = np.zeros(len(allv))
            # PELVIS_BRUSH_PROTECT (default 1) scales the freeze around the protected zones; 0 = none
            _prh = GRAFT_PROTECT_R * 0.5 * float(os.environ.get('PELVIS_BRUSH_PROTECT', '1'))
            _protb = 0
            for _i in _cand.tolist():
                if allv[_i].is_boundary:
                    continue
                _pv = Vector(_Pb[_i].tolist())
                _dsm = _kb.find(_pv)[2]
                if _dsm >= _Rb:
                    continue
                _sb = 1.0 - _dsm / _Rb
                _w = _sb * _sb * (3.0 - 2.0 * _sb)
                if _detail_kd is not None and _prh > 0:
                    _wp = float(np.clip((_detail_kd.find(_pv)[2] - _prh) / _prh, 0.0, 1.0))
                    if _wp <= 0.0:
                        _protb += 1
                    _w *= _wp
                if _w > 0.0:
                    _wb[_i] = _w
            # only verts that HAVE neighbours, and NO trailing dummy index: np.add.reduceat sums the
            # last segment to the end of the array, so a dummy there is added into the last vertex's
            # neighbour sum every iteration (measured: one vertex dragged 190 mm toward vertex 0).
            _act = np.array([_i for _i in np.nonzero(_wb > 0)[0].tolist() if allv[_i].link_edges], np.int64)
            _flb, _ofb = [], [0]
            for _i in _act.tolist():
                _flb.extend(idx[e.other_vert(allv[_i])] for e in allv[_i].link_edges)
                _ofb.append(len(_flb))
            _flb = np.asarray(_flb, np.int64)
            _cnb = np.diff(np.asarray(_ofb, np.int64))
            _sgb = np.asarray(_ofb[:-1], np.int64)
            _wab = _wb[_act][:, None]
            _Pn = _Pb.copy()
            for _ in range(_bit if len(_act) else 0):
                _mb = np.add.reduceat(_Pn[_flb], _sgb, axis=0) / _cnb[:, None]
                _Pn[_act] = _Pn[_act] + 0.5 * _wab * (_mb - _Pn[_act])
            for _i in _act.tolist():
                allv[_i].co = Vector(_Pn[_i].tolist())
            bm2.normal_update()
            _mvb = np.linalg.norm(_Pn[_act] - _Pb[_act], axis=1) / 1.0016 * 1700
            print(f"[cutjoin] brush smooth: reach {_Rb:.5f} ({_Rb / 1.0016 * 1700:.1f} mm, one donor width) "
                  f"straight-line from {len(_seamv)} seam vert(s); {len(_act)} vert(s) smoothed x{_bit}, "
                  f"{_protb} frozen on labelled detail; move med {np.median(_mvb):.2f} max {_mvb.max():.1f} mm",
                  flush=True)
        _stage_dump("8_brush", bm2, A)
    bm2.normal_update(); bm2.to_mesh(me); bm2.free()
    for p in me.polygons: p.use_smooth = True
    _trace("after xsect+seam smooth", bmb, A)
    print(f"[cutjoin] graft + seam-ring smooth: {len(me.vertices)} verts, ring {len(zone)} (inner detail protected)", flush=True)
    return ob, W
    # ===== DEAD (snap / shrinkwrap-morph / cut-bridge) =====
    from collections import defaultdict
    A = np.asarray(A, float); W = np.asarray(W, float); me = ob.data
    # (1) subdivide the genital region
    _bm = bmesh.new(); _bm.from_mesh(me); _bm.faces.ensure_lookup_table()
    _gf = [f for f in _bm.faces if np.linalg.norm(np.array(f.calc_center_median()) - A) < 0.055]
    _ge = list({e for f in _gf for e in f.edges})
    if _ge:
        bmesh.ops.subdivide_edges(_bm, edges=_ge, cuts=SNAP_SUBDIV, use_grid_fill=True)
        bmesh.ops.triangulate(_bm, faces=[f for f in _bm.faces if len(f.verts) > 4])
    _bm.to_mesh(me); _bm.free()
    # (2) controlled snap onto the donor (clamped)
    dme = bpy.data.meshes.new("don"); dme.from_pydata(W.tolist(), [], [list(map(int, f)) for f in Fd]); dme.update()
    don = bpy.data.objects.new("don", dme); bpy.context.scene.collection.objects.link(don)
    V2 = np.array([list(v.co) for v in me.vertices]); dA = np.linalg.norm(V2 - A, axis=1)
    for i in np.where(dA < 0.06)[0]:
        w = float(np.clip((MORPH_R - dA[i]) / MORPH_FALL, 0, 1))
        if w <= 0: continue
        p = Vector(V2[i].tolist()); ok, cp, nn, ix = don.closest_point_on_mesh(p)
        if ok and (cp - p).length < SNAP_CLAMP: me.vertices[i].co = p.lerp(cp, w)
    me.update(); bpy.data.objects.remove(don, do_unlink=True)
    # (3) outward-ring smooth; outward = donor mean normal (robust on any body), donor 2D footprint = in/out
    fn = np.zeros(3)
    for f in Fd:
        a, b, c = W[int(f[0])], W[int(f[1])], W[int(f[2])]; fn = fn + np.cross(b - a, c - a)
    outw = fn / (np.linalg.norm(fn) + 1e-12)
    if outw @ (A - V2.mean(0)) < 0: outw = -outw
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adjr = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adjr[a].append(b); adjr[b].append(a)
    seen = set(); loops = []
    for s in list(adjr):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; cur = s
        while True:
            nx = [ww for ww in adjr[cur] if ww != prev and ww not in seen]
            if not nx: break
            nn2 = nx[0]; lp.append(nn2); seen.add(nn2); prev = cur; cur = nn2
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); rim = loops[0]
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in rim])
    def _inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    allv = list(bm.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    zone = set()
    for i in range(len(allv)):
        if dA2[i] > 0.1: continue
        dd = co[i] - A; px = dd @ u; py = dd @ v
        d2r = float(np.min(np.sqrt(((poly - np.array([px, py])) ** 2).sum(1))))
        inside = _inpoly(px, py, poly)
        if (not inside and d2r < 0.016) or (inside and d2r < 0.010): zone.add(i)  # outward ring + tiny boundary band
    nb = {i: set() for i in zone}
    for e in bm.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in zone: nb[a].add(b)
        if b in zone: nb[b].add(a)
    P = co.copy()
    for _ in range(SNAP_SMOOTH_ITERS):
        nP = P.copy()
        for i in zone:
            if nb[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nb[i]], axis=0)
        P = nP
    for i in zone: allv[i].co = Vector(P[i].tolist())
    bm.normal_update(); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    print(f"[cutjoin] subdivide+snap graft: {len(me.vertices)} verts, ring {len(zone)} (interior protected)", flush=True)
    return ob, W
    # ===== DEAD (old shrinkwrap-morph + cut-bridge) =====
    A = np.asarray(A, float); W = np.asarray(W, float)
    dme = bpy.data.meshes.new("don"); dme.from_pydata(W.tolist(), [], [list(map(int, f)) for f in Fd]); dme.update()
    don = bpy.data.objects.new("don", dme); bpy.context.scene.collection.objects.link(don)
    me = ob.data
    V2 = np.array([list(v.co) for v in me.vertices]); dA = np.linalg.norm(V2 - A, axis=1)
    vg = ob.vertex_groups.new(name="gen")
    for i in range(len(V2)):
        w = float(np.clip((MORPH_R - dA[i]) / MORPH_FALL, 0, 1))   # 1 in the centre -> 0 (soft blend) at MORPH_R
        if w > 0: vg.add([i], w, 'REPLACE')
    sw = ob.modifiers.new("sw", 'SHRINKWRAP'); sw.target = don; sw.vertex_group = "gen"
    # PROJECT (along the body's own normals) NOT nearest-surface: on a coarse body NEAREST folds the surface
    # (adjacent verts snap to non-adjacent donor points) -> lumps/waves; PROJECT is a coherent per-vertex move.
    sw.wrap_method = 'PROJECT'; sw.use_negative_direction = True; sw.use_positive_direction = True
    sw.subsurf_levels = 0; sw.wrap_mode = 'ON_SURFACE'
    bpy.context.view_layer.objects.active = ob
    for x in bpy.context.scene.objects: x.select_set(x == ob)
    bpy.ops.object.modifier_apply(modifier="sw")
    bpy.data.objects.remove(don, do_unlink=True)
    # smooth ONLY the BODY side of the genital boundary ("from the outside") to blend the donor-edge crack,
    # leaving the genital interior (vulva detail) fully untouched. The donor 2D footprint separates in/out.
    from collections import defaultdict
    Vb = _np_verts(ob.data); _cen = Vb.mean(0); outw = A - _cen; outw = outw / (np.linalg.norm(outw) + 1e-12)
    _t = np.array([1.0, 0, 0])
    if abs(outw @ _t) > 0.9: _t = np.array([0, 1.0, 0])
    u = np.cross(outw, _t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adjr = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adjr[a].append(b); adjr[b].append(a)
    seen = set(); loops = []
    for s in list(adjr):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; curr = s
        while True:
            nx = [ww for ww in adjr[curr] if ww != prev and ww not in seen]
            if not nx: break
            nn = nx[0]; lp.append(nn); seen.add(nn); prev = curr; curr = nn
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); rim = loops[0]
    poly = np.array([[(W[li] - A) @ u, (W[li] - A) @ v] for li in rim])
    def _inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    allv = list(bm.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv]); dA2 = np.linalg.norm(co - A, axis=1)
    zone = set()
    for i in range(len(allv)):
        if dA2[i] > 0.085: continue
        d = co[i] - A
        if not _inpoly(d @ u, d @ v, poly): zone.add(i)   # OUTSIDE the genital footprint = body side -> smooth
    nb = {i: set() for i in zone}
    for e in bm.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in zone: nb[a].add(b)
        if b in zone: nb[b].add(a)
    P = co.copy()
    for _ in range(12):
        nP = P.copy()
        for i in zone:
            if nb[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nb[i]], axis=0)
        P = nP
    for i in zone: allv[i].co = Vector(P[i].tolist())
    bm.normal_update(); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    print("[cutjoin] shrinkwrap-morph graft (no bridge -> no shards)", flush=True)
    return ob, W
    # ---- DEAD CODE BELOW (cut+bridge -- shards) ----
    from collections import defaultdict, deque
    A = np.asarray(A, float)
    Vb = _np_verts(ob.data); cen = Vb.mean(0)
    outw = A - cen; outw = outw / (np.linalg.norm(outw) + 1e-12); OW = Vector(outw.tolist())
    t = np.array([1.0, 0, 0])
    if abs(outw @ t) > 0.9: t = np.array([0, 1.0, 0])
    u = np.cross(outw, t); u = u / np.linalg.norm(u); v = np.cross(outw, u)
    # donor OUTER rim = the longest boundary loop
    ec = defaultdict(int)
    for f in Fd:
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1: adj[a].append(b); adj[b].append(a)
    seen = set(); loops = []
    for s in list(adj):
        if s in seen: continue
        lp = [s]; seen.add(s); prev = None; curr = s
        while True:
            nx = [w for w in adj[curr] if w != prev and w not in seen]
            if not nx: break
            n = nx[0]; lp.append(n); seen.add(n); prev = curr; curr = n
        if len(lp) > 6: loops.append(lp)
    loops.sort(key=len, reverse=True); loop = loops[0]
    # snap the rim onto the body along -outward (near crotch surface, never the opposite thigh)
    W = np.asarray(W, float); Wp = W.copy()
    for li in loop:
        o = Vector(W[li].tolist()); hit, lp2, nr, ix = ob.ray_cast(o, -OW)
        if not hit or (Vector(lp2) - o).length > 0.08: hit, lp2, nr, ix = ob.ray_cast(o, OW)
        if hit and (Vector(lp2) - o).length < 0.08: Wp[li] = np.array(lp2.to_tuple())
        else:
            ok, cp, nn, ix = ob.closest_point_on_mesh(o)
            if ok: Wp[li] = np.array(cp.to_tuple())
    poly = np.array([[Wp[li] @ u, Wp[li] @ v] for li in loop])
    def inpoly(px, py, P):
        c = False; n = len(P); j = n - 1
        for i in range(n):
            xi, yi = P[i]; xj, yj = P[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi): c = not c
            j = i
        return c
    bmb = bmesh.new(); bmb.from_mesh(ob.data); bmb.faces.ensure_lookup_table()
    cand = set()
    for f in bmb.faces:
        c = np.array(f.calc_center_median())
        if np.linalg.norm(c - A) > 0.12: continue
        if abs(np.dot(c - A, outw)) > 0.045: continue
        if inpoly(c @ u, c @ v, poly): cand.add(f.index)
    if not cand:
        bmb.free(); print("[cutjoin] WARN: no body faces under donor footprint", flush=True); return ob, Wp
    seedf = min((f for f in bmb.faces if f.index in cand),
                key=lambda f: np.linalg.norm(np.array(f.calc_center_median()) - A))
    flood = set([seedf.index]); dq = deque([seedf])
    while dq:
        f = dq.popleft()
        for e in f.edges:
            for nf in e.link_faces:
                if nf.index in cand and nf.index not in flood: flood.add(nf.index); dq.append(nf)
    changed = True
    while changed:
        changed = False
        for f in bmb.faces:
            if f.index in flood or np.linalg.norm(np.array(f.calc_center_median()) - A) > 0.12: continue
            nbf = [lf.index for e in f.edges for lf in e.link_faces if lf.index != f.index]
            if nbf and all(x in flood for x in nbf): flood.add(f.index); changed = True
    hole_bverts = set()
    for fi in flood:
        for e in bmb.faces[fi].edges:
            if any(lf.index not in flood for lf in e.link_faces):
                hole_bverts.add(e.verts[0]); hole_bverts.add(e.verts[1])
    bmesh.ops.delete(bmb, geom=[bmb.faces[fi] for fi in flood], context='FACES')
    # RELAX the jagged cut boundary into a smooth, evenly-spaced loop BEFORE bridging -> the bridge to the
    # clean donor rim is regular -> kills the long shard triangles at the source. Tangential Laplacian along
    # the boundary loop (move toward midpoint of the two boundary neighbours), kept on the body surface.
    bmb.edges.ensure_lookup_table()
    he0 = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hole_bverts and e.verts[1] in hole_bverts]
    hnb = defaultdict(list)
    for e in he0: hnb[e.verts[0]].append(e.verts[1]); hnb[e.verts[1]].append(e.verts[0])
    hbv = [hv for hv in hole_bverts if hv.is_valid and len(hnb[hv]) == 2]   # clean loop verts only (skip branches)
    for _ in range(12):
        newp = {}
        for hv in hbv:
            mid = 0.5 * (np.array(hnb[hv][0].co) + np.array(hnb[hv][1].co))
            newp[hv] = 0.5 * np.array(hv.co) + 0.5 * mid
        for hv, p in newp.items():
            ok, cp, nn, ix = ob.closest_point_on_mesh(Vector(p.tolist()))
            hv.co = cp if ok else Vector(p.tolist())         # keep the relaxed vert ON the body surface
    dvm = [bmb.verts.new(Wp[i].tolist()) for i in range(len(Wp))]; bmb.verts.ensure_lookup_table()
    for f in Fd:
        try: bmb.faces.new([dvm[int(f[0])], dvm[int(f[1])], dvm[int(f[2])]])
        except Exception: pass
    bmb.edges.ensure_lookup_table()
    # bridge the body hole loop <-> donor rim (watertight base; its shard triangles get evened by the
    # body-side seam smooth below). Branch-aware hole-loop walk + fraction stitch.
    he = [e for e in bmb.edges if len(e.link_faces) == 1 and e.verts[0] in hole_bverts and e.verts[1] in hole_bverts]
    hadj = defaultdict(list)
    for e in he: hadj[e.verts[0]].append(e.verts[1]); hadj[e.verts[1]].append(e.verts[0])
    hstart = he[0].verts[0]; H = [hstart]; prev = None; curr = hstart
    while True:
        nx = [w for w in hadj[curr] if w is not prev]
        if not nx: break
        if len(nx) == 1: n = nx[0]
        else:
            cc = np.array(curr.co); din = (cc - np.array(prev.co)) if prev is not None else np.array([1.0, 0, 0])
            din = din / (np.linalg.norm(din) + 1e-9)
            n = min(nx, key=lambda w: -float(np.dot(din, (np.array(w.co) - cc) / (np.linalg.norm(np.array(w.co) - cc) + 1e-9))))
        if n is hstart or n in H: break
        H.append(n); prev = curr; curr = n
    R = [dvm[li] for li in loop]
    def sarea(L):
        p = np.array([[(np.array(x.co) - A) @ u, (np.array(x.co) - A) @ v] for x in L])
        return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    if sarea(R) * sarea(H) < 0: H = H[::-1]
    def ang(x):
        p = np.array(x.co) - A; return float(np.arctan2(p @ v, p @ u))
    a0 = ang(R[0]); j0 = int(np.argmin([abs(((ang(h) - a0 + np.pi) % (2 * np.pi)) - np.pi) for h in H])); H = H[j0:] + H[:j0]
    nR = len(R); nH = len(H); i = j = 0
    for _ in range(nR + nH):
        fi = (i + 1) / nR; fj = (j + 1) / nH
        if i < nR and (j >= nH or fi <= fj):
            try: bmb.faces.new([R[i % nR], R[(i + 1) % nR], H[j % nH]])
            except Exception: pass
            i += 1
        else:
            try: bmb.faces.new([H[j % nH], R[i % nR], H[(j + 1) % nH]])
            except Exception: pass
            j += 1
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    rim_co = np.array([Wp[li] for li in loop]); rim_v = [dvm[li] for li in loop]; rimset = set(rim_v)
    bmb.edges.ensure_lookup_table()
    openv = set()
    for e in bmb.edges:
        if len(e.link_faces) == 1:
            for vv in e.verts:
                if vv.is_valid and vv not in rimset and np.linalg.norm(np.array(vv.co) - A) < 0.12: openv.add(vv)
    tmap = {}
    for vv in openv:
        p = np.array(vv.co); k = int(np.argmin(((rim_co - p) ** 2).sum(1)))
        if np.linalg.norm(rim_co[k] - p) < 0.02: tmap[vv] = rim_v[k]
    if tmap: bmesh.ops.weld_verts(bmb, targetmap=tmap)
    bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    # flip long sliver bridge edges to well-shaped triangles (Delaunay-like) -> removes most shard faces,
    # without moving the surface (edge flips preserve the shape; the genital tessellation just gets cleaner).
    bmb.faces.ensure_lookup_table()
    sf = [f for f in bmb.faces if len(f.verts) == 3 and np.linalg.norm(np.array(f.calc_center_median()) - A) < 0.09]
    if sf:
        se = list({e for f in sf for e in f.edges})
        bmesh.ops.beautify_fill(bmb, faces=sf, edges=se)
        bmesh.ops.recalc_face_normals(bmb, faces=list(bmb.faces))
    bmb.to_mesh(ob.data); bmb.free(); ob.data.update()
    # BODY-SIDE seam smooth: even out the bridge shards, but ONLY from the rim OUTWARD (~2.5cm onto the
    # body). The genital interior (anything inside the rim radius) and the far surround are PROTECTED, so the
    # vulva/anus detail is never touched. Per-angle rim radius decides inside(genital) vs outside(body).
    bm3 = bmesh.new(); bm3.from_mesh(ob.data); bm3.verts.ensure_lookup_table()
    allv = list(bm3.verts); idx = {x: i for i, x in enumerate(allv)}
    co = np.array([list(x.co) for x in allv])
    rimpts = np.array([Wp[li] for li in loop])
    rim2d = np.stack([(rimpts - A) @ u, (rimpts - A) @ v], 1)
    rim_ang = np.arctan2(rim2d[:, 1], rim2d[:, 0]); rim_rad = np.linalg.norm(rim2d, axis=1)
    near = np.where(np.linalg.norm(co - A, axis=1) < 0.13)[0]
    seam = set()
    for i in near:
        d = co[i] - A; p2 = np.array([d @ u, d @ v]); r = float(np.linalg.norm(p2))
        th = float(np.arctan2(p2[1], p2[0]))
        Rth = rim_rad[int(np.argmin(np.abs(((rim_ang - th + np.pi) % (2 * np.pi)) - np.pi)))]
        if (r > Rth - 0.004) and (r < Rth + 0.030):     # rim + BODY side only; genital interior protected
            seam.add(int(i))
    nbrs = {i: set() for i in seam}
    for e in bm3.edges:
        a = idx[e.verts[0]]; b = idx[e.verts[1]]
        if a in seam: nbrs[a].add(b)
        if b in seam: nbrs[b].add(a)
    P = co.copy()
    for _ in range(18):
        nP = P.copy()
        for i in seam:
            if nbrs[i]: nP[i] = 0.5 * P[i] + 0.5 * np.mean([P[k] for k in nbrs[i]], axis=0)
        P = nP
    for i in seam: allv[i].co = Vector(P[i].tolist())
    bm3.normal_update(); bm3.to_mesh(ob.data); bm3.free(); ob.data.update()
    print(f"[cutjoin] body-side seam smooth on {len(seam)} verts", flush=True)
    # final: close last micro-gaps at the old branch spot
    bmz = bmesh.new(); bmz.from_mesh(ob.data); bmz.edges.ensure_lookup_table()
    rez = [e for e in bmz.edges if len(e.link_faces) == 1 and np.linalg.norm(np.array((e.verts[0].co + e.verts[1].co) / 2) - A) < 0.12]
    if rez:
        bmesh.ops.holes_fill(bmz, edges=rez, sides=6)
        bmesh.ops.triangulate(bmz, faces=[f for f in bmz.faces if len(f.verts) > 4])
        bmesh.ops.recalc_face_normals(bmz, faces=list(bmz.faces))
    bmz.to_mesh(ob.data); bmz.free(); ob.data.update()
    print(f"[cutjoin] grafted: cut {len(flood)} body faces, rim {len(loop)} verts", flush=True)
    return ob, Wp


def graft_pelvis(Vt, Ft, voxel_body=True, local=False):
    """Vt,Ft = body mesh. Returns (V,F) of the clean grafted genital body.

    local=True (PIPELINE): keep the body at its OWN density (~150k), graft the dense donor with ONLY the
    HC seam-flatten, and SKIP the whole-body fuse + GPU sculpt. The fuse would re-voxel the entire body to
    ~1.2M (violates 'detail only in the genital, lean body'); the body is already the smooth gen_seams
    remesh so it needs no surround sculpt. Result stays ~body+donor faces (<200k)."""
    Vt = np.asarray(Vt, float)   # Ft kept as-is (may be a list of QUADS/mixed faces -> don't triangulate the body)
    # 0) body -> smooth remesh (skip if caller already provides the gen_seams remesh)
    ob, _ = _new_obj(Vt, Ft, "body")
    if voxel_body:
        _voxel(ob, BODY_VOX)
    Vb = _np_verts(ob.data); Fb = [list(p.vertices) for p in ob.data.polygons]
    A = np.array([0.0, 0.0, 0.0])

    # 1) place donor
    W, Fd, A = _place(Vt, Ft)
    if GRADE_CUTS > 0:
        _grade_ring(ob, A, GRADE_R1, GRADE_R2, GRADE_CUTS)
    # ===== CUT-AND-JOIN graft (2026-06-24): replaces the old conform+cut+bridge+HC+fuse+GPU-sculpt below.
    # Headless, no surround smoothing -> the body outside the genital is untouched; vulva/anus detail kept.
    # Everything from here to `return V, F, o` supersedes the (now-dead) block beneath this return. =====
    A = np.asarray(A, float)
    # narrow the donor in X (width) about the midline -> rim sits inboard of the thigh junction (the concave
    # spot where the seam fights) -> cleaner seam, off the thighs. Tune DONOR_X_SCALE.
    W = np.asarray(W, float).copy(); W[:, 0] *= DONOR_X_SCALE
    if COLLAR > 0:
        Fd = _collar_trim(W, Fd, COLLAR)     # drop the apron that spans the notch (see COLLAR)
    if NOTCH_TRIM:
        Fd = _notch_trim(ob, W, Fd)          # ...and any skin left hanging over the leg gap
        if RIM_RELAX > 0:
            W = _rim_relax(W, Fd, RIM_RELAX, RIM_RELAX_RINGS)   # straighten the edge it leaves
    W_placed = W.copy()                      # matches genital_donor_zones.npy one-to-one
    o, W = _graft_cutjoin(ob, W, Fd, A)
    me = o.data
    for p in me.polygons: p.use_smooth = True
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects: x.select_set(x == o)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    if not local:
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False); bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    V = _np_verts(me).astype(np.float32); F = [list(p.vertices) for p in me.polygons]
    _dd3 = os.environ.get("PELVIS_DUMP_DONORVERTS", "")
    if _dd3:
        # Read the donor tag HERE, where V is actually taken. Dumping it earlier (inside
        # _graft_cutjoin) misaligns it with the exported vertex order and makes unrelated
        # vertices look like donor verts flung 200 mm down the thigh.
        _m3 = np.zeros(len(me.vertices), bool)
        _at3 = me.attributes.get("pg_donor")
        if _at3 is not None:
            print(f"[pelvis_graft] donor tag attribute: domain={_at3.domain} "
                  f"type={_at3.data_type} len={len(_at3.data)} vs {len(me.vertices)} verts",
                  flush=True)
            _v3 = np.zeros(len(_at3.data), np.int32)
            _at3.data.foreach_get("value", _v3)
            if len(_v3) == len(me.vertices):
                _m3 = _v3 > 0
            else:
                print("[pelvis_graft] donor tag is NOT per-vertex -- cannot use it", flush=True)
        np.save(_dd3, _m3)
        print(f"[pelvis_graft] donor-vertex map: {int(_m3.sum())} of {len(_m3)} -> {_dd3}",
              flush=True)
    if local:
        print("[t] flag: materialising face centres", flush=True)
        cents = np.array([list(p.center) for p in me.polygons], np.float64)
        print("[t] flag: centres done, classifying", flush=True)
        isgen = _genital_flag(cents, W, Fd, W_placed)
        try:
            _gp = os.path.normpath(os.path.join(HERE, "..", "..", "uv_transfer", "_gs_isgenital.npy"))
            np.save(_gp, isgen)
            print(f"[pelvis_graft] genital flag: {int(isgen.sum())}/{len(isgen)} faces -> {_gp}", flush=True)
        except Exception as _e:
            print("[pelvis_graft] genital flag save failed:", _e, flush=True)
    _side = os.environ.get("PELVIS_DONOR_SIDECAR", "")
    if _side:
        # For the RETOPO low poly's genital colours (ComfyUI-Retopo/scripts/genital_colours.py): where the
        # donor ended up and what each donor vertex is. W keeps the donor's vertex order -- the order the
        # zone labels index -- and carries the rim snap; fall back to the placed copy if that ever changes.
        try:
            _Wf = np.asarray(W, float)
            if _Wf.shape != np.asarray(W_placed).shape:
                _Wf = np.asarray(W_placed, float)
            _zn = np.load(A_ZONES)
            _awp = os.environ.get("PELVIS_ANUS_W", "")
            _aw = np.load(_awp) if _awp and os.path.exists(_awp) else np.zeros(len(_Wf), np.float32)
            # FINAL positions (2026-09-16): with the brush smoothing the outer lips, W was up to 11 mm off the mesh
            # (outer edge median 1.7 mm) and the low poly's zone lookup painted a jagged fringe. Take each donor
            # vertex's position in the final mesh through _graft_cutjoin's index map -- but only if the map lines
            # up: the protected detail (never moved) must sit within half a donor edge of W, else keep W.
            # W stays the join-time positions (vestibule_clear.py was tuned and approved on them); W_final = where
            # the donor verts really are, for the colours.
            _Wfin = _Wf
            _mi = globals().get('_DONOR_MESH_IDX')
            if _mi is not None and len(_mi) == len(_Wf):
                _Vn = np.asarray(V, float)
                _ok = (_mi >= 0) & (_mi < len(_Vn))
                _cand = _Wf.copy(); _cand[_ok] = _Vn[_mi[_ok]]
                _Fdn = np.asarray(Fd, np.int64)
                _edge = float(np.median(np.linalg.norm(_Wf[_Fdn[:, 0]] - _Wf[_Fdn[:, 1]], axis=1)))
                _prot = _ok & np.isin(np.asarray(_zn), [3, 4]) if len(_zn) == len(_Wf) else _ok
                _dev = np.linalg.norm(_cand[_prot] - _Wf[_prot], axis=1)
                if len(_dev) and float(np.median(_dev)) < 0.5 * _edge:
                    _mv = np.linalg.norm(_cand - _Wf, axis=1) / 1.0016 * 1700
                    print(f"[pelvis_graft] donor sidecar: FINAL mesh positions for {int(_ok.sum())} of {len(_Wf)} donor verts "
                          f"(moved after the join: med {np.median(_mv):.2f} max {_mv.max():.2f} mm; protected detail off by "
                          f"max {_dev.max() / 1.0016 * 1700:.3f} mm)", flush=True)
                    _Wfin = _cand
                else:
                    print("[pelvis_graft] donor sidecar: index map does not line up with the protected detail -- "
                          "keeping the join-time positions", flush=True)
            np.savez(_side, W=_Wf.astype(np.float32), W_final=np.asarray(_Wfin, np.float32), F=np.asarray(Fd, np.int32), zones=_zn, anus_w=_aw)
            print(f"[pelvis_graft] donor sidecar: {len(_Wf)} verts, {len(_zn)} zone labels, "
                  f"{int((np.asarray(_aw) > 0).sum())} anus-labelled -> {_side}", flush=True)
        except Exception as _e:
            print("[pelvis_graft] donor sidecar save failed:", _e, flush=True)
    return V, F, o
    # ----- DEAD CODE BELOW (old pg14 cut+bridge+GPU-sculpt; kept for reference, never executed) -----
    anchor = Vector((0.0, float(A[1]), float(A[2])))
    if local and GRADE_RATIO < 1.0:
        # GRADE: decimate the donor's RIM toward the body's density (protect the detailed center) so the
        # donor->body seam has no density jump -> no vertex-normal "border". Headless.
        og, _ = _new_obj(W, Fd, "grade")
        _dAg = np.linalg.norm(W - np.array([0.0, A[1], A[2]]), axis=1)
        _vgg = og.vertex_groups.new(name="g")
        for _i in range(len(W)):
            _gw = float(np.clip((_dAg[_i] - GRADE_KEEP) / GRADE_RAMP, 0, 1))   # 0 center -> 1 rim
            _vgg.add([_i], _gw, 'REPLACE')
        _dm = og.modifiers.new("g", 'DECIMATE'); _dm.decimate_type = 'COLLAPSE'
        _dm.ratio = GRADE_RATIO; _dm.vertex_group = "g"; _dm.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = og
        for x in bpy.context.scene.objects:
            x.select_set(x == og)
        bpy.ops.object.modifier_apply(modifier="g")
        W = _np_verts(og.data); Fd = np.array([list(p.vertices) for p in og.data.polygons], np.int64)
        bpy.data.objects.remove(og, do_unlink=True)
        print(f"[graft] graded donor rim -> {len(W)}v/{len(Fd)}f", flush=True)
    # anus anchor = centroid of the placed donor's BACK midline strip (the perineum/anus feature). Used
    # in step 6 to protect the anus from the seam smooth (the depth-stretch slid it into the smooth zone).
    _mid = W[np.abs(W[:, 0]) < 0.02]
    _back = _mid[_mid[:, 2] >= np.percentile(_mid[:, 2], 72)]
    anus_anchor = np.array([0.0, float(_back[:, 1].mean()), float(_back[:, 2].mean())])
    ec = defaultdict(int)
    for t in Fd:
        for a, b in [(int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))]:
            ec[(min(a, b), max(a, b))] += 1
    bidx = sorted({v for (a, b), c in ec.items() if c == 1 for v in (a, b)})
    bco = W[bidx]
    dmin = np.sqrt(((W[:, None, :] - bco[None, :, :]) ** 2).sum(-1).min(1))
    # scale the rim-flush band to the DONOR size: a fixed FLUSH_HI over-reaches on a smaller donor and
    # shrinkwraps the LABIA (not just the outer margin) onto the recessed crotch -> crumple. 0.132 = original
    # donor X-width; _dsc<1 for a tighter cut shrinks the band proportionally so only the rim conforms.
    _dsc = min(1.0, float(np.ptp(np.load(A_DONOR)["V"][:, 0]) / 0.132))
    wgt = np.clip((FLUSH_HI * _dsc - dmin) / (FLUSH_W * _dsc), 0, 1)
    # A-fix: also pull the stretched back perineum onto the body (follow the curve), keeping vulva + anus
    backw = np.clip((W[:, 2] - A[2] - BACK_Z0) / BACK_W, 0, 1)           # 0 at vulva -> 1 toward the back
    backw *= np.clip(np.linalg.norm(W - anus_anchor, axis=1) / BACK_ANUS_KEEP, 0, 1)   # keep the anus dimple
    wgt = np.clip(np.maximum(wgt, backw * BACK_STRENGTH), 0, 1)
    # DIRECTIONAL conform (replaces the NEAREST_SURFACEPOINT shrinkwrap): for each weighted donor vert, ray-cast
    # toward the body interior (both ways) and seat it on the NEAREST hit. NEAREST_SURFACEPOINT grabbed the
    # OPPOSITE thigh in the concave between-legs gap and folded the edge into WALLS; a directional cast hits the
    # surface the vert actually sits over, so a TIGHT cut seats flat with no walls + needs no skin margin.
    _bcen = Vector(Vt.mean(0).tolist()); _mwi = ob.matrix_world.inverted(); _m3 = _mwi.to_3x3()
    Wp = W.copy()
    for _i in np.where(wgt > 0.01)[0]:
        _o = Vector(W[_i].tolist()); _dd0 = _bcen - _o
        if _dd0.length < 1e-6:
            continue
        _dd0.normalize(); _ol = _mwi @ _o; _dl = (_m3 @ _dd0).normalized()
        _best = None; _bd = 1e9
        for _s in (1.0, -1.0):
            _hit, _loc, _nrm, _idx = ob.ray_cast(_ol, _dl * _s)
            if _hit:
                _wl = ob.matrix_world @ _loc; _dist = (_wl - _o).length
                if _dist < _bd:
                    _bd = _dist; _best = _wl
        if _best is not None and _bd < 0.05:
            Wp[_i] = np.array((_o.lerp(_best, float(wgt[_i]))).to_tuple())
    od, _ = _new_obj(Wp, Fd, "don")
    Wp = _np_verts(od.data); rim_pts = Wp[bidx]
    bpy.data.objects.remove(od, do_unlink=True); bpy.data.objects.remove(ob, do_unlink=True)

    # 2) cut the body hole just beyond the donor rim (no overlap)
    gcy = Wp[:, 1].mean(); ax, ay = np.ptp(Wp[:, 0]) / 2, np.ptp(Wp[:, 1]) / 2
    cen = np.array([Vb[fl].mean(0) for fl in Fb])
    rm = ((cen[:, 0] / ax) ** 2 + ((cen[:, 1] - gcy) / ay) ** 2 < CUT ** 2) \
        & (cen[:, 2] > Wp[:, 2].min() - 0.02) & (cen[:, 2] < Wp[:, 2].max() + 0.02)
    Fb_keep = [fl for fl, k in zip(Fb, (~rm).tolist()) if k]

    # 3) join + bridge donor rim to body hole
    nb = len(Vb)
    Vall = np.vstack([Vb, Wp])
    Fall = Fb_keep + [[int(t[0] + nb), int(t[1] + nb), int(t[2] + nb)] for t in Fd]
    o, me = _new_obj(Vall, Fall, "m")
    bpy.ops.object.mode_set(mode='EDIT'); bm = bmesh.from_edit_mesh(me); bm.edges.ensure_lookup_table()
    for e in bm.edges:
        e.select = (len(e.link_faces) == 1 and ((e.verts[0].co + e.verts[1].co) * 0.5 - anchor).length < 0.17)
    bmesh.update_edit_mesh(me)
    bpy.ops.mesh.bridge_edge_loops(number_cuts=BRIDGE_CUTS)
    # beautify the seam band (kill slivers)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(me); bm.faces.ensure_lookup_table()
    kd = kdtree.KDTree(len(rim_pts))
    for i, p in enumerate(rim_pts):
        kd.insert(Vector(p.tolist()), i)
    kd.balance()
    for fc in bm.faces:
        _, _, d = kd.find(fc.calc_center_median())
        if d is not None and d < BEAUTY_R + 0.008:
            fc.select = True
    bmesh.update_edit_mesh(me)
    bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.mesh.beautify_fill(angle_limit=3.14159)
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode='OBJECT')

    if os.environ.get("GRAFT_BRIDGE_ONLY"):   # DEBUG: stop right after the bridge to inspect the attachment
        return _np_verts(me), [list(p.vertices) for p in me.polygons], o   # (skips smooth + crash-prone GPU sculpt)

    # 4) HC-Laplacian wide smooth (flatten seam, no moat; protect central vulva)
    V2 = _np_verts(me); F2 = [list(p.vertices) for p in me.polygons]
    Anp = np.array([0.0, A[1], A[2]])
    dAv = np.linalg.norm(V2 - Anp, axis=1); Xv = np.abs(V2[:, 0])
    protect = np.clip((HC_PX - Xv) / HC_PX, 0, 1) * np.clip((HC_PR - dAv) / 0.020, 0, 1)
    edge = np.clip((HC_REG - dAv) / 0.030, 0, 1)
    sm = np.clip((1.0 - protect) * edge, 0, 1); sm[dAv >= HC_REG] = 0.0
    adj = defaultdict(list); seen = defaultdict(set)
    for fl in F2:
        if dAv[list(fl)].min() < HC_REG + 0.01:
            n = len(fl)
            for k in range(n):
                a, b = fl[k], fl[(k + 1) % n]
                if b not in seen[a]: adj[a].append(b); seen[a].add(b)
                if a not in seen[b]: adj[b].append(a); seen[b].add(a)
    active = [int(i) for i in np.where(sm > 0.01)[0]]
    O = V2.copy(); P = V2.copy()
    for _ in range(HC_ITERS):
        Q = P.copy(); Lap = P.copy()
        for i in active:
            nn = adj[i]
            if nn: Lap[i] = np.mean([P[j] for j in nn], axis=0)
        B = np.zeros_like(P)
        for i in active:
            B[i] = Lap[i] - (HC_ALPHA * O[i] + (1 - HC_ALPHA) * Q[i])
        for i in active:
            nn = adj[i]; bn = np.mean([B[j] for j in nn], axis=0) if nn else B[i]
            P[i] = Q[i] + sm[i] * ((Lap[i] - (HC_BETA * B[i] + (1 - HC_BETA) * bn)) - Q[i])
    for i in active:
        me.vertices[i].co = Vector(P[i].tolist())
    me.update()

    if not local:
        # 5) FUSE -> uniform density (eliminates the density border). LOCAL skips it (keeps body lean).
        _voxel(o, FUSE_VOX)
        me = o.data

    # 6) surround smooth = Blender's GPU sculpt mesh_filter (RELAX + SMOOTH) -- the PROVEN smooth (pg14).
    #    Runs for BOTH now: non-local on the fused mesh; LOCAL on the lean non-fused mesh (smooths the
    #    mons/perineum/inner-thigh surround on the body's OWN density). SMOOTH is volume-preserving so it
    #    does NOT crater; the headless numpy Taubin/RELAX welted + striated when pushed, so the surround
    #    smooth stays on the GPU sculpt. Needs a GPU/VIEW_3D context -> WINDOWED Blender (caller wraps it
    #    in a VIEW_3D temp_override).
    me.update()
    V3 = _np_verts(me)
    dA3 = np.linalg.norm(V3 - Anp, axis=1)
    anus_p = np.clip((ANUS_R - np.linalg.norm(V3 - anus_anchor, axis=1)) / ANUS_FALL, 0, 1)  # keep anus dimple
    if local:
        # SEAM-ONLY: smooth a THIN band at the donor->body seam (distance from the donor rim), protecting
        # EVERYTHING else -- the body's detail (buttock cleft, mons) AND the genital interior. Only the
        # graft transition line gets blended. dist-from-rim only for verts near the genital (rest is far).
        _rp = np.asarray(rim_pts, float)
        dmin_rim = np.full(len(V3), 1e9)
        _near = dA3 < 0.14
        if _near.any():
            _Vn = V3[_near]
            dmin_rim[_near] = np.sqrt(((_Vn[:, None, :] - _rp[None, :, :]) ** 2).sum(-1).min(1))
        # back side is -Z in the graft frame (buttocks; +Z is the vulva/front). Use a NARROWER band there so
        # the smooth never reaches up into the gluteal cleft/buttocks (no flattened cleft, no creases).
        back = np.clip((Anp[2] - V3[:, 2]) / 0.03, 0, 1)
        band_eff = np.clip(SEAM_BAND - SEAM_BACK_NARROW * back, 0.004, None)
        fall_eff = SEAM_BAND_FALL - (SEAM_BAND_FALL - SEAM_BACK_FALL) * back   # back: tight falloff (no feather out)
        # SIDES: widen band + falloff toward the flanks (|X| away from the midline). Independent
        # of the back narrowing above, so the buttocks stay protected while the groin blends out.
        if SEAM_SIDE_WIDEN > 0.0:
            _side = np.clip((np.abs(V3[:, 0] - Anp[0]) - 0.004) / SEAM_SIDE_X, 0, 1) * (1.0 - back)
            band_eff = band_eff + SEAM_SIDE_WIDEN * _side
            fall_eff = fall_eff + SEAM_SIDE_WIDEN * _side
        seam_w = np.clip((band_eff - dmin_rim) / fall_eff, 0, 1)          # 1 on the seam line -> 0 away
        # belly-cut: smooth LESS only on the FRONT belly (+Y above the anchor AND +Z front) -- it must NOT
        # touch the back, where +Y is the buttock seam arc that needs full blending.
        above = np.clip((V3[:, 1] - Anp[1]) / 0.05, 0, 1)
        front = np.clip((V3[:, 2] - Anp[2]) / 0.015, 0, 1)        # +Z = front (vulva side)
        seam_w *= (1.0 - SEAM_BELLY_CUT * above * front)
        # PROTECT the gluteal cleft: midline strip (|X|<CLEFT_W) on the BACK (-Z), but only the UPPER cleft
        # (+Y, above the anchor) -> the natural groove is kept while the LOWER perineum centre still blends.
        cleft_p = (np.clip((CLEFT_W - np.abs(V3[:, 0])) / CLEFT_W, 0, 1)
                   * np.clip((Anp[2] - V3[:, 2]) / 0.015, 0, 1)
                   * np.clip((V3[:, 1] - Anp[1] - 0.005) / 0.02, 0, 1))
        seam_w *= (1.0 - cleft_p)
        mask = 1.0 - seam_w * (1.0 - anus_p)                             # 1 = protect (everything but the line)
    else:
        # standalone: wide surround smooth (matches B40 on the featureless coarse body)
        inner = np.clip((SEAM_INNER - dA3) / INNER_FALL, 0, 1); far = np.clip((dA3 - SEAM_FAR) / FAR_FALL, 0, 1)
        mask = np.clip(1.0 - (1.0 - inner) * (1.0 - far) * (1.0 - anus_p), 0, 1)
    if '.sculpt_mask' not in me.attributes:
        me.attributes.new('.sculpt_mask', 'FLOAT', 'POINT')
    ma = me.attributes['.sculpt_mask']
    for i in range(len(me.vertices)):
        ma.data[i].value = float(mask[i])
    me.update()
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.scene.objects:
        x.select_set(x == o)
    bpy.ops.object.mode_set(mode='SCULPT')
    for _ in range(RELAX_ITERS):
        bpy.ops.sculpt.mesh_filter(type='RELAX', strength=1.0)
    for _ in range(GRAFT_SEAM_ITERS):
        bpy.ops.sculpt.mesh_filter(type='SMOOTH', strength=1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    # (the headless numpy Taubin/RELAX surround polish was removed -- it welted/striated when pushed;
    #  the surround smooth is the GPU sculpt above, for local too.)

    # 7) final light GLOBAL smooth -> remove residual voxel-surface roughness over the whole
    #    fused region uniformly (broad form survives; fine micro-detail = normal-map bake)
    if FINAL_SMOOTH_ITERS > 0:
        bpy.context.view_layer.objects.active = o
        for x in bpy.context.scene.objects:
            x.select_set(x == o)
        fs = o.modifiers.new("fsmooth", 'SMOOTH'); fs.factor = FINAL_SMOOTH_FAC; fs.iterations = FINAL_SMOOTH_ITERS
        bpy.ops.object.modifier_apply(modifier="fsmooth")
        me = o.data
    for p in me.polygons:
        p.use_smooth = True

    # round-trip cleanly. local=True KEEPS the body's quads (only the donor region is tris) so the body
    # stays lean -- do NOT triangulate the whole mesh (that doubles the 150k body). Standalone -> all tris.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if not local:
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    me = o.data
    V = _np_verts(me).astype(np.float32)
    F = [list(p.vertices) for p in me.polygons]   # list (may be mixed quad/tri in local mode)
    if local:
        # save a per-FACE genital flag (centroid within GENITAL_R of the vulva anchor) aligned to THIS
        # mesh's face order, for gen_seams to cut the genital into its own UV island + (later) material.
        # Computed here in the Y-up graft frame (the in_blend rotation that follows preserves face order).
        cents = np.array([list(p.center) for p in me.polygons], np.float64)
        # TIGHT: hug the placed donor surface (the actual genital geometry) + a thin border, NOT a big sphere
        # -> the UV island + slot-2 material cover only the genitals; surrounding skin stays part of the body.
        _gkd = kdtree.KDTree(len(W))
        for _i in range(len(W)):
            _gkd.insert(Vector(W[_i].tolist()), _i)
        _gkd.balance()
        isgen = np.array([_gkd.find(Vector(c.tolist()))[2] < GENITAL_HUG for c in cents])
        try:
            _gp = os.path.normpath(os.path.join(HERE, "..", "..", "uv_transfer", "_gs_isgenital.npy"))
            np.save(_gp, isgen)
            print(f"[pelvis_graft] genital flag: {int(isgen.sum())}/{len(isgen)} faces -> {_gp}", flush=True)
        except Exception as _e:
            print("[pelvis_graft] genital flag save failed:", _e, flush=True)
    return V, F, o


def _argval(flag, default=None):
    a = sys.argv
    if "--" in a:
        a = a[a.index("--") + 1:]
    return a[a.index(flag) + 1] if flag in a else default


def _view3d():
    # GPU sculpt needs a VIEW_3D/region context -> run WINDOWED Blender (not --background). DON'T
    # read_homefile(use_empty=True) -- that wipes the screen. Returns (window, VIEW_3D area, region).
    win = bpy.context.window
    area = next((a for a in (win.screen.areas if (win and win.screen) else []) if a.type == 'VIEW_3D'), None)
    region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
    if area is None:
        raise RuntimeError("no VIEW_3D area -- run WINDOWED Blender (not --background), GPU sculpt needs it")
    return win, area, region


if __name__ == "__main__":
    import traceback
    from mathutils import Matrix
    from math import radians
    in_blend = _argval("--in_blend")   # PIPELINE mode: graft the genital into an existing (remeshed) blend
    body = _argval("--body"); out = _argval("--out"); blend = _argval("--blend")
    vox_body = "--voxel_body" in (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv)
    # --local: use the PIPELINE recipe (lean cut-join, no whole-body fuse, no GPU sculpt) from the
    # CLI too, so the graft can be replayed headlessly on a saved body npz without the 12-min bake.
    use_local = "--local" in (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv)
    try:
        if in_blend:
            # PIPELINE: graft the high-detail genital onto the ALREADY-REMESHED body in `in_blend`
            # (gen_seams' last_seams.blend), in place, BEFORE gen_seams UVs it -> the genital keeps its
            # full fuse detail (it is never re-remeshed). We APPEND the body mesh into the windowed STARTUP
            # scene (NOT open_mainfile -- that drops the VIEW_3D the GPU sculpt needs), graft, then save it
            # back as the SAME object name ('geometry_0') with identity matrix so gen_seams still finds it.
            for ob in list(bpy.data.objects):          # clear startup default cube/camera/light
                bpy.data.objects.remove(ob, do_unlink=True)
            with bpy.data.libraries.load(in_blend) as (_df, _dt):
                _dt.objects = list(_df.objects)
            src = None
            for _ob in _dt.objects:
                if _ob is None:
                    continue
                bpy.context.scene.collection.objects.link(_ob)
                if _ob.type == 'MESH' and (src is None or len(_ob.data.polygons) > len(src.data.polygons)):
                    src = _ob
            if src is None:
                raise RuntimeError("no mesh object in " + in_blend)
            name = src.name
            mw = src.matrix_world
            R = np.array([[mw[r][c] for c in range(3)] for r in range(3)], float)
            T = np.array([mw[r][3] for r in range(3)], float)
            # Extract world V + faces AS-IS (KEEP the body's quads -- triangulating here would double the
            # ~150k body before we even graft). graft_pelvis(local=True) keeps the body quads.
            nv = len(src.data.vertices); _co = np.empty(nv * 3, np.float64)
            src.data.vertices.foreach_get('co', _co)
            Vw = _co.reshape(nv, 3) @ R.T + T
            Fw = [list(p.vertices) for p in src.data.polygons]
            # The pipeline body comes out of the OBJ import Z-UP, but graft_pelvis/_place are tuned for the
            # Y-UP trimesh frame (handgraft_out). If Z is the height axis, rotate -90deg about X into Y-up
            # for the graft, then rotate the RESULT back +90deg about X before saving (gen_seams' frame).
            _zup = float(np.ptp(Vw[:, 2])) > float(np.ptp(Vw[:, 1]))
            if _zup:
                Vw = Vw @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # -90 about X: (x,y,z)->(x,z,-y)
            # cut-and-join graft is fully HEADLESS (no GPU sculpt) -> no VIEW_3D / windowed Blender needed.
            V, F, o = graft_pelvis(Vw, Fw, voxel_body=False, local=True)
            print(f"[pelvis_graft] done (pipeline): {len(V)} verts, {len(F)} faces", flush=True)
            for x in list(bpy.data.objects):           # keep ONLY the grafted result, as 'geometry_0'
                if x is not o:
                    bpy.data.objects.remove(x, do_unlink=True)
            for _lib in list(bpy.data.libraries):      # drop the appended-from library -> can overwrite the file
                bpy.data.libraries.remove(_lib)
            if _zup:                                    # rotate the grafted result back to the pipeline's Z-up frame
                o.data.transform(Matrix.Rotation(radians(90), 4, 'X'))   # +90 about X: (x,y,z)->(x,-z,y)
            o.name = name; o.matrix_world = Matrix.Identity(4)
            bpy.ops.wm.save_as_mainfile(filepath=in_blend)
            # GENITAL TEXTURING (approach A) runs as a SEPARATE HEADLESS launch (genital_texture.py) after this
            # -- it doesn't need the GPU, and the windowed graft session is too unstable for its unwrap.
        else:
            for ob in list(bpy.data.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            d = np.load(body)
            V, F, o = graft_pelvis(d["V"], d["F"], voxel_body=vox_body, local=use_local)
            print(f"[pelvis_graft] done: {len(V)} verts, {len(F)} faces", flush=True)
            if out:
                # graft_pelvis returns the body's ORIGINAL face list, which is quads+tris mixed;
                # np.asarray cannot make that rectangular. Fan-triangulate for the npz so the CLI
                # replay can be inspected. (The pipeline path saves a .blend and keeps the quads.)
                try:
                    _F = np.asarray(F, np.int64)
                except ValueError:
                    _tri = []
                    for _f in F:
                        for _i in range(1, len(_f) - 1):
                            _tri.append([_f[0], _f[_i], _f[_i + 1]])
                    _F = np.asarray(_tri, np.int64)
                    print(f"[pelvis_graft] npz: fan-triangulated {len(F)} polys -> {len(_F)} tris", flush=True)
                np.savez(out, V=np.asarray(V, np.float32), F=_F)
            if blend:
                bpy.ops.wm.save_as_mainfile(filepath=blend)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)   # quit the windowed Blender so the subprocess returns (no lingering window/prompt)
