# charB hair-fix regions (from user's marked screenshots, 2026-06-12)

Gold A = `_goldA_hairfaces.npy` (elf, 24730 faces, frozen). Every change: run on elf,
diff vs gold, **0 hair->skin allowed**, and additions must NOT land on A's face/body.

RED = missing hair (ADD).  GREEN = bleed (REMOVE).

| # | region | mark | coords (gen-norm) | charB now | guard risk |
|---|--------|------|-------------------|-----------|------------|
| 1 | Temples (both, in front of ear) | RED add | \|x\| .07-.13, z .36-.46, fy<.02 | 137/826 hair (gap) | additive = safe |
| 2 | Shoulder/neck drape FRONT edge | RED add | \|x\| .05-.14, z .18-.32, fy<-.005 | 0 hair (gap) | additive = safe |
| 3 | Center forehead | GREEN rm | \|x\|<.05, z .42-.47, fy<-.01 | 1675/4533 hair (bleed) | **conflicts w/ A's hairline (same version-A region)** |
| 4 | Cheek / jaw (below ear, side view) | GREEN rm | TBD (first box \|x\|.04-.10,z.30-.38 found 0) | re-locate | likely conflicts w/ A framing strand |

Coordinate space: z=height (crown~0.48, eyes~0.41, jaw~0.30), x=lateral (face<0.07,
ear>0.10), y=depth (front<0).
