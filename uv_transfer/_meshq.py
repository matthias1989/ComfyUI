"""LOCAL: component-size distribution of the shattered bake's mesh — is the body one
intact piece + floaters, or fragmented into many big pieces?"""
import numpy as np, trimesh, os
p = os.path.join(os.path.dirname(__file__), "last_seams.obj")
m = trimesh.load(p, force='mesh')
parts = m.split(only_watertight=False)
sizes = sorted((len(p.faces) for p in parts), reverse=True)
n = len(m.faces)
print(f"total faces={n}  components={len(parts)}")
print(f"top 15 component face-counts: {sizes[:15]}")
big = [s for s in sizes if s > 500]
print(f"components >500 faces: {len(big)}  holding {100*sum(big)/n:.1f}% of faces")
tiny = [s for s in sizes if s <= 50]
print(f"components <=50 faces (floaters): {len(tiny)}  holding {100*sum(tiny)/n:.1f}% of faces")
print(f"largest single component holds {100*sizes[0]/n:.1f}% of faces")
