"""DEV: mark the path3 hairline loop onto last_seams_DEV.blend so it can be opened + inspected.
Clears existing seams, marks ONLY the path3 hairline (no body/ear yet) to judge the hairline alone.
Run: blender last_seams_DEV.blend --background --python _dev_mark.py"""
import bpy, os, importlib.util as ilu
import numpy as np
UV = os.path.dirname(os.path.abspath(__file__))
obj = bpy.data.objects.get('geometry_0') or next((o for o in bpy.data.objects if o.type == 'MESH'), None)
me = obj.data
for e in me.edges:
    e.use_seam = False
eidx = {}
for e in me.edges:
    eidx[(min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1]))] = e.index
hp = os.path.join(UV, '_hairline_pipe.py')
spec = ilu.spec_from_file_location('_hairline_pipe', hp); hpm = ilu.module_from_spec(spec); spec.loader.exec_module(hpm)
hair_seam = hpm.mark_loop_seam(me, eidx, UV)
print(f"[dev] hairline loop marked: {len(hair_seam)} edges")
# dump for plotting
co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(len(me.vertices), 3)
mat = obj.matrix_world
R = np.array([[mat[r][c] for c in range(3)] for r in range(3)]); T = np.array([mat[r][3] for r in range(3)])
co = co @ R.T + T
se = [(e.vertices[0], e.vertices[1]) for e in me.edges if e.use_seam]
np.savez(os.path.join(UV, '_dev_seam.npz'), co=co.astype('float32'), seam=np.array(se, dtype=np.int64))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(UV, 'last_seams_DEV.blend'))
print(f"[dev] saved last_seams_DEV.blend with {len(se)} seam edges")
