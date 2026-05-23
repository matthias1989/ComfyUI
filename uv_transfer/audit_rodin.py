import bpy, bmesh

bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.wm.obj_import(filepath=r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\Rodin_template.obj')

obj = bpy.context.selected_objects[0]
mesh = obj.data

print(f'[AUDIT] Verts: {len(mesh.vertices)}')
print(f'[AUDIT] Edges: {len(mesh.edges)}')
print(f'[AUDIT] Faces: {len(mesh.polygons)}')
print(f'[AUDIT] has_custom_normals: {mesh.has_custom_normals}')

bm = bmesh.new()
bm.from_mesh(mesh)
bm.edges.ensure_lookup_table()

sharp_edges = [e for e in bm.edges if not e.smooth]
seam_edges  = [e for e in bm.edges if e.seam]
boundary    = [e for e in bm.edges if len(e.link_faces) < 2]

print(f'[AUDIT] Sharp edges (edge.smooth=False): {len(sharp_edges)}')
print(f'[AUDIT] Seam edges (edge.seam=True): {len(seam_edges)}')
print(f'[AUDIT] Boundary edges: {len(boundary)}')

# UV seam detection via UV discontinuity
uv_layer = bm.loops.layers.uv.active
uv_seam_count = 0
if uv_layer:
    for edge in bm.edges:
        if len(edge.link_faces) == 2:
            fa, fb = edge.link_faces
            vset = {edge.verts[0].index, edge.verts[1].index}
            uv_a = {lp.vert.index: lp[uv_layer].uv.to_tuple(5) for lp in fa.loops if lp.vert.index in vset}
            uv_b = {lp.vert.index: lp[uv_layer].uv.to_tuple(5) for lp in fb.loops if lp.vert.index in vset}
            for vi in vset:
                if vi in uv_a and vi in uv_b and uv_a[vi] != uv_b[vi]:
                    uv_seam_count += 1
                    break
    print(f'[AUDIT] UV seam edges (UV discontinuity): {uv_seam_count}')
else:
    print('[AUDIT] No UV layer found')

bm.free()
