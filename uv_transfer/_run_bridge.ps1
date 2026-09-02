$uv = "C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
$bl = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$py = "C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
Remove-Item "$uv\_gs_faceregion.npy" -ErrorAction SilentlyContinue
Remove-Item "$uv\_gs_facemesh.npz" -ErrorAction SilentlyContinue
Copy-Item "$uv\last_seams.obj" "$uv\_gs_stable.obj" -Force   # snapshot (user may be baking concurrently)
$env:GS_OBJ_IN = "$uv\_gs_stable.obj"
Copy-Item "$uv\_genseams_viz.npz" "$uv\_genseams_viz.npz.bakrun" -Force -ErrorAction SilentlyContinue
Copy-Item "$uv\last_seams_hairfaces.npy" "$uv\last_seams_hairfaces.npy.bakrun" -Force -ErrorAction SilentlyContinue
Write-Output "--- pass A (export mesh) ---"
$env:GS_FACE_EXPORT = "1"
& $bl --background --python "$uv\_gs_blender.py" 2>$null | Select-String "pass A:"
$env:GS_FACE_EXPORT = $null
Write-Output "--- bridge (MediaPipe) ---"
& $py "$uv\_gs_facebridge.py" 2>$null | Select-String "compute:", "FaceRegion", "FAILED"
Write-Output "--- pass B (read + unwrap) ---"
& $bl --background --python "$uv\_gs_blender.py" 2>$null | Select-String "face-chart", "UV fill", "exported"
Copy-Item "$uv\_genseams_viz.npz.bakrun" "$uv\_genseams_viz.npz" -Force -ErrorAction SilentlyContinue
Copy-Item "$uv\last_seams_hairfaces.npy.bakrun" "$uv\last_seams_hairfaces.npy" -Force -ErrorAction SilentlyContinue
Write-Output "--- measure ---"
& $py "$uv\_measure_face_uv.py" "$uv\_gs_test_out.obj"
$env:OBJ_PATH = "$uv\_gs_test_out.obj"
& $py "$uv\_uv_layout.py" 2>$null | Select-String "UV islands"
