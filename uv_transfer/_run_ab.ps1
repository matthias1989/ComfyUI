$uv = "C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
$bl = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$py = "C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
Copy-Item "$uv\last_seams.obj" "$uv\_gs_stable.obj" -Force
$env:GS_OBJ_IN = "$uv\_gs_stable.obj"
Remove-Item "$uv\_gs_faceregion.npy" -ErrorAction SilentlyContinue
Copy-Item "$uv\_genseams_viz.npz" "$uv\_genseams_viz.npz.bakrun" -Force -ErrorAction SilentlyContinue
Copy-Item "$uv\last_seams_hairfaces.npy" "$uv\last_seams_hairfaces.npy.bakrun" -Force -ErrorAction SilentlyContinue
Write-Output "=== A: NO face-chart (pre-my-code body) ==="
$env:GS_NO_FACECHART = "1"
& $bl --background --python "$uv\_gs_blender.py" 2>$null | Select-String "face-chart", "UV fill", "exported"
$env:GS_NO_FACECHART = $null
$env:OBJ_PATH = "$uv\_gs_test_out.obj"
& $py "$uv\_uv_layout.py" 2>$null | Select-String "UV islands"
Write-Output "=== B: WITH face-chart (geometric) ==="
& $bl --background --python "$uv\_gs_blender.py" 2>$null | Select-String "face-chart", "UV fill", "exported"
& $py "$uv\_uv_layout.py" 2>$null | Select-String "UV islands"
Copy-Item "$uv\_genseams_viz.npz.bakrun" "$uv\_genseams_viz.npz" -Force -ErrorAction SilentlyContinue
Copy-Item "$uv\last_seams_hairfaces.npy.bakrun" "$uv\last_seams_hairfaces.npy" -Force -ErrorAction SilentlyContinue
