"""ComfyUI node: replace a generated character's webbed hands with clean donor hands BEFORE the remesh.
Skeleton-free (geometric wrist-finding). Drops the donor hands overlapping the wrist stumps; the downstream
Trellis2Remesh dual-contouring fuses them into one watertight mesh, after which UV/texture/rig treat the
hands as native body. Operates on the MeshWithVoxel's vertex/face arrays only (duck-typed, no Trellis import)."""
import os
import numpy as np
from . import hand_replace as hr

HERE = os.path.dirname(os.path.abspath(__file__))
_DONOR_R = os.path.join(HERE, "assets", "donor_geo_R.npz")
_DONOR_L = os.path.join(HERE, "assets", "donor_geo_L.npz")


class ReplaceHandsWithDonor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"trimesh": ("TRIMESH",)},
            "optional": {
                "seam": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "body_faces": ("INT", {"default": 135000, "min": 10000, "max": 5000000, "step": 5000}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "HandGraft"

    def process(self, trimesh, seam=0.5, body_faces=135000):
        # Runs AFTER ToTrimesh, BEFORE SmartUV. Decimates the BODY to `body_faces` (cumesh), then grafts the
        # FULL-RES donor hands on top -> the donor fingers are never decimated. The result is below SmartUV's
        # target_faces, so SmartUV does only seams/UV (no re-decimation) and textures the hands + writes the obj.
        import gc, trimesh as TM, numpy as _np
        Vt = _np.asarray(trimesh.vertices, _np.float64); Ft = _np.asarray(trimesh.faces, _np.int64)
        DBG = os.path.normpath(os.path.join(HERE, "..", "..", "output"))
        try:
            _np.savez(os.path.join(DBG, "handgraft_in.npz"), V=Vt.astype(_np.float32), F=Ft.astype(_np.int64))
        except Exception as e:
            print("[ReplaceHandsWithDonor] dump-in failed:", e, flush=True)
        nf0 = len(Ft)
        if body_faces and nf0 > body_faces:
            import torch, cumesh as CuMesh
            v = torch.tensor(Vt, dtype=torch.float32, device='cuda'); f = torch.tensor(Ft, dtype=torch.int32, device='cuda')
            cm = CuMesh.CuMesh(); cm.init(v, f); cm.simplify(int(body_faces), verbose=False)
            vv, ff = cm.read(); del cm; gc.collect()
            Vt = vv.detach().cpu().numpy().astype(_np.float64); Ft = ff.detach().cpu().numpy().astype(_np.int64)
        V2, F2, info = hr.replace_hands(Vt, Ft, np.load(_DONOR_R), np.load(_DONOR_L), seam=seam)
        print(f"==== [ReplaceHandsWithDonor] RAN ====  body {nf0}->{len(Ft)} faces (decimated), "
              f"arm-axis={info['arm']} wrists={info['wxR']:.3f}/{info['wxL']:.3f}  grafted -> {len(V2)} verts {len(F2)} faces",
              flush=True)
        try:
            _np.savez(os.path.join(DBG, "handgraft_out.npz"), V=V2, F=F2)
        except Exception as e:
            print("[ReplaceHandsWithDonor] dump-out failed:", e, flush=True)
        return (TM.Trimesh(vertices=V2.astype(_np.float64), faces=F2.astype(_np.int64), process=False),)


BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
_PELVIS_GRAFT = os.path.join(HERE, "pelvis_graft.py")


class ReplacePelvisWithDonor:
    """POST-remesh: graft the high-res donor genital onto the SMOOTH remeshed body.

    Runs the validated B40 recipe (build -> fine voxel-fuse -> seam smooth) in a headless
    Blender subprocess (it needs Blender ops: voxel remesh, bridge, beautify). Input is the
    already-remeshed body (gen_seams output) -> leave voxel_body off. Output is a clean,
    border-free, uniform-density mesh; the genital's fine micro-detail comes from a separate
    normal-map bake. See pelvis_graft.py / project memory for the full saga (B1-B40)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"trimesh": ("TRIMESH",)},
            "optional": {
                "genital_scale": ("FLOAT", {"default": 1.10, "min": 0.8, "max": 1.5, "step": 0.05}),
                "voxel_body": ("BOOLEAN", {"default": False,
                               "tooltip": "ON if the input body is NOT yet remeshed (voxels it first). "
                                          "OFF when fed the gen_seams remesh."}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "HandGraft"

    def process(self, trimesh, genital_scale=1.10, voxel_body=False):
        import subprocess, tempfile, trimesh as TM, numpy as _np
        Vt = _np.asarray(trimesh.vertices, _np.float32); Ft = _np.asarray(trimesh.faces, _np.int64)
        td = tempfile.mkdtemp(prefix="pelvisgraft_")
        fin = os.path.join(td, "body.npz"); fout = os.path.join(td, "graft.npz")
        _np.savez(fin, V=Vt, F=Ft)
        env = dict(os.environ, PELVIS_GS=str(genital_scale))
        # WINDOWED on purpose (no --background): step 6 uses Blender's GPU sculpt mesh_filter, which
        # crashes in background and has no headless equal. A Blender window flashes, runs, force-quits.
        cmd = [BLENDER_EXE, "--python", _PELVIS_GRAFT, "--", "--body", fin, "--out", fout]
        if voxel_body:
            cmd.append("--voxel_body")
        # launch the Blender window MINIMIZED (SW_SHOWMINNOACTIVE=7) AND block it from stealing
        # foreground focus by raising the system foreground-lock timeout for the duration (Windows
        # then makes Blender blink in the taskbar instead of popping to front). Restored after.
        si = None; _u = None; _flock_old = None
        if os.name == "nt":
            import ctypes
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 7
            try:
                _u = ctypes.windll.user32
                _buf = ctypes.c_uint()
                _u.SystemParametersInfoW(0x2000, 0, ctypes.byref(_buf), 0)            # SPI_GETFOREGROUNDLOCKTIMEOUT
                _flock_old = _buf.value
                _u.SystemParametersInfoW(0x2001, 0, ctypes.c_void_p(300000), 0)       # SPI_SET... -> 5 min lock
            except Exception:
                _u = None
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, startupinfo=si)
        finally:
            if _u is not None and _flock_old is not None:
                try: _u.SystemParametersInfoW(0x2001, 0, ctypes.c_void_p(int(_flock_old)), 0x02)  # restore + broadcast
                except Exception: pass
        if not os.path.exists(fout):
            print("[ReplacePelvisWithDonor] FAILED — Blender output:\n", r.stdout[-2000:], r.stderr[-1000:], flush=True)
            return (trimesh,)  # passthrough on failure (don't break the graph)
        d = _np.load(fout)
        V2, F2 = d["V"].astype(_np.float64), d["F"].astype(_np.int64)
        print(f"==== [ReplacePelvisWithDonor] RAN ====  body {len(Ft)} -> grafted {len(F2)} faces "
              f"(genital_scale={genital_scale})", flush=True)
        return (TM.Trimesh(vertices=V2, faces=F2, process=False),)


NODE_CLASS_MAPPINGS = {"ReplaceHandsWithDonor": ReplaceHandsWithDonor,
                       "ReplacePelvisWithDonor": ReplacePelvisWithDonor}
NODE_DISPLAY_NAME_MAPPINGS = {"ReplaceHandsWithDonor": "Replace Hands With Donor (skeleton-free)",
                              "ReplacePelvisWithDonor": "Replace Pelvis With Donor (genital graft)"}
