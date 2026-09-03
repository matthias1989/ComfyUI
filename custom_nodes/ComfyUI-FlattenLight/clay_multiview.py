"""Render an UNTEXTURED mesh from several azimuths in ONE Blender session.

Importing a 300k-face OBJ costs ~25 s, so rendering three views as three separate Blender
launches paid that three times. This imports once and only moves the camera. Normalisation and
the orthographic camera are taken from the Trellis2 pack's own blender_render.py, so the frames
match the texturing stage exactly.
"""
import os, sys, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "ComfyUI-Trellis2", "projection")))
import bpy
import blender_render as BR


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--azims", required=True, help="comma separated")
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.15)
    ap.add_argument("--norm_size", type=float, default=1.15)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--shading", default="normal", choices=["normal", "clay"])
    a = ap.parse_args(argv)

    BR.reset_scene()
    obj = BR.import_mesh(a.mesh)
    if obj:
        BR.auto_center_and_scale(obj, a.norm_size)
    BR.setup_lighting()

    if a.shading == "normal" and obj is not None:
        # Lit clay renders the body as a flat white silhouette -- Canny then finds only the
        # outline. Emitting the surface NORMAL as colour gives strong internal contours
        # (spine, shoulder blades, buttocks, hair clumps), which is what the edge reference
        # is for. Lighting-independent, so sample count is irrelevant.
        mat = bpy.data.materials.new("NormalViz")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out_n = nt.nodes.new("ShaderNodeOutputMaterial")
        emis = nt.nodes.new("ShaderNodeEmission")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(emis.inputs["Color"], geo.outputs["Normal"])
        nt.links.new(out_n.inputs["Surface"], emis.outputs["Emission"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    os.makedirs(a.outdir, exist_ok=True)
    for az in [float(x) for x in a.azims.split(",") if x.strip()]:
        for c in [o for o in bpy.data.objects if o.type == 'CAMERA']:
            bpy.data.objects.remove(c, do_unlink=True)
        BR.set_camera(a.elev, az, 1.45, a.scale)
        out = os.path.join(a.outdir, f"view_{int(round(az))}.png")
        BR.render_scene(out, a.resolution, "GPU", "EEVEE", a.samples)
        print(f"[clay] az={az} -> {out}", flush=True)


if __name__ == "__main__":
    main()
