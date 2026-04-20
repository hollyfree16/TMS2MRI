"""
blender_tms_surface.py
======================
Standalone Blender script: load fsaverage pial surfaces and plot TMS
stimulation coordinates as spheres on the surface, coloured by hemisphere.

Run from the command line:
    blender --background --python blender_tms_surface.py -- \\
        --fsaverage-csv path/to/targets_fsaverage.csv \\
        --output-dir ./viz

    # Multiple subjects / group:
    blender --background --python blender_tms_surface.py -- \\
        --fsaverage-csv sub01/targets_fsaverage.csv \\
        --fsaverage-csv sub02/targets_fsaverage.csv \\
        --output-dir ./group_viz \\
        --output-prefix group_M1

    # Save .blend for interactive exploration (no render):
    blender --background --python blender_tms_surface.py -- \\
        --fsaverage-csv targets_fsaverage.csv \\
        --output-dir ./viz \\
        --blend-only

    # Open interactively (Blender GUI):
    blender --python blender_tms_surface.py -- \\
        --fsaverage-csv targets_fsaverage.csv \\
        --output-dir ./viz \\
        --no-render

Requirements
------------
- Blender >= 3.0  (tested on 3.6 and 4.x)
- nilearn must be importable from Blender's Python environment, OR
  pass --nilearn-python /path/to/python to use your system Python to
  export the surface meshes first (see --help).

Outputs
-------
  <output-prefix>_surface.blend   — Blender file (open to explore interactively)
  <output-prefix>_surface.png     — rendered PNG (unless --blend-only / --no-render)
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import tempfile

# ============================================================
# Parse args  (Blender passes everything after "--" to the script)
# ============================================================

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(
        description="Plot TMS stimulation sites on fsaverage pial surface in Blender."
    )
    p.add_argument(
        "--fsaverage-csv",
        action="append",
        dest="fsaverage_csvs",
        required=True,
        metavar="PATH",
        help="Path to a targets_fsaverage.csv (from stage 06). Repeat for multiple subjects.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write outputs into.",
    )
    p.add_argument(
        "--output-prefix",
        default="tms_surface",
        help="Filename prefix for outputs (default: tms_surface).",
    )
    p.add_argument(
        "--fsaverage-mesh",
        default="fsaverage",
        choices=["fsaverage", "fsaverage5", "fsaverage6"],
        help="fsaverage mesh resolution (default: fsaverage). "
             "Must match the mesh used in stage 06.",
    )
    p.add_argument(
        "--nilearn-python",
        default=None,
        metavar="PATH",
        help="Path to a Python interpreter that has nilearn installed, "
             "if Blender's bundled Python does not. "
             "Used to export surface meshes to /tmp before importing.",
    )
    p.add_argument(
        "--sphere-radius",
        type=float,
        default=1.5,
        help="Radius of stimulation point spheres in mm (default: 1.5).",
    )
    p.add_argument(
        "--blend-only",
        action="store_true",
        help="Save .blend file only — skip PNG render.",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Skip render entirely (useful when running in Blender GUI).",
    )
    p.add_argument(
        "--brain-color",
        default="0.75,0.72,0.68",
        help="Brain surface colour as R,G,B floats 0-1 (default: 0.75,0.72,0.68).",
    )
    p.add_argument(
        "--color-left",
        default="0.27,0.53,1.0",
        help="Left hemisphere point colour R,G,B (default: blue 0.27,0.53,1.0).",
    )
    p.add_argument(
        "--color-right",
        default="1.0,0.27,0.27",
        help="Right hemisphere point colour R,G,B (default: red 1.0,0.27,0.27).",
    )
    return p.parse_args(argv)


# ============================================================
# Surface mesh export (runs in system Python if needed)
# ============================================================

EXPORT_SCRIPT = """
import sys, json
mesh = sys.argv[1]
out  = sys.argv[2]

from nilearn import datasets
import nibabel as nib
import numpy as np

fs = datasets.fetch_surf_fsaverage(mesh)

def load_mesh(path):
    path = str(path)
    # Try nibabel FreeSurfer format first
    try:
        coords, faces = nib.freesurfer.read_geometry(path)
        return coords, faces
    except Exception:
        pass
    # Try GIFTI
    try:
        img = nib.load(path)
        arrays = img.darrays
        coords = arrays[0].data
        faces  = arrays[1].data
        return coords, faces
    except Exception:
        pass
    # Try nilearn as last resort
    try:
        from nilearn import surface as surf
        coords, faces = surf.load_surf_mesh(path)
        return coords, faces
    except Exception as e:
        raise RuntimeError(f"Could not load surface {path}: {e}")

result = {}
for key in ['pial_left', 'pial_right']:
    coords, faces = load_mesh(fs[key])
    result[key] = {
        'coords': coords.tolist(),
        'faces':  faces.tolist(),
    }

with open(out, 'w') as f:
    json.dump(result, f)
print('exported', out)
"""


def _export_surfaces_via_subprocess(python_bin: str, mesh: str) -> dict:
    """Use an external Python to export fsaverage meshes to a temp JSON."""
    import json
    tmp = tempfile.mktemp(suffix=".json")
    script = tempfile.mktemp(suffix=".py")
    with open(script, "w") as f:
        f.write(EXPORT_SCRIPT)
    result = subprocess.run(
        [python_bin, script, mesh, tmp],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ERROR exporting surfaces:")
        print(result.stderr)
        sys.exit(1)
    with open(tmp) as f:
        data = json.load(f)
    os.unlink(tmp)
    os.unlink(script)
    return data


def _load_surfaces_direct(mesh: str) -> dict:
    """Load fsaverage meshes directly (nilearn available in Blender Python)."""
    from nilearn import datasets, surface as surf
    fs = datasets.fetch_surf_fsaverage(mesh)
    result = {}
    for key in ['pial_left', 'pial_right']:
        coords, faces = surf.load_surf_mesh(fs[key])
        result[key] = {'coords': coords.tolist(), 'faces': faces.tolist()}
    return result


def _get_surfaces(args: argparse.Namespace) -> dict:
    if args.nilearn_python:
        print(f"[blender_tms] Using external Python: {args.nilearn_python}")
        return _export_surfaces_via_subprocess(args.nilearn_python, args.fsaverage_mesh)
    try:
        return _load_surfaces_direct(args.fsaverage_mesh)
    except ImportError:
        print(
            "[blender_tms] ERROR: nilearn not found in Blender's Python.\n"
            "  Option 1: pip install nilearn into Blender's Python:\n"
            "    <blender>/python/bin/pip install nilearn\n"
            "  Option 2: pass --nilearn-python /path/to/your/python"
        )
        sys.exit(1)


# ============================================================
# Load stimulation coordinates
# ============================================================

def _load_coords(csv_paths: list[str]) -> list[dict]:
    """
    Read fs_x/fs_y/fs_z + hemisphere from one or more targets_fsaverage.csv files.
    Returns list of dicts with keys: x, y, z, hemi
    """
    points = []
    for path in csv_paths:
        with open(path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                row = dict(zip(header, line.strip().split(",")))
                try:
                    x = float(row["fs_x"])
                    y = float(row["fs_y"])
                    z = float(row["fs_z"])
                except (KeyError, ValueError):
                    continue
                # hemisphere: use ef_hemisphere if present, else derive from fs_surface
                hemi = row.get("ef_hemisphere", "")
                if not hemi:
                    surf_name = row.get("fs_surface", "")
                    hemi = "L" if "left" in surf_name else "R"
                points.append({"x": x, "y": y, "z": z, "hemi": hemi.strip()})
    print(f"[blender_tms] Loaded {len(points)} stimulation points from {len(csv_paths)} CSV(s)")
    return points


# ============================================================
# Blender scene construction
# ============================================================

def _parse_color(s: str) -> tuple[float, float, float]:
    r, g, b = [float(v) for v in s.split(",")]
    return (r, g, b)


def _build_scene(surfaces: dict, points: list[dict], args: argparse.Namespace) -> None:
    import bpy
    import mathutils

    brain_color  = _parse_color(args.brain_color)
    color_left   = _parse_color(args.color_left)
    color_right  = _parse_color(args.color_right)

    # ---- Clear scene ----
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.render.resolution_x = 2400
    scene.render.resolution_y = 1200
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)   # dark background
    bg.inputs[1].default_value = 1.0

    # ---- Brain material ----
    brain_mat = bpy.data.materials.new("Brain")
    brain_mat.use_nodes = True
    bsdf = brain_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value  = (*brain_color, 1.0)
    bsdf.inputs["Roughness"].default_value   = 0.85
    bsdf.inputs["Specular IOR Level"].default_value = 0.05
    brain_mat.diffuse_color = (*brain_color, 0.9)

    # ---- Point materials ----
    def _make_mat(name, color, emit=0.4):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value  = (*color, 1.0)
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
        bsdf.inputs["Roughness"].default_value = 0.2
        return mat

    mat_left  = _make_mat("Left",    color_left)
    mat_right = _make_mat("Right",   color_right)
    mat_mid   = _make_mat("Midline", (0.9, 0.9, 0.2))   # yellow for near-midline

    # ---- Import brain meshes ----
    for hemi_key, obj_name in [("pial_left", "Brain_LH"), ("pial_right", "Brain_RH")]:
        if hemi_key not in surfaces:
            continue
        data   = surfaces[hemi_key]
        coords = data["coords"]
        faces  = data["faces"]

        mesh = bpy.data.meshes.new(obj_name)
        verts = [(c[0], c[1], c[2]) for c in coords]

        # Blender needs faces as tuples
        bl_faces = [tuple(f) for f in faces]

        mesh.from_pydata(verts, [], bl_faces)
        mesh.update()

        obj = bpy.data.objects.new(obj_name, mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(brain_mat)

        # Smooth shading
        for poly in obj.data.polygons:
            poly.use_smooth = True

        # Add edge split modifier for cleaner look
        mod = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        mod.split_angle = 0.5

        print(f"[blender_tms] Imported {obj_name}: {len(verts)} verts, {len(bl_faces)} faces")

    # ---- Add stimulation point spheres ----
    # Use instancing for speed: one base sphere per colour, then instance
    def _base_sphere(name, mat, radius):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, segments=16, ring_count=12,
            location=(0, 0, -9999)   # hidden off-screen
        )
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        return obj

    sphere_l = _base_sphere("_sphere_L", mat_left,  args.sphere_radius)
    sphere_r = _base_sphere("_sphere_R", mat_right, args.sphere_radius)
    sphere_m = _base_sphere("_sphere_M", mat_mid,   args.sphere_radius)

    n_left = n_right = n_mid = 0
    for i, pt in enumerate(points):
        x, y, z, hemi = pt["x"], pt["y"], pt["z"], pt["hemi"]

        if hemi == "L":
            src = sphere_l; n_left += 1
        elif hemi == "R":
            src = sphere_r; n_right += 1
        else:
            src = sphere_m; n_mid += 1

        # Duplicate linked (shares mesh data — memory efficient for many points)
        dup = src.copy()
        dup.data = src.data   # shared mesh
        dup.location = mathutils.Vector((x, y, z))
        dup.name = f"stim_{i:04d}_{hemi}"
        scene.collection.objects.link(dup)

    print(f"[blender_tms] Placed {n_left} L points, {n_right} R points, {n_mid} midline points")

    # ---- Camera ----
    # Position for a good superior-lateral view of the whole brain
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = mathutils.Vector((180, -220, 160))
    cam_obj.rotation_euler = mathutils.Euler((1.1, 0.0, 0.65), "XYZ")
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # ---- Key light ----
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.location = mathutils.Vector((100, -100, 200))
    sun_obj.rotation_euler = mathutils.Euler((0.7, 0.0, 0.8), "XYZ")
    scene.collection.objects.link(sun_obj)

    # ---- Fill light ----
    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 500
    fill_data.size   = 200
    fill_obj = bpy.data.objects.new("Fill", fill_data)
    fill_obj.location = mathutils.Vector((-150, 100, 80))
    scene.collection.objects.link(fill_obj)


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    args = parse_result = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    blend_path  = os.path.join(args.output_dir, f"{args.output_prefix}_surface.blend")
    render_path = os.path.join(args.output_dir, f"{args.output_prefix}_surface.png")

    print(f"[blender_tms] Loading surfaces (mesh={args.fsaverage_mesh})...")
    surfaces = _get_surfaces(args)

    print(f"[blender_tms] Loading stimulation coordinates...")
    points = _load_coords(args.fsaverage_csvs)

    if not points:
        print("[blender_tms] ERROR: No valid stimulation points found. Check your CSV(s).")
        sys.exit(1)

    print(f"[blender_tms] Building Blender scene...")
    _build_scene(surfaces, points, args)

    import bpy

    # Save .blend
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[blender_tms] Saved .blend: {blend_path}")
    print(f"[blender_tms] Open in Blender GUI to explore interactively.")

    # Render PNG
    if not args.blend_only and not args.no_render:
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        print(f"[blender_tms] Rendered PNG: {render_path}")
    else:
        print(f"[blender_tms] Skipping render (--blend-only or --no-render set).")

    print("[blender_tms] Done.")


if __name__ == "__main__":
    main()