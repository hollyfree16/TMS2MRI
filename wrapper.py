"""
NextStim Full Pipeline Wrapper
================================
Chains parse_nbe -> nextstim_to_mri -> visualize_mni into a single command.

Steps:
  1. parse_nbe.py        — parse .nbe file into landmarks.csv + targets.csv
  2. nextstim_to_mri.py  — convert NBE coords to subject MRI + MNI space
  3. visualize_mni.py    — plot selected IDs on 3D glass brain

Usage:
    python run_pipeline.py \
        --nbe nbe/auto_TEP004_2021_09_13.nbe \
        --t1 MRI/TEP004/T1.nii.gz \
        --subject-id TEP004 \
        --output-dir ./outputs \
        --mni-template /path/to/MNI152_T1_1mm_brain.nii.gz \
        --mni-template-full /path/to/MNI152_T1_1mm.nii.gz \
        --id 1.26.23 --id 1.48.18 \
        --site M1_L \
        --shared-csv ./all_subjects_mni.csv

Optional flags:
    --no-flip           disable X-axis flip in nextstim_to_mri
    --skip-parse        skip step 1 (if targets/landmarks already exist)
    --skip-registration skip MNI registration in step 2 (subject space only)
    --skip-viz          skip step 3 (no visualization)
    --color-by-hemi     colour L/R points differently in visualization
    --marker-size N     marker size for visualization (default: 5)
    --coord-col         ef or coil (default: ef)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Run the full NextStim -> MRI -> MNI -> Visualization pipeline."
)

# --- Core required ---
parser.add_argument("--nbe",              required=True,
                    help="Path to input .nbe file")
parser.add_argument("--t1",               required=True,
                    help="Path to subject T1 NIfTI")
parser.add_argument("--subject-id",       required=True,
                    help="Subject identifier (e.g. TEP004)")
parser.add_argument("--output-dir",       required=True,
                    help="Root output directory")

# --- MNI (optional) ---
parser.add_argument("--mni-template",     default=None,
                    help="Brain-extracted MNI152 template (MNI152_T1_1mm_brain.nii.gz)")
parser.add_argument("--mni-template-full",default=None,
                    help="Full-head MNI152 template (MNI152_T1_1mm.nii.gz)")

# --- Visualization ---
parser.add_argument("--id",               action="append", dest="ids", default=None,
                    help="ID(s) to visualize. Repeat for multiple: --id 1.26.23 --id 1.48.18. "
                         "Use --id all for everything.")
parser.add_argument("--site",             default=None,
                    help="Stimulation site label for shared CSV (e.g. M1_L)")
parser.add_argument("--shared-csv",       default=None,
                    help="Path to shared CSV to append MNI coordinates to")
parser.add_argument("--color-by-hemi",    action="store_true",
                    help="Colour visualization points by hemisphere")
parser.add_argument("--marker-size",      type=float, default=5,
                    help="Marker size for visualization (default: 5)")
parser.add_argument("--coord-col",        default="ef", choices=["ef", "coil"],
                    help="Coordinate column to visualize: ef (default) or coil")

# --- Pipeline control ---
parser.add_argument("--no-flip",          action="store_true",
                    help="Disable X-axis flip in nextstim_to_mri")
parser.add_argument("--skip-parse",       action="store_true",
                    help="Skip step 1 (parse_nbe) — use existing targets/landmarks CSV")
parser.add_argument("--skip-registration",action="store_true",
                    help="Skip MNI registration — run subject space only")
parser.add_argument("--skip-viz",         action="store_true",
                    help="Skip step 3 (visualization)")

args = parser.parse_args()

# =============================================================================
# HELPERS
# =============================================================================

PYTHON = sys.executable  # use same python that's running this wrapper

def run(cmd, step_name):
    """Run a command, print it, exit on failure."""
    print(f"\n{'='*60}")
    print(f"[{step_name}]")
    print(f"{'='*60}")
    print("$ " + " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

# Resolve script paths relative to this wrapper's location
SCRIPT_DIR   = Path(__file__).parent
PARSE_SCRIPT = SCRIPT_DIR / "parse_nbe.py"
MRI_SCRIPT   = SCRIPT_DIR / "nextstim_to_mri.py"
VIZ_SCRIPT   = SCRIPT_DIR / "visualize_mni.py"

for script in [PARSE_SCRIPT, MRI_SCRIPT, VIZ_SCRIPT]:
    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        print(f"       Make sure parse_nbe.py, nextstim_to_mri.py, and visualize_mni.py "
              f"are in the same directory as run_pipeline.py.")
        sys.exit(1)

subject_id = args.subject_id
output_dir = Path(args.output_dir)

# =============================================================================
# STEP 1: parse_nbe
# =============================================================================

# parse_nbe outputs to: <output_dir>/<subject_id>/xnbe/
xnbe_dir      = output_dir / subject_id / "xnbe"
targets_csv   = xnbe_dir / f"targets_{subject_id}.csv"
landmarks_csv = xnbe_dir / f"landmarks_{subject_id}.csv"

if args.skip_parse:
    print(f"\n[Step 1] Skipping parse_nbe — using existing files:")
    print(f"  Targets  : {targets_csv}")
    print(f"  Landmarks: {landmarks_csv}")
    for f in [targets_csv, landmarks_csv]:
        if not f.exists():
            print(f"  ERROR: File not found: {f}")
            sys.exit(1)
else:
    cmd = [
        PYTHON, PARSE_SCRIPT,
        "-i", args.nbe,
        "-o", str(output_dir),
        "-s", subject_id,
    ]
    run(cmd, "Step 1: parse_nbe")

# =============================================================================
# STEP 2: nextstim_to_mri
# =============================================================================

DO_MNI = (args.mni_template is not None) and not args.skip_registration

cmd = [
    PYTHON, MRI_SCRIPT,
    "--t1",          args.t1,
    "--targets",     str(targets_csv),
    "--landmarks",   str(landmarks_csv),
    "--subject-id",  subject_id,
    "--output-dir",  str(output_dir),
]

if args.no_flip:
    cmd.append("--no-flip")

if DO_MNI:
    cmd += ["--mni-template",      args.mni_template,
            "--mni-template-full", args.mni_template_full]

run(cmd, "Step 2: nextstim_to_mri")

# =============================================================================
# STEP 3: visualize_mni
# =============================================================================

if args.skip_viz:
    print(f"\n[Step 3] Skipping visualization (--skip-viz).")
elif args.ids is None:
    print(f"\n[Step 3] Skipping visualization — no --id flags provided.")
    print(f"         Run visualize_mni.py manually to generate plots.")
else:
    # Determine which CSV to read
    if DO_MNI:
        viz_csv = output_dir / subject_id / "coordinates" / "targets_mni.csv"
    else:
        viz_csv = output_dir / subject_id / "coordinates" / "targets_native.csv"

    if not viz_csv.exists():
        print(f"\n[Step 3] WARNING: Coordinate CSV not found: {viz_csv}")
        print(f"         Skipping visualization.")
    else:
        # Default output paths under subject directory
        viz_output_dir = output_dir / subject_id
        html_out  = viz_output_dir / "stimulation_sites.html"
        png_out   = viz_output_dir / "stimulation_sites.png"

        cmd = [
            PYTHON, VIZ_SCRIPT,
            "--csv",        str(viz_csv),
            "--coord-col",  args.coord_col,
            "--marker-size",str(args.marker_size),
            "--output",     str(html_out),
            "--screenshot", str(png_out),
        ]

        for id_val in args.ids:
            cmd += ["--id", id_val]

        if args.color_by_hemi:
            cmd.append("--color-by-hemi")

        if args.subject_id:
            cmd += ["--subject-id", subject_id]

        if args.site:
            cmd += ["--site", args.site]

        if args.shared_csv:
            cmd += ["--shared-csv", args.shared_csv]

        run(cmd, "Step 3: visualize_mni")

# =============================================================================
# DONE
# =============================================================================

print(f"""
{'='*60}
Pipeline complete  |  {subject_id}
{'='*60}
Output directory : {output_dir / subject_id}

  xnbe/
    targets_{subject_id}.csv
    landmarks_{subject_id}.csv

  coordinates/
    targets_native.csv""")

if DO_MNI:
    print(f"    targets_mni.csv")
    print(f"\n  registration/")
    print(f"    T1_brain.nii.gz")
    print(f"    T1_in_MNI.nii.gz")
    print(f"    sub_to_MNI_0GenericAffine.mat")
    print(f"    sub_to_MNI_1Warp.nii.gz")

if args.ids and not args.skip_viz:
    print(f"\n  stimulation_sites.html  (interactive)")
    print(f"  stimulation_sites.png   (static 4-panel)")

if args.shared_csv:
    print(f"\nShared CSV updated: {args.shared_csv}")

print(f"{'='*60}\n")