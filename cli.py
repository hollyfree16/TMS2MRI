"""
cli.py
======
Argument parsing for tms2mni.  Returns a (args, paths) tuple.

No pipeline logic lives here — only flag definitions and the
PathManifest construction call.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from utils.io import PathManifest


def parse_args() -> tuple[argparse.Namespace, PathManifest]:
    """
    Parse CLI arguments and build the PathManifest.

    Returns
    -------
    (args, paths)
        args:  raw parsed namespace (pipeline control flags, IDs, etc.)
        paths: fully-populated PathManifest derived from args
    """
    p = argparse.ArgumentParser(
        prog="run_tms2mni",
        description="Convert NextStim TMS coordinates to MNI152 space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Full pipeline
  python run_tms2mni.py \\
      --nbe data/TEP004.nbe \\
      --t1  data/TEP004_T1.nii.gz \\
      --subject-id TEP004 \\
      --output-dir ./outputs \\
      --mni-template     /path/to/MNI152_T1_1mm_brain.nii.gz \\
      --mni-template-full /path/to/MNI152_T1_1mm.nii.gz \\
      --id 1.26.23 --id 1.48.18 \\
      --site M1_L \\
      --shared-csv ./all_subjects.csv

  # Subject space only (skip MNI registration)
  python run_tms2mni.py \\
      --nbe data/TEP004.nbe \\
      --t1  data/TEP004_T1.nii.gz \\
      --subject-id TEP004 \\
      --output-dir ./outputs
""",
    )

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #
    req = p.add_argument_group("required")
    req.add_argument("--nbe",         required=True, type=Path,
                     help=".nbe export file from NextStim")
    req.add_argument("--t1",          required=True, type=Path,
                     help="Subject T1 NIfTI (e.g. T1.nii.gz)")
    req.add_argument("--subject-id",  required=True,
                     help="Subject identifier used in output filenames")
    req.add_argument("--output-dir",  required=True, type=Path,
                     help="Root output directory")

    # ------------------------------------------------------------------ #
    # MNI registration (optional — omit to run subject-space only)
    # ------------------------------------------------------------------ #
    mni = p.add_argument_group("MNI registration (optional)")
    mni.add_argument("--mni-template",      default=None, type=Path,
                     help="Brain-extracted MNI152 template for ANTs registration "
                          "(e.g. MNI152_T1_1mm_brain.nii.gz)")
    mni.add_argument("--mni-template-full", default=None, type=Path,
                     help="Full-head MNI152 template — used to warp T1 for visualization "
                          "(e.g. MNI152_T1_1mm.nii.gz)")

    # ------------------------------------------------------------------ #
    # Coordinate conversion
    # ------------------------------------------------------------------ #
    coords = p.add_argument_group("coordinate conversion")
    coords.add_argument("--no-flip", action="store_true",
                        help="Disable X-axis flip in NBE→voxel transform. "
                             "Use if stimulation points appear left/right swapped.")

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    viz = p.add_argument_group("visualization")
    viz.add_argument("--id", action="append", dest="ids", default=None,
                     metavar="ID",
                     help="Row ID(s) to visualize and include in filtered CSVs. "
                          "Repeat for multiple: --id 1.26.23 --id 1.48.18. "
                          "Use --id all for everything.")
    viz.add_argument("--coord-col", default="ef", choices=["ef", "coil"],
                     help="Coordinate set to plot: ef (default) or coil")
    viz.add_argument("--color-by-hemi", action="store_true",
                     help="Colour visualization points by hemisphere (L=blue, R=red)")
    viz.add_argument("--marker-size", type=float, default=5,
                     help="Marker size for glass brain plot (default: 5)")

    # ------------------------------------------------------------------ #
    # Surface snapping (stage 04)
    # ------------------------------------------------------------------ #
    snap = p.add_argument_group("surface snapping")
    snap.add_argument(
        "--fsaverage-mesh",
        default="fsaverage",
        choices=["fsaverage", "fsaverage5", "fsaverage6"],
        help="fsaverage mesh resolution for surface snapping. "
             "fsaverage  = 163,842 vertices/hemi (~0.7 mm, default). "
             "fsaverage6 =  40,962 vertices/hemi (~1.5 mm). "
             "fsaverage5 =  10,242 vertices/hemi (~3.5 mm, fastest).",
    )
    snap.add_argument(
        "--skip-snap", action="store_true",
        help="Skip fsaverage surface snapping even when MNI registration "
             "is available.",
    )

    # ------------------------------------------------------------------ #
    # Shared CSV
    # ------------------------------------------------------------------ #
    shared = p.add_argument_group("shared / multi-subject output")
    shared.add_argument("--site",       default=None,
                        help="Stimulation site label appended to shared CSV "
                             "(e.g. M1_L)")
    shared.add_argument("--shared-csv", default=None, type=Path,
                        help="Cross-subject CSV to append MNI coordinates to. "
                             "Created if it does not exist.")

    # ------------------------------------------------------------------ #
    # Pipeline control
    # ------------------------------------------------------------------ #
    pipeline = p.add_argument_group("pipeline control")
    pipeline.add_argument("--skip-viz", action="store_true",
                          help="Skip stage 05 (visualization) regardless of outputs")
    pipeline.add_argument("--force",    action="store_true",
                          help="Re-run all stages even if outputs already exist")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_grp = p.add_argument_group("logging")
    log_grp.add_argument("--log-level",
                         default="INFO",
                         choices=["DEBUG", "INFO", "WARNING"],
                         help="Console log level (default: INFO). "
                              "Log file always captures DEBUG.")

    # ------------------------------------------------------------------ #
    # Parse
    # ------------------------------------------------------------------ #
    args = p.parse_args()

    # Validate inputs exist
    _require_file(p, args.nbe, "--nbe")
    _require_file(p, args.t1,  "--t1")
    if args.mni_template:
        _require_file(p, args.mni_template,      "--mni-template")
    if args.mni_template_full:
        _require_file(p, args.mni_template_full, "--mni-template-full")

    # Map log-level string to int
    args.log_level_int = getattr(logging, args.log_level)

    # Determine whether IDs were provided (drives filtered CSV generation)
    has_ids = args.ids is not None

    # Build path manifest
    paths = PathManifest.build(
        output_dir        = args.output_dir,
        subject_id        = args.subject_id,
        nbe               = args.nbe,
        t1                = args.t1,
        mni_template      = args.mni_template,
        mni_template_full = args.mni_template_full,
        shared_csv        = args.shared_csv,
        has_ids           = has_ids,
    )

    return args, paths


def _require_file(parser: argparse.ArgumentParser, path: Path, flag: str) -> None:
    if not path.exists():
        parser.error(f"{flag}: file not found: {path}")