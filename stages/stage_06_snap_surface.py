"""
stages/stage_06_snap_surface.py
================================
Stage 06: Subject-level surface visualization (stub).

fsaverage pial surface snapping has moved to stage 04, which now writes
targets_fsaverage.csv alongside targets_native.csv and targets_mni.csv.

This stage is reserved for future subject-level surface rendering work,
e.g. invoking the Blender pipeline (blender_tms_surface.py) automatically
as part of the per-subject run.
"""

from __future__ import annotations

from utils.io import PathManifest
from utils.logger import get_stage_logger

log = get_stage_logger("snap_surface")


def run(args, paths: PathManifest) -> None:
    log.info("Stage 06 is a stub — surface visualization work in progress.")
    log.info("fsaverage snapping is handled by stage 04.")
    log.info("Use blender_tms_surface.py with targets_fsaverage_filtered.csv "
             "for interactive surface rendering.")