"""
run_tms2mni.py
==============
Entry point for the tms2mni pipeline.

Chains:
  01  parse_nbe       — .nbe → landmarks.csv + targets.csv
  02  skullstrip      — T1 → T1_brain  (SynthStrip, BET fallback)
  03  register_mni    — T1_brain → MNI152  (ANTs SyN)
  04  convert_coords  — NBE → native/MNI mm + fsaverage snap
                        writes full CSVs + summary CSV (when --id used)
  05  visualize       — glass brain + HO atlas plot + shared CSV
  06  snap_surface    — stub (reserved for future Blender integration)

Usage
-----
  python run_tms2mni.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging

from cli import parse_args
from staging import make_dirs, run_stage, report
from utils.logger import configure_logging

from stages import (
    stage_01_parse_nbe      as s01,
    stage_02_skullstrip     as s02,
    stage_03_register_mni   as s03,
    stage_04_convert_coords as s04,
    stage_05_visualize      as s05,
    stage_06_snap_surface   as s06,
)


def main() -> None:
    args, paths = parse_args()

    make_dirs(paths)

    configure_logging(
        log_dir    = paths.log_dir,
        subject_id = args.subject_id,
        level      = args.log_level_int,
    )

    log = logging.getLogger("tms2mni")
    log.info("tms2mni pipeline starting")
    log.info("Subject    : %s", args.subject_id)
    log.info("NBE file   : %s", paths.nbe)
    log.info("T1         : %s", paths.t1)
    log.info("Output dir : %s", paths.subject_dir)
    if paths.mni_template:
        log.info("MNI template (brain) : %s", paths.mni_template)
    if paths.mni_template_full:
        log.info("MNI template (full)  : %s", paths.mni_template_full)
    if args.ids:
        log.info("IDs requested : %s", args.ids)

    do_mni = paths.mni_template is not None
    force  = args.force

    run_stage("parse",     s01.run, args, paths, force=force)

    run_stage("skullstrip", s02.run, args, paths,
              enabled = do_mni,
              force   = force)

    run_stage("register",  s03.run, args, paths,
              enabled = do_mni,
              force   = force)

    run_stage("convert",   s04.run, args, paths, force=force)

    run_stage("visualize", s05.run, args, paths,
              enabled = (args.ids is not None) and not args.skip_viz,
              force   = force)

    # Stage 06 is a stub — disabled by default until Blender integration lands
    # run_stage("snap", s06.run, args, paths, enabled=False, force=force)

    report(paths, args.subject_id)


if __name__ == "__main__":
    main()