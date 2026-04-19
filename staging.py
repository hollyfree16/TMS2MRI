"""
staging.py
==========
Stage runner for tms2mni.

Provides:
  - run_stage()     — execute a stage function or skip if outputs exist
  - report()        — print/log a final summary of all stage outputs
  - make_dirs()     — create all required output directories upfront

Skip logic is file-driven: a stage is skipped when all of its declared
output files already exist on disk.  Pass --force on the CLI to override
and re-run everything.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from utils.io import PathManifest
from utils.checks import all_exist, missing

log = logging.getLogger("tms2mni.staging")


# =============================================================================
# Directory creation
# =============================================================================

def make_dirs(paths: PathManifest) -> None:
    """Create all output directories before any stage runs."""
    for d in paths.required_dirs():
        d.mkdir(parents=True, exist_ok=True)
    log.debug("Output directories created under: %s", paths.subject_dir)


# =============================================================================
# Stage runner
# =============================================================================

def run_stage(
    name:     str,
    fn:       Callable,
    args,
    paths:    PathManifest,
    *,
    enabled:  bool = True,
    force:    bool = False,
) -> bool:
    """
    Run a pipeline stage, skipping it if outputs already exist.

    Parameters
    ----------
    name:     Short stage name matching PathManifest.stage_outputs() keys
              (e.g. 'parse', 'skullstrip', 'register', 'convert', 'visualize').
    fn:       Stage callable — signature fn(args, paths) -> None.
    args:     Parsed CLI namespace.
    paths:    PathManifest for this subject.
    enabled:  If False the stage is silently skipped (e.g. MNI stages when
              no template was provided).
    force:    If True, skip the existence check and always re-run.

    Returns
    -------
    True if the stage ran, False if it was skipped.
    """
    banner = f"{'='*60}"

    if not enabled:
        log.info("%s", banner)
        log.info("Stage %-20s  DISABLED (not configured)", name)
        return False

    expected = paths.stage_outputs(name)

    if not force and all_exist(expected):
        log.info("%s", banner)
        log.info("Stage %-20s  SKIPPED  (all outputs exist)", name)
        for p in expected:
            log.debug("  exists: %s", p)
        return False

    if not force:
        absent = missing(expected)
        if absent:
            log.debug("Missing outputs that triggered run:")
            for p in absent:
                log.debug("  missing: %s", p)

    log.info("%s", banner)
    log.info("Stage %-20s  RUNNING", name)

    t0 = time.monotonic()
    try:
        fn(args, paths)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.error("Stage %-20s  FAILED  (%.1fs)", name, elapsed)
        log.exception("Unhandled exception in stage %r:", name)
        raise SystemExit(1) from exc

    elapsed = time.monotonic() - t0

    # Verify outputs were actually produced
    still_missing = missing(expected)
    if still_missing:
        log.error("Stage %-20s  INCOMPLETE — missing outputs:", name)
        for p in still_missing:
            log.error("  %s", p)
        raise SystemExit(1)

    log.info("Stage %-20s  DONE     (%.1fs)", name, elapsed)
    for p in expected:
        log.debug("  wrote: %s", p)

    return True


# =============================================================================
# Final report
# =============================================================================

def report(paths: PathManifest, subject_id: str) -> None:
    """
    Log a human-readable summary of all expected outputs and their status.
    Called at the end of run_tms2mni.py.
    """
    stages = ["parse", "skullstrip", "register", "convert", "visualize"]

    log.info("")
    log.info("=" * 60)
    log.info("Pipeline complete  |  %s", subject_id)
    log.info("=" * 60)
    log.info("Output directory: %s", paths.subject_dir)
    log.info("")

    all_ok = True

    for stage in stages:
        try:
            outputs = paths.stage_outputs(stage)
        except ValueError:
            continue

        log.info("  [%s]", stage)
        for p in outputs:
            status = "OK " if p.exists() else "MISSING"
            if not p.exists():
                all_ok = False
            log.info("    [%s]  %s", status, p)

    if paths.shared_csv:
        status = "OK " if paths.shared_csv.exists() else "MISSING"
        log.info("  [shared csv]")
        log.info("    [%s]  %s", status, paths.shared_csv)

    log.info("")
    if all_ok:
        log.info("All outputs present.")
    else:
        log.warning("Some outputs are missing — check logs above for errors.")

    log.info("Log directory: %s", paths.log_dir)
    log.info("=" * 60)
