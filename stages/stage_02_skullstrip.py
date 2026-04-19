"""
stages/02_skullstrip.py
=======================
Stage 02: Brain extraction.

Strategy:
  1. Try mri_synthstrip (FreeSurfer) — best accuracy on most T1s
  2. On failure (including the PyTorch "input tensor is too large" error
     that occurs with some image orientations/sizes in FreeSurfer 8.1.0),
     fall back to FSL BET with robust-mode enabled

The original image and its voxel grid are NEVER modified — only a
skull-stripped copy is written to paths.t1_brain.  All downstream
coordinate transforms operate on the original T1.

Notes on the SynthStrip tensor error
--------------------------------------
The crash in mri_synthstrip line 281:
    inp /= inp.quantile(0.99)
    RuntimeError: quantile() input tensor is too large

occurs when the image has an unusual orientation (e.g. PIR) that
causes an internal reorient step to produce a malformed tensor.
The image dimensions themselves are not the cause — standard-sized
T1s in non-RAS orientations can trigger it.  BET handles any
orientation natively.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from utils.io import PathManifest
from utils.logger import get_stage_logger

log = get_stage_logger("skullstrip")


def run(args, paths: PathManifest) -> None:
    """
    Produce paths.t1_brain (skull-stripped T1 in original voxel space).
    """
    log.info("Input T1  : %s", paths.t1)
    log.info("Output    : %s", paths.t1_brain)

    paths.t1_brain.parent.mkdir(parents=True, exist_ok=True)

    _try_synthstrip(paths)


def _try_synthstrip(paths: PathManifest) -> None:
    """Attempt SynthStrip; fall back to BET on any failure."""
    log.info("Attempting brain extraction with mri_synthstrip...")

    cmd = ["mri_synthstrip", "-i", str(paths.t1), "-o", str(paths.t1_brain)]
    log.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        log.info("SynthStrip succeeded.")
        log.debug("SynthStrip stdout:\n%s", result.stdout)
        return

    # --- SynthStrip failed ---
    tensor_error = "input tensor is too large" in result.stderr
    if tensor_error:
        reason = "PyTorch quantile() tensor-too-large error (likely non-standard orientation)"
    else:
        reason = f"exit code {result.returncode}"

    log.warning("SynthStrip failed: %s", reason)
    log.debug("SynthStrip stderr:\n%s", result.stderr)

    # Clean up any partial output
    if paths.t1_brain.exists():
        paths.t1_brain.unlink()
        log.debug("Removed partial SynthStrip output.")

    _run_bet(paths)


def _run_bet(paths: PathManifest) -> None:
    """
    FSL BET fallback.

    -R: robust brain centre estimation (iterative; better for unusual images)
    -f: fractional intensity threshold
        0.5 is the FSL default and works for most adult T1s.
        Lower (e.g. 0.3) if BET over-strips; raise (0.6) if under-strips.
    -g: vertical gradient — 0 is neutral (no top/bottom bias)
    """
    log.info("Falling back to FSL BET (robust mode, -f 0.5)...")

    # BET writes to the path we give it, but some versions append _brain
    # automatically.  Give it the final path directly and handle both cases.
    cmd = [
        "bet", str(paths.t1), str(paths.t1_brain),
        "-R", "-f", "0.5", "-g", "0",
    ]
    log.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("BET failed (exit code %d):", result.returncode)
        log.error("BET stderr:\n%s", result.stderr)
        log.error("")
        log.error("Both SynthStrip and BET failed. Suggestions:")
        log.error("  1. Check image: mri_info %s", paths.t1)
        log.error("  2. Inspect the affine matrix:")
        log.error("       python -c \"import nibabel as nib; "
                  "print(nib.load('%s').affine)\"", paths.t1)
        raise RuntimeError("Brain extraction failed — see log for details.")

    log.debug("BET stdout:\n%s", result.stdout)

    # Handle BET versions that append _brain to the output filename
    if not paths.t1_brain.exists():
        auto_name = Path(str(paths.t1_brain).replace(".nii.gz", "_brain.nii.gz"))
        if auto_name.exists():
            auto_name.rename(paths.t1_brain)
            log.debug("Renamed BET output: %s -> %s", auto_name, paths.t1_brain)
        else:
            log.error("BET output not found at: %s", paths.t1_brain)
            log.error("Contents of %s:", paths.t1_brain.parent)
            for f in sorted(paths.t1_brain.parent.iterdir()):
                log.error("  %s", f.name)
            raise RuntimeError(f"BET output missing: {paths.t1_brain}")

    log.info("BET succeeded.")
    log.info("Brain image : %s", paths.t1_brain)
