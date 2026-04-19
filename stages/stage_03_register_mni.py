"""
stages/03_register_mni.py
=========================
Stage 03: Register skull-stripped T1 to MNI152 brain template using ANTs SyN.

Outputs (all written to paths.reg_dir):
  sub_to_MNI_0GenericAffine.mat   — affine component
  sub_to_MNI_1Warp.nii.gz         — nonlinear warp  (forward)
  sub_to_MNI_1InverseWarp.nii.gz  — nonlinear warp  (inverse, for point transforms)
  T1_in_MNI.nii.gz                — full-head T1 warped to MNI (visualization only)

The transforms are consumed by stage 04 to warp stimulation coordinates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from utils.io import PathManifest
from utils.logger import get_stage_logger

log = get_stage_logger("register_mni")


def run(args, paths: PathManifest) -> None:
    """
    Run antsRegistrationSyN.sh (brain→brain) then warp full-head T1 for
    visualization.
    """
    if paths.mni_template is None:
        raise RuntimeError("register_mni called but no MNI template provided.")

    paths.reg_dir.mkdir(parents=True, exist_ok=True)

    log.info("Moving  (brain): %s", paths.t1_brain)
    log.info("Fixed   (brain): %s", paths.mni_template)

    _run_registration(paths)
    _warp_full_t1(args, paths)


def _run_registration(paths: PathManifest) -> None:
    """Run antsRegistrationSyN.sh to produce affine + warp files."""
    reg_prefix = str(paths.reg_dir / "sub_to_MNI_")

    log.info("Running antsRegistrationSyN.sh (this may take several minutes)...")

    cmd = [
        "antsRegistrationSyN.sh",
        "-d", "3",
        "-f", str(paths.mni_template),
        "-m", str(paths.t1_brain),
        "-o", reg_prefix,
        "-t", "s",     # SyN (affine + deformable)
        "-n", "4",     # threads
    ]
    log.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    log.debug("ANTs stdout:\n%s", result.stdout)

    if result.returncode != 0:
        log.error("antsRegistrationSyN.sh failed (exit code %d):", result.returncode)
        log.error("ANTs stderr:\n%s", result.stderr)
        raise RuntimeError("ANTs registration failed — see log for details.")

    # Verify expected transform files were created
    for p in [paths.affine_mat, paths.warp, paths.inv_warp]:
        if not p.exists():
            log.error("Expected transform file missing after registration: %s", p)
            raise RuntimeError(f"ANTs output missing: {p}")

    log.info("Registration complete.")
    log.info("  Affine    : %s", paths.affine_mat)
    log.info("  Warp      : %s", paths.warp)
    log.info("  Inv warp  : %s", paths.inv_warp)


def _warp_full_t1(args, paths: PathManifest) -> None:
    """Apply warp to full-head T1 for visualization (optional)."""
    if paths.mni_template_full is None or paths.t1_in_mni is None:
        log.info("No full-head MNI template provided — skipping warped T1 output.")
        return

    log.info("Warping full-head T1 to MNI space for visualization...")
    log.info("  Reference : %s", paths.mni_template_full)

    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(paths.t1),
        "-r", str(paths.mni_template_full),
        "-t", str(paths.warp),
        "-t", str(paths.affine_mat),
        "-o", str(paths.t1_in_mni),
        "--interpolation", "LanczosWindowedSinc",
    ]
    log.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    log.debug("antsApplyTransforms stdout:\n%s", result.stdout)

    if result.returncode != 0:
        log.error("antsApplyTransforms failed (exit code %d):", result.returncode)
        log.error("stderr:\n%s", result.stderr)
        raise RuntimeError("Warping full-head T1 failed — see log for details.")

    log.info("Warped T1 : %s", paths.t1_in_mni)
