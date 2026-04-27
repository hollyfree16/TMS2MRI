"""
stages/stage_03_register_mni.py
=================================
Stage 03: Register skull-stripped T1 to MNI152 brain template using ANTs SyN.

Outputs (all written to paths.reg_dir):
  sub_to_MNI_0GenericAffine.mat   — affine component
  sub_to_MNI_1Warp.nii.gz         — nonlinear warp  (forward)
  sub_to_MNI_1InverseWarp.nii.gz  — nonlinear warp  (inverse, for point transforms)
  T1_in_MNI.nii.gz                — full-head T1 warped to MNI (visualization only)

Non-RAS orientation handling
-----------------------------
ANTs can fail or produce flipped registrations when the moving image is in a
non-RAS orientation (LAS, PIR etc.) because its centre-of-mass initialisation
assumes standard orientation. To avoid this, we reorient the skull-stripped
T1 to RAS before passing it to ANTs. The resulting transforms are in the same
physical space and are fully valid for warping coordinates from the original
non-RAS image — the reorientation only affects how ANTs initialises.

The reoriented image is written to T1_brain_RAS.nii.gz (temporary, kept for
debugging). The original T1 and T1_brain are never modified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nibabel as nib
import nibabel.orientations as ornt
import numpy as np

from utils.io import PathManifest
from utils.logger import get_stage_logger

log = get_stage_logger("register_mni")


def run(args, paths: PathManifest) -> None:
    if paths.mni_template is None:
        raise RuntimeError("register_mni called but no MNI template provided.")

    paths.reg_dir.mkdir(parents=True, exist_ok=True)

    log.info("Moving  (brain): %s", paths.t1_brain)
    log.info("Fixed   (brain): %s", paths.mni_template)

    # Reorient skull-stripped T1 to RAS if needed
    t1_brain_for_ants = _ensure_ras(paths.t1_brain, paths.reg_dir)

    _run_registration(paths, t1_brain_for_ants)
    _warp_full_t1(args, paths)


def _ensure_ras(t1_brain: Path, reg_dir: Path) -> Path:
    """
    If t1_brain is not RAS, reorient it to RAS and return the path to the
    reoriented image. If already RAS, return t1_brain unchanged.

    The reoriented image is written to reg_dir/T1_brain_RAS.nii.gz.
    This is used only for ANTs registration initialisation — all coordinate
    transforms remain valid for the original image space.
    """
    img    = nib.load(str(t1_brain))
    codes  = ornt.aff2axcodes(img.affine)

    if codes == ("R", "A", "S"):
        log.info("T1_brain is already RAS — no reorientation needed.")
        return t1_brain

    log.info("T1_brain orientation: %s — reorienting to RAS for ANTs initialisation.",
             "".join(codes))

    current_ornt = ornt.io_orientation(img.affine)
    ras_ornt     = ornt.axcodes2ornt(("R", "A", "S"))
    transform    = ornt.ornt_transform(current_ornt, ras_ornt)
    ras_img      = img.as_reoriented(transform)

    ras_path = reg_dir / "T1_brain_RAS.nii.gz"
    nib.save(ras_img, str(ras_path))
    log.info("Reoriented T1_brain saved: %s", ras_path)

    return ras_path


def _run_registration(paths: PathManifest, t1_brain_moving: Path) -> None:
    """Run antsRegistrationSyN.sh to produce affine + warp files."""
    reg_prefix = str(paths.reg_dir / "sub_to_MNI_")

    log.info("Running antsRegistrationSyN.sh (this may take several minutes)...")
    log.info("  Moving : %s", t1_brain_moving)
    log.info("  Fixed  : %s", paths.mni_template)

    cmd = [
        "antsRegistrationSyN.sh",
        "-d", "3",
        "-f", str(paths.mni_template),
        "-m", str(t1_brain_moving),
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