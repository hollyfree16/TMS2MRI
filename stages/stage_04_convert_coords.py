"""
stages/04_convert_coords.py
===========================
Stage 04: Convert NextStim NBE coordinates to:
  - Subject native MRI voxel indices + mm  (targets_native.csv)
  - MNI152 mm coordinates via ANTs point transform  (targets_mni.csv)

Automatic flip retry
--------------------
If the hemisphere check fails OR all EF coordinates land outside the image
volume, the stage automatically retries with the opposite flip and logs a
warning. Common for LAS images.

Orientation-aware ANTs point transform
---------------------------------------
antsApplyTransformsToPoints expects LPS physical coordinates in the same
physical space as the moving image. For non-RAS images (PIR, LAS etc) the
physical coords from voxels_to_mm() are NOT in RAS — they are in scanner
physical space. We use nibabel's orientation transform to reorient to RAS
first, then negate X/Y for LPS, before passing to ANTs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import nibabel.orientations as ornt
import numpy as np
import nibabel as nib
import pandas as pd

from utils.affine import (
    nextstim_to_mri_voxels,
    voxel_sizes_from_affine,
    voxels_to_mm,
    mm_to_voxels,
    label_hemisphere,
    hemisphere_check,
)
from utils.checks import bounds_check, check_voxel_sizes
from utils.io import PathManifest, read_targets, read_landmarks, write_csv
from utils.logger import get_stage_logger

log = get_stage_logger("convert_coords")

TARGET_X = "ef_max_loc_x"
TARGET_Y = "ef_max_loc_y"
TARGET_Z = "ef_max_loc_z"
COIL_X   = "coil_loc_x"
COIL_Y   = "coil_loc_y"
COIL_Z   = "coil_loc_z"


def run(args, paths: PathManifest) -> None:
    # ------------------------------------------------------------------ #
    # Load T1 header
    # ------------------------------------------------------------------ #
    img    = nib.load(str(paths.t1))
    affine = img.affine
    shape  = img.shape[:3]

    vox_x, vox_y, vox_z = voxel_sizes_from_affine(affine)
    check_voxel_sizes(vox_x, vox_y, vox_z)

    dim_x      = shape[0]
    x_midpoint = dim_x / 2.0
    t1_ltr     = bool(affine[0, 0] > 0)

    log.info("T1 dimensions  : %s", shape)
    log.info("Voxel sizes    : %.4f x %.4f x %.4f mm", vox_x, vox_y, vox_z)
    log.info("X orientation  : %s",
             "Left-to-Right (RAS)" if t1_ltr else "Right-to-Left (LAS)")

    # Log the physical orientation so it's always auditable
    current_ornt = ornt.io_orientation(affine)
    ornt_codes   = ornt.ornt2axcodes(current_ornt)
    log.info("Physical space : %s", "".join(ornt_codes))

    tgt_df = read_targets(paths.targets_csv)
    lm_df  = read_landmarks(paths.landmarks_csv)
    log.info("Targets        : %d rows", len(tgt_df))
    log.info("Landmarks      : %d rows", len(lm_df))

    # ------------------------------------------------------------------ #
    # First conversion attempt
    # ------------------------------------------------------------------ #
    flip_x = not args.no_flip
    log.info("X flip         : %s (initial)", "ON" if flip_x else "OFF")

    result = _attempt(tgt_df, lm_df, flip_x,
                      dim_x, vox_x, vox_y, vox_z,
                      affine, shape, x_midpoint, t1_ltr)

    # ------------------------------------------------------------------ #
    # Automatic retry check
    # ------------------------------------------------------------------ #
    hemi_ok         = result["hemi_correct"]
    ef_in, ef_total = result["ef_bounds"]
    all_oob         = (ef_total > 0) and (ef_in == 0)

    if hemi_ok is False or all_oob:
        reasons = []
        if hemi_ok is False:
            reasons.append("hemisphere check failed")
        if all_oob:
            reasons.append("all EF coordinates out-of-bounds")
        log.warning("Auto-flip triggered (%s) — retrying with flip_x=%s",
                    ", ".join(reasons), not flip_x)

        retry = _attempt(tgt_df, lm_df, not flip_x,
                         dim_x, vox_x, vox_y, vox_z,
                         affine, shape, x_midpoint, t1_ltr)

        retry_in, retry_total = retry["ef_bounds"]
        retry_hemi            = retry["hemi_correct"]

        if retry_in > ef_in or retry_hemi is True:
            log.warning(
                "Auto-flip accepted: flip_x %s → %s  "
                "(EF in-bounds %d/%d → %d/%d, hemi_check %s → %s)",
                flip_x, not flip_x,
                ef_in, ef_total, retry_in, retry_total,
                hemi_ok, retry_hemi,
            )
            flip_x = not flip_x
            result = retry
        else:
            log.warning(
                "Auto-flip rejected: retry did not improve results. "
                "Keeping flip_x=%s. Manual review recommended.", flip_x)

    # ------------------------------------------------------------------ #
    # Unpack and save native CSV
    # ------------------------------------------------------------------ #
    lm_vox    = result["lm_vox"]
    lm_mm     = result["lm_mm"]
    ef_vox    = result["ef_vox"]
    ef_mm     = result["ef_mm"]
    ef_mask   = result["ef_mask"]
    coil_vox  = result["coil_vox"]
    coil_mm   = result["coil_mm"]
    coil_mask = result["coil_mask"]

    log.info("X flip used    : %s", "ON" if flip_x else "OFF")

    ef_hemi   = label_hemisphere(ef_vox[:,   0], x_midpoint, t1_ltr)
    coil_hemi = label_hemisphere(coil_vox[:, 0], x_midpoint, t1_ltr)

    native_df = tgt_df.copy()
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        native_df[f"ef_native_vox_{axis}"]   = np.where(
            np.isnan(ef_vox[:, idx]), np.nan, np.round(ef_vox[:, idx]))
        native_df[f"ef_native_mm_{axis}"]    = ef_mm[:, idx]
        native_df[f"coil_native_vox_{axis}"] = np.where(
            np.isnan(coil_vox[:, idx]), np.nan, np.round(coil_vox[:, idx]))
        native_df[f"coil_native_mm_{axis}"]  = coil_mm[:, idx]

    native_df["ef_hemisphere"]   = ef_hemi
    native_df["coil_hemisphere"] = coil_hemi
    native_df["flip_x_used"]     = flip_x

    paths.coords_dir.mkdir(parents=True, exist_ok=True)
    write_csv(native_df, paths.targets_native)
    log.info("Native CSV : %s", paths.targets_native)

    # ------------------------------------------------------------------ #
    # MNI transform
    # ------------------------------------------------------------------ #
    if paths.targets_mni is None:
        log.info("No MNI template configured — skipping MNI coordinate transform.")
        return

    if not paths.affine_mat.exists() or not paths.inv_warp.exists():
        log.error("Registration transforms not found — run stage 03 first.")
        raise RuntimeError("Missing ANTs transforms for MNI coordinate warp.")

    mni_img = nib.load(str(paths.mni_template))

    ef_mni_mm   = _apply_ants_transform(ef_mm,   ef_mask,   paths, affine)
    coil_mni_mm = _apply_ants_transform(coil_mm, coil_mask, paths, affine)
    lm_mni_mm   = _apply_ants_transform(
        lm_mm, np.ones(len(lm_mm), dtype=bool), paths, affine)

    ef_mni_vox   = np.full(ef_mni_mm.shape,   np.nan)
    coil_mni_vox = np.full(coil_mni_mm.shape, np.nan)
    ef_mni_vox[ef_mask]     = mm_to_voxels(ef_mni_mm[ef_mask],     mni_img.affine)
    coil_mni_vox[coil_mask] = mm_to_voxels(coil_mni_mm[coil_mask], mni_img.affine)

    log.info("Landmark MNI mm (RAS):")
    for i, row in lm_df.iterrows():
        m = lm_mni_mm[i]
        log.info("  %-20s  [%6.1f %6.1f %6.1f]",
                 row["landmark_type"], m[0], m[1], m[2])
    log.info("  (sanity: nasion ~[0,+80,-40], ears ~[+/-70,-10,-40])")

    mni_df = tgt_df.copy()
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        mni_df[f"ef_mni_mm_{axis}"]    = ef_mni_mm[:, idx]
        mni_df[f"ef_mni_vox_{axis}"]   = np.where(
            np.isnan(ef_mni_vox[:, idx]), np.nan, np.round(ef_mni_vox[:, idx]))
        mni_df[f"coil_mni_mm_{axis}"]  = coil_mni_mm[:, idx]
        mni_df[f"coil_mni_vox_{axis}"] = np.where(
            np.isnan(coil_mni_vox[:, idx]), np.nan, np.round(coil_mni_vox[:, idx]))

    mni_df["ef_hemisphere"]   = ef_hemi
    mni_df["coil_hemisphere"] = coil_hemi
    mni_df["flip_x_used"]     = flip_x

    write_csv(mni_df, paths.targets_mni)
    log.info("MNI CSV    : %s", paths.targets_mni)


def _attempt(
    tgt_df, lm_df,
    flip_x: bool,
    dim_x, vox_x, vox_y, vox_z,
    affine, shape, x_midpoint, t1_ltr,
) -> dict:
    """One full NBE → voxel → mm conversion attempt for a given flip_x."""
    lm_coords = lm_df[["x", "y", "z"]].values.astype(float)
    lm_vox    = nextstim_to_mri_voxels(
        lm_coords, dim_x, vox_x, vox_y, vox_z, flip_x)
    lm_mm     = voxels_to_mm(lm_vox, affine)

    log.info("Landmark positions (flip_x=%s):", flip_x)
    for i, row in lm_df.iterrows():
        v = lm_vox[i]
        if not all(np.isfinite(v[d]) for d in range(3)):
            log.warning("  %-20s  NON-FINITE ⚠️", row["landmark_type"])
            continue
        in_b = all(0 <= int(round(v[d])) < shape[d] for d in range(3))
        log.info("  %-20s  [%5.0f %5.0f %5.0f]  %s",
                 row["landmark_type"], v[0], v[1], v[2],
                 "OK" if in_b else "OUT OF BOUNDS ⚠️")

    lm_labels = lm_df["landmark_type"].tolist()
    hemi_ok   = hemisphere_check(lm_vox, lm_labels, x_midpoint, t1_ltr)
    if hemi_ok is True:
        log.info("Hemisphere check : ✅ passed (flip_x=%s)", flip_x)
    elif hemi_ok is False:
        log.warning("Hemisphere check : ⚠️  FAILED (flip_x=%s)", flip_x)
    else:
        log.info("Hemisphere check : skipped — landmarks missing (flip_x=%s)", flip_x)

    def _conv(x_col, y_col, z_col):
        coords = tgt_df[[x_col, y_col, z_col]].values.astype(float)
        mask   = ~np.isnan(coords).any(axis=1)
        vox    = np.full(coords.shape, np.nan)
        if mask.any():
            vox[mask] = nextstim_to_mri_voxels(
                coords[mask], dim_x, vox_x, vox_y, vox_z, flip_x)
        mm = np.full(coords.shape, np.nan)
        if mask.any():
            mm[mask] = voxels_to_mm(vox[mask], affine)
        return vox, mm, mask

    ef_vox,   ef_mm,   ef_mask   = _conv(TARGET_X, TARGET_Y, TARGET_Z)
    coil_vox, coil_mm, coil_mask = _conv(COIL_X,   COIL_Y,   COIL_Z)

    ef_in,   ef_total   = bounds_check(ef_vox,   shape)
    coil_in, coil_total = bounds_check(coil_vox, shape)

    log.info("EF   in-bounds : %d / %d  (flip_x=%s)", ef_in,   ef_total,   flip_x)
    log.info("Coil in-bounds : %d / %d  (flip_x=%s)", coil_in, coil_total, flip_x)
    if ef_in < ef_total:
        log.warning("EF: %d coordinate(s) outside image volume", ef_total - ef_in)
    if coil_in < coil_total:
        log.warning("Coil: %d coordinate(s) outside image volume", coil_total - coil_in)

    return dict(
        flip_x       = flip_x,
        lm_vox       = lm_vox,
        lm_mm        = lm_mm,
        ef_vox       = ef_vox,
        ef_mm        = ef_mm,
        ef_mask      = ef_mask,
        coil_vox     = coil_vox,
        coil_mm      = coil_mm,
        coil_mask    = coil_mask,
        hemi_correct = hemi_ok,
        ef_bounds    = (ef_in, ef_total),
        coil_bounds  = (coil_in, coil_total),
    )


def _physical_to_ras(coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Reorient Nx3 coordinates from native scanner physical space to RAS.

    voxels_to_mm() returns coordinates in the scanner's physical space
    (e.g. PIR, LAS), not necessarily RAS. ANTs needs LPS, which is derived
    from RAS by negating X and Y — so we must get to RAS first.
    """
    current_ornt = ornt.io_orientation(affine)
    ras_ornt     = np.array([[0, 1], [1, 1], [2, 1]])
    transform    = ornt.ornt_transform(current_ornt, ras_ornt)

    result = coords.copy().astype(float)
    for out_ax, (in_ax, flip) in enumerate(transform):
        result[:, int(out_ax)] = coords[:, int(in_ax)] * flip
    return result


def _apply_ants_transform(
    mm_native:  np.ndarray,
    mask:       np.ndarray,
    paths:      PathManifest,
    affine:     np.ndarray,
) -> np.ndarray:
    """
    Transform Nx3 native physical mm coordinates to MNI RAS mm via
    antsApplyTransformsToPoints.

    Pipeline:
      native physical → RAS  (orientation-aware reorient)
      RAS → LPS              (negate X and Y — what ANTs expects)
      ANTs transform
      LPS → RAS              (negate X and Y on output)
    """
    result = np.full(mm_native.shape, np.nan)
    if not mask.any():
        return result

    # Native physical → RAS
    valid_ras = _physical_to_ras(mm_native[mask], affine)

    # RAS → LPS for ANTs
    valid_lps = valid_ras.copy()
    valid_lps[:, 0] *= -1
    valid_lps[:, 1] *= -1

    with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False, dir="/tmp") as f:
        pts_in = f.name
        f.write("x,y,z,t\n")
        for row in valid_lps:
            f.write(f"{row[0]},{row[1]},{row[2]},0\n")

    pts_out = pts_in.replace(".csv", "_out.csv")

    cmd = [
        "antsApplyTransformsToPoints",
        "-d", "3",
        "-i", pts_in,
        "-o", pts_out,
        "-t", f"[{paths.affine_mat},1]",
        "-t", str(paths.inv_warp),
    ]
    log.debug("antsApplyTransformsToPoints: %s", " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("antsApplyTransformsToPoints failed (exit code %d):", res.returncode)
        log.error("stderr:\n%s", res.stderr)
        raise RuntimeError("ANTs point transform failed — see log for details.")

    out_lps = pd.read_csv(pts_out)[["x", "y", "z"]].values

    # LPS → RAS
    out_ras = out_lps.copy()
    out_ras[:, 0] *= -1
    out_ras[:, 1] *= -1

    result[mask] = out_ras

    os.unlink(pts_in)
    os.unlink(pts_out)
    return result