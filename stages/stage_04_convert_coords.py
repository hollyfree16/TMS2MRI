"""
stages/04_convert_coords.py
===========================
Stage 04: Convert NextStim NBE coordinates to:
  - Subject native MRI voxel indices + mm  (targets_native.csv)
  - MNI152 mm coordinates via ANTs point transform  (targets_mni.csv)

Uses utils/affine.py for all transform math.
Uses antsApplyTransformsToPoints for the native→MNI warp.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

from utils.affine import (
    nextstim_to_mri_voxels,
    voxel_sizes_from_affine,
    voxels_to_mm,
    mm_to_voxels,
    ras_to_lps,
    lps_to_ras,
    label_hemisphere,
    hemisphere_check,
)
from utils.checks import bounds_check, check_voxel_sizes
from utils.io import PathManifest, read_targets, read_landmarks, write_csv
from utils.logger import get_stage_logger

import pandas as pd

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

    dim_x              = shape[0]
    x_midpoint         = dim_x / 2.0
    t1_ltr             = bool(affine[0, 0] > 0)   # True = RAS (left-to-right)

    log.info("T1 dimensions  : %s", shape)
    log.info("Voxel sizes    : %.4f x %.4f x %.4f mm", vox_x, vox_y, vox_z)
    log.info("X orientation  : %s", "Left-to-Right (RAS)" if t1_ltr else "Right-to-Left (LAS)")

    flip_x = not args.no_flip
    log.info("X flip         : %s", "ON" if flip_x else "OFF (--no-flip)")

    # ------------------------------------------------------------------ #
    # Load input CSVs
    # ------------------------------------------------------------------ #
    tgt_df = read_targets(paths.targets_csv)
    lm_df  = read_landmarks(paths.landmarks_csv)
    log.info("Targets        : %d rows", len(tgt_df))
    log.info("Landmarks      : %d rows", len(lm_df))

    # ------------------------------------------------------------------ #
    # Convert landmarks (for hemisphere check + MNI reporting)
    # ------------------------------------------------------------------ #
    lm_coords = lm_df[["x", "y", "z"]].values.astype(float)
    lm_vox    = nextstim_to_mri_voxels(lm_coords, dim_x, vox_x, vox_y, vox_z, flip_x)
    lm_mm     = voxels_to_mm(lm_vox, affine)

    log.info("Landmark positions (native voxels):")
    for i, row in lm_df.iterrows():
        v    = lm_vox[i]
        finite = all(np.isfinite(v[d]) for d in range(3))
        if not finite:
            log.warning("  %-20s  NON-FINITE ⚠️", row["landmark_type"])
            continue
        in_b = all(0 <= int(round(v[d])) < shape[d] for d in range(3))
        log.info("  %-20s  [%5.0f %5.0f %5.0f]  %s",
                 row["landmark_type"], v[0], v[1], v[2],
                 "OK" if in_b else "OUT OF BOUNDS ⚠️")

    # Hemisphere check
    lm_labels = lm_df["landmark_type"].tolist()
    correct   = hemisphere_check(lm_vox, lm_labels, x_midpoint, t1_ltr)
    if correct is True:
        log.info("Hemisphere check : ✅ passed")
    elif correct is False:
        log.warning("Hemisphere check : ⚠️  FAILED — try toggling --no-flip")
    else:
        log.info("Hemisphere check : skipped (landmarks missing)")

    # ------------------------------------------------------------------ #
    # Convert target / coil coordinates → native voxels + mm
    # ------------------------------------------------------------------ #
    def _convert(df, x_col, y_col, z_col):
        coords = df[[x_col, y_col, z_col]].values.astype(float)
        mask   = ~np.isnan(coords).any(axis=1)
        vox    = np.full(coords.shape, np.nan)
        if mask.any():
            vox[mask] = nextstim_to_mri_voxels(
                coords[mask], dim_x, vox_x, vox_y, vox_z, flip_x
            )
        mm = np.full(coords.shape, np.nan)
        if mask.any():
            mm[mask] = voxels_to_mm(vox[mask], affine)
        return vox, mm, mask

    ef_vox,   ef_mm,   ef_mask   = _convert(tgt_df, TARGET_X, TARGET_Y, TARGET_Z)
    coil_vox, coil_mm, coil_mask = _convert(tgt_df, COIL_X,   COIL_Y,   COIL_Z)

    # Bounds check
    for label, vox, mask in [("EF",   ef_vox,   ef_mask),
                              ("Coil", coil_vox, coil_mask)]:
        n_in, n_total = bounds_check(vox, shape)
        log.info("%s in-bounds : %d / %d", label, n_in, n_total)
        if n_in < n_total:
            log.warning("%s: %d coordinate(s) outside image volume", label, n_total - n_in)

    # Hemisphere labels
    ef_hemi   = label_hemisphere(ef_vox[:,   0], x_midpoint, t1_ltr)
    coil_hemi = label_hemisphere(coil_vox[:, 0], x_midpoint, t1_ltr)

    # ------------------------------------------------------------------ #
    # Build and save native CSV
    # ------------------------------------------------------------------ #
    native_df = tgt_df.copy()
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        native_df[f"ef_native_vox_{axis}"]   = np.where(np.isnan(ef_vox[:,   idx]), np.nan, np.round(ef_vox[:,   idx]))
        native_df[f"ef_native_mm_{axis}"]    = ef_mm[:,   idx]
        native_df[f"coil_native_vox_{axis}"] = np.where(np.isnan(coil_vox[:, idx]), np.nan, np.round(coil_vox[:, idx]))
        native_df[f"coil_native_mm_{axis}"]  = coil_mm[:, idx]

    native_df["ef_hemisphere"]   = ef_hemi
    native_df["coil_hemisphere"] = coil_hemi

    paths.coords_dir.mkdir(parents=True, exist_ok=True)
    write_csv(native_df, paths.targets_native)
    log.info("Native CSV : %s", paths.targets_native)

    # ------------------------------------------------------------------ #
    # MNI coordinate transform (if registration was run)
    # ------------------------------------------------------------------ #
    if paths.targets_mni is None:
        log.info("No MNI template configured — skipping MNI coordinate transform.")
        return

    if not paths.affine_mat.exists() or not paths.inv_warp.exists():
        log.error("Registration transforms not found — run stage 03 first.")
        log.error("  affine : %s", paths.affine_mat)
        log.error("  inv warp: %s", paths.inv_warp)
        raise RuntimeError("Missing ANTs transforms for MNI coordinate warp.")

    mni_img     = nib.load(str(paths.mni_template))
    mni_inv_aff = np.linalg.inv(mni_img.affine)

    ef_mni_mm   = _apply_ants_transform(ef_mm,   ef_mask,   paths)
    coil_mni_mm = _apply_ants_transform(coil_mm, coil_mask, paths)
    lm_mni_mm   = _apply_ants_transform(lm_mm,   np.ones(len(lm_mm), dtype=bool), paths)

    ef_mni_vox   = np.full(ef_mni_mm.shape,   np.nan)
    coil_mni_vox = np.full(coil_mni_mm.shape, np.nan)

    ef_mni_vox[ef_mask]     = mm_to_voxels(ef_mni_mm[ef_mask],     mni_img.affine)
    coil_mni_vox[coil_mask] = mm_to_voxels(coil_mni_mm[coil_mask], mni_img.affine)

    log.info("Landmark MNI mm (RAS):")
    for i, row in lm_df.iterrows():
        m = lm_mni_mm[i]
        log.info("  %-20s  [%6.1f %6.1f %6.1f]", row["landmark_type"], m[0], m[1], m[2])

    # Build and save MNI CSV
    mni_df = tgt_df.copy()
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        mni_df[f"ef_mni_mm_{axis}"]    = ef_mni_mm[:,   idx]
        mni_df[f"ef_mni_vox_{axis}"]   = np.where(np.isnan(ef_mni_vox[:,   idx]), np.nan, np.round(ef_mni_vox[:,   idx]))
        mni_df[f"coil_mni_mm_{axis}"]  = coil_mni_mm[:, idx]
        mni_df[f"coil_mni_vox_{axis}"] = np.where(np.isnan(coil_mni_vox[:, idx]), np.nan, np.round(coil_mni_vox[:, idx]))

    mni_df["ef_hemisphere"]   = ef_hemi
    mni_df["coil_hemisphere"] = coil_hemi

    write_csv(mni_df, paths.targets_mni)
    log.info("MNI CSV    : %s", paths.targets_mni)


def _apply_ants_transform(
    mm_native_ras: np.ndarray,
    mask:          np.ndarray,
    paths:         PathManifest,
) -> np.ndarray:
    """
    Transform Nx3 native RAS mm coordinates to MNI RAS mm coordinates
    using antsApplyTransformsToPoints.

    ANTs uses LPS internally, so we convert RAS→LPS before writing the
    temp CSV and LPS→RAS after reading the output.
    """
    result = np.full(mm_native_ras.shape, np.nan)
    if not mask.any():
        return result

    valid_ras = mm_native_ras[mask]
    valid_lps = ras_to_lps(valid_ras)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                     delete=False, dir="/tmp") as f:
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

    import pandas as pd
    out_lps = pd.read_csv(pts_out)[["x", "y", "z"]].values
    result[mask] = lps_to_ras(out_lps)

    os.unlink(pts_in)
    os.unlink(pts_out)

    return result
