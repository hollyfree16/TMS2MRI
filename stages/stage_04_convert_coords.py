"""
stages/stage_04_convert_coords.py
===================================
Stage 04: Convert NextStim NBE coordinates to:
  - Subject native MRI voxel indices + mm  (targets_native.csv)
  - MNI152 mm coordinates via ANTs point transform  (targets_mni.csv)
  - fsaverage pial surface snapped coordinates  (targets_fsaverage.csv)
  - targets_summary.csv  (filtered rows, when --id flags provided)

Automatic flip detection
------------------------
Two-stage auto-flip system:

  Stage 1 (pre-ANTs): voxel-space hemisphere check using landmark voxel
  positions. Reliable for RAS images but can fail for non-RAS orientations
  where voxel axis 0 is not the R/L axis.

  Stage 2 (post-ANTs): MNI landmark sanity check. After transforming
  landmarks to MNI space, verifies:
    - Nasion X is near midline (|X| < 25mm)
    - Nasion Y is anterior (Y > 50mm)
    - Left ear X is negative, right ear X is positive
  If this fails, automatically retries with the opposite flip and accepts
  the result if MNI landmarks become plausible. This catches flip errors
  that slip through the voxel-space check (e.g. some LAS images).

Orientation-aware NBE→voxel conversion
---------------------------------------
nextstim_to_mri_voxels() reads the image orientation codes from the affine
and assigns NBE axes accordingly. Validated for RAS, PIR, PIL, IPR, LAS.

ANTs point transform
---------------------
The NIfTI affine always maps voxels to RAS mm by definition, so
voxels_to_mm() already returns RAS mm for any orientation. We simply
negate X and Y to convert RAS→LPS for ANTs.

--id ordering
-------------
Summary CSV rows are output in --id flag order, not NBE file order.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import nibabel.orientations as ornt
import numpy as np
import nibabel as nib
import pandas as pd

from utils.atlas import label_coords_all, ATLAS_KEYS, ATLAS_COL_NAMES
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

DEFAULT_MESH = "fsaverage"


# =============================================================================
# MNI landmark sanity check
# =============================================================================

def _check_mni_landmarks(
    lm_mni_mm: np.ndarray,
    lm_df:     pd.DataFrame,
) -> bool:
    """
    Check whether landmark MNI coordinates are anatomically plausible.

    Criteria:
      - Nasion: |X| < 25mm (near midline), Y > 50mm (anterior)
      - Left ear: X < 0 (left hemisphere in RAS)
      - Right ear: X > 0 (right hemisphere in RAS)

    Returns True if all available landmarks pass, False if any fail.
    If no recognisable landmarks are found, returns True (no check possible).

    Landmarks whose MNI coordinates are physically impossible (|X| > 120mm)
    are skipped — this occurs when the landmark falls outside the valid region
    of the ANTs warp (e.g. ear landmarks in subjects with oversized FOV images
    where the ears are far from the brain). In that case the check falls back
    to the voxel-space hemisphere check result from stage 1.
    """
    # MNI brain is ~±80mm in X — anything beyond ±120mm is warp extrapolation
    IMPOSSIBLE_X = 120.0

    labels = lm_df["landmark_type"].str.lower().tolist()

    nasion_idx = next((i for i, l in enumerate(labels)
                       if "nasion" in l or "nose" in l), None)
    left_idx   = next((i for i, l in enumerate(labels)
                       if "left" in l and "ear" in l), None)
    right_idx  = next((i for i, l in enumerate(labels)
                       if "right" in l and "ear" in l), None)

    checks = []

    if nasion_idx is not None:
        n = lm_mni_mm[nasion_idx]
        if np.any(np.isnan(n)):
            pass
        elif abs(n[0]) > IMPOSSIBLE_X:
            log.warning(
                "MNI check: nasion X=%.1f is outside valid warp region — "
                "skipping nasion check (large FOV image?)", n[0])
        else:
            nasion_x_ok = abs(n[0]) < 25
            nasion_y_ok = n[1] > 50
            if not nasion_x_ok:
                log.debug("MNI check: nasion X=%.1f is not near midline", n[0])
            if not nasion_y_ok:
                log.debug("MNI check: nasion Y=%.1f is not anterior", n[1])
            checks.extend([nasion_x_ok, nasion_y_ok])

    if left_idx is not None and right_idx is not None:
        l = lm_mni_mm[left_idx]
        r = lm_mni_mm[right_idx]
        if np.any(np.isnan(l)) or np.any(np.isnan(r)):
            pass
        elif abs(l[0]) > IMPOSSIBLE_X or abs(r[0]) > IMPOSSIBLE_X:
            log.warning(
                "MNI check: ear landmarks outside valid warp region "
                "(left X=%.1f, right X=%.1f) — skipping ear check "
                "(large FOV image?)", l[0], r[0])
        else:
            hemi_ok = l[0] < 0 and r[0] > 0
            if not hemi_ok:
                log.debug(
                    "MNI check: left ear X=%.1f (should be <0), "
                    "right ear X=%.1f (should be >0)", l[0], r[0])
            checks.append(hemi_ok)

    if not checks:
        log.debug("MNI check: no usable landmarks — skipping check")
        return True

    return all(checks)


# =============================================================================
# Stage entry point
# =============================================================================

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

    current_ornt = ornt.io_orientation(affine)
    ornt_codes   = ornt.ornt2axcodes(current_ornt)

    log.info("T1 dimensions  : %s", shape)
    log.info("Voxel sizes    : %.4f x %.4f x %.4f mm", vox_x, vox_y, vox_z)
    log.info("Orientation    : %s", "".join(ornt_codes))
    log.info("X orientation  : %s",
             "Left-to-Right (RAS-like)" if t1_ltr else "Right-to-Left (LAS-like)")

    if "".join(ornt_codes) != "RAS":
        log.info("Non-RAS orientation detected — using orientation-aware "
                 "NBE→voxel mapping (validated for RAS, PIR, PIL, IPR, LAS).")

    tgt_df = read_targets(paths.targets_csv)
    lm_df  = read_landmarks(paths.landmarks_csv)
    log.info("Targets        : %d rows", len(tgt_df))
    log.info("Landmarks      : %d rows", len(lm_df))

    # ------------------------------------------------------------------ #
    # Stage 1 flip check: voxel-space hemisphere check
    # ------------------------------------------------------------------ #
    flip_x = not args.no_flip
    log.info("X flip         : %s (initial)", "ON" if flip_x else "OFF")

    result = _attempt(tgt_df, lm_df, flip_x,
                      affine, shape, x_midpoint, t1_ltr)

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
                "Keeping flip_x=%s.", flip_x)

    # ------------------------------------------------------------------ #
    # Unpack
    # ------------------------------------------------------------------ #
    ef_vox    = result["ef_vox"]
    ef_mm     = result["ef_mm"]
    ef_mask   = result["ef_mask"]
    coil_vox  = result["coil_vox"]
    coil_mm   = result["coil_mm"]
    coil_mask = result["coil_mask"]

    log.info("X flip used    : %s", "ON" if flip_x else "OFF")

    ef_hemi   = label_hemisphere(ef_vox[:,   0], x_midpoint, t1_ltr)
    coil_hemi = label_hemisphere(coil_vox[:, 0], x_midpoint, t1_ltr)

    # ------------------------------------------------------------------ #
    # Write targets_native.csv
    # ------------------------------------------------------------------ #
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
    log.info("Native CSV     : %s", paths.targets_native)

    # ------------------------------------------------------------------ #
    # MNI transform
    # ------------------------------------------------------------------ #
    if paths.targets_mni is None:
        log.info("No MNI template configured — skipping MNI + fsaverage transform.")
        _write_summary(native_df, None, None, args, paths)
        return

    if not paths.affine_mat.exists() or not paths.inv_warp.exists():
        log.error("Registration transforms not found — run stage 03 first.")
        raise RuntimeError("Missing ANTs transforms for MNI coordinate warp.")

    lm_mm   = result["lm_mm"]
    mni_img = nib.load(str(paths.mni_template))

    ef_mni_mm   = _apply_ants_transform(ef_mm,   ef_mask,   paths)
    coil_mni_mm = _apply_ants_transform(coil_mm, coil_mask, paths)
    lm_mni_mm   = _apply_ants_transform(
        lm_mm, np.ones(len(lm_mm), dtype=bool), paths)

    # ------------------------------------------------------------------ #
    # Stage 2 flip check: MNI landmark sanity check
    # ------------------------------------------------------------------ #
    mni_flip_ok = _check_mni_landmarks(lm_mni_mm, lm_df)

    if not mni_flip_ok:
        log.warning(
            "MNI landmark sanity check FAILED (flip_x=%s) — "
            "nasion or ear positions are anatomically implausible. "
            "Retrying with flip_x=%s.", flip_x, not flip_x)

        retry2       = _attempt(tgt_df, lm_df, not flip_x,
                                affine, shape, x_midpoint, t1_ltr)
        retry2_lm_mm = retry2["lm_mm"]
        retry2_lm_mni = _apply_ants_transform(
            retry2_lm_mm, np.ones(len(lm_df), dtype=bool), paths)
        retry2_ok = _check_mni_landmarks(retry2_lm_mni, lm_df)

        if retry2_ok:
            log.warning(
                "MNI auto-flip ACCEPTED: flip_x %s → %s  "
                "(MNI landmarks now anatomically plausible)",
                flip_x, not flip_x)
            flip_x      = not flip_x
            result      = retry2
            ef_mm       = retry2["ef_mm"]
            ef_mask     = retry2["ef_mask"]
            coil_mm     = retry2["coil_mm"]
            coil_mask   = retry2["coil_mask"]
            lm_mni_mm   = retry2_lm_mni
            ef_mni_mm   = _apply_ants_transform(ef_mm,   ef_mask,   paths)
            coil_mni_mm = _apply_ants_transform(coil_mm, coil_mask, paths)
            ef_hemi     = label_hemisphere(
                retry2["ef_vox"][:,   0], x_midpoint, t1_ltr)
            coil_hemi   = label_hemisphere(
                retry2["coil_vox"][:, 0], x_midpoint, t1_ltr)
            # Update native CSV with corrected flip
            native_df["ef_hemisphere"]   = ef_hemi
            native_df["coil_hemisphere"] = coil_hemi
            native_df["flip_x_used"]     = flip_x
            for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
                native_df[f"ef_native_vox_{axis}"] = np.where(
                    np.isnan(retry2["ef_vox"][:, idx]), np.nan,
                    np.round(retry2["ef_vox"][:, idx]))
                native_df[f"ef_native_mm_{axis}"]  = retry2["ef_mm"][:, idx]
            write_csv(native_df, paths.targets_native)
            log.info("Native CSV updated with corrected flip.")
        else:
            log.warning(
                "MNI auto-flip REJECTED: neither flip produces plausible MNI "
                "landmarks. Keeping flip_x=%s. This may indicate a registration "
                "failure — check T1_in_MNI.nii.gz in FreeView.", flip_x)

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
    log.info("MNI CSV        : %s", paths.targets_mni)

    # ------------------------------------------------------------------ #
    # fsaverage surface snapping
    # ------------------------------------------------------------------ #
    fs_df = None
    if paths.targets_fsaverage is not None and not getattr(args, "skip_snap", False):
        fs_df = _snap_to_fsaverage(mni_df, paths, args)
    else:
        log.info("fsaverage snapping skipped (--skip-snap or no MNI template).")

    # ------------------------------------------------------------------ #
    # Summary CSV
    # ------------------------------------------------------------------ #
    _write_summary(native_df, mni_df, fs_df, args, paths)


# =============================================================================
# Summary CSV writer
# =============================================================================

def _norm_id(s) -> str:
    return str(s).strip().rstrip(".")


def _filter_df(df: pd.DataFrame, ids: list[str] | None) -> pd.DataFrame:
    if ids is None:
        return df.iloc[0:0]

    use_all = len(ids) == 1 and ids[0].strip().lower() == "all"
    if use_all:
        return df.copy()

    id_set  = {_norm_id(i) for i in ids}
    mask    = df["id"].apply(_norm_id).isin(id_set)
    matched = df[mask].copy()

    id_order = {_norm_id(i): pos for pos, i in enumerate(ids)}
    matched["_sort_key"] = matched["id"].apply(
        lambda x: id_order.get(_norm_id(x), 999))
    matched = matched.sort_values("_sort_key").drop(columns="_sort_key")
    return matched.reset_index(drop=True)


def _write_summary(
    native_df: pd.DataFrame,
    mni_df:    pd.DataFrame | None,
    fs_df:     pd.DataFrame | None,
    args,
    paths:     PathManifest,
) -> None:
    if args.ids is None or paths.targets_summary is None:
        return

    filtered_native = _filter_df(native_df, args.ids)
    if filtered_native.empty:
        log.warning("Summary CSV: no rows matched the requested IDs — skipping.")
        return

    out = pd.DataFrame()
    out["subject_id"] = [getattr(args, "subject_id", None)] * len(filtered_native)
    out["site"]       = [getattr(args, "site",       None)] * len(filtered_native)
    out["id"]         = filtered_native["id"].values

    for ax in ("x", "y", "z"):
        out[f"ef_native_mm_{ax}"] = filtered_native[f"ef_native_mm_{ax}"].values
    out["ef_hemisphere"] = filtered_native["ef_hemisphere"].values

    mni_source = None
    if fs_df is not None:
        mni_source = _filter_df(fs_df, args.ids)
    elif mni_df is not None:
        mni_source = _filter_df(mni_df, args.ids)

    if mni_source is not None and not mni_source.empty:
        for ax in ("x", "y", "z"):
            out[f"ef_mni_mm_{ax}"] = mni_source[f"ef_mni_mm_{ax}"].values

        mni_x = mni_source["ef_mni_mm_x"].values
        out["mni_hemisphere"] = [
            ("L" if x < 0 else "R") if not np.isnan(float(x)) else None
            for x in mni_x
        ]

        coords     = mni_source[["ef_mni_mm_x", "ef_mni_mm_y",
                                  "ef_mni_mm_z"]].values
        valid_mask = ~np.isnan(coords).any(axis=1)

        all_labels: dict = {k: ["no_region"] * len(coords) for k in ATLAS_KEYS}
        if valid_mask.any():
            log.info("Labelling %d coordinate(s) against %d atlases...",
                     valid_mask.sum(), len(ATLAS_KEYS))
            valid_atlas = label_coords_all(coords[valid_mask])
            for key in ATLAS_KEYS:
                j = 0
                for i, v in enumerate(valid_mask):
                    if v:
                        all_labels[key][i] = valid_atlas[key][j]
                        j += 1

        for key in ATLAS_KEYS:
            out[ATLAS_COL_NAMES[key]] = all_labels[key]

        if "snap_distance_mm" in mni_source.columns:
            for ax in ("x", "y", "z"):
                out[f"fs_{ax}"] = mni_source[f"fs_{ax}"].values
            out["snap_distance_mm"] = mni_source["snap_distance_mm"].values

    write_csv(out, paths.targets_summary)
    log.info("Summary CSV    : %s  (%d rows, %d atlas columns)",
             paths.targets_summary, len(out), len(ATLAS_KEYS))


# =============================================================================
# fsaverage surface snapping
# =============================================================================

def _snap_to_fsaverage(
    mni_df: pd.DataFrame,
    paths:  PathManifest,
    args,
) -> pd.DataFrame:
    mesh = getattr(args, "fsaverage_mesh", DEFAULT_MESH)
    log.info("Loading fsaverage surfaces (mesh=%s)...", mesh)

    from nilearn import datasets, surface as surf

    fsaverage    = datasets.fetch_surf_fsaverage(mesh)
    lh_coords, _ = surf.load_surf_mesh(fsaverage["pial_left"])
    rh_coords, _ = surf.load_surf_mesh(fsaverage["pial_right"])

    log.info("Left  pial : %d vertices", len(lh_coords))
    log.info("Right pial : %d vertices", len(rh_coords))

    results = []
    for _, row in mni_df.iterrows():
        x = row.get("ef_mni_mm_x")
        y = row.get("ef_mni_mm_y")
        z = row.get("ef_mni_mm_z")

        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            results.append(_nan_snap(mesh))
            continue

        pt = np.array([x, y, z], dtype=float)

        if x < 0:
            primary_coords  = lh_coords
            primary_label   = f"{mesh}_pial_left"
            fallback_coords = rh_coords
            fallback_label  = f"{mesh}_pial_right"
        else:
            primary_coords  = rh_coords
            primary_label   = f"{mesh}_pial_right"
            fallback_coords = lh_coords
            fallback_label  = f"{mesh}_pial_left"

        p_idx, p_dist = _nearest_vertex(pt, primary_coords)
        f_idx, f_dist = _nearest_vertex(pt, fallback_coords)

        if p_dist <= f_dist:
            vertex_idx   = p_idx
            surface_name = primary_label
            snapped      = primary_coords[p_idx]
            dist         = p_dist
        else:
            vertex_idx   = f_idx
            surface_name = fallback_label
            snapped      = fallback_coords[f_idx]
            dist         = f_dist

        results.append({
            "fs_vertex":        int(vertex_idx),
            "fs_surface":       surface_name,
            "fs_x":             float(snapped[0]),
            "fs_y":             float(snapped[1]),
            "fs_z":             float(snapped[2]),
            "snap_distance_mm": round(float(dist), 3),
            "fsaverage_mesh":   mesh,
        })

    snap_df   = pd.DataFrame(results)
    output_df = pd.concat([mni_df.reset_index(drop=True), snap_df], axis=1)

    write_csv(output_df, paths.targets_fsaverage)
    log.info("fsaverage CSV  : %s", paths.targets_fsaverage)

    valid = snap_df["snap_distance_mm"].dropna()
    if len(valid):
        log.info("Snap distance summary (mm):")
        log.info("  Mean   : %.2f", valid.mean())
        log.info("  Median : %.2f", valid.median())
        log.info("  Max    : %.2f", valid.max())
        log.info("  Min    : %.2f", valid.min())
        n_large = (valid > 20).sum()
        if n_large:
            log.warning(
                "%d point(s) snapped > 20 mm — may be subcortical or outside cortex.",
                n_large)

    for idx, srow in snap_df.iterrows():
        if not isinstance(srow["fs_surface"], str):
            continue
        mni_x = mni_df.iloc[idx].get("ef_mni_mm_x", 0)
        expected = "left" if mni_x < 0 else "right"
        if not srow["fs_surface"].endswith(expected):
            log.warning(
                "  id=%s  mni_x=%.1f  snapped_to=%s  dist=%.1f mm",
                mni_df.iloc[idx].get("id", idx),
                mni_x,
                srow["fs_surface"],
                srow["snap_distance_mm"],
            )

    return output_df


def _nan_snap(mesh: str) -> dict:
    return {
        "fs_vertex":        pd.NA,
        "fs_surface":       pd.NA,
        "fs_x":             np.nan,
        "fs_y":             np.nan,
        "fs_z":             np.nan,
        "snap_distance_mm": np.nan,
        "fsaverage_mesh":   mesh,
    }


def _nearest_vertex(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[int, float]:
    diff = vertices - point
    dist = np.einsum("ij,ij->i", diff, diff)
    idx  = int(np.argmin(dist))
    return idx, float(np.sqrt(dist[idx]))


# =============================================================================
# NBE → native conversion
# =============================================================================

def _attempt(
    tgt_df, lm_df,
    flip_x: bool,
    affine, shape, x_midpoint, t1_ltr,
) -> dict:
    lm_coords = lm_df[["x", "y", "z"]].apply(
        pd.to_numeric, errors="coerce"
    ).values.astype(float)
    lm_vox = nextstim_to_mri_voxels(lm_coords, affine, shape, flip_x)
    lm_mm  = voxels_to_mm(lm_vox, affine)

    log.info("Landmark positions (flip_x=%s):", flip_x)
    for i, row in lm_df.iterrows():
        v    = lm_vox[i]
        if not all(np.isfinite(v[d]) for d in range(3)):
            log.warning("  %-20s  NON-FINITE", row["landmark_type"])
            continue
        in_b = all(0 <= int(round(v[d])) < shape[d] for d in range(3))
        log.info("  %-20s  [%5.0f %5.0f %5.0f]  %s",
                 row["landmark_type"], v[0], v[1], v[2],
                 "OK" if in_b else "OUT OF BOUNDS")

    lm_labels = lm_df["landmark_type"].tolist()
    hemi_ok   = hemisphere_check(lm_vox, lm_labels, x_midpoint, t1_ltr)
    if hemi_ok is True:
        log.info("Hemisphere check : passed (flip_x=%s)", flip_x)
    elif hemi_ok is False:
        log.warning("Hemisphere check : FAILED (flip_x=%s)", flip_x)
    else:
        log.info("Hemisphere check : skipped — landmarks missing (flip_x=%s)", flip_x)

    def _conv(x_col, y_col, z_col):
        coords = tgt_df[[x_col, y_col, z_col]].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        mask = ~np.isnan(coords).any(axis=1)
        vox  = np.full(coords.shape, np.nan)
        if mask.any():
            vox[mask] = nextstim_to_mri_voxels(coords[mask], affine, shape, flip_x)
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


# =============================================================================
# ANTs coordinate transform
# =============================================================================

def _apply_ants_transform(
    mm_ras: np.ndarray,
    mask:   np.ndarray,
    paths:  PathManifest,
) -> np.ndarray:
    """
    Transform Nx3 RAS mm → MNI RAS mm via antsApplyTransformsToPoints.
    Negates X and Y for LPS, then negates back after.
    """
    result = np.full(mm_ras.shape, np.nan)
    if not mask.any():
        return result

    valid_lps = mm_ras[mask].copy()
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
    out_ras = out_lps.copy()
    out_ras[:, 0] *= -1
    out_ras[:, 1] *= -1

    result[mask] = out_ras

    os.unlink(pts_in)
    os.unlink(pts_out)
    return result