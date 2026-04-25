"""
stages/stage_04_convert_coords.py
===================================
Stage 04: Convert NextStim NBE coordinates to:
  - Subject native MRI voxel indices + mm  (targets_native.csv)
  - MNI152 mm coordinates via ANTs point transform  (targets_mni.csv)
  - fsaverage pial surface snapped coordinates  (targets_fsaverage.csv)

When --id flags are provided, also writes a merged summary CSV containing
only the requested rows with key columns from all three spaces:
  - targets_summary.csv

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

fsaverage surface snapping
---------------------------
MNI mm coordinates are snapped to the nearest fsaverage pial surface vertex.
Hemisphere is routed by the sign of MNI X (negative = left), with cross-
hemisphere fallback for near-midline points.  Snap distance is reported in
the output CSV for QC.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import nibabel.orientations as ornt
import numpy as np
import nibabel as nib
import pandas as pd

from utils.atlas import label_coords
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

    log.info("T1 dimensions  : %s", shape)
    log.info("Voxel sizes    : %.4f x %.4f x %.4f mm", vox_x, vox_y, vox_z)
    log.info("X orientation  : %s",
             "Left-to-Right (RAS)" if t1_ltr else "Right-to-Left (LAS)")

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
    # Write targets_native.csv  (full)
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

    lm_mm     = result["lm_mm"]
    lm_df     = read_landmarks(paths.landmarks_csv)
    mni_img   = nib.load(str(paths.mni_template))

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
    """Return rows matching the requested IDs, or all rows if --id all."""
    if ids is None:
        return df.iloc[0:0]   # empty — no IDs requested

    use_all = len(ids) == 1 and ids[0].strip().lower() == "all"
    if use_all:
        return df.copy()

    id_set = {_norm_id(i) for i in ids}
    mask   = df["id"].apply(_norm_id).isin(id_set)
    return df[mask].copy()


def _write_summary(
    native_df: pd.DataFrame,
    mni_df:    pd.DataFrame | None,
    fs_df:     pd.DataFrame | None,
    args,
    paths:     PathManifest,
) -> None:
    """Merge filtered rows from native, MNI, and fsaverage into one summary CSV."""
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

    # MNI + fsaverage: prefer fs_df (has mni cols + snap cols); fall back to mni_df
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

        coords = mni_source[["ef_mni_mm_x", "ef_mni_mm_y", "ef_mni_mm_z"]].values
        valid_mask = ~np.isnan(coords).any(axis=1)
        labels: list = [None] * len(coords)
        if valid_mask.any():
            valid_labels = label_coords(coords[valid_mask])
            j = 0
            for i, v in enumerate(valid_mask):
                if v:
                    labels[i] = valid_labels[j]
                    j += 1
        out["ho_region"] = labels

        if "snap_distance_mm" in mni_source.columns:
            for ax in ("x", "y", "z"):
                out[f"fs_{ax}"] = mni_source[f"fs_{ax}"].values
            out["snap_distance_mm"] = mni_source["snap_distance_mm"].values

    write_csv(out, paths.targets_summary)
    log.info("Summary CSV    : %s  (%d rows)", paths.targets_summary, len(out))


# =============================================================================
# fsaverage surface snapping  (moved from stage 06)
# =============================================================================

def _snap_to_fsaverage(
    mni_df: pd.DataFrame,
    paths:  PathManifest,
    args,
) -> pd.DataFrame:
    """
    Snap MNI152 EF coordinates to the nearest fsaverage pial vertex.

    Adds columns: fs_vertex, fs_surface, fs_x, fs_y, fs_z,
                  snap_distance_mm, fsaverage_mesh.

    Hemisphere routing uses the sign of ef_mni_mm_x (negative = left) —
    this is always correct in MNI RAS space and avoids the native-space
    mislabelling issue.
    """
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

        # Route by MNI X sign — always correct in RAS
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
            log.debug(
                "Row %s: cross-hemisphere snap (%.1f mm → %s, was %.1f mm → %s)",
                row.get("id", "?"), f_dist, fallback_label, p_dist, primary_label,
            )

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

    # QC summary
    valid   = snap_df["snap_distance_mm"].dropna()
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

    # Cross-hemisphere snap warnings
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
    """Vectorised nearest-vertex search. ~1 ms for 163k-vertex fsaverage."""
    diff = vertices - point
    dist = np.einsum("ij,ij->i", diff, diff)
    idx  = int(np.argmin(dist))
    return idx, float(np.sqrt(dist[idx]))


# =============================================================================
# NBE → native conversion helpers
# =============================================================================

def _attempt(
    tgt_df, lm_df,
    flip_x: bool,
    dim_x, vox_x, vox_y, vox_z,
    affine, shape, x_midpoint, t1_ltr,
) -> dict:
    """One full NBE → voxel → mm conversion attempt for a given flip_x."""
    lm_coords = lm_df[["x", "y", "z"]].apply(
        pd.to_numeric, errors="coerce"
    ).values.astype(float)
    lm_vox = nextstim_to_mri_voxels(
        lm_coords, dim_x, vox_x, vox_y, vox_z, flip_x)
    lm_mm  = voxels_to_mm(lm_vox, affine)

    log.info("Landmark positions (flip_x=%s):", flip_x)
    for i, row in lm_df.iterrows():
        v    = lm_vox[i]
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
        coords = tgt_df[[x_col, y_col, z_col]].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        mask = ~np.isnan(coords).any(axis=1)
        vox  = np.full(coords.shape, np.nan)
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


# =============================================================================
# ANTs coordinate transform helpers
# =============================================================================

def _physical_to_ras(coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Reorient Nx3 native physical coords to RAS for ANTs."""
    current_ornt = ornt.io_orientation(affine)
    ras_ornt     = np.array([[0, 1], [1, 1], [2, 1]])
    transform    = ornt.ornt_transform(current_ornt, ras_ornt)
    result       = coords.copy().astype(float)
    for out_ax, (in_ax, flip) in enumerate(transform):
        result[:, int(out_ax)] = coords[:, int(in_ax)] * flip
    return result


def _apply_ants_transform(
    mm_native: np.ndarray,
    mask:      np.ndarray,
    paths:     PathManifest,
    affine:    np.ndarray,
) -> np.ndarray:
    """
    Transform Nx3 native physical mm → MNI RAS mm via antsApplyTransformsToPoints.

    Pipeline:
      native physical → RAS  (orientation-aware)
      RAS → LPS              (negate X and Y)
      ANTs transform
      LPS → RAS              (negate X and Y)
    """
    result = np.full(mm_native.shape, np.nan)
    if not mask.any():
        return result

    valid_ras = _physical_to_ras(mm_native[mask], affine)
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
    out_ras = out_lps.copy()
    out_ras[:, 0] *= -1
    out_ras[:, 1] *= -1

    result[mask] = out_ras

    os.unlink(pts_in)
    os.unlink(pts_out)
    return result