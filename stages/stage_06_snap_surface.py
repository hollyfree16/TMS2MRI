"""
stages/stage_06_snap_surface.py
================================
Stage 06: Snap MNI152 stimulation coordinates to the nearest fsaverage
pial surface vertex for visualization and coordinate reporting.

Why pial?
---------
The pial surface is the outer cortical surface — the one rendered in
inflated brain visualizations.  Because this stage is primarily motivated
by visualization (points appear to float off the cortex when plotted on
a glass brain due to inter-subject head-size variation vs. MNI152 template),
pial is the natural target.  White matter surface snapping would shift
points ~2–3 mm deeper, which is misleading in an inflated-brain context.

Why fsaverage?
--------------
nilearn ships the fsaverage surface meshes directly via
``datasets.fetch_surf_fsaverage()``, making this dependency-free beyond
nilearn itself (already required).  The fsaverage ↔ MNI152 discrepancy
is on the order of a few mm — acceptable for visualization but worth
disclosing.  The ``snap_distance_mm`` column in the output CSV lets
downstream users quantify how much each point was moved.

Outputs
-------
targets_fsaverage.csv
    All columns from targets_mni.csv, plus:

    fs_vertex              int     Nearest fsaverage vertex index
    fs_surface             str     Surface used  (e.g. fsaverage_pial_left)
    fs_x, fs_y, fs_z       float   Snapped vertex RAS mm coordinates
    snap_distance_mm       float   Euclidean distance from MNI pt → vertex
    fsaverage_mesh         str     Mesh resolution used (e.g. fsaverage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.io import PathManifest
from utils.logger import get_stage_logger

log = get_stage_logger("snap_surface")

# Mesh resolution passed to fetch_surf_fsaverage.
# 'fsaverage'  → 163,842 vertices per hemisphere (~0.7 mm spacing) — highest fidelity
# 'fsaverage5' →  10,242 vertices per hemisphere (~3.5 mm spacing) — faster, lower res
# For publication-quality snapping use 'fsaverage'; for quick QC use 'fsaverage5'.
DEFAULT_MESH = "fsaverage"


def run(args, paths: PathManifest) -> None:
    """
    Snap MNI152 coordinates in targets_mni.csv to the nearest fsaverage
    pial surface vertex and write targets_fsaverage.csv.
    """
    if paths.targets_mni is None or not paths.targets_mni.exists():
        log.error("targets_mni.csv not found — stage 04 must complete with MNI "
                  "registration before stage 06 can run.")
        raise RuntimeError("Missing targets_mni.csv — run stages 03 and 04 first.")

    mesh = getattr(args, "fsaverage_mesh", DEFAULT_MESH)
    log.info("Loading fsaverage surfaces (mesh=%s)...", mesh)

    from nilearn import datasets, surface as surf

    fsaverage  = datasets.fetch_surf_fsaverage(mesh)
    lh_coords, _ = surf.load_surf_mesh(fsaverage["pial_left"])
    rh_coords, _ = surf.load_surf_mesh(fsaverage["pial_right"])

    log.info("Left  pial : %d vertices", len(lh_coords))
    log.info("Right pial : %d vertices", len(rh_coords))

    df = pd.read_csv(paths.targets_mni, na_values=["-", " -", "- "])
    log.info("Loaded %d rows from %s", len(df), paths.targets_mni)

    results = []
    for _, row in df.iterrows():
        x = row.get("ef_mni_mm_x")
        y = row.get("ef_mni_mm_y")
        z = row.get("ef_mni_mm_z")
        hemi = row.get("ef_hemisphere")

        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            results.append({
                "fs_vertex":        pd.NA,
                "fs_surface":       pd.NA,
                "fs_x":             np.nan,
                "fs_y":             np.nan,
                "fs_z":             np.nan,
                "snap_distance_mm": np.nan,
                "fsaverage_mesh":   mesh,
            })
            continue

        pt = np.array([x, y, z], dtype=float)

        # Route to hemisphere surface; fall back to closest across both if
        # hemisphere label is missing or ambiguous (e.g. near midline).
        if hemi == "L":
            primary_coords   = lh_coords
            primary_label    = f"{mesh}_pial_left"
            fallback_coords  = rh_coords
            fallback_label   = f"{mesh}_pial_right"
        else:
            primary_coords   = rh_coords
            primary_label    = f"{mesh}_pial_right"
            fallback_coords  = lh_coords
            fallback_label   = f"{mesh}_pial_left"

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
    output_df = pd.concat([df.reset_index(drop=True), snap_df], axis=1)

    paths.coords_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(paths.targets_fsaverage, index=False)
    log.info("fsaverage CSV : %s", paths.targets_fsaverage)

    # --- QC summary ---
    valid      = snap_df["snap_distance_mm"].dropna()
    n_cross    = snap_df["fs_surface"].str.endswith(
        ("_left", "_right")
    ).sum() if not snap_df.empty else 0

    if len(valid):
        log.info("Snap distance summary (mm):")
        log.info("  Mean   : %.2f", valid.mean())
        log.info("  Median : %.2f", valid.median())
        log.info("  Max    : %.2f", valid.max())
        log.info("  Min    : %.2f", valid.min())

        n_large = (valid > 20).sum()
        if n_large:
            log.warning(
                "%d point(s) snapped > 20 mm from original — "
                "these may be subcortical or outside cortex.", n_large
            )

    # Log any cross-hemisphere snaps explicitly
    cross = snap_df[
        snap_df.apply(
            lambda r: (
                isinstance(r["fs_surface"], str) and
                df.loc[r.name, "ef_hemisphere"] == "L" and
                r["fs_surface"].endswith("_right")
            ) or (
                isinstance(r["fs_surface"], str) and
                df.loc[r.name, "ef_hemisphere"] == "R" and
                r["fs_surface"].endswith("_left")
            ),
            axis=1,
        )
    ]
    if not cross.empty:
        log.warning(
            "%d point(s) snapped to the opposite hemisphere surface "
            "(nearest vertex was across midline):", len(cross)
        )
        for idx in cross.index:
            row_id = df.loc[idx, "id"] if "id" in df.columns else idx
            log.warning(
                "  id=%s  original_hemi=%s  snapped_to=%s  dist=%.1f mm",
                row_id,
                df.loc[idx, "ef_hemisphere"],
                cross.loc[idx, "fs_surface"],
                cross.loc[idx, "snap_distance_mm"],
            )


# =============================================================================
# Internal helpers
# =============================================================================

def _nearest_vertex(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[int, float]:
    """
    Return the index and distance of the closest vertex to ``point``.

    Uses vectorised numpy subtraction — fast even for 163k-vertex fsaverage.
    No KD-tree needed: on a modern CPU a single query over 163k vertices
    takes ~1 ms with numpy.

    Parameters
    ----------
    point:    shape (3,) — query point in mm.
    vertices: shape (N, 3) — surface vertex coordinates in mm.

    Returns
    -------
    (vertex_index, distance_mm)
    """
    diff = vertices - point          # (N, 3)
    dist = np.einsum("ij,ij->i", diff, diff)   # squared distances, no sqrt yet
    idx  = int(np.argmin(dist))
    return idx, float(np.sqrt(dist[idx]))