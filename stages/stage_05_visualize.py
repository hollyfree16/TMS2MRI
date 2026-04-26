"""
stages/stage_05_visualize.py
=============================
Stage 05: Plot selected stimulation sites on a nilearn glass brain,
generate atlas overlay plots for all configured atlases, and optionally
append MNI coordinates to a shared cross-subject CSV.

Outputs (plotted rows selected by --id):
  stimulation_sites.html  — interactive 3D glass brain
  stimulation_sites.png   — static 4-panel (L/coronal/axial/R)
  ho_regions.png          — Harvard-Oxford atlas overlay
  aal_regions.png         — AAL atlas overlay
  destrieux_regions.png   — Destrieux atlas overlay
  schaefer_*_regions.png  — Schaefer atlas overlays (100-1000 parcels)
  yeo_7_regions.png       — Yeo 7-network overlay
  yeo_17_regions.png      — Yeo 17-network overlay
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.atlas import plot_all_atlases, ATLAS_KEYS
from utils.io import PathManifest, write_csv
from utils.logger import get_stage_logger

log = get_stage_logger("visualize")


def run(args, paths: PathManifest) -> None:
    if args.ids is None:
        log.info("No --id flags provided — skipping visualization.")
        return

    # ------------------------------------------------------------------ #
    # Select coordinate source
    # ------------------------------------------------------------------ #
    if paths.targets_mni and paths.targets_mni.exists():
        csv_path  = paths.targets_mni
        mm_x_col  = "ef_mni_mm_x" if args.coord_col == "ef" else "coil_mni_mm_x"
        mm_y_col  = "ef_mni_mm_y" if args.coord_col == "ef" else "coil_mni_mm_y"
        mm_z_col  = "ef_mni_mm_z" if args.coord_col == "ef" else "coil_mni_mm_z"
        space     = "MNI"
        is_mni    = True
    else:
        csv_path  = paths.targets_native
        mm_x_col  = "ef_native_mm_x" if args.coord_col == "ef" else "coil_native_mm_x"
        mm_y_col  = "ef_native_mm_y" if args.coord_col == "ef" else "coil_native_mm_y"
        mm_z_col  = "ef_native_mm_z" if args.coord_col == "ef" else "coil_native_mm_z"
        space     = "native"
        is_mni    = False

    log.info("Coordinate CSV : %s  (%s space)", csv_path, space)

    df = pd.read_csv(csv_path, na_values=["-", " -", "- "])

    def _norm(s):
        return str(s).strip().rstrip(".")

    df["_id_norm"] = df["id"].apply(_norm)

    if len(args.ids) == 1 and args.ids[0].strip().lower() == "all":
        selected = df.dropna(subset=[mm_x_col, mm_y_col, mm_z_col])
        log.info("Plotting all %d points", len(selected))
    else:
        id_norm  = [_norm(x) for x in args.ids]
        selected = df[df["_id_norm"].isin(id_norm)].dropna(
                       subset=[mm_x_col, mm_y_col, mm_z_col])
        missing  = [i for i in id_norm if i not in df["_id_norm"].values]
        if missing:
            log.warning("IDs not found in CSV: %s", missing)
        log.info("Selected %d points from %d requested IDs",
                 len(selected), len(args.ids))

    if selected.empty:
        log.error("No valid coordinates to plot after filtering NaNs.")
        raise RuntimeError("No coordinates to plot.")

    coords = selected[[mm_x_col, mm_y_col, mm_z_col]].values

    if is_mni:
        hemi_labels = ["L" if x < 0 else "R" for x in coords[:, 0]]
    else:
        hemi_col    = "ef_hemisphere"
        hemi_labels = list(selected[hemi_col])

    n_left  = hemi_labels.count("L")
    n_right = hemi_labels.count("R")
    log.info("Left  hemisphere: %d points", n_left)
    log.info("Right hemisphere: %d points", n_right)

    if args.color_by_hemi:
        colors = ["#4488FF" if h == "L" else "#FF4444" for h in hemi_labels]
        log.info("Colour mode: by hemisphere  L=#4488FF  R=#FF4444")
    else:
        colors = "#FF4444"
        log.info("Colour mode: uniform red")

    paths.viz_dir.mkdir(parents=True, exist_ok=True)

    from nilearn import plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------ #
    # Interactive HTML
    # ------------------------------------------------------------------ #
    log.info("Building interactive glass brain...")
    view = plotting.view_markers(
        coords,
        marker_color=colors,
        marker_size=args.marker_size,
    )
    view.save_as_html(str(paths.html_out))
    log.info("HTML saved : %s", paths.html_out)

    # ------------------------------------------------------------------ #
    # Static 4-panel PNG
    # ------------------------------------------------------------------ #
    log.info("Building static glass brain PNG...")
    fig = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=None, annotate=True, black_bg=False,
    )

    hemi_arr  = np.array(hemi_labels)
    left_pts  = coords[hemi_arr == "L"]
    right_pts = coords[hemi_arr == "R"]

    if args.color_by_hemi:
        if len(left_pts):
            disp.add_markers(left_pts,  marker_color="#4488FF",
                             marker_size=args.marker_size * 3)
        if len(right_pts):
            disp.add_markers(right_pts, marker_color="#FF4444",
                             marker_size=args.marker_size * 3)
        patches = [
            mpatches.Patch(color="#4488FF", label="Left"),
            mpatches.Patch(color="#FF4444", label="Right"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=2,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    else:
        disp.add_markers(coords, marker_color="#FF4444",
                         marker_size=args.marker_size * 3)

    plt.savefig(str(paths.png_out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("PNG  saved : %s", paths.png_out)

    # ------------------------------------------------------------------ #
    # Atlas overlay plots (MNI only)
    # ------------------------------------------------------------------ #
    if is_mni:
        log.info("Building atlas overlay plots (%d atlases)...", len(ATLAS_KEYS))
        color_left  = "#4488FF" if args.color_by_hemi else "#FF4444"
        color_right = "#FF4444"

        out_paths = plot_all_atlases(
            coords_mm   = coords,
            hemi_labels = hemi_labels,
            out_dir     = str(paths.viz_dir),
            color_left  = color_left,
            color_right = color_right,
            marker_size = args.marker_size,
        )
        for key, path in out_paths.items():
            log.info("Atlas plot (%s) : %s", key, path)
    else:
        log.info("Skipping atlas plots (native space — no MNI coordinates).")

    # ------------------------------------------------------------------ #
    # Shared CSV append
    # ------------------------------------------------------------------ #
    if paths.shared_csv is not None:
        if args.subject_id is None or args.site is None:
            log.warning("--shared-csv requires --subject-id and --site — skipping.")
        elif not is_mni:
            log.warning("--shared-csv requires MNI coordinates — skipping (native space).")
        else:
            _append_shared_csv(selected, mm_x_col, mm_y_col, mm_z_col,
                               hemi_labels, args, paths)


def _append_shared_csv(
    selected,
    mm_x_col, mm_y_col, mm_z_col,
    hemi_labels: list[str],
    args,
    paths: PathManifest,
) -> None:
    # Re-sort selected to match --id flag order before appending
    def _norm(s):
        return str(s).strip().rstrip(".")

    if args.ids and not (len(args.ids) == 1 and args.ids[0].strip().lower() == "all"):
        id_order = {_norm(i): pos for pos, i in enumerate(args.ids)}
        selected = selected.copy()
        selected["_sort_key"] = selected["id"].apply(
            lambda x: id_order.get(_norm(x), 999))
        selected = selected.sort_values("_sort_key").drop(columns="_sort_key")
        selected = selected.reset_index(drop=True)
        # Rebuild hemi_labels to match new order
        hemi_labels = ["L" if selected.iloc[i][mm_x_col] < 0 else "R"
                       for i in range(len(selected))]

    rows = []
    for i, (_, row) in enumerate(selected.iterrows()):
        x = row[mm_x_col]
        rows.append({
            "subject_id":     args.subject_id,
            "site":           args.site,
            "id":             str(row["id"]).strip().rstrip("."),
            "mni_mm_x":       round(x, 2),
            "mni_mm_y":       round(row[mm_y_col], 2),
            "mni_mm_z":       round(row[mm_z_col], 2),
            "hemisphere":     row.get("ef_hemisphere", hemi_labels[i]),
            "mni_hemisphere": "L" if x < 0 else "R",
        })

    new_df = pd.DataFrame(rows)

    if paths.shared_csv.exists():
        existing = pd.read_csv(paths.shared_csv)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    paths.shared_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(paths.shared_csv, index=False)
    log.info("Shared CSV : %d rows appended → %s", len(rows), paths.shared_csv)