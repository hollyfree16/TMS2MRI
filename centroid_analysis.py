"""
centroid_analysis.py
====================
Group-level centroid analysis and target-coloured glass brain visualization.

Reads a group CSV that has been enriched with intended_area and intended_hemi
columns (added manually after group_visualize.py output).

Produces
--------
  {prefix}_centroids.csv
      Per-group centroid MNI coordinates and subject count.

  {prefix}_distances.csv
      Per-subject distance from their group centroid, with centroid coords
      appended for reference.

  {prefix}_target_glass_brain.png
      Static 4-panel glass brain coloured by intended target:
        L Premotor  → light blue  (#7EC8E3)
        R Premotor  → dark blue   (#0057A8)
        L Parietal  → light pink  (#FFB3C6)
        R Parietal  → red/pink    (#C1003C)

  {prefix}_target_glass_brain.html
      Interactive version of the above.

  {prefix}_target_centroids.png
      Same glass brain with centroid markers added (larger, outlined).

  {prefix}_by_site_{site}.png   (one per site if --by-site is set)
      Per-site glass brain using the same target colour scheme.

Usage
-----
  python centroid_analysis.py \\
      --csv group_viz/all_subjects_mni.csv \\
      --output-dir group_viz \\
      --output-prefix hc

  # Multiple CSVs (one per site — combined for centroid calc)
  python centroid_analysis.py \\
      --csv ou1/all_subjects_mni.csv \\
      --csv ou2/all_subjects_mni.csv \\
      --csv ou3/all_subjects_mni.csv \\
      --output-dir group_viz \\
      --output-prefix hc \\
      --by-site

  # Custom target/hemi column names if yours differ
  python centroid_analysis.py \\
      --csv all_subjects.csv \\
      --output-dir ./out \\
      --target-col intended_area \\
      --hemi-col   intended_hemi
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

# =============================================================================
# Colour scheme — fixed for all plots
# =============================================================================

# (target, hemi) → hex colour
TARGET_COLORS: dict[tuple[str, str], str] = {
    ("Premotor",  "L"): "#7EC8E3",   # light blue
    ("Premotor",  "R"): "#0057A8",   # dark blue
    ("Parietal",  "L"): "#FFB3C6",   # light pink
    ("Parietal",  "R"): "#C1003C",   # red/pink
}

# Fallback for unexpected values — assign grey
FALLBACK_COLOR = "#AAAAAA"

# Centroid marker style
CENTROID_MARKER_SIZE  = 80    # nilearn marker_size units
POINT_MARKER_SIZE     = 15


def _get_color(target: str, hemi: str) -> str:
    return TARGET_COLORS.get((str(target).strip(), str(hemi).strip()),
                              FALLBACK_COLOR)


def _legend_handles() -> list:
    handles = []
    labels  = {
        ("Premotor",  "L"): "L Premotor",
        ("Premotor",  "R"): "R Premotor",
        ("Parietal",  "L"): "L Parietal",
        ("Parietal",  "R"): "R Parietal",
    }
    for (target, hemi), label in labels.items():
        color = TARGET_COLORS[(target, hemi)]
        handles.append(mpatches.Patch(color=color, label=label))
    return handles


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centroid analysis and target-coloured glass brain plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, action="append", dest="csvs",
                   help="Group CSV with intended_area and intended_hemi columns. "
                        "Repeat for multiple sites.")
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--output-prefix", default="group")
    p.add_argument("--target-col",    default="intended_area",
                   help="Column name for intended target area (default: intended_area)")
    p.add_argument("--hemi-col",      default="intended_hemi",
                   help="Column name for intended hemisphere (default: intended_hemi)")
    p.add_argument("--by-site",       action="store_true",
                   help="Also produce one glass brain per site, "
                        "coloured by target within each site.")
    p.add_argument("--no-plots",      action="store_true",
                   help="Skip glass brain plots — output CSVs only.")
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def _load_csvs(
    csv_paths: list[str],
    target_col: str,
    hemi_col: str,
) -> pd.DataFrame:
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, na_values=["-", " -", "- "])
        for col in [target_col, hemi_col, "mni_mm_x", "mni_mm_y", "mni_mm_z"]:
            if col not in df.columns:
                print(f"ERROR: column '{col}' not found in {path}")
                print(f"  Available columns: {list(df.columns)}")
                sys.exit(1)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {path}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(
        subset=["mni_mm_x", "mni_mm_y", "mni_mm_z", target_col, hemi_col])

    print(f"Total: {len(combined)} rows with valid coordinates and target labels")
    return combined


# =============================================================================
# Centroid computation
# =============================================================================

def compute_centroids(
    df: pd.DataFrame,
    target_col: str,
    hemi_col: str,
) -> pd.DataFrame:
    """
    Compute mean MNI coordinate (centroid) for each (target, hemi) group.
    Returns a DataFrame with one row per group.
    """
    groups = df.groupby([target_col, hemi_col])

    rows = []
    for (target, hemi), grp in groups:
        coords = grp[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
        centroid = coords.mean(axis=0)
        rows.append({
            "intended_area":    target,
            "intended_hemi":    hemi,
            "n_subjects":       len(grp),
            "centroid_x":       round(float(centroid[0]), 3),
            "centroid_y":       round(float(centroid[1]), 3),
            "centroid_z":       round(float(centroid[2]), 3),
            # Standard deviation across subjects
            "std_x":            round(float(coords[:, 0].std()), 3),
            "std_y":            round(float(coords[:, 1].std()), 3),
            "std_z":            round(float(coords[:, 2].std()), 3),
            # Mean pairwise spread
            "spread_mm":        round(float(
                np.mean([np.linalg.norm(c - centroid) for c in coords])), 3),
        })
        print(f"  {target:12s} {hemi}  n={len(grp):3d}  "
              f"centroid=[{centroid[0]:6.1f}, {centroid[1]:6.1f}, {centroid[2]:6.1f}]")

    return pd.DataFrame(rows)


# =============================================================================
# Distance computation
# =============================================================================

def compute_distances(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    target_col: str,
    hemi_col: str,
) -> pd.DataFrame:
    """
    Compute Euclidean distance from each subject's actual stimulation point
    to their group centroid.
    """
    # Build centroid lookup
    centroid_map: dict[tuple, np.ndarray] = {}
    for _, row in centroids.iterrows():
        key = (row["intended_area"], row["intended_hemi"])
        centroid_map[key] = np.array([
            row["centroid_x"], row["centroid_y"], row["centroid_z"]])

    out = df.copy()
    distances   = []
    centroid_xs = []
    centroid_ys = []
    centroid_zs = []

    for _, row in df.iterrows():
        key = (str(row[target_col]).strip(), str(row[hemi_col]).strip())
        if key in centroid_map:
            c    = centroid_map[key]
            pt   = np.array([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])
            dist = float(np.linalg.norm(pt - c))
            distances.append(round(dist, 3))
            centroid_xs.append(round(float(c[0]), 3))
            centroid_ys.append(round(float(c[1]), 3))
            centroid_zs.append(round(float(c[2]), 3))
        else:
            distances.append(np.nan)
            centroid_xs.append(np.nan)
            centroid_ys.append(np.nan)
            centroid_zs.append(np.nan)

    out["distance_from_centroid_mm"] = distances
    out["centroid_x"]                = centroid_xs
    out["centroid_y"]                = centroid_ys
    out["centroid_z"]                = centroid_zs

    # Summary stats per group
    print("\nDistance from centroid (mm) — summary:")
    for (target, hemi), grp in out.groupby([target_col, hemi_col]):
        dists = grp["distance_from_centroid_mm"].dropna()
        print(f"  {target:12s} {hemi}  "
              f"mean={dists.mean():.1f}  "
              f"std={dists.std():.1f}  "
              f"min={dists.min():.1f}  "
              f"max={dists.max():.1f}")

    return out


# =============================================================================
# Glass brain plots
# =============================================================================

def _plot_target_brain(
    df: pd.DataFrame,
    centroids: pd.DataFrame | None,
    out_path: str,
    target_col: str,
    hemi_col: str,
    title: str | None = None,
    show_centroids: bool = False,
) -> None:
    """Static 4-panel glass brain coloured by intended target."""
    fig  = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=title, annotate=True, black_bg=False,
    )

    # Plot points grouped by colour
    color_groups: dict[str, list] = {}
    for _, row in df.iterrows():
        color = _get_color(row[target_col], row[hemi_col])
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(
            [row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])

    for color, pts in color_groups.items():
        disp.add_markers(np.array(pts),
                         marker_color=color,
                         marker_size=POINT_MARKER_SIZE)

    # Centroid markers (larger, same colour)
    if show_centroids and centroids is not None:
        for _, crow in centroids.iterrows():
            color = _get_color(crow["intended_area"], crow["intended_hemi"])
            disp.add_markers(
                np.array([[crow["centroid_x"],
                           crow["centroid_y"],
                           crow["centroid_z"]]]),
                marker_color=color,
                marker_size=CENTROID_MARKER_SIZE,
            )

    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_target_interactive(
    df: pd.DataFrame,
    out_path: str,
    target_col: str,
    hemi_col: str,
) -> None:
    """Interactive HTML glass brain coloured by intended target."""
    coords = df[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
    colors = [_get_color(row[target_col], row[hemi_col])
              for _, row in df.iterrows()]

    view = plotting.view_markers(coords, marker_color=colors,
                                 marker_size=5)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    view.save_as_html(out_path)
    print(f"  Saved: {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prefix      = args.output_prefix
    target_col  = args.target_col
    hemi_col    = args.hemi_col

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("\n=== Loading data ===")
    df = _load_csvs(args.csvs, target_col, hemi_col)

    # ------------------------------------------------------------------ #
    # Centroids
    # ------------------------------------------------------------------ #
    print("\n=== Computing centroids ===")
    centroids = compute_centroids(df, target_col, hemi_col)

    centroids_path = os.path.join(args.output_dir, f"{prefix}_centroids.csv")
    centroids.to_csv(centroids_path, index=False)
    print(f"\nCentroids CSV: {centroids_path}")

    # ------------------------------------------------------------------ #
    # Distances
    # ------------------------------------------------------------------ #
    print("\n=== Computing distances from centroids ===")
    df_with_dist = compute_distances(df, centroids, target_col, hemi_col)

    distances_path = os.path.join(args.output_dir, f"{prefix}_distances.csv")

    # Select key output columns
    base_cols = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z",
                 target_col, hemi_col,
                 "distance_from_centroid_mm",
                 "centroid_x", "centroid_y", "centroid_z"]
    # Add atlas cols if present
    atlas_cols = [c for c in df_with_dist.columns if c.startswith("region_")]
    out_cols   = [c for c in base_cols + atlas_cols if c in df_with_dist.columns]

    df_with_dist[out_cols].to_csv(distances_path, index=False)
    print(f"\nDistances CSV: {distances_path}")

    # ------------------------------------------------------------------ #
    # Glass brain plots
    # ------------------------------------------------------------------ #
    if not args.no_plots:
        print("\n=== Generating glass brain plots ===")

        # All subjects, coloured by target
        _plot_target_brain(
            df, None,
            os.path.join(args.output_dir, f"{prefix}_target_glass_brain.png"),
            target_col, hemi_col,
            title="Stimulation sites by intended target",
        )

        # Same with centroids overlaid
        _plot_target_brain(
            df, centroids,
            os.path.join(args.output_dir, f"{prefix}_target_centroids.png"),
            target_col, hemi_col,
            title="Stimulation sites + group centroids",
            show_centroids=True,
        )

        # Interactive HTML
        _plot_target_interactive(
            df,
            os.path.join(args.output_dir, f"{prefix}_target_glass_brain.html"),
            target_col, hemi_col,
        )

        # Per-site plots
        if args.by_site and "site" in df.columns:
            for site in sorted(df["site"].unique()):
                site_df = df[df["site"] == site]
                _plot_target_brain(
                    site_df, centroids,
                    os.path.join(args.output_dir,
                                 f"{prefix}_target_{site}.png"),
                    target_col, hemi_col,
                    title=f"{site} — coloured by intended target",
                    show_centroids=True,
                )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"Output directory : {args.output_dir}")
    print(f"Centroids CSV    : {prefix}_centroids.csv")
    print(f"Distances CSV    : {prefix}_distances.csv")
    if not args.no_plots:
        print(f"Glass brain PNG  : {prefix}_target_glass_brain.png")
        print(f"With centroids   : {prefix}_target_centroids.png")
        print(f"Interactive HTML : {prefix}_target_glass_brain.html")
        if args.by_site:
            sites = sorted(df["site"].unique())
            for s in sites:
                print(f"Site plot        : {prefix}_target_{s}.png")
    print("=" * 60)


if __name__ == "__main__":
    main()