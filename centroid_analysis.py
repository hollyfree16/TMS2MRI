
"""
centroid_analysis_site_colored.py
=================================

Site-specific centroid analysis and one combined centroid plot.

Purpose
-------
Use this when you have multiple sites, e.g. ou1 + ou2, and want ONE combined
glass-brain plot showing site-specific centroids:
  - ou1 = Reds
  - ou2 = Oranges
  - ou3 = Greens
  - ou4 = Blues
  - ou5 = Purples
  - ou6 = Greys
  - additional sites cycle through fallback palettes

Centroids are computed separately for:
  site x intended_area x intended_hemi

So, with ou1 and ou2, you get separate centroids for:
  ou1 L Premotor
  ou1 R Premotor
  ou1 L Parietal
  ou1 R Parietal
  ou2 L Premotor
  ou2 R Premotor
  ou2 L Parietal
  ou2 R Parietal

Coordinates used for centroid computation:
  mni_mm_x, mni_mm_y, mni_mm_z

Snapped atlas columns are preserved in the distance CSV if present, but they are
not used for centroid coordinates.

Usage
-----
python centroid_analysis_site_colored.py \
    --csv /path/to/ou1_all_sites_snapped_centroid.csv \
    --csv /path/to/ou2_all_sites_snapped_centroid.csv \
    --output-dir /path/to/ou1_ou2/ \
    --output-prefix ou1_ou2

Optional:
  --points-faint     Plot all subject points faintly underneath centroids.
  --no-points        Plot centroids only.
  --target-col COL   Default: intended_area
  --hemi-col COL     Default: intended_hemi
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting


# =============================================================================
# Plot settings
# =============================================================================

# Subject points are intentionally faint so the centroid stars carry the plot.
POINT_MARKER_SIZE = 8
POINT_ALPHA = 0.08

# Centroid stars. Black outline keeps light colors readable on white.
CENTROID_OUTLINE_SIZE = 270
CENTROID_INNER_SIZE = 165

# Non-flesh, high-contrast cohort palettes.
#
# Cohort/site color families:
#   ou1 = blues
#   ou2 = purples
#   ou3 = greens
#   ou4 = gold/yellow
#   ou5 = magentas
#
# Within each site:
#   L Premotor  = lighter/saturated family color
#   R Premotor  = darker family color
#   L Parietal  = palest family color
#   R Parietal  = deepest family color
#
# Black star outlines are used so the lighter shades remain visible.
SITE_TARGET_COLORS = {
    "ou1": {
        ("Premotor", "L"): "#4CC9F0",  # bright cyan-blue
        ("Premotor", "R"): "#0057D9",  # strong blue
        ("Parietal", "L"): "#B9F2FF",  # pale ice blue
        ("Parietal", "R"): "#002B7F",  # deep navy
    },
    "ou2": {
        ("Premotor", "L"): "#C77DFF",  # bright violet
        ("Premotor", "R"): "#7B2CBF",  # strong purple
        ("Parietal", "L"): "#E0AAFF",  # pale lavender
        ("Parietal", "R"): "#3C096C",  # deep purple
    },
    "ou3": {
        ("Premotor", "L"): "#80ED99",  # bright green
        ("Premotor", "R"): "#00843D",  # strong green
        ("Parietal", "L"): "#CCFFCC",  # pale mint
        ("Parietal", "R"): "#005A2B",  # deep green
    },
    "ou4": {
        ("Premotor", "L"): "#FFD60A",  # bright gold
        ("Premotor", "R"): "#B8860B",  # dark goldenrod
        ("Parietal", "L"): "#FFF3B0",  # pale yellow
        ("Parietal", "R"): "#7A5C00",  # deep olive-gold
    },
    "ou5": {
        ("Premotor", "L"): "#FF5CDA",  # bright magenta
        ("Premotor", "R"): "#C0007A",  # strong magenta
        ("Parietal", "L"): "#FFC1EF",  # pale pink-magenta
        ("Parietal", "R"): "#7A004D",  # deep magenta
    },
}

FALLBACK_SITE_NAMES = ["ou1", "ou2", "ou3", "ou4", "ou5"]


# =============================================================================
# Helpers
# =============================================================================

def _normalize_target(value) -> str:
    """Normalize target names so premotor/parietal become Premotor/Parietal."""
    s = str(value).strip()
    lower = s.lower()
    if lower == "premotor":
        return "Premotor"
    if lower == "parietal":
        return "Parietal"
    return s


def _normalize_hemi(value) -> str:
    """Normalize hemisphere values to L/R when possible."""
    s = str(value).strip()
    lower = s.lower()
    if lower in {"l", "left", "lh"}:
        return "L"
    if lower in {"r", "right", "rh"}:
        return "R"
    return s


def _site_from_path(path: str) -> str:
    """Fallback site name if no site column exists."""
    p = Path(path)
    for part in reversed(p.parts):
        lower = part.lower()
        if lower.startswith("ou"):
            return part
    return p.stem


def _get_site_palette_map(sites: list[str]) -> dict[str, str]:
    """Map actual site labels to explicit palette names.

    If actual site names are ou1/ou2/etc, they map to themselves.
    Otherwise, sites are assigned in sorted order to ou1/ou2/ou3/ou4/ou5.
    """
    ordered = sorted(str(s) for s in sites)
    site_map = {}
    for i, site in enumerate(ordered):
        if site in SITE_TARGET_COLORS:
            site_map[site] = site
        else:
            site_map[site] = FALLBACK_SITE_NAMES[i % len(FALLBACK_SITE_NAMES)]
    return site_map


def _centroid_color(site: str, target: str, hemi: str, site_cmaps: dict[str, str]) -> str:
    """Return a non-flesh hex color for one site x target x hemisphere centroid."""
    site = str(site)
    palette_name = site_cmaps.get(site, site)
    target = _normalize_target(target)
    hemi = _normalize_hemi(hemi)

    palette = SITE_TARGET_COLORS.get(palette_name, SITE_TARGET_COLORS["ou5"])
    return palette.get((target, hemi), "#000000")


def _add_marker_on_top(fig, disp, coord, *, color, marker_size, marker="*", zorder=1000, alpha=1.0):
    """
    Add a marker with nilearn, then force any newly-created artists to the front.
    Nilearn draws onto multiple projected axes, so this is more reliable than
    passing zorder into add_markers.
    """
    before = {ax: set(ax.collections + ax.lines) for ax in fig.axes}

    disp.add_markers(
        np.asarray(coord),
        marker_color=color,
        marker_size=marker_size,
        marker=marker,
        alpha=alpha,
    )

    after = {ax: set(ax.collections + ax.lines) for ax in fig.axes}

    for ax in fig.axes:
        for artist in after[ax] - before[ax]:
            artist.set_zorder(zorder)


def _legend_handles(centroids: pd.DataFrame, site_cmaps: dict[str, str]) -> list:
    """Build a centroid legend."""
    handles = []

    for _, row in centroids.sort_values(["site", "intended_area", "intended_hemi"]).iterrows():
        site = str(row["site"])
        target = _normalize_target(row["intended_area"])
        hemi = _normalize_hemi(row["intended_hemi"])
        color = _centroid_color(site, target, hemi, site_cmaps)

        handles.append(
            plt.Line2D(
                [], [],
                marker="*",
                color="black",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=14,
                linestyle="None",
                label=f"{site} {hemi} {target}",
            )
        )

    return handles


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Site-specific centroid analysis with one combined site-colored centroid plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--csv", required=True, action="append", dest="csvs",
                   help="Input CSV. Repeat for multiple sites.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-prefix", default="group")
    p.add_argument("--target-col", default="intended_area")
    p.add_argument("--hemi-col", default="intended_hemi")

    point_group = p.add_mutually_exclusive_group()
    point_group.add_argument("--points-faint", action="store_true", default=True,
                             help="Plot subject points faintly under centroids. Default.")
    point_group.add_argument("--no-points", action="store_true",
                             help="Plot centroids only.")

    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def _load_csvs(csv_paths: list[str], target_col: str, hemi_col: str) -> pd.DataFrame:
    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path, na_values=["-", " -", "- "])

        required = [target_col, hemi_col, "mni_mm_x", "mni_mm_y", "mni_mm_z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: missing columns {missing} in {path}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

        if "site" not in df.columns:
            df["site"] = _site_from_path(path)

        df[target_col] = df[target_col].map(_normalize_target)
        df[hemi_col] = df[hemi_col].map(_normalize_hemi)
        df["site"] = df["site"].astype(str)

        dfs.append(df)
        print(f"Loaded {len(df)} rows from {path}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["mni_mm_x", "mni_mm_y", "mni_mm_z", target_col, hemi_col, "site"])

    print(f"Total: {len(combined)} rows with valid coordinates, site, and target labels")
    print(f"Sites: {', '.join(sorted(combined['site'].astype(str).unique()))}")

    return combined


# =============================================================================
# Centroids and distances
# =============================================================================

def compute_site_centroids(df: pd.DataFrame, target_col: str, hemi_col: str) -> pd.DataFrame:
    """Compute centroids by site x target x hemisphere."""
    rows = []

    print("\n=== Computing site-specific centroids ===")

    groups = df.groupby(["site", target_col, hemi_col], dropna=False)
    for (site, target, hemi), grp in groups:
        coords = grp[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].astype(float).values
        centroid = coords.mean(axis=0)

        rows.append({
            "site": site,
            "intended_area": target,
            "intended_hemi": hemi,
            "n_subjects": int(len(grp)),
            "centroid_x": round(float(centroid[0]), 3),
            "centroid_y": round(float(centroid[1]), 3),
            "centroid_z": round(float(centroid[2]), 3),
            "std_x": round(float(coords[:, 0].std()), 3),
            "std_y": round(float(coords[:, 1].std()), 3),
            "std_z": round(float(coords[:, 2].std()), 3),
            "spread_mm": round(float(np.mean([np.linalg.norm(c - centroid) for c in coords])), 3),
        })

        print(
            f"  {site:8s} {str(target):12s} {str(hemi):2s} "
            f"n={len(grp):3d} "
            f"centroid=[{centroid[0]:6.1f}, {centroid[1]:6.1f}, {centroid[2]:6.1f}]"
        )

    return pd.DataFrame(rows)


def compute_site_distances(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    target_col: str,
    hemi_col: str,
) -> pd.DataFrame:
    """Compute distance from each subject point to its own site x target x hemi centroid."""
    centroid_map = {}
    for _, row in centroids.iterrows():
        key = (
            str(row["site"]),
            _normalize_target(row["intended_area"]),
            _normalize_hemi(row["intended_hemi"]),
        )
        centroid_map[key] = np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]], dtype=float)

    out = df.copy()
    distances = []
    centroid_xs = []
    centroid_ys = []
    centroid_zs = []

    for _, row in out.iterrows():
        key = (
            str(row["site"]),
            _normalize_target(row[target_col]),
            _normalize_hemi(row[hemi_col]),
        )
        pt = np.array([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]], dtype=float)

        if key in centroid_map:
            c = centroid_map[key]
            distances.append(round(float(np.linalg.norm(pt - c)), 3))
            centroid_xs.append(round(float(c[0]), 3))
            centroid_ys.append(round(float(c[1]), 3))
            centroid_zs.append(round(float(c[2]), 3))
        else:
            distances.append(np.nan)
            centroid_xs.append(np.nan)
            centroid_ys.append(np.nan)
            centroid_zs.append(np.nan)

    out["site_distance_from_centroid_mm"] = distances
    out["site_centroid_x"] = centroid_xs
    out["site_centroid_y"] = centroid_ys
    out["site_centroid_z"] = centroid_zs

    print("\n=== Distance from site-specific centroid, mm ===")
    for (site, target, hemi), grp in out.groupby(["site", target_col, hemi_col]):
        d = grp["site_distance_from_centroid_mm"].dropna()
        print(
            f"  {site:8s} {str(target):12s} {str(hemi):2s} "
            f"mean={d.mean():5.1f} std={d.std():5.1f} min={d.min():5.1f} max={d.max():5.1f}"
        )

    return out


# =============================================================================
# Plotting
# =============================================================================

def plot_combined_site_centroids(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    output_path: str,
    target_col: str,
    hemi_col: str,
    plot_points: bool = True,
) -> None:
    """One combined glass-brain plot with all site-specific centroids."""
    sites = sorted(df["site"].astype(str).unique())
    site_cmaps = _get_site_palette_map(sites)

    fig = plt.figure(figsize=(16, 4.5), facecolor="white")
    disp = plotting.plot_glass_brain(
        None,
        display_mode="lyrz",
        figure=fig,
        title="Site-specific centroids by cohort and target",
        annotate=True,
        black_bg=False,
    )

    # Faint subject points under the centroids.
    if plot_points:
        for _, row in df.iterrows():
            color = _centroid_color(row["site"], row[target_col], row[hemi_col], site_cmaps)
            coord = np.array([[row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]]], dtype=float)
            disp.add_markers(
                coord,
                marker_color=color,
                marker_size=POINT_MARKER_SIZE,
                alpha=POINT_ALPHA,
            )

    # Centroids are drawn last and forced on top.
    for _, row in centroids.iterrows():
        color = _centroid_color(row["site"], row["intended_area"], row["intended_hemi"], site_cmaps)
        coord = np.array([[row["centroid_x"], row["centroid_y"], row["centroid_z"]]], dtype=float)

        _add_marker_on_top(
            fig,
            disp,
            coord,
            color="black",
            marker_size=CENTROID_OUTLINE_SIZE,
            marker="*",
            zorder=1000,
            alpha=1.0,
        )
        _add_marker_on_top(
            fig,
            disp,
            coord,
            color=color,
            marker_size=CENTROID_INNER_SIZE,
            marker="*",
            zorder=1001,
            alpha=1.0,
        )

    handles = _legend_handles(centroids, site_cmaps)
    ncol = min(4, max(1, len(handles)))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved combined site-centroid plot: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = _load_csvs(args.csvs, args.target_col, args.hemi_col)

    centroids = compute_site_centroids(df, args.target_col, args.hemi_col)
    centroids_path = os.path.join(args.output_dir, f"{args.output_prefix}_site_centroids.csv")
    centroids.to_csv(centroids_path, index=False)
    print(f"\nSite centroids CSV: {centroids_path}")

    df_dist = compute_site_distances(df, centroids, args.target_col, args.hemi_col)
    distances_path = os.path.join(args.output_dir, f"{args.output_prefix}_site_distances.csv")

    base_cols = [
        "subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z",
        args.target_col, args.hemi_col,
        "site_distance_from_centroid_mm",
        "site_centroid_x", "site_centroid_y", "site_centroid_z",
    ]
    atlas_cols = [c for c in df_dist.columns if c.startswith("region_")]
    out_cols = [c for c in base_cols + atlas_cols if c in df_dist.columns]
    df_dist[out_cols].to_csv(distances_path, index=False)
    print(f"Site distances CSV: {distances_path}")

    plot_path = os.path.join(args.output_dir, f"{args.output_prefix}_site_centroids_combined.png")
    plot_combined_site_centroids(
        df,
        centroids,
        plot_path,
        args.target_col,
        args.hemi_col,
        plot_points=not args.no_points,
    )

    print("\n" + "=" * 60)
    print(f"Output directory       : {args.output_dir}")
    print(f"Site centroids CSV     : {os.path.basename(centroids_path)}")
    print(f"Site distances CSV     : {os.path.basename(distances_path)}")
    print(f"Combined centroid plot : {os.path.basename(plot_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
