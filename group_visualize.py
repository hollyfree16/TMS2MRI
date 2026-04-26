"""
group_visualize.py
==================
Multi-site group MNI visualizer.

Reads one or more per-subject shared CSVs (or a single pre-combined CSV)
and plots all stimulation coordinates on a single MNI glass brain.

Each site gets a distinct colour; L and R hemispheres are shown as light
and deep variants of that colour.

Also produces:
  - A density heatmap overlay
  - Atlas overlay plots for all configured atlases (Harvard-Oxford, AAL,
    Destrieux, Schaefer 100-1000, Yeo 7/17) — one PNG per atlas
  - An inflated brain plot using fsaverage surface-snapped coordinates
  - A combined output CSV with atlas label columns for all atlases

Requirements
------------
  pip install nilearn nibabel numpy pandas scipy matplotlib

Usage
-----
  # Multiple site CSVs
  python group_visualize.py \\
      --csv site1.csv --csv site2.csv --csv site3.csv \\
      --output-dir ./group_viz \\
      --output-prefix group_sites

  # Single pre-combined CSV
  python group_visualize.py \\
      --csv all_subjects.csv \\
      --output-dir ./group_viz

  # With per-subject fsaverage CSVs for inflated brain plot
  python group_visualize.py \\
      --csv all_subjects.csv \\
      --fsaverage-csv sub01/coordinates/targets_fsaverage.csv \\
      --fsaverage-csv sub02/coordinates/targets_fsaverage.csv \\
      --output-dir ./group_viz

  # Skip specific outputs
  python group_visualize.py \\
      --csv site1.csv \\
      --output-dir ./group_viz \\
      --no-heatmap \\
      --no-atlas \\
      --no-inflated
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting

import logging

log = logging.getLogger("tms2mni.group_visualize")
logging.basicConfig(
    format="[%(name)s] %(levelname)s  %(message)s",
    level=logging.INFO,
)

# =============================================================================
# Configuration
# =============================================================================

SITE_BASE_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]

LIGHT_FACTOR = 1.45
DEEP_FACTOR  = 0.55
MARKER_SIZE  = 5
HEATMAP_FWHM_MM = 5

REQUIRED_COLS = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z", "hemisphere"]

FSAVERAGE_REQUIRED_COLS = [
    "ef_mni_mm_x", "ef_mni_mm_y", "ef_mni_mm_z",
    "fs_x", "fs_y", "fs_z",
    "fs_surface", "ef_hemisphere",
]


# =============================================================================
# Colour helpers
# =============================================================================

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(min(1.0, max(0.0, r)) * 255),
        int(min(1.0, max(0.0, g)) * 255),
        int(min(1.0, max(0.0, b)) * 255),
    )

def _variant(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r * factor, g * factor, b * factor)

def _site_colors(base: str) -> tuple[str, str]:
    return _variant(base, LIGHT_FACTOR), _variant(base, DEEP_FACTOR)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-site group MNI visualization with multi-atlas labelling."
    )
    p.add_argument("--csv",           required=True, action="append", dest="csvs",
                   help="Path to a site CSV. Repeat: --csv s1.csv --csv s2.csv")
    p.add_argument("--output-dir",    required=True,
                   help="Directory to save all outputs")
    p.add_argument("--output-prefix", default="group",
                   help="Filename prefix for outputs (default: group)")
    p.add_argument("--no-heatmap",    action="store_true",
                   help="Skip heatmap generation")
    p.add_argument("--no-atlas",      action="store_true",
                   help="Skip all atlas labelling and plots")
    p.add_argument(
        "--fsaverage-csv",
        action="append",
        dest="fsaverage_csvs",
        default=None,
        metavar="PATH",
        help="Path to a per-subject targets_fsaverage.csv. Repeat for multiple subjects.",
    )
    p.add_argument(
        "--fsaverage-mesh",
        default="fsaverage",
        choices=["fsaverage", "fsaverage5", "fsaverage6"],
    )
    p.add_argument("--no-inflated",   action="store_true",
                   help="Skip the inflated brain plot.")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def _load_csvs(csv_paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if len(csv_paths) > len(SITE_BASE_COLORS):
        log.error("%d CSVs provided but only %d colours defined.",
                  len(csv_paths), len(SITE_BASE_COLORS))
        sys.exit(1)

    all_dfs, site_labels = [], []

    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            log.error("Missing columns %s in %s", missing, path)
            sys.exit(1)

        df = df.dropna(subset=["mni_mm_x", "mni_mm_y", "mni_mm_z"])
        df["_csv_index"] = i

        site_name = df["site"].iloc[0] if not df.empty else f"site{i+1}"
        site_labels.append(site_name)
        all_dfs.append(df)
        log.info("Site %d (%s): %d points from %s", i+1, site_name, len(df), path)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["mni_hemisphere"] = combined["mni_mm_x"].apply(
        lambda x: "L" if x < 0 else "R")
    log.info("Total: %d points across %d site(s)", len(combined), len(csv_paths))
    return combined, site_labels


def _load_fsaverage_csvs(csv_paths: list[str], site_labels: list[str]) -> pd.DataFrame | None:
    all_dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, na_values=["-", " -", "- "])
        except Exception as e:
            log.error("Could not read fsaverage CSV %s: %s", path, e)
            return None

        missing_cols = [c for c in FSAVERAGE_REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            log.error("fsaverage CSV %s missing columns %s", path, missing_cols)
            return None

        df = df.dropna(subset=["fs_x", "fs_y", "fs_z"])

        if "site" not in df.columns:
            df["site"] = os.path.basename(path)

        def _site_to_idx(s):
            try:
                return site_labels.index(s)
            except ValueError:
                return 0

        df["_csv_index"] = df["site"].apply(_site_to_idx)
        all_dfs.append(df)
        log.info("fsaverage CSV: %d valid points from %s", len(df), path)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Total snapped points for inflated brain: %d", len(combined))
    return combined


# =============================================================================
# Glass brain plots
# =============================================================================

def _plot_interactive(combined, legend_info, out_dir, prefix):
    colors = []
    for _, row in combined.iterrows():
        light, deep = _site_colors(SITE_BASE_COLORS[int(row["_csv_index"])])
        colors.append(light if row["mni_hemisphere"] == "L" else deep)

    coords = combined[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
    view = plotting.view_markers(coords, marker_color=colors, marker_size=MARKER_SIZE)
    path = os.path.join(out_dir, f"{prefix}_interactive.html")
    view.save_as_html(path)
    log.info("Interactive HTML: %s", path)


def _plot_static(combined, legend_info, out_dir, prefix):
    import matplotlib
    matplotlib.use("Agg")

    fig  = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=None, annotate=True, black_bg=False,
    )

    for i, (site_label, base, light, deep) in enumerate(legend_info):
        site_df   = combined[combined["_csv_index"] == i]
        left_pts  = site_df[site_df["mni_hemisphere"] == "L"][
            ["mni_mm_x","mni_mm_y","mni_mm_z"]].values
        right_pts = site_df[site_df["mni_hemisphere"] == "R"][
            ["mni_mm_x","mni_mm_y","mni_mm_z"]].values
        if len(left_pts):
            disp.add_markers(left_pts,  marker_color=light, marker_size=MARKER_SIZE*3)
        if len(right_pts):
            disp.add_markers(right_pts, marker_color=deep,  marker_size=MARKER_SIZE*3)

    patches = []
    for site_label, base, light, deep in legend_info:
        patches += [
            mpatches.Patch(color=light, label=f"{site_label} L"),
            mpatches.Patch(color=deep,  label=f"{site_label} R"),
        ]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(legend_info)*2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.05))

    path = os.path.join(out_dir, f"{prefix}_glass_brain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Static glass brain: %s", path)


# =============================================================================
# Multi-atlas labelling and plots
# =============================================================================

def _run_atlas_labelling(
    combined: pd.DataFrame,
    legend_info: list,
    out_dir: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Label all coordinates against every atlas and produce one PNG per atlas.
    Adds atlas label columns to combined and returns it.
    """
    try:
        from utils.atlas import (
            label_coords_all, plot_all_atlases,
            ATLAS_KEYS, ATLAS_COL_NAMES, ATLAS_NAMES, ATLAS_PNG_STEMS,
        )
    except ImportError:
        log.warning("utils.atlas not importable — skipping atlas labelling.")
        return combined

    coords     = combined[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
    valid_mask = ~np.isnan(coords).any(axis=1)

    log.info("Labelling %d coordinates against %d atlases...",
             valid_mask.sum(), len(ATLAS_KEYS))

    all_labels: dict = {k: ["no_region"] * len(coords) for k in ATLAS_KEYS}
    if valid_mask.any():
        valid_atlas = label_coords_all(coords[valid_mask])
        for key in ATLAS_KEYS:
            j = 0
            for i, v in enumerate(valid_mask):
                if v:
                    all_labels[key][i] = valid_atlas[key][j]
                    j += 1

    for key in ATLAS_KEYS:
        combined[ATLAS_COL_NAMES[key]] = all_labels[key]

    # Log top regions per atlas (Harvard-Oxford and AAL only for brevity)
    for key in ["harvard_oxford", "aal"]:
        col = ATLAS_COL_NAMES[key]
        counts = (combined[combined[col].isin(["no_region", "out_of_bounds"]) == False]
                  [col].value_counts())
        log.info("Top regions (%s):", ATLAS_NAMES[key])
        for region, count in counts.head(5).items():
            log.info("  %4d  %s", count, region)

    # One glass brain PNG per atlas
    log.info("Generating atlas overlay plots...")
    hemi_labels = list(combined["mni_hemisphere"])

    # Use first site's colours for markers in group plot
    color_left  = _variant(SITE_BASE_COLORS[0], LIGHT_FACTOR)
    color_right = _variant(SITE_BASE_COLORS[0], DEEP_FACTOR)

    # We want site-coloured markers, not single colour — build per-point colors
    marker_colors_left  = []
    marker_colors_right = []
    coords_left  = []
    coords_right = []

    for _, row in combined.iterrows():
        light, deep = _site_colors(SITE_BASE_COLORS[int(row["_csv_index"])])
        if row["mni_hemisphere"] == "L":
            coords_left.append([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])
            marker_colors_left.append(light)
        else:
            coords_right.append([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])
            marker_colors_right.append(deep)

    for key in ATLAS_KEYS:
        stem     = ATLAS_PNG_STEMS.get(key, f"{key}_regions")
        out_path = os.path.join(out_dir, f"{prefix}_{stem}.png")
        try:
            _plot_group_atlas(
                key             = key,
                coords_left     = np.array(coords_left)  if coords_left  else np.empty((0,3)),
                coords_right    = np.array(coords_right) if coords_right else np.empty((0,3)),
                colors_left     = marker_colors_left,
                colors_right    = marker_colors_right,
                combined        = combined,
                legend_info     = legend_info,
                out_path        = out_path,
                atlas_col       = ATLAS_COL_NAMES[key],
                atlas_name      = ATLAS_NAMES[key],
            )
        except Exception as e:
            log.warning("Atlas plot failed for %s: %s", key, e)

    return combined


def _plot_group_atlas(
    key: str,
    coords_left: np.ndarray,
    coords_right: np.ndarray,
    colors_left: list,
    colors_right: list,
    combined: pd.DataFrame,
    legend_info: list,
    out_path: str,
    atlas_col: str,
    atlas_name: str,
) -> None:
    """Plot group glass brain with one atlas overlaid and site-coloured markers."""
    import matplotlib
    matplotlib.use("Agg")

    try:
        from utils.atlas import _load_atlas
    except ImportError:
        return

    data, affine, labels = _load_atlas(key)

    targeted = sorted(set(
        r for r in combined[atlas_col].dropna()
        if r not in ("no_region", "out_of_bounds", "atlas_unavailable", "atlas_error")
    ))
    n_regions  = len(targeted)
    cmap_r     = cm.get_cmap("tab20", max(n_regions, 1))
    reg_colors = {r: cmap_r(i) for i, r in enumerate(targeted)}

    fig  = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=atlas_name, annotate=True, black_bg=False,
    )

    # Atlas contours
    if data is not None and n_regions > 0:
        label_to_idx = {}
        for i, lbl in enumerate(labels):
            lbl_str = lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
            label_to_idx[lbl_str] = i

        for region in targeted:
            idx = label_to_idx.get(region)
            if idx is None:
                continue
            region_vol = (data == idx).astype(np.float32)
            region_img = nib.Nifti1Image(region_vol, affine)
            try:
                disp.add_contours(region_img, levels=[0.5],
                                  colors=[reg_colors[region]], linewidths=1.5)
            except Exception:
                pass

    # Site-coloured markers
    if len(coords_left):
        for pt, color in zip(coords_left, colors_left):
            disp.add_markers([pt], marker_color=color, marker_size=MARKER_SIZE*3)
    if len(coords_right):
        for pt, color in zip(coords_right, colors_right):
            disp.add_markers([pt], marker_color=color, marker_size=MARKER_SIZE*3)

    # Legend: regions + sites
    site_handles = []
    for site_label, base, light, deep in legend_info:
        site_handles += [
            mpatches.Patch(color=light, label=f"{site_label} L"),
            mpatches.Patch(color=deep,  label=f"{site_label} R"),
        ]

    if n_regions > 0:
        region_handles = [
            mlines.Line2D([], [], color=reg_colors[r], linewidth=1.5, label=r)
            for r in targeted
        ]
        leg1 = fig.legend(handles=region_handles, loc="lower center",
                          ncol=min(n_regions, 6), fontsize=7, frameon=False,
                          bbox_to_anchor=(0.5, -0.08))
        fig.add_artist(leg1)
        fig.legend(handles=site_handles, loc="lower center",
                   ncol=len(legend_info)*2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.16))
    else:
        fig.legend(handles=site_handles, loc="lower center",
                   ncol=len(legend_info)*2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.05))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Group atlas plot (%s): %s", key, out_path)


# =============================================================================
# Heatmap
# =============================================================================

def _build_heatmap(combined, out_dir, prefix):
    from scipy.ndimage import gaussian_filter

    log.info("Building density heatmap (FWHM=%d mm)...", HEATMAP_FWHM_MM)

    mni_img    = datasets.load_mni152_template(resolution=1)
    mni_affine = mni_img.affine
    mni_shape  = mni_img.shape[:3]
    mni_inv    = np.linalg.inv(mni_affine)

    coords    = combined[["mni_mm_x","mni_mm_y","mni_mm_z"]].values
    coords_h  = np.hstack([coords, np.ones((len(coords), 1))])
    vox_coords = np.round((mni_inv @ coords_h.T).T[:, :3]).astype(int)

    density = np.zeros(mni_shape, dtype=np.float32)
    for vx, vy, vz in vox_coords:
        if all(0 <= v < mni_shape[i] for i, v in enumerate([vx, vy, vz])):
            density[vx, vy, vz] += 1.0

    vox_size = abs(mni_affine[0, 0])
    sigma    = (HEATMAP_FWHM_MM / 2.355) / vox_size
    density  = gaussian_filter(density, sigma=sigma)

    threshold = density.max() * 0.05
    density[density < threshold] = 0

    unit = np.zeros(mni_shape, dtype=np.float32)
    cx, cy, cz = [s // 2 for s in mni_shape]
    unit[cx, cy, cz] = 1.0
    kernel_sum = gaussian_filter(unit, sigma=sigma).max()
    count_data = density / kernel_sum if kernel_sum > 0 else density.copy()
    count_data[count_data < 0.5] = 0
    max_count  = max(1, int(np.round(count_data.max())))

    def _save(img, vmin, vmax, path, label, tick_labels=None):
        fig_h  = plt.figure(figsize=(14, 4), facecolor="white")
        disp_h = plotting.plot_glass_brain(
            img, display_mode="lyrz", colorbar=True, cmap="Reds",
            figure=fig_h, annotate=True, black_bg=False, vmin=vmin, vmax=vmax,
        )
        if disp_h._cbar is not None and tick_labels is not None:
            ticks = np.linspace(vmin, vmax, len(tick_labels))
            disp_h._cbar.set_ticks(ticks)
            disp_h._cbar.set_ticklabels(tick_labels)
            disp_h._cbar.set_label(label, fontsize=9)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_h)
        log.info("Heatmap: %s", path)

    count_img = nib.Nifti1Image(count_data, mni_affine)
    tl_overlap = [str(v) for v in range(1, max_count+1)] if max_count <= 10 else None
    _save(count_img, 1, max_count,
          os.path.join(out_dir, f"{prefix}_heatmap_overlap.png"),
          "N overlapping points", tl_overlap)

    shifted = np.where(count_data > 0, count_data+1, 0).astype(np.float32)
    tl_all  = [str(v) for v in range(1, max_count+2)] if max_count <= 9 else None
    _save(nib.Nifti1Image(shifted, mni_affine), 1, max_count+1,
          os.path.join(out_dir, f"{prefix}_heatmap_all.png"),
          "N points", tl_all)

    nii_path = os.path.join(out_dir, f"{prefix}_heatmap.nii.gz")
    nib.save(nib.Nifti1Image(density, mni_affine), nii_path)
    log.info("Heatmap NIfTI: %s", nii_path)


# =============================================================================
# Inflated brain plot
# =============================================================================

def _plot_inflated(fs_combined, legend_info, out_dir, prefix, mesh="fsaverage"):
    import matplotlib
    matplotlib.use("Agg")
    from nilearn import surface as surf

    log.info("Loading fsaverage surfaces (mesh=%s)...", mesh)
    fsaverage = datasets.fetch_surf_fsaverage(mesh)

    infl_lh_coords, infl_lh_faces = surf.load_surf_mesh(fsaverage["infl_left"])
    infl_rh_coords, infl_rh_faces = surf.load_surf_mesh(fsaverage["infl_right"])
    n_lh = len(infl_lh_coords)
    n_rh = len(infl_rh_coords)

    BG_INTENSITY = 0.82
    SPOT_RADIUS  = 3

    lh_texture = np.zeros(n_lh, dtype=float)
    rh_texture = np.zeros(n_rh, dtype=float)

    lh_adj = _build_adjacency(infl_lh_faces, n_lh)
    rh_adj = _build_adjacency(infl_rh_faces, n_rh)

    for _, row in fs_combined.iterrows():
        vertex_idx = row.get("fs_vertex")
        surface    = row.get("fs_surface", "")
        csv_idx    = int(row.get("_csv_index", 0))
        if pd.isna(vertex_idx):
            continue
        vertex_idx = int(vertex_idx)
        val        = float(csv_idx + 1)
        if "left" in str(surface):
            for v in _k_ring(lh_adj, vertex_idx, SPOT_RADIUS):
                lh_texture[v] = val
        else:
            for v in _k_ring(rh_adj, vertex_idx, SPOT_RADIUS):
                rh_texture[v] = val

    n_sites = len(legend_info)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap_colors = [(BG_INTENSITY, BG_INTENSITY, BG_INTENSITY, 1.0)]
    for _, base, light, deep in legend_info:
        r, g, b = _hex_to_rgb(base)
        cmap_colors.append((r, g, b, 1.0))

    cmap   = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, n_sites + 1.5)
    norm   = BoundaryNorm(bounds, cmap.N)
    vmin, vmax = 0, n_sites

    views = [
        ("left",  "lateral", lh_texture, fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("left",  "medial",  lh_texture, fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("right", "lateral", rh_texture, fsaverage["infl_right"], fsaverage["sulc_right"]),
        ("right", "medial",  rh_texture, fsaverage["infl_right"], fsaverage["sulc_right"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4),
                             subplot_kw={"projection": "3d"},
                             facecolor="white")
    fig.subplots_adjust(wspace=0.01)
    titles = ["LH lateral", "LH medial", "RH lateral", "RH medial"]

    for ax, title, (hemi, view, texture, infl_path, sulc_path) in zip(
            axes, titles, views):
        plotting.plot_surf(
            infl_path, surf_map=texture, hemi=hemi, view=view,
            bg_map=sulc_path, bg_on_data=True, cmap=cmap,
            vmin=vmin, vmax=vmax, colorbar=False, axes=ax, figure=fig,
        )
        ax.set_title(title, fontsize=9, pad=2)

    patches = [mpatches.Patch(color=_hex_to_rgb(base) + (1,) if False else base,
                               label=site_label)
               for site_label, base, light, deep in legend_info]
    fig.legend(handles=[mpatches.Patch(color=base, label=s)
                         for s, base, l, d in legend_info],
               loc="lower center", ncol=len(legend_info),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    path = os.path.join(out_dir, f"{prefix}_inflated_brain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Inflated brain: %s", path)


def _build_adjacency(faces, n_vertices):
    adj = [set() for _ in range(n_vertices)]
    for v0, v1, v2 in faces:
        adj[v0].add(v1); adj[v0].add(v2)
        adj[v1].add(v0); adj[v1].add(v2)
        adj[v2].add(v0); adj[v2].add(v1)
    return adj


def _k_ring(adj, seed, k):
    ring    = {seed}
    current = {seed}
    for _ in range(k):
        nxt = set()
        for v in current:
            nxt.update(adj[v])
        nxt -= ring
        ring.update(nxt)
        current = nxt
        if not current:
            break
    return ring


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = _parse_args()
    log.setLevel(getattr(logging, args.log_level))
    os.makedirs(args.output_dir, exist_ok=True)

    combined, site_labels = _load_csvs(args.csvs)

    legend_info = [
        (site_labels[i], SITE_BASE_COLORS[i], *_site_colors(SITE_BASE_COLORS[i]))
        for i in range(len(args.csvs))
    ]

    _plot_interactive(combined, legend_info, args.output_dir, args.output_prefix)
    _plot_static(combined, legend_info, args.output_dir, args.output_prefix)

    if not args.no_heatmap:
        _build_heatmap(combined, args.output_dir, args.output_prefix)

    if not args.no_atlas:
        combined = _run_atlas_labelling(
            combined, legend_info, args.output_dir, args.output_prefix)
    else:
        log.info("--no-atlas set — skipping atlas labelling and plots.")

    # Inflated brain plot
    if args.fsaverage_csvs and not args.no_inflated:
        fs_combined = _load_fsaverage_csvs(args.fsaverage_csvs, site_labels)
        if fs_combined is not None and not fs_combined.empty:
            _plot_inflated(fs_combined, legend_info,
                           args.output_dir, args.output_prefix,
                           mesh=args.fsaverage_mesh)
        else:
            log.warning("No valid fsaverage coordinates — skipping inflated brain.")
    elif args.fsaverage_csvs and args.no_inflated:
        log.info("--no-inflated set — skipping inflated brain plot.")
    else:
        log.info("No --fsaverage-csv provided — skipping inflated brain plot.")

    # Output CSV — base columns + all atlas columns
    try:
        from utils.atlas import ATLAS_KEYS, ATLAS_COL_NAMES
        atlas_cols = [ATLAS_COL_NAMES[k] for k in ATLAS_KEYS
                      if ATLAS_COL_NAMES[k] in combined.columns]
    except ImportError:
        atlas_cols = []

    out_cols = (["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z",
                 "hemisphere", "mni_hemisphere"] + atlas_cols)
    out_cols = [c for c in out_cols if c in combined.columns]

    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_all_sites.csv")
    combined[out_cols].to_csv(csv_path, index=False)
    log.info("Combined CSV: %s  (%d atlas columns)", csv_path, len(atlas_cols))

    log.info("=" * 60)
    log.info("Group visualization complete")
    log.info("Output directory: %s", args.output_dir)
    log.info("Total points: %d across %d site(s)", len(combined), len(args.csvs))


if __name__ == "__main__":
    main()