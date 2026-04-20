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
  - Harvard-Oxford cortical atlas region labels
  - An inflated brain plot using fsaverage surface-snapped coordinates
    (reads targets_fsaverage.csv files produced by stage 06)
  - A combined output CSV

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

  # Skip heatmap or atlas labelling
  python group_visualize.py \\
      --csv site1.csv \\
      --output-dir ./group_viz \\
      --no-heatmap \\
      --no-atlas
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

LIGHT_FACTOR = 1.45   # L hemisphere
DEEP_FACTOR  = 0.55   # R hemisphere
MARKER_SIZE  = 5
HEATMAP_FWHM_MM = 5
HO_PROBABILITY_THRESHOLD = 25

REQUIRED_COLS = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z", "hemisphere"]

# Columns required in a targets_fsaverage.csv for the inflated brain plot
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
        description="Multi-site group MNI visualization with heatmap and atlas labelling."
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
                   help="Skip Harvard-Oxford atlas labelling")
    p.add_argument(
        "--fsaverage-csv",
        action="append",
        dest="fsaverage_csvs",
        default=None,
        metavar="PATH",
        help="Path to a per-subject targets_fsaverage.csv (produced by stage 06). "
             "Repeat for multiple subjects. When provided, an additional inflated "
             "brain plot is generated using the surface-snapped coordinates.",
    )
    p.add_argument(
        "--fsaverage-mesh",
        default="fsaverage",
        choices=["fsaverage", "fsaverage5", "fsaverage6"],
        help="fsaverage mesh resolution to use for the inflated brain plot "
             "(default: fsaverage). Must match the mesh used in stage 06.",
    )
    p.add_argument(
        "--no-inflated",
        action="store_true",
        help="Skip the inflated brain plot even if --fsaverage-csv is provided.",
    )
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def _load_csvs(csv_paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if len(csv_paths) > len(SITE_BASE_COLORS):
        log.error("%d CSVs provided but only %d colours defined. "
                  "Add more to SITE_BASE_COLORS.", len(csv_paths), len(SITE_BASE_COLORS))
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

        log.info("Site %d (%s): %d points from %s", i + 1, site_name, len(df), path)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["mni_hemisphere"] = combined["mni_mm_x"].apply(
        lambda x: "L" if x < 0 else "R"
    )
    log.info("Total: %d points across %d site(s)", len(combined), len(csv_paths))
    return combined, site_labels


def _load_fsaverage_csvs(
    csv_paths: list[str],
    site_labels: list[str],
) -> pd.DataFrame | None:
    """
    Load and concatenate per-subject targets_fsaverage.csv files.

    Returns a combined DataFrame or None if loading fails.
    """
    all_dfs = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path, na_values=["-", " -", "- "])
        except Exception as e:
            log.error("Could not read fsaverage CSV %s: %s", path, e)
            return None

        missing_cols = [c for c in FSAVERAGE_REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            log.error(
                "fsaverage CSV %s is missing columns %s — "
                "was stage 06 run for this subject?", path, missing_cols
            )
            return None

        df = df.dropna(subset=["fs_x", "fs_y", "fs_z"])

        # Attach site label if available (use 'site' col if present, else filename)
        if "site" not in df.columns:
            df["site"] = os.path.basename(path)

        # Map site label → colour index using the same order as the main CSVs
        def _site_to_idx(s):
            try:
                return site_labels.index(s)
            except ValueError:
                return 0

        df["_csv_index"] = df["site"].apply(_site_to_idx)
        all_dfs.append(df)

        log.info(
            "fsaverage CSV: %d valid snapped points from %s", len(df), path
        )

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Total snapped points for inflated brain: %d", len(combined))
    return combined


# =============================================================================
# Glass brain plots
# =============================================================================

def _plot_interactive(combined: pd.DataFrame, legend_info: list, out_dir: str, prefix: str) -> None:
    colors = []
    for _, row in combined.iterrows():
        light, deep = _site_colors(SITE_BASE_COLORS[int(row["_csv_index"])])
        colors.append(light if row["hemisphere"] == "L" else deep)

    coords = combined[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
    view = plotting.view_markers(coords, marker_color=colors, marker_size=MARKER_SIZE)
    path = os.path.join(out_dir, f"{prefix}_interactive.html")
    view.save_as_html(path)
    log.info("Interactive HTML: %s", path)


def _plot_static(combined: pd.DataFrame, legend_info: list, out_dir: str, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")

    fig = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=None, annotate=True, black_bg=False,
    )

    for i, (site_label, base, light, deep) in enumerate(legend_info):
        site_df   = combined[combined["_csv_index"] == i]
        left_pts  = site_df[site_df["mni_hemisphere"] == "L"][["mni_mm_x","mni_mm_y","mni_mm_z"]].values
        right_pts = site_df[site_df["mni_hemisphere"] == "R"][["mni_mm_x","mni_mm_y","mni_mm_z"]].values
        if len(left_pts):
            disp.add_markers(left_pts,  marker_color=light, marker_size=MARKER_SIZE * 3)
        if len(right_pts):
            disp.add_markers(right_pts, marker_color=deep,  marker_size=MARKER_SIZE * 3)

    patches = []
    for site_label, base, light, deep in legend_info:
        patches += [
            mpatches.Patch(color=light, label=f"{site_label} L"),
            mpatches.Patch(color=deep,  label=f"{site_label} R"),
        ]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(legend_info) * 2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.05))

    path = os.path.join(out_dir, f"{prefix}_glass_brain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Static glass brain: %s", path)


# =============================================================================
# Inflated brain plot (stage 06 snapped coordinates)
# =============================================================================

def _plot_inflated(
    fs_combined: pd.DataFrame,
    legend_info: list,
    out_dir: str,
    prefix: str,
    mesh: str = "fsaverage",
) -> None:
    """
    Plot surface-snapped stimulation coordinates on fsaverage inflated brain.

    Uses the fs_x/fs_y/fs_z columns from targets_fsaverage.csv (which are
    pial surface coordinates) projected onto the inflated surface for display.
    The inflated surface shares the same vertex indices as the pial surface,
    so we can use the saved fs_vertex index to look up inflated coordinates
    directly — giving a clean snap with no additional nearest-vertex search.
    """
    import matplotlib
    matplotlib.use("Agg")

    log.info("Loading fsaverage surfaces for inflated brain plot (mesh=%s)...", mesh)

    fsaverage = datasets.fetch_surf_fsaverage(mesh)

    # We need inflated coords for display but pial vertex indices for mapping.
    # nilearn's plot_surf lets us pass coordinates directly via surf_mesh,
    # but the simplest approach is to use plot_surf_stat_map / plot_surf with
    # a per-vertex colour texture.
    from nilearn import surface as surf

    infl_lh_coords, infl_lh_faces = surf.load_surf_mesh(fsaverage["infl_left"])
    infl_rh_coords, infl_rh_faces = surf.load_surf_mesh(fsaverage["infl_right"])

    n_lh = len(infl_lh_coords)
    n_rh = len(infl_rh_coords)

    log.info("Inflated LH: %d vertices  RH: %d vertices", n_lh, n_rh)

    # Build a per-vertex colour texture for each hemisphere.
    # Background = light grey; stimulation vertices = site colour.
    BG_INTENSITY  = 0.82   # grey level for background vertices
    SPOT_RADIUS   = 3      # spread marker over nearest N-ring neighbours

    # We use a scalar texture and a custom colormap rather than per-vertex
    # RGB (nilearn's plot_surf accepts a 1D stat array).  Encoding: each
    # site gets a unique integer > 0; background = 0.

    lh_texture = np.zeros(n_lh, dtype=float)
    rh_texture = np.zeros(n_rh, dtype=float)

    # Build adjacency for spreading markers over a small neighbourhood
    lh_adj = _build_adjacency(infl_lh_faces, n_lh)
    rh_adj = _build_adjacency(infl_rh_faces, n_rh)

    site_vertex_map: dict[int, list[int]] = {}   # csv_index → vertex list

    for _, row in fs_combined.iterrows():
        vertex_idx = row.get("fs_vertex")
        surface    = row.get("fs_surface", "")
        csv_idx    = int(row.get("_csv_index", 0))

        if pd.isna(vertex_idx):
            continue

        vertex_idx = int(vertex_idx)
        val        = float(csv_idx + 1)   # 1-based so background stays 0

        if "left" in str(surface):
            neighbours = _k_ring(lh_adj, vertex_idx, SPOT_RADIUS)
            for v in neighbours:
                lh_texture[v] = val
        else:
            neighbours = _k_ring(rh_adj, vertex_idx, SPOT_RADIUS)
            for v in neighbours:
                rh_texture[v] = val

        site_vertex_map.setdefault(csv_idx, []).append(vertex_idx)

    n_sites = len(legend_info)

    # Build a colormap: index 0 = background grey, indices 1..n_sites = site colours
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap_colors = [(BG_INTENSITY, BG_INTENSITY, BG_INTENSITY, 1.0)]  # background
    for _, base, light, deep in legend_info:
        # Use the base colour for the surface marker (single colour per site,
        # hemisphere split is already conveyed by which surface it lands on)
        r, g, b = _hex_to_rgb(base)
        cmap_colors.append((r, g, b, 1.0))

    cmap   = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, n_sites + 1.5)
    norm   = BoundaryNorm(bounds, cmap.N)

    vmin, vmax = 0, n_sites

    # Four panels: LH lateral, LH medial, RH lateral, RH medial
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
        axes, titles, views
    ):
        plotting.plot_surf(
            infl_path,
            surf_map=texture,
            hemi=hemi,
            view=view,
            bg_map=sulc_path,
            bg_on_data=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=ax,
            figure=fig,
        )
        ax.set_title(title, fontsize=9, pad=2)

    # Legend
    patches = []
    for site_label, base, light, deep in legend_info:
        r, g, b = _hex_to_rgb(base)
        patches.append(
            mpatches.Patch(color=(r, g, b), label=site_label)
        )
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(legend_info),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    path = os.path.join(out_dir, f"{prefix}_inflated_brain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Inflated brain : %s", path)

    # Log per-site vertex counts
    for csv_idx, vertices in site_vertex_map.items():
        if csv_idx < len(legend_info):
            log.info(
                "  Site %-12s : %d points on surface",
                legend_info[csv_idx][0], len(vertices),
            )


def _build_adjacency(faces: np.ndarray, n_vertices: int) -> list[set]:
    """Build a vertex adjacency list from a triangle mesh face array."""
    adj: list[set] = [set() for _ in range(n_vertices)]
    for v0, v1, v2 in faces:
        adj[v0].add(v1); adj[v0].add(v2)
        adj[v1].add(v0); adj[v1].add(v2)
        adj[v2].add(v0); adj[v2].add(v1)
    return adj


def _k_ring(adj: list[set], seed: int, k: int) -> set[int]:
    """Return all vertices within k hops of seed (inclusive)."""
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
# Heatmap
# =============================================================================

def _build_heatmap(combined: pd.DataFrame, out_dir: str, prefix: str) -> None:
    from scipy.ndimage import gaussian_filter

    log.info("Building density heatmap (FWHM=%d mm)...", HEATMAP_FWHM_MM)

    mni_img    = datasets.load_mni152_template(resolution=1)
    mni_affine = mni_img.affine
    mni_shape  = mni_img.shape[:3]
    mni_inv    = np.linalg.inv(mni_affine)

    coords = combined[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values
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
    max_count = max(1, int(np.round(count_data.max())))

    def _save(img, vmin, vmax, path, label, tick_labels=None):
        fig_h = plt.figure(figsize=(14, 4), facecolor="white")
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
    tl_overlap = [str(v) for v in range(1, max_count + 1)] if max_count <= 10 else None
    _save(count_img, 1, max_count,
          os.path.join(out_dir, f"{prefix}_heatmap_overlap.png"),
          "N overlapping points", tl_overlap)

    shifted = np.where(count_data > 0, count_data + 1, 0).astype(np.float32)
    tl_all = [str(v) for v in range(1, max_count + 2)] if max_count <= 9 else None
    _save(nib.Nifti1Image(shifted, mni_affine), 1, max_count + 1,
          os.path.join(out_dir, f"{prefix}_heatmap_all.png"),
          "N points", tl_all)

    nii_path = os.path.join(out_dir, f"{prefix}_heatmap.nii.gz")
    nib.save(nib.Nifti1Image(density, mni_affine), nii_path)
    log.info("Heatmap NIfTI (FSLeyes): %s", nii_path)


# =============================================================================
# Atlas labelling
# =============================================================================

def _atlas_label(combined: pd.DataFrame, out_dir: str, prefix: str,
                 legend_info: list) -> pd.DataFrame:
    log.info("Fetching Harvard-Oxford cortical atlas (threshold=%d%%)...",
             HO_PROBABILITY_THRESHOLD)

    ho           = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", symmetric_split=True)
    atlas_img    = ho.maps
    atlas_data   = atlas_img.get_fdata().astype(int)
    atlas_inv    = np.linalg.inv(atlas_img.affine)
    atlas_labels = ho.labels

    def _lookup(mm):
        h   = np.array([mm[0], mm[1], mm[2], 1.0])
        vox = np.round((atlas_inv @ h)[:3]).astype(int)
        if not all(0 <= vox[d] < atlas_data.shape[d] for d in range(3)):
            return "out_of_bounds"
        pid = atlas_data[vox[0], vox[1], vox[2]]
        if pid == 0:
            return "no_region"
        lbl = atlas_labels[pid]
        return lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl

    combined = combined.copy()
    combined["ho_region"] = [
        _lookup([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])
        for _, row in combined.iterrows()
    ]

    region_counts = (combined[combined["ho_region"] != "no_region"]
                     ["ho_region"].value_counts())
    log.info("Top 10 regions by stimulation count:")
    for region, count in region_counts.head(10).items():
        log.info("  %4d  %s", count, region)

    targeted = sorted(
        combined[combined["ho_region"] != "no_region"]["ho_region"].dropna().unique()
    )
    n_regions = len(targeted)
    cmap_r    = cm.get_cmap("tab20", max(n_regions, 1))
    reg_colors = {r: cmap_r(i) for i, r in enumerate(targeted)}

    label_to_idx = {}
    for i, lbl in enumerate(atlas_labels):
        lbl_str = lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
        label_to_idx[lbl_str] = i

    fig_ho = plt.figure(figsize=(14, 4), facecolor="white")
    disp_ho = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig_ho,
        title=None, annotate=True, black_bg=False,
    )

    coords_all = combined[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values

    for region in targeted:
        idx = label_to_idx.get(region)
        if idx is None:
            continue
        region_vol = (atlas_data == idx).astype(np.float32)
        region_img = nib.Nifti1Image(region_vol, atlas_img.affine)
        disp_ho.add_contours(region_img, levels=[0.5],
                             colors=[reg_colors[region]], linewidths=1.5)

    for i, (site_label, base, light, deep) in enumerate(legend_info):
        site_mask = combined["_csv_index"].values == i
        hemi      = combined["hemisphere"].values
        left_pts  = coords_all[site_mask & (hemi == "L")]
        right_pts = coords_all[site_mask & (hemi == "R")]
        if len(left_pts):
            disp_ho.add_markers(left_pts,  marker_color=light, marker_size=MARKER_SIZE * 3)
        if len(right_pts):
            disp_ho.add_markers(right_pts, marker_color=deep,  marker_size=MARKER_SIZE * 3)

    region_handles = [
        mlines.Line2D([], [], color=reg_colors[r], linewidth=1.5, label=r)
        for r in targeted
    ]
    site_handles = []
    for site_label, base, light, deep in legend_info:
        site_handles += [
            mpatches.Patch(color=light, label=f"{site_label} L"),
            mpatches.Patch(color=deep,  label=f"{site_label} R"),
        ]

    leg1 = fig_ho.legend(handles=region_handles, loc="lower center",
                         ncol=min(n_regions, 6), fontsize=7, frameon=False,
                         bbox_to_anchor=(0.5, -0.08))
    fig_ho.add_artist(leg1)
    fig_ho.legend(handles=site_handles, loc="lower center",
                  ncol=len(legend_info) * 2, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.16))

    path = os.path.join(out_dir, f"{prefix}_ho_regions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_ho)
    log.info("HO regions plot: %s", path)

    return combined


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
        combined = _atlas_label(combined, args.output_dir, args.output_prefix, legend_info)

    # Inflated brain plot
    if args.fsaverage_csvs and not args.no_inflated:
        fs_combined = _load_fsaverage_csvs(args.fsaverage_csvs, site_labels)
        if fs_combined is not None and not fs_combined.empty:
            _plot_inflated(
                fs_combined, legend_info,
                args.output_dir, args.output_prefix,
                mesh=args.fsaverage_mesh,
            )
        else:
            log.warning("No valid fsaverage coordinates found — skipping inflated brain plot.")
    elif args.fsaverage_csvs and args.no_inflated:
        log.info("--no-inflated set — skipping inflated brain plot.")
    else:
        log.info(
            "No --fsaverage-csv provided — skipping inflated brain plot. "
            "Run stage 06 per subject and pass --fsaverage-csv to enable it."
        )

    # Output CSV
    out_cols = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z",
                "hemisphere", "mni_hemisphere"]
    if not args.no_atlas:
        out_cols.append("ho_region")
    out_cols = [c for c in out_cols if c in combined.columns]

    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_all_sites.csv")
    combined[out_cols].to_csv(csv_path, index=False)
    log.info("Combined CSV: %s", csv_path)

    log.info("=" * 60)
    log.info("Group visualization complete")
    log.info("Output directory: %s", args.output_dir)
    log.info("Total points: %d across %d site(s)", len(combined), len(args.csvs))
    for site_label, base, light, deep in legend_info:
        log.info("  %-12s  L=%s  R=%s", site_label, light, deep)


if __name__ == "__main__":
    main()