"""
Multi-Site Group MNI Visualizer
=================================
Visualizes stimulation coordinates from multiple sites on a single MNI
glass brain. Each site CSV gets a distinct colour; L and R hemispheres
are shown as light and deep variants of that colour.

Also produces:
  - A heatmap overlay showing spatial density of stimulation points
  - A combined output CSV with Schaefer atlas region labels

Requirements:
    pip install nilearn nibabel numpy pandas scipy matplotlib

Usage:
    python visualize_group_mni.py \
        --csv site1.csv \
        --csv site2.csv \
        --csv site3.csv \
        --output-dir ./group_viz \
        --output-prefix group_sites

    # Skip heatmap or atlas labelling if not needed
    python visualize_group_mni.py \
        --csv site1.csv \
        --output-dir ./group_viz \
        --no-heatmap \
        --no-atlas
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# =============================================================================
# USER CONFIGURATION — edit these to customise appearance
# =============================================================================

# Base colours per site (up to 5). Light variant = L hemisphere, deep = R.
# Add or change hex codes here. Must have at least as many entries as --csv args.
SITE_BASE_COLORS = [
    "#2196F3",   # site 1 — blue
    "#F44336",   # site 2 — red
    "#4CAF50",   # site 3 — green
    "#FF9800",   # site 4 — orange
    "#9C27B0",   # site 5 — purple
]

# Light/deep multipliers for L and R hemispheres.
# Light = higher brightness (L), Deep = lower brightness (R).
# Values between 0.0 (black) and 1.5 (washed out).
LIGHT_FACTOR = 1.45   # L hemisphere — light variant
DEEP_FACTOR  = 0.55   # R hemisphere — deep variant

# Marker size for individual site points
MARKER_SIZE = 5

# Heatmap: FWHM of the Gaussian kernel in mm.
# Keep small — these are 1x1x1mm points. 4-6mm is appropriate.
HEATMAP_FWHM_MM = 5

# Heatmap colormap (any matplotlib colormap name)
HEATMAP_CMAP = "Reds"   # overridden below with transparent-zero variant

# Harvard-Oxford cortical atlas
HO_PROBABILITY_THRESHOLD = 25   # % threshold for region membership (standard: 25)

# =============================================================================
# HELPERS
# =============================================================================

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    r = min(1.0, max(0.0, r))
    g = min(1.0, max(0.0, g))
    b = min(1.0, max(0.0, b))
    return "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))

def make_variant(hex_color, factor):
    """Scale RGB brightness by factor to make light or deep variant."""
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex(r * factor, g * factor, b * factor)

def get_site_colors(base_hex):
    """Return (light_hex, deep_hex) for a base colour."""
    return make_variant(base_hex, LIGHT_FACTOR), make_variant(base_hex, DEEP_FACTOR)

# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Multi-site group MNI visualization with heatmap and atlas labelling."
)
parser.add_argument("--csv",           required=True, action="append", dest="csvs",
                    help="Path to a site CSV. Repeat for each site: --csv s1.csv --csv s2.csv")
parser.add_argument("--output-dir",    required=True,
                    help="Directory to save all outputs")
parser.add_argument("--output-prefix", default="group",
                    help="Prefix for output filenames (default: group)")
parser.add_argument("--no-heatmap",    action="store_true",
                    help="Skip heatmap generation")
parser.add_argument("--no-atlas",      action="store_true",
                    help="Skip Schaefer atlas labelling")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

if len(args.csvs) > len(SITE_BASE_COLORS):
    print(f"ERROR: {len(args.csvs)} CSVs provided but only {len(SITE_BASE_COLORS)} colours "
          f"defined in SITE_BASE_COLORS. Add more colours to the config.")
    sys.exit(1)

# =============================================================================
# LOAD ALL SITE CSVs
# =============================================================================

REQUIRED_COLS = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z", "hemisphere"]

all_dfs = []
site_labels = []

print(f"\n[Load] Reading {len(args.csvs)} site CSV(s)...")

for i, csv_path in enumerate(args.csvs):
    df = pd.read_csv(csv_path)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in {csv_path}")
            print(f"  Available: {list(df.columns)}")
            sys.exit(1)

    df = df.dropna(subset=["mni_mm_x", "mni_mm_y", "mni_mm_z"])
    df["_csv_index"] = i

    # Infer site label from 'site' column or filename
    site_name = df["site"].iloc[0] if "site" in df.columns and not df.empty else f"site{i+1}"
    site_labels.append(site_name)

    all_dfs.append(df)
    print(f"  Site {i+1} ({site_name}): {len(df)} points from {csv_path}")

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n[Load] Total: {len(combined_df)} points across {len(args.csvs)} sites")

# hemisphere col = patient orientation (L/R as labelled by NextStim/operator)
# mni_hemisphere = neurological convention (L = negative MNI X, R = positive MNI X)
combined_df["mni_hemisphere"] = combined_df["mni_mm_x"].apply(
    lambda x: "L" if x < 0 else "R"
)

# =============================================================================
# BUILD COLOUR ARRAYS
# =============================================================================

point_colors = []
legend_info  = []   # (site_label, base_color, light_hex, deep_hex)

for i, base in enumerate(SITE_BASE_COLORS[:len(args.csvs)]):
    light, deep = get_site_colors(base)
    legend_info.append((site_labels[i], base, light, deep))

for _, row in combined_df.iterrows():
    i = int(row["_csv_index"])
    light, deep = get_site_colors(SITE_BASE_COLORS[i])
    color = light if row["hemisphere"] == "L" else deep
    point_colors.append(color)

coords_all = combined_df[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values

# =============================================================================
# PLOT 1: Multi-site glass brain — interactive HTML
# =============================================================================

from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np

print(f"\n[Plot] Building interactive multi-site glass brain...")

view = plotting.view_markers(
    coords_all,
    marker_color = point_colors,
    marker_size  = MARKER_SIZE,
)

html_path = os.path.join(args.output_dir, f"{args.output_prefix}_interactive.html")
view.save_as_html(html_path)
print(f"  Saved: {html_path}")

# =============================================================================
# PLOT 2: Static 4-panel glass brain (lyrz) — per site with L/R colours
# =============================================================================

print(f"\n[Plot] Generating static 4-panel glass brain...")

fig = plt.figure(figsize=(14, 4), facecolor="white")

disp = plotting.plot_glass_brain(
    None,
    display_mode = "lyrz",
    figure       = fig,
    title        = None,
    annotate     = True,
    black_bg     = False,
)

for i, (site_label, base, light, deep) in enumerate(legend_info):
    site_df = combined_df[combined_df["_csv_index"] == i]
    left_df  = site_df[site_df["mni_hemisphere"] == "L"]
    right_df = site_df[site_df["mni_hemisphere"] == "R"]

    if len(left_df) > 0:
        disp.add_markers(left_df[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values,
                         marker_color=light,
                         marker_size=MARKER_SIZE * 3)
    if len(right_df) > 0:
        disp.add_markers(right_df[["mni_mm_x", "mni_mm_y", "mni_mm_z"]].values,
                         marker_color=deep,
                         marker_size=MARKER_SIZE * 3)

# Legend
patches = []
for site_label, base, light, deep in legend_info:
    patches.append(mpatches.Patch(color=light, label=f"{site_label} L"))
    patches.append(mpatches.Patch(color=deep,  label=f"{site_label} R"))

fig.legend(handles=patches, loc="lower center", ncol=len(legend_info) * 2,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

png_path = os.path.join(args.output_dir, f"{args.output_prefix}_glass_brain.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {png_path}")

# =============================================================================
# PLOT 3: Heatmap — Gaussian kernel density as statistical overlay
# =============================================================================

if not args.no_heatmap:
    print(f"\n[Heatmap] Building density map (FWHM={HEATMAP_FWHM_MM}mm)...")

    from nilearn.image import smooth_img
    from scipy.ndimage import gaussian_filter

    # Load MNI152 template as reference space
    mni_img    = datasets.load_mni152_template(resolution=1)
    mni_affine = mni_img.affine
    mni_shape  = mni_img.shape[:3]
    mni_inv    = np.linalg.inv(mni_affine)

    # Convert MNI mm -> voxel indices
    coords_h = np.hstack([coords_all, np.ones((len(coords_all), 1))])
    vox_coords = (mni_inv @ coords_h.T).T[:, :3]
    vox_coords = np.round(vox_coords).astype(int)

    # Build density volume — place a 1 at each stimulation voxel
    density = np.zeros(mni_shape, dtype=np.float32)
    for vx, vy, vz in vox_coords:
        if (0 <= vx < mni_shape[0] and
            0 <= vy < mni_shape[1] and
            0 <= vz < mni_shape[2]):
            density[vx, vy, vz] += 1.0

    # Smooth with Gaussian kernel (convert FWHM to sigma: sigma = FWHM / 2.355)
    vox_size = abs(mni_affine[0, 0])
    sigma    = (HEATMAP_FWHM_MM / 2.355) / vox_size
    density  = gaussian_filter(density, sigma=sigma)

    # Threshold: only show voxels above a small fraction of max to reduce noise
    threshold = density.max() * 0.05
    density[density < threshold] = 0

    density_img = nib.Nifti1Image(density, mni_affine)

    # Rescale density to approximate point counts:
    # divide by the volume under a single Gaussian kernel so values reflect
    # "number of overlapping points" rather than raw smoothed intensity.
    from scipy.ndimage import gaussian_filter as gf
    unit = np.zeros(mni_shape, dtype=np.float32)
    cx, cy, cz = [s // 2 for s in mni_shape]
    unit[cx, cy, cz] = 1.0
    unit_smoothed = gf(unit, sigma=sigma)
    kernel_sum = unit_smoothed.max()
    if kernel_sum > 0:
        count_data = density / kernel_sum
    else:
        count_data = density.copy()
    count_data[count_data < 0.5] = 0   # threshold below half a point
    count_img = nib.Nifti1Image(count_data, mni_affine)

    max_count = max(1, int(np.round(count_data.max())))

    def save_heatmap(img, vmin, vmax, out_path, cbar_label, tick_labels=None):
        """Simple clean heatmap using nilearn with vmin=1 so background is white."""
        fig_h = plt.figure(figsize=(14, 4), facecolor="white")
        disp_h = plotting.plot_glass_brain(
            img,
            display_mode = "lyrz",
            colorbar     = True,
            cmap         = "Reds",
            figure       = fig_h,
            title        = None,
            annotate     = True,
            black_bg     = False,
            vmin         = vmin,
            vmax         = vmax,
        )
        if disp_h._cbar is not None and tick_labels is not None:
            ticks = np.linspace(vmin, vmax, len(tick_labels))
            disp_h._cbar.set_ticks(ticks)
            disp_h._cbar.set_ticklabels(tick_labels)
            disp_h._cbar.set_label(cbar_label, fontsize=9)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_h)
        print(f"  Saved: {out_path}")

    heatmap_path     = os.path.join(args.output_dir, f"{args.output_prefix}_heatmap_overlap.png")
    heatmap_all_path = os.path.join(args.output_dir, f"{args.output_prefix}_heatmap_all.png")

    # Overlap map: vmin=1 so white background, only overlapping points coloured
    tick_labels_overlap = [str(v) for v in range(1, max_count + 1)] if max_count <= 10 else None
    save_heatmap(count_img, vmin=1, vmax=max_count,
                 out_path=heatmap_path, cbar_label="N overlapping points",
                 tick_labels=tick_labels_overlap)

    # All-points map: unthresholded data, vmin=1 (single points = lowest colour),
    # tick labels shifted +1 so they read 1,2,3... not 0,1,2...
    count_data_all = count_data.copy()
    count_data_all[count_data_all > 0] = count_data_all[count_data_all > 0]  # keep as-is
    # Shift values up by 1 so that any point > 0 maps to >= 1 on the colorscale
    count_data_shifted = np.where(count_data_all > 0, count_data_all + 1, 0)
    count_img_shifted  = nib.Nifti1Image(count_data_shifted.astype(np.float32), mni_affine)
    vmax_shifted = max_count + 1
    tick_labels_all = [str(v) for v in range(1, max_count + 2)] if max_count <= 9 else None
    save_heatmap(count_img_shifted, vmin=1, vmax=vmax_shifted,
                 out_path=heatmap_all_path, cbar_label="N points",
                 tick_labels=tick_labels_all)

    # Also save density NIfTI for use in other viewers
    heatmap_nii_path = os.path.join(args.output_dir, f"{args.output_prefix}_heatmap.nii.gz")
    nib.save(density_img, heatmap_nii_path)
    print(f"  Saved: {heatmap_nii_path}  (load in FSLeyes for interactive inspection)")

# =============================================================================
# ATLAS LABELLING — Harvard-Oxford Cortical
# =============================================================================

if not args.no_atlas:
    print(f"\n[Atlas] Fetching Harvard-Oxford cortical atlas "
          f"(threshold={HO_PROBABILITY_THRESHOLD}%)...")

    ho = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-1mm",
        symmetric_split=True,
    )

    atlas_img    = ho.maps   # already a Nifti1Image
    atlas_data   = atlas_img.get_fdata().astype(int)
    atlas_affine = atlas_img.affine
    atlas_inv    = np.linalg.inv(atlas_affine)
    atlas_labels = ho.labels   # index 0 = Background

    def lookup_ho(mm_coord):
        """Return Harvard-Oxford region label for a given MNI mm coordinate."""
        h   = np.array([mm_coord[0], mm_coord[1], mm_coord[2], 1.0])
        vox = np.round((atlas_inv @ h)[:3]).astype(int)
        if not all(0 <= vox[d] < atlas_data.shape[d] for d in range(3)):
            return "out_of_bounds"
        parcel_id = atlas_data[vox[0], vox[1], vox[2]]
        if parcel_id == 0:
            return "no_region"
        label = atlas_labels[parcel_id]
        return label.decode("utf-8") if isinstance(label, bytes) else label

    print(f"  Labelling {len(combined_df)} points...")
    combined_df["ho_region"] = [
        lookup_ho([row["mni_mm_x"], row["mni_mm_y"], row["mni_mm_z"]])
        for _, row in combined_df.iterrows()
    ]

    region_counts = (combined_df[combined_df["ho_region"] != "no_region"]
                     ["ho_region"].value_counts())
    print(f"\n  Top 10 regions by stimulation count:")
    for region, count in region_counts.head(10).items():
        print(f"    {count:4d}  {region}")

    # -------------------------------------------------------------------------
    # PLOT: Harvard-Oxford region outlines + stimulation points (single 4-panel)
    # -------------------------------------------------------------------------
    targeted_regions = sorted(
        combined_df[combined_df["ho_region"] != "no_region"]["ho_region"]
        .dropna().unique()
    )
    print(f"\n[Plot] Harvard-Oxford outline panel ({len(targeted_regions)} regions)...")

    # Build a label->index lookup
    label_to_idx = {}
    for i, lbl in enumerate(atlas_labels):
        lbl_str = lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
        label_to_idx[lbl_str] = i

    # Generate a distinct colour per targeted region using a qualitative colormap
    import matplotlib.cm as cm
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    n_regions   = len(targeted_regions)
    cmap        = cm.get_cmap("tab20", max(n_regions, 1))
    region_colors = {r: cmap(i) for i, r in enumerate(targeted_regions)}

    fig_ho = plt.figure(figsize=(14, 4), facecolor="white")
    disp_ho = plotting.plot_glass_brain(
        None,
        display_mode = "lyrz",
        figure       = fig_ho,
        title        = None,
        annotate     = True,
        black_bg     = False,
    )

    # Draw one contour per targeted region in its own colour
    for region in targeted_regions:
        idx = label_to_idx.get(region)
        if idx is None:
            continue
        region_vol = (atlas_data == idx).astype(np.float32)
        region_img = nib.Nifti1Image(region_vol, atlas_affine)
        color      = region_colors[region]
        disp_ho.add_contours(
            region_img,
            levels     = [0.5],
            colors     = [color],
            linewidths = 1.5,
        )

    # Overlay stimulation points on top
    for i, (site_label, base, light, deep) in enumerate(legend_info):
        site_df   = combined_df[combined_df["_csv_index"] == i]
        left_pts  = coords_all[combined_df["_csv_index"].values == i][site_df["hemisphere"].values == "L"]
        right_pts = coords_all[combined_df["_csv_index"].values == i][site_df["hemisphere"].values == "R"]
        if len(left_pts):
            disp_ho.add_markers(left_pts,  marker_color=light, marker_size=MARKER_SIZE*3)
        if len(right_pts):
            disp_ho.add_markers(right_pts, marker_color=deep,  marker_size=MARKER_SIZE*3)

    # Legend: regions as coloured lines + sites as patches
    region_handles = [
        mlines.Line2D([], [], color=region_colors[r], linewidth=1.5, label=r)
        for r in targeted_regions
    ]
    site_handles = []
    for site_label, base, light, deep in legend_info:
        site_handles += [mpatches.Patch(color=light, label=f"{site_label} L"),
                         mpatches.Patch(color=deep,  label=f"{site_label} R")]

    # Two-part legend: regions on top, sites below
    leg1 = fig_ho.legend(handles=region_handles,
                         loc="lower center", ncol=min(n_regions, 6),
                         fontsize=7, frameon=False,
                         bbox_to_anchor=(0.5, -0.08))
    fig_ho.add_artist(leg1)
    fig_ho.legend(handles=site_handles,
                  loc="lower center", ncol=len(legend_info)*2,
                  fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.16))

    ho_png_path = os.path.join(args.output_dir, f"{args.output_prefix}_ho_regions.png")
    plt.savefig(ho_png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_ho)
    print(f"  Saved: {ho_png_path}")

# =============================================================================
# OUTPUT CSV
# =============================================================================

out_cols = ["subject_id", "site", "mni_mm_x", "mni_mm_y", "mni_mm_z",
            "hemisphere", "mni_hemisphere"]
if "id" in combined_df.columns:
    out_cols = ["subject_id", "site", "id"] + [c for c in out_cols if c not in ["subject_id", "site"]]
if not args.no_atlas:
    out_cols.append("ho_region")

# Drop internal working columns
out_df = combined_df[[c for c in out_cols if c in combined_df.columns]].copy()

csv_out_path = os.path.join(args.output_dir, f"{args.output_prefix}_all_sites.csv")
out_df.to_csv(csv_out_path, index=False)
print(f"\n[Output] Combined CSV: {csv_out_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"""
{'='*60}
Group visualization complete
{'='*60}
Output directory: {args.output_dir}

  {args.output_prefix}_interactive.html   — interactive 3D glass brain
  {args.output_prefix}_glass_brain.png    — static 4-panel (L/coronal/axial/R)
  {args.output_prefix}_all_sites.csv      — combined coords + atlas labels""")

if not args.no_heatmap:
    print(f"  {args.output_prefix}_heatmap_overlap.png  — density, thresholded at 1 (overlapping points only)")
    print(f"  {args.output_prefix}_heatmap_all.png      — density, from 0 (all points)")
    print(f"  {args.output_prefix}_heatmap.nii.gz       — density volume (FSLeyes)")

print(f"""
Colour legend:""")
for site_label, base, light, deep in legend_info:
    print(f"  {site_label:<12}  L={light}  R={deep}")

print(f"""
Atlas: Harvard-Oxford cortical (threshold={HO_PROBABILITY_THRESHOLD}%)
Total points: {len(combined_df)}
{'='*60}
""")