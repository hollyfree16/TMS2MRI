"""
NextStim MNI Coordinate Visualizer
===================================
Plots selected stimulation sites from targets_mni.csv as an interactive
3D glass brain in the browser using nilearn.plotting.view_markers.

Coordinates are plotted directly in MNI152 mm space — no fsaverage,
no surface extraction required.

Requirements:
    pip install nilearn

Usage:
    # Plot specific IDs
    python visualize_mni.py \
        --csv targets_mni.csv \
        --id 1.26.23 \
        --id 1.48.18

    # Plot all IDs
    python visualize_mni.py \
        --csv targets_mni.csv \
        --id all

    # Plot coil locations instead of EF
    python visualize_mni.py \
        --csv targets_mni.csv \
        --id 1.26.23 \
        --coord-col coil

    # Save to HTML file instead of opening browser
    python visualize_mni.py \
        --csv targets_mni.csv \
        --id all \
        --output stimulation_sites.html

    # Colour points by hemisphere (L=blue, R=red)
    python visualize_mni.py \
        --csv targets_mni.csv \
        --id all \
        --color-by-hemi
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Visualize NextStim MNI coordinates as interactive 3D glass brain."
)
parser.add_argument("--csv",           required=True,
                    help="Path to targets_mni.csv")
parser.add_argument("--id",            required=True, action="append", dest="ids",
                    help="Row ID to plot. Repeat for multiple: --id 1.26.23 --id 1.48.18. "
                         "Use --id all to plot everything.")
parser.add_argument("--coord-col",     default="ef", choices=["ef", "coil"],
                    help="Which coordinates to plot: ef (default) or coil.")
parser.add_argument("--marker-size",   type=float, default=5,
                    help="Marker size (default: 5).")
parser.add_argument("--color",         default=None,
                    help="Marker colour (e.g. 'red', '#FF4444'). "
                         "Ignored if --color-by-hemi is set.")
parser.add_argument("--color-by-hemi", action="store_true",
                    help="Colour points by hemisphere: L=blue, R=red.")
parser.add_argument("--output",        default=None,
                    help="Save interactive HTML to this path. "
                         "If not provided, opens in browser.")
parser.add_argument("--screenshot",    default=None,
                    help="Save static top-down PNG to this path (e.g. plot.png). "
                         "Uses plot_glass_brain with axial view.")
parser.add_argument("--subject-id",    default=None,
                    help="Subject identifier to record in shared CSV (e.g. TEP004).")
parser.add_argument("--site",          default=None,
                    help="Stimulation site label to record in shared CSV (e.g. M1_L).")
parser.add_argument("--shared-csv",    default=None,
                    help="Path to shared CSV file to append results to. "
                         "Created if it does not exist.")
args = parser.parse_args()

# =============================================================================
# LOAD AND SELECT
# =============================================================================

print(f"[Load] Reading {args.csv}...")
df = pd.read_csv(args.csv)

prefix   = "ef_mni"   if args.coord_col == "ef"   else "coil_mni"
mm_x_col = f"{prefix}_mm_x"
mm_y_col = f"{prefix}_mm_y"
mm_z_col = f"{prefix}_mm_z"
hemi_col = "ef_hemisphere" if args.coord_col == "ef" else "coil_hemisphere"

for col in [mm_x_col, mm_y_col, mm_z_col, hemi_col]:
    if col not in df.columns:
        print(f"ERROR: Column '{col}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

# Normalise IDs: strip whitespace and trailing dots
def normalise_id(s):
    return str(s).strip().rstrip(".")

df["_id_norm"] = df["id"].apply(normalise_id)

if len(args.ids) == 1 and args.ids[0].strip().lower() == "all":
    selected = df.dropna(subset=[mm_x_col, mm_y_col, mm_z_col])
    print(f"[Selection] All rows: {len(selected)} points")
else:
    id_list_norm = [normalise_id(x) for x in args.ids]
    selected     = df[df["_id_norm"].isin(id_list_norm)]

    if selected.empty:
        print(f"ERROR: None of the provided IDs found in the 'id' column.")
        print(f"First few IDs in file: {df['id'].head(10).tolist()}")
        sys.exit(1)

    missing = [i for i in id_list_norm if i not in df["_id_norm"].values]
    if missing:
        print(f"[Warning] IDs not found: {missing}")

    selected = selected.dropna(subset=[mm_x_col, mm_y_col, mm_z_col])
    print(f"[Selection] {len(selected)} points selected")

if selected.empty:
    print("ERROR: No valid coordinates to plot after filtering NaNs.")
    sys.exit(1)

coords = selected[[mm_x_col, mm_y_col, mm_z_col]].values

n_left  = (selected[hemi_col] == "L").sum()
n_right = (selected[hemi_col] == "R").sum()
print(f"  Left hemisphere : {n_left} points")
print(f"  Right hemisphere: {n_right} points")

# =============================================================================
# COLOURS
# =============================================================================

if args.color_by_hemi:
    colors = ["#4488FF" if h == "L" else "#FF4444"
              for h in selected[hemi_col]]
    print(f"[Colour] By hemisphere — L=blue (#4488FF), R=red (#FF4444)")
elif args.color:
    colors = args.color
    print(f"[Colour] {args.color}")
else:
    colors = "#FF4444"
    print(f"[Colour] Default red (#FF4444)")

# =============================================================================
# PLOT
# =============================================================================

from nilearn import plotting
import matplotlib.pyplot as plt

coord_label = "EF Max Location" if args.coord_col == "ef" else "Coil Location"
title = f"{coord_label}  |  n = {len(selected)}"

# --- Interactive HTML ---
print(f"\n[Plot] Building interactive 3D glass brain...")

view = plotting.view_markers(
    coords,
    marker_color = colors,
    marker_size  = args.marker_size,
)

if args.output:
    view.save_as_html(args.output)
    print(f"[HTML]  Saved to: {args.output}")
    print(f"        Open in any browser to view interactively.")
else:
    print(f"[HTML]  Opening in browser...")
    view.open_in_browser()

# --- Static PNG screenshot (top-down axial view) ---
if args.screenshot:
    print(f"\n[PNG]  Generating static top-down glass brain...")

    # plot_glass_brain needs a single colour string or per-point not supported
    # directly — use the first colour if list, otherwise use as-is
    dot_color = colors[0] if isinstance(colors, list) else colors

    # If colouring by hemisphere, plot L and R separately with their colours
    fig = plt.figure(figsize=(14, 4), facecolor="white")

    # display_mode='lyrz': Left sagittal, Coronal, Axial, Right sagittal
    disp = plotting.plot_glass_brain(
        None,
        display_mode = "lyrz",
        figure       = fig,
        title        = None,
        annotate     = True,
        black_bg     = False,
    )

    if args.color_by_hemi:
        left_sel  = selected[selected[hemi_col] == "L"]
        right_sel = selected[selected[hemi_col] == "R"]
        if len(left_sel) > 0:
            disp.add_markers(left_sel[[mm_x_col, mm_y_col, mm_z_col]].values,
                             marker_color="#4488FF",
                             marker_size=args.marker_size * 3)
        if len(right_sel) > 0:
            disp.add_markers(right_sel[[mm_x_col, mm_y_col, mm_z_col]].values,
                             marker_color="#FF4444",
                             marker_size=args.marker_size * 3)
    else:
        disp.add_markers(coords,
                         marker_color=dot_color,
                         marker_size=args.marker_size * 3)

    plt.savefig(args.screenshot, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[PNG]   Saved to: {args.screenshot}")

print(f"\n[Done] {len(selected)} {coord_label} points in MNI152 space.")
if args.color_by_hemi:
    print(f"       Blue = MRI Left  (NextStim Right ear side)")
    print(f"       Red  = MRI Right (NextStim Left ear side)")

# =============================================================================
# SHARED CSV — append selected points with subject/site metadata
# =============================================================================

if args.shared_csv:
    if args.subject_id is None or args.site is None:
        print(f"\n[Warning] --shared-csv requires both --subject-id and --site. Skipping.")
    else:
        rows = []
        for _, row in selected.iterrows():
            rows.append({
                "subject_id":    args.subject_id,
                "site":          args.site,
                "id":            normalise_id(row["id"]),
                "mni_mm_x":      round(row[mm_x_col], 2),
                "mni_mm_y":      round(row[mm_y_col], 2),
                "mni_mm_z":      round(row[mm_z_col], 2),
                "hemisphere":    row[hemi_col],
                "mni_hemisphere": "L" if row[mm_x_col] < 0 else "R",
            })
        new_rows_df = pd.DataFrame(rows)

        if os.path.exists(args.shared_csv):
            existing = pd.read_csv(args.shared_csv)
            combined = pd.concat([existing, new_rows_df], ignore_index=True)
        else:
            combined = new_rows_df

        combined.to_csv(args.shared_csv, index=False)
        print(f"\n[Shared CSV] {len(rows)} rows written to: {args.shared_csv}")