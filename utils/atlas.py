"""
utils/atlas.py
==============
Harvard-Oxford cortical atlas labelling utilities.

Shared between stage 05 (subject-level) and group_visualize.py (group-level).
Extracts region labels for MNI coordinates and generates a glass brain plot
with atlas contours overlaid.
"""

from __future__ import annotations

import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
import numpy as np
import pandas as pd

HO_PROBABILITY_THRESHOLD = 25


def label_coords(
    coords_mm: np.ndarray,
) -> list[str]:
    """
    Look up Harvard-Oxford cortical atlas region for each MNI mm coordinate.

    Parameters
    ----------
    coords_mm : Nx3 array of MNI RAS mm coordinates.

    Returns
    -------
    List of region name strings, one per row.
    'no_region' if coordinate falls outside any labelled region.
    'out_of_bounds' if coordinate is outside the atlas volume.
    """
    from nilearn import datasets

    ho          = datasets.fetch_atlas_harvard_oxford(
                      "cort-maxprob-thr25-1mm", symmetric_split=True)
    atlas_img   = ho.maps
    atlas_data  = atlas_img.get_fdata().astype(int)
    atlas_inv   = np.linalg.inv(atlas_img.affine)
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

    return [_lookup(mm) for mm in coords_mm]


def plot_ho_regions(
    coords_mm:   np.ndarray,
    hemi_labels: list[str],
    out_path:    str,
    color_left:  str = "#4488FF",
    color_right: str = "#FF4444",
    marker_size: float = 5,
    title:       str | None = None,
) -> pd.DataFrame:
    """
    Plot MNI coordinates on a glass brain with Harvard-Oxford atlas contours.

    Parameters
    ----------
    coords_mm   : Nx3 MNI RAS mm coordinates.
    hemi_labels : List of 'L' / 'R' strings, one per coordinate.
    out_path    : Full path for the output PNG.
    color_left  : Hex colour for left hemisphere points.
    color_right : Hex colour for right hemisphere points.
    marker_size : Marker size passed to nilearn add_markers.
    title       : Optional plot title.

    Returns
    -------
    DataFrame with columns [mni_mm_x, mni_mm_y, mni_mm_z, hemisphere, ho_region].
    """
    import matplotlib
    matplotlib.use("Agg")
    from nilearn import datasets, plotting

    ho           = datasets.fetch_atlas_harvard_oxford(
                       "cort-maxprob-thr25-1mm", symmetric_split=True)
    atlas_img    = ho.maps
    atlas_data   = atlas_img.get_fdata().astype(int)
    atlas_labels = ho.labels

    regions = label_coords(coords_mm)

    df = pd.DataFrame({
        "mni_mm_x":  coords_mm[:, 0],
        "mni_mm_y":  coords_mm[:, 1],
        "mni_mm_z":  coords_mm[:, 2],
        "hemisphere": hemi_labels,
        "ho_region":  regions,
    })

    targeted = sorted(
        df[~df["ho_region"].isin(["no_region", "out_of_bounds"])]["ho_region"].unique()
    )
    n_regions  = len(targeted)
    cmap_r     = cm.get_cmap("tab20", max(n_regions, 1))
    reg_colors = {r: cmap_r(i) for i, r in enumerate(targeted)}

    label_to_idx = {}
    for i, lbl in enumerate(atlas_labels):
        lbl_str = lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
        label_to_idx[lbl_str] = i

    if n_regions == 0:
        import logging
        logging.getLogger("tms2mni.utils.atlas").warning(
            "No HO atlas regions found for any coordinate — "
            "coordinates may be out of bounds or registration failed. "
            "Saving glass brain without atlas contours."
        )

    fig = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=title, annotate=True, black_bg=False,
    )

    # Atlas contours (skipped if no regions found)
    for region in targeted:
        idx = label_to_idx.get(region)
        if idx is None:
            continue
        region_vol = (atlas_data == idx).astype(np.float32)
        region_img = nib.Nifti1Image(region_vol, atlas_img.affine)
        disp.add_contours(region_img, levels=[0.5],
                          colors=[reg_colors[region]], linewidths=1.5)

    # Stimulation markers
    hemi_arr  = np.array(hemi_labels)
    left_pts  = coords_mm[hemi_arr == "L"]
    right_pts = coords_mm[hemi_arr == "R"]
    if len(left_pts):
        disp.add_markers(left_pts,  marker_color=color_left,
                         marker_size=marker_size * 3)
    if len(right_pts):
        disp.add_markers(right_pts, marker_color=color_right,
                         marker_size=marker_size * 3)

    # Legend — atlas regions (only if any found)
    site_handles = [
        mpatches.Patch(color=color_left,  label="Left"),
        mpatches.Patch(color=color_right, label="Right"),
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
                   ncol=2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.16))
    else:
        fig.legend(handles=site_handles, loc="lower center",
                   ncol=2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.05))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return df