"""
utils/atlas.py
==============
Multi-atlas labelling and glass brain plotting utilities.

Atlases supported
-----------------
  harvard_oxford   Harvard-Oxford cortical (cort-maxprob-thr25-1mm, symmetric)
  aal              AAL (Automated Anatomical Labeling) 116 regions
  destrieux        Destrieux 2009 cortical parcellation (148 regions)
  schaefer_100     Schaefer 2018 100-parcel 7-network
  schaefer_200     Schaefer 2018 200-parcel 7-network
  schaefer_300     Schaefer 2018 300-parcel 7-network
  schaefer_400     Schaefer 2018 400-parcel 7-network
  schaefer_500     Schaefer 2018 500-parcel 7-network
  schaefer_600     Schaefer 2018 600-parcel 7-network
  schaefer_700     Schaefer 2018 700-parcel 7-network
  schaefer_800     Schaefer 2018 800-parcel 7-network
  schaefer_900     Schaefer 2018 900-parcel 7-network
  schaefer_1000    Schaefer 2018 1000-parcel 7-network
  yeo_7            Yeo 2011 7-network parcellation
  yeo_17           Yeo 2011 17-network parcellation

Each atlas produces:
  - A column in targets_summary.csv / targets_mni.csv
  - A glass brain PNG with atlas contours overlaid

Usage
-----
  from utils.atlas import label_coords_all, plot_all_atlases

  # Label a set of MNI RAS mm coordinates against all atlases
  labels = label_coords_all(coords_mm)   # dict of atlas_key → list of str

  # Plot glass brain for each atlas
  plot_all_atlases(coords_mm, hemi_labels, out_dir, color_left, color_right)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger("tms2mni.utils.atlas")

# ---------------------------------------------------------------------------
# Atlas registry
# ---------------------------------------------------------------------------

# All atlas keys in display order
ATLAS_KEYS = [
    "harvard_oxford",
    "aal",
    "destrieux",
    "schaefer_100",
    "schaefer_200",
    "schaefer_300",
    "schaefer_400",
    "schaefer_500",
    "schaefer_600",
    "schaefer_700",
    "schaefer_800",
    "schaefer_900",
    "schaefer_1000",
    "yeo_7",
    "yeo_17",
]

# Human-readable display names
ATLAS_NAMES = {
    "harvard_oxford":  "Harvard-Oxford",
    "aal":             "AAL",
    "destrieux":       "Destrieux",
    "schaefer_100":    "Schaefer 100",
    "schaefer_200":    "Schaefer 200",
    "schaefer_300":    "Schaefer 300",
    "schaefer_400":    "Schaefer 400",
    "schaefer_500":    "Schaefer 500",
    "schaefer_600":    "Schaefer 600",
    "schaefer_700":    "Schaefer 700",
    "schaefer_800":    "Schaefer 800",
    "schaefer_900":    "Schaefer 900",
    "schaefer_1000":   "Schaefer 1000",
    "yeo_7":           "Yeo 7-network",
    "yeo_17":          "Yeo 17-network",
}

# Output PNG filename stems
ATLAS_PNG_STEMS = {
    "harvard_oxford":  "ho_regions",
    "aal":             "aal_regions",
    "destrieux":       "destrieux_regions",
    "schaefer_100":    "schaefer_100_regions",
    "schaefer_200":    "schaefer_200_regions",
    "schaefer_300":    "schaefer_300_regions",
    "schaefer_400":    "schaefer_400_regions",
    "schaefer_500":    "schaefer_500_regions",
    "schaefer_600":    "schaefer_600_regions",
    "schaefer_700":    "schaefer_700_regions",
    "schaefer_800":    "schaefer_800_regions",
    "schaefer_900":    "schaefer_900_regions",
    "schaefer_1000":   "schaefer_1000_regions",
    "yeo_7":           "yeo_7_regions",
    "yeo_17":          "yeo_17_regions",
}

# CSV column names
ATLAS_COL_NAMES = {k: f"region_{k}" for k in ATLAS_KEYS}


# ---------------------------------------------------------------------------
# Atlas loader (cached per process)
# ---------------------------------------------------------------------------

_atlas_cache: dict = {}


def _load_atlas(key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load atlas and return (atlas_data, affine, labels).
    Results are cached after the first call.
    """
    if key in _atlas_cache:
        return _atlas_cache[key]

    from nilearn import datasets

    try:
        if key == "harvard_oxford":
            ho = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-1mm", symmetric_split=True)
            img    = ho.maps
            labels = [l.decode("utf-8") if isinstance(l, bytes) else l
                      for l in ho.labels]

        elif key == "aal":
            aal    = datasets.fetch_atlas_aal()
            img    = nib.load(aal.maps)
            labels = ["background"] + list(aal.labels)

        elif key == "destrieux":
            dest   = datasets.fetch_atlas_destrieux_2009(lateralized=True)
            img    = nib.load(dest.maps)
            labels = [l.decode("utf-8") if isinstance(l, bytes) else l
                      for l in dest.labels]

        elif key.startswith("schaefer_"):
            n      = int(key.split("_")[1])
            sch    = datasets.fetch_atlas_schaefer_2018(n_rois=n)
            img    = nib.load(sch.maps)
            labels = ["background"] + [
                l.decode("utf-8") if isinstance(l, bytes) else l
                for l in sch.labels
            ]

        elif key == "yeo_7":
            try:
                # nilearn >= 0.10: returns Atlas object with .maps
                yeo  = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
                img  = nib.load(yeo.maps) if isinstance(yeo.maps, str) else yeo.maps
                if hasattr(yeo, "labels") and yeo.labels:
                    labels = ["background"] + list(yeo.labels)
                else:
                    labels = [
                        "background", "Visual", "Somatomotor",
                        "Dorsal Attention", "Ventral Attention",
                        "Limbic", "Frontoparietal", "Default",
                    ]
            except TypeError:
                # older nilearn: fetch_atlas_yeo_2011() returns Bunch with thick_7
                yeo  = datasets.fetch_atlas_yeo_2011()
                img  = nib.load(yeo.thick_7)
                labels = [
                    "background", "Visual", "Somatomotor",
                    "Dorsal Attention", "Ventral Attention",
                    "Limbic", "Frontoparietal", "Default",
                ]

        elif key == "yeo_17":
            try:
                yeo  = datasets.fetch_atlas_yeo_2011(n_networks=17, thickness="thick")
                img  = nib.load(yeo.maps) if isinstance(yeo.maps, str) else yeo.maps
                if hasattr(yeo, "labels") and yeo.labels:
                    labels = ["background"] + list(yeo.labels)
                else:
                    labels = [
                        "background",
                        "Visual A", "Visual B",
                        "Somatomotor A", "Somatomotor B",
                        "Dorsal Attention A", "Dorsal Attention B",
                        "Salience/Ventral Attention A", "Salience/Ventral Attention B",
                        "Limbic A", "Limbic B",
                        "Control A", "Control B", "Control C",
                        "Default A", "Default B", "Default C",
                    ]
            except TypeError:
                yeo  = datasets.fetch_atlas_yeo_2011()
                img  = nib.load(yeo.thick_17)
                labels = [
                    "background",
                    "Visual A", "Visual B",
                    "Somatomotor A", "Somatomotor B",
                "Dorsal Attention A", "Dorsal Attention B",
                "Salience/Ventral Attention A", "Salience/Ventral Attention B",
                "Limbic A", "Limbic B",
                "Control A", "Control B", "Control C",
                "Default A", "Default B", "Default C",
            ]

        else:
            raise ValueError(f"Unknown atlas key: {key!r}")

        if isinstance(img, (str, Path)):
            img = nib.load(str(img))

        data   = img.get_fdata().astype(int)
        affine = img.affine
        _atlas_cache[key] = (data, affine, labels)
        log.debug("Loaded atlas %s: %d labels", key, len(labels))
        return data, affine, labels

    except Exception as e:
        log.warning("Could not load atlas %s: %s", key, e)
        return None, None, None


# ---------------------------------------------------------------------------
# Single-atlas coordinate lookup
# ---------------------------------------------------------------------------

def _lookup_one(
    coords_mm: np.ndarray,
    key: str,
) -> list[str]:
    """
    Look up atlas labels for Nx3 MNI RAS mm coordinates.
    Returns list of strings, one per coordinate.
    """
    data, affine, labels = _load_atlas(key)
    if data is None:
        return ["atlas_unavailable"] * len(coords_mm)

    inv    = np.linalg.inv(affine)
    shape  = data.shape[:3]
    result = []

    for mm in coords_mm:
        h   = np.array([mm[0], mm[1], mm[2], 1.0])
        vox = np.round((inv @ h)[:3]).astype(int)

        if not all(0 <= vox[d] < shape[d] for d in range(3)):
            result.append("out_of_bounds")
            continue

        pid = int(data[vox[0], vox[1], vox[2]])

        if pid == 0 or pid >= len(labels):
            result.append("no_region")
            continue

        lbl = labels[pid]
        result.append(lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl))

    return result


# ---------------------------------------------------------------------------
# Multi-atlas coordinate lookup (main public API)
# ---------------------------------------------------------------------------

def label_coords_all(
    coords_mm: np.ndarray,
    atlas_keys: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Label Nx3 MNI RAS mm coordinates against all (or specified) atlases.

    Parameters
    ----------
    coords_mm  : Nx3 array of MNI RAS mm coordinates.
    atlas_keys : list of atlas keys to use (default: all ATLAS_KEYS).

    Returns
    -------
    dict mapping atlas_key → list of label strings (one per coordinate).
    """
    if atlas_keys is None:
        atlas_keys = ATLAS_KEYS

    result = {}
    for key in atlas_keys:
        try:
            result[key] = _lookup_one(coords_mm, key)
            log.debug("Atlas %s: labelled %d points", key, len(coords_mm))
        except Exception as e:
            log.warning("Atlas %s labelling failed: %s", key, e)
            result[key] = ["atlas_error"] * len(coords_mm)

    return result


def label_coords(
    coords_mm: np.ndarray,
) -> list[str]:
    """
    Backwards-compatible single-atlas lookup (Harvard-Oxford only).
    Used by stage_05_visualize.py.
    """
    return _lookup_one(coords_mm, "harvard_oxford")


# ---------------------------------------------------------------------------
# Glass brain plot (single atlas)
# ---------------------------------------------------------------------------

def _plot_one_atlas(
    key: str,
    coords_mm: np.ndarray,
    hemi_labels: list[str],
    out_path: str,
    color_left:  str = "#4488FF",
    color_right: str = "#FF4444",
    marker_size: float = 5,
    title: str | None = None,
) -> None:
    """Plot stimulation sites on a glass brain with one atlas overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    from nilearn import plotting

    data, affine, labels = _load_atlas(key)

    regions = _lookup_one(coords_mm, key)

    targeted = sorted(set(r for r in regions
                          if r not in ("no_region", "out_of_bounds",
                                       "atlas_unavailable", "atlas_error")))
    n_regions  = len(targeted)
    cmap_r     = cm.get_cmap("tab20", max(n_regions, 1))
    reg_colors = {r: cmap_r(i) for i, r in enumerate(targeted)}

    display_title = title or ATLAS_NAMES.get(key, key)

    fig  = plt.figure(figsize=(14, 4), facecolor="white")
    disp = plotting.plot_glass_brain(
        None, display_mode="lyrz", figure=fig,
        title=display_title, annotate=True, black_bg=False,
    )

    # Atlas contours
    if data is not None and n_regions > 0:
        # Build label→index map
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
    elif n_regions == 0:
        log.warning("Atlas %s: no regions found for any coordinate — "
                    "plotting without contours.", key)

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

    # Legend
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

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("Atlas plot (%s): %s", key, out_path)


# ---------------------------------------------------------------------------
# Plot all atlases (main public API)
# ---------------------------------------------------------------------------

def plot_all_atlases(
    coords_mm:   np.ndarray,
    hemi_labels: list[str],
    out_dir:     str,
    color_left:  str = "#4488FF",
    color_right: str = "#FF4444",
    marker_size: float = 5,
    atlas_keys:  list[str] | None = None,
) -> dict[str, str]:
    """
    Plot glass brain for each atlas.

    Parameters
    ----------
    coords_mm   : Nx3 MNI RAS mm coordinates.
    hemi_labels : list of 'L'/'R' per coordinate.
    out_dir     : directory to write PNG files into.
    color_left  : hex colour for left hemisphere markers.
    color_right : hex colour for right hemisphere markers.
    marker_size : marker size passed to nilearn.
    atlas_keys  : atlases to plot (default: all ATLAS_KEYS).

    Returns
    -------
    dict mapping atlas_key → output PNG path.
    """
    if atlas_keys is None:
        atlas_keys = ATLAS_KEYS

    out_paths = {}
    for key in atlas_keys:
        stem     = ATLAS_PNG_STEMS.get(key, f"{key}_regions")
        out_path = os.path.join(out_dir, f"{stem}.png")
        try:
            _plot_one_atlas(
                key, coords_mm, hemi_labels, out_path,
                color_left=color_left,
                color_right=color_right,
                marker_size=marker_size,
            )
            out_paths[key] = out_path
        except Exception as e:
            log.warning("Atlas plot failed for %s: %s", key, e)

    return out_paths


# ---------------------------------------------------------------------------
# Backwards-compatible single-atlas plot (used by stage_05)
# ---------------------------------------------------------------------------

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
    Backwards-compatible wrapper — plots Harvard-Oxford only.
    Called by stage_05_visualize.py.
    """
    _plot_one_atlas(
        "harvard_oxford", coords_mm, hemi_labels, out_path,
        color_left=color_left,
        color_right=color_right,
        marker_size=marker_size,
        title=title,
    )

    regions = _lookup_one(coords_mm, "harvard_oxford")
    return pd.DataFrame({
        "mni_mm_x":   coords_mm[:, 0],
        "mni_mm_y":   coords_mm[:, 1],
        "mni_mm_z":   coords_mm[:, 2],
        "hemisphere": hemi_labels,
        "ho_region":  regions,
    })