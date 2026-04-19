"""
utils/affine.py
===============
Pure coordinate-transform utilities.  No file IO, no subprocess calls.

All functions operate on numpy arrays and return numpy arrays.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# RAS / LPS conversions
# (ANTs uses LPS; NIfTI/FreeSurfer use RAS)
# =============================================================================

def ras_to_lps(coords: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 RAS coordinates to LPS by negating X and Y.
    Operation is its own inverse — call again to go back.
    """
    c = coords.copy().astype(float)
    c[:, 0] *= -1
    c[:, 1] *= -1
    return c


def lps_to_ras(coords: np.ndarray) -> np.ndarray:
    """LPS → RAS (identical operation to ras_to_lps)."""
    return ras_to_lps(coords)


# =============================================================================
# Voxel ↔ mm conversions
# =============================================================================

def voxels_to_mm(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 voxel indices to mm world coordinates using a NIfTI affine.

    Parameters
    ----------
    vox:    Nx3 array of voxel indices (float or int).
    affine: 4x4 NIfTI affine matrix.

    Returns
    -------
    Nx3 array of mm coordinates in the image's native world space (RAS for
    standard NIfTI images).
    """
    vox = np.asarray(vox, dtype=float)
    h   = np.hstack([vox, np.ones((len(vox), 1))])
    return (affine @ h.T).T[:, :3]


def mm_to_voxels(mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 mm world coordinates to voxel indices using a NIfTI affine.

    Parameters
    ----------
    mm:     Nx3 array of mm coordinates.
    affine: 4x4 NIfTI affine matrix.

    Returns
    -------
    Nx3 array of voxel indices (float — round yourself if integer indices needed).
    """
    mm  = np.asarray(mm, dtype=float)
    inv = np.linalg.inv(affine)
    h   = np.hstack([mm, np.ones((len(mm), 1))])
    return (inv @ h.T).T[:, :3]


# =============================================================================
# NextStim NBE → subject MRI voxel transform
# =============================================================================

def nextstim_to_mri_voxels(
    coords_nbe: np.ndarray,
    dim_x:      int,
    vox_x:      float,
    vox_y:      float,
    vox_z:      float,
    flip_x:     bool = True,
) -> np.ndarray:
    """
    Convert NextStim NBE coordinates to subject T1 voxel indices.

    NextStim uses a different axis convention from NIfTI:
        NBE Z → MRI Y
        NBE Y → MRI Z
        NBE X → MRI X  (with optional flip for L/R orientation)

    Parameters
    ----------
    coords_nbe: Nx3 array of NBE coordinates in mm.
    dim_x:      Number of voxels along the MRI X axis (used for flip).
    vox_x/y/z:  Voxel sizes in mm (from image affine column norms).
    flip_x:     If True, mirror X axis.  Disable with --no-flip if points
                appear left/right swapped.

    Returns
    -------
    Nx3 array of voxel indices (float).
    """
    coords_nbe = np.asarray(coords_nbe, dtype=float)
    x = coords_nbe[:, 0]
    y = coords_nbe[:, 1]
    z = coords_nbe[:, 2]

    x_mri = (dim_x - 1) - (x / vox_x) if flip_x else x / vox_x
    y_mri = z / vox_z   # NBE Z → MRI Y
    z_mri = y / vox_y   # NBE Y → MRI Z

    return np.column_stack([x_mri, y_mri, z_mri])


def voxel_sizes_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """
    Extract voxel sizes from a NIfTI affine using column norms.

    Safer than reading the diagonal directly — works correctly for oblique
    and LAS affines where diagonal values may be near zero.

    Returns
    -------
    (vox_x, vox_y, vox_z) in mm.
    """
    vox_x, vox_y, vox_z = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(vox_x), float(vox_y), float(vox_z)


# =============================================================================
# Hemisphere labelling
# =============================================================================

def label_hemisphere(
    x_vox:              np.ndarray,
    x_midpoint:         float,
    t1_x_is_left_to_right: bool,
) -> list[str | None]:
    """
    Assign hemisphere labels (L / R) based on voxel X position relative
    to the image midpoint.

    Parameters
    ----------
    x_vox:                  1D array of voxel X coordinates.
    x_midpoint:             dim_x / 2.0
    t1_x_is_left_to_right:  True if affine[0,0] > 0  (RAS orientation).

    Returns
    -------
    List of 'L', 'R', or None (for NaN coordinates).
    """
    result = []
    for x in x_vox:
        if np.isnan(x):
            result.append(None)
        elif t1_x_is_left_to_right:
            result.append("L" if x > x_midpoint else "R")
        else:
            result.append("R" if x > x_midpoint else "L")
    return result


def hemisphere_check(
    lm_vox:             np.ndarray,
    lm_labels:          list[str],
    x_midpoint:         float,
    t1_x_is_left_to_right: bool,
) -> bool | None:
    """
    Verify that 'right' landmarks have higher/lower X than 'left' landmarks
    as expected for the image orientation.

    Returns True (correct), False (flipped — toggle --no-flip), or None
    (cannot determine — landmarks missing).
    """
    left_idx  = next((i for i, l in enumerate(lm_labels) if "left"  in l.lower()), None)
    right_idx = next((i for i, l in enumerate(lm_labels) if "right" in l.lower()), None)

    if left_idx is None or right_idx is None:
        return None

    r_x = lm_vox[right_idx][0]
    l_x = lm_vox[left_idx][0]

    if t1_x_is_left_to_right:
        return bool(r_x > l_x)
    else:
        return bool(r_x < l_x)
