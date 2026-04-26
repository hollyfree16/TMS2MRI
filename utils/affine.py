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
    affine:     np.ndarray,
    shape:      tuple,
    flip_x:     bool = True,
) -> np.ndarray:
    """
    Convert NextStim NBE coordinates to subject T1 voxel indices.

    Handles any NIfTI image orientation (RAS, PIR, PIL, IPR, LAS, etc.)
    by reading the orientation codes from the affine and assigning NBE
    axes accordingly.

    Background
    ----------
    NextStim stores stimulation coordinates in a fixed anatomical frame
    tied to the image corner (voxel 0,0,0), in units of voxel-size mm:

        NBE X  →  the image's R/L anatomical axis
        NBE Y  →  the image's S/I anatomical axis
        NBE Z  →  the image's A/P anatomical axis

    For a RAS image the voxel axes happen to align with R, A, S so the
    original hardcoded mapping worked.  For non-RAS images (PIR, PIL, IPR,
    LAS etc.) the voxel axes are permuted and/or flipped, so the mapping
    must be derived from the affine orientation codes.

    Validated orientations (April 2025)
    -------------------------------------
    RAS  — ou1 dataset, majority of subjects
    PIR  — ou1: sub-ou1neuroc012, sub-ou1psychc010, sub-ou1psychc018
    PIL  — ou1: sub-ou1psychc012, sub-ou1psychc014
    IPR  — ou1: sub-ou1psychc013
    LAS  — ou3: TBDPCI_12

    All orientations verified by entering computed landmark voxel
    coordinates into FreeView and confirming the crosshair lands on
    the correct anatomy (nasion, left ear, right ear).

    Parameters
    ----------
    coords_nbe : Nx3 array of NBE coordinates (mm from image corner).
    affine     : 4x4 NIfTI affine of the T1.
    shape      : (dim0, dim1, dim2) voxel dimensions of the T1.
    flip_x     : If True (default), flip the R/L axis.  Pass --no-flip on
                 the CLI if hemisphere labels appear swapped.

    Returns
    -------
    Nx3 array of voxel indices (float).
    """
    import nibabel.orientations as ornt_mod

    coords_nbe = np.asarray(coords_nbe, dtype=float)
    vox_sizes  = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Orientation of each voxel axis expressed as an anatomical code.
    # e.g. PIR → axis0='P', axis1='I', axis2='R'
    current_ornt = ornt_mod.io_orientation(affine)

    # Which NBE axis corresponds to each anatomical direction:
    #   NBE[0] = X  →  R or L
    #   NBE[1] = Y  →  S or I
    #   NBE[2] = Z  →  A or P
    _nbe_axis = {
        'R': 0, 'L': 0,   # NBE X
        'S': 1, 'I': 1,   # NBE Y
        'A': 2, 'P': 2,   # NBE Z
    }
    _opposite = {'R': 'L', 'L': 'R', 'A': 'P', 'P': 'A', 'S': 'I', 'I': 'S'}
    _ras_code  = ['R', 'A', 'S']   # RAS axis index 0/1/2 → anatomical code

    result = np.full(coords_nbe.shape, np.nan)

    for vox_ax, (ras_ax, ras_sign) in enumerate(current_ornt):
        vox_ax  = int(vox_ax)
        ras_ax  = int(ras_ax)

        # Anatomical code for the positive direction of this voxel axis
        code = _ras_code[ras_ax]
        if ras_sign < 0:
            code = _opposite[code]

        nbe_idx  = _nbe_axis[code]
        nbe_vals = coords_nbe[:, nbe_idx]
        vs       = vox_sizes[vox_ax]
        dim      = shape[vox_ax]

        if code == 'R':
            # Validated: NextStim X counts from the R end → flip needed.
            # --no-flip disables if hemispheres appear swapped.
            if flip_x:
                result[:, vox_ax] = (dim - 1) - nbe_vals / vs
            else:
                result[:, vox_ax] = nbe_vals / vs

        elif code == 'L':
            # LAS images: voxel axis runs L→R, opposite to R→L,
            # so flip logic is inverted relative to 'R'.
            # Validated on TBDPCI_12 (ou3 dataset, April 2025) —
            # nasion, left ear, right ear confirmed in FreeView.
            if flip_x:
                result[:, vox_ax] = nbe_vals / vs
            else:
                result[:, vox_ax] = (dim - 1) - nbe_vals / vs

        elif code in ('P', 'I'):
            # Validated: P and I axes are always flipped
            # (NextStim counts from the far end on these axes).
            result[:, vox_ax] = (dim - 1) - nbe_vals / vs

        elif code in ('A', 'S'):
            # Validated: A and S axes are not flipped.
            result[:, vox_ax] = nbe_vals / vs

    return result


def voxel_sizes_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """
    Extract voxel sizes from a NIfTI affine using column norms.

    Safer than reading the diagonal directly — works correctly for oblique
    and non-RAS affines where diagonal values may be near zero.

    Returns
    -------
    (vox_x, vox_y, vox_z) in mm — sizes for voxel axes 0, 1, 2.
    """
    vox_x, vox_y, vox_z = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(vox_x), float(vox_y), float(vox_z)


# =============================================================================
# Hemisphere labelling
# =============================================================================

def label_hemisphere(
    x_vox:                 np.ndarray,
    x_midpoint:            float,
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
    lm_vox:                np.ndarray,
    lm_labels:             list[str],
    x_midpoint:            float,
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