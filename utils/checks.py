"""
utils/checks.py
===============
Output validation helpers used by staging.py and the stages themselves.

All functions are pure (no side effects) except report_missing() which
logs warnings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =============================================================================
# Output existence
# =============================================================================

def all_exist(paths: list[Path]) -> bool:
    """Return True if every path in the list exists on disk."""
    return all(p.exists() for p in paths)


def missing(paths: list[Path]) -> list[Path]:
    """Return the subset of paths that do not exist."""
    return [p for p in paths if not p.exists()]


# =============================================================================
# Coordinate validation
# =============================================================================

def bounds_check(
    vox:   np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int, int]:
    """
    Count how many rows in an Nx3 voxel array fall within image bounds.

    NaN rows and non-finite rows are excluded from the check (not counted
    as in- or out-of-bounds — they are simply missing data).

    Returns
    -------
    (n_in_bounds, n_total_valid)  where n_total_valid excludes NaN rows.
    """
    mask    = ~np.isnan(vox).any(axis=1)
    valid   = vox[mask]

    finite  = np.all(np.isfinite(valid), axis=1)
    valid_f = np.round(valid[finite]).astype(int)

    n_in = sum(
        all(0 <= valid_f[i, d] < shape[d] for d in range(3))
        for i in range(len(valid_f))
    )
    return n_in, int(mask.sum())


def check_voxel_sizes(
    vox_x: float,
    vox_y: float,
    vox_z: float,
) -> None:
    """
    Raise ValueError if any voxel size is degenerate (< 1e-6 mm).
    Catches corrupt or all-zero affines early.
    """
    for name, v in [("vox_x", vox_x), ("vox_y", vox_y), ("vox_z", vox_z)]:
        if v < 1e-6:
            raise ValueError(
                f"Degenerate voxel size {name}={v:.6f} mm. "
                f"The affine matrix may be corrupt or all-zero."
            )
