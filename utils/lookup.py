"""
utils/lookup.py
===============
Look up a stimulation row by EF max coordinates.

Can be used as a library function or run directly as a script.

Script usage
------------
  python -m utils.lookup \\
      --csv outputs/TEP004/coordinates/targets_native.csv \\
      --x 833.5 --y 567.2 --z 522.3

  # Looser tolerance
  python -m utils.lookup \\
      --csv targets_native.csv \\
      --x 833.5 --y 567.2 --z 522.3 --tol 0.1

Library usage
-------------
  from utils.lookup import lookup_by_coords
  matches, closest = lookup_by_coords(df, x=833.5, y=567.2, z=522.3, tol=0.0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


X_COL = "ef_max_loc_x"
Y_COL = "ef_max_loc_y"
Z_COL = "ef_max_loc_z"


def lookup_by_coords(
    df:  pd.DataFrame,
    x:   float,
    y:   float,
    z:   float,
    tol: float = 0.0,
    x_col: str = X_COL,
    y_col: str = Y_COL,
    z_col: str = Z_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find rows in df whose (x_col, y_col, z_col) are within tol mm of (x, y, z).

    Parameters
    ----------
    df:    DataFrame containing coordinate columns.
    x/y/z: Target coordinates in mm.
    tol:   Match tolerance in mm (default 0.0 = exact match).
    x_col, y_col, z_col: Column names to search (default: ef_max_loc_x/y/z).

    Returns
    -------
    (matches, closest_3)
        matches:    Rows within tolerance (may be empty).
        closest_3:  3 nearest rows by Euclidean distance (always populated),
                    with an added 'distance_mm' column.
    """
    for col in [x_col, y_col, z_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(df.columns)}"
            )

    mask = (
        ((df[x_col] - x).abs() <= tol) &
        ((df[y_col] - y).abs() <= tol) &
        ((df[z_col] - z).abs() <= tol)
    )
    matches = df[mask].copy()

    dist_df = df.copy()
    dist_df["distance_mm"] = np.sqrt(
        (dist_df[x_col] - x) ** 2 +
        (dist_df[y_col] - y) ** 2 +
        (dist_df[z_col] - z) ** 2
    )
    closest_3 = dist_df.nsmallest(3, "distance_mm")

    return matches, closest_3


def _print_match(row: pd.Series, x_col: str, y_col: str, z_col: str) -> None:
    print(f"  ID         : {row.get('id', 'N/A')}")
    print(f"  EF max loc : ({row[x_col]}, {row[y_col]}, {row[z_col]})")
    print(f"  Stim type  : {row.get('stim_type', 'N/A')}")
    print(f"  Intensity  : {row.get('first_intens_pct', 'N/A')}%")
    print(f"  EF max     : {row.get('ef_max_value_vm', 'N/A')} V/m")
    print(f"  Time       : {row.get('time_ms', 'N/A')} ms")


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Look up a stimulation row by EF max XYZ coordinates."
    )
    p.add_argument("--csv", required=True, help="Path to targets CSV")
    p.add_argument("--x",   required=True, type=float, help="EF max loc X (mm)")
    p.add_argument("--y",   required=True, type=float, help="EF max loc Y (mm)")
    p.add_argument("--z",   required=True, type=float, help="EF max loc Z (mm)")
    p.add_argument("--tol", type=float, default=0.0,
                   help="Match tolerance in mm (default: 0.0 — exact match)")
    p.add_argument("--x-col", default=X_COL)
    p.add_argument("--y-col", default=Y_COL)
    p.add_argument("--z-col", default=Z_COL)
    args = p.parse_args()

    df = pd.read_csv(args.csv, na_values=["-", " -", "- "])

    try:
        matches, closest = lookup_by_coords(
            df, args.x, args.y, args.z, args.tol,
            x_col=args.x_col, y_col=args.y_col, z_col=args.z_col,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if matches.empty:
        print(f"\nNo match within {args.tol} mm of ({args.x}, {args.y}, {args.z}).")
        print(f"\nClosest 3 rows:\n")
        print(closest[["id", args.x_col, args.y_col, args.z_col, "distance_mm"]]
              .to_string(index=False))
        min_dist = closest["distance_mm"].iloc[0]
        print(f"\nTip: rerun with --tol {min_dist:.2f}")
    else:
        print(f"\n{len(matches)} match(es) for ({args.x}, {args.y}, {args.z}):\n")
        for _, row in matches.iterrows():
            _print_match(row, args.x_col, args.y_col, args.z_col)
            print()


if __name__ == "__main__":
    _main()
