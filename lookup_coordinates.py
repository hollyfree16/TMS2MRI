"""
Lookup stimulation row by EF max coordinates
=============================================
Pass x y z on the command line, get the matching row(s) back.

Usage:
    python lookup_by_coords.py --csv targets_TEP004.csv --x 833.5 --y 567.2 --z 522.3

    # Looser tolerance if needed
    python lookup_by_coords.py --csv targets_TEP004.csv --x 833.5 --y 567.2 --z 522.3 --tol 1.0
"""

import argparse
import sys
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    description="Look up a stimulation row by EF max XYZ coordinates."
)
parser.add_argument("--csv", required=True, help="Path to targets CSV")
parser.add_argument("--x",   required=True, type=float, help="EF max loc X (mm)")
parser.add_argument("--y",   required=True, type=float, help="EF max loc Y (mm)")
parser.add_argument("--z",   required=True, type=float, help="EF max loc Z (mm)")
parser.add_argument("--tol", type=float, default=0.1,
                    help="Match tolerance in mm (default: 0.1)")
args = parser.parse_args()

df = pd.read_csv(args.csv, na_values=['-', ' -', '- '])

x_col, y_col, z_col = "ef_max_loc_x", "ef_max_loc_y", "ef_max_loc_z"
for col in [x_col, y_col, z_col]:
    if col not in df.columns:
        print(f"ERROR: Column '{col}' not found.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

mask = (
    ((df[x_col] - args.x).abs() <= args.tol) &
    ((df[y_col] - args.y).abs() <= args.tol) &
    ((df[z_col] - args.z).abs() <= args.tol)
)
matches = df[mask]

if matches.empty:
    df["_dist"] = np.sqrt(
        (df[x_col] - args.x) ** 2 +
        (df[y_col] - args.y) ** 2 +
        (df[z_col] - args.z) ** 2
    )
    closest = df.nsmallest(3, "_dist")[["id", x_col, y_col, z_col, "_dist"]]
    print(f"\nNo match for ({args.x}, {args.y}, {args.z}) within ±{args.tol}mm.")
    print(f"\nClosest rows:\n")
    print(closest.rename(columns={"_dist": "distance_mm"}).to_string(index=False))
    print(f"\nTip: rerun with --tol {df['_dist'].min():.2f}")
else:
    print(f"\n{len(matches)} match(es) for ({args.x}, {args.y}, {args.z}):\n")
    for _, row in matches.iterrows():
        print(f"  ID         : {row['id']}")
        print(f"  EF max loc : ({row[x_col]}, {row[y_col]}, {row[z_col]})")
        print(f"  Stim type  : {row.get('stim_type', 'N/A')}")
        print(f"  Intensity  : {row.get('first_intens_pct', 'N/A')}%")
        print(f"  EF max     : {row.get('ef_max_value_vm', 'N/A')} V/m")
        print(f"  Time       : {row.get('time_ms', 'N/A')} ms")