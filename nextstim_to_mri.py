"""
NextStim -> MRI + MNI Space Coordinate Conversion
==================================================
Converts coil/EF coordinates from a NextStim .nbe export into:
  - Subject native MRI voxel indices
  - MNI152 voxel indices
  - MNI152 mm coordinates (RAS)

Transform pipeline:
  1. NBE coords -> subject T1 voxels (Y/Z swap + optional X flip)
  2. T1 brain extraction via ANTsPy
  3. ANTs SyN registration: subject T1 brain -> MNI152 brain template
  4. Warp applied to full-head T1 (for visualization)
  5. Native mm coords -> MNI mm coords via ANTs transform
  6. MNI mm coords -> MNI voxel indices via MNI template affine

Registration files are saved and reused if they already exist.

Requirements:
    pip install nibabel numpy pandas antspyx

Usage:
    # Subject space only
    python nextstim_to_mri.py \
        --t1 T1.nii.gz \
        --targets targets.csv \
        --landmarks landmarks.csv \
        --subject-id TEP004 \
        --output-dir ./outputs

    # Subject space + MNI
    python nextstim_to_mri.py \
        --t1 T1.nii.gz \
        --targets targets.csv \
        --landmarks landmarks.csv \
        --subject-id TEP004 \
        --output-dir ./outputs \
        --mni-template MNI152_T1_1mm_brain.nii.gz \
        --mni-template-full MNI152_T1_1mm.nii.gz \
        --no-flip
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Convert NextStim NBE coordinates to subject MRI and MNI space."
)

# Required
parser.add_argument("--t1",           required=True,  help="Path to subject T1 NIfTI (e.g. T1.nii.gz)")
parser.add_argument("--targets",      required=True,  help="Path to NextStim targets CSV")
parser.add_argument("--landmarks",    required=True,  help="Path to NextStim landmarks CSV")
parser.add_argument("--subject-id",   required=True,  help="Subject identifier (e.g. TEP004)")
parser.add_argument("--output-dir",   required=True,  help="Root output directory")

# Optional MNI
parser.add_argument("--mni-template",      default=None,
                    help="Brain-extracted MNI152 template for registration (e.g. MNI152_T1_1mm_brain.nii.gz)")
parser.add_argument("--mni-template-full", default=None,
                    help="Full-head MNI152 template for warped T1 output (e.g. MNI152_T1_1mm.nii.gz)")

# Flip control
parser.add_argument("--no-flip", action="store_true",
                    help="Disable X-axis flip (default: ON). Use if points appear mirrored L/R.")

args = parser.parse_args()

DO_MNI = args.mni_template is not None
if DO_MNI and args.mni_template_full is None:
    print("WARNING: --mni-template provided but --mni-template-full not provided.")
    print("         Warped full-head T1 will not be generated.")

# =============================================================================
# COLUMN SETTINGS — edit if your CSV uses different column names
# =============================================================================

TARGET_X_COL = "ef_max_loc_x"
TARGET_Y_COL = "ef_max_loc_y"
TARGET_Z_COL = "ef_max_loc_z"
COIL_X_COL   = "coil_loc_x"
COIL_Y_COL   = "coil_loc_y"
COIL_Z_COL   = "coil_loc_z"

# =============================================================================
# OUTPUT DIRECTORY STRUCTURE
#
# <output-dir>/
# └── <subject-id>/
#     ├── registration/
#     │   ├── T1_brain.nii.gz
#     │   ├── T1_in_MNI.nii.gz
#     │   ├── sub_to_MNI_0GenericAffine.mat
#     │   └── sub_to_MNI_1Warp.nii.gz
#     └── coordinates/
#         ├── targets_native.csv
#         └── targets_mni.csv
# =============================================================================

sub_dir   = os.path.join(args.output_dir, args.subject_id)
reg_dir   = os.path.join(sub_dir, "registration")
coord_dir = os.path.join(sub_dir, "coordinates")

for d in [sub_dir, reg_dir, coord_dir]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print(f"NextStim -> MRI Coordinate Conversion  |  {args.subject_id}")
print("=" * 60)
print(f"\n[Output] Directory: {sub_dir}")

# =============================================================================
# STEP 1: Load data
# =============================================================================

img    = nib.load(args.t1)
affine = img.affine
shape  = img.shape

vox_x = abs(affine[0, 0])
vox_y = abs(affine[1, 1])
vox_z = abs(affine[2, 2])
dim_x = shape[0]

t1_x_is_left_to_right = affine[0, 0] > 0

print(f"\n[T1] Loaded: {args.t1}")
print(f"     Dimensions  : {shape[0]} x {shape[1]} x {shape[2]} voxels")
print(f"     Voxel size  : {vox_x:.4f} x {vox_y:.4f} x {vox_z:.4f} mm")
print(f"     X orientation: {'Left-to-Right (RAS)' if t1_x_is_left_to_right else 'Right-to-Left (LAS)'}")

lm_df  = pd.read_csv(args.landmarks)
tgt_df = pd.read_csv(args.targets, na_values=['-', ' -', '- '])
print(f"\n[Data] {len(lm_df)} landmarks, {len(tgt_df)} stimulation events")

# =============================================================================
# STEP 2: X flip setting
# =============================================================================

FLIP_X = not args.no_flip
print(f"\n[X Flip] {'ON (default)' if FLIP_X else 'OFF (--no-flip)'}")
if not t1_x_is_left_to_right:
    print(f"  Note: T1 is LAS — if results look mirrored, try toggling --no-flip.")

# =============================================================================
# STEP 3: NBE -> subject native voxel transform
# =============================================================================

def nextstim_to_mri_voxels(coords_nbe, dim_x, vox_x, vox_y, vox_z, flip_x):
    """Convert NextStim NBE coordinates to subject T1 voxel indices."""
    x = coords_nbe[:, 0]
    y = coords_nbe[:, 1]
    z = coords_nbe[:, 2]
    x_mri = (dim_x - 1) - (x / vox_x) if flip_x else x / vox_x
    y_mri = z / vox_z   # NBE Z -> MRI Y
    z_mri = y / vox_y   # NBE Y -> MRI Z
    return np.column_stack([x_mri, y_mri, z_mri])

def voxels_to_mm(vox, affine):
    """Convert Nx3 voxel indices to mm world coords using image affine."""
    h = np.hstack([vox, np.ones((len(vox), 1))])
    return (affine @ h.T).T[:, :3]

def convert_coords(df, x_col, y_col, z_col):
    coords = df[[x_col, y_col, z_col]].values.astype(float)
    mask   = ~np.isnan(coords).any(axis=1)
    result = np.full(coords.shape, np.nan)
    if mask.any():
        result[mask] = nextstim_to_mri_voxels(
            coords[mask], dim_x, vox_x, vox_y, vox_z, FLIP_X
        )
    return result

print(f"\n[Transform] Formula:")
print(f"  x_mri = {'('+str(dim_x)+' - 1) - ' if FLIP_X else ''}x_nbe / {vox_x:.4f}")
print(f"  y_mri = z_nbe / {vox_z:.4f}  [NBE Z -> MRI Y]")
print(f"  z_mri = y_nbe / {vox_y:.4f}  [NBE Y -> MRI Z]")

# Convert landmarks
lm_coords = lm_df[["x", "y", "z"]].values.astype(float)
lm_vox    = nextstim_to_mri_voxels(lm_coords, dim_x, vox_x, vox_y, vox_z, FLIP_X)
lm_mm     = voxels_to_mm(lm_vox, affine)

print(f"\n[Landmarks] Converted to native voxels:")
for i, row in lm_df.iterrows():
    v = lm_vox[i]
    in_b = all(0 <= int(round(v[d])) < shape[d] for d in range(3))
    print(f"  {row['landmark_type']}: [{v[0]:.0f}, {v[1]:.0f}, {v[2]:.0f}]  {'OK' if in_b else 'OUT OF BOUNDS ⚠️'}")

# Hemisphere check
lm_df['_type_lower'] = lm_df['landmark_type'].str.lower()
left_rows  = lm_df[lm_df['_type_lower'].str.contains('left')]
right_rows = lm_df[lm_df['_type_lower'].str.contains('right')]

correct = None
if not left_rows.empty and not right_rows.empty:
    r_x = lm_vox[right_rows.index[0]][0]
    l_x = lm_vox[left_rows.index[0]][0]
    correct = (r_x > l_x) if t1_x_is_left_to_right else (r_x < l_x)
    print(f"\n[Hemisphere Check]")
    print(f"  NextStim Right ear (MRI L) X: {r_x:.1f}")
    print(f"  NextStim Left ear  (MRI R) X: {l_x:.1f}")
    print(f"  {'✅ Correct' if correct else '⚠️  WRONG — toggle --no-flip'}")

x_midpoint = dim_x / 2.0

# Convert targets
print(f"\n[Conversion] Converting EF and coil coordinates...")
ef_vox   = convert_coords(tgt_df, TARGET_X_COL, TARGET_Y_COL, TARGET_Z_COL)
coil_vox = convert_coords(tgt_df, COIL_X_COL,   COIL_Y_COL,   COIL_Z_COL)

# Native mm coordinates
ef_mm_native   = np.full(ef_vox.shape,   np.nan)
coil_mm_native = np.full(coil_vox.shape, np.nan)
ef_mask   = ~np.isnan(ef_vox).any(axis=1)
coil_mask = ~np.isnan(coil_vox).any(axis=1)
if ef_mask.any():
    ef_mm_native[ef_mask]     = voxels_to_mm(ef_vox[ef_mask],     affine)
if coil_mask.any():
    coil_mm_native[coil_mask] = voxels_to_mm(coil_vox[coil_mask], affine)

# Bounds check
def count_in_bounds(vox, shape):
    mask = ~np.isnan(vox).any(axis=1)
    v    = np.round(vox[mask]).astype(int)
    n_ok = sum(all(0 <= v[i, d] < shape[d] for d in range(3)) for i in range(len(v)))
    return n_ok, mask.sum()

ef_ok, ef_total     = count_in_bounds(ef_vox,   shape)
coil_ok, coil_total = count_in_bounds(coil_vox, shape)
print(f"\n[Bounds Check]")
print(f"  EF   in bounds: {ef_ok}/{ef_total}")
print(f"  Coil in bounds: {coil_ok}/{coil_total}")
if ef_ok < ef_total or coil_ok < coil_total:
    print(f"  Warning: some coordinates outside image volume — check manually.")

# Hemisphere labels
def label_hemisphere(x_vox, midpoint, ltr):
    return [None if np.isnan(x) else ("L" if (x > midpoint) == ltr else "R")
            for x in x_vox]

ef_hemi   = label_hemisphere(ef_vox[:, 0],   x_midpoint, t1_x_is_left_to_right)
coil_hemi = label_hemisphere(coil_vox[:, 0], x_midpoint, t1_x_is_left_to_right)

# =============================================================================
# STEP 4: MNI Registration (skipped if --mni-template not provided)
# =============================================================================

ef_mni_mm    = np.full(ef_vox.shape,   np.nan)
coil_mni_mm  = np.full(coil_vox.shape, np.nan)
ef_mni_vox   = np.full(ef_vox.shape,   np.nan)
coil_mni_vox = np.full(coil_vox.shape, np.nan)

if DO_MNI:
    print(f"\n{'='*60}")
    print(f"[MNI] Registration module")
    print(f"{'='*60}")

    brain_path   = os.path.join(reg_dir, "T1_brain.nii.gz")
    affine_path  = os.path.join(reg_dir, "sub_to_MNI_0GenericAffine.mat")
    warp_path    = os.path.join(reg_dir, "sub_to_MNI_1Warp.nii.gz")
    t1_mni_path  = os.path.join(reg_dir, "T1_in_MNI.nii.gz")
    reg_prefix   = os.path.join(reg_dir, "sub_to_MNI_")

    import subprocess

    # -------------------------------------------------------------------------
    # STEP 4a: Brain extraction using SynthStrip (nipreps-synthstrip)
    # -------------------------------------------------------------------------
    if os.path.exists(brain_path):
        print(f"\n[MNI] Brain extraction — reusing existing: {brain_path}")
    else:
        print(f"\n[MNI] Brain extraction via mri_synthstrip...")
        cmd = ["mri_synthstrip", "-i", args.t1, "-o", brain_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"       ERROR during brain extraction:")
            print(result.stderr[-2000:])
            sys.exit(1)
        print(f"       Saved: {brain_path}")

    # -------------------------------------------------------------------------
    # STEP 4b: Register subject brain -> MNI brain using antsRegistrationSyN.sh
    # -------------------------------------------------------------------------
    if os.path.exists(affine_path) and os.path.exists(warp_path):
        print(f"\n[MNI] Registration — reusing existing transforms in {reg_dir}")
        transforms = [warp_path, affine_path]
    else:
        print(f"\n[MNI] Running antsRegistrationSyN.sh (this may take several minutes)...")
        cmd = [
            "antsRegistrationSyN.sh",
            "-d", "3",
            "-f", args.mni_template,       # fixed:  MNI152 brain
            "-m", brain_path,              # moving: subject brain
            "-o", reg_prefix,              # output prefix
            "-t", "s",                     # SyN: rigid + affine + deformable
            "-n", "4"                      # threads
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"       ERROR during registration:")
            print(result.stderr[-2000:])
            sys.exit(1)

        if not os.path.exists(affine_path) or not os.path.exists(warp_path):
            print(f"       ERROR: expected transform files not found:")
            print(f"         {affine_path}")
            print(f"         {warp_path}")
            sys.exit(1)

        transforms = [warp_path, affine_path]
        print(f"       Transforms saved to: {reg_dir}")

    # -------------------------------------------------------------------------
    # STEP 4c: Apply warp to full-head T1 for visualization
    # antsApplyTransforms -d 3 -i T1.nii.gz -r MNI_full.nii.gz
    #                     -t Warp.nii.gz -t Affine.mat -o T1_in_MNI.nii.gz
    # -------------------------------------------------------------------------
    if args.mni_template_full:
        if os.path.exists(t1_mni_path):
            print(f"\n[MNI] Warped T1 — reusing existing: {t1_mni_path}")
        else:
            print(f"\n[MNI] Applying warp to full-head T1...")
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", args.t1,
                "-r", args.mni_template_full,
                "-t", warp_path,
                "-t", affine_path,
                "-o", t1_mni_path,
                "--interpolation", "LanczosWindowedSinc"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"       ERROR applying transforms to T1:")
                print(result.stderr[-2000:])
                sys.exit(1)
            print(f"       Saved: {t1_mni_path}")

    # -------------------------------------------------------------------------
    # STEP 4d: Apply warp to coordinates via antsApplyTransformsToPoints
    #
    # antsApplyTransformsToPoints expects a CSV with columns x,y,z in LPS mm.
    # Our native coords are in RAS mm (from nibabel affine), so we flip
    # X and Y before passing to ANTs, then flip back afterwards.
    #
    # Pipeline:
    #   native voxels -> RAS mm (nibabel affine)
    #   -> LPS mm (flip X,Y)
    #   -> antsApplyTransformsToPoints (using INVERSE transforms for points)
    #   -> LPS mm -> RAS mm (flip X,Y back)
    #   -> MNI voxels (MNI inverse affine)
    #
    # NOTE: For point transforms, ANTs requires the INVERSE transforms:
    #   -t [Warp_inv.nii.gz] -t [Affine.mat, 1]
    # antsRegistrationSyN.sh saves both forward and inverse warps.
    # -------------------------------------------------------------------------
    print(f"\n[MNI] Transforming coordinates to MNI space...")

    inv_warp_path = reg_prefix + "1InverseWarp.nii.gz"
    if not os.path.exists(inv_warp_path):
        print(f"       ERROR: inverse warp not found: {inv_warp_path}")
        print(f"       antsRegistrationSyN.sh should have produced this file.")
        sys.exit(1)

    mni_img        = nib.load(args.mni_template)
    mni_affine     = mni_img.affine
    mni_inv_affine = np.linalg.inv(mni_affine)

    def ras_to_lps(coords):
        """RAS -> LPS: negate X and Y."""
        c = coords.copy()
        c[:, 0] *= -1
        c[:, 1] *= -1
        return c

    def lps_to_ras(coords):
        """LPS -> RAS: same operation as ras_to_lps."""
        return ras_to_lps(coords)

    def mm_to_voxels(coords_mm, inv_affine):
        h = np.hstack([coords_mm, np.ones((len(coords_mm), 1))])
        return (inv_affine @ h.T).T[:, :3]

    import tempfile

    def apply_mni_transform_to_points(mm_native_ras, mask):
        """
        Transform a set of RAS mm coords to MNI RAS mm using ANTs point transform.
        Returns full array (NaN for rows where mask is False).
        """
        result = np.full(mm_native_ras.shape, np.nan)
        if not mask.any():
            return result

        valid_ras = mm_native_ras[mask]
        valid_lps = ras_to_lps(valid_ras)

        # Write LPS points to temp CSV — ANTs format: x,y,z,t (t=0)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
            pts_in_path = f.name
            f.write("x,y,z,t\n")
            for row in valid_lps:
                f.write(f"{row[0]},{row[1]},{row[2]},0\n")

        pts_out_path = pts_in_path.replace(".csv", "_out.csv")

        cmd = [
            "antsApplyTransformsToPoints",
            "-d", "3",
            "-i", pts_in_path,
            "-o", pts_out_path,
            "-t", f"[{affine_path},1]",      # inverse affine
            "-t", inv_warp_path,              # inverse warp
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"       ERROR applying transforms to points:")
            print(res.stderr[-2000:])
            sys.exit(1)

        pts_out = pd.read_csv(pts_out_path)
        out_lps = pts_out[["x", "y", "z"]].values
        out_ras = lps_to_ras(out_lps)

        result[mask] = out_ras

        os.unlink(pts_in_path)
        os.unlink(pts_out_path)
        return result

    ef_mni_mm   = apply_mni_transform_to_points(ef_mm_native,   ef_mask)
    coil_mni_mm = apply_mni_transform_to_points(coil_mm_native, coil_mask)
    lm_mni_mm   = apply_mni_transform_to_points(lm_mm, np.ones(len(lm_mm), dtype=bool))

    # MNI mm (RAS) -> MNI voxel indices
    ef_mni_vox[ef_mask]     = mm_to_voxels(ef_mni_mm[ef_mask],     mni_inv_affine)
    coil_mni_vox[coil_mask] = mm_to_voxels(coil_mni_mm[coil_mask], mni_inv_affine)
    lm_mni_vox              = mm_to_voxels(lm_mni_mm,              mni_inv_affine)

    print(f"\n[MNI] Landmark positions in MNI space:")
    for i, row in lm_df.iterrows():
        v = lm_mni_vox[i]
        m = lm_mni_mm[i]
        print(f"  {row['landmark_type']}")
        print(f"    voxel: [{v[0]:.0f}, {v[1]:.0f}, {v[2]:.0f}]   mm (RAS): [{m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f}]")

    print(f"\n[MNI] Done.")

# =============================================================================
# STEP 5: Assemble and save outputs
# =============================================================================

print(f"\n[Output] Saving coordinate files...")

# --- Native CSV ---
native_df = tgt_df.copy()

native_df["ef_native_vox_x"]  = np.where(np.isnan(ef_vox[:, 0]),   np.nan, np.round(ef_vox[:, 0]))
native_df["ef_native_vox_y"]  = np.where(np.isnan(ef_vox[:, 1]),   np.nan, np.round(ef_vox[:, 1]))
native_df["ef_native_vox_z"]  = np.where(np.isnan(ef_vox[:, 2]),   np.nan, np.round(ef_vox[:, 2]))
native_df["ef_native_mm_x"]   = ef_mm_native[:, 0]
native_df["ef_native_mm_y"]   = ef_mm_native[:, 1]
native_df["ef_native_mm_z"]   = ef_mm_native[:, 2]
native_df["ef_hemisphere"]    = ef_hemi

native_df["coil_native_vox_x"] = np.where(np.isnan(coil_vox[:, 0]), np.nan, np.round(coil_vox[:, 0]))
native_df["coil_native_vox_y"] = np.where(np.isnan(coil_vox[:, 1]), np.nan, np.round(coil_vox[:, 1]))
native_df["coil_native_vox_z"] = np.where(np.isnan(coil_vox[:, 2]), np.nan, np.round(coil_vox[:, 2]))
native_df["coil_native_mm_x"]  = coil_mm_native[:, 0]
native_df["coil_native_mm_y"]  = coil_mm_native[:, 1]
native_df["coil_native_mm_z"]  = coil_mm_native[:, 2]
native_df["coil_hemisphere"]   = coil_hemi

native_path = os.path.join(coord_dir, "targets_native.csv")
native_df.to_csv(native_path, index=False)
print(f"  Native : {native_path}")

# --- MNI CSV ---
if DO_MNI:
    mni_df = tgt_df.copy()

    mni_df["ef_mni_mm_x"]   = ef_mni_mm[:, 0]
    mni_df["ef_mni_mm_y"]   = ef_mni_mm[:, 1]
    mni_df["ef_mni_mm_z"]   = ef_mni_mm[:, 2]
    mni_df["ef_mni_vox_x"]  = np.where(np.isnan(ef_mni_vox[:, 0]),   np.nan, np.round(ef_mni_vox[:, 0]))
    mni_df["ef_mni_vox_y"]  = np.where(np.isnan(ef_mni_vox[:, 1]),   np.nan, np.round(ef_mni_vox[:, 1]))
    mni_df["ef_mni_vox_z"]  = np.where(np.isnan(ef_mni_vox[:, 2]),   np.nan, np.round(ef_mni_vox[:, 2]))
    mni_df["ef_hemisphere"]  = ef_hemi

    mni_df["coil_mni_mm_x"]  = coil_mni_mm[:, 0]
    mni_df["coil_mni_mm_y"]  = coil_mni_mm[:, 1]
    mni_df["coil_mni_mm_z"]  = coil_mni_mm[:, 2]
    mni_df["coil_mni_vox_x"] = np.where(np.isnan(coil_mni_vox[:, 0]), np.nan, np.round(coil_mni_vox[:, 0]))
    mni_df["coil_mni_vox_y"] = np.where(np.isnan(coil_mni_vox[:, 1]), np.nan, np.round(coil_mni_vox[:, 1]))
    mni_df["coil_mni_vox_z"] = np.where(np.isnan(coil_mni_vox[:, 2]), np.nan, np.round(coil_mni_vox[:, 2]))
    mni_df["coil_hemisphere"] = coil_hemi

    mni_path = os.path.join(coord_dir, "targets_mni.csv")
    mni_df.to_csv(mni_path, index=False)
    print(f"  MNI    : {mni_path}")

# =============================================================================
# STEP 6: Summary
# =============================================================================

print(f"""
=============================================================
SUMMARY  |  {args.subject_id}
=============================================================
Output directory : {sub_dir}

Native coordinates : {native_path}
  Columns: ef_native_vox_x/y/z, ef_native_mm_x/y/z, ef_hemisphere
           coil_native_vox_x/y/z, coil_native_mm_x/y/z, coil_hemisphere""")

if DO_MNI:
    print(f"""
MNI coordinates    : {mni_path}
  Columns: ef_mni_mm_x/y/z, ef_mni_vox_x/y/z, ef_hemisphere
           coil_mni_mm_x/y/z, coil_mni_vox_x/y/z, coil_hemisphere

Registration files : {reg_dir}
  T1_brain.nii.gz               — skull-stripped T1
  sub_to_MNI_0GenericAffine.mat — affine transform
  sub_to_MNI_1Warp.nii.gz       — nonlinear warp""")
    if args.mni_template_full:
        print(f"  T1_in_MNI.nii.gz              — full-head T1 warped to MNI (for visualization)")

print(f"""
X flip applied   : {FLIP_X}
Hemisphere check : {'✅ passed' if correct else ('⚠️  FAILED — toggle --no-flip' if correct is not None else 'skipped')}

Hemisphere convention:
  L = MRI Left  = NextStim Right ear side = higher X (RAS)
  R = MRI Right = NextStim Left ear side  = lower  X (RAS)

Landmark native voxels (for FSLeyes verification):""")

for i, row in lm_df.iterrows():
    v = lm_vox[i]
    print(f"  {row['landmark_type']}: [{v[0]:.0f}, {v[1]:.0f}, {v[2]:.0f}]")

if DO_MNI:
    print(f"\nLandmark MNI mm coords (RAS):")
    for i, row in lm_df.iterrows():
        m = lm_mni_mm[i]
        print(f"  {row['landmark_type']}: [{m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f}]")

print("=============================================================\n")