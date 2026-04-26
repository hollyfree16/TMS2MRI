# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

TMS2MRI converts NextStim TMS (Transcranial Magnetic Stimulation) stimulation coordinates to MNI152 standard space and visualizes them on a glass brain. It processes `.nbe` export files alongside subject T1 MRI scans.

## Running the Pipeline

```bash
# Install Python dependencies
pip install -r requirements.txt

# Full pipeline (subject space + MNI registration)
python run_tms2mni.py \
    --nbe data/TEP004.nbe \
    --t1 data/TEP004_T1.nii.gz \
    --subject-id TEP004 \
    --output-dir ./outputs \
    --mni-template /path/to/MNI152_T1_1mm_brain.nii.gz \
    --mni-template-full /path/to/MNI152_T1_1mm.nii.gz \
    --id 1.26.23 --id 1.48.18 \
    --site M1_L \
    --shared-csv ./all_subjects.csv

# Subject space only (skip MNI registration)
python run_tms2mni.py --nbe data/TEP004.nbe --t1 data/TEP004_T1.nii.gz \
    --subject-id TEP004 --output-dir ./outputs

# Group visualization across subjects
python group_visualize.py --csv all_subjects.csv --output-dir ./group_viz --output-prefix my_study

# Coordinate lookup utility
python -m utils.lookup --csv outputs/TEP004/coordinates/targets_native.csv \
    --x 833.5 --y 567.2 --z 522.3
```

Key flags: `--force` (re-run all stages), `--no-flip` (disable X-axis flip), `--coord-col ef|coil`, `--color-by-hemi`, `--skip-snap`, `--log-level DEBUG|INFO|WARNING`.

## External Tool Dependencies (must be in PATH)

- **FreeSurfer ≥ 7.0**: `mri_synthstrip` (brain extraction)
- **FSL ≥ 6.0**: `bet` (fallback brain extraction)
- **ANTs ≥ 2.4**: `antsRegistrationSyN.sh`, `antsApplyTransforms`, `antsApplyTransformsToPoints`

There is no test suite or CI setup.

## Architecture

### Pipeline Stages (linear, file-based)

```
stages/stage_01_parse_nbe.py     → .nbe → landmarks.csv + targets.csv/xlsx
stages/stage_02_skullstrip.py    → T1 → T1_brain (SynthStrip, BET fallback)
stages/stage_03_register_mni.py  → T1_brain → MNI152 via ANTs SyN
stages/stage_04_convert_coords.py → NBE coords → native voxels/mm → MNI mm → fsaverage surface
stages/stage_05_visualize.py     → glass brain plots + Harvard-Oxford atlas overlay
stages/stage_06_snap_surface.py  → stub (Blender integration, future)
```

**Stage skipping**: `staging.py` skips a stage if its output files already exist. This enables crash recovery — re-running picks up from the last incomplete stage. `--force` overrides this.

### Key Abstractions

**`PathManifest`** (`utils/io.py`) — central dataclass defining all input/output paths for a subject. Constructed once from CLI args and passed to every stage. Never construct paths ad hoc outside of this class.

**`staging.py`** — generic stage runner. Validates output file existence post-execution. The only place where stage skip/force logic lives.

**`utils/affine.py`** — pure numpy coordinate transforms: RAS↔LPS conversions, voxel↔mm via NIfTI affine, orientation-aware NextStim axis mapping, optional X-flip. No side effects.

**`utils/atlas.py`** — Harvard-Oxford atlas lookups and glass brain plotting with stimulation markers and atlas contours.

### Coordinate Systems

This is the most complex part of the codebase. Four coordinate systems are involved:
1. **NextStim NBE**: fixed anatomical frame, mm from image corner (voxel 0,0,0)
2. **MRI Native**: NIfTI physical space (RAS, PIR, PIL, IPR, or LAS depending on scanner)
3. **ANTs**: uses LPS (opposite handedness from NIfTI RAS)
4. **MNI152**: standard RAS space; negative X = left hemisphere

### NextStim NBE Coordinate Convention

NextStim stores coordinates in a fixed anatomical frame regardless of how the MRI
was stored on disk:

```
NBE X  →  the image's R/L anatomical axis
NBE Y  →  the image's S/I anatomical axis
NBE Z  →  the image's A/P anatomical axis
```

Values are in mm measured from the image corner (voxel 0,0,0), scaled by voxel size.
The conversion to voxel indices must be orientation-aware — see `nextstim_to_mri_voxels()`
in `utils/affine.py`.

### Non-RAS MRI Orientations

Some MRI scanners store T1 images in non-RAS orientations. This occurs when the image
is loaded directly onto the NextStim neuronavigation device without reorientation.
The pipeline handles all observed orientations correctly via the orientation-aware
NBE→voxel mapping in `nextstim_to_mri_voxels()`.

**Validated orientations (April 2025):**

| Orientation | Example subjects | Dataset |
|-------------|-----------------|---------|
| RAS | sub-ou1neuroc001–013, majority | ou1 |
| PIR | sub-ou1neuroc012, sub-ou1psychc010, sub-ou1psychc018 | ou1 |
| PIL | sub-ou1psychc012, sub-ou1psychc014 | ou1 |
| IPR | sub-ou1psychc013 | ou1 |
| LAS | TBDPCI_12 | ou3 |

**Validation method**: computed landmark voxel coordinates were entered into FreeView
and confirmed to land on the correct anatomy (nasion, left ear, right ear) for
representative subjects of each orientation.

**Hemisphere convention**: L (MRI left) = NextStim right ear side = higher X in RAS
space. Use `--no-flip` if hemisphere appears swapped.

**Auto-flip retry** (stage 04): if hemisphere check fails or all EF coordinates land
out-of-bounds, the stage automatically retries with the opposite X-flip and accepts
whichever result has more in-bounds coordinates.

### Logging

Logger hierarchy: `tms2mni.stages.<name>` and `tms2mni.utils.<name>`. Per-subject
timestamped logs written to `outputs/<subject_id>/logs/`. File handler always captures
DEBUG; console respects `--log-level`.

### Output Structure

```
outputs/<subject_id>/
├── xnbe/           # Stage 01: parsed NBE coordinates
├── registration/   # Stages 02-03: skull-stripped T1, ANTs transforms
├── coordinates/    # Stage 04: native + MNI + fsaverage CSVs
├── visualization/  # Stage 05: .html, .png, ho_regions.png
└── logs/
```