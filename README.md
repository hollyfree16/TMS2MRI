# tms2mni

Convert NextStim TMS stimulation coordinates to MNI152 space and
visualize them on a glass brain.

## Pipeline stages

```
01  parse_nbe       .nbe export → landmarks.csv + targets.csv + targets.xlsx
02  skullstrip      T1 → T1_brain  (SynthStrip preferred; BET fallback)
03  register_mni    T1_brain → MNI152 via ANTs SyN
04  convert_coords  NBE → native voxels/mm → MNI mm (via ANTs point transform)
05  visualize       nilearn glass brain (HTML + PNG) + shared CSV append
```

Stages 02–03 are skipped automatically when no `--mni-template` is provided
(subject-space only run).  Each stage is also skipped if its output files
already exist — re-running after a crash picks up from where it left off.
Pass `--force` to re-run everything regardless.

## Installation

```bash
pip install -r requirements.txt
```

External tools also required (not pip-installable):

| Tool | Version | Used for |
|------|---------|----------|
| [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) | ≥ 7.0 | `mri_synthstrip` (brain extraction) |
| [FSL](https://fsl.fmrib.ox.ac.uk/) | ≥ 6.0 | `bet` (fallback brain extraction) |
| [ANTs](https://github.com/ANTsX/ANTs) | ≥ 2.4 | Registration + coordinate warp |

## Usage

### Full pipeline (subject space + MNI)

```bash
python run_tms2mni.py \
    --nbe  data/TEP004.nbe \
    --t1   data/TEP004_T1.nii.gz \
    --subject-id TEP004 \
    --output-dir ./outputs \
    --mni-template      /path/to/MNI152_T1_1mm_brain.nii.gz \
    --mni-template-full /path/to/MNI152_T1_1mm.nii.gz \
    --id 1.26.23 --id 1.48.18 \
    --site M1_L \
    --shared-csv ./all_subjects.csv
```

### Subject space only (no MNI registration)

```bash
python run_tms2mni.py \
    --nbe  data/TEP004.nbe \
    --t1   data/TEP004_T1.nii.gz \
    --subject-id TEP004 \
    --output-dir ./outputs
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--no-flip` | off | Disable X-axis flip in NBE→voxel transform. Use if points appear L/R swapped. |
| `--id` | — | Row ID(s) to visualize. Repeat for multiple. Use `--id all` for everything. |
| `--coord-col` | `ef` | Plot `ef` (EF max location) or `coil` coordinates. |
| `--color-by-hemi` | off | Colour L=blue, R=red in visualization. |
| `--force` | off | Re-run all stages even if outputs exist. |
| `--log-level` | `INFO` | Console log level. Log file always captures `DEBUG`. |

## Output structure

```
outputs/
└── TEP004/
    ├── xnbe/
    │   ├── landmarks_TEP004.csv
    │   ├── targets_TEP004.csv
    │   └── targets_TEP004.xlsx
    ├── registration/
    │   ├── T1_brain.nii.gz
    │   ├── T1_in_MNI.nii.gz
    │   ├── sub_to_MNI_0GenericAffine.mat
    │   ├── sub_to_MNI_1Warp.nii.gz
    │   └── sub_to_MNI_1InverseWarp.nii.gz
    ├── coordinates/
    │   ├── targets_native.csv
    │   └── targets_mni.csv
    ├── visualization/
    │   ├── stimulation_sites.html
    │   └── stimulation_sites.png
    └── logs/
        └── TEP004_20250418_143022.log
```

## Group visualization

Once multiple subjects have been processed, use `group_visualize.py` to
plot all sites together on a single glass brain with heatmap and atlas
labelling:

```bash
python group_visualize.py \
    --csv all_subjects.csv \
    --output-dir ./group_viz \
    --output-prefix my_study
```

Or pass individual per-site CSVs:

```bash
python group_visualize.py \
    --csv site1.csv --csv site2.csv --csv site3.csv \
    --output-dir ./group_viz
```

## Coordinate lookup

Find a stimulation row by its EF max coordinates:

```bash
python -m utils.lookup \
    --csv outputs/TEP004/coordinates/targets_native.csv \
    --x 833.5 --y 567.2 --z 522.3

# With tolerance
python -m utils.lookup \
    --csv targets_native.csv \
    --x 833.5 --y 567.2 --z 522.3 --tol 0.5
```

## Hemisphere convention

```
L = MRI Left  = NextStim Right ear side = higher X in RAS space
R = MRI Right = NextStim Left ear side  = lower  X in RAS space
```

Use `--no-flip` if the hemisphere check fails (landmarks appear
on the wrong side).

## Notes on brain extraction

SynthStrip (FreeSurfer 8.1.0) can crash with a PyTorch
`quantile() input tensor is too large` error on images with
non-standard orientations (e.g. PIR).  The pipeline automatically
falls back to FSL BET with robust mode (`-R -f 0.5`) in this case.
The original T1 and its voxel grid are never modified — the
skull-stripped image is only used to compute the ANTs registration
transforms.
