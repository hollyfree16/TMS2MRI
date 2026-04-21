"""
utils/io.py
===========
File IO helpers — CSV read/write, NIfTI save, and the central path
manifest that all stages and the orchestrator share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Path manifest
# =============================================================================

@dataclass
class PathManifest:
    """
    All expected input and output paths for a single subject run.
    Constructed once in cli.py and passed through to every stage.

    Stages declare which paths they consume and produce — they never
    construct paths themselves.
    """

    # --- Inputs ---
    nbe:              Path
    t1:               Path
    mni_template:     Path | None   # brain-extracted MNI152
    mni_template_full: Path | None  # full-head MNI152

    # --- Subject root ---
    subject_dir:      Path          # <output_dir>/<subject_id>/

    # --- Stage output directories ---
    xnbe_dir:         Path          # 01_parse_nbe output
    reg_dir:          Path          # 03_register_mni output
    coords_dir:       Path          # 04_convert_coords output
    viz_dir:          Path          # 05_visualize output
    log_dir:          Path          # per-subject logs

    # --- Stage output files ---
    # 01_parse_nbe
    landmarks_csv:    Path
    targets_csv:      Path
    targets_xlsx:     Path

    # 02_skullstrip
    t1_brain:         Path

    # 03_register_mni
    affine_mat:       Path
    warp:             Path
    inv_warp:         Path
    t1_in_mni:        Path | None   # only if mni_template_full provided

    # 04_convert_coords  (full — all rows)
    targets_native:       Path
    targets_mni:          Path | None   # only if mni registration enabled
    targets_fsaverage:    Path | None   # only if mni registration enabled

    # 04_convert_coords  (filtered — only --id rows, written when ids provided)
    targets_native_filtered:     Path | None
    targets_mni_filtered:        Path | None
    targets_fsaverage_filtered:  Path | None

    # 05_visualize
    html_out:         Path
    png_out:          Path
    ho_regions_png:   Path          # subject-level HO atlas plot

    # --- Shared CSV (optional, cross-subject) ---
    shared_csv:       Path | None   = None

    @classmethod
    def build(
        cls,
        output_dir:        Path,
        subject_id:        str,
        nbe:               Path,
        t1:                Path,
        mni_template:      Path | None,
        mni_template_full: Path | None,
        shared_csv:        Path | None = None,
        has_ids:           bool = False,
    ) -> "PathManifest":
        """
        Construct a fully-populated PathManifest from the top-level arguments.
        All paths derived deterministically — no path construction elsewhere.

        Parameters
        ----------
        has_ids : True when --id flags were passed. Controls whether filtered
                  CSV paths are populated (non-None).
        """
        sub    = output_dir / subject_id
        xnbe   = sub / "xnbe"
        reg    = sub / "registration"
        coords = sub / "coordinates"
        viz    = sub / "visualization"
        logs   = sub / "logs"
        reg_px = reg / "sub_to_MNI_"   # ANTs prefix

        # Filtered paths only exist when IDs were requested
        def _filtered(path: Path | None) -> Path | None:
            if path is None or not has_ids:
                return None
            stem = path.stem.replace(".nii", "")   # handle .nii.gz
            suffix = "".join(path.suffixes)
            return path.parent / f"{stem}_filtered{suffix}"

        targets_native    = coords / "targets_native.csv"
        targets_mni       = (coords / "targets_mni.csv")       if mni_template else None
        targets_fsaverage = (coords / "targets_fsaverage.csv") if mni_template else None

        return cls(
            # inputs
            nbe               = nbe,
            t1                = t1,
            mni_template      = mni_template,
            mni_template_full = mni_template_full,

            # dirs
            subject_dir       = sub,
            xnbe_dir          = xnbe,
            reg_dir           = reg,
            coords_dir        = coords,
            viz_dir           = viz,
            log_dir           = logs,

            # 01
            landmarks_csv     = xnbe / f"landmarks_{subject_id}.csv",
            targets_csv       = xnbe / f"targets_{subject_id}.csv",
            targets_xlsx      = xnbe / f"targets_{subject_id}.xlsx",

            # 02
            t1_brain          = reg / "T1_brain.nii.gz",

            # 03
            affine_mat        = Path(str(reg_px) + "0GenericAffine.mat"),
            warp              = Path(str(reg_px) + "1Warp.nii.gz"),
            inv_warp          = Path(str(reg_px) + "1InverseWarp.nii.gz"),
            t1_in_mni         = (reg / "T1_in_MNI.nii.gz") if mni_template_full else None,

            # 04 full
            targets_native    = targets_native,
            targets_mni       = targets_mni,
            targets_fsaverage = targets_fsaverage,

            # 04 filtered
            targets_native_filtered    = _filtered(targets_native),
            targets_mni_filtered       = _filtered(targets_mni),
            targets_fsaverage_filtered = _filtered(targets_fsaverage),

            # 05
            html_out          = viz / "stimulation_sites.html",
            png_out           = viz / "stimulation_sites.png",
            ho_regions_png    = viz / "ho_regions.png",

            # optional
            shared_csv        = shared_csv,
        )

    def required_dirs(self) -> list[Path]:
        """All directories that must exist before the pipeline runs."""
        return [self.xnbe_dir, self.reg_dir, self.coords_dir,
                self.viz_dir,  self.log_dir]

    def stage_outputs(self, stage: str) -> list[Path]:
        """
        Return the list of output paths that determine whether a stage
        can be skipped (all must exist).

        stage: one of 'parse', 'skullstrip', 'register', 'convert',
               'visualize', 'snap'
        """
        if stage == "parse":
            return [self.landmarks_csv, self.targets_csv]

        if stage == "skullstrip":
            return [self.t1_brain]

        if stage == "register":
            outs = [self.affine_mat, self.warp, self.inv_warp]
            if self.t1_in_mni:
                outs.append(self.t1_in_mni)
            return outs

        if stage == "convert":
            outs = [self.targets_native]
            if self.targets_mni:
                outs.append(self.targets_mni)
            if self.targets_fsaverage:
                outs.append(self.targets_fsaverage)
            # Filtered outputs — only checked if paths are set
            for p in [self.targets_native_filtered,
                      self.targets_mni_filtered,
                      self.targets_fsaverage_filtered]:
                if p is not None:
                    outs.append(p)
            return outs

        if stage == "visualize":
            outs = [self.html_out, self.png_out]
            # ho_regions_png is produced when MNI coords are available
            if self.targets_mni is not None:
                outs.append(self.ho_regions_png)
            return outs

        if stage == "snap":
            # Stage 06 is now a stub — no mandatory outputs yet
            return []

        raise ValueError(f"Unknown stage: {stage!r}")


# =============================================================================
# CSV helpers
# =============================================================================

def read_targets(path: Path) -> pd.DataFrame:
    """Read a targets CSV, treating common NA sentinels correctly."""
    return pd.read_csv(path, na_values=["-", " -", "- "])


def read_landmarks(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# NIfTI helpers
# =============================================================================

def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def load_nifti(path: Path):
    import nibabel as nib
    return nib.load(str(path))