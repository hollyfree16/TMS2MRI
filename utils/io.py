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

from utils.atlas import ATLAS_KEYS, ATLAS_PNG_STEMS


# =============================================================================
# Path manifest
# =============================================================================

@dataclass
class PathManifest:
    """
    All expected input and output paths for a single subject run.
    Constructed once in cli.py and passed through to every stage.
    """

    # --- Inputs ---
    nbe:              Path
    t1:               Path
    mni_template:     Path | None
    mni_template_full: Path | None

    # --- Subject root ---
    subject_dir:      Path

    # --- Stage output directories ---
    xnbe_dir:         Path
    reg_dir:          Path
    coords_dir:       Path
    viz_dir:          Path
    log_dir:          Path

    # --- Stage 01 ---
    landmarks_csv:    Path
    targets_csv:      Path
    targets_xlsx:     Path

    # --- Stage 02 ---
    t1_brain:         Path

    # --- Stage 03 ---
    affine_mat:       Path
    warp:             Path
    inv_warp:         Path
    t1_in_mni:        Path | None

    # --- Stage 04 (full) ---
    targets_native:       Path
    targets_mni:          Path | None
    targets_fsaverage:    Path | None

    # --- Stage 04 (summary) ---
    targets_summary:  Path | None

    # --- Stage 05 visualization ---
    html_out:         Path
    png_out:          Path

    # Atlas PNGs — one per atlas key, stored as a dict
    # e.g. atlas_pngs["harvard_oxford"] = viz_dir / "ho_regions.png"
    atlas_pngs:       dict = field(default_factory=dict)

    # --- Shared CSV ---
    shared_csv:       Path | None = None

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
        sub    = output_dir / subject_id
        xnbe   = sub / "xnbe"
        reg    = sub / "registration"
        coords = sub / "coordinates"
        viz    = sub / "visualization"
        logs   = sub / "logs"
        reg_px = reg / "sub_to_MNI_"

        targets_native    = coords / "targets_native.csv"
        targets_mni       = (coords / "targets_mni.csv")       if mni_template else None
        targets_fsaverage = (coords / "targets_fsaverage.csv") if mni_template else None

        # Build atlas PNG paths for all atlases
        atlas_pngs = {}
        if mni_template:
            for key in ATLAS_KEYS:
                stem = ATLAS_PNG_STEMS.get(key, f"{key}_regions")
                atlas_pngs[key] = viz / f"{stem}.png"

        return cls(
            nbe               = nbe,
            t1                = t1,
            mni_template      = mni_template,
            mni_template_full = mni_template_full,

            subject_dir       = sub,
            xnbe_dir          = xnbe,
            reg_dir           = reg,
            coords_dir        = coords,
            viz_dir           = viz,
            log_dir           = logs,

            landmarks_csv     = xnbe / f"landmarks_{subject_id}.csv",
            targets_csv       = xnbe / f"targets_{subject_id}.csv",
            targets_xlsx      = xnbe / f"targets_{subject_id}.xlsx",

            t1_brain          = reg / "T1_brain.nii.gz",

            affine_mat        = Path(str(reg_px) + "0GenericAffine.mat"),
            warp              = Path(str(reg_px) + "1Warp.nii.gz"),
            inv_warp          = Path(str(reg_px) + "1InverseWarp.nii.gz"),
            t1_in_mni         = (reg / "T1_in_MNI.nii.gz") if mni_template_full else None,

            targets_native    = targets_native,
            targets_mni       = targets_mni,
            targets_fsaverage = targets_fsaverage,

            targets_summary   = (coords / "targets_summary.csv") if has_ids else None,

            html_out          = viz / "stimulation_sites.html",
            png_out           = viz / "stimulation_sites.png",

            atlas_pngs        = atlas_pngs,

            shared_csv        = shared_csv,
        )

    def required_dirs(self) -> list[Path]:
        return [self.xnbe_dir, self.reg_dir, self.coords_dir,
                self.viz_dir,  self.log_dir]

    def stage_outputs(self, stage: str) -> list[Path]:
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
            if self.targets_summary is not None:
                outs.append(self.targets_summary)
            return outs

        if stage == "visualize":
            outs = [self.html_out, self.png_out]
            # All atlas PNGs are expected outputs when MNI is available
            outs.extend(self.atlas_pngs.values())
            return outs

        if stage == "snap":
            return []

        raise ValueError(f"Unknown stage: {stage!r}")


# =============================================================================
# CSV helpers
# =============================================================================

def read_targets(path: Path) -> pd.DataFrame:
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