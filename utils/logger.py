"""
utils/logger.py
===============
Logger factory for tms2mni.

Every stage gets its own named logger under the root `tms2mni` hierarchy:
    tms2mni.stages.skullstrip
    tms2mni.stages.register_mni
    etc.

All loggers share the same handlers (console + rotating file) which are
attached once to the root `tms2mni` logger at pipeline startup via
`configure_logging()`.

Usage
-----
# In run_tms2mni.py / staging.py — call once at startup:
    from utils.logger import configure_logging
    configure_logging(log_dir=paths.log_dir, subject_id=args.subject_id,
                      level=logging.DEBUG)

# In each stage module:
    from utils.logger import get_stage_logger
    log = get_stage_logger("skullstrip")
    log.info("Starting brain extraction")
    log.debug("SynthStrip command: %s", cmd)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Root logger name — all stage loggers are children of this
ROOT_LOGGER = "tms2mni"

# Console format: clean, stage-prefixed
CONSOLE_FMT = "[%(name)s] %(levelname)s  %(message)s"

# File format: full timestamp + level + logger name
FILE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_dir: Path | str,
    subject_id: str,
    level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """
    Configure the root tms2mni logger with a console handler and a
    rotating file handler writing to <log_dir>/<subject_id>_<timestamp>.log.

    Parameters
    ----------
    log_dir:    Directory to write log files into (created if absent).
    subject_id: Used in the log filename.
    level:      Console log level  (default INFO).
    file_level: File log level     (default DEBUG — always verbose on disk).

    Returns
    -------
    Path to the log file created.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"{subject_id}_{timestamp}.log"

    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(logging.DEBUG)   # let handlers filter individually

    # Avoid duplicate handlers if configure_logging() is called more than once
    if root.handlers:
        root.handlers.clear()

    # --- Console handler ---
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FMT))
    root.addHandler(console)

    # --- File handler (always DEBUG so full detail is on disk) ---
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
    root.addHandler(fh)

    root.info("Log file: %s", log_path)
    return log_path


def get_stage_logger(stage_name: str) -> logging.Logger:
    """
    Return a named logger for a pipeline stage.

    The logger is a child of the root tms2mni logger so it inherits
    its handlers automatically.

    Parameters
    ----------
    stage_name: Short name for the stage, e.g. "skullstrip", "register_mni".
                Will appear in log output as tms2mni.stages.<stage_name>.

    Example
    -------
        log = get_stage_logger("skullstrip")
        log.info("Using SynthStrip")
    """
    return logging.getLogger(f"{ROOT_LOGGER}.stages.{stage_name}")


def get_util_logger(util_name: str) -> logging.Logger:
    """
    Return a named logger for a utility module.
    Appears as tms2mni.utils.<util_name> in log output.
    """
    return logging.getLogger(f"{ROOT_LOGGER}.utils.{util_name}")
