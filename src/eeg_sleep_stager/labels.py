"""Hypnogram annotation -> AASM 5-class label mapping.

Pure functions only (no I/O): these are called inside Spark workers and are
unit-tested directly. Implemented in S3.
"""

from __future__ import annotations

from typing import Optional

# AASM 5-class ids. Sleep stages 3 and 4 are merged into N3.
STAGE_W = 0
STAGE_N1 = 1
STAGE_N2 = 2
STAGE_N3 = 3
STAGE_REM = 4

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Sleep-EDF hypnogram annotation string -> class id. None means "drop".
ANNOTATION_MAP: dict[str, Optional[int]] = {
    "Sleep stage W": STAGE_W,
    "Sleep stage 1": STAGE_N1,
    "Sleep stage 2": STAGE_N2,
    "Sleep stage 3": STAGE_N3,
    "Sleep stage 4": STAGE_N3,
    "Sleep stage R": STAGE_REM,
    "Sleep stage ?": None,
    "Movement time": None,
}


def map_annotation(annotation: Optional[str]) -> Optional[int]:
    """Map a hypnogram annotation string to an AASM class id, or None to drop it.

    Returns None for the explicitly-dropped annotations ("Sleep stage ?",
    "Movement time"), for unrecognized strings, and for None/empty input — every
    "None" means the epoch is excluded from the dataset. Surrounding whitespace
    is stripped so minor formatting differences don't cause spurious drops.

    >>> map_annotation("Sleep stage 2")
    2
    >>> map_annotation("Sleep stage 4")   # stages 3 and 4 both -> N3
    3
    >>> map_annotation("Movement time") is None
    True
    >>> map_annotation("Sleep stage ?") is None
    True
    >>> map_annotation("something else") is None
    True
    """
    if annotation is None:
        return None
    return ANNOTATION_MAP.get(annotation.strip())
