"""Annotation -> class mapping tests (labels.py)."""

from __future__ import annotations

import pytest

from eeg_sleep_stager import labels
from eeg_sleep_stager.labels import map_annotation


def test_annotation_map_covers_five_classes_plus_drops():
    ids = {v for v in labels.ANNOTATION_MAP.values() if v is not None}
    assert ids == {0, 1, 2, 3, 4}
    # Stages 3 and 4 both map to N3.
    assert labels.ANNOTATION_MAP["Sleep stage 3"] == labels.STAGE_N3
    assert labels.ANNOTATION_MAP["Sleep stage 4"] == labels.STAGE_N3
    # '?' and Movement are dropped.
    assert labels.ANNOTATION_MAP["Sleep stage ?"] is None
    assert labels.ANNOTATION_MAP["Movement time"] is None


@pytest.mark.parametrize(
    "annotation,expected",
    [
        ("Sleep stage W", labels.STAGE_W),
        ("Sleep stage 1", labels.STAGE_N1),
        ("Sleep stage 2", labels.STAGE_N2),
        ("Sleep stage 3", labels.STAGE_N3),
        ("Sleep stage 4", labels.STAGE_N3),   # 3 and 4 merge into N3
        ("Sleep stage R", labels.STAGE_REM),
    ],
)
def test_map_annotation_known_stages(annotation, expected):
    assert map_annotation(annotation) == expected


@pytest.mark.parametrize(
    "annotation",
    ["Sleep stage ?", "Movement time", "unknown label", "", None],
)
def test_map_annotation_dropped_and_unknown(annotation):
    assert map_annotation(annotation) is None


def test_map_annotation_strips_whitespace():
    assert map_annotation("  Sleep stage 2\n") == labels.STAGE_N2


def test_stage_names_align_with_ids():
    assert labels.STAGE_NAMES[labels.STAGE_W] == "W"
    assert labels.STAGE_NAMES[labels.STAGE_REM] == "REM"
    assert len(labels.STAGE_NAMES) == 5
