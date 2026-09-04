"""Tests for the pure ingest helpers (filename parsing, pairing, manifest)."""

from __future__ import annotations

import csv
from pathlib import Path

from eeg_sleep_stager import ingest
from eeg_sleep_stager.ingest import (
    Recording,
    discover_recordings,
    limit_subjects,
    parse_recording_stem,
    write_manifest,
)

# A realistic slice of a PhysioNet sleep-cassette listing.
SAMPLE_FILES = [
    "SC4001E0-PSG.edf",
    "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf",
    "SC4002EC-Hypnogram.edf",
    "SC4011E0-PSG.edf",
    "SC4011EH-Hypnogram.edf",
    "SC4012E0-PSG.edf",
    "SC4012EC-Hypnogram.edf",
    "SC4021E0-PSG.edf",           # subject 02, night 1 — PSG only, no hypnogram
    "index.html",                 # noise
]


def test_parse_recording_stem():
    assert parse_recording_stem("SC4001E0") == ("SC00", 1)
    assert parse_recording_stem("SC4452E0") == ("SC45", 2)
    assert parse_recording_stem("SC4001EC") == ("SC00", 1)
    assert parse_recording_stem("garbage") is None
    assert parse_recording_stem("ST7011J0") is None  # sleep-telemetry, not cassette


def test_discover_pairs_and_skips_unpaired():
    recs = discover_recordings(SAMPLE_FILES)
    keys = [(r.subject_id, r.night) for r in recs]
    # SC4021 (subject 02, night 1) has no hypnogram -> dropped.
    assert keys == [("SC00", 1), ("SC00", 2), ("SC01", 1), ("SC01", 2)]
    r0 = recs[0]
    assert r0.psg_name == "SC4001E0-PSG.edf"
    assert r0.hypnogram_name == "SC4001EC-Hypnogram.edf"
    assert r0.key == "SC4001"


def test_discover_is_sorted_by_subject_then_night():
    shuffled = list(reversed(SAMPLE_FILES))
    recs = discover_recordings(shuffled)
    assert [(r.subject_id, r.night) for r in recs] == [
        ("SC00", 1),
        ("SC00", 2),
        ("SC01", 1),
        ("SC01", 2),
    ]


def test_limit_subjects_keeps_all_nights_of_first_n():
    recs = discover_recordings(SAMPLE_FILES)
    limited = limit_subjects(recs, limit=1)
    assert {r.subject_id for r in limited} == {"SC00"}
    # Both nights of SC00 survive.
    assert sum(r.subject_id == "SC00" for r in limited) == 2
    limited2 = limit_subjects(recs, limit=2)
    assert {r.subject_id for r in limited2} == {"SC00", "SC01"}
    assert limit_subjects(recs, None) == recs


def test_write_manifest_roundtrip(tmp_path: Path):
    recs = [
        Recording("SC00", 1, "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
        Recording("SC01", 2, "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
    ]
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.csv"
    write_manifest(recs, raw, manifest)

    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["subject_id"] for r in rows] == ["SC00", "SC01"]
    assert rows[0]["night"] == "1"
    assert rows[1]["psg_path"].endswith("SC4012E0-PSG.edf")
    assert Path(rows[0]["hypnogram_path"]).name == "SC4001EC-Hypnogram.edf"


def test_manifest_columns_stable():
    assert ingest.MANIFEST_COLUMNS == [
        "subject_id",
        "night",
        "psg_path",
        "hypnogram_path",
    ]
