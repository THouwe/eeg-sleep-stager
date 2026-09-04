"""Tests for the pure ETL epoch-planning helpers (no Spark, no EDF I/O)."""

from __future__ import annotations

import numpy as np

from eeg_sleep_stager.config import BandsConfig
from eeg_sleep_stager.etl import (
    BAND_ORDER,
    EPOCH_COLUMNS,
    EpochPlan,
    EtlParams,
    _epoch_features,
    plan_epochs,
    trim_wake,
)

FS = 100
EPOCH_SEC = 30
N_PER_EPOCH = FS * EPOCH_SEC


def test_plan_epochs_expands_multi_epoch_annotations():
    # One 90 s Wake span (3 epochs) then one 60 s N2 span (2 epochs).
    onsets = [0.0, 90.0]
    durations = [90.0, 60.0]
    descs = ["Sleep stage W", "Sleep stage 2"]
    n_samples = 200 * FS  # plenty
    plans = plan_epochs(onsets, durations, descs, FS, EPOCH_SEC, n_samples)
    assert [p.stage for p in plans] == [0, 0, 0, 2, 2]
    assert [p.start_sample for p in plans] == [
        0, 3000, 6000,        # wake epochs
        9000, 12000,          # n2 epochs
    ]


def test_plan_epochs_drops_unknown_and_out_of_range():
    onsets = [0.0, 30.0, 60.0]
    durations = [30.0, 30.0, 30.0]
    descs = ["Sleep stage ?", "Sleep stage 2", "Sleep stage 2"]
    # Signal only long enough for the first two epochs (0..6000); third is cut.
    n_samples = 6000
    plans = plan_epochs(onsets, durations, descs, FS, EPOCH_SEC, n_samples)
    # '?' dropped, third epoch out of range -> only the 30..60 s N2 survives.
    assert [(p.start_sample, p.stage) for p in plans] == [(3000, 2)]


def test_trim_wake_keeps_pad_around_sleep():
    # 100 wake epochs, then 10 sleep (N2), then 100 wake.
    plans = (
        [EpochPlan(i * N_PER_EPOCH, 0) for i in range(100)]
        + [EpochPlan((100 + i) * N_PER_EPOCH, 2) for i in range(10)]
        + [EpochPlan((110 + i) * N_PER_EPOCH, 0) for i in range(100)]
    )
    # pad of 30 min = 60 epochs each side.
    trimmed = trim_wake(plans, EPOCH_SEC, wake_pad_min=30)
    stages = [p.stage for p in trimmed]
    # 60 wake + 10 sleep + 60 wake.
    assert stages.count(2) == 10
    assert stages.count(0) == 120
    assert len(trimmed) == 130


def test_trim_wake_no_sleep_returns_unchanged():
    plans = [EpochPlan(i * N_PER_EPOCH, 0) for i in range(50)]
    assert trim_wake(plans, EPOCH_SEC, wake_pad_min=30) == plans


def test_trim_wake_short_padding_clips_to_bounds():
    plans = (
        [EpochPlan(i * N_PER_EPOCH, 0) for i in range(5)]
        + [EpochPlan((5 + i) * N_PER_EPOCH, 3) for i in range(3)]
        + [EpochPlan((8 + i) * N_PER_EPOCH, 0) for i in range(5)]
    )
    # pad 0 min -> only the sleep epochs remain.
    trimmed = trim_wake(plans, EPOCH_SEC, wake_pad_min=0)
    assert [p.stage for p in trimmed] == [3, 3, 3]


def test_epoch_features_keys_and_values():
    params = EtlParams(
        channel="EEG Fpz-Cz", sample_rate=FS, epoch_sec=EPOCH_SEC,
        wake_pad_min=30, bands=BandsConfig().as_dict(),
    )
    t = np.arange(N_PER_EPOCH) / FS
    sig = np.sin(2 * np.pi * 10 * t)  # 10 Hz -> alpha
    feats = _epoch_features(sig, params)
    expected_keys = (
        [f"feat_bp_{b}" for b in BAND_ORDER]
        + ["feat_spec_entropy", "feat_hjorth_activity",
           "feat_hjorth_mobility", "feat_hjorth_complexity", "feat_rms"]
    )
    assert set(feats) == set(expected_keys)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["feat_bp_alpha"] > feats["feat_bp_delta"]


def test_epoch_columns_match_feature_layout():
    # The output tuple layout and feature dict must stay in sync.
    assert EPOCH_COLUMNS[:5] == [
        "subject_id", "night", "epoch_idx", "stage", "signal",
    ]
    assert EPOCH_COLUMNS[5:10] == [f"feat_bp_{b}" for b in BAND_ORDER]
    assert len(EPOCH_COLUMNS) == 15
