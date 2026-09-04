"""Dataset-loader tests (datasets.py): class weights and split loading."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eeg_sleep_stager.config import load_config
from eeg_sleep_stager.datasets import (
    FEATURE_COLUMNS,
    compute_class_weights,
    load_feature_split,
    load_signal_split,
)


def test_feature_columns_are_the_ten_engineered_features():
    assert len(FEATURE_COLUMNS) == 10
    assert FEATURE_COLUMNS[0] == "feat_bp_delta"
    assert "feat_rms" in FEATURE_COLUMNS
    assert all(c.startswith("feat_") for c in FEATURE_COLUMNS)


def test_compute_class_weights_upweights_minority():
    # Class 1 (N1-like) is rare -> should get the largest weight.
    y = np.array([0] * 90 + [1] * 5 + [2] * 5)
    weights = compute_class_weights(y)
    assert set(weights) == {0, 1, 2}
    assert weights[1] == max(weights.values())
    assert weights[1] > weights[0]


@pytest.fixture
def tiny_store(tmp_path):
    """A synthetic Spark-layout epoch store + splits.json, wired into a Config."""
    cfg = load_config("config.yaml")
    cfg.paths.processed_dir = tmp_path

    epoch_len = cfg.dataset.epoch_samples
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        sid = f"SC{i:02d}"
        for stage in range(5):
            for _ in range(4):
                row = {
                    "subject_id": sid,
                    "night": 1,
                    "stage": stage,
                    "signal": rng.standard_normal(epoch_len).astype("float32").tolist(),
                }
                for c in FEATURE_COLUMNS:
                    row[c] = float(rng.random())
                rows.append(row)
    pd.DataFrame(rows).to_parquet(
        cfg.paths.epochs_path, engine="pyarrow", index=False,
        partition_cols=["subject_id"],
    )

    splits = {
        "seed": 42, "val_frac": 0.25, "test_frac": 0.25,
        "train": ["SC00", "SC01"], "val": ["SC02"], "test": ["SC03"],
    }
    cfg.paths.splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return cfg


def test_load_signal_split_shapes(tiny_store):
    cfg = tiny_store
    X, y = load_signal_split(cfg, "train")
    # 2 train subjects * 5 stages * 4 epochs = 40 epochs.
    assert X.shape == (40, cfg.dataset.epoch_samples, 1)
    assert X.dtype == np.float32
    assert y.shape == (40,)
    assert set(y.tolist()) == {0, 1, 2, 3, 4}


def test_load_feature_split_shapes(tiny_store):
    cfg = tiny_store
    X, y = load_feature_split(cfg, "val")
    # 1 val subject * 5 stages * 4 epochs = 20 epochs, 10 features.
    assert X.shape == (20, len(FEATURE_COLUMNS))
    assert y.shape == (20,)


def test_split_loading_is_subject_filtered(tiny_store):
    # test split is SC03 only -> 20 epochs, disjoint from train.
    cfg = tiny_store
    Xtr, _ = load_feature_split(cfg, "train")
    Xte, _ = load_feature_split(cfg, "test")
    assert len(Xtr) == 40
    assert len(Xte) == 20
