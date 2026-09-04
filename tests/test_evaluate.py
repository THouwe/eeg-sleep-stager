"""Evaluation metric + plotting tests (evaluate.py)."""

from __future__ import annotations

import numpy as np

from eeg_sleep_stager.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_hypnogram,
)
from eeg_sleep_stager.labels import STAGE_NAMES


def test_compute_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 2, 2, 0])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0
    assert m["n_epochs"] == 8
    # Confusion matrix is diagonal.
    cm = np.array(m["confusion_matrix"])
    assert (cm - np.diag(np.diag(cm))).sum() == 0


def test_compute_metrics_structure_and_labels():
    y_true = np.array([0, 0, 1, 2, 2, 3, 4])
    y_pred = np.array([0, 1, 1, 2, 2, 3, 0])
    m = compute_metrics(y_true, y_pred)
    # Always 5 classes, even if some are absent.
    assert list(m["per_stage"]) == list(STAGE_NAMES)
    assert np.array(m["confusion_matrix"]).shape == (5, 5)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert -1.0 <= m["cohen_kappa"] <= 1.0
    # W has support 2, one misclassified as N1 -> recall 0.5.
    assert m["per_stage"]["W"]["support"] == 2
    assert m["per_stage"]["W"]["recall"] == 0.5


def test_compute_metrics_kappa_zero_for_constant_pred():
    y_true = np.array([0, 1, 2, 3, 4] * 4)
    y_pred = np.zeros_like(y_true)  # always predict W
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.2
    assert abs(m["cohen_kappa"]) < 1e-9  # no agreement beyond chance


def test_plot_confusion_matrix_writes_png(tmp_path):
    cm = [[5, 1, 0, 0, 0], [0, 3, 1, 0, 0], [0, 0, 8, 1, 0],
          [0, 0, 1, 6, 0], [0, 0, 0, 0, 4]]
    out = plot_confusion_matrix(cm, STAGE_NAMES, tmp_path / "cm.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_hypnogram_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    true_seq = rng.integers(0, 5, size=120)
    pred_seq = rng.integers(0, 5, size=120)
    out = plot_hypnogram(true_seq, pred_seq, tmp_path / "hyp.png")
    assert out.exists() and out.stat().st_size > 0
