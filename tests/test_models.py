"""Model-builder tests (models.py). Small inputs keep TensorFlow fast."""

from __future__ import annotations

import numpy as np

from eeg_sleep_stager.config import Config
from eeg_sleep_stager.models import build_baseline, build_cnn1d


def test_build_cnn1d_shapes_and_probabilities():
    cfg = Config()
    model = build_cnn1d(cfg, input_len=300, n_classes=5)
    assert model.input_shape == (None, 300, 1)
    assert model.output_shape == (None, 5)

    x = np.random.default_rng(0).standard_normal((4, 300, 1)).astype("float32")
    probs = model.predict(x, verbose=0)
    assert probs.shape == (4, 5)
    # Softmax rows sum to 1.
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_build_cnn1d_default_input_len_from_config():
    cfg = Config()
    model = build_cnn1d(cfg)
    assert model.input_shape == (None, cfg.dataset.epoch_samples, 1)


def test_build_baseline_uses_config_hyperparams():
    cfg = Config()
    clf = build_baseline(cfg)
    assert clf.n_estimators == cfg.baseline.n_estimators
    assert clf.max_depth == cfg.baseline.max_depth
    assert clf.learning_rate == cfg.baseline.learning_rate
    assert clf.random_state == cfg.baseline.random_state
