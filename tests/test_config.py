"""Tests for the Pydantic run configuration (config.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from eeg_sleep_stager.config import Config, load_config


def test_defaults_load_without_file(tmp_path, monkeypatch):
    # No config.yaml present -> defaults, paths anchored at CWD.
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.dataset.sample_rate == 100
    assert cfg.dataset.epoch_samples == 3000
    assert cfg.paths.processed_dir.is_absolute()


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_repo_config():
    # The checked-in config.yaml must parse and validate.
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.bands.as_dict()["alpha"] == (8.0, 13.0)
    assert set(cfg.bands.as_dict()) == {"delta", "theta", "alpha", "sigma", "beta"}


def test_relative_paths_resolved_against_config_dir(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              processed_dir: ./data/processed
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.paths.processed_dir == (tmp_path / "data" / "processed").resolve()
    assert cfg.paths.epochs_path.name == "epochs.parquet"


def test_bad_band_edges_rejected(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("bands:\n  alpha: [13.0, 8.0]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_file)


def test_split_fractions_must_leave_training_data(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("split:\n  val_frac: 0.6\n  test_frac: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_file)


def test_unknown_key_rejected(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("dataset:\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_file)


def test_class_weights_none_normalized(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("cnn:\n  class_weights: none\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.cnn.class_weights is None
