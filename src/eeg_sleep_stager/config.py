"""Typed run configuration.

`config.yaml` at the repo root is the single source of truth for paths, band
edges, split fractions, and model hyperparameters. It is parsed and validated
here into a nested Pydantic model so the rest of the pipeline reads well-typed,
already-checked values instead of poking at raw dicts.

Usage::

    from eeg_sleep_stager.config import load_config
    cfg = load_config()                 # ./config.yaml
    cfg = load_config("other.yaml")     # explicit path
    cfg.paths.processed_dir             # -> Path, resolved absolute
    cfg.bands.as_dict()["alpha"]        # -> (8.0, 13.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Number of AASM classes (W, N1, N2, N3, REM). Kept here so every module agrees.
N_CLASSES = 5


class _Base(BaseModel):
    # Reject unknown keys so a typo in config.yaml fails loudly instead of
    # silently doing nothing.
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    data_dir: Path = Path("./data")
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    models_dir: Path = Path("./models")
    reports_dir: Path = Path("./reports")

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.csv"

    @property
    def epochs_path(self) -> Path:
        return self.processed_dir / "epochs.parquet"

    @property
    def splits_path(self) -> Path:
        return self.processed_dir / "splits.json"


class DatasetConfig(_Base):
    name: str = "sleep-cassette"
    source_url: str = (
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    )
    channel: str = "EEG Fpz-Cz"
    sample_rate: int = Field(default=100, gt=0)
    epoch_sec: int = Field(default=30, gt=0)
    # Minutes of Wake kept on each side of the sleep period; the rest of the
    # long wake tails are trimmed. Set to a large number to disable trimming.
    wake_pad_min: int = Field(default=30, ge=0)

    @property
    def epoch_samples(self) -> int:
        """Samples per epoch (e.g. 100 Hz * 30 s = 3000)."""
        return self.sample_rate * self.epoch_sec


class BandsConfig(_Base):
    delta: tuple[float, float] = (0.5, 4.0)
    theta: tuple[float, float] = (4.0, 8.0)
    alpha: tuple[float, float] = (8.0, 13.0)
    sigma: tuple[float, float] = (11.0, 16.0)
    beta: tuple[float, float] = (16.0, 30.0)

    @field_validator("*")
    @classmethod
    def _low_below_high(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if low >= high:
            raise ValueError(f"band low ({low}) must be < high ({high})")
        if low < 0:
            raise ValueError(f"band low ({low}) must be >= 0")
        return v

    def as_dict(self) -> dict[str, tuple[float, float]]:
        """Ordered mapping band-name -> (low, high), matching feature columns."""
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "beta": self.beta,
        }


class SplitConfig(_Base):
    seed: int = 42
    val_frac: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_frac: float = Field(default=0.15, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _fractions_leave_training_data(self) -> "SplitConfig":
        if self.val_frac + self.test_frac >= 1.0:
            raise ValueError(
                "val_frac + test_frac must be < 1.0 to leave a training split "
                f"(got {self.val_frac} + {self.test_frac})"
            )
        return self


class SparkConfig(_Base):
    master: str = "local[*]"
    app_name: str = "eeg-sleep-stager-etl"
    shuffle_partitions: int = Field(default=16, gt=0)


class CnnConfig(_Base):
    batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    epochs: int = Field(default=30, gt=0)
    early_stopping_patience: int = Field(default=6, ge=0)
    # "auto" -> inverse-frequency weights; None -> unweighted.
    class_weights: Optional[Literal["auto"]] = "auto"

    @field_validator("class_weights", mode="before")
    @classmethod
    def _normalize_class_weights(cls, v: Union[str, None]) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str) and v.lower() in {"none", "null", ""}:
            return None
        return v


class BaselineConfig(_Base):
    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    max_depth: int = Field(default=3, gt=0)
    random_state: int = 42


class Config(_Base):
    """Top-level validated run configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    bands: BandsConfig = Field(default_factory=BandsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    spark: SparkConfig = Field(default_factory=SparkConfig)
    cnn: CnnConfig = Field(default_factory=CnnConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)


def _resolve_paths(cfg: Config, base_dir: Path) -> None:
    """Make every path in cfg.paths absolute, anchored at base_dir if relative."""
    for name, value in cfg.paths.__dict__.items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(cfg.paths, name, (base_dir / value).resolve())


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Load and validate the run configuration.

    Args:
        path: Path to a YAML config. Defaults to ``config.yaml`` in the current
            working directory. Relative paths inside the config are resolved
            against the config file's directory (or CWD when a default empty
            config is used).

    Returns:
        A validated :class:`Config`. Missing sections fall back to defaults.
    """
    if path is None:
        cfg_path = Path("config.yaml")
    else:
        cfg_path = Path(path)

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        base_dir = cfg_path.resolve().parent
    else:
        if path is not None:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        raw = {}
        base_dir = Path.cwd()

    cfg = Config.model_validate(raw)
    _resolve_paths(cfg, base_dir)
    return cfg
