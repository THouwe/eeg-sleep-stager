"""Load the epoch store into numpy / tf.data for a given split (S6).

Reads `splits.json` and the partitioned `epochs.parquet` (via pandas/pyarrow,
so no Spark is needed at train time), filtering to the subjects of the requested
split. Two views are exposed:

- signal view  -> X shape (N, epoch_samples, 1) for the 1-D CNN
- feature view -> X shape (N, n_features)       for the GBM baseline

Subject filtering happens through Parquet partition pruning, so only the needed
subjects are read off disk.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from .config import Config
from .etl import EPOCH_COLUMNS

# The engineered feature columns, in a fixed order.
FEATURE_COLUMNS = [c for c in EPOCH_COLUMNS if c.startswith("feat_")]


def load_splits(cfg: Config) -> dict:
    """Read splits.json produced by `split`."""
    path = cfg.paths.splits_path
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}. Run `split` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_split_df(cfg: Config, subjects: list[str], columns: list[str]):
    """Read the given columns for the given subjects from the epoch store."""
    import pandas as pd

    epochs_path = cfg.paths.epochs_path
    if not epochs_path.exists():
        raise FileNotFoundError(
            f"Epoch store not found: {epochs_path}. Run `etl` first."
        )
    if not subjects:
        raise ValueError("split has no subjects to load")
    return pd.read_parquet(
        epochs_path,
        columns=columns,
        filters=[("subject_id", "in", list(subjects))],
    )


def load_signal_split(cfg: Config, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the CNN: X is (N, epoch_samples, 1) float32, y int64."""
    subjects = load_splits(cfg)[split]
    df = _read_split_df(cfg, subjects, ["signal", "stage"])
    X = np.stack(df["signal"].to_list()).astype(np.float32)
    X = X[..., np.newaxis]  # add channel axis
    y = df["stage"].to_numpy(dtype=np.int64)
    return X, y


def load_feature_split(cfg: Config, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the baseline: X is (N, n_features) float32, y int64."""
    subjects = load_splits(cfg)[split]
    df = _read_split_df(cfg, subjects, FEATURE_COLUMNS + ["stage"])
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["stage"].to_numpy(dtype=np.int64)
    return X, y


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency (balanced) class weights, e.g. to up-weight rare N1."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
):
    """Wrap arrays in a batched, prefetched tf.data.Dataset."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(X), seed=seed, reshuffle_each_iteration=True
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
