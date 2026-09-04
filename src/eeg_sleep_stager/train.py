"""Training loops with class weights, early stopping, checkpointing (S6).

Two entry points behind one CLI command:

- ``cnn``      — the 1-D CNN on the raw signal, with inverse-frequency class
                 weights (N1 is ~5% of epochs), early stopping on **val macro-F1**
                 (the sleep-staging metric, not accuracy — accuracy is dominated
                 by N2), and best-checkpoint saving.
- ``baseline`` — the GradientBoosting model on engineered features, with
                 balanced sample weights.

Artifacts land in ``models/``: ``cnn.keras`` and ``baseline.joblib``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .datasets import (
    compute_class_weights,
    load_feature_split,
    load_signal_split,
)
from .models import build_baseline, build_cnn1d


def _apply_overrides(
    cfg: Config,
    epochs: Optional[int],
    batch_size: Optional[int],
    lr: Optional[float],
    class_weights: Optional[str],
) -> None:
    if epochs is not None:
        cfg.cnn.epochs = epochs
    if batch_size is not None:
        cfg.cnn.batch_size = batch_size
    if lr is not None:
        cfg.cnn.learning_rate = lr
    if class_weights is not None:
        cfg.cnn.class_weights = None if class_weights.lower() in {
            "none", "null", ""
        } else "auto"


def _macro_f1_callback(val_x: np.ndarray, val_y: np.ndarray):
    """Keras callback logging val macro-F1 each epoch (for early stopping)."""
    import tensorflow as tf
    from sklearn.metrics import f1_score

    class MacroF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            preds = np.argmax(self.model.predict(val_x, verbose=0), axis=1)
            f1 = f1_score(val_y, preds, average="macro")
            logs["val_macro_f1"] = f1
            print(f"  epoch {epoch + 1}: val_macro_f1={f1:.4f}")

    return MacroF1()


def train_cnn(cfg: Config) -> Path:
    """Train the 1-D CNN and checkpoint the best model by val macro-F1."""
    import tensorflow as tf

    X_train, y_train = load_signal_split(cfg, "train")
    X_val, y_val = load_signal_split(cfg, "val")
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    model = build_cnn1d(cfg, input_len=X_train.shape[1])

    class_weight = (
        compute_class_weights(y_train)
        if cfg.cnn.class_weights == "auto"
        else None
    )
    if class_weight:
        print("class weights:", {k: round(v, 2) for k, v in class_weight.items()})

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.paths.models_dir / "cnn.keras"

    callbacks = [
        _macro_f1_callback(X_val, y_val),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=cfg.cnn.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.cnn.epochs,
        batch_size=cfg.cnn.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    # Ensure the restored best weights are on disk even if no epoch improved.
    model.save(ckpt)
    print(f"Saved CNN -> {ckpt}")
    return ckpt


def train_baseline(cfg: Config) -> Path:
    """Train the GradientBoosting baseline on engineered features."""
    import joblib
    from sklearn.utils.class_weight import compute_sample_weight

    X_train, y_train = load_feature_split(cfg, "train")
    print(f"train features: {X_train.shape}")

    model = build_baseline(cfg)
    sample_weight = None
    if cfg.cnn.class_weights == "auto":
        sample_weight = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.models_dir / "baseline.joblib"
    joblib.dump(model, out)
    print(f"Saved baseline -> {out}")
    return out


def train(
    cfg: Config,
    model: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    class_weights: Optional[str] = None,
) -> Path:
    """Train the chosen model ("cnn" or "baseline")."""
    _apply_overrides(cfg, epochs, batch_size, lr, class_weights)
    if model == "cnn":
        return train_cnn(cfg)
    if model == "baseline":
        return train_baseline(cfg)
    raise ValueError(f"unknown model {model!r}; expected 'cnn' or 'baseline'")
