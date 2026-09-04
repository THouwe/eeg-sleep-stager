"""Model builders: 1-D CNN (Keras) and GradientBoosting baseline (sklearn) (S6).

The CNN is a modest Conv1D stack over the raw single-channel epoch — a large
first kernel to capture low-frequency (delta/theta) morphology, then narrower
filters, global average pooling, and a softmax head. It is deliberately small:
the goal is a correct, honest baseline, not SOTA.

TensorFlow/sklearn are imported inside the builders so importing this module
(e.g. for the CLI) stays cheap.
"""

from __future__ import annotations

from typing import Optional

from .config import N_CLASSES, Config


def build_cnn1d(
    cfg: Config,
    input_len: Optional[int] = None,
    n_classes: int = N_CLASSES,
):
    """Build and compile the 1-D CNN over the raw epoch signal.

    Input shape is (input_len, 1); output is a softmax over ``n_classes``.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    length = input_len or cfg.dataset.epoch_samples

    inputs = layers.Input(shape=(length, 1), name="epoch_signal")
    x = inputs
    # (filters, kernel_size) per conv block; first kernel is wide for slow waves.
    for filters, kernel in [(32, 50), (64, 8), (128, 8)]:
        x = layers.Conv1D(filters, kernel, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="stage")(x)

    model = models.Model(inputs, outputs, name="cnn1d_sleep_stager")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.cnn.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline(cfg: Config):
    """Build the sklearn GradientBoosting baseline on engineered features."""
    from sklearn.ensemble import GradientBoostingClassifier

    b = cfg.baseline
    return GradientBoostingClassifier(
        n_estimators=b.n_estimators,
        learning_rate=b.learning_rate,
        max_depth=b.max_depth,
        random_state=b.random_state,
    )
