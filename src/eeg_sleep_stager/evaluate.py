"""Held-out evaluation: metrics, confusion matrix, hypnogram overlay (S7).

Evaluates a trained model on the **test subjects only** (held out by the
subject-wise split) and reports the metrics that matter for sleep staging:
accuracy, macro-F1, and Cohen's kappa (the standard agreement metric), plus
per-stage precision/recall/F1 and a confusion matrix. It also draws a
predicted-vs-expert hypnogram for one test night so the errors are legible, not
just tabulated.

`compute_metrics` is pure and unit-tested; plotting and the model/data wiring
sit on top. Outputs land in ``reports/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import N_CLASSES, Config
from .datasets import FEATURE_COLUMNS, load_splits
from .labels import STAGE_NAMES

# Hypnogram y-axis order (top -> bottom), the conventional clinical layout.
_HYPNO_ORDER = [0, 4, 1, 2, 3]  # W, REM, N1, N2, N3


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict:
    """Compute the evaluation metrics as a JSON-serializable dict."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_stage = {
        STAGE_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "per_stage": per_stage,
        "confusion_matrix": cm.tolist(),
        "class_names": list(STAGE_NAMES),
        "n_epochs": int(len(y_true)),
    }


def plot_confusion_matrix(cm, class_names, path: Path, normalize: bool = True) -> Path:
    """Save a confusion-matrix heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    display = cm.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() or 1)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Expert stage")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    thresh = (display.max() or 1) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j, i, txt, ha="center", va="center",
                color="white" if display[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_hypnogram(
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    path: Path,
    title: str = "Hypnogram: expert vs predicted",
) -> Path:
    """Save a predicted-vs-expert hypnogram overlay for one night."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Map stage id -> vertical position (clinical order, W at top).
    ypos = {stage: len(_HYPNO_ORDER) - 1 - row for row, stage in enumerate(_HYPNO_ORDER)}
    true_y = [ypos[int(s)] for s in y_true_seq]
    pred_y = [ypos[int(s)] for s in y_pred_seq]
    hours = np.arange(len(y_true_seq)) * 30.0 / 3600.0  # 30 s epochs -> hours

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(hours, true_y, where="post", label="expert", color="black", lw=1.4)
    ax.step(hours, pred_y, where="post", label="predicted", color="tab:red",
            lw=1.1, alpha=0.75)
    ax.set_yticks(range(len(_HYPNO_ORDER)),
                  labels=[STAGE_NAMES[s] for s in reversed(_HYPNO_ORDER)])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _predict_test_frame(cfg: Config, model: str):
    """Load the test split, run the trained model, return a frame with predictions."""
    import pandas as pd

    subjects = load_splits(cfg)["test"]
    if not subjects:
        raise ValueError("test split is empty")
    meta_cols = ["subject_id", "night", "epoch_idx", "stage"]
    epochs_path = cfg.paths.epochs_path
    filt = [("subject_id", "in", list(subjects))]

    if model == "cnn":
        df = pd.read_parquet(epochs_path, columns=meta_cols + ["signal"], filters=filt)
        X = np.stack(df["signal"].to_list()).astype(np.float32)[..., np.newaxis]
        import tensorflow as tf

        net = tf.keras.models.load_model(cfg.paths.models_dir / "cnn.keras")
        y_pred = np.argmax(net.predict(X, verbose=0), axis=1)
    elif model == "baseline":
        df = pd.read_parquet(
            epochs_path, columns=meta_cols + FEATURE_COLUMNS, filters=filt
        )
        import joblib

        clf = joblib.load(cfg.paths.models_dir / "baseline.joblib")
        y_pred = clf.predict(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    else:
        raise ValueError(f"unknown model {model!r}; expected 'cnn' or 'baseline'")

    out = df[meta_cols].copy()
    out["pred"] = np.asarray(y_pred).astype(int)
    return out


def evaluate(cfg: Config, model: str = "cnn") -> Path:
    """Evaluate ``model`` on the test subjects; write metrics.json + PNGs.

    Returns the path to metrics.json.
    """
    frame = _predict_test_frame(cfg, model)
    y_true = frame["stage"].to_numpy()
    y_pred = frame["pred"].to_numpy()

    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = model
    metrics["test_subjects"] = sorted(frame["subject_id"].unique().tolist())

    reports = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    metrics_path = reports / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        metrics["confusion_matrix"], STAGE_NAMES,
        reports / f"confusion_matrix_{model}.png",
    )

    # Hypnogram overlay for one test night (first subject-night, in epoch order).
    first = frame.sort_values(["subject_id", "night", "epoch_idx"]).groupby(
        ["subject_id", "night"], sort=False, observed=True
    )
    (sid, night), night_df = next(iter(first))
    plot_hypnogram(
        night_df["stage"].to_numpy(),
        night_df["pred"].to_numpy(),
        reports / f"hypnogram_{model}.png",
        title=f"Hypnogram {sid} night {night}: expert vs predicted ({model})",
    )

    print(f"[{model}] test subjects: {metrics['test_subjects']}")
    print(f"  accuracy    = {metrics['accuracy']:.3f}")
    print(f"  macro-F1    = {metrics['macro_f1']:.3f}")
    print(f"  Cohen kappa = {metrics['cohen_kappa']:.3f}")
    print("  per-stage F1: " + ", ".join(
        f"{name}={v['f1']:.2f}" for name, v in metrics["per_stage"].items()
    ))
    print(f"Wrote {metrics_path} + confusion/hypnogram PNGs -> {reports}")
    return metrics_path
