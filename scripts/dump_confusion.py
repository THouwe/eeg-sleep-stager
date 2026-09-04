"""Print the per-stage confusion matrix for models/cnn.keras to the terminal.

A quick diagnostic that reuses the pipeline's own data/model wiring
(`_predict_test_frame`) and metrics (`compute_metrics`) so the numbers match
what `evaluate` writes to reports/. Prints the raw-count matrix, the
row-normalized matrix (per-stage recall lives on the diagonal), and a
per-stage precision/recall/F1/support table -- the fastest way to see whether
macro-F1 is being capped by N1.

Usage::

    python scripts/dump_confusion.py                 # test split (held out)
    python scripts/dump_confusion.py --split val      # the noisy val estimate
    python scripts/dump_confusion.py --model baseline
"""

from __future__ import annotations

import argparse

import numpy as np

from eeg_sleep_stager.config import load_config
from eeg_sleep_stager.evaluate import _predict_test_frame, compute_metrics
from eeg_sleep_stager.labels import STAGE_NAMES


def _print_matrix(cm: np.ndarray, names: list[str], normalize: bool) -> None:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    header = "true\\pred".ljust(10) + "".join(n.rjust(9) for n in names)
    print(header)
    for i, name in enumerate(names):
        cells = "".join(
            (f"{cm[i, j]:9.2f}" if normalize else f"{int(cm[i, j]):9d}")
            for j in range(len(names))
        )
        print(name.ljust(10) + cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which subject split to score (default: test)",
    )
    parser.add_argument(
        "--model", default="cnn", choices=["cnn", "baseline"], help="model to load"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # _predict_test_frame is hardwired to the 'test' split; for a different
    # split we temporarily point the "test" key at the requested subjects.
    if args.split != "test":
        from eeg_sleep_stager.datasets import load_splits

        splits = load_splits(cfg)
        import eeg_sleep_stager.evaluate as ev

        wanted = splits[args.split]
        orig = ev.load_splits
        ev.load_splits = lambda _cfg: {"test": wanted}  # type: ignore[assignment]
        try:
            frame = _predict_test_frame(cfg, args.model)
        finally:
            ev.load_splits = orig
    else:
        frame = _predict_test_frame(cfg, args.model)

    y_true = frame["stage"].to_numpy()
    y_pred = frame["pred"].to_numpy()
    metrics = compute_metrics(y_true, y_pred)
    names = list(STAGE_NAMES)

    subjects = sorted(frame["subject_id"].unique().tolist())
    print(f"model={args.model}  split={args.split}  subjects={subjects}")
    print(f"n_epochs={metrics['n_epochs']}  "
          f"accuracy={metrics['accuracy']:.3f}  "
          f"macro-F1={metrics['macro_f1']:.3f}  "
          f"kappa={metrics['cohen_kappa']:.3f}")

    print("\nConfusion matrix (counts)")
    _print_matrix(metrics["confusion_matrix"], names, normalize=False)

    print("\nConfusion matrix (row-normalized; diagonal = per-stage recall)")
    _print_matrix(metrics["confusion_matrix"], names, normalize=True)

    print("\nPer-stage")
    print("stage".ljust(8) + "precision".rjust(11) + "recall".rjust(9)
          + "f1".rjust(8) + "support".rjust(10))
    for name in names:
        s = metrics["per_stage"][name]
        print(name.ljust(8)
              + f"{s['precision']:11.2f}{s['recall']:9.2f}"
              + f"{s['f1']:8.2f}{s['support']:10d}")


if __name__ == "__main__":
    main()
