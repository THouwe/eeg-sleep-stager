"""Extract one held-out test subject-night into the bundled demo sample.

The web demo ships a single night so visitors can try it in one click without
finding a 50 MB EDF. That night is pulled from the **test** split -- a subject the
model never saw in training -- so the sample's metrics are honest, not memorized.

The epoch store already holds exactly what the sample needs: the per-recording
z-scored ``signal`` and the expert ``stage``. We copy one subject-night out of it
into a small compressed ``.npz`` plus a JSON sidecar, and the demo feeds that
through the *same* ``inference.stage_epochs`` path an upload uses.

Usage::

    python scripts/make_sample.py                 # default: SC14 night 1
    python scripts/make_sample.py --subject SC14 --night 1
    python scripts/make_sample.py --config config.yaml --out app_assets/sample_night.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_sleep_stager.config import load_config

# Default sample: a test-split subject-night with all five stages well represented
# (incl. clear N3), which makes for a legible demo hypnogram.
DEFAULT_SUBJECT = "SC14"
DEFAULT_NIGHT = 1


def make_sample(
    config: str | None,
    subject: str,
    night: int,
    out: Path,
) -> Path:
    import pandas as pd

    cfg = load_config(config)

    splits = json.loads(cfg.paths.splits_path.read_text(encoding="utf-8"))
    split_name = next((k for k in ("test", "val", "train") if subject in splits.get(k, [])), None)
    if split_name != "test":
        # Not fatal -- but the whole point is an unseen subject, so make noise.
        print(
            f"WARNING: {subject} is in split {split_name!r}, not 'test'. The sample's "
            f"metrics will not be an honest held-out estimate."
        )

    df = pd.read_parquet(
        cfg.paths.epochs_path,
        columns=["subject_id", "night", "epoch_idx", "stage", "signal"],
        filters=[("subject_id", "==", subject), ("night", "==", night)],
    )
    if df.empty:
        raise SystemExit(
            f"No epochs for {subject} night {night} in {cfg.paths.epochs_path}. "
            f"Run the ETL first, or pick another subject-night."
        )
    df = df.sort_values("epoch_idx")

    signal = np.stack(df["signal"].to_list()).astype(np.float32)  # (N, 3000)
    stage = df["stage"].to_numpy(dtype=np.int8)  # (N,)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, signal=signal, stage=stage)

    meta = {
        "subject_id": subject,
        "night": int(night),
        "n_epochs": int(signal.shape[0]),
        "epoch_samples": int(signal.shape[1]),
        "sample_rate": cfg.dataset.sample_rate,
        "epoch_sec": cfg.dataset.epoch_sec,
        "channel": cfg.dataset.channel,
        "split": split_name,
        "note": (
            "Held-out test subject (unseen in training). Signal is per-recording "
            "z-scored; stages are expert AASM labels. Wake padding trimmed by the ETL."
        ),
    }
    meta_path = out.with_name("sample_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(
        f"Wrote {out} ({size_mb:.1f} MB) and {meta_path}: "
        f"{subject} night {night}, {signal.shape[0]} epochs ({split_name} split)."
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the bundled demo sample night.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--subject", default=DEFAULT_SUBJECT, help="Subject id (e.g. SC14).")
    p.add_argument("--night", type=int, default=DEFAULT_NIGHT, help="Night (1 or 2).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("app_assets/sample_night.npz"),
        help="Output .npz path (sidecar sample_meta.json written alongside).",
    )
    args = p.parse_args()
    make_sample(args.config, args.subject, args.night, args.out)


if __name__ == "__main__":
    main()
