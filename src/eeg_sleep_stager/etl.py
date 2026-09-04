"""PySpark ETL: manifest -> partitioned epochs.parquet (S4).

The manifest (one row per recording) is read into a Spark DataFrame and
repartitioned so each recording lands on its own partition; ``mapPartitions``
then runs :func:`process_recording` per recording. That function is where MNE
reads the EDF, the signal is z-scored per recording, epochs are cut and labeled
from the hypnogram, wake padding is trimmed, and per-epoch features are
computed. The resulting rows are written to Parquet partitioned by subject.

The epoch-planning helpers (:func:`plan_epochs`, :func:`trim_wake`) are pure and
unit-tested; the MNE/Spark layers wrap them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from .config import Config
from .features import band_powers, hjorth, spectral_entropy
from .labels import STAGE_NAMES, STAGE_W, map_annotation

# Band feature columns, in a fixed order matching the config band keys.
BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta"]

# Output column order (must match the Spark schema in _epochs_schema()).
EPOCH_COLUMNS = [
    "subject_id",
    "night",
    "epoch_idx",
    "stage",
    "signal",
    "feat_bp_delta",
    "feat_bp_theta",
    "feat_bp_alpha",
    "feat_bp_sigma",
    "feat_bp_beta",
    "feat_spec_entropy",
    "feat_hjorth_activity",
    "feat_hjorth_mobility",
    "feat_hjorth_complexity",
    "feat_rms",
]

_EPS = 1e-8


@dataclass(frozen=True)
class EtlParams:
    """Primitive extraction parameters (picklable, shipped to Spark workers)."""

    channel: str
    sample_rate: int
    epoch_sec: int
    wake_pad_min: int
    bands: dict[str, tuple[float, float]]

    @classmethod
    def from_config(cls, cfg: Config) -> "EtlParams":
        return cls(
            channel=cfg.dataset.channel,
            sample_rate=cfg.dataset.sample_rate,
            epoch_sec=cfg.dataset.epoch_sec,
            wake_pad_min=cfg.dataset.wake_pad_min,
            bands=cfg.bands.as_dict(),
        )


@dataclass(frozen=True)
class EpochPlan:
    """A single planned epoch: where it starts and what stage it is."""

    start_sample: int
    stage: int


def plan_epochs(
    onsets: Iterable[float],
    durations: Iterable[float],
    descriptions: Iterable[str],
    sample_rate: int,
    epoch_sec: int,
    n_samples: int,
) -> list[EpochPlan]:
    """Expand hypnogram annotations into per-epoch (start_sample, stage) plans.

    Each annotation spans ``duration / epoch_sec`` epochs of one stage. Epochs
    whose stage maps to None (``?``/Movement) or that run past the end of the
    signal are dropped. The returned list is in recording time order.
    """
    n_per_epoch = sample_rate * epoch_sec
    plans: list[EpochPlan] = []
    for onset, duration, desc in zip(onsets, durations, descriptions):
        stage = map_annotation(desc)
        if stage is None:
            continue
        n_epochs = int(round(duration / epoch_sec))
        for k in range(n_epochs):
            start = int(round((onset + k * epoch_sec) * sample_rate))
            if start < 0 or start + n_per_epoch > n_samples:
                continue
            plans.append(EpochPlan(start_sample=start, stage=stage))
    return plans


def trim_wake(
    plans: list[EpochPlan],
    epoch_sec: int,
    wake_pad_min: int,
) -> list[EpochPlan]:
    """Keep only Wake epochs within ``wake_pad_min`` of the sleep period.

    Sleep-EDF Cassette recordings include hours of Wake before/after sleep;
    keeping all of it would make Wake ~75% of the data. This keeps the sleep
    epochs plus a pad of Wake on each side and drops the rest. If there is no
    sleep at all, the plans are returned unchanged.
    """
    sleep_idx = [i for i, p in enumerate(plans) if p.stage != STAGE_W]
    if not sleep_idx:
        return plans
    pad = int(round(wake_pad_min * 60 / epoch_sec))
    lo = max(0, sleep_idx[0] - pad)
    hi = min(len(plans), sleep_idx[-1] + pad + 1)
    return plans[lo:hi]


def _epoch_features(signal: np.ndarray, params: EtlParams) -> dict[str, float]:
    """Compute the flat feature dict for one z-scored epoch signal."""
    bp = band_powers(signal, params.sample_rate, params.bands)
    se = spectral_entropy(signal, params.sample_rate)
    activity, mobility, complexity = hjorth(signal)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    feats = {f"feat_bp_{name}": float(bp[name]) for name in BAND_ORDER}
    feats["feat_spec_entropy"] = float(se)
    feats["feat_hjorth_activity"] = float(activity)
    feats["feat_hjorth_mobility"] = float(mobility)
    feats["feat_hjorth_complexity"] = float(complexity)
    feats["feat_rms"] = rms
    return feats


def _read_recording_signal(psg_path: str, params: EtlParams) -> np.ndarray:
    """Read the single EEG channel from an EDF as a 1-D µV array (import-local MNE)."""
    import mne  # imported inside the worker so drivers without MNE still import etl

    raw = mne.io.read_raw_edf(
        psg_path, preload=True, include=[params.channel], verbose="ERROR"
    )
    sfreq = int(round(raw.info["sfreq"]))
    if sfreq != params.sample_rate:
        raise ValueError(
            f"{psg_path}: channel {params.channel!r} is {sfreq} Hz, "
            f"expected {params.sample_rate} Hz"
        )
    return raw.get_data(picks=params.channel)[0] * 1e6  # Volts -> µV


def _read_annotations(hyp_path: str):
    import mne

    ann = mne.read_annotations(hyp_path)
    return ann.onset, ann.duration, ann.description


def process_recording(
    subject_id: str,
    night: int,
    psg_path: str,
    hyp_path: str,
    params: EtlParams,
) -> Iterator[tuple]:
    """Yield one output tuple per kept epoch for a single recording.

    Reads the EDF, z-scores the whole recording, plans+trims epochs from the
    hypnogram, and computes features. Tuple field order matches EPOCH_COLUMNS.
    """
    signal = _read_recording_signal(psg_path, params)
    # Per-recording z-score (data-model spec): removes inter-recording amplitude
    # differences while preserving relative amplitude across epochs.
    mu = float(signal.mean())
    sd = float(signal.std())
    z = (signal - mu) / (sd + _EPS)

    onsets, durations, descriptions = _read_annotations(hyp_path)
    plans = plan_epochs(
        onsets, durations, descriptions,
        params.sample_rate, params.epoch_sec, z.size,
    )
    plans = trim_wake(plans, params.epoch_sec, params.wake_pad_min)

    n_per_epoch = params.sample_rate * params.epoch_sec
    for epoch_idx, plan in enumerate(plans):
        seg = z[plan.start_sample: plan.start_sample + n_per_epoch]
        if seg.size != n_per_epoch:
            continue
        feats = _epoch_features(seg, params)
        yield (
            subject_id,
            int(night),
            epoch_idx,
            int(plan.stage),
            [float(v) for v in seg],
            feats["feat_bp_delta"],
            feats["feat_bp_theta"],
            feats["feat_bp_alpha"],
            feats["feat_bp_sigma"],
            feats["feat_bp_beta"],
            feats["feat_spec_entropy"],
            feats["feat_hjorth_activity"],
            feats["feat_hjorth_mobility"],
            feats["feat_hjorth_complexity"],
            feats["feat_rms"],
        )


def _epochs_schema():
    """Spark schema for epochs.parquet (order matches EPOCH_COLUMNS)."""
    from pyspark.sql.types import (
        ArrayType,
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    feat = lambda name: StructField(name, FloatType(), False)  # noqa: E731
    return StructType(
        [
            StructField("subject_id", StringType(), False),
            StructField("night", IntegerType(), False),
            StructField("epoch_idx", IntegerType(), False),
            StructField("stage", IntegerType(), False),
            StructField("signal", ArrayType(FloatType(), False), False),
            feat("feat_bp_delta"),
            feat("feat_bp_theta"),
            feat("feat_bp_alpha"),
            feat("feat_bp_sigma"),
            feat("feat_bp_beta"),
            feat("feat_spec_entropy"),
            feat("feat_hjorth_activity"),
            feat("feat_hjorth_mobility"),
            feat("feat_hjorth_complexity"),
            feat("feat_rms"),
        ]
    )


def _partition_worker(rows: Iterable, params: EtlParams) -> Iterator[tuple]:
    """mapPartitions body: process each manifest row in this partition."""
    for row in rows:
        yield from process_recording(
            subject_id=row["subject_id"],
            night=int(row["night"]),
            psg_path=row["psg_path"],
            hyp_path=row["hypnogram_path"],
            params=params,
        )


def run_etl(cfg: Config, master: Optional[str] = None) -> None:
    """Spark job: manifest -> epochs.parquet, partitioned by subject.

    Runs in local mode by default; the same code scales to a cluster by passing
    a different ``--master``. Logs per-stage epoch counts at the end so the N1
    class imbalance is visible.
    """
    import os
    import sys

    # Ensure Spark executors use this interpreter. Without this, PySpark on
    # Windows spawns a mismatched Python and workers die with a socket reset.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    manifest_path = cfg.paths.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run `ingest` first."
        )

    spark = (
        SparkSession.builder.master(master or cfg.spark.master)
        .appName(cfg.spark.app_name)
        .config("spark.sql.shuffle.partitions", cfg.spark.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        params = EtlParams.from_config(cfg)

        manifest_df = spark.read.option("header", True).csv(str(manifest_path))
        n_recordings = manifest_df.count()
        print(f"ETL: {n_recordings} recordings -> {cfg.paths.epochs_path}")

        # One recording per partition so process_recording runs in parallel.
        rows_rdd = manifest_df.repartition(max(n_recordings, 1)).rdd.mapPartitions(
            lambda part: _partition_worker(part, params)
        )
        epochs_df = spark.createDataFrame(rows_rdd, schema=_epochs_schema())

        out_path = str(cfg.paths.epochs_path)
        (
            epochs_df.write.mode("overwrite")
            .partitionBy("subject_id")
            .parquet(out_path)
        )

        # Log per-stage counts from the written store (cheap, no recompute).
        written = spark.read.parquet(out_path)
        total = written.count()
        counts = {
            r["stage"]: r["count"]
            for r in written.groupBy("stage").count().collect()
        }
        print(f"Wrote {total} epochs. Per-stage counts:")
        for sid in range(5):
            c = counts.get(sid, 0)
            pct = 100 * c / total if total else 0.0
            print(f"  {STAGE_NAMES[sid]:>3} (id {sid}): {c:6d}  ({pct:4.1f}%)")
    finally:
        spark.stop()
