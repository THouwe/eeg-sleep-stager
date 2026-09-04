"""Serve-time inference: EDF (or bundled sample) -> per-epoch sleep stages.

This is the model-serving counterpart to the Spark ETL. It deliberately shares
**no** Spark code: at inference time there is no cluster, no Parquet store and no
manifest -- just one recording, read with MNE, preprocessed exactly as in
training, and classified with the saved Keras/​sklearn model. The web demo
(`app.py`) is a thin UI over the functions here.

The single invariant that keeps predictions valid is the preprocessing: an epoch
handed to the CNN must be produced the *same way* as in
:func:`eeg_sleep_stager.etl.process_recording` -- single channel ``EEG Fpz-Cz`` at
100 Hz, Volts -> µV, then a **per-recording z-score over the whole night**, then
cut into consecutive non-overlapping 30 s (3000-sample) epochs. :func:`epochs_from_edf`
owns that and is pinned by a test asserting it matches the ETL path (S2).

Heavy/optional dependencies (``mne``, ``tensorflow``, ``joblib``) are imported
inside the functions that need them, so importing this module stays cheap and
Spark-free -- ``import eeg_sleep_stager.inference`` pulls in only NumPy and the
pure feature/label helpers.

Status: S1 scaffold -- signatures and the :class:`InferenceResult` contract are
fixed; bodies are filled in over S2–S4 and raise ``NotImplementedError`` until
then.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import etl
from .config import N_CLASSES, Config
from .etl import EtlParams
from .labels import STAGE_NAMES, STAGE_W, map_annotation

PathLike = Union[str, Path]

# Reuse the exact epsilon the ETL uses in its per-recording z-score, so serve-time
# normalization is bit-identical to training rather than merely similar.
_EPS = etl._EPS

# Model identifiers accepted throughout the serve path.
MODEL_CNN = "cnn"
MODEL_BASELINE = "baseline"

# Sentinel used in per-epoch expert-label arrays for epochs the hypnogram does
# not score ("Sleep stage ?", "Movement time", or gaps).
UNSCORED = -1

# Process-level model cache so a long-lived server (the web demo) loads each model
# once instead of on every request. Keyed by (model, resolved models_dir).
_MODEL_CACHE: dict = {}


def _load_cnn(models_dir: Path):
    # [onnx] retained (unused on the serve path): the TF/Keras loader is kept
    # recoverable per the slim-down rules. Serving now uses _load_cnn_onnx below.
    key = ("cnn", str(models_dir.resolve()))
    if key not in _MODEL_CACHE:
        import tensorflow as tf

        _MODEL_CACHE[key] = tf.keras.models.load_model(models_dir / "cnn.keras")
    return _MODEL_CACHE[key]


def _load_cnn_onnx(models_dir: Path):
    """Return a cached onnxruntime session for ``models/cnn.onnx``.

    Mirrors :func:`_load_cnn`/:func:`_load_baseline`: one load per
    ``(model, models_dir)`` for the life of the process. ``onnxruntime`` is
    imported lazily so ``import eeg_sleep_stager.inference`` stays cheap and
    TF-free. The ONNX graph is exported by ``scripts/keras_to_onnx.py`` with a
    dynamic batch axis; read its input/output names from the session at call
    time (don't hard-code them).
    """
    key = ("cnn_onnx", str(models_dir.resolve()))
    if key not in _MODEL_CACHE:
        import onnxruntime as ort

        _MODEL_CACHE[key] = ort.InferenceSession(
            str(models_dir / "cnn.onnx"), providers=["CPUExecutionProvider"]
        )
    return _MODEL_CACHE[key]


def _load_baseline(models_dir: Path):
    key = ("baseline", str(models_dir.resolve()))
    if key not in _MODEL_CACHE:
        import joblib

        _MODEL_CACHE[key] = joblib.load(models_dir / "baseline.joblib")
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Drop cached models (e.g. after retraining in a live process)."""
    _MODEL_CACHE.clear()


@dataclass
class InferenceResult:
    """Everything the UI needs to render one staged recording.

    Field semantics are frozen here so ``app.py`` and any future client read a
    stable contract; this docstring is the source of truth for that contract.

    Attributes:
        y_pred: predicted stage id per epoch, shape ``(N,)``, values ``0..4``.
        proba: class probabilities per epoch, shape ``(N, 5)`` -- softmax for the
            CNN, ``predict_proba`` for the baseline.
        expert: per-epoch expert label, shape ``(N,)``, with :data:`UNSCORED`
            (``-1``) for unscored epochs; ``None`` when no hypnogram was supplied.
        summary: output of :func:`stage_summary` (per-stage counts/%/minutes plus
            total-sleep-time and efficiency).
        metrics: output of :func:`eeg_sleep_stager.evaluate.compute_metrics` over
            the scored epochs, or ``None`` when there is no expert scoring (no
            labels ⇒ no fabricated numbers).
        epoch_sec: epoch length in seconds (30).
        model: which model produced ``y_pred`` (:data:`MODEL_CNN` /
            :data:`MODEL_BASELINE`).
        source: short human label for the input (e.g. an uploaded filename or the
            bundled sample's ``subject_id night N``), for display only.
    """

    y_pred: np.ndarray
    proba: np.ndarray
    expert: Optional[np.ndarray]
    summary: dict
    metrics: Optional[dict]
    epoch_sec: int
    model: str
    source: str = ""

    @property
    def n_epochs(self) -> int:
        return int(len(self.y_pred))

    @property
    def has_expert(self) -> bool:
        return self.expert is not None


# --------------------------------------------------------------------------- #
# Preprocessing (S2)                                                           #
# --------------------------------------------------------------------------- #
def _zscore(signal: np.ndarray) -> np.ndarray:
    """Per-recording z-score, identical to :func:`etl.process_recording`.

    ``(x - mean) / (std + eps)`` over the whole signal. The shared ``_EPS`` and
    float64 accumulation keep this bit-for-bit equal to the training path.
    """
    sig = np.asarray(signal, dtype=float)
    mu = float(sig.mean())
    sd = float(sig.std())
    return (sig - mu) / (sd + _EPS)


def _cut_epochs(z: np.ndarray, sample_rate: int, epoch_sec: int) -> np.ndarray:
    """Cut a z-scored signal into consecutive non-overlapping epochs.

    Returns ``(N, n_per_epoch)`` float32 with ``N = len // n_per_epoch``; any
    trailing partial epoch is dropped. Raises if the signal is shorter than one
    epoch.
    """
    n_per = sample_rate * epoch_sec
    n_epochs = int(z.size // n_per)
    if n_epochs == 0:
        raise ValueError(
            f"recording has {z.size} samples, fewer than one {epoch_sec}s epoch "
            f"({n_per} samples at {sample_rate} Hz)"
        )
    return z[: n_epochs * n_per].reshape(n_epochs, n_per).astype(np.float32)


def _read_edf_signal(psg_path: PathLike, params: EtlParams) -> np.ndarray:
    """Read the single EEG channel as a 1-D µV array, with actionable errors.

    Validates the channel is present (listing what *is* available on mismatch)
    before delegating to :func:`etl._read_recording_signal`, which enforces the
    sample rate and does the Volts→µV scaling.
    """
    import mne

    path = Path(psg_path)
    if not path.exists():
        raise FileNotFoundError(f"PSG EDF not found: {path}")
    try:
        header = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    except Exception as exc:  # unreadable / not an EDF
        raise ValueError(f"could not read EDF {path.name}: {exc}") from exc
    if params.channel not in header.ch_names:
        raise ValueError(
            f"channel {params.channel!r} not found in {path.name}; available "
            f"channels: {header.ch_names}. This demo expects a Sleep-EDF Cassette "
            f"recording with the {params.channel!r} channel."
        )
    return etl._read_recording_signal(str(path), params)


def epochs_from_edf(psg_path: PathLike, params: EtlParams) -> np.ndarray:
    """Read a PSG EDF and return z-scored 30 s epochs, shape ``(N, 3000)``.

    Mirrors :func:`eeg_sleep_stager.etl.process_recording`'s preprocessing:
    picks ``params.channel``, asserts ``params.sample_rate``, scales to µV,
    z-scores over the whole recording, then cuts consecutive non-overlapping
    epochs of ``params.epoch_sec`` seconds from t=0 (no wake trimming). Raises a
    clear error on the wrong channel/sample rate or a too-short recording.
    """
    signal = _read_edf_signal(psg_path, params)
    z = _zscore(signal)
    return _cut_epochs(z, params.sample_rate, params.epoch_sec)


def expert_labels_per_epoch(
    hyp_path: PathLike, params: EtlParams, n_epochs: int
) -> np.ndarray:
    """Map a hypnogram EDF to a per-epoch label array aligned to the t=0 grid.

    Returns an ``int`` array of length ``n_epochs`` with AASM class ids, using
    :data:`UNSCORED` (``-1``) for epochs the hypnogram leaves unscored
    (``?``/Movement/gaps). Alignment matches :func:`epochs_from_edf`'s grid so a
    prediction and its expert label refer to the same 30 s window.

    Epoch index is ``round(onset / epoch_sec) + k`` for the ``k``-th epoch of an
    annotation -- the same mapping :func:`etl.plan_epochs` uses, so serve-time
    labels line up with the training grid. Annotations that fall outside
    ``[0, n_epochs)`` are clipped.
    """
    path = Path(hyp_path)
    if not path.exists():
        raise FileNotFoundError(f"Hypnogram EDF not found: {path}")

    onsets, durations, descriptions = etl._read_annotations(str(path))
    labels = np.full(int(n_epochs), UNSCORED, dtype=int)
    for onset, duration, desc in zip(onsets, durations, descriptions):
        stage = map_annotation(desc)
        if stage is None:  # "?", "Movement time", or unrecognized -> leave unscored
            continue
        start_ep = int(round(float(onset) / params.epoch_sec))
        n_ep = int(round(float(duration) / params.epoch_sec))
        for k in range(n_ep):
            idx = start_ep + k
            if 0 <= idx < n_epochs:
                labels[idx] = stage
    return labels


# --------------------------------------------------------------------------- #
# Prediction & summary (S3)                                                    #
# --------------------------------------------------------------------------- #
def predict_stages(
    z_epochs: np.ndarray,
    models_dir: PathLike,
    model: str = MODEL_CNN,
    params: Optional[EtlParams] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a trained model over z-scored epochs.

    Args:
        z_epochs: ``(N, epoch_samples)`` float array from :func:`epochs_from_edf`
            or the bundled sample.
        models_dir: directory holding ``cnn.keras`` / ``baseline.joblib``.
        model: :data:`MODEL_CNN` (1-D CNN on the raw epoch) or
            :data:`MODEL_BASELINE` (GBM on the 10 engineered features).
        params: required for :data:`MODEL_BASELINE` -- supplies the bands and
            sample rate used to recompute the engineered features, matching the
            columns the baseline was trained on.

    Returns:
        ``(y_pred, proba)`` with shapes ``(N,)`` int and ``(N, N_CLASSES)`` float.
        ``proba`` is the CNN softmax, or the baseline's ``predict_proba`` expanded
        to the full class axis (zeros for any class unseen in training).
    """
    z = np.asarray(z_epochs, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"z_epochs must be 2-D (N, epoch_samples); got {z.shape}")
    models_dir = Path(models_dir)

    if model == MODEL_CNN:
        # [onnx] disabled: serve path no longer imports TensorFlow; the Keras
        # forward pass is replaced by an onnxruntime session on cnn.onnx (same
        # math, a fraction of the size/RAM). Un-comment these two lines and drop
        # the ONNX block below to restore the TF path (see cnn-to-onnx.md).
        # net = _load_cnn(models_dir)
        # proba = np.asarray(net.predict(z[..., np.newaxis], verbose=0), dtype=float)
        sess = _load_cnn_onnx(models_dir)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        x = z[..., np.newaxis].astype("float32")
        proba = np.asarray(sess.run([out_name], {in_name: x})[0], dtype=float)
        return np.argmax(proba, axis=1).astype(int), proba

    if model == MODEL_BASELINE:
        if params is None:
            raise ValueError(
                "baseline prediction needs EtlParams (bands/sample_rate) to "
                "recompute engineered features"
            )
        from .datasets import FEATURE_COLUMNS

        clf = _load_baseline(models_dir)
        feats = np.array(
            [[etl._epoch_features(seg, params)[c] for c in FEATURE_COLUMNS] for seg in z],
            dtype=np.float32,
        )
        y_pred = clf.predict(feats).astype(int)
        proba = np.zeros((z.shape[0], N_CLASSES), dtype=float)
        proba[:, clf.classes_.astype(int)] = clf.predict_proba(feats)
        return y_pred, proba

    raise ValueError(
        f"unknown model {model!r}; expected {MODEL_CNN!r} or {MODEL_BASELINE!r}"
    )


def stage_summary(y_pred: np.ndarray, epoch_sec: int = 30) -> dict:
    """Per-stage counts/%/minutes plus total-sleep-time and sleep efficiency.

    Pure function of the predicted label sequence. "Sleep" is every non-Wake
    epoch; time-in-bed is the whole recording, so efficiency is TST / TIB. Sleep
    onset is the first non-Wake epoch (``None`` if the night is all Wake).
    """
    y = np.asarray(y_pred).astype(int)
    n = int(y.size)
    min_per_epoch = epoch_sec / 60.0

    per_stage = {}
    for sid, name in enumerate(STAGE_NAMES):
        c = int(np.count_nonzero(y == sid))
        per_stage[name] = {
            "epochs": c,
            "pct": (100.0 * c / n) if n else 0.0,
            "minutes": c * min_per_epoch,
        }

    sleep_epochs = int(np.count_nonzero(y != STAGE_W))
    sleep_idx = np.flatnonzero(y != STAGE_W)
    total_sleep_min = sleep_epochs * min_per_epoch
    time_in_bed_min = n * min_per_epoch

    return {
        "n_epochs": n,
        "epoch_sec": epoch_sec,
        "per_stage": per_stage,
        "total_sleep_min": total_sleep_min,
        "time_in_bed_min": time_in_bed_min,
        "sleep_efficiency_pct": (
            100.0 * total_sleep_min / time_in_bed_min if time_in_bed_min else 0.0
        ),
        "sleep_onset_min": (
            float(sleep_idx[0]) * min_per_epoch if sleep_idx.size else None
        ),
    }


# --------------------------------------------------------------------------- #
# Orchestration (used by app.py)                                              #
# --------------------------------------------------------------------------- #
def stage_epochs(
    z_epochs: np.ndarray,
    cfg: Config,
    expert: Optional[np.ndarray] = None,
    model: str = MODEL_CNN,
    source: str = "",
) -> InferenceResult:
    """Core path shared by EDF uploads and the bundled sample.

    Given already-z-scored epochs (and optionally per-epoch expert labels), run
    the model, build the summary, compute metrics when labels exist, and assemble
    an :class:`InferenceResult`. Metrics are computed only over *scored* epochs
    (``expert != UNSCORED``); with no expert labels ``metrics`` stays ``None`` so
    the UI never shows a fabricated number.
    """
    params = EtlParams.from_config(cfg)
    y_pred, proba = predict_stages(
        z_epochs, cfg.paths.models_dir, model=model, params=params
    )

    metrics: Optional[dict] = None
    if expert is not None:
        expert = np.asarray(expert).astype(int)
        if expert.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"expert labels length {expert.shape[0]} != {y_pred.shape[0]} epochs"
            )
        scored = expert != UNSCORED
        if scored.any():
            from .evaluate import compute_metrics

            metrics = compute_metrics(expert[scored], y_pred[scored])
            metrics["model"] = model
            metrics["n_scored_epochs"] = int(scored.sum())

    return InferenceResult(
        y_pred=y_pred,
        proba=proba,
        expert=expert if expert is not None else None,
        summary=stage_summary(y_pred, cfg.dataset.epoch_sec),
        metrics=metrics,
        epoch_sec=cfg.dataset.epoch_sec,
        model=model,
        source=source,
    )


def stage_edf(
    cfg: Config,
    psg_path: PathLike,
    hyp_path: Optional[PathLike] = None,
    model: str = MODEL_CNN,
) -> InferenceResult:
    """Full upload path: EDF (+ optional hypnogram) -> :class:`InferenceResult`.

    Composes :func:`epochs_from_edf`, optional :func:`expert_labels_per_epoch`,
    and :func:`stage_epochs`. Without a hypnogram it returns predictions and the
    summary; with one it also carries the expert overlay and metrics.
    """
    params = EtlParams.from_config(cfg)
    z = epochs_from_edf(psg_path, params)
    expert = (
        expert_labels_per_epoch(hyp_path, params, n_epochs=z.shape[0])
        if hyp_path is not None
        else None
    )
    return stage_epochs(
        z, cfg, expert=expert, model=model, source=Path(psg_path).name
    )


def load_sample(npz_path: PathLike) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load the bundled sample night, returning ``(z_epochs, expert)``.

    The sample ships already z-scored (extracted from the Parquet store), so it
    feeds straight into :func:`stage_epochs` -- the demo button and the upload
    button share one code path. Expert labels are always present for the sample
    (the store drops unscored epochs), so metrics are shown for it.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sample not found: {path}. Build it with `python scripts/make_sample.py`."
        )
    with np.load(path) as data:
        z = data["signal"].astype(np.float32)
        expert = data["stage"].astype(int) if "stage" in data else None
    return z, expert


def stage_sample(
    cfg: Config, npz_path: PathLike, model: str = MODEL_CNN
) -> InferenceResult:
    """Stage the bundled sample night -> :class:`InferenceResult`.

    Reads the optional ``sample_meta.json`` sidecar for a human-readable source
    label, then runs the same :func:`stage_epochs` path an upload uses.
    """
    import json

    z, expert = load_sample(npz_path)
    source = "bundled sample"
    meta_path = Path(npz_path).with_name("sample_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source = f"sample: {meta.get('subject_id', '?')} night {meta.get('night', '?')}"
    return stage_epochs(z, cfg, expert=expert, model=model, source=source)
