"""Serve-time inference tests.

The centrepiece is the *invariant* test: an epoch produced by
``inference.epochs_from_edf`` must be bit-identical to the one
``etl.process_recording`` produces for the same 30 s window, because the CNN was
trained on the latter. Everything else here guards the pure helpers and the
input validation. Neither test touches Spark, TensorFlow, or a real EDF file --
the EDF read is monkeypatched so the tests are fast and hermetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_sleep_stager import etl, inference
from eeg_sleep_stager.config import load_config
from eeg_sleep_stager.etl import EtlParams

_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}


def _params(**over) -> EtlParams:
    base = dict(
        channel="EEG Fpz-Cz",
        sample_rate=100,
        epoch_sec=30,
        wake_pad_min=30,
        bands=_BANDS,
    )
    base.update(over)
    return EtlParams(**base)


# --------------------------------------------------------------------------- #
# The invariant: serve preprocessing == training preprocessing                #
# --------------------------------------------------------------------------- #
def test_epochs_from_edf_matches_etl(monkeypatch):
    """epochs_from_edf yields the same z-scored epochs as process_recording."""
    params = _params()
    n_per = params.sample_rate * params.epoch_sec  # 3000
    k = 5
    rng = np.random.default_rng(0)
    # A recording with non-zero mean and scale so the z-score is non-trivial.
    signal = (rng.standard_normal(k * n_per) * 12.5 + 3.0).astype(np.float64)

    # Both code paths read "the EDF" through these helpers; feed both the same
    # synthetic signal and a hypnogram of k consecutive N2 epochs from t=0.
    monkeypatch.setattr(inference, "_read_edf_signal", lambda p, pr: signal)
    monkeypatch.setattr(etl, "_read_recording_signal", lambda p, pr: signal)
    monkeypatch.setattr(
        etl,
        "_read_annotations",
        lambda p: ([0.0], [float(k * params.epoch_sec)], ["Sleep stage 2"]),
    )

    serve = inference.epochs_from_edf("dummy-psg.edf", params)
    assert serve.shape == (k, n_per)
    assert serve.dtype == np.float32

    etl_rows = list(
        etl.process_recording("SUB", 1, "dummy-psg.edf", "dummy-hyp.edf", params)
    )
    assert len(etl_rows) == k
    for i, row in enumerate(etl_rows):
        etl_seg = np.asarray(row[4], dtype=np.float32)  # index 4 == signal list
        # Bit-for-bit on the shared 30 s grid, at the float32 the store keeps.
        assert np.array_equal(serve[i], etl_seg)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def test_zscore_matches_formula():
    sig = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    z = inference._zscore(sig)
    expected = (sig - sig.mean()) / (sig.std() + inference._EPS)
    assert np.allclose(z, expected)
    assert abs(float(z.mean())) < 1e-9


def test_cut_epochs_drops_trailing_partial():
    z = np.arange(3000 * 3 + 500, dtype=float)
    epochs = inference._cut_epochs(z, sample_rate=100, epoch_sec=30)
    assert epochs.shape == (3, 3000)
    assert epochs.dtype == np.float32
    # First epoch is the first 3000 samples, in order.
    assert np.array_equal(epochs[0], z[:3000].astype(np.float32))


def test_cut_epochs_rejects_too_short():
    with pytest.raises(ValueError, match="fewer than one"):
        inference._cut_epochs(np.zeros(2999), sample_rate=100, epoch_sec=30)


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #
class _FakeRaw:
    def __init__(self, ch_names, sfreq, data):
        self.ch_names = list(ch_names)
        self.info = {"sfreq": sfreq}
        self._data = np.asarray(data, dtype=float)

    def get_data(self, picks=None):
        return self._data[np.newaxis, :]


def _patch_mne(monkeypatch, raw):
    import mne

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *a, **k: raw)


def test_read_edf_signal_rejects_missing_channel(monkeypatch, tmp_path):
    edf = tmp_path / "rec.edf"
    edf.write_bytes(b"not really an edf")  # only existence + mne are checked
    _patch_mne(monkeypatch, _FakeRaw(["EEG Pz-Oz"], 100, np.zeros(3000)))
    with pytest.raises(ValueError, match="not found"):
        inference._read_edf_signal(edf, _params())


def test_epochs_from_edf_rejects_bad_rate(monkeypatch, tmp_path):
    edf = tmp_path / "rec.edf"
    edf.write_bytes(b"not really an edf")
    # Channel present but sampled at 200 Hz: etl._read_recording_signal must object.
    _patch_mne(monkeypatch, _FakeRaw(["EEG Fpz-Cz"], 200, np.zeros(6000)))
    with pytest.raises(ValueError, match="Hz"):
        inference.epochs_from_edf(edf, _params())


def test_epochs_from_edf_missing_file():
    with pytest.raises(FileNotFoundError):
        inference.epochs_from_edf("does-not-exist.edf", _params())


# --------------------------------------------------------------------------- #
# Prediction & summary (S3)                                                   #
# --------------------------------------------------------------------------- #
def test_stage_summary_math():
    # W W N2 N2 N2 REM  @ 30 s/epoch  -> 0.5 min per epoch.
    y = np.array([0, 0, 2, 2, 2, 4])
    s = inference.stage_summary(y, epoch_sec=30)

    assert s["n_epochs"] == 6
    assert s["per_stage"]["W"]["epochs"] == 2
    assert s["per_stage"]["N2"]["epochs"] == 3
    assert s["per_stage"]["REM"]["epochs"] == 1
    assert s["per_stage"]["N1"]["epochs"] == 0
    assert s["per_stage"]["N2"]["minutes"] == pytest.approx(1.5)
    assert s["per_stage"]["N2"]["pct"] == pytest.approx(50.0)
    # 4 non-Wake epochs -> 2.0 min TST; TIB = 3.0 min; onset at epoch 2 -> 1.0 min.
    assert s["total_sleep_min"] == pytest.approx(2.0)
    assert s["time_in_bed_min"] == pytest.approx(3.0)
    assert s["sleep_efficiency_pct"] == pytest.approx(200.0 / 3.0)
    assert s["sleep_onset_min"] == pytest.approx(1.0)


def test_stage_summary_all_wake_has_no_onset():
    s = inference.stage_summary(np.zeros(4, dtype=int), epoch_sec=30)
    assert s["sleep_onset_min"] is None
    assert s["total_sleep_min"] == 0.0


def _models_dir():
    return load_config().paths.models_dir


@pytest.mark.skipif(
    not (_models_dir() / "cnn.onnx").exists(), reason="cnn.onnx not built"
)
def test_predict_stages_cnn_smoke():
    # The CNN branch now serves via onnxruntime on cnn.onnx (no TensorFlow), so
    # this skips on the ONNX artifact rather than the Keras file. Same contract:
    # (N,) int labels in 0..4 and (N,5) softmax rows summing to 1.
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 3000)).astype(np.float32)
    y_pred, proba = inference.predict_stages(z, _models_dir(), model="cnn")
    assert y_pred.shape == (4,)
    assert proba.shape == (4, 5)
    assert set(np.unique(y_pred)).issubset(set(range(5)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)  # softmax rows


@pytest.mark.skipif(
    not (
        (_models_dir() / "cnn.keras").exists()
        and (_models_dir() / "cnn.onnx").exists()
    ),
    reason="need both cnn.keras and cnn.onnx for the parity check",
)
def test_cnn_onnx_matches_keras():
    """The ONNX serve path reproduces the Keras forward pass.

    This is the equivalence guard for the TF->ONNX slim-down: on the same
    epochs, the onnxruntime backend (via ``predict_stages``) must give the same
    argmax labels and near-identical softmax probabilities as the original Keras
    model. If this passes, swapping backends did not change results.
    """
    rng = np.random.default_rng(7)
    z = rng.standard_normal((16, 3000)).astype(np.float32)
    models_dir = _models_dir()

    # ONNX path (what the serve code runs).
    y_onnx, proba_onnx = inference.predict_stages(z, models_dir, model="cnn")

    # Keras path, direct, via the retained TF loader.
    net = inference._load_cnn(models_dir)
    proba_keras = np.asarray(net.predict(z[..., None], verbose=0), dtype=float)
    y_keras = np.argmax(proba_keras, axis=1).astype(int)

    assert np.array_equal(y_onnx, y_keras)  # identical predicted stages
    assert np.allclose(proba_onnx, proba_keras, atol=1e-4)  # near-identical softmax


@pytest.mark.skipif(
    not (_models_dir() / "baseline.joblib").exists(),
    reason="baseline.joblib not trained",
)
def test_predict_stages_baseline_smoke():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((4, 3000)).astype(np.float32)
    y_pred, proba = inference.predict_stages(
        z, _models_dir(), model="baseline", params=_params()
    )
    assert y_pred.shape == (4,)
    assert proba.shape == (4, 5)
    assert set(np.unique(y_pred)).issubset(set(range(5)))


def test_predict_stages_baseline_requires_params():
    with pytest.raises(ValueError, match="EtlParams"):
        inference.predict_stages(np.zeros((2, 3000), np.float32), ".", model="baseline")


def test_predict_stages_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        inference.predict_stages(np.zeros((2, 3000), np.float32), ".", model="nope")


def test_stage_epochs_metrics_gating(monkeypatch):
    """metrics appear only when scored expert labels are present."""
    cfg = load_config()
    n = 6
    canned = (np.array([0, 1, 2, 3, 4, 2]), np.zeros((n, 5), dtype=float))
    monkeypatch.setattr(inference, "predict_stages", lambda *a, **k: canned)

    z = np.zeros((n, 3000), dtype=np.float32)

    # No expert -> no metrics.
    res = inference.stage_epochs(z, cfg, expert=None, model="cnn")
    assert res.metrics is None
    assert res.has_expert is False
    assert res.n_epochs == n
    assert res.summary["n_epochs"] == n

    # Expert with one unscored epoch -> metrics over the 5 scored epochs only.
    expert = np.array([0, 1, 2, 3, 4, inference.UNSCORED])
    res = inference.stage_epochs(z, cfg, expert=expert, model="cnn")
    assert res.metrics is not None
    assert res.metrics["n_scored_epochs"] == 5
    assert res.metrics["accuracy"] == pytest.approx(1.0)  # canned preds match


# --------------------------------------------------------------------------- #
# Expert labels (S4)                                                          #
# --------------------------------------------------------------------------- #
def test_expert_labels_alignment(monkeypatch, tmp_path):
    """Annotations map onto the t=0 epoch grid; ? spans and gaps stay unscored."""
    hyp = tmp_path / "hyp.edf"
    hyp.write_bytes(b"stub")  # only existence is checked; _read_annotations is patched

    # onset / duration (s) / description, at 30 s epochs:
    #   W  @0  60s -> epochs 0,1
    #   N1 @60 30s -> epoch  2
    #   ?  @90 60s -> epochs 3,4 (dropped)
    #   R  @150 30s -> epoch 5
    #   epochs 6,7 have no annotation -> unscored
    ann = (
        [0.0, 60.0, 90.0, 150.0],
        [60.0, 30.0, 60.0, 30.0],
        ["Sleep stage W", "Sleep stage 1", "Sleep stage ?", "Sleep stage R"],
    )
    monkeypatch.setattr(etl, "_read_annotations", lambda p: ann)

    labels = inference.expert_labels_per_epoch(hyp, _params(), n_epochs=8)
    U = inference.UNSCORED
    assert list(labels) == [0, 0, 1, U, U, 4, U, U]


def test_expert_labels_clips_past_end(monkeypatch, tmp_path):
    hyp = tmp_path / "hyp.edf"
    hyp.write_bytes(b"stub")
    # One long N2 annotation covering more epochs than the signal has.
    monkeypatch.setattr(
        etl, "_read_annotations", lambda p: ([0.0], [30.0 * 10], ["Sleep stage 2"])
    )
    labels = inference.expert_labels_per_epoch(hyp, _params(), n_epochs=3)
    assert list(labels) == [2, 2, 2]  # clipped to n_epochs, no index error


def test_expert_labels_missing_file():
    with pytest.raises(FileNotFoundError):
        inference.expert_labels_per_epoch("nope-hyp.edf", _params(), n_epochs=4)


def test_stage_epochs_rejects_length_mismatch(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(
        inference, "predict_stages", lambda *a, **k: (np.zeros(6, int), np.zeros((6, 5)))
    )
    with pytest.raises(ValueError, match="length"):
        inference.stage_epochs(
            np.zeros((6, 3000), np.float32), cfg, expert=np.zeros(5, int)
        )


# --------------------------------------------------------------------------- #
# Bundled sample (S5)                                                         #
# --------------------------------------------------------------------------- #
def test_load_sample_roundtrip(tmp_path):
    z = np.random.default_rng(3).standard_normal((10, 3000)).astype(np.float32)
    stage = np.array([0, 1, 2, 3, 4, 2, 2, 0, 4, 3], dtype=np.int8)
    npz = tmp_path / "s.npz"
    np.savez_compressed(npz, signal=z, stage=stage)

    got_z, got_expert = inference.load_sample(npz)
    assert got_z.shape == (10, 3000)
    assert got_z.dtype == np.float32
    assert np.array_equal(got_expert, stage.astype(int))


def test_load_sample_missing():
    with pytest.raises(FileNotFoundError, match="make_sample"):
        inference.load_sample("no-such-sample.npz")


_SAMPLE = Path("app_assets/sample_night.npz")


@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample not built")
def test_bundled_sample_shape():
    z, expert = inference.load_sample(_SAMPLE)
    assert z.ndim == 2 and z.shape[1] == 3000
    assert expert is not None and expert.shape[0] == z.shape[0]
    assert set(np.unique(expert)).issubset(set(range(5)))  # all scored, valid ids
