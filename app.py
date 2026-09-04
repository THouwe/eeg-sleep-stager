"""Gradio web demo for the EEG sleep stager (Hugging Face Spaces entrypoint).

Upload a full-night PSG EDF -- or click "Load sample night" -- and get the trained
1-D CNN's sleep staging back as a hypnogram, a stage summary, and a downloadable
per-epoch table; supply the matching hypnogram EDF for an expert-vs-predicted
overlay and honest metrics. All model logic lives in
:mod:`eeg_sleep_stager.inference`; this file is only UI wiring, kept deliberately
thin (and untested) so the tested code carries the weight.

No Spark runs here -- inference is plain MNE/NumPy/TensorFlow. Run locally with::

    pip install -e ".[app]"
    python app.py

On Hugging Face Spaces the package is not pip-installed; the ``src`` shim below
puts it on the path so ``import eeg_sleep_stager`` resolves from a bare checkout.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Import the package straight from the checkout (see module docstring).
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eeg_sleep_stager import inference  # noqa: E402
from eeg_sleep_stager.config import load_config  # noqa: E402
from eeg_sleep_stager.labels import STAGE_NAMES  # noqa: E402

CFG = load_config(ROOT / "config.yaml")
SAMPLE_NPZ = ROOT / "app_assets" / "sample_night.npz"
MAX_UPLOAD_MB = 200  # Sleep-EDF Cassette PSGs are ~50 MB; guard against surprises.

# Hypnogram vertical order (top -> bottom): the conventional clinical layout.
_HYPNO_ORDER = [0, 4, 1, 2, 3]  # W, REM, N1, N2, N3

MODEL_CHOICES = [("1-D CNN (deep)", "cnn"), ("GBM baseline (features)", "baseline")]

DISCLAIMER = """\
### About this demo
This is a **research demonstration**, not a medical device, and produces no
clinical or diagnostic output. It automatically scores 30-second epochs of a
single EEG channel (**Fpz-Cz, 100 Hz**) into the five AASM stages using a small
1-D CNN trained on the open [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
corpus.

- **Input:** a full-night Sleep-EDF *Cassette* PSG EDF. Other montages or sample
  rates are rejected with a clear message rather than silently mis-scored.
- **Honest numbers:** metrics only appear when you also upload the expert
  hypnogram. The bundled sample is a **held-out test subject** the model never
  saw in training, so its numbers are an honest estimate.
- **Why a whole night:** the signal is z-scored per recording (as in training),
  so the demo needs the full recording, not an isolated epoch.
- **Note on raw uploads:** Sleep-EDF Cassette recordings span ~20 h including the
  waking day, so a raw upload often shows long Wake stretches. The bundled sample
  has its Wake padding trimmed (as the training ETL does).
"""


# --------------------------------------------------------------------------- #
# Rendering helpers (matplotlib Figures for gr.Plot; pure of Gradio)          #
# --------------------------------------------------------------------------- #
def _new_fig(figsize):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def _hypnogram_fig(res: "inference.InferenceResult"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ypos = {s: len(_HYPNO_ORDER) - 1 - row for row, s in enumerate(_HYPNO_ORDER)}
    hours = np.arange(res.n_epochs) * res.epoch_sec / 3600.0
    total_h = res.n_epochs * res.epoch_sec / 3600.0

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(11, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )

    # --- hypnogram (predicted, with expert overlay when available) ---
    if res.expert is not None:
        # NaN at unscored epochs leaves gaps rather than jumping to a fake stage.
        ev = np.array([ypos.get(int(s), np.nan) for s in res.expert], dtype=float)
        ax.step(hours, ev, where="post", color="black", lw=1.4, label="expert", alpha=0.8)
    pv = [ypos[int(s)] for s in res.y_pred]
    ax.step(hours, pv, where="post", color="tab:red", lw=1.1, label="predicted", alpha=0.85)
    ax.set_yticks(range(len(_HYPNO_ORDER)),
                  labels=[STAGE_NAMES[s] for s in reversed(_HYPNO_ORDER)])
    ax.set_title(f"Hypnogram — {res.source} ({res.model})")
    ax.legend(loc="upper right")
    ax.margins(x=0.01)

    # --- probability strip: per-epoch P(stage), clinical row order (W at top) ---
    disp = res.proba[:, _HYPNO_ORDER].T  # (5, N): rows W, REM, N1, N2, N3
    im = axp.imshow(
        disp, aspect="auto", origin="upper", cmap="magma", vmin=0, vmax=1,
        extent=[0, total_h, len(_HYPNO_ORDER), 0], interpolation="nearest",
    )
    axp.set_yticks([i + 0.5 for i in range(len(_HYPNO_ORDER))],
                   labels=[STAGE_NAMES[s] for s in _HYPNO_ORDER])
    axp.set_ylabel("P(stage)")
    axp.set_xlabel("Time (hours)")
    fig.colorbar(im, ax=axp, fraction=0.03, pad=0.01)
    fig.tight_layout()
    return fig


def _confusion_fig(res: "inference.InferenceResult"):
    if res.metrics is None:
        return None
    cm = np.asarray(res.metrics["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = _new_fig((5.0, 4.4))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_NAMES)), labels=STAGE_NAMES)
    ax.set_yticks(range(len(STAGE_NAMES)), labels=STAGE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expert")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{disp[i, j]:.2f}", ha="center", va="center",
                    color="white" if disp[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _summary_frame(res: "inference.InferenceResult"):
    import pandas as pd

    rows = [
        {
            "Stage": name,
            "Epochs": d["epochs"],
            "%": round(d["pct"], 1),
            "Minutes": round(d["minutes"], 1),
        }
        for name, d in res.summary["per_stage"].items()
    ]
    return pd.DataFrame(rows)


def _summary_markdown(res: "inference.InferenceResult") -> str:
    s = res.summary
    rec_h = s["time_in_bed_min"] / 60.0
    onset = s["sleep_onset_min"]
    onset_txt = f"{onset:.0f} min" if onset is not None else "—"
    return (
        f"**Recording:** {res.n_epochs} epochs · {rec_h:.1f} h &nbsp;|&nbsp; "
        f"**Total sleep:** {s['total_sleep_min']:.0f} min &nbsp;|&nbsp; "
        f"**Efficiency:** {s['sleep_efficiency_pct']:.0f}% &nbsp;|&nbsp; "
        f"**Sleep onset:** {onset_txt}"
    )


def _metrics_markdown(res: "inference.InferenceResult") -> str:
    m = res.metrics
    if m is None:
        return (
            "_No expert hypnogram supplied — upload the matching "
            "`*-Hypnogram.edf` to see accuracy, macro-F1 and Cohen's κ versus "
            "expert scoring._"
        )
    per = " · ".join(f"{n} {v['f1']:.2f}" for n, v in m["per_stage"].items())
    return (
        f"**Accuracy** {m['accuracy']:.1%} &nbsp;|&nbsp; "
        f"**macro-F1** {m['macro_f1']:.3f} &nbsp;|&nbsp; "
        f"**Cohen's κ** {m['cohen_kappa']:.3f}  \n"
        f"over {m['n_scored_epochs']} expert-scored epochs. "
        f"Per-stage F1: {per}."
    )


def _write_csv(res: "inference.InferenceResult") -> str:
    import pandas as pd

    data = {
        "epoch_idx": np.arange(res.n_epochs),
        "start_sec": np.arange(res.n_epochs) * res.epoch_sec,
        "pred_stage": [STAGE_NAMES[int(s)] for s in res.y_pred],
    }
    if res.expert is not None:
        data["expert_stage"] = [
            STAGE_NAMES[int(s)] if int(s) >= 0 else "unscored" for s in res.expert
        ]
    fd, path = tempfile.mkstemp(prefix="staging_", suffix=".csv")
    os.close(fd)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _render(res: "inference.InferenceResult"):
    """InferenceResult -> the tuple of output values, in output-component order."""
    return (
        f"### Result — {res.source}",
        _hypnogram_fig(res),
        _summary_frame(res),
        _summary_markdown(res),
        _metrics_markdown(res),
        _confusion_fig(res),
        _write_csv(res),
    )


# --------------------------------------------------------------------------- #
# Callbacks                                                                    #
# --------------------------------------------------------------------------- #
def _check_size(path: str, label: str) -> None:
    import gradio as gr

    mb = os.path.getsize(path) / 1e6
    if mb > MAX_UPLOAD_MB:
        raise gr.Error(
            f"{label} is {mb:.0f} MB, over the {MAX_UPLOAD_MB} MB limit for this demo."
        )


def on_stage(psg_path, hyp_path, model, progress=None):
    """Stage an uploaded PSG (+ optional hypnogram)."""
    import gradio as gr

    if not psg_path:
        raise gr.Error("Upload a PSG EDF, or click “Load sample night”.")
    _check_size(psg_path, "PSG file")
    if hyp_path:
        _check_size(hyp_path, "Hypnogram file")
    try:
        res = inference.stage_edf(CFG, psg_path, hyp_path or None, model=model)
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc))
    return _render(res)


def on_sample(model):
    """Stage the bundled held-out sample night."""
    import gradio as gr

    if not SAMPLE_NPZ.exists():
        raise gr.Error(
            "Sample not found. Build it with `python scripts/make_sample.py`."
        )
    res = inference.stage_sample(CFG, SAMPLE_NPZ, model=model)
    return _render(res)


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #
def build_demo():
    import gradio as gr

    with gr.Blocks(title="EEG Sleep Stager") as demo:
        gr.Markdown(
            "# EEG Sleep Stager\n"
            "Automatic AASM sleep staging from a single EEG channel — "
            "a Keras 1-D CNN trained on Sleep-EDF, with subject-wise validation.  \n"
            "**[📂 Source code & write-up on GitHub →](https://github.com/THouwe/eeg-sleep-stager/)**"
        )
        with gr.Accordion("About / disclaimer", open=False):
            gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                psg = gr.File(label="PSG EDF (*-PSG.edf)", file_types=[".edf"],
                              type="filepath")
                hyp = gr.File(label="Hypnogram EDF (optional, for metrics)",
                              file_types=[".edf"], type="filepath")
                model = gr.Radio(choices=MODEL_CHOICES, value="cnn", label="Model")
                with gr.Row():
                    stage_btn = gr.Button("Stage recording", variant="primary")
                    sample_btn = gr.Button("Load sample night")
            with gr.Column(scale=2):
                source_md = gr.Markdown("### Result")
                hypno_plot = gr.Plot(label="Hypnogram (predicted vs expert)")
                summary_md = gr.Markdown()

        with gr.Row():
            summary_df = gr.Dataframe(label="Stage summary", interactive=False)
            cm_plot = gr.Plot(label="Confusion matrix (with expert labels)")

        metrics_md = gr.Markdown()
        csv_file = gr.File(label="Per-epoch predictions (CSV)")

        outputs = [source_md, hypno_plot, summary_df, summary_md, metrics_md,
                   cm_plot, csv_file]
        stage_btn.click(on_stage, inputs=[psg, hyp, model], outputs=outputs)
        sample_btn.click(on_sample, inputs=[model], outputs=outputs)

    return demo


def main() -> None:
    # [onnx] disabled: the bare launch() only served localhost:7860, which a
    # container host can't reach. Bind all interfaces and the injected $PORT
    # (Koyeb/Render set it) so the same entrypoint works in Docker and locally.
    # Restore the line below to revert to the local-only launch.
    # build_demo().launch()
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )


if __name__ == "__main__":
    main()
