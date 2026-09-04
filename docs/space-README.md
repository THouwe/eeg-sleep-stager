---
title: EEG Sleep Stager
emoji: 😴
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# EEG Sleep Stager

Automatic **sleep staging** from a single EEG channel. Upload a full night of
polysomnography and a 1‑D convolutional neural network scores every 30‑second
epoch into the five AASM stages — **W, N1, N2, N3, REM** — and draws the
hypnogram, the way a sleep technician would.

Trained on the open [Sleep‑EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
corpus (78 subjects), with **subject‑wise validation** — no epoch from a test
subject is ever seen in training — so the reported numbers are honest rather than
leaked.

## Try it

- **No file?** Click **Load sample night**. It runs a bundled recording from a
  **held‑out test subject** (unseen in training), so what you see is an honest
  estimate of real performance.
- **Your own recording?** Upload a full‑night **`*-PSG.edf`** in Sleep‑EDF Cassette
  format (channel **Fpz‑Cz**, sampled at **100 Hz**). Other montages or sample
  rates are rejected with a clear message rather than silently mis‑scored.
- **Have the expert scoring too?** Add the matching **`*-Hypnogram.edf`** to get an
  expert‑vs‑predicted overlay and honest metrics (accuracy, macro‑F1, Cohen's κ,
  per‑stage F1, confusion matrix).
- Switch between the **1‑D CNN** and a classical **features + GBM baseline** with
  the radio.

You get back a **predicted hypnogram** (with a per‑epoch probability strip), a
**stage summary** (minutes, total sleep time, efficiency), metrics when a
hypnogram is supplied, and a **downloadable per‑epoch CSV**.

## How well does it work?

On the 12 held‑out test subjects (31,388 epochs):

| Model | Accuracy | Macro‑F1 | Cohen's κ |
|---|---:|---:|---:|
| **1‑D CNN** | **0.78** | **0.71** | **0.69** |
| Features + GBM baseline | 0.68 | 0.62 | 0.57 |

Cohen's κ ≈ 0.69 is **substantial** agreement, approaching the human inter‑scorer
band (κ ≈ 0.76–0.80). The CNN beats the classical baseline on every stage, most
clearly on **REM** and deep sleep (**N3**). As everywhere in the sleep literature,
**N1** (the wake→sleep transition) is the hardest stage for both the model and
human scorers.

## Why a whole night, not one epoch?

Each epoch is normalized using statistics from the **entire recording** (exactly as
in training), so the model needs the full night, not an isolated 30‑second window.
Inference is plain MNE/NumPy/TensorFlow — the heavy distributed feature‑extraction
pipeline (PySpark) is only used offline to build the training data.

## Scope & honesty

This is a **research demonstration, not a medical device**, and produces no
clinical or diagnostic output. It uses a single EEG channel (Fpz‑Cz); a full
clinical montage also uses EOG and EMG, which especially help distinguish REM — so
single‑channel REM scoring is inherently harder here. Ground truth itself is an
expert convention, not absolute truth: sleep stages are defined by scoring rules,
and even trained humans disagree, particularly on N1.

## Learn more

Code, the full pipeline, and a detailed metrics write‑up:
**https://github.com/THouwe/eeg-sleep-stager**
