# The 1-D CNN

*Part of [eeg-sleep-stager](README.md) — see also [storyline.md](storyline.md) and [metrics.md](metrics.md).*

An in-depth description of the deep model that scores sleep stages in this
project: a small 1-D convolutional network that reads the **raw single-channel
EEG epoch** and outputs a probability over the five AASM stages (W, N1, N2, N3,
REM). This document covers the architecture layer by layer, why it is shaped the
way it is, how it is trained, and how it is kept honest at serve time.

- Built by `build_cnn1d()` in [src/eeg_sleep_stager/models.py](src/eeg_sleep_stager/models.py).
- Trained by `train_cnn()` in [src/eeg_sleep_stager/train.py](src/eeg_sleep_stager/train.py).
- Hyperparameters live under `cnn:` in [config.yaml](config.yaml), validated by `CnnConfig` in [src/eeg_sleep_stager/config.py](src/eeg_sleep_stager/config.py).
- Scored by the metrics in [metrics.md](metrics.md); served (no Spark) by [src/eeg_sleep_stager/inference.py](src/eeg_sleep_stager/inference.py).

The design goal is stated up front and honoured throughout: **a correct, honest
baseline, not state of the art.** The model is deliberately small (~93 K
parameters), single-channel, and single-epoch. Where it could be made stronger is
called out in [§8](#8-limitations-and-what-would-move-the-numbers).

---

## 1. What the model sees

One training example is **one 30-second epoch of one EEG channel**:

- Channel `EEG Fpz-Cz`, sampled at **100 Hz** → 30 s × 100 Hz = **3000 samples**.
- Amplitude in µV, then **z-scored per recording over the whole night** (not per
  epoch — see [§7](#7-serve-time-consistency-the-one-invariant)).
- Fed to Keras as shape `(3000, 1)` — 3000 timesteps, 1 channel.

No spectral transform, no hand-crafted features: the network works on the raw
waveform and is expected to learn the frequency-domain structure (slow waves,
spindles, the mixed-frequency look of REM/N1) on its own. The engineered features
in [features.py](src/eeg_sleep_stager/features.py) go instead to the GBM baseline
([§6](#6-the-baseline-it-is-measured-against)); the whole point of the comparison
is "does letting the CNN learn its own features beat the hand-crafted ones?"

---

## 2. Architecture at a glance

Three convolutional blocks (Conv → BatchNorm → MaxPool), global average pooling,
a dropout + dense bottleneck, and a softmax head:

```
Input (3000, 1)
  │
  ├─ Conv1D(32, kernel=50, pad=same, ReLU) → BatchNorm → MaxPool1D(4)   → (750, 32)
  ├─ Conv1D(64, kernel=8,  pad=same, ReLU) → BatchNorm → MaxPool1D(4)   → (187, 64)
  ├─ Conv1D(128,kernel=8,  pad=same, ReLU) → BatchNorm → MaxPool1D(4)   → (46, 128)
  │
  ├─ GlobalAveragePooling1D                                             → (128,)
  ├─ Dropout(0.5)
  ├─ Dense(64, ReLU)                                                    → (64,)
  └─ Dense(5, softmax)                                                  → (5,)
```

Layer-by-layer, with output shapes and parameter counts (input length 3000):

| # | Layer | Output shape | Params | Note |
|---|---|---|---:|---|
| 0 | Input | (3000, 1) | 0 | one z-scored epoch |
| 1 | Conv1D(32, k=50, same, ReLU) | (3000, 32) | 1,632 | **wide** kernel — 0.5 s of signal |
| 2 | BatchNormalization | (3000, 32) | 128 | 64 trainable |
| 3 | MaxPool1D(4) | (750, 32) | 0 | ↓4 in time |
| 4 | Conv1D(64, k=8, same, ReLU) | (750, 64) | 16,448 | narrower, deeper |
| 5 | BatchNormalization | (750, 64) | 256 | 128 trainable |
| 6 | MaxPool1D(4) | (187, 64) | 0 | ↓4 |
| 7 | Conv1D(128, k=8, same, ReLU) | (187, 128) | 65,664 | narrowest, widest |
| 8 | BatchNormalization | (187, 128) | 512 | 256 trainable |
| 9 | MaxPool1D(4) | (46, 128) | 0 | ↓4 |
| 10 | GlobalAveragePooling1D | (128,) | 0 | collapse time axis |
| 11 | Dropout(0.5) | (128,) | 0 | train-time only |
| 12 | Dense(64, ReLU) | (64,) | 8,256 | classification bottleneck |
| 13 | Dense(5, softmax) | (5,) | 325 | stage probabilities |

**Total: ~93,221 parameters (~92,773 trainable, 448 non-trainable BN buffers).**
That is a very small network by deep-learning standards — it fits comfortably on a
laptop and trains in minutes on the subset. Small is a feature here, not a
compromise: it is hard to overfit ~150 nights with 93 K parameters, and the honest
subject-wise numbers ([metrics.md](metrics.md)) are what the project is selling.

> Padding is `same` on every conv (length preserved), so all downsampling is done
> by the pooling layers. The three `MaxPool1D(4)` layers reduce the time axis by
> 4³ = 64× (3000 → 750 → 187 → 46; integer floor at each step).

---

## 3. Why it is shaped this way

### The wide first kernel (50 samples)
The first convolution uses a **kernel of 50 samples ≈ 0.5 s**, far wider than the
size-8 kernels above it. At 100 Hz, 0.5 s spans a full cycle of a 2 Hz slow wave
and most of a delta/theta oscillation — the low-frequency morphology that
separates deep sleep (N3, delta-dominated) from wake and REM. A wide first
receptive field lets a single filter respond to slow-wave *shape* directly instead
of having to reconstruct it from many tiny kernels. The deeper layers then use
narrow size-8 kernels over the pooled, downsampled signal, where each step already
covers more real time, to compose these primitives into higher-level patterns
(spindle bursts, K-complexes, the ragged mixed-frequency texture of N1/REM).

By the last conv, a single unit's **temporal receptive field is ~253 samples ≈
2.5 s** of the original signal; global average pooling then integrates those
detections across the **entire 30 s epoch**, so the final decision is informed by
the whole window, not one spot in it.

### BatchNorm after every conv
Each conv is followed by `BatchNormalization`, which stabilises and speeds up
training (less sensitivity to the learning rate and to the exact input scale) and
adds a mild regularising effect. It also makes the network more robust to the
residual scale differences that survive per-recording z-scoring.

### Global average pooling instead of Flatten
After the conv stack the `(46, 128)` feature map is collapsed by
**GlobalAveragePooling1D** to a 128-vector — one average per filter over time.
This is a deliberate choice over `Flatten` + a large dense layer:

- It makes the classifier **position-invariant**: a spindle counts the same
  wherever in the 30 s epoch it occurs.
- It **cuts parameters drastically**. `Flatten` here would produce a 46×128 = 5,888
  vector and a `Dense(64)` on it would cost ~377 K parameters — four times the
  whole current network — with far more overfitting risk.

### The dropout + dense head
A `Dropout(0.5)` sits between the pooled features and a small `Dense(64, ReLU)`
bottleneck, before the 5-way softmax. Half the pooled activations are dropped at
each training step, which is aggressive on purpose: with only ~150 nights, the
head is the easiest place to overfit, and this is the model's main explicit
regulariser alongside BatchNorm and the class-weighted loss.

---

## 4. Output head and loss

- **Head:** `Dense(5, softmax)` → a probability distribution over
  `[W, N1, N2, N3, REM]` (ids 0–4, mapping in [labels.py](src/eeg_sleep_stager/labels.py)).
- **Loss:** `sparse_categorical_crossentropy` — the labels are integer stage ids,
  not one-hot, so the sparse variant is used directly.
- **Reported training metric:** `accuracy` is compiled in for convenience, but
  **accuracy is not the selection criterion** — early stopping and checkpointing
  key off macro-F1 ([§5](#5-training)), because accuracy is dominated by the
  majority N2 class and rewards ignoring the rare stages. See
  [metrics.md §2](metrics.md) for why.

At serve time the same softmax vector is returned as `proba` in
`InferenceResult`, and `argmax` gives the predicted stage
([inference.py](src/eeg_sleep_stager/inference.py) `predict_stages`).

---

## 5. Training

Driven by `train_cnn()` in [train.py](src/eeg_sleep_stager/train.py); config under
`cnn:` in [config.yaml](config.yaml).

| Setting | Value | Where |
|---|---|---|
| Optimizer | Adam | `build_cnn1d` |
| Learning rate | 1e-3 | `cnn.learning_rate` |
| Batch size | 128 | `cnn.batch_size` |
| Max epochs | 30 | `cnn.epochs` |
| Early-stopping patience | 6 | `cnn.early_stopping_patience` |
| Class weights | inverse-frequency ("auto") | `cnn.class_weights` |
| Selection metric | **val macro-F1** | `_macro_f1_callback` + `EarlyStopping` |

### Class weights (the N1 problem)
Sleep stages are severely imbalanced — N1 is only ~13 % of epochs, N2 ~36 %
([metrics.md §2](metrics.md)). Left unaddressed, the network minimises loss by
under-calling the rare stages. `compute_class_weights()` in
[datasets.py](src/eeg_sleep_stager/datasets.py) computes **balanced
(inverse-frequency) weights** and passes them to `model.fit(class_weight=...)`, so
a misclassified N1 epoch contributes proportionally more loss than a
misclassified N2. This trades a little raw accuracy for much better minority-class
recall — exactly the trade the macro-F1 headline is meant to reward.

### Selecting on macro-F1, not val loss
Keras has no built-in macro-F1 metric, so `_macro_f1_callback` computes it on the
validation set at each epoch end and injects `val_macro_f1` into the logs. Both
`EarlyStopping` and `ModelCheckpoint` then monitor `val_macro_f1` with
`mode="max"`:

- **Early stopping** halts once macro-F1 has not improved for 6 epochs and
  restores the best weights.
- **ModelCheckpoint** saves the best-so-far network to `models/cnn.keras`.
- A final `model.save()` guarantees the restored best weights are on disk even if
  no epoch triggered a checkpoint.

Selecting on macro-F1 rather than validation loss or accuracy means the chosen
checkpoint is the one that scored the *five stages most evenly*, which is the
metric the project actually reports.

### What is held out
Training only ever sees the **train** split; macro-F1 is monitored on **val**; the
**test** subjects are never touched here. The split is **subject-wise**
([split.py](src/eeg_sleep_stager/split.py)) — no epoch from a test subject appears
in training — which is the single discipline point that keeps the reported numbers
from being inflated by cross-subject leakage ([metrics.md §1](metrics.md)).

---

## 6. The baseline it is measured against

The CNN is not reported alone. `build_baseline()` builds a scikit-learn
**GradientBoostingClassifier** (300 trees, depth 3, lr 0.05) over the **10
engineered features** per epoch — band powers, spectral entropy, Hjorth
parameters, RMS ([features.py](src/eeg_sleep_stager/features.py)). It is trained
with balanced sample weights, the same imbalance correction the CNN gets.

This is the "does deep learning earn its keep here?" control. On the current
held-out test set the CNN beats the baseline on every stage and decisively on REM
(F1 0.70 vs 0.50) — the CNN learns REM morphology from the raw signal that the
hand-crafted spectral features miss ([metrics.md §5](metrics.md)). That contrast,
not a leaderboard rank, is the result.

---

## 7. Serve-time consistency (the one invariant)

The network is only valid if an epoch at inference time is built **exactly** the
way it was in training. The invariant, owned by `epochs_from_edf()` in
[inference.py](src/eeg_sleep_stager/inference.py) and pinned by a test against the
ETL path:

> Single channel `EEG Fpz-Cz`, 100 Hz, Volts→µV, then a **per-recording z-score
> over the whole night**, then consecutive non-overlapping 30 s (3000-sample)
> epochs from t=0.

The z-score is over the **entire recording**, not per epoch. That is why the web
demo takes a **full-night EDF** rather than a single clip: a per-epoch z-score
would normalise away exactly the amplitude information (e.g. the large slow waves
of N3) that distinguishes the stages, producing train/serve skew. The serve path
reuses the ETL's epsilon and float64 accumulation so the normalisation is
bit-for-bit identical to training, not merely similar.

Recordings with a different channel or sample rate raise a clear error rather than
being silently resampled and guessed at.

**Serve backend (ONNX).** Training writes `models/cnn.keras` on TensorFlow, but the
hosted demo serves TF-free: `cnn.keras` is exported to `models/cnn.onnx`
(**opset 13**, dynamic batch axis) by [`scripts/keras_to_onnx.py`](scripts/keras_to_onnx.py)
and run with **onnxruntime 1.17.3**. ONNX replaces only the matrix multiply, not the
preprocessing above — a parity test (`test_cnn_onnx_matches_keras`) pins the two
backends to identical argmax and `atol=1e-4` softmax, so the slim-down is provably
result-preserving. Inference is mini-batched (128 epochs) to keep peak RAM under the
free host's 512 MB. See [cnn-to-onnx.md](cnn-to-onnx.md).

---

## 8. Limitations and what would move the numbers

The model is honest about what it is *not*:

- **Single epoch, no context.** Each 30 s epoch is classified independently. Human
  scorers and the strongest models (DeepSleepNet, U-Sleep) use the **sequence** of
  neighbouring epochs — sleep has strong temporal structure (you rarely jump W→N3).
  A recurrent or transformer layer over the per-epoch CNN embeddings is the single
  most likely improvement, especially for N1, the hardest and most context-dependent
  stage.
- **Single channel.** Only EEG Fpz-Cz. Adding EOG (for REM) and EMG (for
  wake/atonia) is standard in clinical scoring and would most help REM and N1; the
  Parquet schema leaves room for extra channels, but fusion is out of scope here.
- **Small capacity by design.** ~93 K parameters on ~150 nights. More data and a
  larger network would raise the ceiling, but the project deliberately targets a
  correct, reproducible baseline (~78–82 % accuracy, κ ≈ 0.75 on Fpz-Cz) rather than
  SOTA.
- **Not a medical device.** Batch inference over a full night for demonstration only
  — not real-time, not streaming, not clinical.

For how the resulting predictions are scored and why macro-F1 and Cohen's κ are the
headline numbers, see [metrics.md](metrics.md).
