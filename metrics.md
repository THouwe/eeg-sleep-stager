# Evaluation metrics

*Part of [eeg-sleep-stager](README.md) — see also [storyline.md](storyline.md) and [cnn.md](cnn.md).*

How this project measures a sleep stager, why these particular metrics, and how to
read them. The short version: **accuracy alone is misleading on sleep data**, so the
headline numbers are **macro-F1** and **Cohen's κ**, reported on **held-out
subjects**.

- Computed by `compute_metrics()` in [src/eeg_sleep_stager/evaluate.py](src/eeg_sleep_stager/evaluate.py) (pure, unit-tested).
- Written to `reports/metrics.json` + `reports/confusion_matrix_<model>.png` by `eeg-sleep-stager evaluate`.
- Shown live in the web demo whenever an expert hypnogram is supplied.

---

## 1. The evaluation setup (why the numbers are honest)

Two decisions make these metrics trustworthy rather than flattering:

- **Subject-wise test split.** No epoch from a test subject ever appears in
  training. Epoch-level random splits leak physiology across the split (adjacent
  30 s epochs from the same night are near-duplicates) and inflate accuracy by
  ~10 points. Numbers here are on subjects the model has never seen.
- **Evaluated on the test split only.** Never on train or validation data.

Everything below is computed over the set of test epochs (predicted stage vs the
expert technician's label, per 30 s epoch).

---

## 2. Read this first: the class-imbalance problem

Sleep stages are wildly imbalanced. On the held-out test set (12 subjects,
31,388 epochs) the expert labels break down as:

| Stage | Epochs | Share |
|---|---:|---:|
| N2 | 11,161 | 36% |
| W  | 11,108 | 35% |
| N1 |  4,067 | 13% |
| REM|  3,518 | 11% |
| N3 |  1,534 |  5% |

N2 and W together are ~71% of epochs, while N3 is only ~5% and N1 ~13%. A
do-nothing classifier that **always predicts N2** would already score **~36%
accuracy** while being clinically useless. N1 — the transitional light-sleep stage
— and N3 (deep sleep) are the rare classes, and N1 is the hardest to score even for
humans. This imbalance is the reason a single accuracy number is not enough, and
why the metrics below deliberately give the rare stages an equal vote.

---

## 3. The metrics, one by one

Notation, per stage: **TP** true positives, **FP** false positives, **FN** false
negatives, **TN** true negatives.

### Accuracy
Fraction of epochs whose predicted stage equals the expert's.

```
accuracy = (correct epochs) / (total epochs)
```

- **Range** 0–1 (higher better). **Reports** overall agreement at a glance.
- **Pitfall** dominated by the majority class. High accuracy can hide total failure
  on N1/N3/REM (see the "always predict N2" case above). Use it as a headline
  *only* alongside macro-F1 and κ.

### Precision (per stage)
When the model says "this is stage X", how often is it right?

```
precision_X = TP_X / (TP_X + FP_X)
```

- Low precision ⇒ the model over-calls that stage (many false alarms).
- Clinically: your *trust* in a positive call for stage X.

### Recall (a.k.a. sensitivity, per stage)
Of all the true stage-X epochs, how many did the model catch?

```
recall_X = TP_X / (TP_X + FN_X)
```

- Low recall ⇒ the model misses that stage (under-detects it).
- Clinically: the *detection rate* for stage X. Recall is the diagonal of a
  row-normalized confusion matrix (section 4).

### F1 (per stage)
The harmonic mean of precision and recall — a single per-stage score that punishes
imbalance between the two.

```
F1_X = 2 · (precision_X · recall_X) / (precision_X + recall_X)
```

- **Range** 0–1. The harmonic mean (not the arithmetic mean) means F1 is high only
  when *both* precision and recall are high; being great at one and terrible at the
  other still yields a low F1.

### Macro-F1 — the primary headline
The **unweighted** average of the five per-stage F1 scores.

```
macro_F1 = mean(F1_W, F1_N1, F1_N2, F1_N3, F1_REM)
```

- **Why it's the headline:** every stage counts equally regardless of how rare it
  is, so a model that ignores N1 is penalized hard. This is exactly what accuracy
  fails to do.
- **Contrast:** *weighted*-F1 (average F1 weighted by support) and *micro*-F1
  (≈ accuracy on a single-label problem) both let the majority class dominate again
  — so macro-F1 is the honest choice for imbalanced staging.

### Cohen's κ (kappa) — the sleep-staging standard
Agreement between model and expert **corrected for the agreement you'd expect by
chance**.

```
κ = (p_o − p_e) / (1 − p_e)
```

where `p_o` is observed agreement (= accuracy) and `p_e` is the agreement expected
if both labelled independently at each class's base rate.

- **Range** −1 to 1. `κ = 1` perfect, `κ = 0` no better than chance, `κ < 0` worse
  than chance.
- **Why it matters here:** because the classes are imbalanced, a lot of raw
  agreement is "free" (both label the frequent N2). κ subtracts that free credit, so
  it's a much fairer number — and it's the metric the clinical sleep literature
  reports, which lets you compare against **human inter-scorer agreement**
  (typically κ ≈ 0.76–0.80).
- **Interpretation (Landis & Koch):**

  | κ | Agreement |
  |---|---|
  | < 0.00 | Poor (worse than chance) |
  | 0.00–0.20 | Slight |
  | 0.21–0.40 | Fair |
  | 0.41–0.60 | Moderate |
  | 0.61–0.80 | Substantial |
  | 0.81–1.00 | Almost perfect |

### Support (per stage)
The number of true epochs of each stage in the test set. Not a score — context. A
per-stage F1 computed on tiny support (e.g. N3 with 670 epochs) is noisier and
swings more between subjects, so read the minority-class F1s with that in mind.

---

## 4. The confusion matrix

A 5×5 table: **rows = expert (true) stage, columns = predicted stage**. Cell
`(i, j)` counts epochs whose true stage was `i` and prediction was `j`. The
**diagonal is correct**; everything off-diagonal is an error, and *where* the errors
land is diagnostic.

`plot_confusion_matrix()` writes a **row-normalized** version (each row sums to 1),
so the diagonal reads directly as **per-stage recall**, and each row shows where a
stage's epochs get misrouted.

**Physiologically expected confusions** (a model making *these* mistakes is failing
sensibly, the way humans do):
- **N1 ↔ W** and **N1 ↔ REM** — N1 is low-amplitude mixed-frequency EEG that looks
  like drowsy wake or REM; it's the classic hard class.
- **N2 ↔ N3** — the N2/N3 boundary is a slow-wave *percentage* threshold, genuinely
  gradual.
- **N2 ↔ N1** — light-sleep boundary, spindle presence is subtle on one channel.

---

## 5. Worked example (current model, full corpus)

Full **Sleep-EDF Cassette** corpus, evaluated on the 12 held-out test subjects
(`SC08, SC15, SC33, SC48, SC49, SC51, SC52, SC53, SC54, SC62, SC70, SC71`;
31,388 epochs). These are the current `models/` numbers (`reports/metrics_*.json`).

| Metric | 1-D CNN | GBM baseline |
|---|---:|---:|
| Accuracy | **0.776** | 0.677 |
| Macro-F1 | **0.714** | 0.617 |
| Cohen's κ | **0.694** | 0.571 |

Per-stage **F1**:

| Stage | CNN | Baseline | Support |
|---|---:|---:|---:|
| W   | 0.94 | 0.84 | 11,108 |
| N1  | 0.44 | 0.39 |  4,067 |
| N2  | 0.78 | 0.73 | 11,161 |
| N3  | 0.71 | 0.62 |  1,534 |
| REM | 0.70 | 0.50 |  3,518 |

**How to read this:**
- κ = 0.694 is **substantial** agreement, approaching the human inter-scorer band
  (κ ≈ 0.76–0.80) — a solid, honest result for a plain single-channel CNN on 12
  unseen subjects.
- The CNN **beats the baseline on every stage**, decisively on **REM** (0.70 vs
  0.50) and **N3** (0.71 vs 0.62) — the "does deep learning earn its keep?" answer:
  yes, mainly by learning REM/deep-sleep morphology the hand-crafted band-power and
  Hjorth features miss.
- **N1 is the weakest** for both models (0.44 / 0.39), exactly as expected — low
  prevalence and intrinsically ambiguous. This is where macro-F1 (0.714) sits below
  accuracy (0.776): the rare hard classes drag the *fair* average down, which is the
  whole point of reporting it.

CNN confusion matrix, **row-normalized** (rows = expert, diagonal = recall):

| true ↓ / pred → | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| **W**   | **0.94** | 0.04 | 0.00 | 0.00 | 0.01 |
| **N1**  | 0.18 | **0.46** | 0.14 | 0.02 | 0.20 |
| **N2**  | 0.00 | 0.16 | **0.70** | 0.05 | 0.08 |
| **N3**  | 0.00 | 0.01 | 0.19 | **0.79** | 0.01 |
| **REM** | 0.02 | 0.12 | 0.04 | 0.00 | **0.83** |

The off-diagonal mass is exactly the expected confusions: N1 (recall 0.46) leaks
into W, N2 and REM roughly equally — it looks like all of them on one channel; REM
leaks mainly into N1 (0.12); N2 leaks into N1 (0.16) and N3 (0.05). W (0.94) and REM
(0.83) recall are strong. Note N1's *F1* (0.44) is close to its recall (0.46)
because precision is also low — other stages get mislabelled as N1, so it's both
under-detected and over-called.

---

## 6. Regenerating the numbers

```bash
eeg-sleep-stager evaluate --model cnn        # -> reports/metrics.json + PNGs
eeg-sleep-stager evaluate --model baseline
```

This recomputes everything on the current test split and overwrites
`reports/metrics.json`, `reports/confusion_matrix_<model>.png`, and
`reports/hypnogram_<model>.png`. `reports/metrics.json` holds whichever model was
evaluated last; `reports/metrics_cnn.json` and `reports/metrics_baseline.json` keep
each model's results side by side.

The checked-in `reports/` reflect the current full-corpus models (section 5).
Regenerate after any retrain or re-split so the numbers stay in sync with
`models/`.

---

## 7. What "good" looks like on this dataset

- **Realistic target** for a plain single-channel (Fpz-Cz) CNN: **~78–82%
  accuracy, κ ≈ 0.75**. The current model sits right in that band.
- **Human inter-scorer agreement** on the same task is roughly **κ ≈ 0.76–0.80**, so
  a κ in the low-0.7s is close to the human ceiling on easy stages — most remaining
  disagreement is on N1, where humans also disagree.
- **State of the art** (e.g. U-Sleep, DeepSleepNet) reaches higher, often with more
  channels, larger corpora, and sequence models over neighbouring epochs. This
  project deliberately aims for a **correct, honest baseline**, not SOTA — so the
  interesting comparison is CNN-vs-features and model-vs-human-agreement, not the
  leaderboard.
