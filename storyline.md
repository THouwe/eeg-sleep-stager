# Storyline: Sleep EEG and Polysomnography

*Part of [eeg-sleep-stager](README.md) — see also [cnn.md](cnn.md) and [metrics.md](metrics.md).*

> Background and history for **eeg-sleep-stager**. Why do we cut a night of brain
> activity into 30-second windows and label each one W / N1 / N2 / N3 / REM? Where
> did that convention come from, and what is actually being measured? This document
> is the "why" behind the pipeline — the clinical and scientific story that the code
> automates.

---

## 1. What sleep staging is

Human sleep is not a uniform "off" state. Across a night the brain and body cycle
through distinct physiological states, each with a recognizable electrophysiological
signature. **Sleep staging** (or "sleep scoring") is the process of dividing a
whole-night recording into consecutive fixed-length windows — historically 30
seconds, a convention inherited from the speed of paper polygraph charts — and
assigning each window to one of a small set of stages.

Under the modern American Academy of Sleep Medicine (AASM) framework, the five
scored states are:

| Stage | Name | Character |
|---|---|---|
| **W** | Wake | Alpha rhythm (8–13 Hz) with eyes closed; low-voltage mixed-frequency with eyes open; high muscle tone and eye movements. |
| **N1** | Light sleep (drowsy) | Transition from wake; attenuation of alpha, low-amplitude mixed-frequency activity, slow rolling eye movements, vertex sharp waves. |
| **N2** | Established sleep | **Sleep spindles** (11–16 Hz bursts) and **K-complexes**; the numerically dominant stage of the night. |
| **N3** | Deep / slow-wave sleep | High-amplitude slow delta waves (0.5–2 Hz) over ≥20% of the epoch; hardest to wake from; most physically restorative. |
| **REM** | Rapid eye movement | Low-voltage mixed-frequency EEG resembling wake, but with rapid eye movements and near-complete skeletal muscle atonia; the stage most associated with vivid dreaming. |

A full night is typically 4–6 cycles of roughly 90 minutes, progressing
wake → N1 → N2 → N3 → back up → REM, with deep sleep dominating the first half of
the night and REM periods lengthening toward morning. The plot of stage against
time is called a **hypnogram**, and it is the primary clinical output of a sleep
study.

---

## 2. A short history

### 2.1 The discovery of the human EEG (1920s)

The whole field rests on the electroencephalogram. The German psychiatrist **Hans
Berger** recorded the first human EEG in 1924 and published it in 1929,
demonstrating that rhythmic electrical activity could be measured non-invasively
from the scalp and that it changed with the subject's state — most famously the
**alpha rhythm** that appears when a relaxed subject closes their eyes and
disappears on mental effort (Berger, 1929). Berger's work was initially met with
scepticism until it was independently confirmed by Adrian and Matthews (1934),
after which the EEG became a standard tool of neurophysiology.

### 2.2 Sleep is not uniform: the discovery of REM (1930s–1953)

Early sleep researchers already noticed that the EEG changed as subjects fell
asleep. **Loomis, Harvey, and Hobart (1937)** produced the first systematic
classification of sleep depth from the EEG, labelling stages A through E based on
alpha content, spindles, and slow waves — the direct ancestor of every scoring
system since.

The pivotal discovery came in 1953, when **Aserinsky and Kleitman** at the
University of Chicago reported periods of **rapid, jerky eye movements** recurring
during sleep, accompanied by a low-voltage, fast EEG and correlated with dream
recall (Aserinsky & Kleitman, 1953). This established that sleep contains a
paradoxical, active state — REM sleep — and effectively founded modern sleep
science. **Dement and Kleitman (1957)** then described the cyclic alternation
between REM and non-REM across the night and proposed a stage-classification scheme
tied to those cycles.

### 2.3 Standardization: the Rechtschaffen & Kales manual (1968)

As labs multiplied, so did incompatible scoring conventions. In 1968 a committee
chaired by **Allan Rechtschaffen and Anthony Kales** published *A Manual of
Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human
Subjects* — universally cited as the **"R&K rules"** (Rechtschaffen & Kales, 1968).
It fixed the conventions the field still lives by: scoring in 30-second **epochs**,
a defined set of derivations (EEG, EOG, EMG), and the stage set **Wake, Stage 1,
Stage 2, Stage 3, Stage 4, and REM**. Stages 3 and 4 were distinguished only by how
much of the epoch was occupied by slow waves. R&K governed clinical and research
sleep scoring for nearly four decades.

### 2.4 The AASM revision (2007)

In 2007 the **American Academy of Sleep Medicine** issued a revised manual (Iber,
Ancoli-Israel, Chesson, & Quan, 2007) that remains the current clinical standard
(with periodic updates by Berry et al.). The most consequential changes for this
project:

- Stages 3 and 4 were **merged into a single "N3"** (slow-wave sleep), because the
  4-vs-3 boundary had poor inter-scorer reliability and little physiological
  meaning.
- Non-REM stages were renamed **N1, N2, N3**; REM became **R**; the result is the
  **five-class W / N1 / N2 / N3 / REM scheme** used throughout `eeg-sleep-stager`.
- Recommended electrode derivations and digital-recording standards were formalized.

This five-class AASM mapping is exactly the `stage: 0..4` label space defined in the
project's data model.

---

## 3. Polysomnography: what is actually recorded

**Polysomnography (PSG)** is the simultaneous, multi-channel recording of
physiological signals during sleep — literally "many-sleep-writing." A clinical
montage combines several modalities, because the stages are defined by their
*joint* behaviour, not by the EEG alone:

- **EEG (electroencephalogram)** — cortical electrical activity from scalp
  electrodes placed by the international 10–20 system. This is the signal that
  carries spindles, K-complexes, and slow waves, and it is the single channel this
  project models (**Fpz-Cz** in Sleep-EDF).
- **EOG (electrooculogram)** — eye-movement potentials, essential for detecting the
  rapid eye movements of REM and the slow rolling movements of drowsy N1.
- **EMG (electromyogram)** — muscle tone, usually from the chin; the near-silence of
  chin EMG is a defining feature of REM atonia.
- **ECG, respiratory effort, airflow, pulse oximetry, leg movement** — added in
  full clinical studies to diagnose sleep apnoea, periodic limb movement, and
  arrhythmia.

Scoring rules combine these: for example, REM requires low-voltage mixed-frequency
EEG **and** rapid eye movements **and** low muscle tone occurring together. A single
EEG channel — as used here — cannot see EOG/EMG directly, which is one honest reason
an automatic single-channel stager will never perfectly match a technician who had
the full montage. It is also why single-channel deep learning is an interesting
problem: the network must infer, from the EEG alone, correlates of information the
human scorer read from other channels.

### 3.1 Why 30-second epochs and 100 Hz

The 30-second epoch is a historical artifact of paper polygraphs run at 10 mm/s,
where a standard 30 cm page equalled 30 seconds — and it survived digitization
because decades of normative data and clinical training are built on it. The
Sleep-EDF EEG is sampled at **100 Hz**, so one 30 s epoch is exactly **3000
samples** — the fixed-length raw vector the CNN consumes.

---

## 4. The Sleep-EDF Expanded corpus

This project uses the open-access **Sleep-EDF Database Expanded** hosted on
PhysioNet (Kemp, Zwinderman, Tuk, Kamphuisen, & Oberyé, 2000; Goldberger et al.,
2000). The *Sleep Cassette* subset comprises whole-night ambulatory
polysomnograms from healthy volunteers, each provided as a `*-PSG.edf` signal file
and a matching `*-Hypnogram.edf` of expert manual scoring. The data are distributed
in the **European Data Format (EDF/EDF+)**, the standard interchange format for
clinical physiological signals (Kemp & Olivan, 2003), which the ETL reads with
MNE-Python.

Because these recordings were scored under the older R&K rules, Stages 3 and 4 both
appear in the annotations and are collapsed to N3 in the project's label mapping —
the practical bridge from the 1968 convention to the 2007 five-class target.

---

## 5. Why automatic staging — and why subject-wise evaluation

Manual scoring is slow (roughly 900–1200 epochs per night, each inspected by eye),
expensive, and imperfectly reproducible: inter-scorer agreement between trained
technicians is typically around **80–83% epoch agreement** (Cohen's κ ≈ 0.75–0.80),
with N1 the least reliable stage (Rosenberg & Van Hout, 2013). That human ceiling is
the honest yardstick for any automatic system — a model near ~80% accuracy /
κ ≈ 0.75 on single-channel EEG is performing at roughly the level scorers agree with
*each other*, not failing.

Machine and deep-learning stagers have matured accordingly. **Supratak, Dong, Wu,
and Guo (2017)** introduced *DeepSleepNet*, a CNN + LSTM operating on raw
single-channel EEG, and **Perslev et al. (2021)** produced *U-Sleep*, a
convolutional model that generalizes across cohorts. The `eeg-sleep-stager` CNN
does not chase these state-of-the-art numbers; it aims to reproduce a **correct,
honest baseline**.

The central methodological point — the reason this is a data-science project rather
than a leaderboard chase — is **subject-wise splitting**. Epochs from one sleeper are
highly autocorrelated and share that individual's physiology; if epochs from the same
subject land in both training and test sets, the model can memorize person-specific
traits and report accuracy inflated by roughly 10 points. Reporting numbers under a
strict subject-disjoint split, with **macro-F1** (which refuses to let the rare N1
class be ignored) and **Cohen's κ** (the field-standard agreement metric), is what
makes the reported result trustworthy.

---

## References

Adrian, E. D., & Matthews, B. H. C. (1934). The Berger rhythm: Potential changes
from the occipital lobes in man. *Brain, 57*(4), 355–385.
https://doi.org/10.1093/brain/57.4.355

Aserinsky, E., & Kleitman, N. (1953). Regularly occurring periods of eye motility,
and concomitant phenomena, during sleep. *Science, 118*(3062), 273–274.
https://doi.org/10.1126/science.118.3062.273

Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. *Archiv für
Psychiatrie und Nervenkrankheiten, 87*(1), 527–570.
https://doi.org/10.1007/BF01797193

Dement, W., & Kleitman, N. (1957). Cyclic variations in EEG during sleep and their
relation to eye movements, body motility, and dreaming. *Electroencephalography and
Clinical Neurophysiology, 9*(4), 673–690.
https://doi.org/10.1016/0013-4694(57)90088-3

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C.,
Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000).
PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for
complex physiologic signals. *Circulation, 101*(23), e215–e220.
https://doi.org/10.1161/01.CIR.101.23.e215

Iber, C., Ancoli-Israel, S., Chesson, A. L., & Quan, S. F. (2007). *The AASM manual
for the scoring of sleep and associated events: Rules, terminology and technical
specifications* (1st ed.). American Academy of Sleep Medicine.

Kemp, B., & Olivan, J. (2003). European data format 'plus' (EDF+), an EDF alike
standard format for the exchange of physiological data. *Clinical Neurophysiology,
114*(9), 1755–1761. https://doi.org/10.1016/S1388-2457(03)00123-8

Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberyé, J. J. L.
(2000). Analysis of a sleep-dependent neuronal feedback loop: The slow-wave
microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering, 47*(9),
1185–1194. https://doi.org/10.1109/10.867928

Loomis, A. L., Harvey, E. N., & Hobart, G. A. (1937). Cerebral states during sleep,
as studied by human brain potentials. *Journal of Experimental Psychology, 21*(2),
127–144. https://doi.org/10.1037/h0057431

Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C.
(2021). U-Sleep: Resilient high-frequency sleep staging. *npj Digital Medicine,
4*, 72. https://doi.org/10.1038/s41746-021-00440-5

Rechtschaffen, A., & Kales, A. (Eds.). (1968). *A manual of standardized
terminology, techniques and scoring system for sleep stages of human subjects*.
U.S. Government Printing Office; National Institutes of Health Publication No. 204.

Rosenberg, R. S., & Van Hout, S. (2013). The American Academy of Sleep Medicine
inter-scorer reliability program: Sleep stage scoring. *Journal of Clinical Sleep
Medicine, 9*(1), 81–87. https://doi.org/10.5664/jcsm.2350

Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017). DeepSleepNet: A model for
automatic sleep stage scoring based on raw single-channel EEG. *IEEE Transactions on
Neural Systems and Rehabilitation Engineering, 25*(11), 1998–2008.
https://doi.org/10.1109/TNSRE.2017.2721116
