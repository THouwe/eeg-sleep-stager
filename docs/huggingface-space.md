# Deploying the web demo to Hugging Face Spaces

The Gradio demo (`app.py`) runs unchanged on a free **Gradio Space**. This is the
model-serving path only — no Spark, no dataset, no training. Everything the Space
needs is already in the repo except that two directories are `.gitignore`d
(`models/`, and `data/`); you push the two model files explicitly and leave the
data behind.

> **Python must be 3.11.** `tensorflow==2.15.0` and `scipy==1.11.4` have no wheels
> for 3.12+, so a Space on a newer Python tries to compile from source and fails.
> The front-matter below pins it.

---

## 1. What the Space needs

| Path | Why | Notes |
|---|---|---|
| `app.py` | entry point | puts `./src` on `sys.path` — no package install |
| `src/eeg_sleep_stager/` | the package (inference, config, etc.) | imported from source |
| `config.yaml` | paths, bands, channel, epoch length | read at startup |
| `requirements.txt` | serve-time deps (no pyspark, no `.`) | pinned, mutually compatible |
| `app_assets/sample_night.npz` | one-click sample (held-out test night) | ~5.5 MB binary |
| `app_assets/sample_meta.json` | sample label | tiny |
| `models/cnn.keras` | trained CNN | ~1.2 MB binary, **git-ignored in this repo** |
| `models/baseline.joblib` | trained GBM baseline | ~1.9 MB binary, **git-ignored in this repo** |

**Not needed / do not push:** `data/` (raw EDFs + Parquet, gigabytes), `.venv*/`,
`reports/`, `tests/`, `docs/`, `Makefile`, `scripts/` (the sample is already built).

## 2. Space README front-matter

A Space is configured by YAML front-matter at the top of its **`README.md`**.
Create the Space's README with this header (edit title/emoji to taste):

```yaml
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
```

`sdk_version` must match the pinned `gradio` in `requirements.txt` (4.44.1); if you
bump one, bump both. Everything after the `---` block is the Space's description
page (reuse the project README's intro if you like).

## 3. Model + sample binaries (Git LFS)

`models/` is `.gitignore`d in this project, so the two model files won't be pushed
by a normal `git add`. They're small (<2 MB each) and the sample is ~5.5 MB — all
under Hugging Face's 10 MB soft limit, so **plain git works**. Using Git LFS is
still the tidy choice for binaries and is required if a retrained `cnn.keras` grows
past 10 MB:

```bash
git lfs install
git lfs track "*.keras" "*.joblib" "*.npz"
git add .gitattributes
git add -f models/cnn.keras models/baseline.joblib          # -f overrides .gitignore
git add app_assets/sample_night.npz app_assets/sample_meta.json
```

If you skip LFS, still use `git add -f models/…` to get past `.gitignore`.

## 4. Two ways to deploy

### A. Push to the Space repo directly
1. On huggingface.co → **New Space** → SDK **Gradio**, name it, create.
2. Clone it and copy the needed files in (or add the Space as a second remote to
   this repo and push a curated tree):
   ```bash
   git clone https://huggingface.co/spaces/<user>/<space> hf-space
   ```
   Copy `app.py`, `src/`, `config.yaml`, `requirements.txt`, `app_assets/`,
   `models/cnn.keras`, `models/baseline.joblib`, and write the `README.md` with the
   front-matter above.
3. `git lfs track` the binaries (section 3), commit, `git push`. The Space builds
   automatically and shows a live log.

### B. Link the GitHub repo
Spaces can build from a connected GitHub repo. This works, but note the two model
files are git-ignored here — either remove them from `.gitignore` on a deploy
branch, or use option A. The data/ dir stays ignored (good — it's huge).

## 5. Runtime notes

- **Cold start.** The TensorFlow image is heavy; the first request loads
  `cnn.keras` (a few seconds). The current `predict_stages` reloads the model per
  request — fine for a demo. If you want snappier repeat calls, cache the loaded
  model at module scope in `app.py` (a small S8 polish).
- **Free-tier sleep.** Free Spaces sleep after inactivity and cold-start on the
  next visit. Expect the first hit after idle to be slow.
- **Uploads.** `app.py` caps uploads at 200 MB (Sleep-EDF Cassette PSGs are
  ~50 MB). Adjust `MAX_UPLOAD_MB` if needed.
- **CPU is enough.** Inference on one night is fast on CPU; no GPU hardware tier
  required.

## 6. After a retrain

The serve contract (3000-sample epochs, Fpz-Cz, 5 classes) doesn't change, so a
retrained `models/cnn.keras` is drop-in — just push the new file. **But regenerate
the bundled sample**, because the subject-wise split changes with new data and the
old sample subject may no longer be held out:

```bash
python scripts/make_sample.py     # confirm the WARNING does NOT fire (subject in test split)
```

Then push the refreshed `app_assets/sample_night.npz` + `sample_meta.json`.
