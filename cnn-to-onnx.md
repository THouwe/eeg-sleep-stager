# CNN → ONNX slim-down & free deployment (Koyeb)

Goal: make the **serve path** TensorFlow-free so the Gradio demo fits a free
512 MB, no-card host (**Koyeb**, Render as fallback). We convert the trained
`models/cnn.keras` to `models/cnn.onnx`, run inference with **onnxruntime**
(a fraction of TF's size and RAM), containerize, and deploy.

**Why:** Hugging Face now gates free `cpu-basic` Gradio Spaces behind PRO, and its
free ZeroGPU tier is PyTorch-only — so a TensorFlow Gradio app has no free HF home.
onnxruntime-cpu (~tens of MB) replaces TensorFlow (~hundreds of MB) at serve time
and fits the free tiers.

---

## Guardrails (read first — these are hard rules for this refactor)

1. **Do not delete any file or any code.** If a line must go, **comment it out**
   with a clear marker: `# [onnx] disabled: <reason>`. The TF/Keras path must stay
   recoverable.
2. **Training stays on TensorFlow.** `models.py`, `train.py`, `datasets.py`,
   `etl.py` are **untouched** — the pipeline still trains and writes `cnn.keras`.
   Only the **serve path** (`inference.predict_stages` CNN branch, `app.py`,
   `requirements.txt`) switches to ONNX.
3. **The preprocessing invariant is sacred.** `epochs_from_edf` / `_zscore` /
   `_cut_epochs` do not change. ONNX only replaces the *matrix multiply*, not the
   inputs to it.
4. **Keep `pytest` green at every step**, and keep the two model backends
   producing the same predictions (S3 parity test).
5. Conversion (Keras→ONNX) runs in the **TF-enabled `.venv`**; the serve/Docker
   env is **TF-free** and only needs onnxruntime.

Two Python environments are in play:
- **`.venv`** — the full pipeline env (has TensorFlow). Used for training,
  conversion, and running the parity test. Add `onnxruntime` here too.
- **serve env** — `requirements.txt` (Docker / Koyeb). TF removed, onnxruntime added.

---

## S1 — Conversion tooling: `cnn.keras` → `cnn.onnx`
- Add `tf2onnx` + `onnxruntime` to the pipeline env: `.venv/Scripts/python.exe -m pip install tf2onnx onnxruntime`.
- New file `scripts/keras_to_onnx.py` (does **not** replace anything): load
  `models/cnn.keras`, export to `models/cnn.onnx` with a **dynamic batch axis** and
  a fixed opset (13+). Prefer the programmatic API
  (`tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec((None,3000,1),tf.float32)], opset=13, output_path=...)`);
  if that fights TF 2.15, fall back to exporting a `SavedModel` and
  `python -m tf2onnx.convert --saved-model ... --output models/cnn.onnx --opset 13`.
- Run it; then sanity-check in onnxruntime: input name via
  `sess.get_inputs()[0].name`, output via `sess.get_outputs()[0].name`, output
  shape `(N,5)`, softmax rows sum to 1.
- **Commit the artifact:** `git add -f models/cnn.onnx` (like the other model files).
- **Done when:** `models/cnn.onnx` exists and loads; input/output names recorded in
  a comment in `keras_to_onnx.py`.

## S2 — ONNX backend in `inference.py`
- Add a cached loader `_load_cnn_onnx(models_dir)` returning an
  `onnxruntime.InferenceSession` (mirror the existing `_MODEL_CACHE` pattern; import
  `onnxruntime` lazily inside the function).
- In `predict_stages`, CNN branch: **comment out** the two TF lines
  (`net = _load_cnn(...)` / `net.predict(...)`) with the `# [onnx] disabled:` marker
  and replace with an onnxruntime call: build `x = z[..., None].astype("float32")`,
  `proba = sess.run([out_name], {in_name: x})[0]`, then `argmax`. Read `in_name` /
  `out_name` from the session (don't hard-code).
- Leave `_load_cnn` (the TF loader) defined but unused — recoverable per the rules.
- **Done when:** `stage_sample(..., model="cnn")` returns predictions via ONNX and
  the numbers match the Keras run (checked properly in S3).

## S3 — Parity test (the equivalence guard)
- New test in `tests/test_inference.py` (skip if either `cnn.keras` or `cnn.onnx`
  is absent): load both backends, run on the bundled sample (or fixed random
  epochs), assert **argmax labels are identical** and **softmax probs are close**
  (`np.allclose(atol=1e-4)`). This is the guarantee the slim-down didn't change
  results.
- Update `test_predict_stages_cnn_smoke` to the ONNX path (skip on missing
  `cnn.onnx`). Do not remove the old assertions — adapt them.
- **Done when:** `pytest -q` green, parity test passing.

## S4 — Make the serve env TensorFlow-free
- Rewrite `requirements.txt` (serve list): **remove `tensorflow`**, **add
  `onnxruntime==<pinned>`** (a version with 3.11 wheels), keep
  `mne, scipy, numpy, pandas, pyarrow, scikit-learn, matplotlib, pyyaml, pydantic`
  and the gradio stack. Keep it TF-free and pyspark-free.
- Ensure **no TF import survives on the serve path**: the TF lines in
  `predict_stages` are commented (S2); confirm `import app` and
  `stage_sample(model="cnn")` never import `tensorflow`
  (`"tensorflow" not in sys.modules` after a sample run).
- **Done when:** a fresh venv built from `requirements.txt` (no TF installed) runs
  `python app.py` and stages the sample end-to-end.

## S5 — Container entrypoint: bind `$PORT`
- In `app.py` `main()`: **comment out** `build_demo().launch()` and replace with a
  container-friendly launch: `server_name="0.0.0.0"`,
  `server_port=int(os.environ.get("PORT", 7860))` (Koyeb/Render inject `$PORT`).
  Local `python app.py` still works (default 7860).
- **Done when:** `PORT=8000 python app.py` serves on 8000; bare `python app.py` on 7860.

## S6 — Dockerfile
- New `Dockerfile`: `python:3.11-slim` base, `COPY` the repo, `pip install -r
  requirements.txt`, run `python app.py`. New `.dockerignore` excluding
  `data/`, `.venv*`, `__pycache__/`, `reports/`, and **`models/cnn.keras`** (the
  Keras file isn't needed at serve — only `cnn.onnx` + `baseline.joblib`), to keep
  the image small.
- Build and run locally: `docker build -t eeg-stager . && docker run -p 8000:8000 -e PORT=8000 eeg-stager`; open `localhost:8000`, click **Load sample night**.
- **Done when:** the container serves and stages the sample; note the image size and
  peak RAM (must sit under ~512 MB — if tight, trim: e.g. gate the baseline toggle
  to drop `scipy`, or drop `pyarrow`).

## S7 — Deploy to Koyeb (free, no card)
- Commit everything (`cnn.onnx`, `requirements.txt`, `Dockerfile`, `.dockerignore`,
  code changes) and push to GitHub.
- Create a **Koyeb** account (verify **no card** at signup; if they now require one,
  pivot to Render — same artifact). New service → **GitHub** → this repo →
  **Dockerfile** build → **Free (nano)** instance → deploy.
- Watch the build log; confirm it reaches **Running** and the sample stages within
  the RAM limit.
- **Troubleshooting:** OOM → trim deps (S6 note); blank page → confirm `$PORT`
  binding (S5); model-not-found → confirm `cnn.onnx`/`baseline.joblib` are in the
  repo (they're `git add -f`'d).
- **Done when:** the public Koyeb URL serves the demo.

## S8 — Docs, URL, polish
- Fill the **live-URL placeholders** (README top line + the web-demo comment) with
  the Koyeb URL. Add one line to the README **Web demo** section: hosted on Koyeb
  (onnxruntime), since HF gates free Gradio behind PRO.
- Note in `docs/huggingface-space.md` that the HF path needs PRO; Koyeb is the free
  route. The in-app GitHub link is already correct.
- Optional: delete/pause the broken HF Space (or leave it), and record the
  onnxruntime version + opset in `cnn.md` §7 (serve consistency) alongside the
  existing invariant.
- **Done when:** README/URLs point at the live Koyeb demo and `pytest` is green.

---

### Test matrix (additions)
```
tests/test_inference.py
  test_cnn_onnx_matches_keras     # S3 — parity: onnx argmax == keras argmax, probs close
  test_predict_stages_cnn_smoke   # S3 — adapted to the onnx backend
```

### Rollback
Because nothing was deleted, reverting to the TF serve path is: un-comment the TF
lines in `predict_stages` and `app.main()`, and restore `tensorflow` in
`requirements.txt`. The Keras model and loaders are still present.
