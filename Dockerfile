# Serve-only image for the Gradio sleep-staging demo (Koyeb / Render, free tier).
#
# TensorFlow-free by design: the CNN runs from models/cnn.onnx via onnxruntime
# (see cnn-to-onnx.md), so this image fits a free 512 MB host. Every serve
# dependency in requirements.txt ships a manylinux 3.11 wheel (numpy, scipy,
# pandas, pyarrow, scikit-learn, matplotlib, onnxruntime, gradio, mne), so no
# compiler/build-essential is needed on the base image.

FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image layer
# - PYTHONUNBUFFERED: logs stream straight to the container stdout (build logs)
# - MPLBACKEND=Agg: matplotlib renders the hypnogram/confusion-matrix PNGs
#   headless (no GUI toolkit in the slim base)
# - GRADIO_ANALYTICS_ENABLED=False: no phone-home from the demo
# - PORT: default; Koyeb/Render override this at runtime and app.py binds it
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

WORKDIR /app

# Install deps first so the (slow) pip layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo. .dockerignore keeps out data/, venvs, caches,
# reports/, and the unused models/cnn.keras (serve needs only cnn.onnx +
# baseline.joblib), so the build context and image stay small.
COPY . .

# app.py adds ./src to sys.path itself; no `pip install .` (avoids dragging the
# pipeline's pyspark/typer pins back in). main() binds 0.0.0.0:$PORT (S5).
EXPOSE 7860
CMD ["python", "app.py"]
