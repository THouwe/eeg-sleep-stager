"""Convert the trained Keras CNN to ONNX for a TensorFlow-free serve path.

Part of the CNN→ONNX slim-down (see ``cnn-to-onnx.md``, S1). Training stays on
TensorFlow and keeps writing ``models/cnn.keras``; this script is a one-off
conversion tool that reads that file and writes ``models/cnn.onnx`` so the web
demo can run inference with onnxruntime alone (a fraction of TF's size/RAM) and
fit a free 512 MB host.

**Environment:** run this in the *pipeline* ``.venv`` (the one with TensorFlow).
It needs ``tf2onnx`` + ``onnxruntime`` alongside TF::

    .venv/Scripts/python.exe -m pip install tf2onnx onnxruntime
    .venv/Scripts/python.exe scripts/keras_to_onnx.py

The exported graph has a **dynamic batch axis** (``(None, 3000, 1)``) and a fixed
opset so any epoch count stages in one ``session.run``. After writing, we reload
the ONNX in onnxruntime and sanity-check the output shape ``(N, 5)`` and that the
softmax rows sum to 1.

Recorded I/O names (read them from the session at serve time, don't hard-code):

    input  name: "epoch_signal"     shape (None, 3000, 1)  float32
    output name: "stage"            shape (None, 5)        float32

The invariant (single channel, µV, per-recording z-score, 3000-sample epochs) is
unchanged: ONNX replaces only the matrix multiply, not the inputs to it.

Usage::

    python scripts/keras_to_onnx.py
    python scripts/keras_to_onnx.py --keras models/cnn.keras --out models/cnn.onnx --opset 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Fixed opset for the export. 13+ covers every op in the Conv1D/BN/GAP/softmax
# stack and is widely supported by onnxruntime-cpu wheels.
DEFAULT_OPSET = 13

# The trained CNN's fixed epoch length; the batch axis stays dynamic (None).
EPOCH_SAMPLES = 3000


def convert(keras_path: Path, out_path: Path, opset: int = DEFAULT_OPSET) -> tuple[str, str]:
    """Load ``keras_path`` and export it to ONNX at ``out_path``.

    Uses the programmatic ``tf2onnx.convert.from_keras`` API with an explicit
    ``TensorSpec`` so the batch axis is dynamic. Returns the ONNX graph's
    ``(input_name, output_name)``.
    """
    import tensorflow as tf
    import tf2onnx

    if not keras_path.exists():
        raise FileNotFoundError(
            f"Keras model not found: {keras_path}. Train it first (see README)."
        )

    model = tf.keras.models.load_model(keras_path)
    input_signature = [
        tf.TensorSpec((None, EPOCH_SAMPLES, 1), tf.float32, name="epoch_signal")
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(out_path),
    )
    in_name = model_proto.graph.input[0].name
    out_name = model_proto.graph.output[0].name
    return in_name, out_name


def sanity_check(out_path: Path) -> None:
    """Reload the ONNX in onnxruntime and assert the serve contract holds.

    Runs a small random batch and checks: output shape ``(N, 5)`` and each
    softmax row sums to 1. Prints the I/O names so they can be recorded/verified.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    n = 7
    x = np.random.randn(n, EPOCH_SAMPLES, 1).astype("float32")
    proba = sess.run([out_name], {in_name: x})[0]

    assert proba.shape == (n, 5), f"expected (N, 5), got {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f"softmax rows sum to {row_sums}"

    print(f"[onnx] input name : {in_name!r}")
    print(f"[onnx] output name: {out_name!r}")
    print(f"[onnx] output shape: {proba.shape}, softmax rows sum to 1: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, default=Path("models/cnn.keras"))
    parser.add_argument("--out", type=Path, default=Path("models/cnn.onnx"))
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    in_name, out_name = convert(args.keras, args.out, opset=args.opset)
    print(f"[onnx] wrote {args.out} (input={in_name!r}, output={out_name!r})")
    sanity_check(args.out)


if __name__ == "__main__":
    main()
