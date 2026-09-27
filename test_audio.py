"""
Interactive Audio Tester for Keyword: "Amaze"
--------------------------------------------
Evaluates single WAV audio files or batches against the trained KWS model:
- Computes 40-band Log-Mel spectrogram
- Runs inference and displays detection probability, binary decision, and latency
"""

import sys
import wave
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from features import AudioFeatureExtractor
from models import get_model
from flax import serialization


def load_model(checkpoint_path: str = "best_kws_model.flax", arch: str = "bcresnet"):
    model = get_model(arch, num_classes=2)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)

    with open(checkpoint_path, "rb") as f:
        encoded = f.read()

    template = {
        "params": variables['params'],
        "arch": arch,
        "param_count": 0,
        "val_acc": 0.0,
        "tpr": 0.0,
        "fpr": 0.0
    }
    state_dict = serialization.from_bytes(template, encoded)
    params = state_dict["params"]

    @jax.jit
    def predict_fn(x):
        logits = model.apply({'params': params}, x, train=False)
        probs = jax.nn.softmax(logits, axis=-1)
        return probs[0, 1]

    # Warmup
    _ = predict_fn(dummy_input)
    return predict_fn


def test_file(predict_fn, extractor, wav_path: str, threshold: float = 0.85):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if sr != 16000:
        print(f"[Warning] Audio sample rate is {sr} Hz (expected 16000 Hz).")

    spec = extractor.compute_spectrogram(audio)
    tensor = jnp.expand_dims(jnp.expand_dims(jnp.array(spec), axis=0), axis=-1)

    t0 = np.datetime64("now")
    prob = float(predict_fn(tensor))
    t1 = np.datetime64("now")
    lat_ms = (t1 - t0) / np.timedelta64(1, "ms")

    is_amaze = prob >= threshold
    status = "[DETECTED 'AMAZE']" if is_amaze else "[NOT DETECTED]   "
    bar = "=" * int(prob * 20)

    print(f"{status} | Conf: {prob:6.3f} [{bar:<20}] | File: {wav_path}")
    return is_amaze, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WAV audio file against 'Amaze' KWS model")
    parser.add_argument("--file", type=str, default="dataset/positive/pos_0001.wav", help="Path to WAV file")
    parser.add_argument("--thresh", type=float, default=0.85, help="Confidence threshold")
    args = parser.parse_args()

    extractor = AudioFeatureExtractor()
    predict_fn = load_model("best_kws_model.flax", "bcresnet")

    print("\n" + "=" * 75)
    print(" Testing Single Audio File against 'Amaze' Keyword Model")
    print("=" * 75)
    test_file(predict_fn, extractor, args.file, args.thresh)

    print("\nTesting 5 random positive ('Amaze') samples:")
    for i in [5, 12, 27, 45, 99]:
        test_file(predict_fn, extractor, f"dataset/positive/pos_{i:04d}.wav", args.thresh)

    print("\nTesting 5 random negative (confusable / noise) samples:")
    for i in [3, 18, 50, 110, 240]:
        test_file(predict_fn, extractor, f"dataset/negative/neg_{i:04d}.wav", args.thresh)
    print("=" * 75 + "\n")
