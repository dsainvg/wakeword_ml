"""Mine continuous-speech false alarms as hard negatives, batched on the GPU.

Why this exists: agents.md 9.1 measured that validation FPR does not predict deployment
false alarms, and 11 names the missing ingredient -- natural, un-normalised long-form
speech as a negative class. Sweeping the run-2 checkpoint showed exactly that failure:
0.48 FA/hour on soundscapes but 11-15 FA/hour on continuous LibriSpeech. The corpus
negatives are distorted 1 s slices, so the model never saw the deployment distribution.

This scans continuous speech with the *current* model at the deployment stride, keeps
the windows it actually fires on, and writes them as a fine-tune corpus. That is the
negative class the training set was missing.
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf
from scipy import signal

import jax
import jax.numpy as jnp
from flax import serialization

from features import AudioFeatureExtractor
from models import get_model

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000


def load_scorer(ckpt, arch):
    with open(ckpt, "rb") as f:
        state = serialization.msgpack_restore(f.read())
    model = get_model(arch, num_classes=2)
    params = jax.tree_util.tree_map(jnp.asarray, state["params"])

    @jax.jit
    def score(x):
        return jax.nn.softmax(model.apply({"params": params}, x, train=False), axis=-1)[:, 1]

    return score


def load_wav(path):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    if sr != SAMPLE_RATE:
        d = signal.resample(d, int(round(len(d) * SAMPLE_RATE / sr)))
    return d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="bcconformer_v3")
    ap.add_argument("--out", default="/home/user123/kws_run/hard_speech.npz")
    ap.add_argument("--speech_dir", default="/home/user123/wakeword_ml/speech_corpus/LibriSpeech")
    ap.add_argument("--files", type=int, default=700)
    ap.add_argument("--stride_ms", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="keep windows scoring at or above this; the sweep says the "
                         "operating point will sit near 0.94, so 0.40 keeps a margin")
    ap.add_argument("--noise_index", default="",
                    help="index.csv of the soundscape bank; when given, the same mining "
                         "runs over it and the windows are tagged MINE_soundscape. "
                         "Fine-tuning on speech negatives alone trades soundscape false "
                         "alarms for recall, so both sources have to be mined together.")
    ap.add_argument("--noise_threshold", type=float, default=0.50)
    ap.add_argument("--keep_top", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    score = load_scorer(args.model, args.arch)
    fx = AudioFeatureExtractor()

    flacs = []
    for root, _, files in os.walk(args.speech_dir):
        flacs += [os.path.join(root, f) for f in files if f.endswith(".flac")]
    rng = np.random.default_rng(args.seed)
    if args.files and len(flacs) > args.files:
        flacs = [flacs[i] for i in rng.choice(len(flacs), args.files, replace=False)]
    print(f"[mine] {len(flacs)} continuous-speech files, stride {args.stride_ms} ms", flush=True)

    stride = int(args.stride_ms * SAMPLE_RATE / 1000)
    specs, probs, src = [], [], []
    pending, pending_src = [], []
    t0 = time.time()
    scanned = 0
    hours = 0.0

    def flush():
        if not pending:
            return
        x = np.asarray(pending, dtype=np.float32)[:, :, :, None]
        p = np.array(score(jnp.asarray(x)))
        for s, pp, k in zip(pending, p, pending_src):
            if pp >= args.threshold:
                specs.append(s)
                probs.append(float(pp))
                src.append(k)
        pending.clear()
        pending_src.clear()

    for fi, path in enumerate(flacs):
        try:
            audio = load_wav(path)
        except Exception:
            continue
        hours += len(audio) / SAMPLE_RATE / 3600
        n = max(0, (len(audio) - TOTAL_SAMPLES) // stride)
        for s in range(n):
            chunk = audio[s * stride: s * stride + TOTAL_SAMPLES]
            if np.sqrt(np.mean(chunk ** 2)) < 0.010:      # skip near-silence
                continue
            pending.append(fx.compute_spectrogram(chunk))
            pending_src.append(fi)
            scanned += 1
            if len(pending) >= args.batch:
                flush()
        if (fi + 1) % 100 == 0:
            print(f"[mine] {fi + 1}/{len(flacs)} files  {scanned:,} windows  "
                  f"hard={len(specs):,}  {hours:.2f} h  {time.time() - t0:.0f}s", flush=True)
    flush()

    if not specs:
        print("[mine] nothing above threshold; lower --threshold")
        return
    X = np.asarray(specs, dtype=np.float16)[:, :, :, None]
    p = np.asarray(probs, dtype=np.float32)
    if len(X) > args.keep_top:
        idx = np.argsort(-p)[:args.keep_top]
        X, p, src = X[idx], p[idx], [src[i] for i in idx]
    tag = np.array(["MINE_speech_continuous"] * len(X))

    np.savez_compressed(args.out, X=X, y=np.zeros(len(X), dtype=np.int32), probs=p,
                        src=np.asarray(src, dtype=np.int32))
    print(f"[mine] kept {len(X):,} hard speech negatives  p: max {p.max():.3f} "
          f"mean {p.mean():.3f}  -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)  from {hours:.2f} h scanned")

    if args.noise_index:
        mine_soundscapes(score, fx, args)


def mine_soundscapes(score, fx, args):
    """Same procedure over the soundscape bank: every window the model actually fires on."""
    import csv

    paths = []
    with open(args.noise_index, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append((row["path"].replace("/", os.sep), row["class"]))
    print(f"[mine] {len(paths)} soundscape clips", flush=True)

    stride = int(args.stride_ms * SAMPLE_RATE / 1000)
    specs, probs, src = [], [], []
    pending, pending_src = [], []
    hours = 0.0
    t0 = time.time()

    def flush():
        if not pending:
            return
        x = np.asarray(pending, dtype=np.float32)[:, :, :, None]
        p = np.array(score(jnp.asarray(x)))
        for s, pp, k in zip(pending, p, pending_src):
            if pp >= args.noise_threshold:
                specs.append(s)
                probs.append(float(pp))
                src.append(k)
        pending.clear()
        pending_src.clear()

    for i, (path, cls) in enumerate(paths):
        try:
            audio = load_wav(path)
        except Exception:
            continue
        hours += len(audio) / SAMPLE_RATE / 3600
        for s in range(max(0, (len(audio) - TOTAL_SAMPLES) // stride)):
            chunk = audio[s * stride: s * stride + TOTAL_SAMPLES]
            if np.sqrt(np.mean(chunk ** 2)) < 0.005:
                continue
            pending.append(fx.compute_spectrogram(chunk))
            pending_src.append(cls)
            if len(pending) >= args.batch:
                flush()
        if (i + 1) % 400 == 0:
            print(f"[mine] {i + 1}/{len(paths)} clips  hard={len(specs):,}  "
                  f"{hours:.2f} h  {time.time() - t0:.0f}s", flush=True)
    flush()

    if not specs:
        print("[mine] no soundscape windows above threshold")
        return
    X = np.asarray(specs, dtype=np.float16)[:, :, :, None]
    p = np.asarray(probs, dtype=np.float32)
    out = args.out.replace(".npz", "_soundscape.npz")
    np.savez_compressed(out, X=X, y=np.zeros(len(X), dtype=np.int32), probs=p,
                        src=np.asarray(src))
    print(f"[mine] kept {len(X):,} hard soundscape negatives  p: max {p.max():.3f} "
          f"mean {p.mean():.3f}  -> {out}  from {hours:.2f} h scanned")


if __name__ == "__main__":
    main()
