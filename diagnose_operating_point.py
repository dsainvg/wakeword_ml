"""
Operating-Point Diagnostic
---------------------------
Loads a trained checkpoint and reports the full TPR / FPR tradeoff curve so the
streaming threshold can be chosen from evidence instead of a hardcoded 0.85.

Usage:
    python diagnose_operating_point.py --model best_50k_kws_model.flax \
        --arch bcconformer_50k --data dataset_mega.npz
"""

import argparse
import glob
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from sklearn.metrics import roc_auc_score

from features import AudioFeatureExtractor
from models import get_model

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000


def load_predict_fn(checkpoint_path: str, arch: str):
    with open(checkpoint_path, "rb") as f:
        data = f.read()
    checkpoint = serialization.msgpack_restore(data)
    model = get_model(arch, num_classes=2)
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])

    @jax.jit
    def predict_batch(x):
        logits = model.apply({"params": params}, x, train=False)
        return jax.nn.softmax(logits, axis=-1)[:, 1]

    return predict_batch, checkpoint


def score_spectrograms(predict_fn, specs: np.ndarray, batch_size: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(specs), batch_size):
        batch = specs[i:i + batch_size]
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=-1)
        out.extend(np.array(predict_fn(jnp.array(batch.astype(np.float32)))))
    return np.array(out)


def load_noise_bank_probs(predict_fn, extractor, noise_dir: str = "noise_bank") -> np.ndarray:
    probs = []
    for w in sorted(glob.glob(os.path.join(noise_dir, "*.wav"))):
        try:
            import soundfile as sf
            from scipy import signal
            d, sr = sf.read(w)
            if d.ndim > 1:
                d = d[:, 0]
            if sr != SAMPLE_RATE:
                d = signal.resample(d, int(len(d) * SAMPLE_RATE / sr))
            n_chunks = min(20, max(1, len(d) // TOTAL_SAMPLES))
            for c in range(n_chunks):
                chunk = d[c * TOTAL_SAMPLES:(c + 1) * TOTAL_SAMPLES]
                if len(chunk) < TOTAL_SAMPLES:
                    chunk = np.pad(chunk, (0, TOTAL_SAMPLES - len(chunk)))
                spec = extractor.compute_spectrogram(chunk)
                probs.append(float(predict_fn(jnp.expand_dims(jnp.expand_dims(jnp.array(spec), 0), -1))[0]))
        except Exception:
            pass
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="best_50k_kws_model.flax")
    ap.add_argument("--arch", type=str, default="bcconformer_50k")
    ap.add_argument("--data", type=str, default="dataset_mega.npz")
    ap.add_argument("--fpr_budget", type=float, default=0.0012,
                    help="Max acceptable FPR on the validation split")
    args = ap.parse_args()

    predict_fn, ckpt = load_predict_fn(args.model, args.arch)
    extractor = AudioFeatureExtractor()

    d = np.load(args.data)
    X_val, y_val = d["X_val"], d["y_val"]
    probs = score_spectrograms(predict_fn, X_val)
    auc = roc_auc_score(y_val, probs)

    pos = probs[y_val == 1]
    neg = probs[y_val == 0]

    print("=" * 72)
    print(f" OPERATING-POINT DIAGNOSTIC  {args.model}  [{args.arch}]")
    print("=" * 72)
    print(f" Checkpoint epoch={ckpt.get('epoch','?')}  stored_auc={ckpt.get('val_auc',0)*100:.2f}%  "
          f"stored_tpr={ckpt.get('tpr',0)*100:.2f}%  stored_fpr={ckpt.get('fpr',0)*100:.2f}%")
    print(f" Recomputed Val AUC: {auc*100:.3f}%   (pos n={len(pos)}, neg n={len(neg)})")
    print("-" * 72)
    print(f"{'Thresh':>8} | {'TPR':>8} | {'FPR':>8} | {'F1':>7} | {'NoiseFA':>8}")
    print("-" * 72)

    noise_probs = load_noise_bank_probs(predict_fn, extractor)

    rows = []
    for t in np.arange(0.05, 0.99, 0.05):
        tpr = float(np.mean(pos >= t))
        fpr = float(np.mean(neg >= t))
        prec = tpr / max(1e-9, tpr * float(np.mean(y_val == 1)) + fpr * float(np.mean(y_val == 0)))
        f1 = 2 * prec * tpr / max(1e-9, prec + tpr)
        nfa = int(np.sum(noise_probs >= t))
        rows.append((t, tpr, fpr, f1, nfa))
        print(f"{t:8.2f} | {tpr*100:7.2f}% | {fpr*100:7.3f}% | {f1*100:6.2f}% | {nfa:5d}/{len(noise_probs)}")

    # Best achievable TPR under the FPR budget
    ok = [r for r in rows if r[2] <= args.fpr_budget]
    if ok:
        best = max(ok, key=lambda r: r[1])
        print("-" * 72)
        print(f" Best TPR with FPR <= {args.fpr_budget*100:.3f}%:  "
              f"TPR={best[1]*100:.2f}% @ threshold {best[0]:.2f}  (FPR={best[2]*100:.3f}%)")
    else:
        print("-" * 72)
        print(f" No threshold in the sweep meets FPR <= {args.fpr_budget*100:.3f}%")

    # Empirical EER-ish reference
    print(f"\n Noise bank: {len(noise_probs)} slices, max conf = {noise_probs.max():.4f}, "
          f"99.9th pct = {np.percentile(noise_probs, 99.9):.4f}")
    print(f" Positives : 5th pct = {np.percentile(pos,5):.4f}, 10th pct = {np.percentile(pos,10):.4f}, "
          f"median = {np.median(pos):.4f}")
    print(f" Negatives : 99.9th pct = {np.percentile(neg,99.9):.4f}, max = {neg.max():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
