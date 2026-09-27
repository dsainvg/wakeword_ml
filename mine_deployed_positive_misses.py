"""Mine missed positives in the *deployed* configuration, on fresh keyword material.

Two gaps this targets, both measured on v11 at its 0.70 operating point:

  * 9.2% of CLEAN held-out keyword utterances are missed. In-corpus mining finds only
    68 clean misses in 40,000 training positives, so this is a generalisation gap, not a
    data-quantity one -- it needs keyword material the model has not trained on.
  * Streaming recall falls to 53.5% at 0 dB SNR, the regime agents.md 11 flags as the
    weakest.

So: synthesise a fresh base pool (different rate/pitch draws from the evaluation set,
which is left untouched for measurement), splice each base into a 1 s window at a random
offset over a real noise bed, score, and keep what the model gets wrong.
"""
import argparse

import numpy as np
import soundfile as sf

import jax
import jax.numpy as jnp
from flax import serialization

import build_corpus_v2 as B
from features import AudioFeatureExtractor
from models import get_model

SR = 16000
WIN = 16000


def load_scorer(ckpt, arch):
    with open(ckpt, "rb") as f:
        state = serialization.msgpack_restore(f.read())
    model = get_model(arch, num_classes=2)
    params = jax.tree_util.tree_map(jnp.asarray, state["params"])
    thr = float(state.get("threshold", 0.85))

    @jax.jit
    def score(x):
        return jax.nn.softmax(model.apply({"params": params}, x, train=False), axis=-1)[:, 1]

    return score, thr


def splice(base, bed, rng, snr_db):
    """Places base at a random offset in a 1 s window over a real noise bed at snr_db."""
    out = np.zeros(WIN, np.float32)
    if bed is not None and len(bed) >= WIN:
        st = int(rng.integers(0, len(bed) - WIN + 1))
        out[:] = bed[st:st + WIN]
    if base is None or len(base) == 0:
        return out
    if len(base) > WIN:
        base = base[:WIN]
    pos = int(rng.integers(0, max(1, WIN - len(base) + 1)))
    seg = np.zeros(WIN, np.float32)
    seg[pos:pos + len(base)] = base
    sp = np.sqrt(np.mean(seg ** 2)) + 1e-9
    npow = np.sqrt(np.mean(out ** 2)) + 1e-12
    # snr_db is a SIGNAL-TO-NOISE ratio, so the signal sits that many dB ABOVE the bed.
    gain = (sp / npow) * (10 ** (snr_db / 20.0))
    out = out + seg * gain
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.35            # the corpus level policy: -50..-14 dBFS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="bcconformer_v3")
    ap.add_argument("--bases", required=True, help="fresh keyword pool (not the eval set)")
    ap.add_argument("--noise_index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_base", type=int, default=6)
    ap.add_argument("--snr", type=float, nargs="+", default=[99.0, 20.0, 12.0, 6.0, 0.0])
    ap.add_argument("--keep", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    score, thr = load_scorer(args.model, args.arch)
    fx = AudioFeatureExtractor()
    nb = B.NoiseBank(index_csv=args.noise_index)
    bases = [np.asarray(w, dtype=np.float32) for w in
             np.load(args.bases, allow_pickle=True)]
    bases = [b for b in bases if len(b) > 800]
    rng = np.random.default_rng(args.seed)
    print(f"[mine] {len(bases)} fresh bases x {args.per_base} variants, "
          f"threshold {thr:.3f}", flush=True)

    specs, probs, snrs = [], [], []
    for b in bases:
        for _ in range(args.per_base):
            snr = float(args.snr[int(rng.integers(0, len(args.snr)))])
            bed, _ = (None, None) if snr > 90 else nb.sample_slice(rng)
            wav = splice(b, bed, rng, snr)
            specs.append(fx.compute_spectrogram(wav))
            probs.append(-1.0)
            snrs.append(snr)

    X = np.asarray(specs, dtype=np.float32)[:, :, :, None]
    p = np.array(score(jnp.asarray(X)))
    snrs = np.asarray(snrs, np.float32)
    miss = p < thr
    print(f"[mine] {len(p):,} variants, {miss.sum():,} missed ({100*miss.mean():.1f}%)",
          flush=True)
    for s in sorted(set(snrs.tolist())):
        m = snrs == s
        print(f"    snr {s:>5.0f} dB   miss {100*(p[m] < thr).mean():5.1f}%   "
              f"median conf {np.median(p[m]):.3f}")

    Xm, pm, sm = X[miss], p[miss], snrs[miss]
    if len(Xm) > args.keep:
        keep = np.argsort(pm)[:args.keep]
        Xm, pm, sm = Xm[keep], pm[keep], sm[keep]
    np.savez_compressed(args.out, X=Xm.astype(np.float16),
                        y=np.ones(len(Xm), dtype=np.int32), probs=pm, snr=sm)
    print(f"[mine] kept {len(Xm):,} missed deployed positives -> {args.out}")


if __name__ == "__main__":
    main()
