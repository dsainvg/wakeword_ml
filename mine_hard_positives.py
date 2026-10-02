"""Mine hard POSITIVES: the keyword windows the current model scores below threshold.

The hard-negative half of this project (mine_speech_fa.py) took continuous-speech
false alarms from 11-15/hour to zero, then mining the remaining soundscape alarms took
the total below 0.3/hour. The mirror-image failure is recall: at the 0.70 operating
point v11 still misses 9.2% of *clean* held-out keyword utterances and 22% of the
deployed streaming trials.

This finds the positives the model gets wrong and writes them as a fine-tune set.
Sourced from the training split, not the validation or evaluation split, so the
held-out numbers stay honest; the eval keyword bases are never read here.
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from models import get_model


def load_scorer(ckpt, arch, threshold):
    with open(ckpt, "rb") as f:
        state = serialization.msgpack_restore(f.read())
    model = get_model(arch, num_classes=2)
    params = jax.tree_util.tree_map(jnp.asarray, state["params"])
    thr = float(threshold if threshold >= 0 else state.get("threshold", 0.85))

    @jax.jit
    def score(x):
        return jax.nn.softmax(model.apply({"params": params}, x, train=False), axis=-1)[:, 1]

    return score, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="bcconformer_v3")
    ap.add_argument("--corpus", required=True, help="the corpus to mine positives from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="score below which a positive counts as missed; -1 = checkpoint")
    args = ap.parse_args()

    score, thr = load_scorer(args.model, args.arch, args.threshold)
    d = np.load(args.corpus, allow_pickle=True)
    X, y, b = d["X_train"], d["y_train"], d["b_train"]
    pos = np.flatnonzero(y == 1)
    print(f"[mine] {len(pos):,} training positives, operating threshold {thr:.3f}", flush=True)

    out_idx, out_p = [], []
    for i in range(0, len(pos), args.batch):
        idx = pos[i:i + args.batch]
        x = np.asarray(X[idx], dtype=np.float32)
        p = np.array(score(jnp.asarray(x)))
        miss = p < thr
        if miss.any():
            out_idx.append(idx[miss])
            out_p.append(p[miss])
        done = min(i + args.batch, len(pos))
        print(f"[mine] {done:,}/{len(pos):,}  missed {sum(len(a) for a in out_idx):,}",
              flush=True)

    idx = np.concatenate(out_idx)
    p = np.concatenate(out_p)
    order = np.argsort(p)                       # worst first
    idx, p = idx[order], p[order]
    if len(idx) > args.keep:
        idx, p = idx[:args.keep], p[:args.keep]

    Xh = np.asarray(X[idx], dtype=np.float16)
    if Xh.ndim == 5:                      # corpus X is already (n, 49, 40, 1)
        Xh = Xh[..., 0]
    np.savez_compressed(args.out, X=Xh, y=np.ones(len(Xh), dtype=np.int32), probs=p,
                        bucket=np.asarray(b[idx]))
    print(f"[mine] kept {len(Xh):,} missed positives  p: max {p.max():.3f} "
          f"median {np.median(p):.3f}  -> {args.out}")

    buckets, counts = np.unique(b[idx], return_counts=True)
    for bk, c in sorted(zip(buckets, counts), key=lambda t: -t[1]):
        print(f"    {bk:<32} {c:>6,}")


if __name__ == "__main__":
    main()
