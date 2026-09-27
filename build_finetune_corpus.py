"""Assemble a fine-tune corpus: mined continuous-speech false alarms + anchors.

The mined windows are the deployment negative class the bucketed corpus never
produced. Training on them alone would trade recall away, so the set also carries
positives (to hold recall) and a slice of the ordinary corpus negatives (so the model
does not simply learn "reject continuous speech"). Bucket labels reuse the existing
taxonomy names so per-bucket logging still works: the mined windows are tagged
N7_real_two_noise_hard's natural-speech sibling, and the anchors keep their own.
"""
import argparse
import os

import numpy as np


def take(rng, arr, n):
    if n <= 0 or len(arr) == 0:
        return arr[:0]
    idx = rng.choice(len(arr), min(n, len(arr)), replace=False)
    return arr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the bucketed corpus to draw anchors from")
    ap.add_argument("--hard", nargs="*", default=[],
                    help="mined hard-negative npz files (continuous speech, soundscapes)")
    ap.add_argument("--hard_pos", nargs="*", default=[],
                    help="mined hard-POSITIVE npz files (missed keyword windows). Without "
                         "these a fine-tune can only trade recall for a lower false-alarm "
                         "rate, which is what iterations 3 and 4 kept doing.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_hard", type=int, default=12000)
    ap.add_argument("--n_hard_pos", type=int, default=12000)
    ap.add_argument("--n_pos", type=int, default=9000)
    ap.add_argument("--n_neg", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = np.load(args.base, allow_pickle=True)
    Xt, yt = base["X_train"], base["y_train"]
    bt = base["b_train"]
    pos_idx = np.flatnonzero(yt == 1)
    neg_idx = np.flatnonzero(yt == 0)
    print(f"[ft] base corpus: {len(yt):,} train ({len(pos_idx):,} pos / {len(neg_idx):,} neg)")

    X_pos = take(rng, Xt[pos_idx], args.n_pos)
    X_neg = take(rng, Xt[neg_idx], args.n_neg)
    b_pos = take(rng, bt[pos_idx], args.n_pos)
    b_neg = take(rng, bt[neg_idx], args.n_neg)

    X_hard, y_hard = [], []
    for f in args.hard:
        h = np.load(f, allow_pickle=True)
        X_hard.append(h["X"])
        y_hard.append(np.zeros(len(h["X"]), dtype=np.int32))
        print(f"[ft] hard neg  {os.path.basename(f)}: {len(h['X']):,}")
    X_hard = np.concatenate(X_hard, axis=0) if X_hard else np.zeros((0,) + Xt.shape[1:],
                                                                    dtype=np.float16)
    y_hard = np.concatenate(y_hard, axis=0) if len(y_hard) else np.zeros(0, dtype=np.int32)
    if len(X_hard) > args.n_hard:
        keep = rng.choice(len(X_hard), args.n_hard, replace=False)
        X_hard, y_hard = X_hard[keep], y_hard[keep]

    X_hp, b_hp = [], []
    for f in args.hard_pos:
        h = np.load(f, allow_pickle=True)
        X_hp.append(h["X"])
        bk = h["bucket"] if "bucket" in h.files else None
        b_hp.append(np.asarray(bk if bk is not None else
                               ["MINE_deployed_miss"] * len(h["X"])))
        print(f"[ft] hard pos  {os.path.basename(f)}: {len(h['X']):,}")
    if X_hp:
        X_hp = np.concatenate(X_hp, axis=0)
        b_hp = np.concatenate(b_hp, axis=0)
        if len(X_hp) > args.n_hard_pos:
            keep = rng.choice(len(X_hp), args.n_hard_pos, replace=False)
            X_hp, b_hp = X_hp[keep], b_hp[keep]
    else:
        X_hp = np.zeros((0,) + Xt.shape[1:], dtype=np.float16)
        b_hp = np.array([], dtype=bt.dtype)

    X = np.concatenate([X_pos, X_neg, X_hard, X_hp], axis=0)
    y = np.concatenate([np.ones(len(X_pos), dtype=np.int32),
                        np.zeros(len(X_neg), dtype=np.int32), y_hard,
                        np.ones(len(X_hp), dtype=np.int32)], axis=0)
    # A dedicated bucket so the trainer reports this regime separately instead of
    # letting it hide inside the corpus average.
    b = np.concatenate([b_pos, b_neg,
                        np.array(["MINE_negative"] * len(X_hard), dtype=bt.dtype),
                        b_hp], axis=0)
    snr = np.zeros(len(X), np.float32)
    lvl = np.zeros(len(X), np.float32)
    g = np.zeros(len(X), np.float32)

    order = rng.permutation(len(X))
    X, y, b, snr, lvl = X[order], y[order], b[order], snr[order], lvl[order]

    n_val = max(600, len(X) // 20)
    val_idx = np.arange(n_val)
    tr_idx = np.arange(n_val, len(X))

    out = {
        "X_train": X[tr_idx].astype(np.float16), "y_train": y[tr_idx],
        "g_train": g[tr_idx], "snr_train": snr[tr_idx], "b_train": b[tr_idx],
        "lvl_train": lvl[tr_idx],
        "X_val": X[val_idx].astype(np.float16), "y_val": y[val_idx],
        "g_val": g[val_idx], "snr_val": snr[val_idx], "b_val": b[val_idx],
        "lvl_val": lvl[val_idx],
    }
    np.savez(args.out, **out)
    print(f"[ft] wrote {args.out}: train {out['X_train'].shape} "
          f"({int((y[tr_idx] == 1).sum()):,} pos / {int((y[tr_idx] == 0).sum()):,} neg)  "
          f"val {out['X_val'].shape}")


if __name__ == "__main__":
    main()
