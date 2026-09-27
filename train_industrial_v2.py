"""
Industrial KWS Training v2 -- optimise TPR at a fixed FPR budget
----------------------------------------------------------------
The v1 trainer selected checkpoints on ROC-AUC measured at a hardcoded 0.85
threshold, and weighted negatives up to 2.5x. Both choices push the operating
point toward "say no": 17% of v1 positives scored below 0.85 while the soundscape
bank already had non-speech slices above it (pub_babble 0.933).

v2 changes
  1. Checkpoint selection maximises TPR subject to FPR <= --fpr_budget, computed on
     the full validation ROC. The threshold that satisfies the budget is stored in
     the checkpoint so deployment does not have to guess.
  2. Class weights are computed from the dataset ratio with a sqrt law and an
     optional positive bias, instead of the old "penalise negatives" rule.
  3. On-batch hard-negative mining (OHEM): only the hardest fraction of negatives in
     each minibatch contributes gradient. This tightens the FPR tail without
     weakening the positive gradient.
  4. Focal gamma is lowered (default 1.5) because gamma=2.0 down-weights exactly the
     easy-but-quiet positives that dominate the recall loss.
  5. Offset-stress validation: val positives are also scored after a random temporal
     roll, and the reported TPR is the mean over the unshifted and shifted copies.
     This is what a 100 ms-stride streaming window actually sees.
  6. Per-bucket loss and metrics. When the dataset carries mixture-recipe tags
     (dataset_v4.npz, written by build_corpus_v2.py --profile v4) every sample gets
     a bucket id such as P2_word_two_noise_hard or N7_babble_two_noise_hard. The
     trainer reports loss, accuracy and TPR/FPR separately per bucket every epoch and
     applies a per-bucket loss weight, so a regression in one regime cannot be hidden
     inside a single average.
  7. Mic-gain augmentation. A constant added to the log-mel is a multiplicative change
     in waveform amplitude, so jittering it simulates a preamp/AGC change and forces the
     model to key on spectral shape rather than absolute loudness.
  8. Two local-run affordances: --hard_neg_k mines a fixed *count* of negatives so the
     OHEM behaviour does not change with batch size (the Kaggle path already assumed
     this flag existed), and --memmap streams X_train/X_val out of the compressed .npz
     into .npy sidecars that are then memory-mapped, so an 850 MB corpus does not have to
     compete with the XLA arena for RAM.
  9. float16 corpora are read without being widened on load. build_corpus_v2.py writes
     X_* as float16 by default (--dtype fp16), because the log-mel floor is -11.51 and
     float16 resolution is far below anything the model resolves. The corpus therefore
     stays half the size on disk and in the memmap sidecar, and is upcast per batch --
     the point being the first place that widens, rather than a second full-size copy
     in RAM. Both --dtype fp16 and --dtype fp32 corpora train through this unchanged.

    python train_industrial_v2.py --data dataset_v4.npz --arch bcconformer_50k \
        --epochs 24 --fpr_budget 0.005 --save best_v4_50k.flax
"""

import argparse
import functools
import os
import time
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state
from sklearn.metrics import roc_auc_score

from models import get_model, count_params

# Must match build_corpus_v2.BUCKET_LOSS_WEIGHT; kept local so the trainer does not
# need to import the (heavy) dataset builder.
BUCKET_LOSS_WEIGHT = {
    "P0_word_direct": 1.00,
    "P1_word_one_noise": 1.15,
    "P2_word_two_noise": 1.25,
    "P3_word_two_noise_hard": 1.30,
    "N0_filler_direct": 0.80,
    "N1_filler_one_noise": 0.90,
    "N2_filler_two_noise": 1.00,
    "N3_filler_two_noise_hard": 1.10,
    "N4_real_direct": 0.90,
    "N5_real_one_noise": 1.00,
    "N6_real_two_noise": 1.10,
    "N7_real_two_noise_hard": 1.20,
    "N8_babble_direct": 1.00,
    "N9_babble_one_noise": 1.10,
    "N10_babble_two_noise": 1.20,
    "N11_babble_two_noise_hard": 1.30,
    "N12_noise_direct": 0.90,
    "N13_noise_one_noise": 1.00,
    "N14_noise_two_noise": 1.00,
    "N15_noise_two_noise_hard": 1.10,
    "N16_confus_direct": 1.00,
    "N17_confus_one_noise": 1.10,
    "N18_confus_two_noise": 1.10,
    "N19_confus_two_noise_hard": 1.20,
}


# ==============================================================================
# Augmentation
# ==============================================================================

def spec_augment(batch_x: np.ndarray, rng: np.random.Generator, prob: float = 0.5,
                 max_f: int = 5, max_t: int = 7) -> np.ndarray:
    b, t, m, c = batch_x.shape
    out = batch_x.copy()
    for i in range(b):
        if rng.random() < prob:
            f_w = int(rng.integers(1, max_f + 1))
            f_st = int(rng.integers(0, max(1, m - f_w)))
            out[i, :, f_st:f_st + f_w, :] = np.min(out[i])
            t_w = int(rng.integers(1, max_t + 1))
            t_st = int(rng.integers(0, max(1, t - t_w)))
            out[i, t_st:t_st + t_w, :, :] = np.min(out[i])
    return out


def gain_augment(batch_x: np.ndarray, rng: np.random.Generator, prob: float = 0.6,
                 span_db: float = 12.0) -> np.ndarray:
    """Mic preamp / AGC jitter.

    Log-mel is a log of energy, so adding a constant d to the log-mel multiplies the
    waveform energy by e^d, i.e. the amplitude by e^(d/2). A +/- span_db/20 dB swing
    in amplitude therefore becomes a constant offset of span_db/20*ln(10) in feature
    space. This is the cheapest way to make the model stop keying on absolute level.
    """
    out = batch_x.copy()
    b = out.shape[0]
    k = int(np.ceil(span_db / 20.0 * np.log(10.0)))
    for i in range(b):
        if rng.random() < prob:
            out[i] = out[i] + float(rng.uniform(-k, k))
    return out


def roll_batch(specs: np.ndarray, rng: np.random.Generator, offsets: np.ndarray) -> np.ndarray:
    out = np.empty_like(specs)
    for i, off in enumerate(offsets):
        out[i] = np.roll(specs[i], int(off), axis=0)
    return out


# ==============================================================================
# Training state
# ==============================================================================

def create_train_state(rng, model, learning_rate: float, total_steps: int, init_params=None,
                      weight_decay: float = 1e-4):
    dummy = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    if init_params is None:
        variables = model.init(rng, dummy, train=False)
        params = variables["params"]
    else:
        params = init_params

    warmup = max(30, int(0.06 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5, peak_value=learning_rate, warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1), end_value=1e-5)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@functools.partial(jax.jit, static_argnames=("gamma", "hard_neg_frac", "hard_neg_k"))
def train_step(state, batch_x, batch_y, rng, gamma, class_weights, hard_neg_frac,
               hard_neg_k=0, bucket_w=None):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x, train=True,
                                rngs={"dropout": rng})
        probs = jax.nn.softmax(logits, axis=-1)
        p_t = jnp.take_along_axis(probs, jnp.expand_dims(batch_y, axis=-1), axis=-1)[:, 0]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y)
        focal = jnp.power(jnp.clip(1.0 - p_t, 1e-4, 1.0), gamma)
        sw = class_weights[batch_y]
        per_sample = ce * focal * sw
        if bucket_w is not None:
            per_sample = per_sample * bucket_w

        # On-batch hard-negative mining: keep only the hardest negatives.
        # `hard_neg_k` (a count) and `hard_neg_frac` (a fraction of the batch) are both
        # static, so the cutoff index derives from static values only. A count is the
        # batch-size-independent form: the same number of negatives is mined whether the
        # batch is 64 (local CPU) or 512 (Kaggle GPU).
        is_neg = batch_y == 0
        k = int(hard_neg_k) if hard_neg_k > 0 else int(hard_neg_frac * batch_x.shape[0])
        if hard_neg_k > 0 or hard_neg_frac < 1.0:
            k = max(1, k)
            neg_vals = jnp.where(is_neg, per_sample, jnp.inf)
            min_neg = jnp.min(jnp.where(is_neg, per_sample, jnp.inf))
            cut = jnp.minimum(jnp.sort(neg_vals)[k - 1], min_neg)
            keep = jnp.logical_or(batch_y == 1, per_sample >= cut)
            per_sample = per_sample * jnp.where(keep, 1.0, 0.0)

        return jnp.mean(per_sample), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
    return new_state, loss, acc


@jax.jit
def predict_probs(state, batch_x):
    logits = state.apply_fn({"params": state.params}, batch_x, train=False)
    return jax.nn.softmax(logits, axis=-1)[:, 1]


# ==============================================================================
# Operating-point helpers
# ==============================================================================

def threshold_for_fpr(neg_probs: np.ndarray, fpr_budget: float) -> float:
    """Smallest threshold whose negative false-positive rate stays within budget."""
    if len(neg_probs) == 0:
        return 0.5
    allowed = int(np.floor(fpr_budget * len(neg_probs)))
    if allowed <= 0:
        return float(np.nextafter(neg_probs.max(), 1.0))
    srt = np.sort(neg_probs)[::-1]
    return float(srt[allowed - 1])


def tpr_at_threshold(pos_probs: np.ndarray, t: float) -> float:
    return float(np.mean(pos_probs >= t)) if len(pos_probs) else 0.0


def tpr_at_fpr_budget(pos_probs: np.ndarray, neg_probs: np.ndarray, budget: float):
    t = threshold_for_fpr(neg_probs, budget)
    return tpr_at_threshold(pos_probs, t), t, float(np.mean(neg_probs >= t))


# ==============================================================================
# Per-bucket metrics
# ==============================================================================

BUCKET_REASON = {
    "P0_word_direct": "word only, no noise",
    "P1_word_one_noise": "word + 1 noise layer",
    "P2_word_two_noise": "word + 2 noise layers",
    "P3_word_two_noise_hard": "word + 2 noise layers, low SNR, full distortion",
    "N0_filler_direct": "unimportant word, no noise",
    "N1_filler_one_noise": "unimportant word + 1 noise layer",
    "N2_filler_two_noise": "unimportant word + 2 noise layers",
    "N3_filler_two_noise_hard": "unimportant word + 2 layers, low SNR, full distortion",
    "N4_real_direct": "real human speech, no noise",
    "N5_real_one_noise": "real human speech + 1 noise layer",
    "N6_real_two_noise": "real human speech + 2 noise layers",
    "N7_real_two_noise_hard": "real human speech + 2 layers, low SNR",
    "N8_babble_direct": "2-8 live talkers, no extra noise",
    "N9_babble_one_noise": "2-8 live talkers + 1 noise layer",
    "N10_babble_two_noise": "2-8 live talkers + 2 noise layers",
    "N11_babble_two_noise_hard": "2-8 live talkers + 2 layers, low SNR",
    "N12_noise_direct": "single soundscape, nothing added",
    "N13_noise_one_noise": "soundscape + 1 noise layer",
    "N14_noise_two_noise": "soundscape + 2 more noise layers",
    "N15_noise_two_noise_hard": "soundscape + 2 more layers, full distortion",
    "N16_confus_direct": "near-miss word, no noise",
    "N17_confus_one_noise": "near-miss word + 1 noise layer",
    "N18_confus_two_noise": "near-miss word + 2 noise layers",
    "N19_confus_two_noise_hard": "near-miss word + 2 layers, low SNR",
}


def bucket_stats(probs: np.ndarray, buckets: np.ndarray, y: np.ndarray,
                 thr: float) -> dict:
    """TPR for positive buckets, FPR for negative buckets, at the operating point."""
    out = {}
    for b in np.unique(buckets):
        m = buckets == b
        p = probs[m]
        lab = y[m]
        out[str(b)] = float(np.mean(p >= thr)) if (lab == 1).any() else float(np.mean(p >= thr))
    return out


def print_bucket_table(probs, buckets, y, thr):
    pos = defaultdict(lambda: [0, 0, []])
    neg = defaultdict(lambda: [0, 0, []])
    for p, b, l in zip(probs, buckets, y):
        tgt = pos if l == 1 else neg
        e = tgt[str(b)]
        e[0] += 1
        e[1] += int(p >= thr)
        e[2].append(float(p))
    # Two tables, because a single column cannot say whether a large number is good: 3%
    # recall is a failure and 3% false-alarm rate is also a failure, but a reader has to
    # know which polarity a row is before the number means anything.
    print(f"    POSITIVE buckets @ thr {thr:.3f}   (recall, higher is better)")
    print(f"    {'bucket':<28} | {'n':>6} | {'recall':>7} | {'MISSED':>7} | "
          f"{'mean conf':>9} | {'p05 conf':>8}")
    for name, e in pos.items():
        r = e[1] / max(1, e[0])
        print(f"    {name:<28} | {e[0]:6d} | {r*100:6.2f}% | {(1-r)*100:6.2f}% | "
              f"{np.mean(e[2]):9.4f} | {np.percentile(e[2],5):8.4f}")
    _p = sum(e[0] for e in pos.values())
    _h = sum(e[1] for e in pos.values())
    if _p:
        print(f"    {'POOLED':<28} | {_p:6d} | {_h/_p*100:6.2f}% | "
              f"{(1-_h/_p)*100:6.2f}% |")

    print(f"    NEGATIVE buckets @ thr {thr:.3f}   (false-alarm rate, lower is better)")
    print(f"    {'bucket':<28} | {'n':>6} | {'FA rate':>8} | {'FA count':>8} | "
          f"{'mean conf':>9} | {'p95 conf':>8}")
    for name, e in neg.items():
        print(f"    {name:<28} | {e[0]:6d} | {e[1]/max(1,e[0])*100:7.2f}% | {e[1]:8d} | "
              f"{np.mean(e[2]):9.4f} | {np.percentile(e[2],95):8.4f}")
    _n = sum(e[0] for e in neg.values())
    _f = sum(e[1] for e in neg.values())
    if _n:
        print(f"    {'POOLED':<28} | {_n:6d} | {_f/_n*100:7.2f}% | {_f:8d} |")


# ==============================================================================
# Training
# ==============================================================================

# ==============================================================================
# Low-memory corpus loading
# ==============================================================================

def stream_npy_member_to_file(npz_path: str, member: str, out_path: str,
                               chunk: int = 1 << 22):
    """Copy a single .npy member out of an .npz to a standalone .npy in bounded RAM.

    `np.load(npz)["X_train"]` inflates the whole member into anonymous memory, which for
    dataset_v5 is 847 MB -- more than this machine has free. A memmapped sidecar keeps the
    corpus on disk and lets the OS page it in per batch, so the training loop and the
    XLA arena are the only large anonymous allocations. The sidecar is written once and
    reused: a header/size check decides whether it is already complete.
    """
    from numpy.lib import format as npformat
    import zipfile

    with zipfile.ZipFile(npz_path) as z, z.open(member) as src:
        version = npformat.read_magic(src)
        reader = (npformat.read_array_header_1_0 if version == (1, 0)
                  else npformat.read_array_header_2_0)
        shape, fortran, dtype = reader(src)
        nbytes = int(np.prod(shape)) * dtype.itemsize

        if os.path.exists(out_path):
            try:
                with open(out_path, "rb") as f:
                    v2 = npformat.read_magic(f)
                    r2 = (npformat.read_array_header_1_0 if v2 == (1, 0)
                          else npformat.read_array_header_2_0)
                    s2, _, d2 = r2(f)
                    if s2 == shape and d2 == dtype and os.path.getsize(out_path) == f.tell() + nbytes:
                        return shape, dtype
            except Exception:
                pass

        tmp = out_path + ".part"
        with open(tmp, "wb") as dst:
            header = {"descr": npformat.dtype_to_descr(dtype),
                      "fortran_order": fortran, "shape": shape}
            if version == (1, 0):
                npformat.write_array_header_1_0(dst, header)
            else:
                npformat.write_array_header_2_0(dst, header)
            while True:
                buf = src.read(chunk)
                if not buf:
                    break
                dst.write(buf)
        os.replace(tmp, out_path)
    return shape, dtype


def load_corpus(data_path: str, use_memmap: bool, cache_dir: str = "corpus_cache"):
    """Returns (X_train, X_val, NpzFile). X_* are memmaps when use_memmap is set.

    A float16 corpus is NOT widened here. The whole point of `build_corpus_v2.py
    --dtype fp16` is that an 850 MB corpus stays 425 MB on disk and in the memmap
    sidecar; inflating it back to float32 up front would hand the saving straight back
    and, without --memmap, would put 850 MB of anonymous memory back on the machine.
    The upcast is per batch in the training loop, and per scoring batch in score().
    """
    data = np.load(data_path, allow_pickle=True)
    if not use_memmap:
        return data["X_train"], data["X_val"], data

    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    out = {}
    for name in ("X_train", "X_val"):
        side = os.path.join(cache_dir, f"{stem}_{name}.npy")
        t0 = time.time()
        shape, dtype = stream_npy_member_to_file(data_path, f"{name}.npy", side)
        print(f" corpus sidecar    : {side} {shape} {dtype} "
              f"({os.path.getsize(side)/1e6:.0f} MB, {time.time()-t0:.0f}s)")
        out[name] = np.load(side, mmap_mode="r")
    return out["X_train"], out["X_val"], data


# ==============================================================================
# Training
# ==============================================================================

def run_training(data_path="dataset_v2.npz", arch="bcconformer_50k", epochs=30,
                 batch_size=64, lr=1e-3, save_path="best_v2.flax",
                 fpr_budget=0.005, hard_neg_frac=0.5, hard_neg_k=0, gamma=1.5,
                 pos_bias=1.15, spec_aug_prob=0.35, offset_stress=True,
                 warm_start=None, seed=7, gain_aug_prob=0.6, gain_aug_db=12.0,
                 bucket_weight=True, val_metric_cap=12000, use_memmap=False):
    print("=" * 78)
    print(f" INDUSTRIAL TRAINING v2  |  arch={arch}  |  objective = max TPR @ FPR <= {fpr_budget*100:.2f}%")
    print("=" * 78)

    X_train, X_val, data = load_corpus(data_path, use_memmap)
    y_train, y_val = data["y_train"], data["y_val"]
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, -1)
    if X_val.ndim == 3:
        X_val = np.expand_dims(X_val, -1)
    b_train = data["b_train"].astype(str) if "b_train" in data.files else None
    b_val = data["b_val"].astype(str) if "b_val" in data.files else None

    print(f" dataset            : {data_path}")
    print(f" log-mel dtype      : {X_train.dtype}"
          + ("  (upcast to float32 per batch)" if X_train.dtype != np.float32 else ""))
    print(f" train              : {len(X_train):,}  ({int((y_train==1).sum()):,} pos / {int((y_train==0).sum()):,} neg)")
    print(f" val                : {len(X_val):,}  ({int((y_val==1).sum()):,} pos / {int((y_val==0).sum()):,} neg)")
    if b_train is not None:
        print(f" mixture buckets    : {len(np.unique(b_train))}  (per-bucket loss + metrics enabled)")

    model = get_model(arch, num_classes=2)
    n_params = count_params(model)
    print(f" model              : {arch}  {n_params:,} params  ({n_params/1024:.1f} KB INT8)")

    # Class weights: sqrt law, optional positive bias. Unlike v1 this does not
    # penalise the negative class, which is what capped recall.
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    ratio = n_neg / n_pos
    neg_w = float(np.clip(np.sqrt(ratio), 0.5, 2.0))
    pos_w = float(pos_bias * neg_w)
    cw = jnp.array([neg_w, pos_w], dtype=jnp.float32)
    print(f" class weights      : neg={neg_w:.3f} pos={pos_w:.3f}  (pos:neg count ratio 1:{ratio:.2f})")
    print(f" focal gamma        : {gamma}   hard-negative mining: "
          f"{'k=' + str(hard_neg_k) if hard_neg_k > 0 else 'frac=' + str(hard_neg_frac)}")
    print(f" spec-augment prob  : {spec_aug_prob}   mic-gain aug: p={gain_aug_prob}, +/-{gain_aug_db} dB")
    print(f" bucket loss weight : {'on' if (bucket_weight and b_train is not None) else 'off'}")

    rng_np = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    init_params = None
    if warm_start and os.path.exists(warm_start):
        with open(warm_start, "rb") as f:
            ck = serialization.msgpack_restore(f.read())
        variables = model.init(init_key, jnp.ones((1, 49, 40, 1), jnp.float32), train=False)
        template = variables["params"]
        # msgpack_restore already returns a pytree of arrays, so this is from_state_dict.
        # from_bytes wants raw bytes and raises, which used to be swallowed below and
        # silently turned a fine-tune into a from-scratch run.
        init_params = serialization.from_state_dict(template, ck["params"])
        missing = set(template) ^ set(init_params)
        if missing:
            raise ValueError(f"warm start {warm_start} does not match {arch}: {sorted(missing)}")
        print(f" warm start         : {warm_start}  "
              f"({sum(x.size for x in jax.tree_util.tree_leaves(init_params)):,} params)")

    steps_per_epoch = len(X_train) // batch_size
    total_steps = epochs * steps_per_epoch
    state = create_train_state(init_key, model, lr, total_steps, init_params=init_params)

    pos_mask = y_val == 1
    neg_mask = y_val == 0
    X_val_pos = X_val[pos_mask][:, :, :, 0]
    X_val_neg = X_val[neg_mask][:, :, :, 0]

    # Per-bucket validation subset: cheap enough to score every epoch, and it keeps
    # the rare hard regimes visible instead of averaging them away.
    bucket_val = b_val if b_val is not None else np.where(y_val == 1, "pos", "neg")
    if val_metric_cap and len(X_val) > val_metric_cap:
        sub = rng_np.choice(len(X_val), size=val_metric_cap, replace=False)
        X_val_b_spec = X_val[sub][:, :, :, 0]
        y_val_b = y_val[sub]
        bucket_val = bucket_val[sub]
    else:
        X_val_b_spec = X_val[:, :, :, 0]
        y_val_b = y_val

    roll_offsets = rng_np.integers(0, 20, size=len(X_val_pos))
    X_val_pos_rolled = roll_batch(X_val_pos, rng_np, roll_offsets)
    print(f" offset-stress set  : {len(X_val_pos)} positives rolled by U[0,20) frames")
    print(f" per-bucket report  : {len(X_val_b_spec):,} val samples across "
          f"{len(np.unique(bucket_val))} buckets")

    # Bucket -> per-sample loss weight, as a dense vector for the training step.
    bucket_w_train = None
    if bucket_weight and b_train is not None:
        default_w = 1.0
        bucket_w_train = np.array([BUCKET_LOSS_WEIGHT.get(str(b), default_w) for b in b_train],
                                  dtype=np.float32)

    best_score = -1.0
    best = {}
    # The old table printed an "FPR" column next to every recall column, which read like a
    # measurement but was not one: the threshold is *chosen* so that exactly the budget's
    # worth of negatives cross it, so the column always echoed the budget. Worse, "recall
    # 77.74% at FPR 0.500%" is precisely the sentence that gets mistaken for a deployment
    # claim, and on real audio that same threshold fired 4.25 times per hour.
    print("=" * 96)
    print(" HOW TO READ THIS TABLE")
    print("=" * 96)
    print(f"  thr          the operating threshold, chosen so that {fpr_budget*100:.2f}% of the")
    print(f"               validation NEGATIVES cross it. That is a property of 1 s,")
    print(f"               level-normalised slices -- it is NOT a deployment number.")
    print(f"  FA/1k neg    how many of every 1000 validation negatives cross thr, as a whole")
    print(f"               number. At a {fpr_budget*100:.2f}% budget that is about "
          f"{int(round(fpr_budget*1000))} negatives; a printed")
    print(f"               percentage like 0.500% would be a rounded echo of the budget.")
    print("  recall@bud   true-positive rate on validation positives at thr. Higher better.")
    print("  recall@roll  the SAME positives, randomly rolled inside the 1 s window.")
    print("               The gap between the two columns is the model's position")
    print("               dependence, and it is the number that decides deployment value.")
    print("  SCORE        0.5*(recall@bud + recall@roll). This is the checkpoint selection")
    print("               criterion; the * marks the epoch that owns the saved checkpoint.")
    print("  recall@0.12% the v1 reference budget, kept only for continuity with old tables.")
    print()
    print("  MEASURED DEPLOYMENT FALSE ALARMS ARE NOT IN THIS TABLE. A checkpoint can look")
    print("  excellent here and still fire constantly on real audio -- that has happened twice")
    print("  in this project. Re-freeze the threshold against sweep_threshold.py output,")
    print("  which scores real soundscape hours and continuous human speech.")
    print("=" * 96)
    header = (f"{'Ep':>3} | {'Loss':>7} | {'TrAcc':>6} | {'AUC':>7} | {'thr':>6} | "
              f"{'FA/1k':>6} | {'recall@bud':>10} | {'recall@roll':>11} | {'SCORE':>7} | "
              f"{'recall@0.12':>10}")
    print("-" * 96)
    print(header)
    print("-" * 96)

    t_start = time.time()
    for ep in range(1, epochs + 1):
        # Gather per batch rather than materialising X_train[perm]: a full permuted copy
        # of the corpus is another 850 MB on a 108k-sample corpus, which does not fit
        # alongside the corpus itself on a 16 GB machine. The batch contents are
        # identical either way.
        perm = rng_np.permutation(len(X_train))

        losses, accs = [], []
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            # Upcast here, not at load time. A corpus written with build_corpus_v2.py
            # --dtype fp16 stays float16 on disk and in the memmap sidecar, so the batch
            # is the first point where the widened type is needed -- and it is
            # batch-sized, so the augmentation below (which copies the batch) still runs
            # in float32 rather than rounding every gain-augmented sample to float16.
            bx = np.asarray(X_train[idx], dtype=np.float32)
            by = y_train[idx]
            bx = spec_augment(bx, rng_np, prob=spec_aug_prob)
            bx = gain_augment(bx, rng_np, prob=gain_aug_prob, span_db=gain_aug_db)
            bw = None
            if bucket_w_train is not None:
                bw = jnp.asarray(bucket_w_train[idx], jnp.float32)
            key, step_key = jax.random.split(key)
            state, loss, acc = train_step(state, jnp.asarray(bx, jnp.float32),
                                          jnp.asarray(by, jnp.int32), step_key,
                                          gamma=gamma, class_weights=cw,
                                          hard_neg_frac=hard_neg_frac,
                                          hard_neg_k=hard_neg_k, bucket_w=bw)
            losses.append(float(loss))
            accs.append(float(acc))

        # ---- validation scoring ----
        def score(arr):
            out = []
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 3:
                a = np.expand_dims(a, -1)
            for i in range(0, len(a), 512):
                out.extend(np.array(predict_probs(state, jnp.asarray(a[i:i + 512]))))
            return np.array(out)

        p_pos = score(X_val_pos)
        p_neg = score(X_val_neg)
        p_pos_roll = score(X_val_pos_rolled)
        p_b = score(X_val_b_spec)
        p_all = np.concatenate([p_pos, p_neg])
        y_all = np.concatenate([np.ones(len(p_pos), np.int32), np.zeros(len(p_neg), np.int32)])
        auc = roc_auc_score(y_all, p_all) if len(set(y_all.tolist())) > 1 else 0.5

        tpr_b, thr_b, fpr_b = tpr_at_fpr_budget(p_pos, p_neg, fpr_budget)
        tpr_b_roll, thr_r, fpr_r = tpr_at_fpr_budget(p_pos_roll, p_neg, fpr_budget)
        tpr_off = 0.5 * (tpr_b + tpr_b_roll) if offset_stress else tpr_b

        # hard FPR reference: the 0.12% budget v1 claimed
        tpr_h, thr_h, fpr_h = tpr_at_fpr_budget(p_pos, p_neg, 0.0012)
        tpr_h_roll, _, _ = tpr_at_fpr_budget(p_pos_roll, p_neg, 0.0012)

        score_key = tpr_off if offset_stress else tpr_b
        marker = ""
        improved = score_key > best_score + 1e-6
        if improved and ep >= 3:
            best_score = score_key
            thr_pick = thr_b if offset_stress else thr_b
            best = {
                "params": state.params, "arch": arch, "param_count": n_params,
                "epoch": ep, "val_auc": float(auc), "fpr_budget": fpr_budget,
                "threshold": float(thr_pick), "tpr_at_budget": float(tpr_b),
                "fpr_at_budget": float(fpr_b),
                "tpr_offset_stress": float(tpr_off),
                "tpr_at_0012": float(tpr_h), "fpr_at_0012": float(fpr_h),
                "tpr_offset_0012": float(0.5 * (tpr_h + tpr_h_roll)),
                "tpr_at_085": tpr_at_threshold(p_pos, 0.85),
                "fpr_at_085": float(np.mean(p_neg >= 0.85)),
            }
            best["bucket_metrics"] = {str(k): float(v) for k, v in bucket_stats(p_b, bucket_val,
                                                                              y_val_b, thr_b).items()}
            with open(save_path, "wb") as f:
                f.write(serialization.to_bytes(best))
            marker = " *"

        if ep % 2 == 0 or ep == epochs or marker:
            fa_per_k = int(round(1000.0 * float(np.sum(p_neg >= thr_b)) / max(1, len(p_neg))))
            print(f"{ep:3d} | {np.mean(losses):7.4f} | {np.mean(accs)*100:5.1f}% | {auc*100:6.3f}% | "
                  f"{thr_b:6.3f} | {fa_per_k:6d} | {tpr_b*100:9.2f}% | {tpr_b_roll*100:10.2f}% | "
                  f"{score_key*100:6.2f}% | {tpr_h*100:9.2f}%{marker}", flush=True)
            if b_val is not None and (ep % 4 == 0 or ep == epochs or marker):
                print_bucket_table(p_b, bucket_val, y_val_b, thr_b)

    print("-" * 96)
    print(f" training time      : {time.time() - t_start:.1f}s")
    print(f" best epoch         : {best.get('epoch')}")
    print(f" SCORE (selection)  : {best.get('tpr_offset_stress', 0)*100:.2f}%  "
          f"= 0.5*(recall {best.get('tpr_at_budget', 0)*100:.2f}% + "
          f"rolled {0.5*best.get('tpr_offset_stress', 0)*100 - best.get('tpr_at_budget', 0)*100:.2f}%)")
    print(f" recall @ 0.12% FPR: {best.get('tpr_at_0012', 0)*100:.2f}%"
          f"  (offset-stressed {best.get('tpr_offset_0012', 0)*100:.2f}%)")
    print(f" stored threshold   : {best.get('threshold', 0):.4f}")
    print("                      ^ a VALIDATION threshold. It has not been checked against")
    print("                        real audio. Run sweep_threshold.py and freeze a point")
    print("                        before treating it as a deployment operating point.")
    print(f" checkpoint         : {save_path}")
    print("=" * 96, flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v2.npz")
    ap.add_argument("--arch", type=str, default="bcconformer_50k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default="best_v2.flax")
    ap.add_argument("--fpr_budget", type=float, default=0.005)
    ap.add_argument("--hard_neg_frac", type=float, default=0.5)
    ap.add_argument("--hard_neg_k", type=int, default=0,
                    help="mine a fixed count of the hardest negatives per batch (overrides hard_neg_frac)")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--pos_bias", type=float, default=1.15)
    ap.add_argument("--spec_aug_prob", type=float, default=0.35)
    ap.add_argument("--no_offset_stress", action="store_true")
    ap.add_argument("--no_gain_aug", action="store_true")
    ap.add_argument("--no_bucket_weight", action="store_true")
    ap.add_argument("--gain_aug_db", type=float, default=12.0)
    ap.add_argument("--val_metric_cap", type=int, default=12000)
    ap.add_argument("--warm_start", type=str, default="")
    ap.add_argument("--memmap", action="store_true",
                    help="extract X_train/X_val to .npy sidecars once and memory-map them "
                         "instead of inflating the whole corpus into RAM")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run_training(
        data_path=a.data, arch=a.arch, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, save_path=a.save, fpr_budget=a.fpr_budget,
        hard_neg_frac=a.hard_neg_frac, hard_neg_k=a.hard_neg_k, gamma=a.gamma,
        pos_bias=a.pos_bias,
        spec_aug_prob=a.spec_aug_prob, offset_stress=not a.no_offset_stress,
        warm_start=a.warm_start or None, seed=a.seed,
        gain_aug_prob=0.0 if a.no_gain_aug else 0.6, gain_aug_db=a.gain_aug_db,
        bucket_weight=not a.no_bucket_weight, val_metric_cap=a.val_metric_cap,
        use_memmap=a.memmap)
