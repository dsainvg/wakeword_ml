"""
PyTorch trainer for the budget-compliant KWSNet architectures
-------------------------------------------------------------
PyTorch counterpart of `train_industrial_v2.py` (which is JAX/Flax). Kept
deliberately parallel in structure so the two can be compared, and it carries
over the objective and metrics that the JAX version established:

  * TPR-at-FPR-budget as the selection criterion, NOT accuracy. Accuracy is a
    bad criterion for this problem: the corpus is 2:1 negative, so a model that
    always answers "no" scores 66% and looks fine.
  * per-bucket loss weighting, so the rare hard buckets are not drowned out
  * on-batch hard-negative mining (OHEM)
  * offset stress: roll the spectrogram in time, because the deployed keyword
    lands at an arbitrary position inside the 1 s window. v1/v2 showed a 62.6
    point recall spread across the window from a positional prior, so this is the
    metric that matters most.
  * mic-gain augmentation, because log-mel is a log of energy and a constant
    offset there is exactly a gain change

    py train_torch.py --data dataset_v6.npz --arch kws_tight --epochs 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_torch import get_torch_model, count_macs   # noqa: E402

# Must match build_corpus_v2.BUCKET_LOSS_WEIGHT. Copied rather than imported so
# the trainer does not drag in the (heavy) corpus builder.
BUCKET_LOSS_WEIGHT = {
    "P0_word_direct": 1.00, "P1_word_one_noise": 1.15,
    "P2_word_two_noise": 1.25, "P3_word_two_noise_hard": 1.30,
    "N0_filler_direct": 0.80, "N1_filler_one_noise": 0.90,
    "N2_filler_two_noise": 1.00, "N3_filler_two_noise_hard": 1.10,
    "N4_real_direct": 0.90, "N5_real_one_noise": 1.00,
    "N6_real_two_noise": 1.10, "N7_real_two_noise_hard": 1.20,
    "N8_babble_direct": 1.00, "N9_babble_one_noise": 1.10,
    "N10_babble_two_noise": 1.20, "N11_babble_two_noise_hard": 1.30,
    "N12_noise_direct": 0.90, "N13_noise_one_noise": 1.00,
    "N14_noise_two_noise": 1.00, "N15_noise_two_noise_hard": 1.10,
    "N16_confus_direct": 1.00, "N17_confus_one_noise": 1.10,
    "N18_confus_two_noise": 1.10, "N19_confus_two_noise_hard": 1.20,
}


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------
def threshold_for_fpr(neg_probs: np.ndarray, budget: float) -> float:
    """Smallest threshold whose negative false-positive rate stays within budget."""
    if len(neg_probs) == 0:
        return 0.5
    allowed = int(np.floor(budget * len(neg_probs)))
    if allowed <= 0:
        return float(np.nextafter(neg_probs.max(), 1.0))
    return float(np.sort(neg_probs)[::-1][allowed - 1])


def tpr_at_fpr_budget(pos, neg, budget):
    t = threshold_for_fpr(neg, budget)
    tpr = float(np.mean(pos >= t)) if len(pos) else 0.0
    fpr = float(np.mean(neg >= t)) if len(neg) else 0.0
    return tpr, t, fpr


def auc_score(pos, neg) -> float:
    """Rank-based AUC, tie-safe. Avoids a sklearn import in the hot path."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # Average ranks across ties, otherwise a run of identical scores (common
    # with a saturating net) inflates AUC.
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
def load_corpus(path, use_memmap=True, cache_dir="corpus_cache"):
    """Load X_train/X_val, streaming big members to .npy sidecars.

    A float16 corpus is NOT widened here. The whole point of `--dtype fp16` is
    that the artifact stays half-size; inflating it to float32 up front hands the
    saving straight back. The upcast happens per batch, on GPU.
    """
    from numpy.lib import format as npformat
    import zipfile

    data = np.load(path, allow_pickle=True)
    if not use_memmap:
        return data["X_train"], data["X_val"], data

    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = {}
    for name in ("X_train", "X_val"):
        side = os.path.join(cache_dir, f"{stem}_{name}.npy")
        if os.path.exists(side):
            out[name] = np.load(side, mmap_mode="r")
            continue
        t0 = time.time()
        with zipfile.ZipFile(path) as z, z.open(f"{name}.npy") as src:
            version = npformat.read_magic(src)
            reader = (npformat.read_array_header_1_0 if version == (1, 0)
                      else npformat.read_array_header_2_0)
            shape, _fortran, dtype = reader(src)
            tmp = side + ".part"
            with open(tmp, "wb") as dst:
                npformat.write_array_header_1_0(
                    dst, {"descr": npformat.dtype_to_descr(dtype),
                          "fortran_order": False, "shape": shape})
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    dst.write(buf)
            os.replace(tmp, side)
        print(f"  sidecar {side} {shape} {dtype} "
              f"({os.path.getsize(side)/1e6:.0f} MB, {time.time()-t0:.0f}s)")
        out[name] = np.load(side, mmap_mode="r")
    return out["X_train"], out["X_val"], data


def augment(xb, rng, spec_aug_prob=0.35, gain_prob=0.6, span_db=12.0,
            roll_prob=0.5):
    """SpecAugment + mic-gain jitter + TIME ROLL, on a (B,T,M,1) GPU batch.

    gain:  log-mel is log(energy), so a constant offset d multiplies waveform
           energy by e^d. A +/- span_db/20 dB amplitude swing is therefore a
           constant offset of span_db/20*ln(10) in feature space -- the cheapest
           way to stop the model keying on absolute level.

    roll:  THE fix for offset fragility. The deployed keyword lands at an
           arbitrary position inside the 1 s window, but a model trained only on
           one placement learns that placement. Rolling each clip to a random
           time offset makes every position equally likely to carry the keyword.
           This is the training-side twin of the offset_stress metric, which is
           the weakest number in the iteration log (~43% vs ~78% offset-free).
           Applied PER SAMPLE: a shared offset would leave the rest of the batch
           un-augmented and teach nothing.
    """
    b, t, m, _ = xb.shape
    if roll_prob > 0:
        for i in range(b):
            if rng.random() < roll_prob:
                xb[i] = torch.roll(xb[i], int(rng.integers(0, t)), dims=0)
    if spec_aug_prob > 0:
        for i in range(b):
            if rng.random() < spec_aug_prob:
                fw = int(rng.integers(1, 5))
                fs = int(rng.integers(0, max(1, m - fw)))
                xb[i, :, fs:fs + fw, :] = xb[i].min()
                tw = int(rng.integers(1, 8))
                ts = int(rng.integers(0, max(1, t - tw)))
                xb[i, ts:ts + tw, :, :] = xb[i].min()
    if gain_prob > 0:
        k = np.ceil(span_db / 20.0 * np.log(10.0))
        for i in range(b):
            if rng.random() < gain_prob:
                xb[i] += float(rng.uniform(-k, k))
    return xb


