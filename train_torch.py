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

def offset_stress_probs(model, xb, device, n_offsets=8, bs=512):
    """Recall when the same clip is rolled to different positions in the window.

    This is the metric that separates a position-robust detector from one that
    learned a prior. v1/v2 lost 62.6 points of recall across the window because
    of an absolute position embedding; v3's relative bias fixed it.
    """
    probs = []
    model.eval()
    t = xb.shape[1]
    with torch.no_grad():
        for off in range(0, t, max(1, t // n_offsets)):
            rolled = torch.roll(xb, shifts=off, dims=1).contiguous()
            for i in range(0, rolled.shape[0], bs):
                p = torch.softmax(model(rolled[i:i + bs].to(device)), -1)[:, 1]
                probs.append(p.float().cpu().numpy())
    model.train()
    return np.concatenate(probs) if probs else np.array([])

@torch.no_grad()
def score_all(model, X, y, device, bs=1024):
    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device)
        p = torch.softmax(model(xb), -1)[:, 1]
        probs[i:i + bs] = p.float().cpu().numpy()
    model.train()
    return probs

def resolve_arch(name: str):
    """Build a model for `name`, accepting every preset family.

    `kws_*` -> models_torch (RAM/CPU-floor presets)
    `hc_*`, `m1_*` -> models_torch_hc (high-capacity and 1.5 MMAC tiers)

    Note the fallback behaviour: models_torch.get_torch_model warns and returns
    'kws_tight' for an unknown name. Silently training the wrong architecture
    under the requested name is worse than failing, so unknown names raise here
    instead. That bug already cost one run (m1_a trained as kws_tight).
    """
    if name.startswith("m1_") or name.startswith("hc_"):
        from models_torch_hc import ARCHS_HC, ARCHS_M1, get_hc_model
        if name not in ARCHS_HC and name not in ARCHS_M1:
            raise ValueError(f"unknown arch '{name}'. "
                             f"Available: {sorted(ARCHS_HC) + sorted(ARCHS_M1)}")
        return get_hc_model(name)
    from models_torch import get_torch_model, ARCHS
    if name not in ARCHS and name not in ("kwsnet", "kwsnet_tiny"):
        raise ValueError(f"unknown arch '{name}'. Available: {sorted(ARCHS)}")
    return get_torch_model(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v6.npz")
    ap.add_argument("--arch", default="kws_tight")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=1.5, help="focal gamma")
    ap.add_argument("--pos_bias", type=float, default=1.15)
    ap.add_argument("--fpr_budget", type=float, default=0.005)
    ap.add_argument("--hard_neg_k", type=int, default=64)
    ap.add_argument("--spec_aug_prob", type=float, default=0.35)
    ap.add_argument("--gain_prob", type=float, default=0.6)
    ap.add_argument("--roll_prob", type=float, default=0.5,
                    help="probability of a random TIME ROLL per training sample. "
                         "The direct attack on offset fragility: makes every "
                         "position in the 1 s window equally likely to carry the "
                         "keyword, which is what the deployed case looks like.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_memmap", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78, flush=True)
    print(f"data {args.data}  arch {args.arch}  epochs {args.epochs} "
          f"batch {args.batch}  device {device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB",
              flush=True)
    print("=" * 78, flush=True)

    Xtr, Xva, data = load_corpus(args.data, use_memmap=not args.no_memmap)
    ytr = np.asarray(data["y_train"])
    yva = np.asarray(data["y_val"])
    print(f"train {Xtr.shape} {Xtr.dtype}  pos {int((ytr==1).sum()):,} "
          f"neg {int((ytr==0).sum()):,}", flush=True)
    print(f"val   {Xva.shape} {Xva.dtype}  pos {int((yva==1).sum()):,} "
          f"neg {int((yva==0).sum()):,}", flush=True)

    btr = data["b_train"] if "b_train" in data.files else None
    if btr is not None:
        print(f"buckets: {len(np.unique(btr))} distinct", flush=True)

    model = resolve_arch(args.arch).to(device)
    mac = count_macs(model)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {nparam:,} params  {mac:,} MAC  "
          f"floor {mac/3_840_000:.3f} ms  "
          f"({11.11/(mac/3_840_000):.1f}x headroom in 11.11 ms)", flush=True)
    print(flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    nsteps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=nsteps, pct_start=0.15)

    sw = torch.tensor([args.pos_bias, 1.0], device=device, dtype=torch.float32)
    bw_tr = None
    if btr is not None:
        bw_tr = torch.tensor(
            [BUCKET_LOSS_WEIGHT.get(str(b), 1.0) for b in btr],
            device=device, dtype=torch.float32)

    best = {"tpr": -1.0}
    hist = []
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(Xtr[idx], dtype=np.float32)).to(device)
            yb = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
            xb = augment(xb, rng, args.spec_aug_prob, args.gain_prob,
                           roll_prob=args.roll_prob)

            logits = model(xb)
            ce = F.cross_entropy(logits, yb, reduction="none",
                                 label_smoothing=0.02)
            pt = torch.softmax(logits, -1).gather(1, yb[:, None]).squeeze(1)
            focal = (1.0 - pt).clamp_min(1e-4).pow(args.gamma)
            per = ce * focal * sw[yb]
            if bw_tr is not None:
                per = per * bw_tr[torch.from_numpy(idx).to(device)]
            if args.hard_neg_k > 0:
                is_neg = yb == 0
                k = min(args.hard_neg_k, int(is_neg.sum()))
                if k > 0:
                    negv = torch.where(is_neg, per,
                                      torch.full_like(per, float("inf")))
                    cut = torch.topk(negv, k, largest=False).values[-1]
                    keep = (yb == 1) | (per >= cut)
                    per = per * keep.float()
            loss = per.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1

        pva = score_all(model, Xva, yva, device)
        pos_p, neg_p = pva[yva == 1], pva[yva == 0]
        auc = auc_score(pos_p, neg_p)
        tpr, thr, fpr = tpr_at_fpr_budget(pos_p, neg_p, args.fpr_budget)
        acc = float(((pva >= thr).astype(np.int32) == yva).mean())

        # position robustness: the same positives rolled across the window
        xpos = torch.from_numpy(np.asarray(Xva[yva == 1][:512], dtype=np.float32))
        op = offset_stress_probs(model, xpos, device)
        os_tpr = float(np.mean(op >= thr)) if len(op) else float("nan")

        hist.append({"epoch": ep, "loss": tot / max(1, nb), "auc": auc,
                     "tpr": tpr, "thr": thr, "fpr": fpr, "acc": acc,
                     "offset_tpr": os_tpr})
        print("ep %2d  loss %.4f  auc %.4f  tpr@%.3f%% %.4f  thr %.4f  "
              "acc %.4f  offset-recall %.4f"
              % (ep, tot / max(1, nb), auc, args.fpr_budget * 100, tpr, thr,
                 acc, os_tpr), flush=True)

        if tpr > best["tpr"]:
            best = {"tpr": tpr, "epoch": ep, "thr": thr, "auc": auc,
                    "fpr": fpr, "offset_tpr": os_tpr, "acc": acc}
            if args.out:
                torch.save({"arch": args.arch, "state_dict": model.state_dict(),
                            "threshold": thr, "epoch": ep, "tpr": tpr,
                            "auc": auc, "mac": mac, "params": nparam}, args.out)

    print(flush=True)
    print("=" * 78)
    print("BEST (by TPR at %.3f%% FPR budget)" % (args.fpr_budget * 100))
    print("  epoch          %d" % best["epoch"])
    print("  TPR            %.4f" % best["tpr"])
    print("  FPR            %.4f" % best["fpr"])
    print("  threshold      %.4f" % best["thr"])
    print("  AUC            %.4f" % best["auc"])
    print("  accuracy@thr   %.4f" % best["acc"])
    if best["offset_tpr"] == best["offset_tpr"]:
        print("  offset recall  %.4f  <- position robustness" % best["offset_tpr"])
    print("  MAC            %d (%.3f ms floor)" % (mac, mac / 3_840_000))
    print("  params         %d" % nparam)
    print("  wall clock     %.1f min" % ((time.time() - t_start) / 60))
    if args.out:
        print("  checkpoint     %s" % args.out)
    print("=" * 78)

    with open("train_result.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "best": best, "history": hist,
                   "mac": mac, "params": nparam,
                   "floor_ms": mac / 3_840_000}, fh, indent=2)
    print("history -> train_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
