"""Independently verify the trained checkpoint actually loads and scores.

The training log is not evidence: a run can print a good-looking table and still
save nothing usable. This re-loads best_kwstight.pt from disk, rebuilds the model
from its recorded arch name, runs it on real val data, and recomputes the
operating point from scratch.

    py verify_checkpoint.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_torch import resolve_arch  # noqa: E402
from models_torch import count_macs  # noqa: E402
from train_torch import (load_corpus, score_all, tpr_at_fpr_budget,  # noqa: E402
                         auc_score, offset_stress_probs)

OUT = []


def emit(s=""):
    OUT.append(str(s))
    print(s, flush=True)


CKPT = "best_hc_balanced.pt"
if not os.path.exists(CKPT):
    emit("FATAL: %s not found" % CKPT)
    raise SystemExit(1)

emit("=" * 74)
emit("checkpoint: %s  (%.1f KB)" % (CKPT, os.path.getsize(CKPT) / 1024))
emit("=" * 74)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
emit("arch           %s" % ck["arch"])
emit("epoch          %s" % ck["epoch"])
emit("recorded TPR   %.4f" % ck["tpr"])
emit("recorded AUC   %.4f" % ck["auc"])
emit("threshold      %.4f" % ck["threshold"])
emit("params         %d" % ck["params"])
emit("MAC            %d (floor %.3f ms)" % (ck["mac"], ck["mac"] / 3_840_000))
emit("state_dict     %d tensors" % len(ck["state_dict"]))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resolve_arch(ck["arch"]).to(device)
model.load_state_dict(ck["state_dict"], strict=True)
emit("load_state_dict OK (strict=True, no missing/unexpected keys)")

# recomputed MAC must match what was recorded
mac = count_macs(model)
emit("MAC recomputed  %d  -> matches recorded: %s"
     % (mac, "YES" if mac == ck["mac"] else "NO (MISMATCH)"))

Xtr, Xva, data = load_corpus("dataset_v7.npz", use_memmap=True)
yva = np.asarray(data["y_val"])
emit("")
emit("val %s  pos %d  neg %d" % (Xva.shape, int((yva == 1).sum()), int((yva == 0).sum())))

p = score_all(model, Xva, yva, device)
pos, neg = p[yva == 1], p[yva == 0]
emit("")
emit("RECOMPUTED from the reloaded weights:")
emit("  AUC                 %.4f" % auc_score(pos, neg))
for budget in (0.005, 0.01, 0.02):
    tpr, thr, fpr = tpr_at_fpr_budget(pos, neg, budget)
    emit("  TPR @ %.1f%% FPR     %.4f   (thr %.4f, actual fpr %.4f)"
         % (budget * 100, tpr, thr, fpr))

xpos = torch.from_numpy(np.asarray(Xva[yva == 1][:512], dtype=np.float32))
op = offset_stress_probs(model, xpos, device)
tpr_saved, thr_saved, _ = tpr_at_fpr_budget(pos, neg, 0.005)
emit("  offset recall @saved thr  %.4f" % float(np.mean(op >= thr_saved)))
emit("  mean P(kw) speech %.4f   noise %.4f"
     % (pos.mean(), neg.mean()))
emit("")
emit("saved threshold %.4f  vs  recomputed %.4f  -> %s"
     % (ck["threshold"], thr_saved,
        "consistent" if abs(ck["threshold"] - thr_saved) < 0.15 else "DIFFERS"))

open("verify_ckpt.out", "w", encoding="utf-8").write("\n".join(OUT))