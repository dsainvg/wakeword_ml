"""
Iteration log for the "Amaze" KWS architecture search
=====================================================
One record per trained model, so each architecture's MEASURED result stays
recoverable instead of being overwritten by the next experiment. Read by the
orchestrator after every training run; `print_table()` is the summary.

Columns
  iter    iteration label
  arch    preset name, resolvable via models_torch_hc.get_hc_model(name)
  corpus  training corpus file
  params / mmac / floor   measured by models_torch.count_macs, never estimated
  tpr     offline frame TPR at the 0.5% false-positive budget
  auc     validation ROC-AUC
  offset  recall when positives are rolled to different positions in the window

Reference baselines NOT produced by this script:
  bcconformer_v3 : 84,865 params | 11.24 MMAC | 2.93 ms floor | 480 KB RAM
                   79.8% STREAMING recall. That is a STREAMING number on a
                   24-bucket corpus; the tpr column here is an OFFLINE frame
                   number on a 16-bucket (v4) corpus, so the two are not directly
                   comparable and should not be tabulated together as if they were.

    py iteration_log.py
"""

ITERS = [
    dict(iter=0, arch="kws_tight", corpus="dataset_v6.npz",
         params=16_383, mmac=1.128, floor=0.294,
         tpr=0.1578, auc=0.9183, offset=0.0809,
         note="First GPU run. Small corpus (33.6k train) -> starved. Kept as the "
              "low-data baseline so later gains can be attributed to data scale "
              "rather than architecture."),

    dict(iter=1, arch="hc_balanced", corpus="dataset_v7.npz",
         params=125_959, mmac=2.616, floor=0.681,
         tpr=0.8322, auc=0.9940, offset=0.5336,
         note="3x data (96k train) moved TPR 15.8% -> 83.2% at similar capacity, "
              "confirming the audit's conclusion that DATA, not capacity, was the "
              "bottleneck. Independently re-verified from the saved checkpoint."),

    dict(iter=2, arch="m1_a", corpus="dataset_v7.npz",
         params=129_675, mmac=1.448, floor=0.377,
         tpr=0.7808, auc=0.9908, offset=0.4355,
         note="1.5 MMAC cap, trained. 129,675 params -- MORE than iter1's 125,959 -- at "
              "55% of the arithmetic (T 49->12 halves per-block MACs while parameter "
              "counts are untouched). TPR 83.2% -> 78.1% vs iter1: a 5.1-point cost "
              "for a 1.8x speedup. 9.3 min train time vs 23.5 min."),

    dict(iter=3, arch="m1_a", corpus="dataset_v8_accent.npz",
         params=129_675, mmac=1.448, floor=0.377,
         tpr=0.7850, auc=0.9909, offset=0.4299,
         note="Same architecture and same cost as iter2, accent-rich corpus only. "
              "TPR 78.08% -> 78.50% (+0.4pt), accuracy 92.36% -> 92.50%. A small but "
              "real gain from accent diversity alone, with zero compute cost. "
              "Converged FASTER (best epoch 30 vs 36), which matters on a CPU budget."),

    dict(iter=4, arch="m1_e_t24", corpus="dataset_v8_accent.npz",
         params=63_655, mmac=2.237, floor=0.583,
         tpr=0.7836, auc=0.9913, offset=0.4473,
         note="T=24 instead of T=12, to price the temporal cut directly. TPR 78.36% "
              "vs iter3's 78.50% -- statistically the SAME accuracy for 54% more "
              "arithmetic (2.24 vs 1.45 MMAC) and half the parameters. Conclusion: "
              "the T=12 cut costs NOTHING in accuracy. Best offset-recall so far "
              "(44.7%). SECOND FINDING, which redirected the search: comparing "
              "against iter1 showed WIDTH beats TIME -- d=64/T=24 scored 83.2% vs "
              "d=40/T=24 at 78.4%, same T and depth. Led directly to iter5."),

    dict(iter=5, arch="m1_g_wide", corpus="dataset_v8_accent.npz",
         params=146_187, mmac=1.474, floor=0.384,
         tpr=0.7873, auc=0.9905, offset=0.7253,
         note="CURRENT BEST. Two changes from iter4's findings. (1) WIDTH: ffn_mult "
              "cut to 1.0 so the FFN is a 1:1 projection, which paid for d=96 -- "
              "146,187 params at the SAME 1.474 MMAC as iter3. (2) TIME-ROLL "
              "augmentation (roll_prob=0.5, applied per sample) -- the training-side "
              "twin of the offset_stress metric. Offset recall 42.99% -> 72.53% "
              "(+29.5 pts, +69% relative) with overall TPR held flat, cutting the "
              "position-robustness gap from -35.5 pts to -6.2 pts."),
]


def fmt(v, spec="%.4f"):
    return "   --   " if v is None else spec % v


def print_table():
    print("=" * 122)
    print("%-4s %-12s %-22s %8s %8s %9s %9s %8s %8s"
          % ("iter", "arch", "corpus", "params", "MMAC", "floor", "TPR@.5%",
             "AUC", "offset"))
    print("=" * 122)
    for r in ITERS:
        print("%-4s %-12s %-22s %8d %7.3fM %7.3fms %9s %8s %8s"
              % (r["iter"], r["arch"], r["corpus"], r["params"], r["mmac"],
                 r["floor"], fmt(r["tpr"]), fmt(r["auc"]), fmt(r["offset"])))
    print("=" * 122)
    print("v3 reference: 84,865 params | 11.24 MMAC | 2.93 ms floor | 480 KB RAM")
    print("  (79.8% STREAMING recall -- different metric, not comparable to TPR@.5%)")
    for r in ITERS:
        print("")
        print("iter %s  %s on %s" % (r["iter"], r["arch"], r["corpus"]))
        for line in _wrap(r["note"], 92):
            print("  %s" % line)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    print_table()