# Iteration 1 — `hc_balanced`: data scale is the bottleneck

**Status:** superseded by iteration 5 on the time budget, but still the highest
TPR measured. Keep it if accuracy matters more than CPU.

| | |
|---|---|
| Architecture | `hc_balanced` |
| Corpus | `dataset_v7.npz` (96,000 train / 24,000 val) |
| Parameters | 125,959 |
| MAC | 2,615,506 (2.616 MMAC) |
| Arithmetic floor | 0.681 ms |
| **TPR @ 0.5% FPR** | **83.22%** (threshold 0.8457) |
| TPR @ 1% FPR | 88.53% |
| TPR @ 2% FPR | 93.77% |
| AUC | 0.9940 |
| Offset recall | 53.36% |
| Wall clock | 23.5 min |

## The architecture

The first "high capacity" design, built on the observation that **params and MACs
are decoupled along the time axis**.

v3 runs 3 blocks at width 48 over all 49 frames. Each block costs `T·d·hidden` on
the FFN and `h·T²·dh` on attention, at T=49. Halving T before the stack halves
every block's arithmetic while leaving parameter counts untouched — a Linear's
parameters depend on `d`, not `T`.

Measured on this machine (`explore_arch.py`, 32 configs, analytic MAC walker):

| T | params | MAC | floor |
|---|---|---|---|
| 49 | 124,583 | 4.053 MMAC | 1.055 ms |
| **24** | **125,959** | **2.616 MMAC** | **0.681 ms** |
| 12 | 125,959 | 1.532 MMAC | 0.399 ms |

Nearly identical parameter count, 2.6× cheaper. The freed budget buys width.

## Result: +67 TPR points from data alone

15.78% → **83.22%**, at similar capacity (16k → 126k params, so not a pure
data comparison — but iteration 2 isolates the data effect properly).

Independently re-verified from the checkpoint: strict load passes, recomputed MAC
matches exactly (2,615,506), saved threshold 0.8457 vs recomputed 0.8457.

## Design mandates (enforced as executable assertions, not comments)

- No absolute position embedding — it measured 77.7% → 15.1% recall across the window
- Relative position bias `b[i-j]` only (shift-invariant)
- Soft-OR pooling (mean + logsumexp + max), permutation-invariant over time —
  verified numerically
- Learned attention pooling over mel bins (mean-pooling discards formant structure)
- `DeltaStack` with 0 learned parameters
- Depthwise-separable convs throughout

## Against v3

| | v3 | iter 1 |
|---|---|---|
| params | 84,865 | 125,959 (+48%) |
| MAC | 11.24 MMAC | 2.616 MMAC (−77%) |
| floor | 2.93 ms | 0.681 ms (4.3× faster) |

**Honest caveat:** v3's 79.8% is a *streaming* recall figure on a 24-bucket corpus.
This is an *offline frame* TPR on a 16-bucket corpus. Not directly comparable, and
I have not pretended otherwise.

## Artefacts

- `best_hc_balanced.pt`
- `models_torch_hc.py` (`ARCHS_HC`)