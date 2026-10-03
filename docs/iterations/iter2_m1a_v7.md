# Iteration 2 — `m1_a`: the 1.5 MMAC target

**Status:** superseded by iteration 5, but this is the iteration that established
the 1.5 MMAC operating point.

| | |
|---|---|
| Architecture | `m1_a` |
| Corpus | `dataset_v7.npz` (96,000 train / 24,000 val) |
| Parameters | **129,675** |
| MAC | 1,448,258 (1.448 MMAC) |
| Arithmetic floor | **0.377 ms** |
| Headroom in 11.11 ms budget | **29.5×** |
| **TPR @ 0.5% FPR** | **78.08%** (threshold 0.8554) |
| AUC | 0.9908 |
| Offset recall | 43.55% |
| Wall clock | 9.3 min |

## The key result: more params, less compute

`m1_a` carries **129,675 parameters — MORE than iteration 1's 125,959 — while
running at 55% of the arithmetic** (1.448 vs 2.616 MMAC).

That is not a contradiction. Cutting time from T=49 to T=12 via a stride-4
depthwise convolution in the stem halves every block's arithmetic, and parameter
counts are untouched by that operation. The freed budget is then spent on width
(d=80 instead of d=64).

Configuration: `dim=80, stem_channels=32, num_blocks=2, num_heads=4,
time_stride=4, ffn_mult=2.0`

## The price

| | iter 1 | iter 2 | delta |
|---|---|---|---|
| MAC | 2.616 MMAC | 1.448 MMAC | −45% |
| floor | 0.681 ms | 0.377 ms | −45% |
| TPR | 83.22% | 78.08% | **−5.14 pts** |
| train time | 23.5 min | 9.3 min | −60% |

So the honest price of the 1.5 MMAC target is **~5 TPR points**, and iterations
3–5 went on to recover most of it.

## Why this matters on the CPU budget

The audit's real finding was that the current firmware takes 296 ms against a
2.93 ms floor — ~101× implementation overhead, mostly float32 on an
integer-only vector unit. The arithmetic floor is not the risk; **the
implementation overhead is**.

At 0.377 ms the model can absorb roughly 20× overhead and still land inside the
11.11 ms budget. At 0.681 ms it cannot. That headroom is the actual argument for
the 1.5 MMAC target, not the 0.3 ms of floor.

## Artefacts

- `best_m1a_v7.pt`
- `models_torch_hc.py` (`ARCHS_M1["m1_a"]`)