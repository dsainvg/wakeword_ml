# Iteration 4 — `m1_e_t24`: pricing the temporal cut

**Status:** the experiment that redirected the whole search. Its architecture lost,
but its *finding* won.

| | |
|---|---|
| Architecture | `m1_e_t24` |
| Corpus | `dataset_v8_accent.npz` |
| Parameters | 63,655 |
| MAC | 2,237,200 (2.237 MMAC) |
| Arithmetic floor | 0.583 ms |
| **TPR @ 0.5% FPR** | **78.36%** (threshold 0.8084) |
| AUC | 0.9913 |
| Offset recall | **44.73%** (best so far) |
| Wall clock | 22.5 min |

## The experiment

Iteration 3 established that cutting T to 12 was free. This iteration tests the
opposite direction: **keep T=24 (double the temporal resolution) and spend the
extra MACs.** If more temporal resolution bought accuracy, then T=12 was the
wrong trade and iteration 3 was leaving performance on the table.

Configuration: `dim=40, stem_channels=32, num_blocks=3, num_heads=4, time_stride=2, ffn_mult=3.0`

## Result: the temporal cut is free

| | iter 3 (T=12) | iter 4 (T=24) | delta |
|---|---|---|---|
| T | 12 | 24 | 2× resolution |
| MAC | 1.448 MMAC | 2.237 MMAC | **+54%** |
| floor | 0.377 ms | 0.583 ms | +55% |
| params | 129,675 | 63,655 | **−51%** |
| TPR | 78.50% | 78.36% | **−0.14 pts** |
| Offset recall | 42.99% | 44.73% | +1.74 pts |

**78.36% vs 78.50% — statistically identical, for 54% more arithmetic and half
the parameters.** Doubling temporal resolution bought nothing.

Conclusion: `m1_a`'s T=12 cut is the correct choice on a time budget, and the
saving is real rather than a compromise.

## The second finding: WIDTH > TIME

Comparing this against iteration 1 exposed something the original sweep had
missed:

| model | dim | T | blocks | MAC class | TPR |
|---|---|---|---|---|---|
| `hc_balanced` | **64** | 24 | 3 | ~2.6 MMAC | **83.22%** |
| `m1_e_t24` | **40** | 24 | 3 | ~2.2 MMAC | **78.36%** |

Same T, same depth, same MAC class — **~5 TPR points apart on width alone.**

The first sweep (`sweep_15m.py`) only varied `ffn_mult` over {2.0, 3.0, 4.0} and
`dim` over a coarse grid, so it never saw the cheap wide-and-thin configurations.
At 1.5 MMAC the FFN dominates cost (at d=64, T=24: FFN 393k MAC/block vs 295k for
attention qkv), so **`ffn_mult` below 2.0 is the cheapest way to buy width**.

This finding directly produced iteration 5's architecture.

## Artefacts

- `best_m1e_t24.pt`
- `logs/it3_t24.log`