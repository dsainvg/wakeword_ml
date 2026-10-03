# Iteration history — "Amaze" KWS

Every trained architecture, in order, with the measured numbers and the reasoning
that produced the next one. One file per iteration.

| iter | architecture | corpus | params | MMAC | floor | TPR@0.5%FPR | AUC | offset | [write-up](iterations/) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `kws_tight` | v6 (33k) | 16,383 | 1.128 | 0.294 ms | 15.78% | 0.9183 | 8.09% | [md](iterations/iter0_kws_tight.md) |
| 1 | `hc_balanced` | v7 (96k) | 125,959 | 2.616 | 0.681 ms | **83.22%** | 0.9940 | 53.36% | [md](iterations/iter1_hc_balanced.md) |
| 2 | `m1_a` | v7 (96k) | 129,675 | 1.448 | 0.377 ms | 78.08% | 0.9908 | 43.55% | [md](iterations/iter2_m1a_v7.md) |
| 3 | `m1_a` | v8 accent | 129,675 | 1.448 | 0.377 ms | 78.50% | 0.9909 | 42.99% | [md](iterations/iter3_m1a_accent.md) |
| 4 | `m1_e_t24` | v8 accent | 63,655 | 2.237 | 0.583 ms | 78.36% | 0.9913 | 44.73% | [md](iterations/iter4_m1e_t24.md) |
| **5** | **`m1_g_wide`** | **v8 accent** | **146,187** | **1.474** | **0.384 ms** | **78.73%** | 0.9905 | **72.53%** | **[md](iterations/iter5_m1g_wide.md)** |

**v3 reference:** 84,865 params · 11.24 MMAC · 2.93 ms floor · 480 KB RAM ·
79.8% *streaming* recall.

> v3's recall is a **streaming** figure on a 24-bucket corpus. The TPR column above
> is an **offline frame** TPR at a 0.5% false-positive budget on a 16-bucket (v4)
> corpus. The two are not directly comparable and are not tabulated together as if
> they were.

## The three findings that drove the search

1. **Data, not capacity, was the bottleneck** (iter 0 → 1). 33k → 96k samples moved
   TPR 15.78% → 83.22%. `agents.md` §9.3 had already reached this conclusion; this
   confirmed it at 100× the parameter count.

2. **Time is the cheap axis; width is the valuable one** (iter 1 → 4). Cutting T
   49 → 12 halves every block's MACs while leaving parameter counts untouched, so
   params and MACs are decoupled. But once that was exploited, iteration 4 showed
   that at matched T and depth, **width was worth ~5 TPR points** — and `ffn_mult`
   below 2.0 is the cheapest way to buy it.

3. **The model was never trained on the condition it was measured in** (iter 5).
   Offset recall sat at ~43% against ~78% offset-free in every prior iteration,
   because nothing in training randomised where the keyword sat in the window.
   Adding per-sample time-roll augmentation moved offset recall to 72.53% with
   overall TPR held flat.

## Current best

**`m1_g_wide`** (`best_m1g_wide.pt`) — 146,187 params (+72% over v3) at
**1.474 MMAC**, a **0.384 ms** arithmetic floor leaving **28.9× headroom** in the
11.11 ms budget. 78.73% TPR @ 0.5% FPR, 72.53% offset recall.

The headroom is the real argument for this operating point, not the floor itself.
The audit measured 296 ms against a 2.93 ms floor — ~101× implementation overhead,
mostly float32 on a core whose only vector unit is integer. At 0.384 ms the design
can absorb roughly 20× overhead and still land inside budget.

## Known open gaps

1. Iteration 1 still holds the highest TPR (83.22% at 2.616 MMAC). The 1.5 MMAC cap
   costs ~4.5 points and nothing tested has fully recovered it.
2. **Accent robustness is only TTS-measured.** Every human speech negative comes from
   LibriSpeech (US/UK read speech). Real accented human speech is untested.
3. **No on-device validation.** All figures are offline. The audit's central warning —
   that no positive has ever been confirmed to fire on hardware — is untouched.
4. Offset recall at 72.5% is good, not solved; a ~6 point gap to offset-free TPR remains.

## Reproducing

```bash
py download_datasets_hf.py                                    # LibriSpeech, ESC-50, UrbanSound8K
py build_noise_bank_v2.py --procedural                        # babble + device layers
py build_corpus_v2.py --profile v4 --pos 40000 --neg 80000 \
   --workers 3 --per_voice 5 --dtype fp16 --out dataset_v8_accent.npz
py models_torch_hc.py                                         # architecture table + mandate checks
py train_torch.py --data dataset_v8_accent.npz --arch m1_g_wide \
   --epochs 40 --batch 256 --roll_prob 0.5 --out best_m1g_wide.pt
py verify_checkpoint.py                                       # independent re-verification
py iteration_log.py                                           # this table
```