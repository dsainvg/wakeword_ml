# Iteration 5 — `m1_g_wide`: **current best**

Two changes at once, both driven by iteration 4's findings: **width** (from the
`ffn_mult` discovery) and **time-roll augmentation** (to fix offset fragility).

| | |
|---|---|
| Architecture | `m1_g_wide` |
| Corpus | `dataset_v8_accent.npz` |
| Parameters | **146,187** (+72% vs v3's 84,865) |
| MAC | 1,473,962 (1.474 MMAC) |
| Arithmetic floor | **0.384 ms** |
| Headroom in 11.11 ms budget | **28.9×** |
| Weights int8 | 142.8 KB |
| **TPR @ 0.5% FPR** | **78.73%** (threshold 0.7961) |
| AUC | 0.9905 |
| **Offset recall** | **72.53%** |
| Wall clock | 22.6 min |

Configuration: `dim=96, stem_channels=32, num_blocks=2, num_heads=4, time_stride=4, ffn_mult=1.0`

## Change 1 — width, bought with `ffn_mult=1.0`

`sweep_fine.py` (132 configs under the cap) found `d=96, b=2, ff=1.0, T=12` as the
best quality-per-MAC point. `ffn_mult=1.0` means the FFN is a plain 1:1 projection
instead of the usual 2× expansion — at T=12 the FFN was the dominant cost, so
dropping the expansion is what pays for d=96.

Result: **146,187 parameters at 1.474 MMAC** — more parameters than any prior
iteration, at the same compute as iteration 3.

## Change 2 — time-roll augmentation (the big win)

Added `augment(..., roll_prob=0.5)`: each training sample is independently rolled
to a random time offset, so every position in the 1 s window is equally likely to
carry the keyword.

This is the training-side twin of the `offset_stress` metric, and that metric was
the weakest number in every prior iteration (~43% against ~78% offset-free). The
model had never been trained on the condition it was being measured in.

Applied **per sample**, not per batch — a shared offset would leave the rest of
the batch un-augmented and teach nothing.

### The offset collapse it fixed

| iteration | offset recall | TPR | gap |
|---|---|---|---|
| iter 3 (`m1_a`, no roll) | 42.99% | 78.50% | −35.5 pts |
| **iter 5 (`m1_g_wide`, roll)** | **72.53%** | **78.73%** | **−6.2 pts** |

**+29.5 points of offset recall — a 69% relative improvement** — while overall TPR
held flat. The position-robustness gap, previously the single biggest weakness in
the project, is now largely closed.

This also matters more than the raw number suggests: the deployed 2-of-2
peak-hold detector requires two *consecutive* frames above threshold, so a model
that only fires when the keyword sits at one particular offset will miss most real
utterances.

## Where it stands vs everything measured

| | v3 | iter 1 | iter 3 | **iter 5** |
|---|---|---|---|---|
| params | 84,865 | 125,959 | 129,675 | **146,187** |
| MAC | 11.24 M | 2.616 M | 1.448 M | **1.474 M** |
| floor | 2.93 ms | 0.681 ms | 0.377 ms | **0.384 ms** |
| TPR @0.5% FPR | — | 83.22% | 78.50% | **78.73%** |
| offset recall | — | 53.36% | 42.99% | **72.53%** |

**Iteration 5 beats iteration 3 on every metric at identical cost**, and beats
iteration 1's offset recall by 19 points at 56% of the compute.

## Honest remaining gaps

1. **Iteration 1 still holds the highest TPR** (83.22% at 2.616 MMAC). The 1.5 MMAC
   cap costs ~4.5 points and no configuration tested has fully recovered it.
2. **Accent robustness is still only TTS-measured.** All human speech negatives come
   from LibriSpeech (US/UK read speech). Real accented human speech remains untested
   and is the top open gap.
3. **No on-device validation.** Every number here is offline. The audit's central
   warning — that no positive has ever been confirmed to fire on hardware — is
   untouched by this work.
4. **Offset recall at 72.5% is good, not solved.** A ~6 point gap to offset-free
   TPR remains.

## Artefacts

- `best_m1g_wide.pt` (590.8 KB)
- `models_torch_hc.py` (`ARCHS_M1["m1_g_wide"]`)
- `logs/it5_m1g.log`