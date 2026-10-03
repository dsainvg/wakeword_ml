# Iteration 3 — `m1_a` on the accent-rich corpus

**Status:** superseded by iteration 5. This is the controlled **data-only**
comparison against iteration 2: identical architecture, identical cost, different
data.

| | |
|---|---|
| Architecture | `m1_a` (unchanged) |
| Corpus | `dataset_v8_accent.npz` (96,000 train / 24,000 val) |
| Parameters | 129,675 |
| MAC | 1.448 MMAC |
| Arithmetic floor | 0.377 ms |
| **TPR @ 0.5% FPR** | **78.50%** (threshold 0.8172) |
| AUC | 0.9909 |
| Offset recall | 42.99% |
| Wall clock | 9.4 min |

## What changed in the data

The TTS pool was **deleted and rebuilt** at `--per_voice 5` (up from 3), spanning
**10 locales**: US, GB, AU, IN, CA, IE, NZ, ZA, NG, PH — 32 distinct neural voices.

This affects both halves of the corpus:

- **Positives**: the keyword "amaze" is spoken in every accent
- **Negatives**: the filler-word negatives (the `N0`–`N3` buckets) are also spoken
  in every accent, so the model cannot learn "foreign accent ⇒ not the keyword"

## Result

| | iter 2 (v7) | iter 3 (v8 accent) | delta |
|---|---|---|---|
| TPR @ 0.5% FPR | 78.08% | **78.50%** | +0.42 pts |
| AUC | 0.9908 | 0.9909 | +0.0001 |
| Offset recall | 43.55% | 42.99% | −0.56 pts |
| Best epoch | 36 | **30** | converged 6 epochs sooner |

## Honest assessment: a small win

**+0.42 points is real but modest**, and I should not oversell it. The reason is
that `EDGE_VOICES` in `build_corpus_v2.py` was *already* configured with all 32
voices across those 10 locales before this iteration. I increased sampling
density (per_voice 3 → 5); I did not introduce new accents.

The most useful side effect was **faster convergence** — best epoch 30 vs 36.
On a CPU budget where every epoch of retraining costs engineering time, that is
worth something.

## What this iteration did NOT solve

Accent robustness measured against **real accented human speech** is still
untested. TTS accents are not the same distribution as human ones, and all
human speech negatives in this corpus come from LibriSpeech, which is US/UK read
speech only. That remains the top open gap.

## Artefacts

- `best_m1a_accent.pt`
- `dataset_v8_accent.npz` (385.6 MB)
- Build command:
  `py build_corpus_v2.py --profile v4 --pos 40000 --neg 80000 --workers 3 --per_voice 5 --dtype fp16 --out dataset_v8_accent.npz`