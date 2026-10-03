# Iteration 0 — baseline: `kws_tight`

**Status:** superseded. Kept as the low-data baseline so later gains can be attributed
to data scale rather than architecture.

| | |
|---|---|
| Architecture | `kws_tight` |
| Corpus | `dataset_v6.npz` (33,602 train / 8,398 val) |
| Parameters | 16,383 |
| MAC | 1,127,798 (1.128 MMAC) |
| Arithmetic floor | 0.294 ms |
| **TPR @ 0.5% FPR** | **15.78%** (threshold 0.7407) |
| AUC | 0.9183 |
| Offset recall | 8.09% |
| Wall clock | 5.6 min |

## What happened

First successful GPU training run. Verified end-to-end on the RTX 4050
(`device cuda`, 563 MiB VRAM, 19–23% utilisation) and independently re-verified
from the saved checkpoint: `load_state_dict(strict=True)` passes, recomputed MAC
matches to the digit, and every metric reproduces from disk.

## Why the numbers are bad

AUC 0.918 says the model *ranks* speech above noise reasonably. The TPR is
crushed by the operating point: mean P(keyword) was 0.61 on speech vs 0.31 on
noise — heavily overlapping distributions rather than separated ones.

Three causes, in order of likely impact:

1. **Corpus scale.** 33.6k training samples. `agents.md` §9.3 already concluded the
   model was data-limited, not capacity-limited.
2. **Wrong bucket profile.** `--profile v5` does not exist in this checkout
   (only v2/v3/v4), so the 16-bucket v4 corpus was used instead of the 24-bucket
   v5 corpus the production model trained on.
3. **Not converged.** AUC was still climbing at epoch 29 of 30.

## What this iteration established

The architecture, forward pass and optimiser are all sound on real audio — this is
a *data* result, not a broken-model result. That distinction is what made
iteration 1 worth running at all.

## Artefacts

- `best_kwstight.pt`
- `train_result.json` (overwritten by later runs; history for this iteration is in
  the transcript)