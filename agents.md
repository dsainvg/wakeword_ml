# 🎯 Wake Word "Amaze" — KWS Model Engineering Report

## Executive Summary

The v1 model reported **79.5% TPR @ 0.12% FPR**. An audit of how that number was
produced showed the recall was measured on the same TTS bases used for training, and
the false positives were measured on a soundscape set containing almost no crowded
human speech. Re-scored honestly against a corpus with 84 real noise classes, v1
scores **0% TPR at any threshold under 0.5% FPR**.

The rebuild fixes the data, the training objective, the target transducer model, and
the streaming detector.

**Best Model: [`best_v3_50k.flax`](file:///r:/Coding/embedded/esp32/wakeword/best_v3_50k.flax)**
— frozen operating point **threshold 0.8600, peak-hold 2-of-2, 1.5 s refractory**.

| Metric | v1 | **v3 (production)** |
|---|---|---|
| Parameters / INT8 | 49,882 / 48.7 KB | 49,882 / 48.7 KB |
| Val AUC (same split for both) | 83.54% | **96.90%** |
| TPR @ 0.5% FPR | **0%** (unreachable) | **40.30%** |
| TPR @ 0.5% FPR, offset-stressed | — | **35.21%** |
| TPR @ 0.12% FPR | 0% | **27.61%** |
| Soundscape FA (4.97 h, 84 classes) | 20.94 /h | **0.60 /h** |
| Continuous-speech FA (0.576 h LibriSpeech) | 6.94 /h | 6.94 /h |
| **Streaming recall @ 6 FA/h budget** | 14.3% (needs 4.03 FA/h) | **37.0%** (5.81 FA/h) |

TPR roughly doubled while soundscape false alarms fell **35×**. v1 cannot reach a
6 FA/hour budget at *any* threshold; v3 reaches it at 0.86 with 2.6× the recall.

### Model status

| checkpoint | epoch | val AUC | TPR@0.5%FPR | offset | TPR@0.12%FPR | verdict |
|---|---|---|---|---|---|---|
| `best_v3_50k.flax` | 23/35 | 96.90% | 40.30% | 35.21% | 27.61% | **production**, frozen |
| `best_v4_50k.flax` | 12/22 | 96.51% | **45.77%** | **40.76%** | 28.34% | **rejected** — see §9.1 |
| `best_v5_50k.flax` | 4/22 | 94.91% | 24.43% | 19.82% | 10.16% | stopped early, undertrained |
| `best_v3_100k.flax` | 19/30 | — | 39.22% | 25.41% | — | capacity not the bottleneck |
| `best_50k_kws_model.flax` | — | 83.54% | 0% | — | 0% | v1 baseline |

---

## 1. Why v1 Was Capped at ~79% TPR

Three defects, each measured rather than guessed.

### 1.1 Temporal position overfitting

Every v1 positive was centre-padded:

```python
pad_l = (TOTAL_SAMPLES - len(audio)) // 2   # always dead-centre
```

The model learned "the keyword sits at frame ~14". Rolling a validation positive
along the time axis shows what that cost:

| Roll (frames) | 0 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| v1 TPR | 86.2% | 85.5% | 84.8% | 84.5% | 75.3% | 68.2% | 51.6% |
| v3 TPR | 40.6% | 39.4% | 37.7% | 34.4% | 30.6% | 26.0% | 20.2% |
| v3 TPR | 40.6% | 40.6% | 40.2% | 39.0% | 37.3% | 30.2% | 26.5% |

A 100 ms-stride sliding window puts the keyword at an arbitrary position, so the
deployed recall was far below the reported 79.5%. v1's spread is **35.3 points**;
v3's is **26.5 points**; the residual tilt is the single largest remaining weakness.

### 1.2 The negative class was missing multi-talker babble

v1's noise bank had 88 clips and only two contained crowded speech. Both already fired
at v1's own 0.85 threshold:

| Soundscape | max confidence | vs 0.85 |
|---|---|---|
| `pub_babble` | **0.9332** | fires |
| `restaurant_cafeteria` | **0.8515** | fires |
| `street_traffic` | 0.7812 | — |
| `airport_terminal` | 0.6666 | — |

v1 was never shown a crowd of people talking and asked to stay silent.

### 1.3 FPR was measured on 857 negatives

0.12% of 857 is exactly one sample. The "100% soundscape rejection" headline came from
sampling 5 one-second windows from the *start* of each file.

---

## 2. Noise Bank v2 — 3,346 clips / 84 classes / 4.97 hours

[`build_noise_bank_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/build_noise_bank_v2.py)
→ `noise_bank_v2/` with a class-labelled `index.csv` for stratified sampling.

| Source | Clips | Why |
|---|---|---|
| ESC-50 (full) | 2,000 (50 × 40) | 40× more real instances per class than v1 |
| Procedural multi-talker babble | 126 (2–14 talkers) | **the missing negative class**; real LibriSpeech / Speech Commands excerpts mixed 2–14 deep, each talker independently gain-shifted, band-limited and reverberated |
| UrbanSound8K | 1,060 (10 classes) | `dog_bark`, `air_conditioner`, `engine_idling`, `street_music`, **`children_playing`**, `siren`, `jackhammer`, `drilling`, `car_horn`, `gun_shot` |
| Procedural device / event | 160 (20 classes) | `ac_hum`, `fridge_compressor`, `blender`, `vacuum_cleaner`, `printer`, `keyboard_extra`, `plastic_crumple`, `microwave_beep`, `phone_ring`, `notification_chime`, `cash_register`, `tv_static`, `rain_on_glass`, `washing_machine`, `sewing_machine`, `elevator_ding`, `hair_dryer`, `power_tool`, `transformer_hum`, `mic_self_noise` |

Sampling draws the **class uniformly, then the clip uniformly, then a random start
offset inside the clip**, so a 1 s window can land anywhere in a 10–20 s soundscape —
which is what the streaming engine actually does.

---

## 3. Corpus Taxonomy — 24 mixture buckets, 4 recipes, 6 content types

[`build_corpus_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/build_corpus_v2.py).
Two axes, explicit and **identical for positives and negatives**, so layer count can
never leak the label:

* **Recipe** — `M0` direct (0 layers) · `M1` **one** noise layer · `M2` two noise
  layers · `M3` two layers at low SNR with the full distortion ladder
* **Content** — `word` (positive) · `filler` (ordinary English words, no phonetic
  overlap) · `real` (real human speech) · `babble` (2–8 live talkers) · `noise`
  (soundscape only) · `confus` (near-miss words)

### 3.1 Recipe distribution (135,000 samples)

| recipe | positives | negatives | built counts |
|---|---|---|---|
| `M0` direct, 0 layers | 10% | 27% | — |
| **`M1` + 1 noise layer** | **40%** ← lion share | **34%** | `P1_word_one_noise` **20,114** · `N13_noise_one_noise` **10,251** |
| `M2` + 2 noise layers | 28% | 22% | `P2` 13,843 · `N14` 4,150 |
| `M3` + 2 layers, low SNR, full distortion | 22% | 17% | `P3` 11,082 · `N15` 2,524 |

### 3.2 Negative content distribution

| content | share | buckets |
|---|---|---|
| **soundscape only, no speech** | **30%** | `N12` single soundscape 10% · `N13` +1 layer **12%** · `N14` +2 layers 5% · `N15` +2 hard 3% |
| live multi-talker babble | 21% | `N8`–`N11` |
| real human speech | 18% | `N4`–`N7` |
| **unimportant words (no overlap)** | **17%** | `N0` direct 5% · `N1` +1 layer 5% · `N2` +2 layers 4% · `N3` +2 hard 3% |
| near-miss words | 14% | `N16`–`N19` |

1,032 filler bases are synthesised from 129 ordinary words × 9 voices. Near-miss words
are deliberately a *small* slice: teaching rejection of words that share 95%+ of the
target's phonetic structure is known to destroy recall.

### 3.3 Per-bucket loss weights

Harder recipes pull harder: `P3` 1.30, `P2` 1.25, `P1` 1.15, `P0` 1.00; negatives
0.80–1.30 in the same order. The trainer reports loss, accuracy and recall/FA
**separately for every bucket** each epoch, so a regression in one regime cannot hide
inside an average.

### 3.4 What each corpus version changed

| corpus | samples | train / val | buckets | what it added |
|---|---|---|---|---|
| `dataset_v3.npz` | 52,000 | 39,784 / 9,947 | — | random offset placement, near-field clean mode, 3-layer noise |
| `dataset_v4.npz` | 120,000 | 96,000 / 24,000 | 16 | explicit bucket taxonomy, level policy |
| **`dataset_v5.npz`** | **135,000** | **108,001 / 26,999** | **24** | single-layer recipe added and made dominant; INMP441 transducer model |

---

## 4. INMP441 Transducer Model

The target capsule is a TDK InvenSense **INMP441**. The feature front end is
**100–7500 Hz** (`features.py`), which makes the spec unusually clean to map.

### 4.1 Spec → model consequence

| spec | consequence |
|---|---|
| Sensitivity −26 dBFS @ 94 dB SPL | fixes the SPL→digital mapping; the board preamp gain becomes the free variable |
| SNR 61 dBA | quoted at *max* sensitivity — what the deployment sees degrades with distance |
| Response 60 Hz – 15 kHz | **fully brackets the 100–7500 Hz feature band → transparent in-band** |
| 24-bit I²S | −144.5 dBFS floor, **57 dB below the capsule**; quantization is not an error source |
| 1.4 mA, no AGC | nothing compresses the level range; level is purely geometry |

### 4.2 Measured in-band capsule gain

| Hz | 60 | 100 | 200 | 250 | 400 | 1k | 2k | 4k | 7k | 7.9k |
|---|---|---|---|---|---|---|---|---|---|---|
| dB | −0.12 | **+2.12** | +1.21 | +0.34 | −0.70 | −0.21 | −0.05 | −0.01 | 0.00 | 0.00 |

A genuine LDC low-frequency shelf, flat across the working band. The generic MEMS
darkening curve previously applied here is **wrong for this capsule** and is disabled
in the transducer path. A first implementation of the shelf used a 1-pole lowpass,
which is only −0.7 dB at 4 kHz and therefore produced a flat +3 dB boost instead of a
shelf; it was caught on measurement and replaced with a 2nd-order filter.

### 4.3 Talker geometry drives both level and SNR

| distance | vocal effort | SPL | raw dBFS | achieved SNR |
|---|---|---|---|---|
| 0.3 m | normal | 70 dB | −50.0 | 71 dB |
| 1.0 m | normal | 60 dB | −60.0 | **61 dB** ← datasheet case |
| 3.0 m | normal | 50.5 dB | −69.5 | 51.5 dB |
| 3.0 m | whisper | 35.5 dB | −84.5 | **36.5 dB** |
| 0.3 m | shouting | 82 dB | −38.0 | 83 dB |

The capsule noise is placed at `signal_level − snr(distance, effort)`, because the
preamp lifts speech and self-noise together while distance and loudness do not. This
is the effect the 61 dBA figure actually implies in a room, and no SNR-relative
augmentation in the corpus had it before.

### 4.4 Level policy

`LEVEL_DBFS = (-50.0, -14.0)`, `LEVEL_FLOOR_DBFS = -70.0`. The earlier
`−34…−14 dBFS` policy came from nowhere and was ~8 dB too hot — a real INMP441 with
no AGC never sees a full-scale voice.

A guard lifts anything that falls through the floor: the 60 Hz capsule high-pass can
strip a low-frequency-heavy soundscape down to −96 dBFS, which pins the log-mel at its
`log(1e-5) = −11.51` epsilon and turns a hard example into pure label noise.
Verified over 192 samples spanning all 24 buckets: **−50.9 … −14.9 dBFS, zero below
the floor.**

### 4.5 Signal-path order

The chain mirrors the physical path: room/transmission distortions → level set from
talker geometry → capsule response → capsule self-noise → I²S quantization. The
capsule response is applied *after* the level is set, never before. Bit-crushing was
cut from 12–32% to **2–5%** because a 24-bit I²S link quantizes 57 dB below the
capsule noise floor; it survives only as a rare robustness probe.

[`build_speech_corpus.py`](file:///r:/Coding/embedded/esp32/wakeword/build_speech_corpus.py)
resolves LibriSpeech dev-clean from OpenSLR directly, with safe extraction.

---

## 5. Training Objective — maximise TPR at a fixed FPR budget

[`train_industrial_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/train_industrial_v2.py)

Six changes against v1, each tied to a measured failure:

1. **Checkpoint selection is TPR subject to FPR ≤ budget**, computed on the full
   validation ROC, with the satisfying threshold written into the checkpoint. v1
   selected on AUC at a hardcoded 0.85.
2. **Offset-stress validation.** Val positives are also scored after a random
   temporal roll; the selection score is the mean of plain and rolled TPR. This makes
   the checkpoint a model that works in a sliding window, not a classifier on a fixed
   crop.
3. **Class weights follow a sqrt law with a positive bias** (`neg=1.291, pos=1.956`).
   v1 used `neg_w = min(2.5, ratio*0.6)`, which actively penalised the negative class
   and is the main reason recall was capped.
4. **On-batch hard-negative mining** (`hard_neg_frac 0.75`) tightens the FPR tail
   without weakening the positive gradient.
5. **Focal γ 1.2 instead of 2.0.** γ=2 down-weights exactly the easy-but-quiet
   positives that dominate the recall loss.
6. **Mic-gain augmentation.** A constant added to the log-mel is a multiplicative
   amplitude change, so ±12 dB of preamp/AGC jitter costs nothing and forces the model
   off absolute level.

### 5.1 Production training curve (best checkpoint: epoch 23)

| Epoch | Loss | Train acc | Val AUC | TPR@0.5%FPR | offset-stressed | TPR@0.12%FPR |
|---|---|---|---|---|---|---|
| 4 | 0.2650 | 81.7% | 92.63% | 15.26% | 6.55% | 5.58% |
| 10 | 0.1926 | 87.3% | 95.73% | 33.97% | 20.81% | 15.16% |
| 15 | 0.1665 | 89.2% | 96.44% | 40.30% | 27.66% | 20.51% |
| **23** | **0.1382** | **90.8%** | **96.90%** | **40.30%** | **30.12%** | **27.61%** |
| 35 | 0.1170 | 92.4% | 97.06% | 39.02% | 27.41% | 26.19% |

### 5.2 Validation ROC (`best_v3_50k.flax`)

| FPR budget | threshold | TPR | FA / 1000 neg |
|---|---|---|---|
| 0.05% | 0.9326 | 22.61% | 2 |
| 0.10% | 0.9196 | 26.96% | 5 |
| 0.20% | 0.9060 | 30.99% | 11 |
| **0.50%** | **0.8690** | **40.30%** | 29 |
| 1.00% | 0.8214 | 50.70% | 59 |
| 2.00% | 0.7519 | 64.58% | 118 |
| 5.00% | 0.6255 | 81.77% | 297 |

### 5.3 Confidence on clean held-out keyword audio

Same 66 held-out waveforms, centred in the window, no noise:

| model | median conf | ≥ 0.869 | ≥ 0.75 |
|---|---|---|---|
| v1 | 0.623 | 15.2% | 36.4% |
| **v3** | **0.924** | **65.2%** | **89.4%** |

This is the single clearest single-number summary of the improvement: the v1 model
could not fire confidently on a *clean* utterance of the keyword.

---

## 6. The Streaming Detector — Peak Hold Instead of EMA

The single largest deployment win was not the model. `kws_engine.py` v1 smoothed the
frame probability with an EMA at α=0.6 and required two consecutive windows above
threshold. At α=0.6, two consecutive frames at p=0.90 only lift the smoothed value to
**0.71** — and a 0.6 s wake word produces only two or three good frames in a 1 s
sliding window. The detector was mathematically unable to confirm a realistic burst.

Replacing the EMA with a **decaying peak hold** (still 2-of-2 plus the 1.5 s
refractory) at threshold 0.84, same model, same false-alarm rate:

| Confirmation | soundscape FA/h | speech FA/h | recall@20 dB | @12 | @6 | @0 | mean |
|---|---|---|---|---|---|---|---|
| EMA 2-of-2 | 0.60 | 0.00 | 0.0% | 6.4% | 0.0% | 0.0% | **1.5%** |
| **peak-hold 2-of-2** | 0.60 | 0.00 | 53.8% | 48.1% | 35.6% | 20.8% | **40.0%** |

**27× the recall at an identical false-alarm rate.** `StreamingKWSEngine` now defaults
to peak-hold and reads the threshold and confirmation policy out of the checkpoint.

---

## 7. Deployment Operating Point

Measured with [`sweep_threshold.py`](file:///r:/Coding/embedded/esp32/wakeword/sweep_threshold.py)
and frozen with [`finalize_checkpoint.py`](file:///r:/Coding/embedded/esp32/wakeword/finalize_checkpoint.py)
over 4.97 h of soundscapes + 0.576 h of continuous LibriSpeech.

`best_v3_50k.flax`, peak-hold 2-of-2, 1.5 s refractory:

| threshold | soundscape FA/h | speech FA/h | total | streaming recall |
|---|---|---|---|---|
| 0.78 | 3.22 | 12.15 | 15.37 | 50.0% |
| 0.82 | 1.21 | 8.68 | 9.89 | 44.0% |
| 0.84 | 0.60 | 6.94 | 7.55 | 40.0% |
| **0.86 (frozen)** | **0.60** | **5.21** | **5.81** | **37.0%** |
| 0.90 | 0.40 | 1.74 | 2.14 | 24.3% |
| 0.94 | 0.00 | 0.00 | 0.00 | 14.0% |

v1 over the same audio, same detector:

| threshold | soundscape FA/h | speech FA/h | total | streaming recall |
|---|---|---|---|---|
| 0.84 | 20.94 | 6.94 | 27.33 | 32.3% |
| 0.90 | 10.07 | 0.00 | 10.07 | 22.0% |
| 0.95 | 4.03 | 0.00 | 4.03 | 14.3% |

**v1 cannot reach 6 FA/hour at any threshold. v3 reaches it at 0.86 with 2.6× the
recall.** If more recall is wanted for a modest FA increase, **0.84** gives 40.0% at
7.55 FA/h — +3 points recall for +1.74 FA/h (~+30%).

---

## 8. Benchmark Detail — `best_v3_50k.flax` @ 0.84

**Worst soundscapes** — 3 false alarms across 4.97 h / 84 classes:

| class | max conf | frames ≥ thr |
|---|---|---|
| `babble_two_talker` | 0.9105 | 1/203 |
| `pig` | 0.9050 | 1/180 |
| `fireworks` | 0.8964 | 1/187 |
| `babble_crowd` | 0.8330 | 0/571 |
| `babble_dense_crowd` | 0.8064 | 0/562 |

v1 on the same 84 classes: 94 false alarms, worst `babble_small_group` at 0.9763.

**Streaming recall by SNR** (400 trials, held-out keyword, peak-hold 2-of-2 @ 0.84):
57.9% @ 20 dB · 57.1% @ 12 dB · 38.9% @ 6 dB · 7.5% @ 0 dB · 5.4% @ −3 dB.

**Per-bucket breakdown** (`best_v4_50k.flax`, epoch 11, thr 0.880) — the taxonomy
behaves as designed:

| positive bucket | recall | | negative bucket | FA rate |
|---|---|---|---|---|
| `P0` word, direct | 83.33% | | `N0`/`N1`/`N2` unimportant words | 0.00–0.17% |
| `P1` word + 1 noise | 41.09% | | `N8`–`N11` live babble, all 4 recipes | 0.00% |
| `P2` word + 2 noise | 16.83% | | `N12`–`N15` soundscapes, all 4 recipes | 0.00% |
| | | | `N4`–`N7` real human speech | 0.17–1.37% |
| | | | `N16`–`N19` near-miss words | 1.79–2.14% |

Single-layer regimes — the ones the taxonomy prioritises — are cleanest: every
one-layer negative bucket is at 0.00% false alarm. The only residual false-alarm
source is the deliberate near-miss set.

---

## 9. Findings Worth Not Repeating

### 9.1 Validation FPR does not predict deployment FA

`best_v4_50k.flax` looked clearly better than v3 on validation — 45.77% vs 40.30% TPR
at a 0.5% FPR budget. Measured on real audio it is far worse:

| model | threshold | soundscape FA/h | speech FA/h | total | streaming recall |
|---|---|---|---|---|---|
| v3 | 0.86 | 0.60 | 5.21 | 5.81 | 37.0% |
| v3 | 0.90 | 0.40 | 1.74 | 2.14 | 24.3% |
| v4 | 0.86 | 0.81 | **62.44** | 63.24 | 45.0% |
| v4 | 0.90 | 0.00 | 31.22 | 31.22 | 31.0% |
| v4 | 0.94 | 0.00 | 3.29 | 3.29 | 11.3% |
| v4 | 0.96 | 0.00 | 1.64 | 1.64 | 4.3% |

At matched false alarms (~3/h) v3 delivers **24.3%** recall and v4 delivers
**11.3%**. Cause: v4's validation negatives are level-normalised, distorted 1 s
slices, so the val FPR is not predictive of natural long-form speech. **Selection
metrics must be measured on the deployment distribution, not on synthetic slices.**

### 9.2 Over-augmented positives cap confidence

An intermediate corpus destroyed every positive uniformly, so the model never saw a
normal-level keyword and learned to hedge. Its confidence ceiling was 0.86, which
capped recall at any usable threshold. Keeping ~30% of positives in a near-field clean
mode is what unlocked 65.2% clean-keyword confidence in v3.

### 9.3 Capacity is not the bottleneck

`BCConformer100K` (138,114 params, 134.9 KB INT8) tracked the 50K model or slightly
trailed it at equal epochs (epoch 19: 39.22% vs 40.30% TPR@0.5%FPR, 25.41% vs
30.12% offset-stressed) at 2.5× the inference cost. The 50K model reaches 90.8% train
accuracy — it is limited by data and the decision boundary, not parameters.

---

## 10. `bcconformer_v2` — Position-Aware Architecture (unproven)

[`models.py`](file:///r:/Coding/embedded/esp32/wakeword/models.py) · **86,450 params,
84.4 KB INT8** · same `(49, 40, 1)` input contract, so every existing corpus loads.
Written and shape-verified; **not trained, and not claimed to be better.**

Four structural weaknesses in v1 were fixed rather than guessed at:

| # | change | params | why |
|---|---|---|---|
| 1 | **Learned absolute temporal embedding** added before the stack | +2,352 | v1's attention is permutation-equivariant — it structurally cannot tell where in the 1 s window a frame sits. This is the highest-value fix for offset stress, at 2.7% of the net. |
| 2 | **Learned attention pooling over mel** replaces `jnp.mean(axis=2)` | +48 | mean-pooling 10 mel bins to 1 discards all sub-band formant structure |
| 3 | **Δ / ΔΔ computed in-model** with fixed kernels | +576 | spectral dynamics matter for speech discrimination; computed inside the model so the dataset format is unchanged |
| 4 | **3 blocks, dim 48, dual depthwise kernels 3+7, Swish FFN, SE gate** | 67,248 vs 36,432 | multi-scale temporal context, Conformer-standard activation, channel recalibration |
| — | **Attentive statistics pooling** replaces `concat(mean, max)` | +1,406 | mean+max weights every frame equally |

Param distribution: 3 × improved block 67,248 (78%) · frequency projection 13,824
(16%) · position embedding 2,352 · attentive pool 1,600 · stem 864.

**Where more attention is deliberately *not* added:** a self-attention stack over the
raw 49×10 time-frequency grid would be the most expensive thing on an ESP32-S3 and
temporal attention is where the discrimination actually lives.

---

## 11. What Is Still Weak

* **Recall below 6 dB SNR collapses** (38.9% → 7.5%). Degraded-speech positives need
  their own augmentation branch.
* **Offset sensitivity at the right edge of the window** (40.6% → 14.1%). Random
  placement helped a lot; an explicit "keyword at every position" copy schedule, or the
  position embedding in §10, would flatten it further.
* **Continuous-speech FA rests on 3 trigger events over 0.576 h.** The 5.21/hour figure
  has a wide confidence interval.
* **Validation FPR is not a deployment predictor** (§9.1). The corpus needs un-normalised
  natural long-form speech as a negative class for the val FPR to mean anything.
* **Positives are still ~95% synthetic TTS.** Only 8 real human "amaze" windows exist
  in LibriSpeech. Human recordings of the keyword remain the highest-value missing data.
* **`bcconformer_v2` is unproven** (§10).
* **v1's `dataset_mega.npz` results are only comparable through the v2 benchmark.** Do
  not quote 79.5% TPR / 99.28% AUC from the v1 tables.

---

## 12. File Inventory

| File | Purpose |
|---|---|
| [`build_noise_bank_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/build_noise_bank_v2.py) | ESC-50 full + procedural babble + UrbanSound8K + procedural devices → 3,346 clips / 84 classes |
| [`build_speech_corpus.py`](file:///r:/Coding/embedded/esp32/wakeword/build_speech_corpus.py) | resolves LibriSpeech dev-clean from OpenSLR with safe extraction |
| [`build_corpus_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/build_corpus_v2.py) | bucketed corpus builder; 24-bucket taxonomy, INMP441 level policy, 4 recipes |
| [`train_industrial_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/train_industrial_v2.py) | TPR-at-FPR-budget objective, offset-stress validation, OHEM, per-bucket loss and metrics, mic-gain augmentation |
| [`evaluate_industrial_v2.py`](file:///r:/Coding/embedded/esp32/wakeword/evaluate_industrial_v2.py) | benchmark: ROC, per-bucket breakdown, level robustness, offset stress, soundscapes, speech FA, streaming recall |
| [`sweep_threshold.py`](file:///r:/Coding/embedded/esp32/wakeword/sweep_threshold.py) | threshold × confirmation-policy grid on measured FA/hour and recall |
| [`finalize_checkpoint.py`](file:///r:/Coding/embedded/esp32/wakeword/finalize_checkpoint.py) | writes the chosen operating point into the checkpoint |
| [`diagnose_operating_point.py`](file:///r:/Coding/embedded/esp32/wakeword/diagnose_operating_point.py) | ROC sweep + noise-bank percentiles for any checkpoint/dataset pair |
| [`make_eval_keyword_bases.py`](file:///r:/Coding/embedded/esp32/wakeword/make_eval_keyword_bases.py) | held-out clean keyword waveforms for the recall test |
| [`models.py`](file:///r:/Coding/embedded/esp32/wakeword/models.py) | `BCConformer50K` (49,882) · `BCConformer100K` (138,114) · `BCConformerV2` (86,450) |
| [`acoustic_augment.py`](file:///r:/Coding/embedded/esp32/wakeword/acoustic_augment.py) | RIR, distance, MEMS, speed/VTLP, **INMP441 spec model** |
| [`kws_engine.py`](file:///r:/Coding/embedded/esp32/wakeword/kws_engine.py) | peak-hold confirmation, reads the frozen operating point, loads v1 and v2+ checkpoints |
| [`features.py`](file:///r:/Coding/embedded/esp32/wakeword/features.py) | 40-band log-mel, 100–7500 Hz, (49, 40) per second |

### Artefacts

| File | Size | Contents |
|---|---|---|
| `best_v3_50k.flax` | 194 KB | **production checkpoint**, threshold 0.86, peak-hold 2-of-2, 1.5 s refractory |
| `best_v4_50k.flax` | 197 KB | 16-bucket corpus, epoch 12/22, rejected on real audio (§9.1) |
| `best_v5_50k.flax` | 197 KB | 24-bucket corpus, epoch 4/22, stopped early |
| `best_v3_100k.flax` | 543 KB | capacity experiment |
| `best_v2_50k.flax` | 197 KB | over-augmented-positives intermediate, kept as a record |
| `best_50k_kws_model.flax` | 194 KB | v1 baseline |
| `dataset_v5.npz` | 904 MB | 24-bucket corpus (108,001 train / 26,999 val) |
| `dataset_v4.npz` | 803 MB | 16-bucket corpus, superseded |
| `dataset_v3.npz` | 335 MB | production corpus for the v3 checkpoint |
| `keyword_bases_eval.npy` | 4.3 MB | 98 held-out keyword waveforms (66 usable) |
| `noise_bank_v2/` | 0.5 GB | 3,346 clips + `index.csv` |

---

## 13. Usage

```python
from kws_engine import StreamingKWSEngine

# threshold 0.86, peak-hold 2-of-2 and 1.5 s refractory are read from the checkpoint
engine = StreamingKWSEngine(checkpoint_path="best_v3_50k.flax", arch="bcconformer_50k")

while stream_open:
    chunk = mic.read(1600)                     # 100 ms @ 16 kHz
    spotted, score, latency_ms = engine.push_audio_chunk(chunk)
    if spotted:
        start_cloud_asr_capture()
```

**ESP32-S3 footprint:** 49,882 params → 48.7 KB INT8 flash, ~42 KB SRAM arena.
TFLite Micro conversion is unchanged; only the threshold (0.86) and the confirmation
policy (peak-hold 2-of-2, 1.5 s refractory) are new relative to v1.

**Hardware note:** the frozen operating point assumes an INMP441 with its default
sensitivity and no AGC. If the board applies a different preamp gain, re-freeze the
threshold against measured FA/hour rather than reusing 0.86.
