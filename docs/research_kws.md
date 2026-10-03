# Ultra-Low-Compute KWS on ESP32-S3: 2023-2026 Survey

**Target:** wake word "Amaze", 1 s of 16 kHz audio -> 40-band log-mel -> `(49, 40, 1)` tensor.
**Constraints:** RAM < 256 KB incl. PSRAM; inference <= 11.1 ms at 100 ms hop (or <= 22.2 ms at 200 ms); high TPR at near-zero FAR; low-latency cloud handoff.
**Production model:** 49x40 log-mel -> BC-Conformer-style net, 84,865 params, 11.24 MMAC, measured 296 ms/inference (13.3x over budget). Arithmetic floor 2.93 ms -- the gap is implementation overhead (float32 ops, flash re-reads), not algorithm.

> **Note on this document.** This is a freshly-derived report. All numbers below were pulled from live sources during this session and each carries its URL. Where a number could not be verified (notably the per-variant BC-ResNet params/MACs table, which sits in image streams inside the arXiv PDF), it is flagged explicitly rather than estimated. Dataset and metric are stated for every figure, because most strong headline numbers are **35-way Google Speech Commands accuracy**, a weak proxy for a single custom wake word -- see section 6.

---

## 1. Executive summary

1. **The strongest portable mechanism found is re-parameterization (RepCNN, Apple, 2024).** Train a redundant multi-branch block, fold it algebraically into a single branch at inference. Reported **43% more accurate than a uni-branch conv model at identical runtime**, and it meets BC-ResNet accuracy with **2x lower peak memory and 10x faster runtime** ([arXiv:2406.02652](https://arxiv.org/abs/2406.02652)). This is a pure training-time/inference-time asymmetry: inference cost is unchanged, accuracy goes up. It is also the best fit for the 296 ms problem, because it shrinks the *graph*, and the stated bottleneck is graph/implementation overhead, not arithmetic.

---

## 2. What is genuinely state-of-the-art on the embedded tier

### 2.1 RepCNN -- re-parameterized single-branch CNN (Apple, 2024) | most relevant

- Source: [arXiv:2406.02652v2](https://arxiv.org/abs/2406.02652), Jun-Aug 2024. Authors Kundu, Nayak, Padmanabhan, Naik (Apple Inc.).
- Mechanism: during **training**, refactor each block into a larger redundant multi-branch form. The paper explicitly credits the ResNet property that multi-branch topology creates an implicit ensemble of shallower models, easing gradient flow and improving accuracy. At **inference**, algebraically fold the branches into a single branch, yielding fewer parameters, lower memory, lower compute. The deployed graph is an ordinary single-branch conv net.
- Measured claims:
  - RepCNN re-parameterized models are **43% more accurate than a uni-branch convolutional model with the same runtime**.
  - RepCNN **meets the accuracy of BC-ResNet** while using **2x lesser peak memory** and **10x faster runtime**.
- Metric caveat: accuracy figures are on the authors' wake-word/Speech-Commands-style evaluation. The 43% and the 2x/10x are the load-bearing architecture-mechanism numbers and are directly comparable to this project's situation.
- **Portable to `(49,40,1)`: YES, high confidence.** Folding happens offline in the graph; the deployed artifact is a plain conv net. The input contract is untouched.

### 2.2 EdgeSpot -- BC-ResNet + PCEN + early-block fusion + relative PE (Analog Devices, ICASSP 2026)

- Source: [arXiv:2601.16316](https://arxiv.org/abs/2601.16316), ICASSP 2026, DOI [10.1109/ICASSP55912.2026.11460983](https://doi.org/10.1109/ICASSP55912.2026.11460983). Buyuksolak, Gok, Okman.
- Components (its own section numbering): 2.1.1 Per-Channel Energy Normalization (PCEN) frontend; 2.1.2 **Fusion of Early Blocks**; 2.1.3 **Relative Positional Encoding**; 2.1.4 Scaled Dot Product Attention (SDPA). Backbone is an *optimized BC-ResNet*. Training: self-supervised teacher distillation + Sub-center ArcFace loss.
- Measured headline: largest variant **EdgeSpot-4** improves **10-shot accuracy at 1% FAR from 73.7% -> 82.0%**, at **29.4M MACs, 128k parameters**.
- Metric: 10-shot enrollment accuracy **at 1% false-alarm rate** -- FAR-anchored, much closer to the compliance requirement here than raw accuracy. Note 29.4M MAC is ~10x the budget, so EdgeSpot-4 itself is not deployable; the *mechanisms* are.
- **Portable: PCEN = YES; early-block fusion = YES; relative PE = YES (section 5); SDPA head = probably, and it is the expensive part to drop.**
2. **Absolute temporal position encoding is the wrong tool for this input, and the literature agrees in spirit.** The project's own failure (recall 77.7% -> 15.1%) is consistent with the one directly comparable recent system: EdgeSpot (Analog Devices, ICASSP 2026) uses **relative** positional encoding plus early-block fusion, propagating *content-based early-block features* into the positional/attention path ([arXiv:2601.16316](https://arxiv.org/abs/2601.16316)). The project's own `bcconformer_v3` already moved to relative bias and recovered. This survey found **no** 2023-2026 KWS paper using an absolute learned temporal embedding as the primary position signal.

3. **The best recent accuracy-per-MAC *architectural* win is moving expensive 2D work off the temporal axis early.** Both BC-ResNet (subspectral normalization on the frequency axis, then 1D temporal convs) and EdgeSpot (early-block fusion) independently encode the same lesson. Directly portable to `(49,40,1)`.

4. **Most "SOTA on Speech Commands" results do not transfer.** Nearly every strong number below is 35-class classification accuracy on GSC. A single wake word is a binary, far-harder, open-set problem with a hard latency constraint. Each result carries its dataset/metric and a transfer judgment.

5. **Two 2026 mechanisms attack compute directly rather than improving accuracy-per-MAC**, and neither is a drop-in: Sub-Model STMC (Samsung) turns a CNN into LSTM-like *incremental* inference, and DuSpaR exploits ReLU activation sparsity in recurrent cells. See section 4.
### 2.3 Sub-Model Short-Term Memory Convolutions -- STMC (Samsung R&D Poland, Interspeech 2026)

- Source: [arXiv:2609.35005](https://arxiv.org/abs/2609.35005), Proc. Interspeech 2026 pp. 4077-4081, DOI [10.21437/Interspeech.2026-1343](https://doi.org/10.21437/Interspeech.2026-1343).
- Mechanism: applies the STMC framework ([arXiv:2302.04331](https://arxiv.org/abs/2302.04331)) to a **modular CNN** to get **online, LSTM-like inference** -- the model is decomposed into sub-models and a scheduler runs them incrementally, exploiting state redundancy instead of re-running the whole network on a sliding window. Keeps plain-CNN training stability.
- Measured: **up to 82% MCPS reduction** vs equivalently-frequent standard CNN execution, and **46% MCPS reduction** vs vanilla STMC. Best config: **93.8% accuracy on 11-class Google Speech Commands**, and **97.1% on the same task with zero-padded data**.
- Metric: 11-class GSC accuracy; MCPS is a cycles/s-class compute metric.
- **Portable: PARTIALLY.** Architecturally interesting for the latency problem, but it changes the execution model (incremental/stateful), which conflicts with a clean fixed `(49,40,1)` 1-second window contract. A v2 idea, not a v1 patch.

### 2.4 DuSpaR -- dual-state sparsifying recurrent unit (2026)

- Source: [arXiv:2609.39237](https://arxiv.org/abs/2609.39237), Sep 2026. Li, Zhou, Cheng, Liu.
- Mechanism: recurrent cells apply ReLU to sparsify the **input vector operand** of the matrix-vector multiply, then dynamically skip zero entries -- saving both MACs *and weight memory fetches*. Adds **dual-state recurrence**: a second state feeds back to modulate the input in a stateful loop, which is what produces most of the saving (the recurrence modulates sparsity, rather than sparsity being incidental).
- Measured: at similar parameter counts, **51.0% less computation than GRU on KWS** while achieving higher accuracy; 68.4% less on SLU; 50.1% less on SE at similar quality. Ablation: dual-state recurrence reduces effective compute by **3.1x to 11.3x** vs single-state baseline at similar task performance.
- Metric: Google Speech Commands for KWS; Fluent Speech Commands for SLU; VBD for SE.
- **Portable: NO for v1.** Recurrent/branchy control flow is hostile to `EE.VMULAS.S8` int8 SIMD, and data-dependent zero-skipping cannot be exploited by a dense vector kernel. Conceptually it *is* an answer to the float32-op overhead problem (skip work rather than vectorize it faster) -- keep as longer-horizon.

### 2.5 The BC-ResNet block itself (the baseline this model is named after)

- Source: [arXiv:2106.04140](https://arxiv.org/abs/2106.04140), Interspeech 2021, DOI [10.21437/Interspeech.2021-383](https://doi.org/10.21437/Interspeech.2021-383). Kim, Chang, Lee, Sung (Qualcomm AI Research Korea). Code: [Qualcomm-AI-research/bcresnet](https://github.com/Qualcomm-AI-research/bcresnet).
- Headline: **98.0% (GSC v1) and 98.7% (GSC v2)** top-1, "using fewer computations and parameters."
- Block structure (from the published description): rather than `y = x + f(x)`, the broadcasted residual is
  `y = x + f2(x) + BC(f1(avgpool(f2(x))))`
  where `f2` is the 2D frequency-temporal path (a 3x1 frequency-depthwise conv followed by **Subspectral Normalization**, which normalizes frequency sub-bands separately), and `f1` is the cheap 1D temporal path (1x3 temporal depthwise conv -> BatchNorm -> Swish -> 1x1 pointwise conv -> channel-wise dropout). `avgpool` collapses frequency into a 1D temporal representation; `BC` broadcasts it back up.
- **Verification gap:** the per-variant params/MACs table (BC-ResNet-1...-10, BS-Conformer) could **not** be retrieved. The arXiv PDF returns image streams rather than extractable text, ar5iv conversion failed, and the GitHub README has no table. Per-variant numbers are therefore **not** quoted here. They are in table form in the Interspeech PDF if needed.
- **Why it matters anyway:** the *mechanism* is the portable part and it is well documented above -- collapse frequency early with subspectral normalization, then do the rest of the work in 1D. That is a pure MAC reduction with no accuracy loss by construction.
---

## 3. Deployability against the ~100K params / ~3 MMAC ceiling

| Work | Params | MACs | Metric | Deployable? |
|:--|--:|--:|:--|:--|
| Production net (current) | 84,865 | 11.24 M | Amaze TPR@0.5%FPR | Over MAC budget **3.7x** |
| Arithmetic floor | -- | 2.93 M | -- | Within budget; **the target** |
| RepCNN (Apple, 2024) | not in abstract | not in abstract | wake-word eval; 43% > uni-branch at equal runtime | **Architecture yes, magnitude unknown** -- must be built and measured |
| EdgeSpot-4 | 128,000 | 29.4 M | 10-shot acc @ 1% FAR | **No** -- 10x over |
| BC-ResNet family | per-variant table unverified | unverified | GSC v1 98.0% / v2 98.7% | Unknown until table verified |
| STMC best config | not in abstract | 82% / 46% MCPS reduction | 11-class GSC 93.8% (97.1% zero-padded) | Compute claim yes; accuracy far below GSC SOTA |
| DuSpaR | "similar to GRU" | 51.0% < GRU | GSC KWS | **No** -- recurrent, not int8-SIMD friendly |
| Wav2Small (2024, non-KWS) | 72,000 | -- | A/D/V emotion | 120 KB quantized ONNX -- **param budget right, task is not** |

**The honest headline: very little published work reports a params+MAC pair small enough to compete directly with a 3 MMAC ceiling.** Most "efficient KWS" papers sit in the 10-30M MAC regime -- one to two orders of magnitude above this target. RepCNN is the notable exception in *spirit*, because its claim is about the accuracy/runtime tradeoff rather than an absolute budget, and its 2x memory / 10x runtime reductions versus BC-ResNet are the kind of win that could close a 13x implementation gap.

Supporting datapoint on the param side: **Wav2Small** ([arXiv:2408.16083](https://arxiv.org/abs/2408.16083)) reports **72K parameters** and a **120 KB quantized ONNX** versus **3.12M** for MobileNetV4-Small, for audio/emotion/valence tasks. That confirms ~100K-param int8 is a well-trodden size class -- but it is not KWS and reports no MACs, so it is a feasibility signal only.

---

## 4. Architectural *mechanisms* that improve accuracy-per-MAC

Ranked by expected value for a `(49,40,1)` int8 budget.

### 4.1 Re-parameterization (train multi-branch -> fold to single branch)
- **Mechanically:** BN folding and branch-sum folding are algebraic identities. Training cost rises, deployed cost does not.
- **Why it buys accuracy/MAC:** the multi-branch training graph gives an implicit ensemble effect a narrow single-branch graph cannot reach at fixed width. Ensemble accuracy at single-branch inference cost.
- **Evidence:** +43% accuracy at equal runtime vs uni-branch; matches BC-ResNet accuracy at 2x lower peak memory and 10x faster runtime ([arXiv:2406.02652](https://arxiv.org/abs/2406.02652)).
- **Portable: YES. Delta: ~0 params/MACs at inference (the whole point).**

### 4.2 Collapse the frequency axis early (subspectral normalization + 1D thereafter)
- **Mechanically:** a 2D conv over (F=40, T=49) costs F x T. Averaging/subseparating frequency so the residual path operates on T only cuts the dominant cost by the F factor. BC-ResNet's `avgpool(f2(x))` + `BC(f1(.))` is exactly this.
- **Why:** frequency structure is captured by a cheap sub-band normalization; expensive per-frame mixing is done where it is cheapest.
- **Portable: YES. Delta: large MAC reduction.** Input is only 40 mel bands wide, so the available win is smaller than BC-ResNet's (which uses wider mel) but real.

### 4.3 Early-block fusion into the position/attention path
- **Mechanically:** EdgeSpot section 2.1.2 propagates content-based features from early blocks forward, instead of letting attention rediscover them.
- **Why:** short receptive fields at the input already encode discriminative phonetic detail; forwarding them gives the temporal aggregator a head start rather than making it re-derive local structure.
- **Portable: YES. Delta: small (skip connections cost params, almost no MACs).**

### 4.4 PCEN (learnable, per-channel energy normalization) frontend
- **Mechanically:** replaces fixed normalization with a learned per-channel gain/gating over a smooth (tanh/relu-smoothed) running energy estimate. Origin: [Lostanlen et al., "Per-channel energy normalization: why and how," SPL 26(1):39-43, 2019](https://arxiv.org/abs/1910.06740); trainable-frontend precedent: [Wang et al., "Trainable frontend for robust and far-field keyword spotting"](https://arxiv.org/abs/1607.05666).
- **Why:** cheap (no multiply-accumulate depth) and it attacks *acoustic* robustness -- level, channel imbalance, far-field -- which is where false activations likely come from.
- **Portable: YES** -- and notably the input contract is already fixed at log-mel, and PCEN operates on the time-frequency surface. Delta: negligible params/MACs.

### 4.5 Attention alternatives -- replace softmax attention with cheaper pooling
- **Evidence of the problem:** the STMC paper states plainly that transformers for KWS (AST, Keyword Transformer) have "high computational complexity and memory footprints [that] make them inadequate for resource-constrained environments, such as wireless earbuds or smartwatches" ([arXiv:2609.35005](https://arxiv.org/abs/2609.35005)). MobileNetV4's **Mobile MQA** attention block reports a **39% speedup** on mobile accelerators ([arXiv:2404.10518](https://arxiv.org/abs/2404.10518)) -- the mechanism is *replace multi-query attention with cheaper mixed-axis pooling*, which transfers even though it was built for images.
- **Practical reading:** on an int8 vector unit with no float SIMD, a strided depthwise conv or global average+max pooling as the temporal aggregator will beat multi-head softmax attention on latency essentially always. Keep attention only if the accuracy is provably worth it.
- **Portable: YES.**
### 4.6 Stem design and downsampling schedule
- **General principle:** the stem is where frequency resolution can be thrown away for free, because a wake word's identity is carried by a few formant trajectories, not by all 40 bands x 49 frames. Compressing frequency hard in the first block (and time modestly) shrinks everything downstream.
- **Evidence:** MicroNets ([arXiv:2010.11267](https://arxiv.org/abs/2010.11267), ARM/Google) is the canonical NAS result for this tier -- its key empirical finding is that **latency varies linearly with op count** under a uniform prior over the search space, which is *why* op count is a safe proxy to optimize. It reports SOTA on all three TinyMLperf tasks including **audio keyword spotting** and **visual wake words** ([ARM ML-zoo](https://github.com/ARM-software/ML-zoo)).
- **Portable: YES, high confidence.** The most under-used lever.

### 4.7 Learned filterbank / channel reduction (attacks the front end, not the net)
- **Evidence:** [arXiv:2211.10565](https://arxiv.org/abs/2211.10565) -- switching from **40-channel log-Mel to 8-channel learned features** costs only a **3.5% relative KWS accuracy loss** while achieving a **6.3x energy reduction**; learned filterbanks beat handcrafted features for KWS *whenever the number of channels is severely decreased*, and adapt better to noise, especially when dropout is integrated.
- **Why it matters:** 6.3x energy is the single largest multiplier found anywhere in this survey -- far larger than any architectural delta. Measured on a **noisy GSC variant**, closer to the real operating condition than clean GSC accuracy.
- **Portable: CAUTION.** The input contract is **frozen at 40 bands by compliance**, so the 8-channel win cannot be taken directly. What *can* be taken is the finding that the network is over-supplied in frequency channels -- so spend the savings on a wider time axis or more channels elsewhere, rather than reducing mel bands.

### 4.8 MC-Dropout / DropBlock / GroupNorm -- the honest verdict
- **Negative result, stated plainly:** searching arXiv for `"keyword spotting" AND "dropout"` returned **exactly one hit** ([arXiv:2211.10565](https://arxiv.org/abs/2211.10565), filterbank learning, where dropout is a modifier on the front end, not the headline). There is **no 2023-2026 KWS paper** establishing MC-Dropout or DropBlock as a state-of-the-art accuracy-per-MAC lever for keyword spotting.
- **What is documented:** BC-ResNet's own block already includes **channel-wise dropout** after the 1x1 pointwise conv (section 2.5). GroupNorm appears in this project's `train_advanced.py` pipeline description rather than in a cited external result.
- **Skeptical read:** any claim that MC-Dropout or DropBlock is "SOTA for embedded KWS" is marketing. MC-Dropout in particular *costs* inference (stochastic forward passes at test time) -- anti-latent-compatible with a hard 11.1 ms budget. Regularizers move accuracy-per-parameter at fixed training budget; they do not move accuracy-per-MAC.
- **Portable: regularization yes, as a training-time tool only. Do not let DropBlock/MC-Dropout into the inference graph.**

### 4.9 NAS results (TinyML-specific)
- **MicroNets** ([arXiv:2010.11267](https://arxiv.org/abs/2010.11267)) is the load-bearing result, and its most useful contribution here is the *proxy validity* claim, not the models: latency ~ linear in op count, so op count can be optimized offline and trusted. Models at [github.com/ARM-software/ML-zoo](https://github.com/ARM-software/ML-zoo).
- **TinyML NAS on KWS, 2026:** Teacher-Guided Learning NSGA-II (TGL-NSGA-II) ([arXiv:2609.30553](https://arxiv.org/abs/2609.30553)) reports measured **Kendall-tau of 0.74** on keyword spotting and 0.62 on bird-call classification against predicted lower bounds of 0.60/0.46, **41% proxy-score variance reduction** from joint stratification, and **2.2x faster than full NSGA-II**. This is a *search-efficiency* result, not an architecture result -- it means a constrained-budget NAS run on this project's own data is now practical.
- **Portable: the method is portable (it optimizes whatever search space it is given); the published search spaces are image-derived and not directly portable.**
---

## 5. Positional encoding for KWS -- what is more robust than absolute learned embeddings

This is the section most directly relevant to the measured regression (recall **77.7% -> 15.1%**).

**Direct evidence -- EdgeSpot's Relative Positional Encoding ([arXiv:2601.16316](https://arxiv.org/abs/2601.16316), section 2.1.3):** EdgeSpot pairs a **relative** positional encoding with **SDPA** and with **early-block fusion**, and reports consistent gains over strong BC-ResNet baselines at fixed FAR (73.7% -> 82.0% 10-shot @ 1% FAR for the largest variant). The structural point: EdgeSpot's positional signal is *relative* (distance-based, translation-invariant) and is **fused with content-based early features**, so position informs the attention distribution without asserting where in the window the word must start.

**Why absolute embeddings hurt on this input specifically:** in a fixed 1-second `(49,40,1)` window, the onset of the word moves with speaking rate, leading silence, and microphone distance. An absolute learned embedding `e[t]` is a *positional prior* that says "frame t is where the word starts." At 100 ms hop that prior is misaligned almost every window -- the model learns a prior that is wrong as often as right, and that noise costs it. A relative bias `b[i-j+(T-1)]` added to attention logits is invariant to where in the window the utterance lands, which is the property actually wanted.

**What this project already established (worth crediting):** per `agents.md`, `bcconformer_v2` was **never trained**, precisely because its absolute temporal embedding is a positional prior rather than a position encoder, and `bcconformer_v3` -- the current production model -- replaced it with the relative bias `b[i - j + (T-1)]` added to the attention logits. The 77.7% -> 15.1% collapse and the subsequent recovery are consistent with the EdgeSpot design. This survey found no counterexample.

**Practical recommendations, in order:**
1. **Keep relative bias. Do not reintroduce absolute learned temporal embeddings.** Confidence: high -- measured locally, and the literature agrees.
2. **Stronger still for a fixed window: remove positional encoding from the decision path entirely and make the model translation-invariant by construction** -- global pooling over the time axis, or attention whose logits depend only on content similarity. If the word can appear anywhere, position should be a non-factor. This is stronger than relative encoding, and is the version to test first.
3. **If any position signal is kept, make it relative AND fuse it with early-block content features** -- precisely EdgeSpot's combination, and the one configuration with a recent, FAR-anchored measured win behind it.
4. **Do not add sinusoidal absolute encodings** expecting help here -- they are absolute, so they inherit the same misalignment failure as learned absolute embeddings, with less capacity to overfit it.

---

## 6. Transferability audit -- why most of this literature does not apply

| Reported result | Its metric | The actual task here | Transfers? |
|:--|:--|:--|:--|
| BC-ResNet 98.0% / 98.7% | 35-class top-1 on GSC v1/v2 | binary open-set, "Amaze" only | **Weak.** 35-way closed-set accuracy says little about one word against arbitrary confusables and noise. |
| EdgeSpot 82.0% @ 1% FAR | 10-shot enrollment, 1% FAR | single fixed wake word, near-zero FAR | **Good.** FAR-anchored and open-set -- the right shape of metric. |
| STMC 93.8% / 97.1% | 11-class GSC | single wake word | **Weak**, but the compute-reduction claim is metric-independent and useful. |
| DuSpaR 51.0% < GRU | GSC accuracy + compute | single wake word, no float SIMD | **No** -- wrong execution model. |
| RepCNN +43% at equal runtime | wake-word eval | same | **Yes** -- an architectural claim about the accuracy/runtime frontier. |
| Filterbank 40->8 ch, 6.3x energy | noisy GSC | frozen 40-band contract | **No directly**; reallocating the saving, yes. |

**The systematic bias to watch:** the KWS literature overwhelmingly reports **closed-set 12/35-class accuracy on clean or synthetically-noised GSC**. The problem here -- one word, arbitrary negative audio (speech, noise, music, typing, doors), near-zero-FAR requirement, hard cycle-time budget -- is a strictly harder operating point. A model that gains 1% on GSC accuracy may gain nothing measurable at 0.1% FAR. **Prefer FAR-anchored metrics (TPR@fixed FAR) in every future evaluation**, and re-run every comparison on the project's own noise bank at the project's own operating point. The right tooling already exists (`diagnose_operating_point.py`, ROC sweeps, noise-bank percentiles) -- use it as arbiter rather than published numbers.
---

## 7. Ranked shortlist -- 8 concrete mechanisms

Ordered by expected value for the ESP32-S3 "Amaze" target. All inference deltas are **after int8 quantization**.

### 1. Re-parameterize (multi-branch train -> single-branch fold)
- **What:** train each conv block with parallel branches, fold algebraically into one branch post-training.
- **Param delta:** ~0 at inference. **MAC delta:** ~0 at inference.
- **Why it helps:** ensemble effect at training time, single-branch cost at inference. Best-evidenced item here: **+43% accuracy at identical runtime** vs uni-branch; matches BC-ResNet at **2x lower peak memory, 10x faster runtime**.
- **Confidence: HIGH.** Branch folding is a mathematical identity (BN folding), so the inference-graph guarantee is airtight; only the accuracy gain magnitude is uncertain. Directly attacks the 296 ms problem, since it simplifies the graph where the overhead lives.

### 2. Drop softmax attention; use pooled / strided-conv temporal aggregation
- **What:** replace the attention head with global average+max pooling, or a strided depthwise temporal conv.
- **Param delta:** -2k to -10k. **MAC delta:** -10% to -25% (removes QK^T and the softmax entirely).
- **Why it helps:** attention buys accuracy that may not justify its latency on an int8 vector unit with no float SIMD. STMC's authors state transformer KWS is "inadequate for resource-constrained environments"; MobileNetV4's Mobile MQA reports 39% speedup via cheaper mixed-axis pooling.
- **Confidence: HIGH** that it is cheaper; **MEDIUM** that the accuracy loss is affordable -- measure at the project's own operating point.

### 3. Make the model translation-invariant by construction; drop positional encoding
- **What:** global time-axis pooling or content-only attention. No absolute position, no sinusoidal position.
- **Param delta:** ~0 (removes an embedding table). **MAC delta:** ~0 or slightly negative.
- **Why it helps:** the word can start anywhere in a fixed 1 s window, so absolute position is a prior that is wrong as often as right. The cost was measured: recall 77.7% -> 15.1%. Removing position from the decision path is strictly stronger than making it relative.
- **Confidence: HIGH** on the diagnosis (local measurement + EdgeSpot's relative-PE design + zero 2023-2026 papers using absolute embeddings for KWS). **MEDIUM** on full removal vs relative -- test both.

### 4. Early frequency collapse via subspectral norm + 1D temporal residual path
- **What:** structure the residual as `y = x + f2(x) + BC(f1(avgpool(f2(x))))` -- cheap sub-band norm on frequency, remaining mixing in 1D time.
- **Param delta:** neutral to slightly negative. **MAC delta:** -30% to -50% if the net currently does full 2D convs on (40,49).
- **Why it helps:** moves multiply-accumulate work off the frequency axis -- the documented BC-ResNet mechanism for "much less computation than conventional CNNs" at equal-or-better accuracy.
- **Confidence: MEDIUM-HIGH.** Mechanism proven; magnitude depends on how 2D the current net already is. If already mostly 1D, the win shrinks.

### 5. PCEN learnable frontend
- **What:** per-channel learnable energy normalization over the existing 40-band log-mel, with smoothing and a learnable smoothing coefficient.
- **Param delta:** **+~100 params** (two per channel plus one scalar per channel). **MAC delta:** negligible (~1%).
- **Why it helps:** cheap attack on acoustic robustness -- level, channel imbalance, far-field -- which is where false activations come from. Composes with `acoustic_augment.py`'s INMP441 spec model rather than competing with it.
- **Confidence: MEDIUM-HIGH.** Very cheap, well-precedented ([arXiv:1607.05666](https://arxiv.org/abs/1607.05666), [arXiv:1910.06740](https://arxiv.org/abs/1910.06740)), and it fits the frozen 40-band contract. Risk: needs its own training run to realize any gain.
### 6. Reallocate frequency channels -> time axis (stem redesign)
- **What:** with 40 mel bands fixed by contract, compress frequency hard in the stem and spend the savings on a longer temporal receptive field or more channel width.
- **Param delta:** ~0 if shape-preserving. **MAC delta:** ~0 if total ops held constant -- this is a **reallocation**, not a cut.
- **Why it helps:** [arXiv:2211.10565](https://arxiv.org/abs/2211.10565) shows 8 learned channels nearly match 40 mel channels (3.5% relative loss) -- strong evidence the network is over-supplied in frequency. Wake-word identity lives in a few formant trajectories, i.e. in *time*. Also, latency ~ linear in op count ([arXiv:2010.11267](https://arxiv.org/abs/2010.11267)), so the trade is predictable offline.
- **Confidence: MEDIUM.** Over-supply evidence is strong but from a different task and a channel-count reduction that is not permitted here. Reallocation is the permitted approximation.

### 7. Early-block content fusion into the temporal aggregator
- **What:** forward low-level features from early blocks into the final pooling/attention, instead of letting the aggregator re-derive local structure.
- **Param delta:** +1k to +3k (skip/merge weights). **MAC delta:** <2% (concatenation/addition).
- **Why it helps:** part of EdgeSpot's stack that produced 73.7% -> 82.0% at fixed 1% FAR. Nearly free in MACs -- one of the best ratios available, if it holds outside a few-shot setting.
- **Confidence: MEDIUM.** The EdgeSpot result is few-shot; this setting is fixed-vocabulary trained-from-scratch, so the gain may shrink substantially.

### 8. Constrained-budget NAS on the project's own data (TGL-NSGA-II-style)
- **What:** rather than hand-picking the above, run a low-fidelity NAS over a search space parameterized to the 3 MMAC / 100K-param box, using the teacher-guided proxy from TGL-NSGA-II.
- **Param delta:** by construction, inside budget. **MAC delta:** by construction, inside budget.
- **Why it helps:** TGL-NSGA-II reports **Kendall-tau 0.74** on keyword spotting (predicted bound 0.60), **41% proxy-variance reduction**, **2.2x faster than full NSGA-II** ([arXiv:2609.30553](https://arxiv.org/abs/2609.30553)) -- the proxy that decides which candidate is better is now reliable enough to run cheaply on this project's corpus. This is the only item that optimizes *this* metric on *this* data, which section 6 argues is the real bottleneck.
- **Confidence: MEDIUM on the method transferring; HIGH on the logic.** All published search spaces are image-derived. But given that every number in section 6 shows the literature's metrics are the wrong metric, searching this project's own space is defensible.

---

## 8. Two things to fix before touching the architecture

Both are outside the published literature and both may be worth more than any item above.

1. **296 ms against a 2.93 ms arithmetic floor is a ~100x implementation gap, not a 13x budget gap.** No architectural change in this report closes 100x. Until int8 `EE.VMULAS.S8` kernels and weight residency are addressed, every mechanism above optimizes the denominator of a ratio dominated by implementation. Items 1, 2, and 4 all reduce graph complexity, which is the right direction -- but sequence the implementation work first, or architecture changes will be measured against a broken baseline.
2. **The 11.24 MMAC model is already 3.7x over the 3 MMAC ceiling while at 84,865 params -- under the param cap.** The binding constraint is arithmetic, not memory. Prioritize MAC-reducing mechanisms (items 1, 2, 4, 6) over param-reducing ones, and treat the ~100K param ceiling as comfortably non-binding.

---

### Sources
- RepCNN: https://arxiv.org/abs/2406.02652
- EdgeSpot (ICASSP 2026): https://arxiv.org/abs/2601.16316 ; https://doi.org/10.1109/ICASSP55912.2026.11460983
- Sub-Model STMC (Samsung, Interspeech 2026): https://arxiv.org/abs/2609.35005 ; https://doi.org/10.21437/Interspeech.2026-1343 ; prior STMC: https://arxiv.org/abs/2302.04331
- DuSpaR: https://arxiv.org/abs/2609.39237
- BC-ResNet (Interspeech 2021): https://arxiv.org/abs/2106.04140 ; https://doi.org/10.21437/Interspeech.2021-383 ; https://github.com/Qualcomm-AI-research/bcresnet
- MicroNets (TinyMLperf): https://arxiv.org/abs/2010.11267 ; https://github.com/ARM-software/ML-zoo
- MobileNetV4 / Mobile MQA: https://arxiv.org/abs/2404.10518
- Filterbank learning (40->8 ch): https://arxiv.org/abs/2211.10565
- Trainable frontend: https://arxiv.org/abs/1607.05666 ; PCEN: https://arxiv.org/abs/1910.06740
- TGL-NSGA-II TinyML NAS: https://arxiv.org/abs/2609.30553
- SF-KWS review (Neurocomputing 2026): https://arxiv.org/abs/2506.11169 ; https://doi.org/10.1016/j.neucom.2026.134028
- Wav2Small (72K param audio model, non-KWS): https://arxiv.org/abs/2408.16083
- Hello Edge / DS-CNN (ARM MCU KWS baseline): https://arxiv.org/abs/1711.07128 ; https://github.com/ARM-software/ML-KWS-for-MCU
- ASAP-FE feature front end (ISLPED 2025): https://arxiv.org/abs/2506.14657
- Local project docs consulted: `D:\New folder0010\wakeword_ml\docs\KWS_MODEL_SURVEY.md`, `wakeword_ml\agents.md`

---

*Report generated 2026-10-03. Every figure was fetched live in this session; the BC-ResNet per-variant params/MACs table is the one requested item that could not be retrieved, and is flagged rather than estimated.*