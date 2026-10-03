# INT8 Quantization & MCU Acceleration for the ESP32-S3 KWS Net

**Research report — 2023–2026 survey**
Target: ESP32-S3 (240 MHz Xtensa LX7 dual-core, 8 MB octal PSRAM, `EE.VMULAS.S8` int8 SIMD, **no** hardware float SIMD)
Model: 49x40 log-mel → small conv/conformer KWS net, 84,865 params, 11.24 MMAC
Problem: **296 ms** measured / inference vs **22.2 ms** budget; arithmetic floor **2.93 ms**; ~87 % of runtime is float32
Known headroom: worst measured int8-vs-fp32 error **5.738e-4** against a **5e-3** budget → **~8.7x error headroom available**
Known RAM hog: one float32 conv output tensor = **94,080 bytes**

> **Provenance note.** Every number below is either (a) quoted with a URL from a page fetched during this survey, or (b) labelled as an engineering estimate / project measurement rather than a published result. Google and DuckDuckGo were bot-blocked during this survey; arXiv was queried through its official API (`export.arxiv.org/api/query`), so all arXiv IDs below are verified-real and were read from the returned abstracts. Where I could not verify a number I say so explicitly rather than filling the gap.

---

## Executive summary (the short version)

1. **Adopt plain TFLite-spec W8A8 int8**: symmetric per-output-channel int8 weights (zero-point 0, range [-127,127]), asymmetric per-tensor uint8 activations (zero-point in [0,255]), int32 accumulators. This is *the* format ESP-NN and CMSIS-NN are built to execute, and you already measured 5.738e-4 error against a 5e-3 budget — i.e. you do **not** need anything cleverer.
2. **The quantization scheme is not your bottleneck.** You are 101x over the arithmetic floor (296 ms vs 2.93 ms). Perfect int8 convs get you to ~3 ms of math, but the *real* 296 ms is float32 scalar code on an integer-only vector unit. Quantization is necessary but not sufficient — you must also eliminate float from norms/activations/attention and fix weight re-read traffic.
3. **ESP-NN is proven on your exact chip.** VWW int8 on ESP32-S3 @240 MHz: **2300 ms → 54 ms (42.6x)** with ESP-NN vs TFLM. Kernel speedups **1.37x–11.48x** (softmax only 1.37x; prelu/relu6 11.48x). Source: https://github.com/espressif/esp-nn — the single most important citation in this report.
4. **ESP-DSP has NO int8 kernels.** It ships float32 and **int16 only**. Do not plan on it for int8 GEMM. Source: https://github.com/espressif/esp-dsp
5. **Do not use PyTorch's int8 stack for the firmware.** torchao targets XNNPACK/NEON/fbgemm (x86 + ARM CPU *with float SIMD*); ExecuTorch's Cortex-M backend is Arm Cortex-M; its "Cadence Xtensa" backend targets Cadence HiFi DSPs, **not** ESP32-S3's Xtensa LX7. Both are usable on your *training* box; neither emits ESP32-S3 firmware. Details in §6.
6. **The biggest single lever is weight-traffic, not bit-width**: your audit measured conv2 re-reading its weight matrix **49x** per inference. Weights at int8 are only **84,865 B ≈ 83 KB**, and ESP-NN's reference ESP32-S3 config uses a **64 KB data cache** with **80 MHz octal PSRAM** — the entire int8 weight tensor nearly fits in cache. Restructuring to weight-stationary register blocking removes the 49x re-read and is orthogonal to quantization.
7. **Do not go to int4 yet.** int4 would halve weight bytes to ~42 KB but ESP-NN ships no int4 GEMM and `EE.VMULAS.S8` is int8-native, so int4 buys bandwidth, not compute, and costs an unpack. Revisit only if post-int8 profiling still shows flash/PSRAM as the limiter.

---

## 1. PTQ / QAT best practice 2023–2026

### 1.1 The canonical MCU-grade scheme (and why it wins)

The TFLite/LiteRT integer quantization spec is the lingua franca of MCU inference and is what both ESP-NN and CMSIS-NN implement:

> `real_value = (int8_value - zero_point) × scale`
> - **Per-axis (per-channel) weights**: int8 two's complement in **[-127, 127]**, **zero-point = 0** (symmetric)
> - **Per-tensor activations/inputs**: int8 in **[-128, 127]** with **zero-point in [-128, 127]** (asymmetric)

Source: https://ai.google.dev/edge/lite/models/post_training_quantization (page last updated 2026-05-28)

CMSIS-NN states it is **bit-exact with TensorFlow Lite reference kernels** and follows TFLM where TFL and TFLM disagree.
Source: https://github.com/ARM-software/CMSIS-NN

**Implication for you:** quantize in the TFLite scheme, implement the integer graph the way CMSIS-NN/ESP-NN expect, and your on-device numerics are not a re-derivation — they are *the reference*. Do **not** invent a bespoke int8 scheme; you would lose bit-exactness with the two kernels libraries you intend to use and gain nothing, because your 8.7x error headroom means the *simple* scheme already fits.

### 1.2 Per-channel vs per-tensor

- **Weights: always per-output-channel.** Cost is C_out scales (a few hundred bytes here). Benefit is large when weight distributions are non-uniform — standard practice.
- **Activations: per-tensor asymmetric (uint8 with zero-point) is the MCU default.** Per-channel activations are the expensive/problematic case — see §1.3 and §2.
- Your project already measured a worst-case int8-vs-fp32 error of **5.738e-4** against a **5e-3** budget (8.7x headroom). That was presumably per-tensor; per-channel weights should improve on it. **Do not spend complexity here until you have per-layer error attribution.**
### 1.3 Symmetric vs asymmetric

- Weights symmetric (zp=0) — required for the fast integer paths, and it lets you skip the zero-point subtract in the accumulator loop.
- Activations asymmetric (uint8) — post-ReLU / post-swish activations are strictly non-negative, so you get the *entire* 0–255 range instead of half of it. For swish output specifically the range is [0, ~max], so unsigned is right and costs nothing.
- Caveat for this model: **pre-softmax attention probabilities are in [0,1] and heavily skewed.** Asymmetric uint8 works, but the softmax *input* (pre-activation logits) is the real problem — see §2.1.

### 1.4 SmoothQuant — useful, but the wrong tool here

SmoothQuant migrates activation outliers into weights via a mathematically equivalent per-channel scaling `s`, folded into the preceding LayerNorm, enabling W8A8 for **all** matmuls. Reports **up to 1.56x speedup and 2x memory reduction with negligible accuracy loss**, across OPT, BLOOM, GLM, MT-NLG, Llama-1/2, Falcon, Mistral, Mixtral. ICML 2023.
Source: https://arxiv.org/abs/2211.10438

**Verdict for this model: skip (for now).**
- SmoothQuant's premise is *activation outliers* in transformer FFN inputs. Your measured failure mode is float32 scalar execution cost, not outliers — you have 8.7x error headroom, i.e. no accuracy problem to solve.
- Its equivalence requires folding `1/s` into a LayerNorm, which on a conformer block means GroupNorm + relative position bias, and it perturbs the residual stream. That is architectural surgery for no proven gain.
- The cost side (a per-channel scale vector folded into the preceding norm) is cheap; the *risk* side is invalidating current checkpoints for zero measured benefit.

**Revisit only if**, after int8 conversion, per-layer error attribution shows one attention or FFN input tensor blowing the budget.

### 1.5 AdaRound / BRECQ — real, but aimed *below* int8

- **AdaRound** (ICML 2020): formulates weight rounding as a layer-wise quadratic binary optimization derived from a Taylor expansion of the task loss, solved with a soft relaxation. Result: quantize **ResNet18/ResNet50 to 4 bits within 1 % accuracy loss, no fine-tuning, using only a small amount of unlabelled data.** Source: https://arxiv.org/abs/2004.10568
- **BRECQ**: blockwise reconstruction; first PTQ framework to reach **INT2**; shows PTQ can make **4-bit ResNet/MobileNetV2 comparable to QAT**, with **240x faster** quantized-model production. Source: https://arxiv.org/abs/2102.05426

**Verdict: both are the wrong level for you.** They exist to rescue 2–4 bit. At int8 with 8.7x measured headroom, round-to-nearest per-channel is already inside budget. AdaRound also needs unlabelled calibration data through the layer plus a per-weight optimization loop — real cost, zero measurable return at 8 bit.

### 1.6 LSQ / learned step size — the one worth knowing

LSQ learns the quantizer **step size** as a parameter, with a gradient scaling that stabilizes the step-size gradient. Reports the highest accuracy to date on ImageNet with weights **and** activations at 2/3/4 bit, and can train **3-bit models that reach full-precision baseline accuracy**.
Source: https://arxiv.org/abs/1902.08153

**Verdict: the correct tool if and only if you go to QAT.** It is the best-supported learned-quantizer mechanism and ~15 lines of PyTorch. But you only need it if per-layer PTQ error attribution shows a problem. Given 8.7x headroom: **PTQ first; LSQ-based QAT only as the escalation path.**