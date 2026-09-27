# Keyword Spotting (KWS) Architecture Survey & TinyML Analysis

This document evaluates candidate Keyword Spotting (KWS) architectures, TinyML optimization techniques, and custom keyword training strategies for the **ESP32-S3** (Xtensa dual-core LX7 with Vector Instructions, 16MB Flash, 8MB PSRAM) under strict evaluation boundaries:
- **RAM footprint:** < 256 KB
- **Idle CPU utilization:** < 10%
- **Open-source only:** No proprietary blobs (e.g., ESP-Skainet WakeNet binary libraries prohibited)
- **Zero pre-trained global assistant wake words:** Must train on a custom target phrase
- **Low-latency cloud ASR streaming:** Minimal delta between wake-up and cloud transcription

---

## 1. Architectural Comparison Matrix

| Architecture | Parameters | Quantized Size (INT8) | Tensor Arena (RAM) | Inference Latency (ESP32-S3 @ 240MHz) | Accuracy (GSC v2) | Quantization Robustness | Hardware Acceleration Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-CNN-S (Depthwise Separable CNN)** | **24.5 K** | **~26 KB** | **~22 KB** | **~11 - 14 ms** | **95.2%** | **Extremely High** | **Native ESP-NN SIMD (Xtensa PIE)** |
| **BC-ResNet-1 (Broadcast Residual)** | **9.8 K** | **~12 KB** | **~18 KB** | **~8 - 12 ms** | **96.6%** | High (needs calibrated PTQ) | Standard Conv2D/MatMul SIMD |
| **TC-ResNet-8 (Temporal Conv ResNet)** | 18.2 K | ~20 KB | ~25 KB | ~14 - 18 ms | 94.3% | High | Partial 1D Conv |
| **MatchboxNet-3x1x64** | 35.0 K | ~38 KB | ~28 KB | ~16 - 22 ms | 95.0% | Moderate | Partial 1D Depthwise |
| **Tiny-Conv (MicroSpeech Baseline)** | 19.0 K | ~20 KB | ~16 KB | ~7 - 10 ms | 88.5% | Very High | Basic 2D Conv |
| **CRNN (Conv + GRU)** | 62.0 K | ~65 KB | ~45 KB | ~28 - 36 ms | 94.8% | Moderate | Sequential / Poor SIMD fit |
| **Wav2Vec2 / Whisper-Tiny (Transformers)**| 5.8M - 38M | > 6 MB - 40 MB | > 8 MB - 32 MB | > 600 - 2500 ms | 98.2% | Poor for MCU | **DISQUALIFIED (>256KB RAM)** |

### Primary Recommendation: DS-CNN-S with Dual-Mode BC-ResNet Option
1. **DS-CNN-S (Depthwise Separable Convolutional Neural Network)**:
   - Designed specifically for microcontroller speech recognition (Zhang et al., ARM).
   - Decomposes standard 2D convolutions into a depthwise spatial convolution followed by a $1 \times 1$ pointwise convolution. This slashes multiply-accumulate (MAC) operations by **85-90%** compared to standard CNNs.
   - **ESP-NN Vector Acceleration:** Espressif provides hand-tuned assembly kernels (`esp_nn_depthwise_conv_s8` and `esp_nn_conv_s8`) utilizing the ESP32-S3's Xtensa LX7 Vector Extension (PIE instructions). This executes 4 int8 MAC operations per clock cycle, bringing single-inference latency down to **~12 ms**.
2. **BC-ResNet-1 (Broadcasting-Residual Network)**:
   - Kim et al. (NAVER Corp). Uses 1D temporal convolutions and broadcasts frequency representations across residual connections.
   - Yields the highest accuracy-to-parameter ratio (< 10k parameters), fitting easily into < 15KB of Flash.

---

## 2. Audio Preprocessing & Feature Extraction Pipeline

Microcontroller audio feature extraction must be deterministic, fixed-point capable, and compute in < 2ms per window.

```
+-----------------------------------------------------------------------------------+
|                            ESP32-S3 Audio Pipeline                               |
|                                                                                   |
|  [I2S MEMS Mic] (INMP441 / ICS-43434)                                             |
|        |                                                                          |
|        v (16 kHz, 16-bit Mono, DMA Double Buffer)                                |
|  [Energy Gater / Fixed-Point RMS VAD] ---> (Below threshold? Sleep / Skip FFT)    |
|        |                                                                          |
|        v (Energy > Threshold)                                                     |
|  [Sliding Hanning Window] (30 ms window = 480 samples, 20 ms stride = 320 samples)|
|        |                                                                          |
|        v                                                                          |
|  [512-point Real FFT] (esp-dsp / MicroFrontend fixed-point RFFT)                 |
|        |                                                                          |
|        v                                                                          |
|  [40-Channel Mel Filterbank] (Filter weights precomputed in Flash, 100Hz-7500Hz)  |
|        |                                                                          |
|        v                                                                          |
|  [Log Compression & Int8 Dynamic Quantization]                                   |
|        |                                                                          |
|        v                                                                          |
|  [Spectrogram Tensor Buffer] [49 frames x 40 mel bands = 1,960 bytes]            |
+-----------------------------------------------------------------------------------+
```

- **Tensor Dimensions:** Input shape is `(1, 49, 40, 1)` representing 1.0 second of audio (49 time frames with 20ms step, 40 Mel frequency channels).
- **Execution Overhead:** Computing 1 frame of 40-band Mel spectrum takes **~0.6 ms** on ESP32-S3 using `esp-dsp` fixed-point FFT.

---

## 3. Custom Keyword Generation Without Human Voice Actors

The challenge rules prohibit using pre-trained global assistant keywords (like "Hey Siri", "OK Google", or "Alexa"). Building an industrial-grade custom keyword model requires thousands of diverse acoustic samples. We solve this via a 4-pillar synthetic-to-real transfer pipeline:

### Pillar 1: Multi-Speaker Synthetic Speech Generation (TTS Synthesis)
- Use open-source multi-speaker neural TTS engines (`edge-tts`, `piper-tts`, or `gTTS`) covering **50+ diverse speaker identities**:
  - Genders: Male, Female, Child-pitch shifted.
  - Accents: North American, British, Indian, Australian, etc.
  - Prosody variations: Speech rates from $0.75\times$ to $1.35\times$, pitch shifts from $-30\text{Hz}$ to $+30\text{Hz}$.
- Generates 1,500 - 3,000 pristine positive utterances of the custom keyword phrase (e.g., *"Activate Echo"* or *"Computer Now"*).

### Pillar 2: Confusable Mining & Negative Phoneme Rejection
False activations occur when phonetically similar words are spoken. We specifically mine:
- **Hard Negative Confusables:** Words with overlapping vowels or consonant transitions (e.g., for *"Hey Jarvis"*, mine *"Hey Harvest"*, *"Hay Service"*, *"Starfish"*, *"Tarvis"*).
- **General Negative Speech:** 30 words from the Google Speech Commands v2 dataset (100,000+ utterances: *"yes"*, *"no"*, *"stop"*, *"go"*, *"left"*, *"right"*, etc.) + Common Voice sentences.
- **Ambient Noise & Non-Speech:** White noise, pink noise, typing, coughing, doors closing, running water, cafe murmur (from DEMAND / MUSAN / Speech Commands `_background_noise_`).

### Pillar 3: Audio Augmentation Pipeline
- Room Impulse Response (RIR) convolution: Simulates room reverberation (small room, large living room, echo).
- Additive Environmental Noise: Mixed at Signal-to-Noise Ratios (SNR) ranging dynamically from **-5 dB to +20 dB**.
- SpecAugment: Random frequency masking ($f \in [1, 6]$ channels) and time masking ($t \in [1, 8]$ frames) applied directly on the spectrogram tensors during training to prevent overfitting.

### Pillar 4: Few-Shot Calibration (Closing the Domain Gap)
- Record **5 to 15 real samples** using the actual ESP32-S3 I2S microphone (or phone mic).
- The training script weights these real samples at $4\times$ the loss penalty. This adapts the model to the exact microphone frequency response and room acoustics of the user.

---

## 4. Hardware Efficiency & Evaluation Budget on ESP32-S3

| Subsystem | RAM Allocated | Flash Footprint | CPU Load (Idle Listening) | Note |
| :--- | :--- | :--- | :--- | :--- |
| **KWS Model (INT8)** | 0 KB (runs from flash) | ~26 KB | - | Mapped to DROM via ESP32-S3 MMU |
| **TFLM Tensor Arena** | 22 KB | 0 KB | - | Statically allocated in internal SRAM |
| **I2S DMA Double Buffer** | 2 KB | 0 KB | < 0.2% | Handled autonomously by DMA hardware |
| **Pre-roll Ring Buffer (1.0s)**| 32 KB | 0 KB | - | Holds raw 16kHz PCM to prevent cutoffs |
| **Mel Spectrogram History** | 2 KB | 0 KB | - | 49 x 40 x 1 int8 array |
| **Energy / RMS VAD Pre-filter**| < 1 KB | < 2 KB | 0.4% | Evaluated every 20ms |
| **DSP (Mel Filterbank)** | 4 KB | 8 KB | 1.8% | Only runs if Energy > VAD threshold |
| **TFLite Micro Inference** | - | - | 3.5% | Runs every 100ms stride (12ms / 100ms * VAD duty) |
| **WiFi & WebSocket Client** | ~35 KB | ~120 KB | 0.8% | Pre-connected keepalive state |
| **FreeRTOS & Tasks** | ~12 KB | ~40 KB | 0.3% | Core 0: Network / Core 1: DSP + ML |
| **TOTAL** | **~110 KB** | **~196 KB** | **~7.0% CPU** | **Fully meets <256KB RAM and <10% CPU!** |

---

## 5. Edge-to-Cloud Latency & Streaming Protocol

To minimize the delta between keyword recognition and cloud ASR reception:
1. **Persistent WebSocket Handshake:** Rather than establishing a new TCP/TLS connection upon wake-up (which would add 80-250ms of DNS, TCP handshake, and TLS negotiation), the ESP32-S3 maintains a lightweight WebSocket connection with periodic 15-second heartbeat pings.
2. **Circular Pre-Roll Flushing:** When the KWS model fires (confidence threshold $P(\text{target}) > 0.85$ for 2 consecutive windows):
   - The ring buffer immediately transmits the preceding 200ms of audio (to capture any lingering trailing phonemes without clipping).
   - Subsequent audio frames are streamed synchronously in 50ms chunks (1,600 bytes) over WebSocket binary frames.
3. **Latency Profile:**
   - Network transmission delay over local 802.11n Wi-Fi: **4 - 12 ms**
   - Cloud ASR ingest latency: **< 5 ms**
   - **Total time delta:** **< 20 ms** from wake word completion to cloud ingestion.
