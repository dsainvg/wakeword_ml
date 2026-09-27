# Ultra-Lightweight Keyword Spotting (KWS): "Amaze"

> ## Current status (2026-09-27)
>
> The production model is **`best_v11_production.flax`** — `bcconformer_v3`, 84,865
> params (82.9 KB INT8), frozen at threshold **0.68**, peak-hold 2-of-2, 1.5 s refractory.
> Those settings live in the checkpoint and are read automatically by
> `StreamingKWSEngine`.
>
> Measured on real audio — 3.78 h of soundscapes across 74 classes plus 0.772 h of
> continuous LibriSpeech:
>
> | | previous production | current |
> |---|---|---|
> | streaming recall (20/12/6/0 dB) | 37.0% | **78.0%** |
> | false alarms per hour | 5.81 | **4.42** |
> | continuous-speech false alarms | 5.21/h | **0.00/h** |
> | clean held-out keyword detection | 65.2% | **91.8%** |
> | params | 49,882 | 84,865 |
>
> `agents.md` is the engineering record: §10.1 why the architecture changed, §10.2 why
> the confirmation policy is part of the model, §11 what is still weak.
>
> The figures in the intro below are the **v1 prototype** and describe a 5,266-parameter
> BC-ResNet-1 on a synthetic dataset. They are kept for the history of the pipeline and
> must not be quoted as the model's accuracy — `agents.md` §11 says the same.

An end-to-end Python prototype for **TinyML Keyword Spotting (KWS)** targeting the custom wake word **"Amaze"** (/əˈmeɪz/), designed to satisfy strict low-power microcontroller evaluation metrics:
- **Target Keyword:** `"Amaze"`
- **RAM Footprint:** < 256 KB (~18.0 KB utilized, **7.0% of budget**)
- **Idle CPU Load:** < 10% (**2.07% measured** at 100ms stride)
- **Model Footprint:** ~5.14 KB INT8 (**5,266 parameters** for BC-ResNet-1)
- **Accuracy:** **100.0% True Positive Rate** with **0.00% False Positive Rate**
- **Edge-to-Cloud Latency:** Sub-20ms streaming dispatch to Cloud ASR

---

## 1. Quickstart Commands

### Step 1: Synthesize Custom Keyword Dataset for "Amaze"
```bash
python dataset_generator.py --keyword "Amaze" --pos 250 --neg 500 --outdir dataset
```

### Step 2: Train State-of-the-Art BC-ResNet-1
```bash
python train.py --arch bcresnet --epochs 18 --batch 16 --lr 0.003
```
*Saves the trained model checkpoint to `best_kws_model.flax`.*

### Step 3: Test Audio Files Individually
```bash
python test_audio.py --file dataset/positive/pos_0001.wav
```

### Step 4: Run Real-Time Streaming Audio Simulation
```bash
python simulate_stream.py
```

### Step 4: Run TinyML Hardware Benchmark
Profiles memory usage, parameter counts, FLOPs/MACs, and inference latency:
```bash
python benchmark.py
```

---

## 2. TinyML Benchmark Results

Benchmarked on standard 16 kHz, 40-channel Mel filterbanks over 100 iterations:

| Metric | BC-ResNet-1 (Broadcasting-Residual) | DS-CNN-S (Depthwise Separable) | Target Evaluation Limit |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | **5,266** | **12,866** | Ultra-lightweight |
| **INT8 Flash Size** | **5.14 KB** | **12.56 KB** | < 1 MB |
| **Tensor Arena (RAM)** | **18.0 KB** | **18.0 KB** | **< 256 KB (7.0% used)** |
| **Inference Latency** | **2.07 ms** | **1.33 ms** | Real-time |
| **Idle CPU Load (100ms stride)** | **2.07%** | **1.33%** | **< 10% CPU** |
| **Validation Recall (TPR)** | **100.0%** | **97.8%** | High True Positive |
| **False Positive Rate (FPR)** | **0.00% - 2.67%** | **1.5% - 3.2%** | Near-zero false alarms |

---

## 3. Project Structure

- [`models.py`](file:///r:/Coding/embedded/esp32/wakeword/models.py): SOTA **BC-ResNet-1** and **DS-CNN-S** architectures in Flax Linen with GroupNorm.
- [`features.py`](file:///r:/Coding/embedded/esp32/wakeword/features.py): 40-band Log-Mel Spectrogram and PCEN (Per-Channel Energy Normalization) extractor.
- [`dataset_generator.py`](file:///r:/Coding/embedded/esp32/wakeword/dataset_generator.py): Formant-trajectory acoustic speech generator with noise mixing.
- [`train.py`](file:///r:/Coding/embedded/esp32/wakeword/train.py): JIT-compiled training pipeline with class-weighted loss and metric tracking.
- [`kws_engine.py`](file:///r:/Coding/embedded/esp32/wakeword/kws_engine.py): Streaming inference engine with 1.0s sliding window and moving average debouncing.
- [`simulate_stream.py`](file:///r:/Coding/embedded/esp32/wakeword/simulate_stream.py): End-to-end streaming simulation and cloud ASR ingestion demonstration.
- [`benchmark.py`](file:///r:/Coding/embedded/esp32/wakeword/benchmark.py): Microcontroller memory and CPU profiling suite.
- [`cloud_asr_server/`](file:///r:/Coding/embedded/esp32/wakeword/cloud_asr_server/server.py): FastAPI WebSocket streaming ASR server.
- [`docs/`](file:///r:/Coding/embedded/esp32/wakeword/docs/KWS_MODEL_SURVEY.md): In-depth TinyML architectural research and evaluation survey.
