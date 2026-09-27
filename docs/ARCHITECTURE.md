# End-to-End System Architecture: Edge KWS + Cloud ASR

This document describes the complete hybrid edge-to-cloud architecture designed for the ESP32-S3 running an ultra-lightweight Keyword Spotting (KWS) model coupled with a real-time streaming Automatic Speech Recognition (ASR) cloud backend.

---

## 1. System Topology & Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant Mic as I2S MEMS Mic (INMP441)
    participant Core1 as ESP32-S3 Core 1 (DSP & ML)
    participant Core0 as ESP32-S3 Core 0 (WiFi / Net)
    participant Cloud as Remote ASR Server (WebSocket)

    Note over Core0,Cloud: System boots; establishes persistent WebSocket connection
    loop Continuous Listening (Idle <10% CPU, <256KB RAM)
        Mic->>Core1: DMA audio buffer (16kHz, 16-bit, 20ms chunks)
        Core1->>Core1: Energy Gate (RMS check). If silent, sleep.
        alt Speech Energy Detected
            Core1->>Core1: Compute 40-band Mel-filterbank
            Core1->>Core1: Run INT8 DS-CNN inference (12ms via ESP-NN)
        end
    end

    Note over Core1: Keyword Detected! (e.g. "Activate Echo", confidence > 0.85)
    Core1->>Core0: Post WAKE_EVENT with pre-roll timestamp via FreeRTOS Queue
    Core0->>Cloud: Send JSON Header: {"event": "wake", "timestamp_ms": 12840}
    Core0->>Cloud: Stream binary audio frames (Pre-roll buffer + Live 50ms chunks)
    
    loop Real-Time Streaming
        Mic->>Core1: Live audio stream
        Core1->>Core0: Stream raw PCM
        Core0->>Cloud: Binary WebSocket frames (16kHz PCM @ 32 KB/s)
        Cloud->>Cloud: Stream into Faster-Whisper / Vosk decoder
        Cloud-->>Core0: Partial transcript JSON: {"partial": "turn on the..."}
    end

    Note over Core1: Silence detected for 1.2s -> STOP_STREAM
    Core0->>Cloud: Send JSON: {"event": "end_of_speech"}
    Cloud-->>Core0: Final transcript JSON: {"text": "Turn on the living room lights.", "latency_ms": 18.4}
```

---

## 2. ESP32-S3 Dual-Core Thread Allocation

The ESP32-S3 contains two 32-bit Xtensa LX7 cores running at 240 MHz. Allocating tasks symmetrically avoids thread contention and guarantees real-time audio deadlines.

### Core 1 (Real-Time DSP & Machine Learning Core)
- **`Audio_Capture_Task` (Priority 10, Core 1)**:
  - Consumes 320 samples (20ms) from the I2S DMA queue.
  - Computes root-mean-square (RMS) energy.
  - Pushes audio samples into the circular pre-roll buffer (32 KB capacity).
- **`KWS_Inference_Task` (Priority 8, Core 1)**:
  - If RMS energy exceeds noise floor, computes 40-band log Mel filterbank.
  - Shifts the $49 \times 40$ spectrogram matrix by 5 frames (100ms stride).
  - Invokes `tflite::MicroInterpreter::Invoke()` with ESP-NN SIMD vector acceleration.
  - Applies 2-window moving average confidence debouncing to suppress false positives.
  - Dispatches `WAKE_EVENT` to Core 0 upon positive spotting.

### Core 0 (Network & Protocol Core)
- **`WiFi_WebSocket_Task` (Priority 5, Core 0)**:
  - Maintains persistent WebSocket connection to the cloud ASR server (`ws://<server_ip>:8000/ws/audio`).
  - Handles WebSocket ping/pong heartbeats every 15 seconds.
  - When notified by `WAKE_EVENT`, transitions to `STREAMING` mode.
  - Pipes the audio ring buffer contents followed by incoming live PCM directly out of the Wi-Fi stack via zero-copy LwIP socket writes.

---

## 3. Circular Ring Buffer & Zero-Clipping Pre-Roll

A major challenge in wake-word systems is **speech clipping**:
- If audio streaming starts only *after* the wake-word is recognized, the start of the follow-on command (or the last syllable of the wake-word) is often clipped.
- **Solution:** A 1.0-second circular audio buffer (32,000 bytes for 16-bit 16kHz audio) continuously logs audio in SRAM.
- When the wake word triggers at time $T_{\text{wake}}$:
  - The pointer rewinds by $300\text{ms}$ (9,600 bytes) to capture any transition audio.
  - This pre-roll slice is transmitted as the very first binary payload over the existing WebSocket connection.
  - No follow-up words are lost, ensuring near 100% cloud ASR transcription accuracy.

---

## 4. Hardware Pinout Configuration (ESP32-S3 to INMP441)

The INMP441 is an omnidirectional MEMS microphone with standard I2S digital output.

| INMP441 Pin | ESP32-S3 GPIO | Function | Description |
| :--- | :--- | :--- | :--- |
| **VDD** | **3V3** | Power | 3.3V DC regulated power rail |
| **GND** | **GND** | Ground | Common ground |
| **SD** | **GPIO 4** | I2S Serial Data (DIN) | Serial PCM audio data from mic |
| **WS / L/R CLK** | **GPIO 5** | I2S Word Select (LRCLK) | Frame clock (16 kHz) |
| **SCK / BCLK** | **GPIO 6** | I2S Bit Clock (BCLK) | Bit clock (16 kHz * 32 bits = 512 kHz) |
| **L/R** | **GND** | Channel Select | Tie to GND for Left Channel (Mono) |

---

## 5. Latency Decomposition Breakdown

The evaluation metric requires minimizing the time delta between the keyword ending and the cloud ASR receiving the audio stream.

```
+--------------------------------------------------------------------------------+
| Time Delta Event                                                       Duration|
+--------------------------------------------------------------------------------+
| [T0] Wake word utterance concludes                                           0 ms|
| [T1] KWS feature window closes & inference executes (100ms stride)        ~10 ms|
| [T2] Moving average debouncer verifies target threshold                   ~10 ms|
| [T3] FreeRTOS inter-core queue notification from Core 1 to Core 0         <0.1 ms|
| [T4] Core 0 flushes pre-roll packet into TCP/WebSocket socket               ~1.5 ms|
| [T5] 802.11n Wi-Fi local airtime propagation to Access Point / Server     ~4 - 8 ms|
| [T6] Cloud WebSocket server reads binary chunk into memory buffer          <1.0 ms|
+--------------------------------------------------------------------------------+
| TOTAL LATENCY DELTA (T6 - T0):                                      ~26 - 30 ms|
+--------------------------------------------------------------------------------+
```

Because the WebSocket connection is **pre-established**, there is zero DNS or TCP 3-way handshake overhead when the wake event occurs. Cloud ingestion begins in under 30 milliseconds.
