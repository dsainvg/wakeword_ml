"""
Hard Phonetic Confusable & Sub-Word Negative Mining Engine
---------------------------------------------------------
Synthesizes targeted phonetic distractors that trick wake word models:
1. Suffix extensions: 'amazed', 'amazing', 'amazingly', 'amazement', 'amazon'
2. Prefix confusables: 'amass', 'amuse', 'amateur', 'amount', 'amends'
3. Rhymes & near-homophones: 'blaze', 'craze', 'graze', 'glaze', 'phase', 'phrase', 'raise', 'praise', 'gaze', 'daze', 'maze', 'maize'
Synthesizes across diverse regional accents via Edge-TTS and mixes with dual-layer noise.
"""

import os
import sys
import io
import asyncio
import numpy as np
import soundfile as sf
from scipy import signal
import edge_tts
from features import AudioFeatureExtractor
from acoustic_augment import (
    apply_reverberation,
    apply_distance_attenuation,
    apply_mems_mic_response,
)
from build_mega_corpus import (
    mix_dual_independent_noises,
    apply_telephonic_bandpass,
    apply_burst_dropout,
    apply_adc_bitcrush,
    apply_nonlinear_cadence_warp,
)

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000

HARD_WORDS = [
    "amazed",
    "amazing",
    "amazingly",
    "amazement",
    "amazon",
    "amateur",
    "amass",
    "amuse",
    "amount",
    "amends",
    "blaze",
    "craze",
    "graze",
    "glaze",
    "phase",
    "phrase",
    "raise",
    "praise",
    "gaze",
    "daze",
    "maze",
    "maize"
]

VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-CA-ClaraNeural",
    "en-IE-EmilyNeural",
    "en-ZA-LeahNeural"
]


async def synthesize_word(word: str, voice: str) -> np.ndarray:
    comm = edge_tts.Communicate(word, voice)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])
    out.seek(0)
    data, sr = sf.read(out)
    if len(data.shape) > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    return data.astype(np.float32)


async def generate_hard_confusables(output_npz: str = "dataset_hard_confusables.npz"):
    print("=" * 70)
    print(" MINING HARD PHONETIC CONFUSABLES & SUB-WORD DISTRACTORS")
    print("=" * 70)
    sem = asyncio.Semaphore(12)

    async def fetch_one(w, v):
        async with sem:
            try:
                return await synthesize_word(w, v)
            except Exception:
                return None

    tasks = []
    task_labels = []
    for w in HARD_WORDS:
        for v in VOICES:
            tasks.append(fetch_one(w, v))
            task_labels.append(w)

    print(f" Synthesizing {len(tasks)} targeted hard negative recordings across {len(VOICES)} regional accents...", flush=True)
    raw_audios = await asyncio.gather(*tasks)

    # Load noise bank for dual-noise mixing
    import glob
    noise_wavs = glob.glob(os.path.join("noise_bank", "*.wav"))
    cached_noises = []
    for nw in noise_wavs:
        try:
            nd, nsr = sf.read(nw)
            if len(nd.shape) > 1:
                nd = nd[:, 0]
            if nsr != SAMPLE_RATE:
                nd = signal.resample(nd, int(len(nd) * SAMPLE_RATE / nsr))
            cached_noises.append(nd.astype(np.float32))
        except Exception:
            pass

    extractor = AudioFeatureExtractor()
    specs = []
    labels = []

    print(f" Applying dual-layer cocktail party noise and extreme acoustic distortions...", flush=True)
    for idx, audio in enumerate(raw_audios):
        if audio is None or len(audio) < 1000:
            continue

        # Generate 3 augmented variations per base recording
        for var in range(3):
            aug = audio.copy()

            # Align into 1.0s window with random temporal offset
            if len(aug) < TOTAL_SAMPLES:
                max_offset = TOTAL_SAMPLES - len(aug)
                offset = np.random.randint(0, max_offset + 1)
                padded = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
                padded[offset:offset + len(aug)] = aug
                aug = padded
            else:
                aug = aug[:TOTAL_SAMPLES]

            # Nonlinear cadence warp
            if np.random.random() < 0.60:
                aug = apply_nonlinear_cadence_warp(aug)

            # Reverb
            if np.random.random() < 0.70:
                aug = apply_reverberation(aug)

            # Dual-layer noise
            if len(cached_noises) >= 2 and np.random.random() < 0.85:
                n1, n2 = np.random.choice(len(cached_noises), 2, replace=False)
                nd1, nd2 = cached_noises[n1], cached_noises[n2]

                def get_slice(nd):
                    if len(nd) >= TOTAL_SAMPLES:
                        s = np.random.randint(0, len(nd) - TOTAL_SAMPLES + 1)
                        return nd[s:s + TOTAL_SAMPLES]
                    return np.pad(nd, (0, TOTAL_SAMPLES - len(nd)))

                ns1 = get_slice(nd1)
                ns2 = get_slice(nd2)
                snr1 = np.random.uniform(6.0, 20.0)
                snr2 = np.random.uniform(-3.0, 14.0)
                aug = mix_dual_independent_noises(aug, ns1, ns2, snr1, snr2)

            # Telephonic bandpass
            if np.random.random() < 0.20:
                aug = apply_telephonic_bandpass(aug)

            # ADC Bitcrush
            if np.random.random() < 0.35:
                bits = np.random.choice([6, 8])
                aug = apply_adc_bitcrush(aug, bits=bits)

            # MEMS mic
            if np.random.random() < 0.85:
                aug = apply_mems_mic_response(aug)

            spec = extractor.compute_spectrogram(aug)
            specs.append(spec)
            labels.append(0)  # STRICT NEGATIVE

    X_hard = np.array(specs, dtype=np.float32)
    y_hard = np.array(labels, dtype=np.int32)
    print(f" Produced {len(X_hard)} hard negative spectrograms.")

    np.savez_compressed(output_npz, X=X_hard, y=y_hard)
    print(f" Saved to '{output_npz}'.")
    return output_npz


if __name__ == "__main__":
    asyncio.run(generate_hard_confusables())
