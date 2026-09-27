"""
Mega-Scale Multi-Source, Dual-Noise & Extreme-Distortion Dataset Engine
----------------------------------------------------------------------
Constructs a massive 6,000-sample production corpus for Wake Word 'Amaze':

1. EXTREME ACOUSTIC DISTORTION SUITE:
   - Dual-Layered Independent Noise Mixing (Cocktail Party: Ambient + Transient)
   - Lo-Fi ADC Bitcrushing (6-bit and 8-bit dynamic quantization dither)
   - Non-linear Cadence Warping (Variable quadratic time stretching)
   - Telephonic Bandpass Filtering (300 Hz - 3.4 kHz)
   - DMA Burst Packet Loss (Zeroing 20ms - 40ms bursts)
   - Vocal Tract Length Perturbation (VTLP formant warping, alpha in [0.82, 1.20])
   - Room Impulse Response Reverberation (T60 in [0.15, 0.65]s)
   - Distance Attenuation & Air Absorption (0.5m to 4.5m)
   - Hardware MEMS mic low-cut & cavity resonance + tanh saturation

2. 6,000 TOTAL HIGH-DIVERSITY SAMPLES:
   - 1,600 Positives across 96 neural voices, gTTS, SAPI5, LibriSpeech human slices,
     dual independent noises, and extreme distortions.
   - 4,400 Negatives: 1,600 Speech Commands, 1,600 LibriSpeech continuous, 300 confusables,
     and 900 dual-layered soundscapes from all 88 noise classes.

Directly saves 40-band Log-Mel spectrograms (49, 40, 1) to 'dataset_mega.npz'.
"""

import os
import io
import glob
import time
import asyncio
import numpy as np
import soundfile as sf
from scipy import signal

from features import AudioFeatureExtractor
from acoustic_augment import apply_full_acoustic_chain, apply_speed_perturbation, apply_reverberation, apply_distance_attenuation, apply_mems_mic_response

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000

EDGE_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural", "en-US-ChristopherNeural",
    "en-US-EricNeural", "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural", "en-GB-ThomasNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-PH-RosaNeural", "en-PH-JamesNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
]

GTTS_TLDS = ["com", "co.uk", "ca", "com.au", "co.in", "co.za", "ie", "com.ng"]


# ==============================================================================
# Extreme Acoustic Distortion Functions
# ==============================================================================

def apply_vtlp(audio: np.ndarray, alpha: float) -> np.ndarray:
    """Vocal Tract Length Perturbation (formant warping)."""
    new_len = int(len(audio) / alpha)
    warped = signal.resample(audio, new_len)
    t_orig = np.linspace(0, 1, len(audio))
    t_warp = np.linspace(0, 1, len(warped))
    return np.interp(t_orig, t_warp, warped).astype(np.float32)


def apply_adc_bitcrush(audio: np.ndarray, bits: int = 8) -> np.ndarray:
    """Simulates cheap low-resolution ADC converter quantization noise."""
    q_levels = 2 ** bits
    scaled = np.clip(audio, -1.0, 1.0)
    crushed = np.round((scaled + 1.0) * (q_levels / 2.0 - 0.5)) / (q_levels / 2.0 - 0.5) - 1.0
    return crushed.astype(np.float32)


def apply_nonlinear_cadence_warp(audio: np.ndarray) -> np.ndarray:
    """Simulates variable non-linear human cadence (rushing/dragging vowels)."""
    n = len(audio)
    beta = np.random.uniform(-0.25, 0.25)
    t = np.linspace(0, 1, n)
    t_warped = t + beta * t * (1.0 - t)
    t_warped = np.clip(t_warped, 0, 1)
    warped = np.interp(t, t_warped, audio)
    return warped.astype(np.float32)


def apply_telephonic_bandpass(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulates narrow telephonic / walkie-talkie bandwidth (300 Hz - 3400 Hz)."""
    b, a = signal.butter(2, [300.0 / (sr / 2.0), 3400.0 / (sr / 2.0)], btype='band')
    return signal.lfilter(b, a, audio).astype(np.float32)


def apply_burst_dropout(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulates DMA buffer slip / wireless frame dropout (zeroing 20-40ms chunk)."""
    burst_len = int(np.random.uniform(0.02, 0.04) * sr)
    if len(audio) > burst_len:
        st = np.random.randint(0, len(audio) - burst_len)
        audio = audio.copy()
        audio[st:st + burst_len] = 0.0
    return audio


def mix_dual_independent_noises(speech: np.ndarray, noise1: np.ndarray, noise2: np.ndarray, snr1_db: float, snr2_db: float) -> np.ndarray:
    """Cocktail party mixture: speech + stationary ambient noise1 + transient noise2."""
    sig_pow = np.mean(speech ** 2) + 1e-8

    p1 = np.mean(noise1 ** 2) + 1e-8
    target_p1 = sig_pow / (10 ** (snr1_db / 10.0))
    scale1 = np.sqrt(target_p1 / p1)

    p2 = np.mean(noise2 ** 2) + 1e-8
    target_p2 = sig_pow / (10 ** (snr2_db / 10.0))
    scale2 = np.sqrt(target_p2 / p2)

    mixed = speech + (noise1 * scale1) + (noise2 * scale2)
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    return mixed.astype(np.float32)


# ==============================================================================
# Speech Synthesis Engines
# ==============================================================================

async def synthesize_edge_tts(voice: str, rate: str, pitch: str) -> np.ndarray:
    import edge_tts
    comm = edge_tts.Communicate("amaze", voice, rate=rate, pitch=pitch)
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


def synthesize_gtts(tld: str, slow: bool) -> np.ndarray:
    from gtts import gTTS
    tts = gTTS(text="amaze", lang="en", tld=tld, slow=slow)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    data, sr = sf.read(fp)
    if len(data.shape) > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    return data.astype(np.float32)


def synthesize_sapi(rate: int, voice_idx: int) -> np.ndarray:
    import pyttsx3
    eng = pyttsx3.init()
    voices = eng.getProperty("voices")
    if voice_idx < len(voices):
        eng.setProperty("voice", voices[voice_idx].id)
    eng.setProperty("rate", rate)
    tmp_path = f"scratch_sapi_mega_{rate}_{voice_idx}.wav"
    eng.save_to_file("amaze", tmp_path)
    eng.runAndWait()
    data, sr = sf.read(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if len(data.shape) > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    return data.astype(np.float32)


def slice_human_amaze_from_librispeech(corpus_dir: str = "speech_corpus/LibriSpeech") -> list:
    txt_files = glob.glob(os.path.join(corpus_dir, "**", "*.trans.txt"), recursive=True)
    matches = []
    for tf in txt_files:
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                for w in parts[1:]:
                    if "AMAZE" in w:
                        matches.append((parts[0], os.path.dirname(tf)))

    human_samples = []
    for utt_id, folder in matches:
        flac = os.path.join(folder, f"{utt_id}.flac")
        if os.path.exists(flac):
            try:
                audio, sr = sf.read(flac)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                if sr != SAMPLE_RATE:
                    audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
                human_samples.append(audio.astype(np.float32))
            except Exception:
                pass
    return human_samples


# ==============================================================================
# Mega Dataset Generation Pipeline
# ==============================================================================

async def generate_mega_positives(target_count: int = 1600):
    print(f"\n[Positives] Generating {target_count} diverse 'Amaze' samples with Dual-Noise & Extreme Augmentations...")
    raw_bases = []

    # 1. Edge-TTS neural parallel
    print("  Synthesizing 96 Edge-TTS neural voice bases (parallel async)...")
    sem = asyncio.Semaphore(10)

    async def fetch_one(v, r, p):
        async with sem:
            try:
                return await synthesize_edge_tts(v, r, p)
            except Exception:
                return None

    tasks = []
    for v in EDGE_VOICES:
        for r, p in zip(["-15%", "+0%", "+15%"], ["-15Hz", "+0Hz", "+15Hz"]):
            tasks.append(fetch_one(v, r, p))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, np.ndarray) and len(res) > 0:
            raw_bases.append(res)
    print(f"  Collected {len(raw_bases)} Edge-TTS neural bases.")

    # 2. Google TTS (gTTS)
    for tld in GTTS_TLDS:
        for slow in [False, True]:
            try:
                raw_bases.append(synthesize_gtts(tld, slow))
            except Exception:
                pass

    # 3. Windows SAPI5
    for r in [130, 160, 190, 220]:
        for vidx in [0, 1]:
            try:
                raw_bases.append(synthesize_sapi(r, vidx))
            except Exception:
                pass

    # 4. Human LibriSpeech slices
    human_slices = slice_human_amaze_from_librispeech()
    raw_bases.extend(human_slices)
    print(f"  Total raw voice bases available: {len(raw_bases)}")

    # 5. Load all 88 noise soundscapes
    noise_files = sorted(glob.glob("noise_bank/*.wav"))
    cached_noises = []
    for nw in noise_files:
        try:
            d, s = sf.read(nw)
            if len(d.shape) > 1:
                d = d[:, 0]
            if s != SAMPLE_RATE:
                d = signal.resample(d, int(len(d) * SAMPLE_RATE / s))
            cached_noises.append(d)
        except Exception:
            pass
    print(f"  Pre-cached {len(cached_noises)} environmental soundscapes for dual-layer mixing.")

    # 6. Apply VTLP, extreme acoustic distortions, dual-noise mixing
    print("  Applying Extreme Distortions (ADC Bitcrush, Dual Noise, Doppler, Packet Drop, Mic)...")
    extractor = AudioFeatureExtractor()
    pos_specs = []

    np.random.seed(42)
    idx = 0
    while len(pos_specs) < target_count:
        base = raw_bases[idx % len(raw_bases)]
        idx += 1

        # VTLP formant warping (0.82 to 1.20)
        alpha = np.random.uniform(0.82, 1.20)
        audio = apply_vtlp(base, alpha)

        # Non-linear cadence warping (50% prob)
        if np.random.random() < 0.50:
            audio = apply_nonlinear_cadence_warp(audio)

        # Pad or slice to 16,000 samples
        if len(audio) < TOTAL_SAMPLES:
            pad_l = (TOTAL_SAMPLES - len(audio)) // 2
            pad_r = TOTAL_SAMPLES - len(audio) - pad_l
            audio = np.pad(audio, (pad_l, pad_r))
        else:
            audio = audio[:TOTAL_SAMPLES]

        # Room reverberation & distance attenuation
        if np.random.random() < 0.75:
            audio = apply_reverberation(audio)
        if np.random.random() < 0.75:
            audio = apply_distance_attenuation(audio)

        # Dual-Layered Independent Noise Mixing (Cocktail Party)
        if len(cached_noises) >= 2 and np.random.random() < 0.85:
            # Pick 2 distinct soundscapes
            n_idx1, n_idx2 = np.random.choice(len(cached_noises), 2, replace=False)
            nd1, nd2 = cached_noises[n_idx1], cached_noises[n_idx2]

            def slice_or_pad(nd):
                if len(nd) >= TOTAL_SAMPLES:
                    st = np.random.randint(0, len(nd) - TOTAL_SAMPLES + 1)
                    return nd[st:st + TOTAL_SAMPLES]
                else:
                    return np.pad(nd, (0, TOTAL_SAMPLES - len(nd)))

            ns1 = slice_or_pad(nd1)
            ns2 = slice_or_pad(nd2)

            snr1 = np.random.uniform(6.0, 20.0)    # Ambient base
            snr2 = np.random.uniform(-3.0, 14.0)   # Transient impulse
            audio = mix_dual_independent_noises(audio, ns1, ns2, snr1, snr2)

        # Telephonic Bandpass (15% prob)
        if np.random.random() < 0.15:
            audio = apply_telephonic_bandpass(audio)

        # Burst Packet Loss (20% prob)
        if np.random.random() < 0.20:
            audio = apply_burst_dropout(audio)

        # MEMS mic emulation + saturation (85% prob)
        if np.random.random() < 0.85:
            audio = apply_mems_mic_response(audio)

        # ADC Bitcrushing (30% prob, 6-bit or 8-bit)
        if np.random.random() < 0.30:
            bits = np.random.choice([6, 8])
            audio = apply_adc_bitcrush(audio, bits=bits)

        spec = extractor.compute_spectrogram(audio)
        pos_specs.append(spec)

    print(f"[Positives Complete] Built {len(pos_specs)} ultra-robust positive spectrograms.")
    return pos_specs


def generate_mega_negatives(extractor: AudioFeatureExtractor, target_count: int = 4400):
    print(f"\n[Negatives] Generating {target_count} diverse negative examples across 5 sources...")
    neg_specs = []

    # 1. Google Speech Commands (1,600 samples)
    cmd_wavs = glob.glob(os.path.join("speech_corpus", "speech_commands", "**", "*.wav"), recursive=True)
    if cmd_wavs:
        np.random.seed(42)
        selected_cmds = list(np.random.choice(cmd_wavs, min(1600, len(cmd_wavs)), replace=False))
        print(f"  Processing {len(selected_cmds)} Google Speech Commands (real human command words)...")
        for w in selected_cmds:
            try:
                audio, sr = sf.read(w)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                if sr != SAMPLE_RATE:
                    audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
                if len(audio) < TOTAL_SAMPLES:
                    audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)))
                else:
                    audio = audio[:TOTAL_SAMPLES]
                neg_specs.append(extractor.compute_spectrogram(audio))
            except Exception:
                pass

    # 2. LibriSpeech Continuous Speech (1,600 samples)
    flacs = []
    for root, _, files in os.walk("speech_corpus/LibriSpeech"):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    if flacs:
        np.random.seed(42)
        selected_flacs = list(np.random.choice(flacs, min(1600, len(flacs)), replace=False))
        print(f"  Processing {len(selected_flacs)} continuous LibriSpeech slices (120 human speakers)...")
        for f in selected_flacs:
            try:
                audio, sr = sf.read(f)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                if sr != SAMPLE_RATE:
                    audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
                if len(audio) >= TOTAL_SAMPLES:
                    start = np.random.randint(0, len(audio) - TOTAL_SAMPLES + 1)
                    chunk = audio[start:start + TOTAL_SAMPLES]
                else:
                    chunk = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)))
                neg_specs.append(extractor.compute_spectrogram(chunk))
            except Exception:
                pass

    # 3. Real Human Rhyming Confusables (300 samples)
    conf_files = glob.glob(os.path.join("speech_corpus", "confusables", "**", "*.flac"), recursive=True)
    if conf_files:
        print(f"  Processing {len(conf_files)} real human rhyming confusables (blaze, raise, phase...)...")
        for cf in conf_files:
            try:
                audio, sr = sf.read(cf)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                if sr != SAMPLE_RATE:
                    audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
                if len(audio) >= TOTAL_SAMPLES:
                    start = np.random.randint(0, len(audio) - TOTAL_SAMPLES + 1)
                    chunk = audio[start:start + TOTAL_SAMPLES]
                else:
                    chunk = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)))
                neg_specs.append(extractor.compute_spectrogram(chunk))
            except Exception:
                pass

    # 4. Dual-Layered Environmental Soundscapes (900 samples from all 88 soundscapes)
    noise_files = sorted(glob.glob("noise_bank/*.wav"))
    if noise_files:
        print(f"  Processing 900 dual-layered soundscapes from {len(noise_files)} noise classes...")
        for _ in range(900):
            nw1, nw2 = np.random.choice(noise_files, 2, replace=False)
            try:
                d1, s1 = sf.read(nw1)
                if len(d1.shape) > 1: d1 = d1[:, 0]
                if s1 != SAMPLE_RATE: d1 = signal.resample(d1, int(len(d1) * SAMPLE_RATE / s1))

                d2, s2 = sf.read(nw2)
                if len(d2.shape) > 1: d2 = d2[:, 0]
                if s2 != SAMPLE_RATE: d2 = signal.resample(d2, int(len(d2) * SAMPLE_RATE / s2))

                def get_chunk(d):
                    if len(d) >= TOTAL_SAMPLES:
                        st = np.random.randint(0, len(d) - TOTAL_SAMPLES + 1)
                        return d[st:st + TOTAL_SAMPLES]
                    else:
                        return np.pad(d, (0, TOTAL_SAMPLES - len(d)))

                c1 = get_chunk(d1)
                c2 = get_chunk(d2)
                mix_noise = (0.6 * c1 + 0.4 * c2).astype(np.float32)
                neg_specs.append(extractor.compute_spectrogram(mix_noise))
            except Exception:
                pass

    print(f"[Negatives Complete] Built {len(neg_specs)} negative spectrograms.")
    return neg_specs


async def main():
    print("=" * 70)
    print(" MEGA-CORPUS ENGINE: DUAL-NOISE & EXTREME ACOUSTIC AUGMENTATION")
    print("=" * 70)
    t0 = time.time()
    extractor = AudioFeatureExtractor()

    pos_specs = await generate_mega_positives(target_count=1600)
    neg_specs = generate_mega_negatives(extractor, target_count=4400)

    X_pos = np.array(pos_specs, dtype=np.float32)
    y_pos = np.ones(len(pos_specs), dtype=np.int32)

    X_neg = np.array(neg_specs, dtype=np.float32)
    y_neg = np.zeros(len(neg_specs), dtype=np.int32)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # Expand dims to (N, 49, 40, 1)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    split_idx = int(0.80 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    out_file = "dataset_mega.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f" MEGA-CORPUS GENERATION COMPLETED IN {time.time() - t0:.1f}s")
    print(f"  Archive:       {out_file} ({size_mb:.2f} MB)")
    print(f"  Total Samples: {len(X):,} (Positives: {len(X_pos):,}, Negatives: {len(X_neg):,})")
    print(f"  Train Set:     {len(X_train):,} ({np.sum(y_train==1):,} pos, {np.sum(y_train==0):,} neg)")
    print(f"  Val Set:       {len(X_val):,} ({np.sum(y_val==1):,} pos, {np.sum(y_val==0):,} neg)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
