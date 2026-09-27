"""
Robust Multi-Accent Neural Dataset Generator for "Amaze" with Real HF Noise Bank
--------------------------------------------------------------------------------
Incorporates 33 diverse environmental noise tracks from Hugging Face & ESC-50:
- Pub & Cocktail party babble (Hugging Face DPDFNet)
- Restaurant & Cafeteria chatter & cutlery (Hugging Face DPDFNet)
- Office murmur & PC ambience (Hugging Face DPDFNet)
- Airport terminal announcements & crowds (Hugging Face DPDFNet)
- Subway transit & urban street traffic (Hugging Face DPDFNet)
- Fan room & living room tone (Hugging Face sdialog)
- Keyboard typing, coughing, sneezing, laughing, crying baby, footsteps, clapping (ESC-50)
- Washing machine, vacuum, toilet flush, water drops, fire, glass breaking (ESC-50)
- Rain, thunderstorm, wind gust, car horn, siren, dog bark, door knock (ESC-50)
- Calibrated pink noise and flat white noise
"""

import os
import sys
import glob
import wave
import random
import asyncio
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import edge_tts

DEFAULT_KEYWORD = "Amaze"
SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 1.0
TOTAL_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)
NOISE_BANK_DIR = "noise_bank"

# Diverse global neural voices (20+ accents and genders)
VOICES = [
    # US English
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    # UK English
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    # Indian English
    "en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-IN-NeerjaExpressiveNeural",
    # Australian English
    "en-AU-WilliamMultilingualNeural", "en-AU-NatashaNeural",
    # Canadian English
    "en-CA-LiamNeural", "en-CA-ClaraNeural",
    # Irish & New Zealand English
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural"
]

# Hard phonetic confusables
CONFUSABLES = [
    "amused", "always", "amazed", "amaze me", "appraise", "blaze",
    "glaze", "phrase", "raise", "craze", "graze", "brave", "space",
    "base", "face", "chase", "erase", "ahead", "array", "astray",
    "decay", "delay", "display", "essay", "portray", "replay",
    "survey", "today", "away", "a maze"
]

# Everyday smart-home voice commands & conversational speech
NEGATIVE_PHRASES = [
    "turn on the lights", "what is the weather", "set a timer for ten minutes",
    "play some jazz", "stop music", "cancel command", "open front door",
    "volume up", "increase sound", "navigate home", "yes", "no", "hello",
    "good morning", "how are you today", "check my schedule", "send a message",
    "close the window", "turn off heater", "system reboot", "start timer"
]

# Cache pre-loaded real-world noise audio in memory
s_noise_cache = []


def load_noise_bank():
    """Loads all noise WAV files from noise_bank/ into RAM for ultra-fast slicing."""
    global s_noise_cache
    s_noise_cache.clear()
    wav_files = glob.glob(os.path.join(NOISE_BANK_DIR, "*.wav"))
    for f in wav_files:
        try:
            data, sr = sf.read(f)
            if len(data.shape) > 1:
                data = data[:, 0]
            if sr != SAMPLE_RATE:
                num_target = int(len(data) * SAMPLE_RATE / sr)
                data = signal.resample(data, num_target)
            s_noise_cache.append(data.astype(np.float32))
        except Exception as e:
            print(f"[Warning] Failed loading noise {f}: {e}")

    print(f"[NoiseBank] Loaded {len(s_noise_cache)} real environmental noise tracks from '{NOISE_BANK_DIR}'.", flush=True)


def get_real_noise_slice(num_samples: int = TOTAL_SAMPLES) -> np.ndarray:
    """Extracts a random 1.0-second slice from the real-world noise bank (Hugging Face / ESC-50)."""
    if s_noise_cache and random.random() < 0.85:
        track = random.choice(s_noise_cache)
        if len(track) >= num_samples:
            start = random.randint(0, len(track) - num_samples)
            slice_data = track[start:start + num_samples]
        else:
            slice_data = np.pad(track, (0, num_samples - len(track)), mode="wrap")
        return slice_data.copy()
    else:
        # Fallback synthetic pink/white noise
        white = np.random.normal(0, 1.0, num_samples)
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / SAMPLE_RATE)
        freqs[0] = 1.0
        pink = np.fft.irfft(fft_white / np.sqrt(freqs), n=num_samples)
        return (pink / (np.max(np.abs(pink)) + 1e-6) * 0.08).astype(np.float32)


def save_wav_16k(filepath: str, audio: np.ndarray):
    """Saves audio to 16kHz 16-bit mono WAV file padded/trimmed to 1.0s."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if len(audio) < TOTAL_SAMPLES:
        pad_l = (TOTAL_SAMPLES - len(audio)) // 2
        pad_r = TOTAL_SAMPLES - len(audio) - pad_l
        audio = np.pad(audio, (pad_l, pad_r), mode="constant")
    elif len(audio) > TOTAL_SAMPLES:
        audio = audio[:TOTAL_SAMPLES]

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def apply_real_noise_augmentation(audio: np.ndarray) -> np.ndarray:
    """Applies time shift and mixes real Hugging Face environmental noise at random SNR (-3 to 25 dB)."""
    n = len(audio)

    # 1. Random time shift (-90ms to +90ms)
    max_shift = int(0.09 * SAMPLE_RATE)
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        audio = np.pad(audio, (shift, 0), mode="constant")[:n]
    elif shift < 0:
        audio = np.pad(audio, (0, -shift), mode="constant")[-shift:n - shift]

    # 2. Extract random slice from real-world noise bank
    noise = get_real_noise_slice(n)

    # 3. Dynamic SNR mixing (-3 dB to 25 dB)
    snr_db = np.random.uniform(-3.0, 25.0)
    sig_pow = np.mean(audio ** 2) + 1e-8
    noise_pow = np.mean(noise ** 2) + 1e-8
    target_noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_pow / noise_pow)

    augmented = audio + (noise * scale)

    # Soft peak normalization
    max_val = np.max(np.abs(augmented))
    if max_val > 0.95:
        augmented = augmented / max_val * 0.95

    return augmented.astype(np.float32)


async def synthesize_neural_clip(text: str, voice: str, pitch_hz: int = 0, rate_pct: int = 0) -> np.ndarray:
    pitch_str = f"{'+' if pitch_hz >= 0 else ''}{pitch_hz}Hz"
    rate_str = f"{'+' if rate_pct >= 0 else ''}{rate_pct}%"

    communicate = edge_tts.Communicate(text, voice, pitch=pitch_str, rate=rate_str)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    raw_mp3 = b"".join(audio_chunks)
    if not raw_mp3:
        return np.zeros(TOTAL_SAMPLES, dtype=np.float32)

    import io
    mp3_io = io.BytesIO(raw_mp3)
    data, orig_sr = sf.read(mp3_io)

    if len(data.shape) > 1:
        data = data[:, 0]

    if orig_sr != SAMPLE_RATE:
        num_target = int(len(data) * SAMPLE_RATE / orig_sr)
        data = signal.resample(data, num_target)

    return data.astype(np.float32)


async def generate_dataset_async(
    output_dir: str = "dataset",
    keyword: str = DEFAULT_KEYWORD,
    num_positives: int = 250,
    num_negatives: int = 500
):
    load_noise_bank()

    pos_dir = os.path.join(output_dir, "positive")
    neg_dir = os.path.join(output_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    print("=====================================================================", flush=True)
    print(f" Robust Multi-Accent KWS Generator with Real-World HF Noise", flush=True)
    print(f" Target Keyword   : '{keyword}'", flush=True)
    print(f" Neural Voices    : {len(VOICES)} global English accents (US, UK, IN, AU, CA, IE, NZ)", flush=True)
    print(f" Real Noise Bank  : {len(s_noise_cache)} soundscapes (Babble, Cafeteria, Office, Airport, Fan, Typing, Rain...)", flush=True)
    print(f" Positive Samples : {num_positives}", flush=True)
    print(f" Negative Samples : {num_negatives}", flush=True)
    print("=====================================================================\n", flush=True)

    semaphore = asyncio.Semaphore(10)

    # 1. Synthesize Positives across accents + Real Noise Injection
    print(f"[1/2] Generating {num_positives} positive '{keyword}' samples across accents & real noise...", flush=True)

    async def worker_positive(idx: int):
        async with semaphore:
            voice = VOICES[idx % len(VOICES)]
            pitch = random.choice([-25, -15, 0, 15, 25, 35])
            rate = random.choice([-15, -10, 0, 10, 15, 20])
            try:
                raw_audio = await synthesize_neural_clip(keyword, voice, pitch_hz=pitch, rate_pct=rate)
                aug_audio = apply_real_noise_augmentation(raw_audio)
                out_path = os.path.join(pos_dir, f"pos_{idx:04d}.wav")
                save_wav_16k(out_path, aug_audio)
            except Exception as e:
                pass

    pos_tasks = [worker_positive(i) for i in range(num_positives)]
    await asyncio.gather(*pos_tasks)
    print(f"  -> Positives complete! Total saved: {len(os.listdir(pos_dir))}\n", flush=True)

    # 2. Synthesize Negatives (Real Noise Slices + Confusables + Everyday Commands)
    print(f"[2/2] Generating {num_negatives} negative samples (Real Noise + Confusables + Commands)...", flush=True)

    async def worker_negative(idx: int):
        async with semaphore:
            out_path = os.path.join(neg_dir, f"neg_{idx:04d}.wav")
            # 35% Pure Real Environmental Noise (Fan, Room tone, Vacuum, Rain, Dogs, Knocks)
            if idx % 3 == 0:
                noise_slice = get_real_noise_slice(TOTAL_SAMPLES)
                # Gain jitter
                noise_slice = noise_slice * np.random.uniform(0.5, 1.2)
                save_wav_16k(out_path, noise_slice)
                return

            # 45% Hard Phonetic Confusables, 20% Everyday Speech Commands
            if idx % 2 == 0:
                phrase = random.choice(CONFUSABLES)
            else:
                phrase = random.choice(NEGATIVE_PHRASES)

            voice = VOICES[idx % len(VOICES)]
            pitch = random.choice([-20, 0, 20])
            rate = random.choice([-10, 0, 15])
            try:
                raw_audio = await synthesize_neural_clip(phrase, voice, pitch_hz=pitch, rate_pct=rate)
                aug_audio = apply_real_noise_augmentation(raw_audio)
                save_wav_16k(out_path, aug_audio)
            except Exception as e:
                pass

    neg_tasks = [worker_negative(i) for i in range(num_negatives)]
    await asyncio.gather(*neg_tasks)
    print(f"  -> Negatives complete! Total saved: {len(os.listdir(neg_dir))}\n", flush=True)

    print("=====================================================================", flush=True)
    print(f" Dataset successfully augmented with Hugging Face & ESC-50 noise!", flush=True)
    print(f" Positives: {len(os.listdir(pos_dir))} | Negatives: {len(os.listdir(neg_dir))}", flush=True)
    print("=====================================================================", flush=True)


def build_dataset(output_dir="dataset", keyword=DEFAULT_KEYWORD, num_positives=250, num_negatives=500):
    asyncio.run(generate_dataset_async(output_dir, keyword, num_positives, num_negatives))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Accent Noise Augmented Generator for 'Amaze'")
    parser.add_argument("--keyword", type=str, default=DEFAULT_KEYWORD, help="Target keyword")
    parser.add_argument("--outdir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--pos", type=int, default=250, help="Positive count")
    parser.add_argument("--neg", type=int, default=500, help="Negative count")
    args = parser.parse_args()

    build_dataset(output_dir=args.outdir, keyword=args.keyword, num_positives=args.pos, num_negatives=args.neg)
