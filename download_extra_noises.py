"""
Downloads and pre-processes an extensive collection of real-world environmental noises
from Hugging Face (DPDFNet Evaluation Set), ESC-50, and synthetic calibration noise.
All files are resampled to 16,000 Hz 16-bit mono WAV and stored in 'noise_bank/'.
"""

import os
import sys
import wave
import io
import urllib.request
import numpy as np

NOISE_DIR = "noise_bank"
os.makedirs(NOISE_DIR, exist_ok=True)

def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    num_samples = int(round(len(audio) * target_sr / orig_sr))
    return np.interp(
        np.linspace(0.0, 1.0, num_samples, endpoint=False),
        np.linspace(0.0, 1.0, len(audio), endpoint=False),
        audio
    ).astype(np.float32)

def save_wav_16k(audio: np.ndarray, filepath: str):
    # Normalize peak to 0.70 to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * 0.70
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    dur = len(pcm) / 16000.0
    print(f"  [SAVED] {os.path.basename(filepath)} ({dur:.1f}s, {len(pcm)} samples)")

def read_wav_any(data_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data_bytes), "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return samples, sr

def download_and_process(url: str, out_name: str, desc: str):
    out_path = os.path.join(NOISE_DIR, out_name)
    if os.path.exists(out_path):
        print(f"[EXISTS] {out_name} already present, skipping.")
        return

    print(f"Downloading {desc} ({out_name})...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        samples, sr = read_wav_any(data)
        audio_16k = resample_linear(samples, sr, 16000)
        save_wav_16k(audio_16k, out_path)
    except Exception as e:
        print(f"  [FAILED] {out_name}: {e}")

def generate_pink_noise(duration_sec=30, sr=16000):
    n_samples = duration_sec * sr
    # Voss-McCartney pink noise generation
    num_rows = 16
    array = np.random.randn(num_rows, n_samples // 16 + 1)
    array = np.repeat(array, [2**i for i in range(num_rows)][::-1][:num_rows], axis=1)[:, :n_samples]
    white = np.random.randn(n_samples)
    pink = array.sum(axis=0) + white
    pink = pink - np.mean(pink)
    return pink.astype(np.float32)

def generate_white_noise(duration_sec=30, sr=16000):
    n_samples = duration_sec * sr
    return np.random.randn(n_samples).astype(np.float32)

def main():
    print("=" * 65)
    print(" Expanding Noise Bank with Hugging Face & ESC-50 Datasets")
    print("=" * 65)

    # 1. Hugging Face DPDFNet Evaluation Set (Real acoustic environments)
    hf_dpdfnet_files = [
        ("pub_babble.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_pub_snr0_rt60none_0_mixture_noisy.wav", "Pub & crowd babble noise"),
        ("restaurant_cafeteria.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_restaurant_snr0_rt60none_0_mixture_noisy.wav", "Restaurant & cafeteria chatter"),
        ("office_ambience.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_office_snr0_rt60none_0_mixture_noisy.wav", "Office murmur & keyboard/pc noise"),
        ("airport_terminal.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_airport_snr0_rt60none_0_mixture_noisy.wav", "Airport terminal announcements & crowds"),
        ("subway_transit.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_subway_snr0_rt60none_0_mixture_noisy.wav", "Subway metro train noise"),
        ("street_traffic.wav", "https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/resolve/main/Noisy/arabic_street_snr0_rt60none_0_mixture_noisy.wav", "City street traffic & vehicles"),
    ]

    print("\n--- 1. Downloading Hugging Face DPDFNet Real Noise Tracks ---")
    for fname, url, desc in hf_dpdfnet_files:
        download_and_process(url, fname, desc)

    # 2. ESC-50 Environmental Sound Categories
    esc50_base = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
    esc50_files = [
        ("keyboard_typing.wav", esc50_base + "1-137-A-32.wav", "Keyboard typing keys"),
        ("clapping.wav", esc50_base + "1-104089-A-22.wav", "Hand clapping"),
        ("coughing.wav", esc50_base + "1-19111-A-24.wav", "Human coughing distractor"),
        ("sneezing.wav", esc50_base + "1-26143-A-21.wav", "Human sneezing distractor"),
        ("laughing.wav", esc50_base + "1-1791-A-26.wav", "Human laughter distractor"),
        ("crying_baby.wav", esc50_base + "1-187207-A-20.wav", "Crying baby distractor"),
        ("footsteps.wav", esc50_base + "1-155858-A-25.wav", "Footsteps walking"),
        ("washing_machine.wav", esc50_base + "1-21896-A-35.wav", "Washing machine cycle"),
        ("toilet_flush.wav", esc50_base + "1-20736-A-18.wav", "Toilet flush impulse"),
        ("wind_gust.wav", esc50_base + "1-137296-A-16.wav", "Wind gust noise"),
        ("siren.wav", esc50_base + "1-31482-A-42.wav", "Emergency siren"),
        ("clock_tick.wav", esc50_base + "1-21934-A-38.wav", "Clock ticking"),
        ("water_drops.wav", esc50_base + "1-12653-A-15.wav", "Water dripping"),
        ("crackling_fire.wav", esc50_base + "1-17150-A-12.wav", "Fireplace crackling"),
        ("glass_breaking.wav", esc50_base + "1-20133-A-39.wav", "Glass breaking transient"),
        ("drinking_sipping.wav", esc50_base + "1-17295-A-29.wav", "Drinking & sipping"),
    ]

    print("\n--- 2. Downloading ESC-50 Real Sound Categories ---")
    for fname, url, desc in esc50_files:
        download_and_process(url, fname, desc)

    # 3. Synthetic Calibration Noise
    print("\n--- 3. Generating Calibrated Pink and White Noise ---")
    pink_path = os.path.join(NOISE_DIR, "pink_noise.wav")
    if not os.path.exists(pink_path):
        save_wav_16k(generate_pink_noise(duration_sec=30), pink_path)
    else:
        print("[EXISTS] pink_noise.wav already present.")

    white_path = os.path.join(NOISE_DIR, "white_noise.wav")
    if not os.path.exists(white_path):
        save_wav_16k(generate_white_noise(duration_sec=30), white_path)
    else:
        print("[EXISTS] white_noise.wav already present.")

    print("\n" + "=" * 65)
    all_noises = [f for f in os.listdir(NOISE_DIR) if f.endswith(".wav")]
    print(f" TOTAL NOISE BANK TRACKS: {len(all_noises)}")
    print("=" * 65)
    for n in sorted(all_noises):
        fsize = os.path.getsize(os.path.join(NOISE_DIR, n))
        print(f"  - {n:<26} ({fsize/1024:.1f} KB)")

if __name__ == "__main__":
    main()
