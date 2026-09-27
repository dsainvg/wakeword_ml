"""
Hugging Face & ESC-50 Real-World Environmental Noise Downloader
--------------------------------------------------------------
Downloads realistic non-speech noise sources:
- Room Big Fan noise (Hugging Face sdialog/background)
- Room Tone Ambient noise (Hugging Face sdialog/background)
- Keyboard typing (ESC-50)
- Rain & storm (ESC-50)
- Car engine & street traffic (ESC-50)
- Door knock & footsteps (ESC-50)
- Dog bark & animal noises (ESC-50)
- Vacuum cleaner & domestic noise (ESC-50)
- Coughing & human non-speech sounds (ESC-50)
- Siren & alarm (ESC-50)

Pre-resamples each file to 16,000 Hz 16-bit mono into `noise_bank/`.
"""

import os
import sys
import urllib.request
import numpy as np
import soundfile as sf
from scipy import signal

NOISE_SOURCES = {
    # 1. Hugging Face sdialog/background real ambient noise
    "fan_room": "https://huggingface.co/datasets/sdialog/background/resolve/main/fan_noise/210098__yuval__room-big-fan.wav",
    "room_tone": "https://huggingface.co/datasets/sdialog/background/resolve/main/fan_noise/674563__klankbeeld__room-tone-fan-801am-220813_0497.wav",
    
    # 2. ESC-50 real environmental and domestic noises
    "typing": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-103995-A-32.wav",
    "rain": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-17367-A-10.wav",
    "traffic_engine": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-36398-A-44.wav",
    "door_knock": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-101336-A-30.wav",
    "footsteps": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-34094-A-25.wav",
    "vacuum_cleaner": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-39923-A-36.wav",
    "coughing": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-30691-A-24.wav",
    "dog_bark": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-100032-A-0.wav",
    "siren": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-47819-A-42.wav",
    "car_horn": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-24074-A-43.wav"
}

OUTPUT_DIR = "noise_bank"
TARGET_SR = 16000


def download_and_resample_noises():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=====================================================================", flush=True)
    print(" Downloading Real-World Noise Bank from Hugging Face & ESC-50", flush=True)
    print(f" Target Directory: {OUTPUT_DIR} (16kHz 16-bit Mono)", flush=True)
    print("=====================================================================\n", flush=True)

    for name, url in NOISE_SOURCES.items():
        dst_wav = os.path.join(OUTPUT_DIR, f"{name}.wav")
        if os.path.exists(dst_wav):
            print(f" [Skip] {name}.wav already exists.", flush=True)
            continue

        temp_dl = os.path.join(OUTPUT_DIR, f"temp_{name}.wav")
        try:
            print(f" [Downloading] {name} from {url[:55]}...", flush=True)
            urllib.request.urlretrieve(url, temp_dl)

            # Read with soundfile
            data, sr = sf.read(temp_dl)
            if len(data.shape) > 1:
                data = data[:, 0]  # Mono

            # Resample to 16,000 Hz
            if sr != TARGET_SR:
                target_len = int(len(data) * TARGET_SR / sr)
                data = signal.resample(data, target_len)

            # Normalize & save 16-bit PCM WAV
            data = np.clip(data, -1.0, 1.0)
            data_int16 = (data * 32767.0).astype(np.int16)
            sf.write(dst_wav, data_int16, TARGET_SR, subtype="PCM_16")

            if os.path.exists(temp_dl):
                os.remove(temp_dl)

            print(f"   -> Saved: {dst_wav} ({len(data_int16):,} samples, {len(data_int16) / TARGET_SR:.1f}s)", flush=True)
        except Exception as e:
            print(f"   -> [Warning] Failed to download {name}: {e}", flush=True)
            if os.path.exists(temp_dl):
                os.remove(temp_dl)

    print("\n Noise bank preparation complete!", flush=True)
    files = os.listdir(OUTPUT_DIR)
    print(f" Total noise soundscapes in {OUTPUT_DIR}: {len(files)} files\n", flush=True)


if __name__ == "__main__":
    download_and_resample_noises()
