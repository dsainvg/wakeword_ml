"""
Downloads 12 additional distinct soundscapes from ESC-50:
- mouse click, snoring, cat meow, helicopter, chainsaw, sea waves,
  crickets, can opening, fireworks, church bells, airplane, breathing
"""

import os
import urllib.request
import soundfile as sf
from scipy import signal

ADDITIONAL_NOISES = {
    "mouse_click": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-20133-A-39.wav",
    "snoring": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-23052-A-28.wav",
    "cat_meow": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-34094-A-5.wav",
    "helicopter": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-17264-A-40.wav",
    "chainsaw": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-19872-A-41.wav",
    "sea_waves": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-28135-A-11.wav",
    "crickets": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-1791-A-26.wav",
    "can_opening": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-18755-A-37.wav",
    "fireworks": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-39923-A-48.wav",
    "church_bells": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-32986-A-46.wav",
    "airplane": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-36929-A-47.wav",
    "breathing": "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-43760-A-23.wav"
}

OUTPUT_DIR = "noise_bank"
TARGET_SR = 16000

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading {len(ADDITIONAL_NOISES)} additional soundscapes into '{OUTPUT_DIR}'...")
    for name, url in ADDITIONAL_NOISES.items():
        dst_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        if os.path.exists(dst_path):
            print(f"  [Skip] {name}.wav already exists.")
            continue
        try:
            tmp_path = f"tmp_{name}.wav"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp, open(tmp_path, "wb") as f:
                f.write(resp.read())
            data, sr = sf.read(tmp_path)
            if len(data.shape) > 1:
                data = data[:, 0]
            if sr != TARGET_SR:
                data = signal.resample(data, int(len(data) * TARGET_SR / sr))
            sf.write(dst_path, data.astype("float32"), TARGET_SR)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"  [Done] {name}.wav downloaded and resampled to {TARGET_SR}Hz.")
        except Exception as e:
            print(f"  [Error] {name}: {e}")

    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
    print(f"\nTotal soundscapes in '{OUTPUT_DIR}': {total}")

if __name__ == "__main__":
    main()
