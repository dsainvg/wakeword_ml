"""
Downloads 1 audio file for each of the 50 ESC-50 classes from GitHub
and pre-resamples them to 16,000 Hz into `noise_bank/`.
"""

import os
import json
import urllib.request
import soundfile as sf
from scipy import signal

CLASS_NAMES = {
    0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog', 5: 'cat', 6: 'hen', 7: 'insects', 8: 'sheep', 9: 'crow',
    10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets', 14: 'chirping_birds', 15: 'water_drops',
    16: 'wind', 17: 'pouring_water', 18: 'toilet_flush', 19: 'thunderstorm', 20: 'crying_baby', 21: 'sneezing',
    22: 'clapping', 23: 'breathing', 24: 'coughing', 25: 'footsteps', 26: 'laughing', 27: 'brushing_teeth',
    28: 'snoring', 29: 'drinking_sipping', 30: 'door_knock', 31: 'mouse_click', 32: 'keyboard_typing',
    33: 'door_creak', 34: 'can_opening', 35: 'washing_machine', 36: 'vacuum_cleaner', 37: 'clock_alarm',
    38: 'clock_tick', 39: 'glass_breaking', 40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn',
    44: 'engine', 45: 'train', 46: 'church_bells', 47: 'airplane', 48: 'fireworks', 49: 'hand_saw'
}

def main():
    url = 'https://api.github.com/repos/karoldvl/ESC-50/contents/audio'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print("Fetching ESC-50 manifest from GitHub...")
    with urllib.request.urlopen(req) as r:
        files = json.loads(r.read())

    by_cat = {}
    for f in files:
        name = f['name']
        if name.endswith('.wav'):
            try:
                cat_id = int(name.rsplit('-', 1)[1].split('.')[0])
                if cat_id not in by_cat:
                    by_cat[cat_id] = f['download_url']
            except Exception:
                pass

    print(f"Found files for {len(by_cat)} categories. Downloading...")
    os.makedirs('noise_bank', exist_ok=True)
    downloaded = 0
    for cat_id, dl_url in by_cat.items():
        cname = f"esc_{cat_id:02d}_{CLASS_NAMES.get(cat_id, 'sound')}"
        dst = os.path.join('noise_bank', f"{cname}.wav")
        if not os.path.exists(dst):
            try:
                req2 = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req2, timeout=10) as resp, open('tmp_esc.wav', 'wb') as out_f:
                    out_f.write(resp.read())
                d, s = sf.read('tmp_esc.wav')
                if len(d.shape) > 1:
                    d = d[:, 0]
                if s != 16000:
                    d = signal.resample(d, int(len(d) * 16000 / s))
                sf.write(dst, d.astype('float32'), 16000)
                if os.path.exists('tmp_esc.wav'):
                    os.remove('tmp_esc.wav')
                downloaded += 1
            except Exception as e:
                print(f"  [Error] {cname}: {e}")

    total = len([f for f in os.listdir('noise_bank') if f.endswith('.wav')])
    print(f"Downloaded {downloaded} new category files. Total soundscapes in 'noise_bank': {total}")

if __name__ == "__main__":
    main()
