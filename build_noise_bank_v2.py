"""
Noise Bank v2 Builder
---------------------
Expands the background-noise inventory so the model sees far more acoustic variety,
with explicit coverage of the classes it currently fails on (multi-talker babble,
crowded human chatter) plus 50x more real recordings per ESC-50 class.

Sources
  1. ESC-50 (full, 2000 clips / 50 classes) from the public GitHub mirror.
  2. UrbanSound8K (8732 clips / 18 classes incl. children_playing, crowd_hailing,
     street_music, police_car) from the HuggingFace parquet mirror.
  3. Procedural multi-talker babble built from the real human speech already in
     speech_corpus (mixes 2-14 independent talkers with per-talker gain, band
     limiting, overlap and light RIR) -- this is the class the v1 model has
     essentially never seen.
  4. Procedural device / human-event noises (HVAC hum, blender, printer, microwave
     beeps, cash register, TV static, rain on glass, ...) for appliance and office
     realism that ESC-50 does not cover.

Everything is resampled to 16 kHz mono float32 and written to noise_bank_v2/ with
an index CSV so the corpus builder can do class-stratified sampling.

    python build_noise_bank_v2.py --esc50 --urbansound --procedural
"""

import argparse
import csv
import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 16000
OUT_DIR = "noise_bank_v2"
RAW_DIR = os.path.join(OUT_DIR, "raw")
INDEX_CSV = os.path.join(OUT_DIR, "index.csv")

USER_AGENT = {"User-Agent": "Mozilla/5.0"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def to_16k_mono(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(round(len(data) * SAMPLE_RATE / sr)))
    if len(data) < SAMPLE_RATE // 2:
        data = np.pad(data, (0, SAMPLE_RATE // 2 - len(data)))
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.89
    return data.astype(np.float32)


def save_clip(clip: np.ndarray, path: str, sr: int = SAMPLE_RATE) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, clip.astype(np.float32), sr)
        return True
    except Exception:
        return False


def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# -----------------------------------------------------------------------------
# 1. ESC-50 full set
# -----------------------------------------------------------------------------

ESC50_META_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
ESC50_TREE_URL = "https://api.github.com/repos/karoldvl/ESC-50/git/trees/master?recursive=1"
ESC50_AUDIO_BASE = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/"


def esc50_class_map() -> dict:
    import json
    text = _fetch(ESC50_META_URL).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    return {r["filename"]: r["category"] for r in rows}


def build_esc50(max_files: int = 2000, workers: int = 16):
    print("\n[1/4] ESC-50 full set")
    try:
        import json
        tree = json.loads(_fetch(ESC50_TREE_URL, timeout=40).decode("utf-8"))["tree"]
        wavs = [t["path"] for t in tree
                if t["path"].startswith("audio/") and t["path"].endswith(".wav")]
        wavs.sort()
    except Exception as e:
        print(f"  could not list ESC-50: {e}")
        return []
    if max_files and len(wavs) > max_files:
        step = len(wavs) / max_files
        wavs = [wavs[int(i * step)] for i in range(max_files)]

    try:
        cmap = esc50_class_map()
    except Exception as e:
        print(f"  could not read ESC-50 metadata: {e}")
        cmap = {}

    written = []

    def grab(rel):
        fname = os.path.basename(rel)
        cls = cmap.get(fname, "esc50_unknown")
        dst = os.path.join(RAW_DIR, "esc50", f"{cls}__{fname}")
        if os.path.exists(dst):
            return (dst, cls)
        data = _fetch(ESC50_AUDIO_BASE + rel, timeout=40)
        arr, sr = sf.read(io.BytesIO(data))
        if save_clip(to_16k_mono(arr, sr), dst):
            return (dst, cls)
        return None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(grab, w): w for w in wavs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    written.append(r)
            except Exception:
                pass
            done += 1
            if done % 200 == 0:
                print(f"  ESC-50 {done}/{len(wavs)}", flush=True)
    print(f"  ESC-50 clips on disk: {len(written)}")
    return written


# -----------------------------------------------------------------------------
# 2. UrbanSound8K
# -----------------------------------------------------------------------------

US8K_REPO = "danavery/urbansound8K"


def build_urbansound(shard_limit: int = 4, per_class_cap: int = 220):
    """Streams a few UrbanSound8K parquet shards and unpacks the WAV payloads.

    The HF mirror stores clips as raw WAV bytes inside a struct column, so each
    shard is large; we only pull a handful of shards and cap clips per class.
    """
    print("\n[2/4] UrbanSound8K")
    try:
        import json
        api = json.loads(_fetch(f"https://huggingface.co/api/datasets/{US8K_REPO}", timeout=30).decode("utf-8"))
        parquets = sorted(s["rfilename"] for s in api.get("siblings", [])
                          if s["rfilename"].endswith(".parquet"))
        if not parquets:
            print("  no parquet shards listed")
            return []
    except Exception as e:
        print(f"  could not list UrbanSound8K: {e}")
        return []

    try:
        import pyarrow.parquet as pq
    except Exception as e:
        print(f"  pyarrow unavailable ({e}); skipping UrbanSound8K")
        return []

    written = []
    counts = {}
    for pq_name in parquets[:max(0, shard_limit)]:
        try:
            url = f"https://huggingface.co/datasets/{US8K_REPO}/resolve/main/{pq_name}"
            raw = _fetch(url, timeout=300)
            table = pq.read_table(io.BytesIO(raw))
        except Exception as e:
            print(f"  shard {pq_name} failed: {str(e)[:70]}")
            continue

        cols = table.column_names
        if "audio" not in cols or "class" not in cols:
            print(f"  shard {pq_name}: unexpected columns {cols}")
            continue

        audio_data = table.column("audio").to_pylist()
        labels = table.column("class").to_pylist()

        for rec, lbl in zip(audio_data, labels):
            lbl = str(lbl)
            if counts.get(lbl, 0) >= per_class_cap:
                continue
            try:
                if not isinstance(rec, dict) or not rec.get("bytes"):
                    continue
                arr, sr = sf.read(io.BytesIO(rec["bytes"]))
                counts[lbl] = counts.get(lbl, 0) + 1
                dst = os.path.join(RAW_DIR, "urbansound", f"us8k_{counts[lbl]:05d}_{lbl}.wav")
                if save_clip(to_16k_mono(arr, sr), dst):
                    written.append((dst, f"us8k_{lbl}"))
            except Exception:
                continue
        print(f"  shard {pq_name}: cumulative {len(written)} clips", flush=True)

    print(f"  UrbanSound8K clips on disk: {len(written)}")
    return written


# -----------------------------------------------------------------------------
# 3. Procedural multi-talker babble from real human speech
# -----------------------------------------------------------------------------

def _load_speech_pool(speech_dir: str = "speech_corpus", max_files: int = 3000) -> list:
    """Loads short real-human speech excerpts to use as babble talkers."""
    import glob
    pool = []
    flacs = []
    for root, _, files in os.walk(os.path.join(speech_dir, "LibriSpeech")):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    np.random.shuffle(flacs)
    for p in flacs[:max_files]:
        try:
            d, sr = sf.read(p)
            d = to_16k_mono(d, sr)
            if len(d) >= SAMPLE_RATE:
                pool.append(d)
        except Exception:
            pass
    wavs = glob.glob(os.path.join(speech_dir, "speech_commands", "**", "*.wav"), recursive=True)
    np.random.shuffle(wavs)
    for p in wavs[:2000]:
        try:
            d, sr = sf.read(p)
            d = to_16k_mono(d, sr)
            if len(d) >= SAMPLE_RATE // 2:
                pool.append(d)
        except Exception:
            pass
    print(f"  babble talker pool: {len(pool)} excerpts")
    return pool


def _fit_len(a: np.ndarray, n: int, rng) -> np.ndarray:
    if len(a) >= n:
        st = rng.integers(0, len(a) - n + 1)
        return a[st:st + n]
    reps = int(np.ceil(n / max(1, len(a))))
    return np.tile(a, reps)[:n]


def make_babble(pool: list, rng, n_talkers: int, dur: float = 12.0) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    mix = np.zeros(n, dtype=np.float32)
    for _ in range(n_talkers):
        voice = pool[rng.integers(0, len(pool))]
        seg = _fit_len(voice, n, rng).astype(np.float32)
        # per-talker gain spread so some voices dominate, others sit in the background
        g = float(10 ** (rng.uniform(-14.0, 3.0) / 20.0))
        # occasional lowpass (distant / across-the-room talkers)
        if rng.random() < 0.45:
            fc = float(rng.uniform(900.0, 3800.0))
            b, a = signal.butter(1, fc / (SAMPLE_RATE / 2.0), btype="low")
            seg = signal.lfilter(b, a, seg)
        # occasional highpass (close mic, plosives)
        if rng.random() < 0.3:
            b, a = signal.butter(1, float(rng.uniform(90.0, 260.0)) / (SAMPLE_RATE / 2.0), btype="high")
            seg = signal.lfilter(b, a, seg)
        # slow level drift (turn-taking)
        drift = 1.0 + 0.35 * np.sin(np.linspace(0, rng.uniform(1.0, 5.0), n) + rng.uniform(0, 6.28))
        mix += (seg * g * drift).astype(np.float32)

    mix = mix / (np.max(np.abs(mix)) + 1e-8)

    # small room reverb so the crowd sits in a space
    if rng.random() < 0.7:
        t60 = float(rng.uniform(0.25, 0.9))
        L = int(t60 * SAMPLE_RATE)
        ir = np.exp(-6.91 * np.arange(L) / SAMPLE_RATE / t60).astype(np.float32)
        ir *= rng.standard_normal(L).astype(np.float32) * 0.4
        ir[0] = 1.0
        ir = signal.lfilter(*signal.butter(1, 3800.0 / (SAMPLE_RATE / 2.0), btype="low"), ir)
        ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-8)
        mix = signal.fftconvolve(mix, ir)[:n].astype(np.float32)

    # channel colouring
    if rng.random() < 0.5:
        b, a = signal.butter(2, float(rng.uniform(3000.0, 7500.0)) / (SAMPLE_RATE / 2.0), btype="low")
        mix = signal.lfilter(b, a, mix)
    if rng.random() < 0.4:
        b, a = signal.butter(1, float(rng.uniform(90.0, 200.0)) / (SAMPLE_RATE / 2.0), btype="high")
        mix = signal.lfilter(b, a, mix)
    if rng.random() < 0.35:  # light bitcrush (cheap mic / codec in a crowd)
        bits = int(rng.choice([6, 8]))
        q = 2 ** bits / 2.0 - 0.5
        mix = np.round((np.clip(mix, -1, 1) + 1.0) * q) / q - 1.0
    peak = np.max(np.abs(mix))
    return (mix / peak * float(rng.uniform(0.35, 0.9))).astype(np.float32) if peak > 0 else mix


def build_babble(per_config: int = 6):
    print("\n[3/4] Procedural multi-talker babble")
    pool = _load_speech_pool()
    if not pool:
        print("  no speech pool; skipping babble")
        return []
    rng = np.random.default_rng(1234)
    written = []
    configs = [(2, "two_talker"), (3, "small_group"), (4, "small_group"), (5, "crowd"),
               (6, "crowd"), (8, "crowd"), (10, "dense_crowd"), (12, "dense_crowd"), (14, "dense_crowd")]
    for n_t, name in configs:
        for i in range(per_config):
            clip = make_babble(pool, rng, n_t, dur=float(rng.uniform(8.0, 20.0)))
            dst = os.path.join(RAW_DIR, f"babble_{name}_{n_t}t_{i:03d}.wav")
            if save_clip(clip, dst):
                written.append((dst, f"babble_{name}"))
    print(f"  babble clips on disk: {len(written)}")
    return written


# -----------------------------------------------------------------------------
# 4. Procedural device / human-event noises
# -----------------------------------------------------------------------------

def _env(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _tone(freq, n, sr=SAMPLE_RATE, phase=0.0) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def _motor(n, rng, base_hz=48.0):
    """Combustion / compressor motor: harmonic stack + roughness + broadband."""
    sig = np.zeros(n, dtype=np.float32)
    h = 1
    while base_hz * h < 3000:
        amp = 1.0 / (h ** 1.15)
        sig += amp * _tone(base_hz * h * float(rng.uniform(0.985, 1.015)), n)
        h += 1
    sig /= np.max(np.abs(sig)) + 1e-8
    rough = _env(n, rng)
    b, a = signal.butter(2, float(rng.uniform(700.0, 2600.0)) / (SAMPLE_RATE / 2.0), btype="low")
    rough = signal.lfilter(b, a, rough)
    return (0.75 * sig + 0.55 * rough).astype(np.float32)


def gen_noise(name: str, rng, dur: float = 12.0) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    r = rng

    if name == "ac_hum":
        base = float(r.choice([50.0, 60.0]))
        sig = 0.6 * _tone(base, n) + 0.4 * _tone(2 * base, n) + 0.2 * _tone(3 * base, n)
        hiss = signal.lfilter(*signal.butter(2, 3000.0 / 8000.0, btype="low"), _env(n, r))
        am = 1.0 + 0.06 * _tone(float(r.uniform(0.2, 1.5)), n)
        return (sig * am + 0.12 * hiss).astype(np.float32)

    if name == "fridge_compressor":
        sig = _motor(n, r, base_hz=float(r.uniform(48.0, 122.0)))
        b, a = signal.butter(2, 1400.0 / 8000.0, btype="low")
        return signal.lfilter(b, a, sig)

    if name == "transformer_hum":
        base = float(r.choice([50.0, 60.0]))
        sig = _tone(base, n) + 0.5 * _tone(2 * base, n) + 0.3 * _tone(4 * base, n) + 0.15 * _tone(6 * base, n)
        return (sig / 1.95).astype(np.float32)

    if name == "blender":
        sig = _motor(n, r, base_hz=float(r.uniform(90.0, 190.0)))
        jet = signal.lfilter(*signal.butter(2, float(r.uniform(1800.0, 5000.0)) / 8000.0, btype="low"), _env(n, r))
        am = 0.75 + 0.25 * np.abs(_tone(float(r.uniform(3.0, 9.0)), n))
        return (0.6 * sig + 0.7 * jet * am).astype(np.float32)

    if name == "vacuum_cleaner":
        sig = _motor(n, r, base_hz=float(r.uniform(70.0, 150.0)))
        hiss = signal.lfilter(*signal.butter(2, float(r.uniform(1200.0, 3600.0)) / 8000.0, btype="low"), _env(n, r))
        am = 0.8 + 0.2 * _tone(float(r.uniform(4.0, 11.0)), n)
        return (0.5 * sig + 0.9 * hiss * am).astype(np.float32)

    if name == "hair_dryer":
        sig = _motor(n, r, base_hz=float(r.uniform(110.0, 200.0)))
        hiss = signal.lfilter(*signal.butter(2, float(r.uniform(2000.0, 5500.0)) / 8000.0, btype="low"), _env(n, r))
        return (0.45 * sig + 1.0 * hiss).astype(np.float32)

    if name == "power_tool":
        sig = _motor(n, r, base_hz=float(r.uniform(140.0, 300.0)))
        chip = np.zeros(n, dtype=np.float32)
        period = max(8, int(SAMPLE_RATE / float(r.uniform(30.0, 90.0))))
        L = 60
        for k in range(0, n - L, period):
            chip[k:k + L] += r.standard_normal(L).astype(np.float32) * 2.5
        return (0.6 * sig + 0.35 * chip).astype(np.float32)

    if name == "printer":
        out = np.zeros(n, dtype=np.float32)
        t = 0.0
        while t < dur - 1.0:
            seg_len = min(int(r.uniform(0.8, 3.0) * SAMPLE_RATE), n)
            if seg_len < 100:
                break
            step = _motor(seg_len, r, base_hz=float(r.uniform(80.0, 180.0)))
            fader = np.linspace(1.0, 0.4, seg_len).astype(np.float32)
            st = int(t * SAMPLE_RATE)
            if st >= n:
                break
            end = min(st + seg_len, n)
            out[st:end] += step[:end - st] * fader[:end - st]
            for _ in range(int(r.integers(3, 12))):
                p = st + int(r.integers(0, max(1, end - st)))
                if p < n:
                    out[p] += float(r.uniform(0.2, 0.8))
            t += seg_len / SAMPLE_RATE + float(r.uniform(0.05, 0.4))
        return out

    if name == "keyboard_extra":
        out = np.zeros(n, dtype=np.float32)
        for _ in range(int(dur * float(r.uniform(6.0, 16.0)))):
            p = int(r.integers(0, n - 400))
            L = int(r.uniform(120, 320))
            burst = r.standard_normal(L).astype(np.float32)
            burst *= np.exp(-np.linspace(0, 1, L) * 6)
            b, a = signal.butter(2, float(r.uniform(1500.0, 5000.0)) / 8000.0, btype="low")
            out[p:p + L] += signal.lfilter(b, a, burst) * float(r.uniform(0.3, 1.0))
        return out

    if name == "plastic_crumple":
        out = np.zeros(n, dtype=np.float32)
        for _ in range(int(dur * float(r.uniform(2.0, 6.0)))):
            p = int(r.integers(0, max(1, n - 2000)))
            L = int(r.uniform(800, 2600))
            burst = r.standard_normal(L).astype(np.float32)
            burst *= np.exp(-np.linspace(0, 1, L) * 2.0)
            b, a = signal.butter(2, float(r.uniform(2500.0, 7000.0)) / 8000.0, btype="low")
            out[p:p + L] += signal.lfilter(b, a, burst) * float(r.uniform(0.4, 1.0))
        return out

    if name == "microwave_beep":
        out = np.zeros(n, dtype=np.float32)
        f = float(r.uniform(880.0, 2600.0))
        t = 0.0
        while t < dur:
            for _ in range(int(r.integers(1, 5))):
                L = int(0.12 * SAMPLE_RATE)
                gap = int(r.uniform(0.08, 0.5) * SAMPLE_RATE)
                st = int(t * SAMPLE_RATE)
                if st + L < n:
                    tone = _tone(f, L) * np.hanning(L).astype(np.float32)
                    out[st:st + L] += tone * float(r.uniform(0.5, 1.0))
            t += float(r.uniform(1.5, 5.0))
        return out

    if name == "phone_ring":
        out = np.zeros(n, dtype=np.float32)
        t = float(r.uniform(0.0, 1.0))
        f0 = float(r.uniform(400.0, 480.0))
        while t < dur:
            st = int(t * SAMPLE_RATE)
            for rep in range(2):
                s2 = st + rep * int(0.02 * SAMPLE_RATE)
                L = int(0.9 * SAMPLE_RATE)
                if s2 + L >= n:
                    break
                tone = (_tone(f0 * 2, L) + _tone(f0 * 3.2, L) + _tone(f0 * 4.1, L)) / 3.0
                out[s2:s2 + L] += tone.astype(np.float32) * float(r.uniform(0.6, 1.0))
            t += float(r.uniform(2.5, 6.0))
        return out

    if name == "notification_chime":
        out = np.zeros(n, dtype=np.float32)
        roots = [523.25, 587.33, 659.25, 783.99, 880.0, 987.77, 1046.5]
        t = 0.0
        while t < dur:
            st = int(t * SAMPLE_RATE)
            for k in range(int(r.integers(2, 5))):
                f = float(r.choice(roots))
                L = int(float(r.uniform(0.25, 0.9)) * SAMPLE_RATE)
                s2 = st + k * int(0.14 * SAMPLE_RATE)
                if s2 + L < n:
                    out[s2:s2 + L] += _tone(f, L) * np.exp(-np.linspace(0, 1, L) * 3).astype(np.float32) * 0.5
            t += float(r.uniform(2.0, 6.0))
        return out

    if name == "cash_register":
        out = np.zeros(n, dtype=np.float32)
        t = float(r.uniform(0.0, 2.0))
        while t < dur:
            st = int(t * SAMPLE_RATE)
            drawer = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
            drawer[:] = r.standard_normal(len(drawer)).astype(np.float32) * np.exp(-np.linspace(0, 1, len(drawer)) * 5)
            if st + len(drawer) < n:
                out[st:st + len(drawer)] += signal.lfilter(*signal.butter(2, 1200.0 / 8000.0, btype="low"), drawer) * 0.8
            for _ in range(int(r.integers(4, 14))):
                p = st + int(r.integers(0, int(0.4 * SAMPLE_RATE)))
                L = 200
                if p + L < n:
                    out[p:p + L] += r.standard_normal(L).astype(np.float32) * 0.4
            t += float(r.uniform(3.0, 9.0))
        return out

    if name == "tv_static":
        base = _env(n, r)
        out = base
        if r.random() < 0.5:  # AM radio: narrowband + carrier whistles
            fc = float(r.uniform(400.0, 1500.0))
            b, a = signal.butter(2, 1200.0 / 8000.0, btype="low")
            out = signal.lfilter(b, a, base) * 3.0
            for f in (fc, fc * 1.5, fc * 2.0):
                out = out + 0.12 * _tone(f, n) * (1.0 + 0.5 * _tone(float(r.uniform(0.3, 2.0)), n))
        voice = _tone(float(r.uniform(80.0, 250.0)), n)
        out = out * (0.5 + 0.5 * np.abs(voice))
        return out.astype(np.float32)

    if name == "rain_on_glass":
        base = _env(n, r)
        taps = np.zeros(n, dtype=np.float32)
        for _ in range(int(dur * float(r.uniform(20.0, 70.0)))):
            p = int(r.integers(0, max(1, n - 300)))
            L = int(r.uniform(60, 240))
            b = r.standard_normal(L).astype(np.float32) * np.exp(-np.linspace(0, 1, L) * 8)
            taps[p:p + L] += b
        hiss = signal.lfilter(*signal.butter(2, 4500.0 / 8000.0, btype="low"), base)
        return (0.75 * hiss + 0.6 * signal.lfilter(*signal.butter(2, 3000.0 / 8000.0, btype="low"), taps)).astype(np.float32)

    if name == "washing_machine":
        sig = _motor(n, r, base_hz=float(r.uniform(40.0, 90.0)))
        slosh = signal.lfilter(*signal.butter(2, 500.0 / 8000.0, btype="low"), _env(n, r))
        am = 0.7 + 0.3 * np.abs(_tone(float(r.uniform(0.6, 2.4)), n))
        return (0.6 * sig + 0.7 * slosh * am).astype(np.float32)

    if name == "sewing_machine":
        sig = _motor(n, r, base_hz=float(r.uniform(150.0, 320.0)))
        clack = np.zeros(n, dtype=np.float32)
        period = int(SAMPLE_RATE / float(r.uniform(5.0, 14.0)))
        for k in range(0, n, period):
            L = 60
            clack[k:k + L] += r.standard_normal(L).astype(np.float32) * 0.8
        return (0.6 * sig + 0.5 * clack).astype(np.float32)

    if name == "elevator_ding":
        out = np.zeros(n, dtype=np.float32)
        t = float(r.uniform(0.0, 1.0))
        while t < dur:
            st = int(t * SAMPLE_RATE)
            for f in (1567.98, 2093.0, 2637.0):
                L = int(float(r.uniform(0.5, 1.2)) * SAMPLE_RATE)
                if st + L < n:
                    out[st:st + L] += _tone(f, L) * np.exp(-np.linspace(0, 1, L) * 3.5).astype(np.float32) * 0.4
            t += float(r.uniform(4.0, 12.0))
        return out

    if name == "mic_self_noise":
        hiss = signal.lfilter(*signal.butter(2, float(r.uniform(2000.0, 6000.0)) / 8000.0, btype="low"), _env(n, r))
        agc = 1.0 / (1.0 + np.abs(hiss))
        return (0.25 * hiss * agc).astype(np.float32)

    # fallback
    return _env(n, r).astype(np.float32)


PROCEDURAL_NAMES = [
    "ac_hum", "fridge_compressor", "transformer_hum", "blender", "vacuum_cleaner",
    "hair_dryer", "power_tool", "printer", "keyboard_extra", "plastic_crumple",
    "microwave_beep", "phone_ring", "notification_chime", "cash_register",
    "tv_static", "rain_on_glass", "washing_machine", "sewing_machine",
    "elevator_ding", "mic_self_noise",
]


def build_procedural(per_name: int = 4):
    print("\n[4/4] Procedural device / event noises")
    written = []
    for name in PROCEDURAL_NAMES:
        for i in range(per_name):
            rng = np.random.default_rng(abs(hash((name, i))) % (2 ** 31))
            try:
                clip = gen_noise(name, rng, dur=float(rng.uniform(8.0, 18.0)))
                peak = np.max(np.abs(clip))
                if not np.isfinite(peak) or peak <= 0:
                    continue
                clip = clip / peak * float(rng.uniform(0.3, 0.85))
                dst = os.path.join(RAW_DIR, f"proc_{name}_{i:02d}.wav")
                if save_clip(clip.astype(np.float32), dst):
                    written.append((dst, f"proc_{name}"))
            except Exception as e:
                print(f"  {name}[{i}] failed: {str(e)[:60]}")
    print(f"  procedural clips on disk: {len(written)}")
    return written


# -----------------------------------------------------------------------------
# index
# -----------------------------------------------------------------------------

def _class_of(path: str) -> str:
    base = os.path.basename(path)
    if "__" in base:
        return base.split("__", 1)[0]
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if parts and parts[0] == "esc50":
        return "_".join(parts[1:-1]) or "esc50"
    if parts and parts[0] == "us8k":
        return "us8k_" + parts[-1]
    if parts and parts[0] == "proc":
        return "proc_" + "_".join(parts[1:-1])
    if parts and parts[0] == "babble":
        return "_".join(parts[:-2])
    return stem


def write_index(entries=None):
    """Indexes every clip already on disk (idempotent, safe to re-run)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for dirpath, _, files in os.walk(RAW_DIR):
        for f in sorted(files):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(dirpath, f)
            try:
                info = sf.info(path)
                rows.append({"path": path.replace("\\", "/"), "class": _class_of(path),
                             "duration_s": round(info.duration, 3), "sr": info.samplerate})
            except Exception:
                continue
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "duration_s", "sr"])
        w.writeheader()
        w.writerows(rows)
    classes = sorted({r["class"] for r in rows})
    print(f"\nIndex written: {INDEX_CSV}  ({len(rows)} clips, {len(classes)} classes)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", action="store_true")
    ap.add_argument("--urbansound", action="store_true")
    ap.add_argument("--procedural", action="store_true")
    ap.add_argument("--babble", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--esc50_max", type=int, default=2000)
    ap.add_argument("--babble_per_config", type=int, default=6)
    ap.add_argument("--procedural_per_name", type=int, default=4)
    ap.add_argument("--urbansound_shards", type=int, default=4)
    args = ap.parse_args()

    do_all = args.all or not any([args.esc50, args.urbansound, args.procedural, args.babble])
    entries = []
    if do_all or args.esc50:
        entries += build_esc50(max_files=args.esc50_max)
    if do_all or args.babble:
        entries += build_babble(per_config=args.babble_per_config)
    if do_all or args.procedural:
        entries += build_procedural(per_name=args.procedural_per_name)
    if args.urbansound or (do_all and args.urbansound_shards > 0):
        entries += build_urbansound(shard_limit=args.urbansound_shards)

    rows = write_index(entries)
    if rows:
        total_s = sum(r["duration_s"] for r in rows)
        print(f"Total noise material: {total_s/3600:.2f} hours across {len(rows)} clips")
    else:
        print("\n[Notice] Nothing new was written. Existing files stay listed only if rebuilt.")


if __name__ == "__main__":
    main()
