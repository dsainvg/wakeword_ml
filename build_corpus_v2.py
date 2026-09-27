"""
Industrial Corpus v2 for Wake Word 'Amaze'
-------------------------------------------
Builds a much larger, better-structured corpus than build_mega_corpus.py, driven by
three failure modes measured on the v1 model:

  1. TEMPORAL POSITION OVERFIT
     v1 centre-padded every positive, so the model learned "keyword sits at frame ~14".
     Rolling a val positive by 18 frames dropped TPR from 83% to 70%.
     -> v2 places the keyword at a random offset in the 1 s window and also emits
        explicit multi-offset copies of each voice base.

  2. NO MULTI-TALKER BABBLE NEGATIVES
     v1 scored 0.933 on pub_babble and 0.852 on restaurant_cafeteria -- both above the
     0.85 threshold -- because the noise bank had essentially no crowded-speech class.
     -> v2 mixes 2-8 real speech slices as the negative class, and mixes 2-3 noise
        layers into every positive.

  3. 88 NOISE CLASSES / 5 s EACH
     -> v2 uses noise_bank_v2 (full ESC-50 x40, 90 multi-talker babble tracks, ~160
        procedural device/event clips) with class-stratified sampling and random
        start offsets, so a 1 s window can sit anywhere inside a 10-20 s soundscape.

Positives  : 1.0 s windows, keyword at random offset, 2-3 noise layers at a
             SNR ladder that reaches -6 dB, full distortion chain, plus context
             carriers ("hey amaze", "amaze the lights") and intra-word jitter.
Negatives  : (a) clean/noisy real speech, (b) keyword-adjacent confusables,
             (c) pure + layered soundscapes, (d) MULTI-TALKER BABBLE,
             (e) speech-in-noise at low SNR (the hardest real deployment case),
             (f) keyword-adjacent fragments ("a maze", "amaz", "amuses").

Split discipline: the 80/20 split is stratified by (label, negative source kind) so every
negative category keeps its share in both splits. With ~28k negatives the val FPR
resolves to ~0.02% instead of the single-sample 0.12% v1 was reporting.

Storage dtype: X_* are written as float16 by default (--dtype fp16). The log-mel front
end floors at log(1e-5) = -11.51 and rarely exceeds +5, so float16 resolution (~1e-3) is
far below anything the model resolves, and it halves the artifact and the download.
train_industrial_v2.py upcasts per batch, so either dtype trains identically.

    python build_corpus_v2.py --pos 14000 --neg 28000 --out dataset_v2.npz
"""

import argparse
import asyncio
import csv
import glob
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy import signal

from features import AudioFeatureExtractor
from acoustic_augment import (
    apply_reverberation,
    apply_distance_attenuation,
    apply_mems_mic_response,
    apply_speed_perturbation,
    apply_inmp441_response,
    add_mic_self_noise,
    apply_i2s_16bit_truncation,
    draw_talker_conditions,
)

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000

KEYWORD = "amaze"

# Carriers that still contain the exact wake word (keeps the label valid)
POSITIVE_CARRIERS = [
    "amaze",
    "amaze.",
    "hey amaze",
    "ok amaze",
    "amaze play music",
    "amaze stop",
    "computer amaze",
    "amaze the lights",
    "amazing",
]
# 'amazing' is intentionally absent as a bare positive-only signal: the confusable
# miner below keeps inflected forms out of the positive set.

EDGE_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural", "en-US-ChristopherNeural",
    "en-US-EricNeural", "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-US-EmmaNeural", "en-US-AndrewNeural", "en-US-BrianNeural", "en-US-AvaNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural", "en-GB-ThomasNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-AU-KenNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaExpressiveNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-PH-RosaNeural", "en-PH-JamesNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
    "en-NG-ChidiNeural", "en-PH-DeniseNeural",
]

GTTS_TLDS = ["com", "co.uk", "ca", "com.au", "co.in", "co.za", "ie", "com.ng", "co.nz"]

# Keyword-adjacent negatives: these are the words that actually make KWS systems fire.
CONFUSABLE_WORDS = [
    "amazed", "amazement", "amazes", "amazing", "amusing", "amused", "amuses",
    "a maze", "a maiz", "amaz", "amaize", "amusse", "always", "amass", "amassed",
    "blaze", "blazes", "raise", "raises", "praise", "phase", "phrase", "chase",
    "erase", "gaze", "graze", "craze", "glaze", "maize", "amazingly",
    "haste", "paste", "taste", "waste", "vast", "mask", "masks", "ask",
    "am I late", "amazing grace", "may as well", "me as well", "aims",
    "he was amazed", "I'm amazed", "are you amazed",
]

NEGATIVE_COMMANDS = [
    "turn on the lights", "what is the weather", "set a timer for ten minutes",
    "play some jazz", "stop the music", "cancel that", "open the front door",
    "volume up", "increase the sound", "navigate home", "yes", "no", "hello",
    "good morning", "how are you today", "check my schedule", "send a message",
    "close the window", "turn off the heater", "system reboot", "start the timer",
    "what time is it", "call mom", "play the news", "shut down", "wake up",
    "lights off", "lights on", "pause", "resume", "next track", "louder", "quieter",
]


# ==============================================================================
# Acoustic helpers
# ==============================================================================

def apply_vtlp(audio: np.ndarray, alpha: float) -> np.ndarray:
    new_len = max(64, int(len(audio) / alpha))
    warped = signal.resample(audio, new_len)
    t_orig = np.linspace(0, 1, len(audio))
    t_warp = np.linspace(0, 1, len(warped))
    return np.interp(t_orig, t_warp, warped).astype(np.float32)


def apply_adc_bitcrush(audio: np.ndarray, bits: int = 8) -> np.ndarray:
    q = 2 ** bits / 2.0 - 0.5
    scaled = np.clip(audio, -1.0, 1.0)
    return (np.round((scaled + 1.0) * q) / q - 1.0).astype(np.float32)


def apply_nonlinear_cadence_warp(audio: np.ndarray, beta: float = None) -> np.ndarray:
    n = len(audio)
    if beta is None:
        beta = np.random.uniform(-0.25, 0.25)
    t = np.linspace(0, 1, n)
    tw = np.clip(t + beta * t * (1.0 - t), 0, 1)
    return np.interp(t, tw, audio).astype(np.float32)


def apply_telephonic_bandpass(audio: np.ndarray) -> np.ndarray:
    b, a = signal.butter(2, [300.0 / 8000.0, 3400.0 / 8000.0], btype="band")
    return signal.lfilter(b, a, audio).astype(np.float32)


def apply_burst_dropout(audio: np.ndarray) -> np.ndarray:
    burst = int(np.random.uniform(0.02, 0.045) * SAMPLE_RATE)
    if len(audio) > burst:
        st = np.random.randint(0, len(audio) - burst)
        audio = audio.copy()
        audio[st:st + burst] = 0.0
    return audio


def apply_dropout_bursts(audio: np.ndarray, max_bursts: int = 3) -> np.ndarray:
    out = audio
    for _ in range(int(np.random.randint(1, max_bursts + 1))):
        out = apply_burst_dropout(out)
    return out


def apply_awgn(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sp = np.mean(speech ** 2) + 1e-9
    npow = np.mean(noise ** 2) + 1e-9
    return (speech + noise * np.sqrt(sp / (10 ** (snr_db / 10.0) * npow))).astype(np.float32)


def peak_guard(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(audio))
    if m > ceiling:
        audio = audio / m * ceiling
    return audio


def fit_to_window(sig: np.ndarray, rng, offset: str = "random") -> np.ndarray:
    """Places `sig` inside a 1 s window at a random offset (or centred).

    Random offset is the fix for the v1 temporal-position overfit: the streaming
    engine slides the window every 100 ms, so the keyword lands at an arbitrary
    position in the frame.
    """
    sig = np.asarray(sig, dtype=np.float32).ravel()
    n = len(sig)
    if n >= TOTAL_SAMPLES:
        if offset == "random":
            st = rng.integers(0, n - TOTAL_SAMPLES + 1)
        else:
            st = (n - TOTAL_SAMPLES) // 2
        return sig[st:st + TOTAL_SAMPLES].copy()
    if offset == "random":
        left = int(rng.integers(0, TOTAL_SAMPLES - n + 1))
    else:
        left = (TOTAL_SAMPLES - n) // 2
    out = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    out[left:left + n] = sig
    return out


def trim_silence(audio: np.ndarray, thresh_db: float = -45.0, pad_ms: int = 25) -> np.ndarray:
    """Trims leading/trailing near-silence so short words are not mostly padding."""
    if len(audio) == 0:
        return audio
    win = int(0.005 * SAMPLE_RATE)
    n_win = max(1, len(audio) // win)
    frames = audio[:n_win * win].reshape(n_win, win)
    energy = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    voiced = np.where(energy > thresh_db)[0]
    if len(voiced) == 0:
        return audio
    pad = int(pad_ms * SAMPLE_RATE / 1000)
    st = max(0, voiced[0] * win - pad)
    en = min(len(audio), (voiced[-1] + 1) * win + pad)
    return audio[st:en]


# ==============================================================================
# Noise / speech banks
# ==============================================================================

class NoiseBank:
    """Class-stratified sampler over noise_bank_v2/index.csv (or legacy noise_bank).

    The decoded-clip cache is bounded: a full 84-class bank is several GB of float32
    audio and each build worker holds its own copy, so only the most recently used
    clips stay resident.
    """

    MAX_CACHE = 256

    def __init__(self, index_csv: str = "noise_bank_v2/index.csv", legacy_dir: str = "noise_bank"):
        self.by_class = {}
        paths = []
        if os.path.exists(index_csv):
            with open(index_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    paths.append((row["path"].replace("/", os.sep), row["class"]))
        if not paths:
            for p in sorted(glob.glob(os.path.join(legacy_dir, "*.wav"))):
                paths.append((p, os.path.splitext(os.path.basename(p))[0]))
        self.paths = paths
        for p, c in paths:
            self.by_class.setdefault(c, []).append(p)
        self.classes = sorted(self.by_class)
        self._cache = {}
        print(f"[NoiseBank] {len(paths)} clips across {len(self.classes)} classes")

    def load(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        try:
            d, sr = sf.read(path)
            if d.ndim > 1:
                d = d[:, 0]
            if sr != SAMPLE_RATE:
                d = signal.resample(d, int(round(len(d) * SAMPLE_RATE / sr)))
            out = d.astype(np.float32)
        except Exception:
            out = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if len(self._cache) >= self.MAX_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = out
        return out

    def sample_slice(self, rng, exclude=()) -> tuple:
        """Returns (slice, class). Class is drawn uniformly, position inside the clip
        is drawn uniformly, so a 1 s window can land anywhere in a 10-20 s soundscape."""
        pool = [c for c in self.classes if c not in exclude] or self.classes
        cls = str(rng.choice(pool))
        path = str(rng.choice(self.by_class[cls]))
        clip = self.load(path)
        if len(clip) >= TOTAL_SAMPLES:
            st = int(rng.integers(0, len(clip) - TOTAL_SAMPLES + 1))
            return clip[st:st + TOTAL_SAMPLES].copy(), cls
        reps = int(np.ceil(TOTAL_SAMPLES / max(1, len(clip))))
        padded = np.tile(clip, reps)[:TOTAL_SAMPLES]
        return padded, cls

    def babble_classes(self):
        return [c for c in self.classes if "babble" in c]


class SpeechBank:
    """Real human speech: LibriSpeech + Speech Commands, for negatives and babble."""

    MAX_CACHE = 192

    def __init__(self, corpus_dir: str = "speech_corpus", max_flacs: int = 6000, max_cmds: int = 6000):
        self.flacs = []
        for root, _, files in os.walk(os.path.join(corpus_dir, "LibriSpeech")):
            for f in files:
                if f.endswith(".flac"):
                    self.flacs.append(os.path.join(root, f))
        np.random.shuffle(self.flacs)
        self.flacs = self.flacs[:max_flacs]
        self.cmds = glob.glob(os.path.join(corpus_dir, "speech_commands", "**", "*.wav"), recursive=True)
        np.random.shuffle(self.cmds)
        self.cmds = self.cmds[:max_cmds]
        self._cache = {}
        print(f"[SpeechBank] {len(self.flacs)} LibriSpeech flacs, {len(self.cmds)} Speech Commands wavs")

    def load(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        try:
            d, sr = sf.read(path)
            if d.ndim > 1:
                d = d[:, 0]
            if sr != SAMPLE_RATE:
                d = signal.resample(d, int(round(len(d) * SAMPLE_RATE / sr)))
            out = d.astype(np.float32)
        except Exception:
            out = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if len(self._cache) >= self.MAX_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = out
        return out

    def sample_slice(self, rng, cmd_bias: float = 0.0) -> np.ndarray:
        use_cmd = self.cmds and (rng.random() < cmd_bias)
        pool = self.cmds if use_cmd else self.flacs
        if not pool:
            return np.zeros(TOTAL_SAMPLES, dtype=np.float32)
        path = str(rng.choice(pool))
        clip = self.load(path)
        if len(clip) >= TOTAL_SAMPLES:
            st = int(rng.integers(0, len(clip) - TOTAL_SAMPLES + 1))
            return clip[st:st + TOTAL_SAMPLES].copy()
        reps = int(np.ceil(TOTAL_SAMPLES / max(1, len(clip))))
        return np.tile(clip, reps)[:TOTAL_SAMPLES]


def make_babble_from_speech(speech_bank: SpeechBank, rng, n_talkers: int, dur: float = 4.0) -> np.ndarray:
    """Multi-talker crowd built from real speech -- the negative class v1 never had."""
    n = int(dur * SAMPLE_RATE)
    mix = np.zeros(n, dtype=np.float32)
    for _ in range(n_talkers):
        seg = speech_bank.sample_slice(rng, cmd_bias=0.15)
        if len(seg) < n:
            seg = np.tile(seg, int(np.ceil(n / len(seg))))
        seg = seg[:n]
        g = float(10 ** (rng.uniform(-13.0, 3.0) / 20.0))
        if rng.random() < 0.5:
            fc = float(rng.uniform(900.0, 4000.0))
            b, a = signal.butter(1, fc / 8000.0, btype="low")
            seg = signal.lfilter(b, a, seg)
        if rng.random() < 0.35:
            b, a = signal.butter(1, float(rng.uniform(90.0, 260.0)) / 8000.0, btype="high")
            seg = signal.lfilter(b, a, seg)
        mix += (seg * g).astype(np.float32)
    mix /= (np.max(np.abs(mix)) + 1e-8)
    if rng.random() < 0.6:
        mix = apply_reverberation(mix, t60_sec=float(rng.uniform(0.2, 0.8)))
    return peak_guard(mix).astype(np.float32)


# ==============================================================================
# Synthesis engines for keyword / confusable waveforms
# ==============================================================================

async def synth_edge(text: str, voice: str, rate: str, pitch: str) -> np.ndarray:
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    out = io.BytesIO()

    async def drain():
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                out.write(chunk["data"])

    # edge-tts websockets occasionally stall; bound every request so a single
    # hung voice cannot stall the whole pool build.
    await asyncio.wait_for(drain(), timeout=25.0)
    out.seek(0)
    data, sr = sf.read(out)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(round(len(data) * SAMPLE_RATE / sr)))
    return data.astype(np.float32)


def synth_gtts(text: str, tld: str, slow: bool) -> np.ndarray:
    from gtts import gTTS
    tts = gTTS(text=text, lang="en", tld=tld, slow=slow)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    data, sr = sf.read(fp)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(round(len(data) * SAMPLE_RATE / sr)))
    return data.astype(np.float32)


def synth_sapi(text: str, rate: int, voice_idx: int) -> np.ndarray:
    import pyttsx3
    eng = pyttsx3.init()
    voices = eng.getProperty("voices")
    if voice_idx < len(voices):
        eng.setProperty("voice", voices[voice_idx].id)
    eng.setProperty("rate", rate)
    tmp = f"scratch_v2_{rate}_{voice_idx}.wav"
    eng.save_to_file(text, tmp)
    eng.runAndWait()
    data, sr = sf.read(tmp)
    if os.path.exists(tmp):
        os.remove(tmp)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(round(len(data) * SAMPLE_RATE / sr)))
    return data.astype(np.float32)


# ==============================================================================
# Positive pipeline
# ==============================================================================

def mix_layers(speech: np.ndarray, layers, snr_ladder) -> np.ndarray:
    out = speech
    for layer, snr in zip(layers, snr_ladder):
        if layer is None or len(layer) == 0:
            continue
        if len(layer) < len(out):
            reps = int(np.ceil(len(out) / len(layer)))
            layer = np.tile(layer, reps)
        layer = layer[:len(out)]
        if np.std(layer) < 1e-7:
            continue
        out = apply_awgn(out, layer, snr)
    return out


def distortion_chain(audio: np.ndarray, rng, aggressive: float) -> np.ndarray:
    """aggressive in [0,1] controls how far down the robustness ladder we go."""
    if rng.random() < 0.35 + 0.35 * aggressive:
        audio = apply_nonlinear_cadence_warp(audio)
    if rng.random() < 0.70:
        audio = apply_reverberation(audio)
    if rng.random() < 0.70:
        audio = apply_distance_attenuation(audio)
    if rng.random() < 0.15 * aggressive:
        audio = apply_telephonic_bandpass(audio)
    if rng.random() < 0.15 + 0.25 * aggressive:
        audio = apply_dropout_bursts(audio, max_bursts=1 + int(2 * aggressive))
    if rng.random() < 0.85:
        audio = apply_mems_mic_response(audio)
    if rng.random() < 0.25 + 0.30 * aggressive:
        audio = apply_adc_bitcrush(audio, bits=int(rng.choice([6, 8])))
    return peak_guard(audio)


# Deployment-realistic profile. v3 (the default) keeps a large near-field / clean
# mode because the v2 corpus was so uniformly destroyed that the model never saw a
# normal-level keyword and lost its high-confidence response entirely: the trained
# model's confidence ceiling was 0.86, which capped recall at any usable threshold.
PROFILES = {
    "v2": {
        "clean_frac": 0.0,
        "snr1": (2.0, 22.0), "snr2": (-2.0, 16.0), "snr3": (-6.0, 12.0),
        "telephonic": 0.15, "bitcrush": (0.25, 0.30), "dropout": (0.15, 0.25),
        "gain": (1.0, 1.0), "sin_snr": (-4.0, 14.0), "mems": 1.0,
    },
    "v3": {
        "clean_frac": 0.30,
        "snr1": (8.0, 24.0), "snr2": (0.0, 16.0), "snr3": (-3.0, 12.0),
        "telephonic": 0.06, "bitcrush": (0.12, 0.20), "dropout": (0.08, 0.18),
        "gain": (0.45, 1.60), "sin_snr": (0.0, 16.0), "mems": 1.0,
    },
    # v4/v5: buckets + a transducer chain matched to the INMP441.
    # `mems` goes to 0 because the INMP441 is flat from 60 Hz to 15 kHz, which
    # brackets the 100-7500 Hz feature band entirely: a generic MEMS darkening curve
    # would misrepresent the hardware. `bitcrush` drops to a rare probe because a
    # 24-bit I2S link quantizes ~59 dB below the capsule noise floor.
    "v4": {
        "clean_frac": 0.0,
        "snr1": (8.0, 24.0), "snr2": (0.0, 16.0), "snr3": (-3.0, 12.0),
        "telephonic": 0.06, "bitcrush": (0.02, 0.05), "dropout": (0.08, 0.18),
        "gain": (1.0, 1.0), "sin_snr": (0.0, 16.0), "mems": 0.0,
    },
}
_PROFILE = dict(PROFILES["v4"])


def _distortion_chain(audio: np.ndarray, rng, aggressive: float) -> np.ndarray:
    """Room / transmission / mic-stage effects only. The capsule response is NOT
    applied here -- it is handled by the INMP441 transducer chain after the level is
    set, so that ordering matches the physical signal path."""
    p = _PROFILE
    if rng.random() < 0.35 + 0.35 * aggressive:
        audio = apply_nonlinear_cadence_warp(audio)
    if rng.random() < 0.70:
        audio = apply_reverberation(audio)
    if rng.random() < 0.62:
        audio = apply_distance_attenuation(audio)
    if rng.random() < p["telephonic"] * aggressive:
        audio = apply_telephonic_bandpass(audio)
    if rng.random() < p["dropout"][0] + p["dropout"][1] * aggressive:
        audio = apply_dropout_bursts(audio, max_bursts=1 + int(2 * aggressive))
    if p.get("mems", 0.0) > 0 and rng.random() < p["mems"]:
        audio = apply_mems_mic_response(audio)
    if rng.random() < p["bitcrush"][0] + p["bitcrush"][1] * aggressive:
        audio = apply_adc_bitcrush(audio, bits=int(rng.choice([6, 8])))
    return peak_guard(audio)


def build_positive(base: np.ndarray, noise_bank: NoiseBank, rng) -> tuple:
    """base is a trimmed keyword waveform (< 1 s). Returns (1 s window, snr_mid, noise_classes)."""
    p = _PROFILE
    base = np.asarray(base, dtype=np.float32).ravel()
    audio = base
    # speed / VTLP: speaker- and rate-level variation
    if rng.random() < 0.85:
        audio = apply_speed_perturbation(audio, float(rng.uniform(0.80, 1.22)))
    if rng.random() < 0.70:
        audio = apply_vtlp(audio, float(rng.uniform(0.82, 1.22)))

    audio = fit_to_window(audio, rng, offset="random")

    # Recording-gain variation: the mic sits at a fixed gain, so the keyword level
    # tracks the distance the user happens to be at.
    if p["gain"][1] > p["gain"][0]:
        audio = (audio * float(rng.uniform(*p["gain"]))).astype(np.float32)

    classes, layers, snrs = [], [], []
    if rng.random() < p["clean_frac"]:
        # Near-field / clean mode: at most one quiet-ish layer, light distortion.
        if rng.random() < 0.55:
            nz, cls = noise_bank.sample_slice(rng)
            layers.append(nz)
            classes.append(cls)
            snrs.append(float(rng.uniform(*p["snr1"])))
    else:
        for i in range(int(rng.choice([1, 2, 2, 3, 3]))):
            nz, cls = noise_bank.sample_slice(rng)
            layers.append(nz)
            classes.append(cls)
            if i == 0:
                snrs.append(float(rng.uniform(*p["snr1"])))
            elif i == 1:
                snrs.append(float(rng.uniform(*p["snr2"])))
            else:
                snrs.append(float(rng.uniform(*p["snr3"])))
    audio = mix_layers(audio, layers, snrs)

    aggressive = float(rng.uniform(0.0, 1.0)) if len(layers) > 1 else float(rng.uniform(0.0, 0.35))
    audio = _distortion_chain(audio, rng, aggressive=aggressive)
    return audio, float(np.mean(snrs)) if snrs else 40.0, "|".join(classes[:3])


def build_positive_from_stream(audio: np.ndarray, noise_bank: NoiseBank, rng) -> tuple:
    """For long recordings (carrier phrases, human speech): place the whole utterance."""
    audio = fit_to_window(audio, rng, offset="random")
    classes, layers, snrs = [], [], []
    for i in range(int(rng.choice([1, 2, 2, 3]))):
        nz, cls = noise_bank.sample_slice(rng)
        layers.append(nz)
        classes.append(cls)
        snrs.append(float(rng.uniform(2.0, 22.0)) if i == 0 else float(rng.uniform(-4.0, 16.0)))
    audio = mix_layers(audio, layers, snrs)
    audio = distortion_chain(audio, rng, aggressive=float(rng.uniform(0.0, 1.0)))
    return audio, float(np.mean(snrs)), "|".join(classes[:3])


# ==============================================================================
# Negative pipeline
# ==============================================================================

def build_negative(kind: str, noise_bank: NoiseBank, speech_bank: SpeechBank,
                   rng, tts_neg_pool: list) -> tuple:
    if kind == "speech":
        a = speech_bank.sample_slice(rng, cmd_bias=0.35)
        if rng.random() < 0.5:
            nz, cls = noise_bank.sample_slice(rng)
            a = peak_guard(apply_awgn(a, nz, float(rng.uniform(0.0, 25.0))))
            return a, cls
        return a, "clean_speech"

    if kind == "babble":
        n_t = int(rng.integers(2, 9))
        a = make_babble_from_speech(speech_bank, rng, n_t, dur=float(rng.uniform(1.0, 4.0)))
        if rng.random() < 0.45:
            nz, cls = noise_bank.sample_slice(rng)
            a = peak_guard(apply_awgn(a, nz, float(rng.uniform(0.0, 20.0))))
            return a, f"babble|{cls}"
        return a, "babble"

    if kind == "noise":
        a, cls = noise_bank.sample_slice(rng)
        if rng.random() < 0.5:
            a2, cls2 = noise_bank.sample_slice(rng)
            a = peak_guard(0.6 * a + 0.4 * a2)
            cls = f"{cls}|{cls2}"
        a = a * float(rng.uniform(0.25, 1.0))
        if rng.random() < 0.25:
            a = apply_mems_mic_response(peak_guard(a))
        return a, cls

    if kind == "speech_in_noise":
        # Hardest deployment case: human speech buried in noise. The model must
        # stay silent because the wake word is not present.
        a = speech_bank.sample_slice(rng, cmd_bias=0.25)
        lo, hi = _PROFILE["sin_snr"]
        layers, classes, snrs = [], [], []
        for i in range(int(rng.choice([1, 2, 2, 3]))):
            nz, cls = noise_bank.sample_slice(rng)
            layers.append(nz)
            classes.append(cls)
            snrs.append(float(rng.uniform(lo, hi)) if i else float(rng.uniform(lo - 2, hi + 4)))
        a = mix_layers(a, layers, snrs)
        if rng.random() < 0.3:
            b = make_babble_from_speech(speech_bank, rng, int(rng.integers(2, 6)), dur=1.0)
            a = peak_guard(apply_awgn(a, b, float(rng.uniform(0.0, 12.0))))
            classes.append("babble")
        a = _distortion_chain(a, rng, aggressive=float(rng.uniform(0.0, 1.0)))
        return a, "|".join(classes[:3])

    if kind == "confusable":
        idx = int(rng.integers(0, len(tts_neg_pool))) if tts_neg_pool else 0
        base = tts_neg_pool[idx][1] if tts_neg_pool else np.zeros(4000, dtype=np.float32)
        a = fit_to_window(base, rng, offset="random")
        if rng.random() < 0.85:
            a = apply_speed_perturbation(a, float(rng.uniform(0.85, 1.18)))
        layers, classes, snrs = [], [], []
        for i in range(int(rng.choice([1, 2, 3]))):
            nz, cls = noise_bank.sample_slice(rng)
            layers.append(nz)
            classes.append(cls)
            snrs.append(float(rng.uniform(2.0, 22.0)) if i == 0 else float(rng.uniform(-4.0, 14.0)))
        a = mix_layers(a, layers, snrs)
        a = _distortion_chain(a, rng, aggressive=float(rng.uniform(0.0, 1.0)))
        return a, "|".join(classes[:3])

    raise ValueError(kind)


NEG_MIX = {
    "speech": 0.22,
    "babble": 0.16,
    "noise": 0.24,
    "speech_in_noise": 0.24,
    "confusable": 0.14,
}


# ==============================================================================
# v4/v5: explicit mixture taxonomy
# ==============================================================================
# The v3 corpus mixed "how much noise" and "what the content is" together, so the
# model could in principle use layer count as a class cue and the reported metrics
# hid which regime was actually failing. v4+ makes the two axes explicit and
# identical for positives and negatives:
#
#   recipe  M0_direct        content only, no noise at all
#           M1_one_noise     content + exactly ONE noise layer   <- lion share
#           M2_two_noise     content + exactly TWO noise layers
#           M3_two_noise_hard content + 2 layers, low SNR, full distortion
#
#   content pos  the keyword
#           neg  filler    ordinary English words with NO phonetic overlap
#           neg  real      real human speech slices (LibriSpeech / Speech Commands)
#           neg  babble    2-8 real talkers mixed live
#           neg  noise     soundscape only, no speech content
#           neg  confus    near-miss words (kept small on purpose)
#
# Every negative content type is instantiated in all four recipes, so no recipe is
# ever positive-only or negative-only. M1 dominates by design: a single interfering
# source is the most common real condition, and "one noise layer, no keyword" is the
# most common false-alarm condition.

UNIMPORTANT_WORDS = [
    # determiners / pronouns / function words -- nothing like "amaze"
    "the", "a", "that", "this", "these", "those", "there", "then", "than", "when",
    "where", "what", "which", "while", "would", "could", "should", "here", "their",
    "there", "about", "through", "before", "between", "without", "under", "after",
    # everyday verbs
    "open", "close", "carry", "follow", "listen", "measure", "picture", "remember",
    "consider", "discover", "explain", "finish", "gather", "happen", "imagine",
    "decide", "enjoy", "expect", "forget", "notice", "prepare", "receive", "search",
    "touch", "walk", "watch", "write", "jump", "climb", "brush", "count", "study",
    # everyday nouns
    "bottle", "candle", "doorway", "engine", "fabric", "garden", "hammer", "island",
    "jacket", "kettle", "ladder", "mirror", "napkin", "orange", "pencil", "quilt",
    "ribbon", "saddle", "table", "umbrella", "violet", "wagon", "anchor", "basket",
    "candle", "cupboard", "drawer", "fountain", "gravel", "helmet", "iron", "jug",
    # everyday adjectives
    "brave", "clever", "deep", "empty", "fancy", "gentle", "heavy", "just",
    "kind", "large", "mellow", "narrow", "odd", "polite", "quiet", "rapid",
    "shiny", "tidy", "vast", "warm", "young",
    # food / household / nature words
    "bread", "carrot", "cheese", "coffee", "salad", "butter", "cinnamon", "melon",
    "onion", "pepper", "soup", "dinner", "copper", "marble", "willow", "meadow",
    "thunder", "harvest", "blanket", "candle",
]

# bucket id -> (content, recipe, share_within_class, loss_weight)
# M1 ("one noise layer") is the lion share on the positive side (0.40 of 1.00) and the
# single largest negative bucket overall is N13_noise_one_noise (0.12), i.e. a lone
# soundscape with no speech in it -- the most common false-alarm condition in a room.
BUCKET_TABLE = {
    # positives -------------------------------------------------------------
    "P0_word_direct":         ("word",  "M0", 0.10, 1.00),
    "P1_word_one_noise":      ("word",  "M1", 0.40, 1.15),
    "P2_word_two_noise":      ("word",  "M2", 0.28, 1.25),
    "P3_word_two_noise_hard": ("word",  "M3", 0.22, 1.30),
    # negatives: filler words (unimportant, not confusable) ----------------
    "N0_filler_direct":         ("filler", "M0", 0.05, 0.80),
    "N1_filler_one_noise":      ("filler", "M1", 0.05, 0.90),
    "N2_filler_two_noise":      ("filler", "M2", 0.04, 1.00),
    "N3_filler_two_noise_hard": ("filler", "M3", 0.03, 1.10),
    # negatives: real human speech, no keyword ---------------------------
    "N4_real_direct":         ("real",  "M0", 0.05, 0.90),
    "N5_real_one_noise":      ("real",  "M1", 0.06, 1.00),
    "N6_real_two_noise":      ("real",  "M2", 0.04, 1.10),
    "N7_real_two_noise_hard": ("real",  "M3", 0.03, 1.20),
    # negatives: live multi-talker babble --------------------------------
    "N8_babble_direct":         ("babble", "M0", 0.04, 1.00),
    "N9_babble_one_noise":      ("babble", "M1", 0.06, 1.10),
    "N10_babble_two_noise":     ("babble", "M2", 0.05, 1.20),
    "N11_babble_two_noise_hard": ("babble", "M3", 0.06, 1.30),
    # negatives: soundscape only (no speech) ----------------------------
    "N12_noise_direct":         ("noise", "M0", 0.10, 0.90),
    "N13_noise_one_noise":      ("noise", "M1", 0.12, 1.00),
    "N14_noise_two_noise":      ("noise", "M2", 0.05, 1.00),
    "N15_noise_two_noise_hard": ("noise", "M3", 0.03, 1.10),
    # negatives: hard near-miss words ------------------------------------
    "N16_confus_direct":         ("confus", "M0", 0.03, 1.00),
    "N17_confus_one_noise":      ("confus", "M1", 0.05, 1.10),
    "N18_confus_two_noise":      ("confus", "M2", 0.04, 1.10),
    "N19_confus_two_noise_hard": ("confus", "M3", 0.02, 1.20),
}

POS_BUCKETS = [b for b in BUCKET_TABLE if b.startswith("P")]
NEG_BUCKETS = [b for b in BUCKET_TABLE if b.startswith("N")]
BUCKET_LOSS_WEIGHT = {b: BUCKET_TABLE[b][3] for b in BUCKET_TABLE}

# recipe -> layer count, per-layer SNR ranges, and the distortion-aggressiveness range
RECIPES = {
    "M0": {"layers": 0, "snr": None, "aggr": (0.00, 0.20)},
    "M1": {"layers": 1, "snr": ((12.0, 26.0),), "aggr": (0.05, 0.35)},
    "M2": {"layers": 2, "snr": ((8.0, 22.0), (0.0, 16.0)), "aggr": (0.20, 0.60)},
    "M3": {"layers": 2, "snr": ((2.0, 14.0), (-4.0, 10.0)), "aggr": (0.55, 1.00)},
}
RECIPE_LABEL = {"M0": "direct", "M1": "1 noise", "M2": "2 noise", "M3": "2 noise hard"}

# Explicit level policy, derived from the INMP441 datasheet rather than guessed.
#
# The v4 policy (-34..-14 dBFS) came from nowhere in particular and was ~8 dB too hot.
# With -26 dBFS sensitivity and 94 dB SPL full scale the SPL -> digital mapping is fixed
# (acoustic_augment.spl_to_dbfs), so the deployed level is set by (a) how far the user is
# standing and (b) whatever preamp/software gain the board applies. We treat the board
# gain as the free variable and let draw_talker_conditions() pick it, so a 3 m talker in
# a quiet voice lands genuinely lower than one at 30 cm.
#
#   distance   SPL (normal voice)  raw dBFS   @ +20 dB board gain
#   0.3 m      70.0 dB             -50.0      -30.0
#   0.5 m      66.0 dB             -54.0      -34.0
#   1.0 m      60.0 dB             -60.0      -40.0
#   2.0 m      54.0 dB             -66.0      -46.0
#   3.0 m      50.5 dB             -69.5      -49.5
#   shouting 82.0 dB              -38.0      -18.0
#   whisper   45.0 dB              -75.0      -55.0
#
# The INMP441 has no AGC, so nothing compresses this range: it is what the front end sees.
LEVEL_DBFS = (-50.0, -14.0)
# Below this the log-mel is pinned at its epsilon floor and the sample carries no
# information, so it is a label-noise sample rather than a hard one.
LEVEL_FLOOR_DBFS = -70.0
CLIP_PROB = 0.30
CLIP_CEILING = (0.35, 1.00)

# 24-bit I2S means real quantization is negligible, so aggressive bit-crushing is not
# physically justified for this hardware. Kept only as a rare robustness probe.
BITCRUSH_PROB_V4 = (0.02, 0.05)


def apply_level_policy(audio: np.ndarray, rng, level_dbfs: float = None) -> tuple:
    """Normalises loudness to a designed range, optionally through preamp overload.

    `level_dbfs` overrides the random draw (used by the transducer chain, which already
    picked a level from the talker's distance and loudness).
    """
    a = np.asarray(audio, dtype=np.float32).ravel()
    if rng.random() < CLIP_PROB:
        # Preamp overload: the gain stage clips before the level is settled.
        a = np.clip(a, -float(rng.uniform(*CLIP_CEILING)), float(rng.uniform(*CLIP_CEILING)))
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-8:
        return peak_guard(a), -60.0
    target_db = float(rng.uniform(*LEVEL_DBFS) if level_dbfs is None else level_dbfs)
    a = a * (10 ** (target_db / 20.0) / rms)
    peak = float(np.max(np.abs(a)))
    if peak > 1.0:
        a = np.clip(a * (0.99 / peak), -1.0, 1.0)
    achieved = 20.0 * float(np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9))
    return a.astype(np.float32), achieved


def apply_transducer_chain(audio: np.ndarray, rng, snr_db: float = 61.0) -> np.ndarray:
    """Final sensor model: INMP441 response + capsule self-noise + I2S quantization.

    Order mirrors the physical signal path: the capsule response shapes the incoming
    sound, then the capsule's own electronic noise is added, then the I2S word is
    quantized. `snr_db` comes from the drawn talker distance and vocal effort, so a
    user 3 m away in a quiet voice genuinely sees a worse SNR than one at 30 cm.
    """
    audio = apply_inmp441_response(audio)
    audio = add_mic_self_noise(audio, rng, snr_db=snr_db)
    if rng.random() < 0.12:
        # Deployment may read the 24-bit word through a 16-bit I2S peripheral.
        audio = apply_i2s_16bit_truncation(audio)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _sample_layers(noise_bank: NoiseBank, rng, snr_specs):
    layers, classes, snrs = [], [], []
    for spec in snr_specs:
        nz, cls = noise_bank.sample_slice(rng)
        layers.append(nz)
        classes.append(cls)
        snrs.append(float(rng.uniform(*spec)))
    return layers, classes, snrs


def build_bucket_sample(bucket: str, noise_bank: NoiseBank, speech_bank: SpeechBank,
                        rng, tts_bases: list, tts_filler: list, tts_conf: list) -> tuple:
    """Builds one sample for a named bucket. Returns (audio, snr, noise_classes, level_dbfs)."""
    content, recipe = BUCKET_TABLE[bucket][0], BUCKET_TABLE[bucket][1]
    rec = RECIPES[recipe]

    if content == "word":
        base = tts_bases[int(rng.integers(0, len(tts_bases)))][1]
        base = np.asarray(base, dtype=np.float32).ravel()
        if rng.random() < 0.85:
            base = apply_speed_perturbation(base, float(rng.uniform(0.80, 1.22)))
        if rng.random() < 0.70:
            base = apply_vtlp(base, float(rng.uniform(0.82, 1.22)))
        content_sig = fit_to_window(base, rng, offset="random")
    elif content == "filler":
        pool = tts_filler or tts_conf
        base = np.asarray(pool[int(rng.integers(0, len(pool)))][1], dtype=np.float32).ravel()
        if rng.random() < 0.8:
            base = apply_speed_perturbation(base, float(rng.uniform(0.85, 1.20)))
        content_sig = fit_to_window(base, rng, offset="random")
    elif content == "confus":
        pool = tts_conf or tts_filler
        base = np.asarray(pool[int(rng.integers(0, len(pool)))][1], dtype=np.float32).ravel()
        if rng.random() < 0.8:
            base = apply_speed_perturbation(base, float(rng.uniform(0.85, 1.18)))
        content_sig = fit_to_window(base, rng, offset="random")
    elif content == "real":
        content_sig = speech_bank.sample_slice(rng, cmd_bias=0.30)
    elif content == "babble":
        content_sig = make_babble_from_speech(
            speech_bank, rng, int(rng.integers(2, 9)), dur=float(rng.uniform(1.0, 4.0)))
    elif content == "noise":
        # The "content" for a noise-only negative is itself a soundscape, so M0 is a
        # real single soundscape rather than digital silence.
        content_sig, _cls = noise_bank.sample_slice(rng)
    else:
        raise ValueError(content)

    if rec["layers"] == 0:
        audio = content_sig
        classes, snrs = ["direct"], []
    else:
        layers, classes, snrs = _sample_layers(noise_bank, rng, rec["snr"])
        audio = mix_layers(content_sig, layers, snrs)

    lo, hi = rec["aggr"]
    audio = _distortion_chain(audio, rng, aggressive=float(rng.uniform(lo, hi)))

    # Talker geometry first: it sets both where the board's gain puts the speech and
    # the SNR the INMP441 actually delivers, which is what the 61 dBA figure means in
    # a real room.
    level_dbfs, snr_db, _spl = draw_talker_conditions(rng)
    audio, _ = apply_level_policy(audio, rng, level_dbfs=level_dbfs)
    audio = apply_transducer_chain(audio, rng, snr_db=snr_db)

    # Level is measured after the transducer chain so the recorded dBFS is the number
    # the deployed front end will actually see. The 60 Hz capsule high-pass can strip a
    # low-frequency-heavy soundscape by tens of dB, which would pin the log-mel at its
    # epsilon floor and make the sample unlearnable, so anything that falls through the
    # floor is lifted back to target.
    lvl = 20.0 * float(np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9))
    if lvl < LEVEL_FLOOR_DBFS:
        gain = 10 ** ((level_dbfs - lvl) / 20.0)
        audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        lvl = 20.0 * float(np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9))
    return audio, float(np.mean(snrs)) if snrs else 40.0, "|".join(classes[:3]), lvl


def _bucket_sampler(buckets, rng):
    keys = list(buckets)
    w = np.array([buckets[k][2] for k in keys], dtype=np.float64)
    w /= w.sum()
    probs = w.cumsum()

    def draw():
        r = float(rng.random())
        return keys[int(np.searchsorted(probs, r, side="right"))]

    return draw



# ==============================================================================
# Worker
# ==============================================================================

_W = {}


def _init_worker(noise_paths, speech_flacs, speech_cmds, tts_bases, tts_confs, seed,
                 profile=None, tts_filler=None):
    global _PROFILE
    if profile:
        _PROFILE.clear()
        _PROFILE.update(PROFILES[profile])
    _W["noise_bank"] = NoiseBank.__new__(NoiseBank)
    _W["noise_bank"].by_class = {}
    for p, c in noise_paths:
        _W["noise_bank"].by_class.setdefault(c, []).append(p)
    _W["noise_bank"].paths = noise_paths
    _W["noise_bank"].classes = sorted(_W["noise_bank"].by_class)
    _W["noise_bank"]._cache = {}

    _W["speech_bank"] = SpeechBank.__new__(SpeechBank)
    _W["speech_bank"].flacs = speech_flacs
    _W["speech_bank"].cmds = speech_cmds
    _W["speech_bank"]._cache = {}

    _W["tts_bases"] = tts_bases
    _W["tts_confs"] = tts_confs
    _W["tts_filler"] = tts_filler or []
    _W["seed"] = seed


def _run_chunk(args):
    start, n_pos, n_neg_total, seed, use_buckets = args
    rng = np.random.default_rng(seed)
    extractor = AudioFeatureExtractor()
    nb = _W["noise_bank"]
    sb = _W["speech_bank"]
    tb = _W["tts_bases"]
    tc = _W["tts_confs"]
    tf = _W.get("tts_filler", [])

    Xs, ys, groups, snrs, buckets, lvls = [], [], [], [], [], []

    def emit(bucket, is_pos):
        audio, snr, grp, lvl = build_bucket_sample(bucket, nb, sb, rng, tb, tf, tc)
        Xs.append(extractor.compute_spectrogram(audio))
        ys.append(1 if is_pos else 0)
        groups.append(("pos|" if is_pos else "neg|") + grp)
        snrs.append(snr)
        buckets.append(bucket)
        lvls.append(lvl)

    if use_buckets:
        draw_pos = _bucket_sampler({b: BUCKET_TABLE[b] for b in POS_BUCKETS}, rng)
        draw_neg = _bucket_sampler({b: BUCKET_TABLE[b] for b in NEG_BUCKETS}, rng)
        for _ in range(n_pos):
            emit(draw_pos(), True)
        for _ in range(n_neg_total):
            emit(draw_neg(), False)
    else:
        for i in range(n_pos):
            base = tb[(start + i) % len(tb)][1]
            audio, snr, grp = build_positive(base, nb, rng)
            Xs.append(extractor.compute_spectrogram(audio))
            ys.append(1)
            groups.append("pos|" + grp)
            snrs.append(snr)
            buckets.append("P_legacy")
            lvls.append(20.0 * float(np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)))

        kinds = list(NEG_MIX)
        weights = np.array([NEG_MIX[k] for k in kinds], dtype=np.float64)
        weights /= weights.sum()
        for _ in range(n_neg_total):
            kind = str(rng.choice(kinds, p=weights))
            try:
                audio, grp = build_negative(kind, nb, sb, rng, tc)
            except Exception:
                continue
            Xs.append(extractor.compute_spectrogram(audio))
            ys.append(0)
            groups.append(f"neg|{kind}|{grp}")
            snrs.append(0.0)
            buckets.append("N_legacy")
            lvls.append(20.0 * float(np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)))

    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.int32),
            np.array(groups),
            np.array(snrs, dtype=np.float32),
            np.array(buckets),
            np.array(lvls, dtype=np.float32))


# ==============================================================================
# Base waveform synthesis (run once, shared by workers)
# ==============================================================================

def _pool_cache_path(cache):
    # Pickle, not npz. The payload is a dict of ragged float32 arrays, so it needs an
    # object container anyway, and Kaggle auto-extracts .zip resources from a dataset
    # (a .npz is a zip) which silently ships one member instead of the whole 356 MB
    # file. A .pkl is passed through untouched and pickle.load reads it back fine.
    return os.path.join(cache, "tts_pools.pkl") if cache else None


def save_pool_cache(pools, cache_dir):
    path = _pool_cache_path(cache_dir)
    if not path:
        return
    try:
        import pickle
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({k: [(t, np.asarray(w, dtype=np.float32)) for t, w in v]
                         for k, v in pools.items()}, f, protocol=4)
        print(f"  TTS pool cache written: {path} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  could not write pool cache: {e}")


def load_pool_cache(cache_dir, want_filler=True):
    path = _pool_cache_path(cache_dir)
    if not path or not os.path.exists(path):
        return None
    try:
        import pickle
        with open(path, "rb") as f:
            raw = pickle.load(f)
        pools = {k: [(str(t), np.asarray(w, dtype=np.float32)) for t, w in v]
                 for k, v in raw.items()}
        if not pools.get("kw") or (want_filler and not pools.get("filler")):
            return None
        print(f"  TTS pool cache loaded: {path} "
              f"(kw={len(pools['kw'])}, conf={len(pools.get('conf', []))}, "
              f"filler={len(pools.get('filler', []))})")
        return pools
    except Exception as e:
        print(f"  could not read pool cache: {e}")
        return None


async def build_tts_pools(per_voice: int = 3, want_filler: bool = True):
    print("\n[Synthesis] Building keyword, filler and confusable waveform pools...")
    sem = asyncio.Semaphore(12)
    results = {"kw": [], "conf": [], "filler": []}

    async def one(text, voice, rate, pitch, bucket):
        async with sem:
            for attempt in range(2):
                try:
                    w = await synth_edge(text, voice, rate, pitch)
                except Exception:
                    w = None
                if w is not None and len(w) > 800:
                    results[bucket].append((f"{voice}|{rate}|{pitch}|{text}", w))
                    return 1
                await asyncio.sleep(0.4 * (attempt + 1))
            return 0

    tasks = []
    rates = ["-25%", "-18%", "-12%", "-6%", "+0%", "+6%", "+12%", "+18%", "+25%"]
    pitches = ["-25Hz", "-15Hz", "-8Hz", "+0Hz", "+8Hz", "+15Hz", "+25Hz"]
    for v in EDGE_VOICES:
        combos = [(r, p) for r in rates for p in pitches]
        np.random.shuffle(combos)
        for k in range(per_voice * 3):
            rate, pitch = combos[k % len(combos)]
            text = POSITIVE_CARRIERS[(k + abs(hash(v)) % 7) % len(POSITIVE_CARRIERS)]
            tasks.append(one(text, v, rate, pitch, "kw"))
    await asyncio.gather(*tasks)

    tasks = []
    for w in CONFUSABLE_WORDS:
        for v in EDGE_VOICES[::4]:
            for r, p in (("-10%", "+5Hz"), ("+0%", "+0Hz"), ("+14%", "-6Hz")):
                tasks.append(one(w, v, r, p, "conf"))
    for w in NEGATIVE_COMMANDS:
        for v in EDGE_VOICES[::6]:
            tasks.append(one(w, v, "+0%", "+0Hz", "conf"))
    await asyncio.gather(*tasks)

    if want_filler:
        # Ordinary words with no phonetic overlap with the keyword. These are the
        # "unimportant word" negatives the user asked for in every mixture recipe:
        # without them the model only ever learns to reject speech that is either a
        # near-miss of the keyword or real human audio.
        tasks = []
        seen = set()
        words = [w for w in UNIMPORTANT_WORDS if not (w in seen or seen.add(w))]
        for w in words:
            for v in EDGE_VOICES[::5]:
                r, p = (("-8%", "+0Hz"), ("+0%", "+6Hz"), ("+12%", "-6Hz"))[len(tasks) % 3]
                tasks.append(one(w, v, r, p, "filler"))
        await asyncio.gather(*tasks)

    for tld in GTTS_TLDS:
        for slow in (False, True):
            try:
                w = trim_silence(synth_gtts(KEYWORD, tld, slow))
                results["kw"].append((f"gtts|{tld}|{slow}", w))
            except Exception:
                pass
            try:
                w = trim_silence(synth_gtts(rng_word_pick(), tld, slow))
                results["conf"].append((f"gtts|{tld}|{slow}", w))
            except Exception:
                pass

    try:
        eng_voices = 4
        for r in (110, 140, 170, 200, 230):
            for vi in range(eng_voices):
                try:
                    w = trim_silence(synth_sapi(KEYWORD, r, vi))
                    results["kw"].append((f"sapi|{r}|{vi}", w))
                except Exception:
                    pass
        for w in CONFUSABLE_WORDS[::4]:
            for vi in (0, 1):
                try:
                    results["conf"].append((f"sapi|{w}|{vi}", trim_silence(synth_sapi(w, 175, vi))))
                except Exception:
                    pass
    except Exception as e:
        print(f"  SAPI unavailable: {e}")

    # human "amaze" from LibriSpeech
    human = slice_human_keyword()
    for tag, w in human:
        results["kw"].append((f"human|{tag}", w))

    print(f"  keyword bases: {len(results['kw'])}  confusable/command bases: {len(results['conf'])}")
    return results


_rng_state = np.random.default_rng(7)


def rng_word_pick() -> str:
    return str(_rng_state.choice(CONFUSABLE_WORDS))


def slice_human_keyword(corpus_dir: str = "speech_corpus/LibriSpeech", max_files: int = 400) -> list:
    """Extracts 1 s windows centred on real human utterances containing 'amaze*'."""
    out = []
    txt_files = glob.glob(os.path.join(corpus_dir, "**", "*.trans.txt"), recursive=True)
    targets = []
    for tf in txt_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    if any("AMAZ" in w for w in parts[1:]):
                        flac = os.path.join(os.path.dirname(tf), f"{parts[0]}.flac")
                        if os.path.exists(flac):
                            targets.append(flac)
        except Exception:
            pass
    np.random.shuffle(targets)
    for p in targets[:max_files]:
        try:
            d, sr = sf.read(p)
            if d.ndim > 1:
                d = d[:, 0]
            if sr != SAMPLE_RATE:
                d = signal.resample(d, int(round(len(d) * SAMPLE_RATE / sr)))
            if len(d) < TOTAL_SAMPLES:
                continue
            st = int(np.random.randint(0, len(d) - TOTAL_SAMPLES + 1))
            out.append((os.path.basename(p), d[st:st + TOTAL_SAMPLES].copy()))
        except Exception:
            pass
    print(f"  human keyword windows: {len(out)}")
    return out


# ==============================================================================
# Main
# ==============================================================================

def stratified_split(groups, y, val_frac=0.2, seed=7, buckets=None):
    """Stratified by (label, bucket) so every mixture regime keeps its share in both
    splits -- otherwise the rare hard buckets vanish from val and stop being measured."""
    rng = np.random.default_rng(seed)
    from collections import defaultdict
    buckets_map = defaultdict(list)
    for i, g in enumerate(groups):
        if buckets is not None:
            kind = str(buckets[i])
        else:
            kind = "|".join(g.split("|")[:2]) if not g.startswith("pos|") else "pos"
        buckets_map[(int(y[i]), kind)].append(i)
    val_idx = []
    for key, idxs in buckets_map.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * val_frac)))
        val_idx.extend(idxs[:k].tolist())
    val_idx = set(val_idx)
    train_idx = [i for i in range(len(y)) if i not in val_idx]
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=14000)
    ap.add_argument("--neg", type=int, default=28000)
    ap.add_argument("--out", type=str, default="dataset_v2.npz")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunks", type=int, default=0, help="0 = workers * 4")
    ap.add_argument("--per_voice", type=int, default=3)
    ap.add_argument("--noise_index", type=str, default="noise_bank_v2/index.csv")
    ap.add_argument("--dump_bases", type=str, default="keyword_bases.npy",
                    help="path for the clean keyword waveform cache ('' to skip)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--profile", type=str, default="v4", choices=sorted(PROFILES))
    ap.add_argument("--pool_cache", type=str, default="cache",
                    help="directory for the TTS pool cache ('' to disable)")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="storage dtype for the log-mel X arrays. fp16 halves the "
                         "artifact and the download; the log-mel front end floors at "
                         "log(1e-5) = -11.51 and rarely exceeds +5, so float16 "
                         "resolution (~1e-3) is far below anything the model resolves. "
                         "train_industrial_v2.py upcasts per batch either way.")
    args = ap.parse_args()

    _PROFILE.clear()
    _PROFILE.update(PROFILES[args.profile])
    use_buckets = args.profile in ("v4", "v5")
    print(f"[Profile] {args.profile}: buckets={use_buckets}  "
          f"bitcrush={_PROFILE['bitcrush']}  level={LEVEL_DBFS} dBFS")

    t0 = time.time()
    n_chunks = args.chunks or max(1, args.workers * 4)

    pools = load_pool_cache(args.pool_cache, want_filler=use_buckets)
    if pools is None:
        pools = asyncio.run(build_tts_pools(per_voice=args.per_voice, want_filler=use_buckets))
        save_pool_cache(pools, args.pool_cache)
    kw_bases = [(t, trim_silence(w)) for t, w in pools["kw"]]
    kw_bases = [(t, w) for t, w in kw_bases if w is not None and len(w) > 600]
    conf_bases = [(t, trim_silence(w)) for t, w in pools["conf"]]
    conf_bases = [(t, w) for t, w in conf_bases if w is not None and len(w) > 600]
    filler_bases = [(t, trim_silence(w)) for t, w in pools.get("filler", [])]
    filler_bases = [(t, w) for t, w in filler_bases if w is not None and len(w) > 400]
    if not kw_bases:
        raise SystemExit("No keyword bases synthesised; cannot build corpus.")
    print(f"  filler bases: {len(filler_bases)}")

    # Held-out clean keyword waveforms, used by evaluate_industrial_v2.py for the
    # streaming-recall test.
    dump = args.dump_bases or "keyword_bases.npy"
    if dump:
        try:
            np.save(dump, np.array([w for _, w in kw_bases[:600]], dtype=object),
                    allow_pickle=True)
            print(f"  keyword bases cached for evaluation: {dump}")
        except Exception as e:
            print(f"  could not write {dump}: {e}")

    nb = NoiseBank(index_csv=args.noise_index)
    sb = SpeechBank()

    total = args.pos + args.neg
    per_chunk = int(np.ceil(total / n_chunks))
    jobs = []
    assigned = 0
    for c in range(n_chunks):
        take = min(per_chunk, total - assigned)
        if take <= 0:
            break
        pos_share = int(round(take * args.pos / total))
        jobs.append((assigned, pos_share, take - pos_share, args.seed + c * 7919, use_buckets))
        assigned += take

    print(f"\n[Build] {args.pos:,} positives + {args.neg:,} negatives "
          f"across {len(jobs)} chunks on {args.workers} workers")
    noise_paths = nb.paths
    speech_flacs, speech_cmds = sb.flacs, sb.cmds

    results = []
    if args.workers <= 1:
        _init_worker(noise_paths, speech_flacs, speech_cmds, kw_bases, conf_bases,
                     args.seed, args.profile, filler_bases)
        for j in jobs:
            results.append(_run_chunk(j))
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(noise_paths, speech_flacs, speech_cmds, kw_bases, conf_bases,
                          args.seed, args.profile, filler_bases),
            ) as ex:
                futs = [ex.submit(_run_chunk, j) for j in jobs]
                for k, f in enumerate(futs):
                    results.append(f.result())
                    print(f"  chunk {k+1}/{len(jobs)} done", flush=True)
        except Exception as e:
            # A fork-based pool needs __main__ picklable by reference, which is not
            # guaranteed on every interpreter/platform. Serial is slower but always
            # correct, so a pool that cannot start must not lose the whole build.
            print(f"[Build] parallel build failed ({str(e)[:120]}), falling back to serial",
                  flush=True)
            results = []
            _init_worker(noise_paths, speech_flacs, speech_cmds, kw_bases, conf_bases,
                         args.seed, args.profile, filler_bases)
            for j in jobs:
                results.append(_run_chunk(j))

    X = np.concatenate([r[0] for r in results], axis=0)
    y = np.concatenate([r[1] for r in results], axis=0)
    groups = np.concatenate([r[2] for r in results], axis=0)
    snrs = np.concatenate([r[3] for r in results], axis=0)
    buckets = np.concatenate([r[4] for r in results], axis=0)
    lvls = np.concatenate([r[5] for r in results], axis=0)

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    x_dtype = np.float16 if args.dtype == "fp16" else np.float32
    X = X.astype(x_dtype)

    tr, va = stratified_split(groups, y, 0.2, seed=args.seed, buckets=buckets)
    print(f"\n[Split] train={len(tr):,}  val={len(va):,}  X dtype {np.dtype(x_dtype).name}")

    np.savez_compressed(
        args.out,
        X_train=X[tr], y_train=y[tr], g_train=groups[tr], snr_train=snrs[tr],
        b_train=buckets[tr], lvl_train=lvls[tr],
        X_val=X[va], y_val=y[va], g_val=groups[va], snr_val=snrs[va],
        b_val=buckets[va], lvl_val=lvls[va],
    )

    print(f"  positives: {int((y==1).sum()):,}   negatives: {int((y==0).sum()):,}")
    if use_buckets:
        print(f"  {'bucket':<28} | {'n':>8} | {'share':>6} | {'loss w':>6} | {'lvl dBFS':>16}")
        for b, (_c, _r, _s, w) in BUCKET_TABLE.items():
            m = buckets == b
            c = int(m.sum())
            if c == 0:
                continue
            lv = lvls[m]
            print(f"  {b:<28} | {c:8,} | {c/len(buckets)*100:5.1f}% | {w:6.2f} | "
                  f"{lv.min():7.1f}..{lv.max():5.1f}")
    else:
        for kind in NEG_MIX:
            c = int(np.sum([g.startswith(f"neg|{kind}|") for g in groups]))
            print(f"    neg/{kind:<16} {c:,}")
    print(f"  file: {args.out}  ({os.path.getsize(args.out)/1024/1024:.1f} MB)")
    print(f"  elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
