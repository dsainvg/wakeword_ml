"""
Automated Hard Negative Miner (OHEM for Keyword Spotting)
-------------------------------------------------------
Passes continuous soundscapes (sirens, dog barks, babble), real speech streams
(LibriSpeech), and human command words (Google Speech Commands) through the current
best model to harvest false positive slices (P(Amaze) >= 0.15).

These mined hard negatives are the most difficult examples for the model,
and retraining on them eliminates false alarms.
"""

import os
import glob
import time
import numpy as np
import soundfile as sf
from scipy import signal
import jax.numpy as jnp
from flax import serialization

from features import AudioFeatureExtractor

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000
from models import get_model


def load_model(checkpoint_path: str = "best_kws_model.flax", arch: str = "bcresnet3"):
    with open(checkpoint_path, "rb") as f:
        data = f.read()
    checkpoint = serialization.msgpack_restore(data)
    model = get_model(arch, num_classes=2)
    params = checkpoint["params"]

    def predict_batch(x):
        logits = model.apply({"params": params}, x, train=False)
        probs = jnp.exp(logits[:, 1] - jnp.max(logits, axis=-1)) / jnp.sum(
            jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True)), axis=-1
        )
        return probs

    return predict_batch


def mine_from_noises(predict_fn, extractor, noise_dir: str = "noise_bank", stride_ms: int = 100):
    hard_specs = []
    hard_probs = []
    wav_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    stride_samples = int(stride_ms * SAMPLE_RATE / 1000)

    print(f"\n[Mining] Scanning {len(wav_files)} environmental soundscapes...")
    for wpath in wav_files:
        fname = os.path.basename(wpath)
        try:
            audio, sr = sf.read(wpath)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

            num_steps = (len(audio) - TOTAL_SAMPLES) // stride_samples
            if num_steps <= 0:
                continue

            for s in range(0, num_steps, 2):  # step every 200ms
                chunk = audio[s * stride_samples: s * stride_samples + TOTAL_SAMPLES]
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.005:
                    continue
                spec = extractor.compute_spectrogram(chunk)
                tensor = jnp.expand_dims(jnp.expand_dims(jnp.array(spec), axis=0), axis=-1)
                prob = float(predict_fn(tensor)[0])
                if prob >= 0.15:
                    hard_specs.append(spec)
                    hard_probs.append(prob)
        except Exception:
            pass

    print(f"  Mined {len(hard_specs)} hard negatives from environmental soundscapes.")
    return hard_specs, hard_probs


def mine_from_speech_commands(predict_fn, extractor, commands_dir: str = "speech_corpus/speech_commands", max_files: int = 2000):
    hard_specs = []
    hard_probs = []
    wavs = glob.glob(os.path.join(commands_dir, "**", "*.wav"), recursive=True)
    np.random.seed(42)
    if len(wavs) > max_files:
        wavs = list(np.random.choice(wavs, max_files, replace=False))

    print(f"\n[Mining] Scanning {len(wavs)} Google Speech Commands audio files...")
    for wpath in wavs:
        try:
            audio, sr = sf.read(wpath)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

            # Pad or slice to exactly 1.0s (TOTAL_SAMPLES)
            if len(audio) < TOTAL_SAMPLES:
                audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)))
            else:
                audio = audio[:TOTAL_SAMPLES]

            spec = extractor.compute_spectrogram(audio)
            tensor = jnp.expand_dims(jnp.expand_dims(jnp.array(spec), axis=0), axis=-1)
            prob = float(predict_fn(tensor)[0])
            if prob >= 0.15:
                hard_specs.append(spec)
                hard_probs.append(prob)
        except Exception:
            pass

    print(f"  Mined {len(hard_specs)} hard negatives from Google Speech Commands.")
    return hard_specs, hard_probs


def mine_from_librispeech(predict_fn, extractor, corpus_dir: str = "speech_corpus/LibriSpeech", num_files: int = 200):
    hard_specs = []
    hard_probs = []
    flacs = []
    for root, _, files in os.walk(corpus_dir):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))

    np.random.seed(42)
    selected = list(np.random.choice(flacs, min(num_files, len(flacs)), replace=False))
    stride_samples = int(0.10 * SAMPLE_RATE)

    print(f"\n[Mining] Scanning {len(selected)} LibriSpeech continuous audiobook files...")
    for fpath in selected:
        try:
            audio, sr = sf.read(fpath)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

            num_steps = (len(audio) - TOTAL_SAMPLES) // stride_samples
            if num_steps <= 0:
                continue

            for s in range(0, num_steps, 2):
                chunk = audio[s * stride_samples: s * stride_samples + TOTAL_SAMPLES]
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.015:
                    continue
                spec = extractor.compute_spectrogram(chunk)
                tensor = jnp.expand_dims(jnp.expand_dims(jnp.array(spec), axis=0), axis=-1)
                prob = float(predict_fn(tensor)[0])
                if prob >= 0.15:
                    hard_specs.append(spec)
                    hard_probs.append(prob)
        except Exception:
            pass

    print(f"  Mined {len(hard_specs)} hard negatives from continuous LibriSpeech.")
    return hard_specs, hard_probs


def main():
    print("=" * 65)
    print(" HARD NEGATIVE MINING ENGINE (OHEM FOR WAKE-WORD 'AMAZE')")
    print("=" * 65)
    t0 = time.time()
    predict_fn = load_model("best_kws_model.flax", "bcresnet3")
    extractor = AudioFeatureExtractor()

    specs_noise, probs_noise = mine_from_noises(predict_fn, extractor)
    specs_cmd, probs_cmd = mine_from_speech_commands(predict_fn, extractor)
    specs_libri, probs_libri = mine_from_librispeech(predict_fn, extractor)

    all_hard_specs = specs_noise + specs_cmd + specs_libri
    all_hard_probs = probs_noise + probs_cmd + probs_libri

    if all_hard_specs:
        X_hard = np.array(all_hard_specs, dtype=np.float32)
        y_hard = np.zeros(len(all_hard_specs), dtype=np.int32)
        np.savez_compressed("hard_negatives.npz", X_hard=X_hard, y_hard=y_hard, probs=all_hard_probs)
        print(f"\n[Success] Harvested {len(all_hard_specs)} HARD NEGATIVE slices in {time.time() - t0:.1f}s.")
        print(f"          Saved to 'hard_negatives.npz' ({os.path.getsize('hard_negatives.npz') / (1024*1024):.2f} MB).")
        print(f"          Average false-alarm confidence of mined slices: {np.mean(all_hard_probs):.4f}")
        print(f"          Max false-alarm confidence of mined slices:     {np.max(all_hard_probs):.4f}")
    else:
        print("\n[Notice] No hard negatives found with P(Amaze) >= 0.15.")


if __name__ == "__main__":
    main()
