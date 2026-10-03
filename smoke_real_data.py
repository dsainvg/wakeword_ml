"""
End-to-end smoke test: real downloaded audio -> features -> new KWSNet -> loss.

Everything so far has been verified on synthetic tensors. This checks the actual
chain a training run will use, against the real corpora on disk:

    LibriSpeech / ESC-50 / UrbanSound8K wav
      -> AudioFeatureExtractor  (features.py, 40-band log-mel, (49, 40))
      -> KWSNet                 (models_torch.py, (N, 49, 40, 1))
      -> cross-entropy + backward on GPU

It also reports the feature-domain statistics of real audio, because the corpus
builders assume particular log-mel energy values and a level policy. If the real
data sits elsewhere, every downstream bucket/SNR calculation would be quietly
wrong -- and that would be invisible in the loss curve.

    py smoke_real_data.py
"""

import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import AudioFeatureExtractor          # noqa: E402
from models_torch import get_torch_model, count_macs, INPUT_SHAPE  # noqa: E402

OUT = []


def emit(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def load_wave_16k(path, seconds=1.0):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        n_out = int(round(len(data) * 16000 / sr))
        xo = np.linspace(0, 1, len(data), endpoint=False)
        xn = np.linspace(0, 1, n_out, endpoint=False)
        data = np.interp(xn, xo, data).astype(np.float32)
    need = int(seconds * 16000)
    if len(data) < need:
        data = np.pad(data, (0, need - len(data)))
    return data[:need]


def sample_files(pattern, n):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return []
    step = max(1, len(files) // n)
    return files[::step][:n]
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emit("device: %s" % (torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))
    emit("")

    ex = AudioFeatureExtractor()
    emit("AudioFeatureExtractor: %d mel bands, %.0f-%.0f Hz, expect %s"
         % (ex.n_mels, ex.f_min, ex.f_max, (49, 40)))
    emit("")

    groups = {
        "LibriSpeech": sample_files(os.path.join("speech_corpus", "LibriSpeech", "**", "*.flac"), 64),
        "ESC-50": sample_files(os.path.join("noise_bank_v2", "raw", "esc50", "*.wav"), 64),
        "UrbanSound8K": sample_files(os.path.join("noise_bank_v2", "raw", "urbansound", "*.wav"), 64),
    }

    emit("%-14s %6s %8s %11s %11s %11s"
         % ("corpus", "n", "rms", "logmel_min", "logmel_max", "logmel_std"))
    emit("-" * 68)
    feats_by_group = {}
    for name, files in groups.items():
        if not files:
            emit("%-14s %6d  (no files found)" % (name, 0))
            continue
        feats, rmss = [], []
        for p in files:
            try:
                w = load_wave_16k(p)
                rmss.append(float(np.sqrt(np.mean(w ** 2) + 1e-12)))
                feats.append(ex.compute_spectrogram(w))
            except Exception as e:
                emit("  UNREADABLE %s: %s" % (os.path.basename(p), str(e)[:60]))
        if not feats:
            continue
        Fm = np.stack(feats)
        feats_by_group[name] = Fm
        emit("%-14s %6d %8.4f %11.3f %11.3f %11.3f"
             % (name, len(Fm), float(np.mean(rmss)), Fm.min(), Fm.max(), Fm.std()))
    emit("")

    if "LibriSpeech" not in feats_by_group:
        emit("FATAL: no LibriSpeech features; cannot smoke test.")
        open("smoke.out", "w", encoding="utf-8").write("\n".join(OUT))
        return 1

    allf = np.concatenate([v for v in feats_by_group.values()], axis=0)
    x = torch.from_numpy(allf[:, :, :, None]).float()
    # Labels are arbitrary: this is a plumbing test, not a training run.
    y = torch.randint(0, 2, (x.shape[0],))
    n_lib = feats_by_group["LibriSpeech"].shape[0]
    emit("mixed batch: %s from %d corpora" % (tuple(x.shape), len(feats_by_group)))
    emit("")

    for arch in ("kws_ultra", "kws_tight", "kws_moderate"):
        model = get_torch_model(arch).to(dev)
        model.train()
        t0 = time.time()
        logits = model(x.to(dev))
        loss = F.cross_entropy(logits, y.to(dev))
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        mac = count_macs(model)
        params = sum(p.numel() for p in model.parameters())
        ok = bool(torch.isfinite(logits).all() and torch.isfinite(loss)
                  and torch.isfinite(gnorm))
        emit("  %-13s params %6d  MAC %9d  logits %s  loss %.4f  |g| %9.3f  %s  %.3fs"
             % (arch, params, mac, tuple(logits.shape), loss.item(), gnorm.item(),
                "finite" if ok else "NON-FINITE", dt))
        emit("                 floor %.3f ms -> %.1fx headroom in the 11.11 ms budget"
             % (mac / 3_840_000, 11.11 / (mac / 3_840_000)))

    emit("")
    model = get_torch_model("kws_tight").to(dev).eval()
    with torch.no_grad():
        a = model(x[:16].to(dev))
        b = model(x[:16].to(dev))
    same = bool(torch.allclose(a, b, atol=0, rtol=0))
    emit("determinism (eval, identical inputs -> bitwise identical): %s" % same)
    if not same:
        emit("  max abs diff: %.3e" % (a - b).abs().max().item())

    with torch.no_grad():
        p = torch.softmax(model(x.to(dev)), dim=-1)[:, 1]
    emit("")
    emit("untrained net, mean P(keyword):")
    emit("  LibriSpeech (speech) : %.4f" % p[:n_lib].mean().item())
    if p[n_lib:].numel():
        emit("  noise corpora        : %.4f" % p[n_lib:].mean().item())
    emit("  (untrained, so this only proves the forward pass is well-conditioned.)")
    emit("   It is NOT an accuracy result.")

    open("smoke.out", "w", encoding="utf-8").write("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())