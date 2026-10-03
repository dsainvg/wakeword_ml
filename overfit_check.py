"""
Overfit sanity check: can KWSNet actually learn this data on this GPU?

This is not a benchmark. It trains kws_tight on ~120 REAL clips (LibriSpeech
speech as positives, ESC-50/UrbanSound8K noise as negatives) for a few dozen
steps and asks one question: does the loss go down and does train accuracy rise?

That distinguishes three very different failure modes that look identical in a
loss curve on synthetic data:
  - the model cannot fit the labels (architecture or optimiser problem)
  - the forward pass is silently wrong (data/plumbing problem)
  - everything works and it just needs more data / steps

    py overfit_check.py
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
from models_torch import get_torch_model, count_macs  # noqa: E402

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
        data = np.interp(np.linspace(0, 1, n_out, endpoint=False),
                         np.linspace(0, 1, len(data), endpoint=False),
                         data).astype(np.float32)
    need = int(seconds * 16000)
    if len(data) < need:
        data = np.pad(data, (0, need - len(data)))
    return data[:need]


def pick(pattern, n):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return []
    step = max(1, len(files) // n)
    return files[::step][:n]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ex = AudioFeatureExtractor()

    speech = pick(os.path.join("speech_corpus", "LibriSpeech", "**", "*.flac"), 80)
    noise = (pick(os.path.join("noise_bank_v2", "raw", "esc50", "*.wav"), 40)
             + pick(os.path.join("noise_bank_v2", "raw", "urbansound", "*.wav"), 40))
    emit("positives (LibriSpeech speech): %d" % len(speech))
    emit("negatives (ESC-50 + UrbanSound8K): %d" % len(noise))
    if len(speech) < 8 or len(noise) < 8:
        emit("FATAL: not enough real data on disk.")
        return 1

    X, Y = [], []
    for p in speech:
        X.append(ex.compute_spectrogram(load_wave_16k(p)))
        Y.append(1)
    for p in noise:
        X.append(ex.compute_spectrogram(load_wave_16k(p)))
        Y.append(0)

    X = torch.from_numpy(np.stack(X)[:, :, :, None]).float().to(dev)
    Y = torch.tensor(Y, dtype=torch.long, device=dev)
    emit("dataset: %s  positives %d  negatives %d"
         % (tuple(X.shape), int(Y.sum()), int((Y == 0).sum())))
    emit("")

    torch.manual_seed(0)
    model = get_torch_model("kws_tight").to(dev)
    mac = count_macs(model)
    emit("kws_tight: %d params, %d MAC, floor %.3f ms"
         % (sum(p.numel() for p in model.parameters()), mac, mac / 3_840_000))
    emit("")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = 150
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=steps, pct_start=0.2)

    emit("step   loss     acc    P(speech)")
    emit("-" * 40)
    t0 = time.time()
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, Y, label_smoothing=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 15 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                lg = model(X)
                pr = torch.softmax(lg, -1)
                acc = (lg.argmax(-1) == Y).float().mean().item()
                ps = pr[Y == 1, 1].mean().item()
            emit("%4d  %7.4f  %5.1f%%  %.4f"
                 % (step, loss.item(), acc * 100, ps))

    dt = time.time() - t0
    emit("")
    emit("%d steps in %.1f s  (%.3f s/step incl. full-batch fwd+bwd)"
         % (steps, dt, dt / steps))

    model.eval()
    with torch.no_grad():
        lg = model(X)
        pr = torch.softmax(lg, -1)
    acc = (lg.argmax(-1) == Y).float().mean().item()
    emit("final train accuracy: %.1f%%" % (acc * 100))
    emit("mean P(keyword) on speech : %.4f" % pr[Y == 1, 1].mean().item())
    emit("mean P(keyword) on noise  : %.4f" % pr[Y == 0, 1].mean().item())
    emit("")
    if acc > 0.9:
        emit("VERDICT: the model FITS this data. Architecture, forward pass and")
        emit("         optimiser are all sound end-to-end on real audio.")
    elif acc > 0.7:
        emit("VERDICT: partial fit. Learning is happening but 150 full-batch steps")
        emit("         on ~%d samples is a small budget. Not alarming." % len(Y))
    else:
        emit("VERDICT: did NOT fit. Investigate before scaling to a real run.")
    emit("")
    emit("NOTE: this is a train-set fit on %d samples with TTS-free real audio." % len(Y))
    emit("      It says nothing about held-out recall or false-alarm rate, which")
    emit("      is the number the audit actually cares about.")

    open("overfit.out", "w", encoding="utf-8").write("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())