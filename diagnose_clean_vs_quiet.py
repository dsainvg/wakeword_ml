"""Why clean audio scores well and the quiet room does not: separate the two causes.

Clean held-out audio is measured with the keyword centred in a 1 s window and no noise
bed. The quiet-room test puts it at 0.5-1 dB above a real bed. That is a ~40 dB drop in
effective SNR, so the model sees far less of the keyword -- but the streaming number can
also fall for a completely different reason: the detector requires 2-of-2 peak-hold
confirmation, and a 0.5-0.7 s keyword only occupies 2-3 of the windows in a 1 s sliding
stream. A single hot window raises the reported score without ever confirming.

This measures, per SNR:

  frame recall     fraction of keyword-bearing windows scoring above threshold
  confirmed recall fraction of utterances the detector actually fires on
  confirmation gap frame recall minus confirmed recall -- pure detector loss

and re-runs the confirmed column across confirmation policies, so the fix (more model
training vs a different confirmation rule) is chosen from data rather than guessed.
"""
import argparse
from collections import defaultdict

import numpy as np

import build_corpus_v2 as B
from features import AudioFeatureExtractor
from kws_engine import StreamingKWSEngine

SR = 16000
WIN = 16000
CHUNK = 1600


def splice(base, bed, rng, snr_db):
    out = np.zeros(WIN, np.float32)
    if bed is not None and len(bed) >= WIN:
        st = int(rng.integers(0, len(bed) - WIN + 1))
        out[:] = bed[st:st + WIN]
    if base is not None and len(base):
        if len(base) > WIN:
            base = base[:WIN]
        pos = int(rng.integers(0, max(1, WIN - len(base) + 1)))
        seg = np.zeros(WIN, np.float32)
        seg[pos:pos + len(base)] = base
        sp = np.sqrt(np.mean(seg ** 2)) + 1e-9
        npow = np.sqrt(np.mean(out ** 2)) + 1e-12
        out = out + seg * (sp / npow) * (10 ** (snr_db / 20.0))
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak * 0.35
    return out


def stream_windows(engine, wav):
    """Returns (confirmed, per-window scores) for one 1 s window fed in 100 ms chunks."""
    scores = []
    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        spotted, s, _ = engine.push_audio_chunk(wav[i:i + CHUNK])
        scores.append(s)
        if spotted:
            engine.reset()
            return True, scores
    return False, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="bcconformer_v3")
    ap.add_argument("--bases", required=True)
    ap.add_argument("--noise_index", required=True)
    ap.add_argument("--snr", type=float, nargs="+", default=[99.0, 20.0, 12.0, 6.0, 1.0, 0.5])
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=-1.0)
    ap.add_argument("--policies", type=str, nargs="+", default=["1/2", "2/2", "2/3", "3/4"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    engine = StreamingKWSEngine(checkpoint_path=args.model, arch=args.arch)
    if args.threshold >= 0:
        engine.confidence_threshold = float(args.threshold)
    thr = engine.confidence_threshold
    fx = AudioFeatureExtractor()
    print(f"model {args.model}\nthreshold {thr:.3f}   agg {engine.aggregation}   "
          f"refractory {engine.refractory_steps * 0.1:.1f}s\n", flush=True)

    bases = [np.asarray(w, np.float32) for w in np.load(args.bases, allow_pickle=True)]
    bases = [b for b in bases if 1600 < len(b) < WIN]
    nb = B.NoiseBank(index_csv=args.noise_index)
    rng = np.random.default_rng(args.seed)

    # Build the trial set once, so every policy sees identical audio.
    trials = []
    for snr in args.snr:
        for _ in range(args.trials):
            base = bases[int(rng.integers(0, len(bases)))]
            bed, _ = (None, None) if snr > 90 else nb.sample_slice(rng)
            trials.append((snr, splice(base, bed, rng, snr)))

    def fresh(need=None, window=0):
        e = StreamingKWSEngine(checkpoint_path=args.model, arch=args.arch)
        e.confidence_threshold = thr
        e.confirm_window = window
        if need is not None:
            e.consecutive_required = need
        return e

    # --- model vs detector -------------------------------------------------
    print(f"{'SNR dB':>7} {'frame recall':>13} {'confirmed':>10} {'gap':>7} "
          f"{'median score':>13} {'p10':>7}")
    for snr in args.snr:
        sel = [t for t in trials if t[0] == snr]
        eng = fresh()
        frame_hits = 0
        conf_hits = 0
        all_scores = []
        for _, wav in sel:
            eng.reset()
            hit, scores = stream_windows(eng, wav)
            conf_hits += int(hit)
            # frame recall is independent of confirmation: did ANY window clear the bar
            frame_hits += int(any(s >= thr for s in scores))
            all_scores.extend(scores)
        fr = 100.0 * frame_hits / len(sel)
        cr = 100.0 * conf_hits / len(sel)
        print(f"{snr:>7.1f} {fr:>12.1f}% {cr:>9.1f}% {fr - cr:>6.1f} "
              f"{np.median(all_scores):>13.3f} {np.percentile(all_scores, 10):>7.3f}",
              flush=True)

    # --- confirmation policies --------------------------------------------
    print(f"\n{'policy':>8} " + " ".join(f"{s:>7.1f}dB" for s in args.snr) + f" {'mean':>7}")
    for pol in args.policies:
        need, win = (int(x) for x in pol.split("/"))
        eng = fresh(need=need, window=(0 if win <= need else win))
        row = []
        for snr in args.snr:
            sel = [t for t in trials if t[0] == snr]
            hits = 0
            for _, wav in sel:
                eng.reset()
                hit, _ = stream_windows(eng, wav)
                hits += int(hit)
            row.append(100.0 * hits / len(sel))
        print(f"{pol:>8} " + " ".join(f"{v:>8.1f}%" for v in row) +
              f" {np.mean(row):>6.1f}%", flush=True)


if __name__ == "__main__":
    main()
