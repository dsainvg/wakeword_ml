"""Realistic quiet-room test: the keyword barely above the noise floor, plus a control arm.

The headline number in this project is streaming recall averaged over 20/12/6/0 dB SNR.
That average hides the case that actually decides whether a wake word feels reliable in
a quiet room: someone two metres away, room tone and mic self-noise present, the keyword
sitting 0.5-1 dB ABOVE the bed. A model can post an 80% average and still miss most of
those, because the average is carried by the loud near-field trials.

Detection alone is not a metric here -- at 0.5 dB SNR a model that fires on everything
scores 100%. So every trial is paired with a control arm built from the SAME bed with no
keyword, and both numbers are reported: detection, and the false-alarm rate the detector
would incur on that bed if the user never spoke.

Trials run through StreamingKWSEngine in 100 ms chunks, so the peak-hold confirmation and
the refractory are in the measurement, not just the raw frame probability.
"""
import argparse

import numpy as np
import soundfile as sf

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


def run_stream(engine, wav):
    engine.reset()
    hit = False
    score = 0.0
    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        spotted, s, _ = engine.push_audio_chunk(wav[i:i + CHUNK])
        score = max(score, s)
        hit = hit or spotted
    return hit, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="bcconformer_v3")
    ap.add_argument("--bases", required=True)
    ap.add_argument("--noise_index", required=True)
    ap.add_argument("--snr", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="override the checkpoint's stored threshold; -1 = use it. "
                         "Needed for a fair comparison, because a fine-tune moves the "
                         "stored point and the two models then get measured at "
                         "different thresholds.")
    args = ap.parse_args()

    fx = AudioFeatureExtractor()
    # One engine for the whole run: constructing it per trial re-reads the checkpoint and
    # re-jits the model, which at a few hundred trials costs far more than the measurement.
    engine = StreamingKWSEngine(checkpoint_path=args.model, arch=args.arch)
    if args.threshold >= 0:
        engine.confidence_threshold = float(args.threshold)
    print(f"[quiet] model {args.model}  threshold {engine.confidence_threshold:.3f}", flush=True)
    bases = [np.asarray(w, np.float32) for w in np.load(args.bases, allow_pickle=True)]
    bases = [b for b in bases if 1600 < len(b) < WIN]
    nb = B.NoiseBank(index_csv=args.noise_index)
    rng = np.random.default_rng(args.seed)

    results = {}
    for snr in args.snr:
        hits = 0
        ctl_hits = 0
        ctl_seconds = 0.0
        scores = []
        for t in range(args.trials):
            base = bases[int(rng.integers(0, len(bases)))]
            bed, cls = nb.sample_slice(rng)

            kw = splice(base, bed, rng, snr)
            hit, sc = run_stream(engine, kw)
            hits += int(hit)
            scores.append(sc)

            # control: identical bed, no keyword, run for 10 windows of 1 s
            for _ in range(10):
                ctl = splice(None, bed, rng, snr)
                chit, _ = run_stream(engine, ctl)
                ctl_hits += int(chit)
                ctl_seconds += 1.0

        det = 100.0 * hits / args.trials
        fa = ctl_hits / max(ctl_seconds / 3600.0, 1e-9)
        results[snr] = (det, fa)
        print(f"[quiet] SNR {snr:>4.1f} dB  detection {det:5.1f}%  "
              f"control FA on the same beds {fa:6.2f}/hour  "
              f"median peak score {np.median(scores):.3f}", flush=True)

    print("-" * 78)
    dets = [v[0] for v in results.values()]
    print(f"[quiet] mean detection over {args.snr} dB: {np.mean(dets):.1f}%")
    if args.out:
        np.savez(args.out, snr=np.array(list(results.keys())),
                 detection=np.array([v[0] for v in results.values()]),
                 control_fa_per_hour=np.array([v[1] for v in results.values()]))
        print(f"[quiet] wrote {args.out}")


if __name__ == "__main__":
    main()
