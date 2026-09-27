"""
Operating-Point Sweep for the Streaming Detector
------------------------------------------------
Picks the deployment threshold from measured behaviour rather than a frame-level FPR
proxy. For each candidate threshold it reports

  * soundscape false alarms per hour over noise_bank_v2
  * debounced continuous-speech false alarms per hour over LibriSpeech
  * streaming recall with a held-out keyword spliced into real noise beds

The right threshold is the lowest one that keeps total streaming false alarms under
the budget, because every point of recall is worth more than a threshold that is
tighter than the hardware ever required.

    python sweep_threshold.py --model best_v2_50k.flax --arch bcconformer_50k \
        --thresholds 0.60 0.70 0.75 0.80 0.8246 0.85
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

from evaluate_industrial_v2 import (
    StreamingDetector, batch_score, load_index, load_keyword_bases, load_predict, to_16k,
)
from features import AudioFeatureExtractor

SAMPLE_RATE = 16000
WIN = 16000
HOP = 800  # 0.5 s stride for the FA scan (denser than deployment, so FA/hr is an upper bound)


def soundscape_scan(predict, extractor, rows, stride_s=1.0):
    """Pre-computes every soundscape frame confidence once; thresholds are applied later."""
    per_class = {}
    step = int(stride_s * SAMPLE_RATE)
    for path, cls in rows:
        try:
            d = to_16k(*sf.read(path))
        except Exception:
            continue
        n = max(0, (len(d) - WIN) // step + 1)
        if n == 0:
            continue
        arr = np.empty((n, 49, 40), dtype=np.float32)
        keep = np.zeros(n, dtype=bool)
        for s in range(n):
            chunk = d[s * step:s * step + WIN]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.002:
                continue
            arr[s] = extractor.compute_spectrogram(chunk)
            keep[s] = True
        if not keep.any():
            continue
        probs = batch_score(predict, arr[keep][:, :, :])
        per_class.setdefault(cls, []).append(probs)
    merged = {c: np.concatenate(v) for c, v in per_class.items()}
    return merged


def speech_stream_scan(predict, extractor, threshold, num_files, **det_kw):
    det = StreamingDetector(predict, extractor, threshold, **det_kw)
    flacs = []
    for root, _, files in os.walk("speech_corpus/LibriSpeech"):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(flacs), size=min(num_files, len(flacs)), replace=False)
    total = 0.0
    hits = 0
    for i in sel:
        try:
            d = to_16k(*sf.read(flacs[i]))
        except Exception:
            continue
        total += len(d) / SAMPLE_RATE
        dets, _, _ = det.run(d)
        hits += len(dets)
    return total / 3600.0, hits


def recall_scan(predict, extractor, threshold, bases, rows, trials, snr_points, seed=5, **det_kw):
    rng = np.random.default_rng(seed)
    paths = [p for p, _ in rows]
    det = StreamingDetector(predict, extractor, threshold, **det_kw)
    hits = defaultdict(int)
    counts = defaultdict(int)
    lats = defaultdict(list)
    for _ in range(trials):
        try:
            nz = to_16k(*sf.read(paths[int(rng.integers(0, len(paths)))]))
        except Exception:
            continue
        kw = np.asarray(bases[int(rng.integers(0, len(bases)))], dtype=np.float32).ravel()
        if len(kw) > 11000:
            kw = kw[:11000]
        if len(kw) < 1500:
            continue
        bed_n = WIN + 7000 + int(rng.integers(0, 3000))
        bed = np.tile(nz, int(np.ceil(bed_n / max(1, len(nz)))))[:bed_n]
        start = int(rng.integers(1200, 3200))
        snr = float(snr_points[int(rng.integers(0, len(snr_points)))])
        sp = np.mean(kw ** 2) + 1e-9
        npow = np.mean(bed ** 2) + 1e-9
        bed = (bed * np.sqrt(sp / (10 ** (snr / 10.0) * npow))).astype(np.float32)
        bed[start:start + len(kw)] += kw
        pk = float(np.max(np.abs(bed)))
        if pk > 0.95:
            bed = (bed / pk * 0.95).astype(np.float32)
        dets, first, _ = det.run(bed)
        counts[snr] += 1
        if first is not None and first <= start + len(kw) + 3200:
            hits[snr] += 1
            lats[snr].append((first - start) / SAMPLE_RATE)
    return hits, counts, lats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="best_v3_50k.flax")
    ap.add_argument("--arch", type=str, default="bcconformer_50k")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.78, 0.84, 0.90])
    ap.add_argument("--policies", type=str, nargs="+", default=["2/2"],
                    help="need/span confirmation policies, e.g. 2/2 2/3 3/4 1/2")
    ap.add_argument("--aggs", type=str, nargs="+", default=["ema", "maxhold"],
                    help="frame-score aggregation: ema or maxhold")
    ap.add_argument("--refractory", type=int, default=15)
    ap.add_argument("--index", type=str, default="noise_bank_v2/index.csv")
    ap.add_argument("--speech_files", type=int, default=80)
    ap.add_argument("--recall_trials", type=int, default=250)
    ap.add_argument("--snr", type=float, nargs="+", default=[20.0, 12.0, 6.0, 0.0])
    ap.add_argument("--fa_budget", type=float, default=1.0, help="max total FA per hour")
    args = ap.parse_args()

    predict, ck = load_predict(args.model, args.arch)
    extractor = AudioFeatureExtractor()
    rows = load_index(args.index)
    bases = load_keyword_bases()

    print("=" * 108)
    print(f" THRESHOLD x POLICY SWEEP  |  {args.model}  [{args.arch}]  "
          f"| refractory {args.refractory*0.1:.1f}s  | FA budget {args.fa_budget}/hour")
    print("=" * 108)
    print(" scanning soundscapes once ...", flush=True)
    t0 = time.time()
    noise = soundscape_scan(predict, extractor, rows)
    noise_secs = 0.0
    for path, _ in rows:
        try:
            noise_secs += len(to_16k(*sf.read(path))) / SAMPLE_RATE
        except Exception:
            pass
    noise_hours = noise_secs / 3600.0
    print(f" {len(noise)} classes, {noise_hours:.2f} h scored in {time.time()-t0:.0f}s", flush=True)

    print(" scanning continuous speech ...", flush=True)
    speech_cache = []
    flacs = []
    for root, _, files in os.walk("speech_corpus/LibriSpeech"):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(flacs), size=min(args.speech_files, len(flacs)), replace=False)
    for i in sel:
        try:
            speech_cache.append(to_16k(*sf.read(flacs[i])))
        except Exception:
            pass
    speech_hours = sum(len(d) for d in speech_cache) / SAMPLE_RATE / 3600.0

    print("=" * 108)
    print(f"{'agg policy':>13} | {'thresh':>7} | {'soundscFA/h':>11} | {'speechFA/h':>9} | "
          f"{'totFA/h':>8} | {'rec@20':>7} | {'rec@12':>7} | {'rec@6':>7} | {'rec@0':>7} | {'mean':>7}")
    print("-" * 108)
    best = None
    results = []
    for agg in args.aggs:
      for pol in args.policies:
        need, span = (int(x) for x in pol.split("/"))
        for t in args.thresholds:
            det_kw = dict(need=need, span=span, refractory_steps=args.refractory, agg=agg)
            det = StreamingDetector(predict, extractor, t, **det_kw)
            hits = 0
            for d in speech_cache:
                dets, _, _ = det.run(d)
                hits += len(dets)
            speech_fa = hits / max(1e-9, speech_hours)
            noise_fa = sum(int(np.sum(p >= t)) for p in noise.values()) / max(1e-9, noise_hours)
            total_fa = noise_fa + speech_fa
            rh, rc, _ = recall_scan(predict, extractor, t, bases, rows,
                                    args.recall_trials, args.snr, **det_kw)
            def r_of(s):
                return rh.get(s, 0) / max(1, rc.get(s, 0)) * 100
            mean_r = sum(rh.get(s, 0) for s in args.snr) / max(1, sum(rc.values())) * 100
            flag = "" if total_fa > args.fa_budget else "  *"
            print(f"{agg+' '+pol:>13} | {t:7.4f} | {noise_fa:11.2f} | {speech_fa:9.2f} | {total_fa:8.2f} | "
                  f"{r_of(20.0):6.1f}% | {r_of(12.0):6.1f}% | {r_of(6.0):6.1f}% | "
                  f"{r_of(0.0):6.1f}% | {mean_r:6.1f}%{flag}", flush=True)
            results.append((agg, pol, t, total_fa, mean_r))
            if total_fa <= args.fa_budget:
                if best is None or mean_r > best[4]:
                    best = (agg, pol, t, total_fa, mean_r)
    print("-" * 108)
    if best:
        print(f" RECOMMENDED: threshold {best[2]:.4f}, agg={best[0]}, policy {best[1]} "
              f"(streaming recall {best[4]:.1f}%, total FA {best[3]:.2f}/hour)")
    else:
        print(" No (threshold, policy) met the FA budget.")
    print("=" * 108)


if __name__ == "__main__":
    main()
