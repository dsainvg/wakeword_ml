"""
Freeze a deployment operating point into a checkpoint
-----------------------------------------------------
The trainer stores the threshold that meets the frame-level FPR budget on the
validation split. That is a useful default, but the number you actually ship is the
one that meets the *streaming* false-alarm budget, which sweep_threshold.py measures
against real soundscape hours and continuous human speech.

This script re-reads a checkpoint, runs the streaming sweep, and writes the chosen
(threshold, aggregation, confirmation policy, refractory) into the checkpoint so
StreamingKWSEngine picks it up with no arguments.

    python finalize_checkpoint.py --model best_v3_50k.flax --arch bcconformer_50k \
        --data dataset_v3.npz --fa_budget 1.0
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
from flax import serialization

from evaluate_industrial_v2 import (
    StreamingDetector, load_index, load_keyword_bases, load_predict, to_16k,
)
from features import AudioFeatureExtractor
from sweep_threshold import recall_scan, soundscape_scan

SAMPLE_RATE = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="best_v3_50k.flax")
    ap.add_argument("--arch", type=str, default="bcconformer_50k")
    ap.add_argument("--index", type=str, default="noise_bank_v2/index.csv")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.72, 0.78, 0.82, 0.84, 0.86, 0.90, 0.94])
    ap.add_argument("--aggs", type=str, nargs="+", default=["maxhold"])
    ap.add_argument("--policy", type=str, default="2/2")
    ap.add_argument("--refractory", type=int, default=15)
    ap.add_argument("--speech_files", type=int, default=150)
    ap.add_argument("--recall_trials", type=int, default=300)
    ap.add_argument("--snr", type=float, nargs="+", default=[20.0, 12.0, 6.0, 0.0])
    ap.add_argument("--fa_budget", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    predict, ck = load_predict(args.model, args.arch)
    extractor = AudioFeatureExtractor()
    rows = load_index(args.index)
    bases = load_keyword_bases()
    need, span = (int(x) for x in args.policy.split("/"))

    print("scanning soundscapes ...", flush=True)
    noise = soundscape_scan(predict, extractor, rows)
    noise_hours = sum(len(to_16k(*sf.read(p))) for p, _ in rows) / SAMPLE_RATE / 3600.0

    flacs = []
    for root, _, files in os.walk("speech_corpus/LibriSpeech"):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    rng = np.random.default_rng(3)
    sel = rng.choice(len(flacs), size=min(args.speech_files, len(flacs)), replace=False)
    cache = []
    for i in sel:
        try:
            cache.append(to_16k(*sf.read(flacs[i])))
        except Exception:
            pass
    speech_hours = sum(len(d) for d in cache) / SAMPLE_RATE / 3600.0
    print(f"soundscapes {noise_hours:.2f} h | speech {speech_hours:.3f} h", flush=True)

    print(f"\n{'agg':>8} | {'thresh':>7} | {'soundscFA/h':>11} | {'speechFA/h':>10} | "
          f"{'totalFA/h':>9} | {'recall':>7}")
    print("-" * 68)
    best = None
    for agg in args.aggs:
        for t in args.thresholds:
            kw = dict(need=need, span=span, refractory_steps=args.refractory, agg=agg)
            det = StreamingDetector(predict, extractor, t, **kw)
            hits = 0
            for a in cache:
                dets, _, _ = det.run(a)
                hits += len(dets)
            speech_fa = hits / max(1e-9, speech_hours)
            noise_fa = sum(int(np.sum(p >= t)) for p in noise.values()) / max(1e-9, noise_hours)
            total = noise_fa + speech_fa
            rh, rc, _ = recall_scan(predict, extractor, t, bases, rows,
                                    args.recall_trials, args.snr, **kw)
            recall = sum(rh.values()) / max(1, sum(rc.values())) * 100
            ok = " *" if total <= args.fa_budget else ""
            print(f"{agg:>8} | {t:7.4f} | {noise_fa:11.2f} | {speech_fa:10.2f} | {total:9.2f} | "
                  f"{recall:6.1f}%{ok}", flush=True)
            if total <= args.fa_budget and (best is None or recall > best[2]):
                best = (agg, t, recall, total)
    print("-" * 68)
    if not best:
        print("No operating point met the FA budget; nothing written.")
        return
    agg, t, recall, total = best
    print(f" CHOSEN: threshold {t:.4f}, agg={agg}, policy {args.policy}, "
          f"refractory {args.refractory*0.1:.1f}s")
    print(f"         measured streaming recall {recall:.1f}%, total FA {total:.2f}/hour")
    if args.dry_run:
        return

    out = dict(ck)
    out["threshold"] = float(t)
    out["deploy_threshold"] = float(t)
    out["deploy_agg"] = agg
    out["deploy_need"] = need
    out["deploy_span"] = span
    # The span has to reach the deployed engine, otherwise the checkpoint says "2 of 3"
    # while kws_engine.py still requires two strictly consecutive windows.
    out["deploy_confirm_window"] = int(span) if span > need else 0
    out["deploy_refractory_steps"] = args.refractory
    out["deploy_recall"] = float(recall)
    out["deploy_fa_per_hour"] = float(total)
    with open(args.model, "wb") as f:
        f.write(serialization.to_bytes(out))
    print(f" Written into {args.model} ({os.path.getsize(args.model)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
