"""
Industrial Evaluation v2
------------------------
End-to-end benchmark that scores the thing that actually ships: a 1 s sliding
window detector, not a single-frame classifier.

Sections
  1. Validation ROC + TPR at several FPR budgets (with the checkpoint's stored
     threshold highlighted).
  2. Offset stress: TPR as a function of where the keyword sits inside the 1 s
     window. v1 fell from 83% to 70% across this sweep; anything close to flat is
     what a 100 ms-stride streamer needs.
  3. Soundscape stress over noise_bank_v2: per-class max confidence, false alarms
     per hour of audio, and the worst offenders by name.
  4. Continuous-speech false alarm rate with the industrial debouncer
     (EMA + 2 consecutive frames + 1.5 s refractory).
  5. Streaming recall: keyword spliced into real noise+speech beds at random
     offsets and random SNR, then detected through the sliding-window engine.
     Reports hit rate and detection latency.

    python evaluate_industrial_v2.py --model best_v2_50k.flax --arch bcconformer_50k
"""

import argparse
import csv
import glob
import os
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import jax
import jax.numpy as jnp
from flax import serialization
from scipy import signal

from features import AudioFeatureExtractor
from models import get_model

SAMPLE_RATE = 16000
WIN = 16000
STRIDE = 1600  # 100 ms


# ==============================================================================
# Streaming detector (mirrors kws_engine.py / evaluate_industrial.py)
# ==============================================================================

class StreamingDetector:
    """1 s sliding-window detector with a configurable confirmation policy.

    `need` hits inside any `span` consecutive windows is the general form of the
    classic "2 consecutive frames" rule. A 0.6 s wake word only fully fits one or two
    1 s windows, so requiring 2 *consecutive* frames is a much harsher test than
    requiring 2 hits inside 3 windows, while still rejecting isolated single spikes.
    """

    def __init__(self, predict_fn, extractor, threshold, ema_alpha=0.6,
                 need=2, span=2, refractory_steps=15, rms_gate=0.008, agg="ema",
                 hold_decay=0.85):
        self.predict_fn = predict_fn
        self.extractor = extractor
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.need = need
        self.span = span
        self.refractory = refractory_steps
        self.rms_gate = rms_gate
        self.agg = agg
        self.hold_decay = hold_decay
        self.reset()

    def reset(self):
        self.ema = 0.0
        self.hold = 0.0
        self.hits = []
        self.lock = 0
        self.raw_peak = 0.0

    def _score(self, raw: float) -> float:
        """Aggregate the raw frame probability into a confirmation score.

        'ema'     -- exponential moving average. Penalises bursts: two high frames
                     in a row only reach ~0.71 of the way to the raw peak at alpha=0.6.
        'maxhold' -- decaying peak hold. A two-frame burst keeps its full value for
                     `span` windows, so a 0.6 s keyword still confirms, while an
                     isolated single-frame spike still does not.
        """
        if self.agg == "maxhold":
            self.hold = max(raw, self.hold * self.hold_decay)
            return self.hold
        self.ema = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self.ema
        return self.ema

    def _decay(self):
        if self.agg == "maxhold":
            self.hold *= self.hold_decay
        else:
            self.ema = (1.0 - self.ema_alpha) * self.ema

    def run(self, audio, collect_raw=True):
        """Streams `audio`, returns (detections, first_detect_step, raw_triggers)."""
        self.reset()
        dets = []
        first = None
        raw_hits = 0
        n_steps = max(0, (len(audio) - WIN) // STRIDE + 1)
        for s in range(n_steps):
            if self.lock > 0:
                self.lock -= 1
            w = audio[s * STRIDE:s * STRIDE + WIN]
            rms = float(np.sqrt(np.mean(w ** 2)))
            if rms < self.rms_gate:
                self._decay()
                self.hits.append(0)
                if len(self.hits) > self.span:
                    self.hits.pop(0)
                continue
            spec = self.extractor.compute_spectrogram(w)
            p = float(self.predict_fn(jnp.expand_dims(jnp.expand_dims(jnp.array(spec), 0), -1))[0])
            if p > self.raw_peak:
                self.raw_peak = p
            if collect_raw and p >= self.threshold:
                raw_hits += 1
            score = self._score(p)
            self.hits.append(1 if score >= self.threshold else 0)
            if len(self.hits) > self.span:
                self.hits.pop(0)
            if sum(self.hits) >= self.need and self.lock == 0:
                dets.append(s * STRIDE)
                if first is None:
                    first = s * STRIDE
                self.lock = self.refractory
                self.hits = [0] * len(self.hits)
                self.ema = 0.0
                self.hold = 0.0
        return dets, first, raw_hits


# ==============================================================================
# Loading helpers
# ==============================================================================

def load_predict(checkpoint_path, arch):
    with open(checkpoint_path, "rb") as f:
        ck = serialization.msgpack_restore(f.read())
    model = get_model(arch, num_classes=2)
    # msgpack_restore hands back numpy leaves. jitting with them is fine for most ops but
    # makes any numpy-style indexing inside the model convert a tracer, so promote them.
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])

    @jax.jit
    def predict(x):
        logits = model.apply({"params": params}, x, train=False)
        return jax.nn.softmax(logits, axis=-1)[:, 1]

    return predict, ck


def to_16k(d, sr):
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SAMPLE_RATE:
        d = signal.resample(d, int(round(len(d) * SAMPLE_RATE / sr)))
    return d.astype(np.float32)


def batch_score(predict, specs, bs=512):
    out = []
    for i in range(0, len(specs), bs):
        chunk = np.asarray(specs[i:i + bs], dtype=np.float32)
        if chunk.ndim == 3:
            chunk = np.expand_dims(chunk, -1)
        out.extend(np.array(predict(jnp.asarray(chunk))))
    return np.array(out)


def load_index(index_csv="noise_bank_v2/index.csv"):
    rows = []
    if os.path.exists(index_csv):
        with open(index_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["path"].replace("/", os.sep), r["class"]))
    if not rows:
        for p in sorted(glob.glob(os.path.join("noise_bank", "*.wav"))):
            rows.append((p, os.path.splitext(os.path.basename(p))[0]))
    return rows


# ==============================================================================
# Sections
# ==============================================================================

def sec1_roc(predict, data_path):
    print("\n" + "=" * 86)
    print(" 1. VALIDATION ROC AND OPERATING POINTS")
    print("=" * 86)
    d = np.load(data_path, allow_pickle=True)
    X_val, y_val = d["X_val"], d["y_val"]
    p = batch_score(predict, X_val[:, :, :, 0])
    pos, neg = p[y_val == 1], p[y_val == 0]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, p)
    print(f" val: {len(pos)} pos / {len(neg)} neg   AUC = {auc*100:.3f}%")
    print(f" positives : 1%={np.percentile(pos,1):.3f}  5%={np.percentile(pos,5):.3f}  "
          f"10%={np.percentile(pos,10):.3f}  median={np.median(pos):.3f}")
    print(f" negatives : 99%={np.percentile(neg,99):.3f}  99.9%={np.percentile(neg,99.9):.3f}  "
          f"max={neg.max():.3f}")
    print("-" * 86)
    print(f"{'FPR budget':>11} | {'threshold':>9} | {'TPR':>7} | {'actual FPR':>10} | {'FA / 1000 neg':>14}")
    for b in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        allowed = int(np.floor(b * len(neg)))
        t = float(np.sort(neg)[::-1][allowed - 1]) if allowed > 0 else 1.0
        tpr = float(np.mean(pos >= t))
        print(f"{b*100:10.2f}% | {t:9.4f} | {tpr*100:6.2f}% | {float(np.mean(neg>=t))*100:9.3f}% | "
              f"{int(np.sum(neg>=t)):14d}")
    return pos, neg, auc


def sec1b_buckets(predict, data_path, threshold):
    print("\n" + "=" * 96)
    print(" 1b. PER-BUCKET BREAKDOWN AT THE OPERATING POINT")
    print("=" * 96)
    d = np.load(data_path, allow_pickle=True)
    if "b_val" not in d.files:
        print(" dataset has no bucket tags; skipping")
        return
    X_val, y_val = d["X_val"], d["y_val"]
    b_val = d["b_val"].astype(str)
    p = batch_score(predict, X_val[:, :, :, 0])
    order = sorted(np.unique(b_val), key=lambda b: (b[0] != "P", b))
    print(f" threshold {threshold:.4f}    val {len(p):,} samples")
    print(" Two tables on purpose: in a single column a 3% figure means failure for a")
    print(" positive bucket and success for a negative one, so the number is unreadable")
    print(" without knowing its polarity.")

    posb = [b for b in order if b.startswith("P")]
    negb = [b for b in order if b.startswith("N")]

    print(f"\n {'bucket':<28} | {'n':>6} | {'recall':>7} | {'MISSED':>7} | "
          f"{'mean conf':>9} | {'p05 conf':>8}   (higher recall is better)")
    print("-" * 96)
    for b in posb:
        m = b_val == b
        r = float(np.mean(p[m] >= threshold))
        print(f" {b:<28} | {int(m.sum()):6d} | {r*100:6.2f}% | {(1-r)*100:6.2f}% | "
              f"{p[m].mean():9.4f} | {np.percentile(p[m], 5):8.4f}")

    print(f"\n {'bucket':<28} | {'n':>6} | {'FA rate':>8} | {'FA count':>8} | "
          f"{'mean conf':>9} | {'p95 conf':>8}   (lower is better)")
    print("-" * 96)
    for b in negb:
        m = b_val == b
        r = float(np.mean(p[m] >= threshold))
        print(f" {b:<28} | {int(m.sum()):6d} | {r*100:7.2f}% | {int(np.sum(p[m] >= threshold)):8d} | "
              f"{p[m].mean():9.4f} | {np.percentile(p[m], 95):8.4f}")
    print("-" * 96)
    for prefix, name in (("P0", "direct"), ("P1", "+1 noise"),
                         ("P2", "+2 noise"), ("P3", "+2 noise hard")):
        sel = [b for b in posb if b.startswith(prefix)]
        if not sel:
            continue
        m = np.isin(b_val, sel)
        print(f" positive recipe {name:<14} pooled recall: {float(np.mean(p[m] >= threshold))*100:6.2f}%"
              f"  (n={int(m.sum()):,})")
    for grp, gname in (("N0", "filler direct"), ("N1", "filler +1 noise"),
                       ("N2", "filler +2 noise"), ("N3", "filler +2 noise hard")):
        sel = [b for b in negb if b.startswith(grp)]
        if not sel:
            continue
        m = np.isin(b_val, sel)
        print(f" filler recipe   {gname:<14} pooled FA  : {float(np.mean(p[m] >= threshold))*100:6.3f}%"
              f"  (n={int(m.sum()):,})")
    for grp, gname in (("N12", "noise alone"), ("N13", "noise +1 layer"),
                       ("N14", "noise +2 layers"), ("N15", "noise +2 hard")):
        sel = [b for b in negb if b.startswith(grp)]
        if not sel:
            continue
        m = np.isin(b_val, sel)
        print(f" noise  recipe   {gname:<14} pooled FA  : {float(np.mean(p[m] >= threshold))*100:6.3f}%"
              f"  (n={int(m.sum()):,})")


def sec1c_level(predict, data_path, threshold):
    print("\n" + "=" * 96)
    print(" 1c. LEVEL ROBUSTNESS (input gain / amplitude normalisation)")
    print("=" * 96)
    d = np.load(data_path, allow_pickle=True)
    X_val, y_val = d["X_val"], d["y_val"]
    pos = X_val[y_val == 1][:, :, :, 0]
    neg = X_val[y_val == 0][:, :, :, 0]
    if pos.shape[0] > 4000:
        pos = pos[:4000]
    if neg.shape[0] > 4000:
        neg = neg[:4000]
    k = 12.0 / 20.0 * np.log(10.0)  # dB amplitude -> log-mel offset
    print(f"{'gain':>8} | {'dB':>7} | {'pos recall':>10} | {'pos mean':>9} | {'neg FA':>8}")
    print("-" * 96)
    for g_db in (-24, -18, -12, -6, 0, 6, 12, 18):
        off = g_db / 20.0 * np.log(10.0)
        pp = batch_score(predict, np.clip(pos + off, -11.5129, 5.0))
        pn = batch_score(predict, np.clip(neg + off, -11.5129, 5.0))
        print(f"{10**(g_db/20.0):8.3f} | {g_db:+7.1f} | {float(np.mean(pp>=threshold))*100:9.2f}% | "
              f"{pp.mean():9.4f} | {float(np.mean(pn>=threshold))*100:7.3f}%")


def sec2_offset(predict, data_path, extractor):
    print("\n" + "=" * 86)
    print(" 2. OFFSET STRESS (keyword position inside the 1 s window)")
    print("=" * 86)
    d = np.load(data_path, allow_pickle=True)
    X_val, y_val = d["X_val"], d["y_val"]
    pos_specs = X_val[y_val == 1][:, :, :, 0]
    neg_probs = batch_score(predict, X_val[y_val == 0][:, :, :, 0])
    allowed = int(np.floor(0.005 * len(neg_probs)))
    thr = float(np.sort(neg_probs)[::-1][allowed - 1]) if allowed > 0 else 1.0
    print(f" threshold from 0.5% FPR budget: {thr:.4f}")
    print(f"{'roll (frames / ms)':>18} | {'TPR':>7} | {'mean conf':>10}")
    tprs = []
    for off in range(0, 25, 2):
        rolled = np.stack([np.roll(s, off, axis=0) for s in pos_specs])
        p = batch_score(predict, rolled)
        tpr = float(np.mean(p >= thr))
        tprs.append(tpr)
        print(f"{off:8d} / {off*20:4d} ms | {tpr*100:6.2f}% | {p.mean():10.4f}")
    print("-" * 86)
    print(f" TPR spread across offsets: min {min(tprs)*100:.2f}%  max {max(tprs)*100:.2f}%  "
          f"drop {100*(max(tprs)-min(tprs)):.2f} pts")


def sec3_soundscapes(predict, extractor, threshold, index_csv):
    print("\n" + "=" * 86)
    print(" 3. SOUNDSCAPE STRESS (noise_bank_v2)")
    print("=" * 86)
    rows = load_index(index_csv)
    by_class = defaultdict(list)
    for p, c in rows:
        by_class[c].append(p)
    total_sec = 0.0
    per_class = {}
    raw_fa = 0
    for cls, paths in sorted(by_class.items()):
        peaks = []
        secs = 0.0
        for p in paths:
            try:
                d = to_16k(*sf.read(p))
            except Exception:
                continue
            secs += len(d) / SAMPLE_RATE
            n = max(0, (len(d) - WIN) // 1600 + 1)
            for s in range(0, n, 10):  # every 1 s
                chunk = d[s * 1600:s * 1600 + WIN]
                if len(chunk) < WIN:
                    chunk = np.pad(chunk, (0, WIN - len(chunk)))
                if np.sqrt(np.mean(chunk ** 2)) < 0.002:
                    continue
                spec = extractor.compute_spectrogram(chunk)
                peaks.append(float(predict(jnp.expand_dims(jnp.expand_dims(jnp.array(spec), 0), -1))[0]))
        if not peaks:
            continue
        peaks = np.array(peaks)
        fa = int(np.sum(peaks >= threshold))
        raw_fa += fa
        total_sec += secs
        per_class[cls] = (peaks.max(), fa, secs, len(peaks))
    worst = sorted(per_class.items(), key=lambda kv: -kv[1][0])[:14]
    print(f"{'class':<28} | {'max conf':>8} | {'frames>=thr':>11} | {'hours':>6}")
    for cls, (mx, fa, secs, n) in worst:
        print(f"{cls:<28} | {mx:8.4f} | {fa:6d}/{n:<5d} | {secs/3600:6.2f}")
    hours = total_sec / 3600.0
    print("-" * 86)
    print(f" classes tested: {len(per_class)}   audio: {hours:.2f} h   "
          f"raw false alarms: {raw_fa}  ({raw_fa/max(hours,1e-9):.2f} FA/hour, 1 s stride)")


def sec4_speech_fa(predict, extractor, threshold, num_files=120, agg="maxhold"):
    print("\n" + "=" * 86)
    print(" 4. CONTINUOUS HUMAN SPEECH FALSE ALARMS (industrial debouncer)")
    print("=" * 86)
    flacs = []
    for root, _, files in os.walk("speech_corpus/LibriSpeech"):
        for f in files:
            if f.endswith(".flac"):
                flacs.append(os.path.join(root, f))
    if not flacs:
        print(" [warn] no LibriSpeech found")
        return
    rng = np.random.default_rng(0)
    sel = rng.choice(len(flacs), size=min(num_files, len(flacs)), replace=False)
    det = StreamingDetector(predict, extractor, threshold, agg=agg)
    total = 0.0
    hits = 0
    raw = 0
    t0 = time.time()
    for i in sel:
        try:
            d = to_16k(*sf.read(flacs[i]))
        except Exception:
            continue
        total += len(d) / SAMPLE_RATE
        dets, _, r = det.run(d)
        hits += len(dets)
        raw += r
    hrs = total / 3600.0
    el = time.time() - t0
    print(f" audio            : {total:.1f}s ({hrs:.3f} h) from {len(sel)} recordings")
    print(f" raw frame FAs    : {raw}  ({raw/max(hrs,1e-9):.2f} / hour)")
    print(f" debounced triggers: {hits}  ({hits/max(hrs,1e-9):.2f} / hour)")
    print(f" speed            : {el:.1f}s  ({total/max(el,1e-9):.1f}x real time)")


def sec5_streaming_recall(predict, extractor, threshold, keyword_bases, n_trials=300, agg="maxhold"):
    print("\n" + "=" * 86)
    print(f" 5. STREAMING RECALL (keyword spliced into real noise + speech beds, agg={agg})")
    print("=" * 86)
    if not keyword_bases:
        print(" [warn] no keyword bases available; skipping")
        return
    noise_rows = load_index()
    noise_paths = [p for p, _ in noise_rows]
    rng = np.random.default_rng(11)
    det = StreamingDetector(predict, extractor, threshold, agg=agg)

    snr_points = [20.0, 12.0, 6.0, 0.0, -3.0]
    results = {s: [0, 0, []] for s in snr_points}
    for _ in range(n_trials):
        try:
            nz_path = noise_paths[int(rng.integers(0, len(noise_paths)))]
            nz = to_16k(*sf.read(nz_path))
        except Exception:
            continue
        kw = keyword_bases[int(rng.integers(0, len(keyword_bases)))]
        kw = np.asarray(kw, dtype=np.float32).ravel()
        # Keep the whole utterance: truncating the keyword destroys the phonetic
        # pattern the detector is supposed to key on.
        if len(kw) > 11000:
            kw = kw[:11000]
        if len(kw) < 1500:
            continue
        bed_n = 16000 + 7000 + int(rng.integers(0, 3000))
        reps = int(np.ceil(bed_n / max(1, len(nz))))
        bed = np.tile(nz, reps)[:bed_n]
        start = int(rng.integers(1200, 3200))
        snr = float(snr_points[int(rng.integers(0, len(snr_points)))])
        # Scale the NOISE to the requested SNR against the keyword. Scaling the whole
        # mixture would be a no-op: SNR is scale invariant.
        sp = np.mean(kw ** 2) + 1e-9
        npow = np.mean(bed ** 2) + 1e-9
        bed = (bed * np.sqrt(sp / (10 ** (snr / 10.0) * npow))).astype(np.float32)
        bed[start:start + len(kw)] += kw
        peak = float(np.max(np.abs(bed)))
        if peak > 0.95:
            bed = (bed / peak * 0.95).astype(np.float32)
        dets, first, _ = det.run(bed)
        key = snr_points[int(np.argmin([abs(s - snr) for s in snr_points]))]
        res = results[key]
        res[1] += 1
        if first is not None and first <= start + len(kw) + 3200:
            res[0] += 1
            # Latency measured from the end of the spoken keyword to the trigger.
            res[2].append((first - (start + len(kw))) / SAMPLE_RATE)
    print(f"{'SNR (dB)':>9} | {'detected':>9} | {'recall':>7} | {'median lat':>11} | {'p90 lat':>8}")
    tot_hit = 0
    tot_n = 0
    for s in snr_points:
        hit, n, lats = results[s]
        if n == 0:
            continue
        tot_hit += hit
        tot_n += n
        med = np.median(lats) * 1000 if lats else float("nan")
        p90 = np.percentile(lats, 90) * 1000 if lats else float("nan")
        print(f"{s:9.1f} | {hit:4d}/{n:<4d} | {hit/n*100:6.2f}% | {med:9.0f}ms | {p90:6.0f}ms")
    print("-" * 86)
    if tot_n:
        print(f" overall streaming recall: {tot_hit}/{tot_n} = {tot_hit/tot_n*100:.2f}%")


def load_keyword_bases(path="keyword_bases_eval.npy", limit=400):
    """Held-out clean keyword waveforms for the streaming-recall test."""
    for cand in (path, "keyword_bases_eval.npy", "keyword_bases.npy"):
        if os.path.exists(cand):
            arr = np.load(cand, allow_pickle=True)
            bases = [np.asarray(x, dtype=np.float32).ravel() for x in list(arr)]
            bases = [b for b in bases if 1500 <= len(b) <= 13000]
            print(f"[eval] keyword bases: {cand} ({len(bases)} usable)")
            return bases[:limit]
    print("[eval] no keyword base cache found; streaming recall will be skipped")
    return []


# ==============================================================================
# main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="best_v2_50k.flax")
    ap.add_argument("--arch", type=str, default="bcconformer_50k")
    ap.add_argument("--data", type=str, default="dataset_v2.npz")
    ap.add_argument("--index", type=str, default="noise_bank_v2/index.csv")
    ap.add_argument("--threshold", type=float, default=-1.0, help="-1 = use value stored in checkpoint")
    ap.add_argument("--agg", type=str, default="maxhold", choices=["ema", "maxhold"])
    ap.add_argument("--speech_files", type=int, default=120)
    ap.add_argument("--recall_trials", type=int, default=300)
    args = ap.parse_args()

    predict, ck = load_predict(args.model, args.arch)
    extractor = AudioFeatureExtractor()
    thr = args.threshold if args.threshold >= 0 else float(ck.get("threshold", 0.85))

    print("#" * 86)
    print(f" INDUSTRIAL BENCHMARK v2  |  keyword = 'amaze'")
    print(f" model      : {args.model}  [{args.arch}]  {ck.get('param_count', 0):,} params")
    print(f" checkpoint : epoch {ck.get('epoch','?')}  stored threshold {thr:.4f}  "
          f"confirmation: {args.agg} 2-of-2 + 1.5 s refractory")
    if "tpr_at_budget" in ck:
        print(f" stored     : TPR@budget {ck.get('tpr_at_budget',0)*100:.2f}%  "
              f"offset-stress {ck.get('tpr_offset_stress',0)*100:.2f}%  "
              f"TPR@FPR0.12% {ck.get('tpr_at_0012',0)*100:.2f}%")
    print("#" * 86)

    sec1_roc(predict, args.data)
    sec1b_buckets(predict, args.data, thr)
    sec1c_level(predict, args.data, thr)
    sec2_offset(predict, args.data, extractor)
    sec3_soundscapes(predict, extractor, thr, args.index)
    sec4_speech_fa(predict, extractor, thr, num_files=args.speech_files, agg=args.agg)
    sec5_streaming_recall(predict, extractor, thr, load_keyword_bases(),
                          n_trials=args.recall_trials, agg=args.agg)


if __name__ == "__main__":
    main()
