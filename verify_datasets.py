"""Verify the fetched Amaze KWS datasets and (re)build noise_bank_v2/index.csv.

Run:  py verify_datasets.py
"""

import collections
import csv
import os

import numpy as np
import soundfile as sf

SPEECH = os.path.join("speech_corpus", "LibriSpeech")
RAW = os.path.join("noise_bank_v2", "raw")
INDEX = os.path.join("noise_bank_v2", "index.csv")

out = []


def emit(s=""):
    out.append(str(s))


# --- LibriSpeech -------------------------------------------------------------
flacs = [os.path.join(r, f) for r, _, fs in os.walk(SPEECH) for f in fs
         if f.endswith(".flac")]
emit("LibriSpeech flac : %d files" % len(flacs))
if flacs:
    durs, srs = [], set()
    for p in flacs[:400]:
        i = sf.info(p)
        durs.append(i.frames / i.samplerate)
        srs.add(i.samplerate)
    emit("  sample rate(s) : %s" % sorted(srs))
    emit("  duration       : %.2f h total, mean %.2f s, min %.2f, max %.2f"
         % (sum(durs) / 3600, np.mean(durs), np.min(durs), np.max(durs)))

# --- noise bank --------------------------------------------------------------
emit("")
emit("Noise bank:")
total_rows = []
for sub in sorted(os.listdir(RAW)) if os.path.isdir(RAW) else []:
    d = os.path.join(RAW, sub)
    if not os.path.isdir(d):
        continue
    wavs = sorted(os.listdir(d))
    wavs = [w for w in wavs if w.endswith(".wav")]
    if not wavs:
        continue
    secs = 0.0
    classes = set()
    for w in wavs:
        try:
            i = sf.info(os.path.join(d, w))
            secs += i.frames / i.samplerate
        except Exception:
            pass
        # The class is the part BEFORE the "__" separator, e.g.
        # "airplane__1-11687-A-47.wav" -> class "airplane". Splitting on the wrong
        # side yields one class per clip, which silently destroys the
        # class-stratified sampling the corpus builder depends on.
        if "__" in w:
            classes.add(w.split("__", 1)[0])
        elif sub == "urbansound":
            parts = w.split("_")
            classes.add(parts[2].rsplit(".", 1)[0] if len(parts) > 2 else "?")
    emit("  %-12s %5d clips  %6.2f h  %3d classes"
         % (sub, len(wavs), secs / 3600, len(classes)))
    for w in wavs:
        try:
            i = sf.info(os.path.join(d, w))
            dur = i.frames / i.samplerate
        except Exception:
            dur = 0.0
        if "__" in w:
            c = w.split("__", 1)[0]
        elif sub == "urbansound":
            parts = w.split("_")
            c = parts[2].rsplit(".", 1)[0] if len(parts) > 2 else "us8k_unknown"
        else:
            c = sub
        total_rows.append((os.path.join(sub, w), c, "%.3f" % dur))

emit("")
emit("TOTAL noise bank : %d clips, %d classes, %.2f h"
     % (len(total_rows), len({r[1] for r in total_rows}),
        sum(float(r[2]) for r in total_rows) / 3600))

# --- rebuild index.csv ------------------------------------------------------
with open(INDEX, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "class", "duration_s"])
    w.writerows(total_rows)
emit("index.csv written: %s (%d rows)" % (INDEX, len(total_rows)))

# --- sanity: decodability ----------------------------------------------------
emit("")
bad = 0
for sub in ("esc50", "urbansound"):
    d = os.path.join(RAW, sub)
    if not os.path.isdir(d):
        continue
    for w in sorted(os.listdir(d))[:60]:
        try:
            a, sr = sf.read(os.path.join(d, w))
            if a.size == 0 or sr != 16000:
                bad += 1
        except Exception as e:
            bad += 1
            emit("  UNREADABLE %s/%s: %s" % (sub, w, str(e)[:60]))
emit("decodability spot-check: %d problems in first 60 of each noise subdir" % bad)

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify.out"),
     "w", encoding="utf-8").write("\n".join(out))