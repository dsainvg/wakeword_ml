"""
Held-out clean keyword waveforms for the streaming-recall benchmark
-------------------------------------------------------------------
The v1 recall figure was measured on the same TTS bases used for training, so it
never tested the deployed case: a fresh utterance of the bare wake word dropped into
a live 1 s sliding window. This script synthesises a clean, bare-"amaze" set (plus a
few realistic carriers) that the corpus builder does not use for evaluation, and
caches it for evaluate_industrial_v2.py.

    python make_eval_keyword_bases.py --out keyword_bases_eval.npy
"""

import argparse
import asyncio
import os

import numpy as np

import build_corpus_v2 as B

EVAL_RATES = ["-30%", "-22%", "-16%", "-8%", "+4%", "+10%", "+18%", "+28%", "+36%"]
EVAL_PITCHES = ["-30Hz", "-18Hz", "-6Hz", "+6Hz", "+18Hz", "+30Hz"]


async def gather(texts, voices, rates, pitches, limit_per_voice=2):
    sem = asyncio.Semaphore(12)
    out = []

    async def one(text, v, r, p):
        async with sem:
            for _ in range(2):
                try:
                    w = await B.synth_edge(text, v, rate=r, pitch=p)
                except Exception:
                    w = None
                if w is not None and len(w) > 800:
                    out.append((f"{text}|{v}|{r}|{p}", B.trim_silence(w)))
                    return
                await asyncio.sleep(0.3)
            return

    tasks = []
    for v in voices:
        combos = [(r, p) for r in rates for p in pitches]
        np.random.shuffle(combos)
        for k in range(limit_per_voice):
            r, p = combos[k]
            for t in texts:
                tasks.append(one(t, v, r, p))
    await asyncio.gather(*tasks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="keyword_bases_eval.npy")
    ap.add_argument("--per_voice", type=int, default=2)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[skip] {args.out} already exists")
        return

    bare = asyncio.run(gather(["amaze"], B.EDGE_VOICES, EVAL_RATES, EVAL_PITCHES,
                              args.per_voice))
    carrier = asyncio.run(gather(["hey amaze", "ok amaze"], B.EDGE_VOICES[::3],
                                 EVAL_RATES[:5], EVAL_PITCHES[:4], 1))
    pool = bare + carrier
    arrays = np.array([w for _, w in pool], dtype=object)
    np.save(args.out, arrays, allow_pickle=True)
    lens = [len(w) for _, w in pool]
    print(f"[done] {len(pool)} held-out keyword waveforms "
          f"(bare {len(bare)}, carrier {len(carrier)}) -> {args.out}")
    print(f"        duration median {np.median(lens)/16000:.2f}s  "
          f"min {min(lens)/16000:.2f}s  max {max(lens)/16000:.2f}s")


if __name__ == "__main__":
    main()
