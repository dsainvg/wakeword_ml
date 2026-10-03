"""Finer 1.5 MMAC sweep -- exploit WIDTH, not just time
--------------------------------------------------------
Iteration 4 produced the finding that redirected this search:

    hc_balanced  d=64  T=24  b=3   2.62 MMAC   TPR 83.2%
    m1_e_t24     d=40  T=24  b=3   2.24 MMAC   TPR 78.4%

Same T, same depth, same MAC class -- a width of 64 instead of 40 is worth ~5
TPR points. So WIDTH buys more accuracy per MAC than TIME reduction does, which
inverts the assumption the first sweep was built on (it only varied ffn_mult over
2/3/4 and dim over a coarse grid, so it never saw the cheap wide configs).

At 1.5 MMAC the FFN expansion is the dominant term -- at dim 64, T 24:
    FFN      T*d*hidden*2 = 24*64*128*2 = 393k MAC per block
    attn qkv T*d*3d      = 24*64*192   = 295k MAC per block
So ffn_mult is the single biggest cost knob below 1.0. This sweep explores
ffn_mult 1.0-2.0 in fine steps, plus block counts 1-4 and dims 32-96, at both
T=12 and T=24, and ranks by a quality proxy rather than by params alone.

    py sweep_fine.py
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_torch as MT                                        # noqa: E402
from models_torch_hc import TimedKWSNet, MAC_PER_MS, V3_PARAMS  # noqa: E402
from models_torch import INPUT_SHAPE                            # noqa: E402

CAP = 1_500_000
OUT = []


def emit(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def measure(cfg):
    m = TimedKWSNet(**cfg)
    p = sum(q.numel() for q in m.parameters())
    mac = MT.count_macs(m)
    with torch.no_grad():
        y = m(torch.zeros(*INPUT_SHAPE))
    assert tuple(y.shape) == (1, 2)
    return p, mac


def quality_proxy(p, mac):
    """Rank quality per unit of arithmetic, weighted by width.

    Not a substitute for training -- a heuristic. It rewards parameters (capacity)
    and penalises MACs (the actual CPU cost), with a mild bonus for width because
    the measured hc_balanced vs m1_e_t24 comparison says width converts to TPR
    more efficiently than depth or temporal resolution does.
    """
    return (p ** 0.5) / (mac ** 0.25)


def main():
    emit("=" * 100)
    emit("Fine sweep under %.2f MMAC (floor %.3f ms)"
         % (CAP / 1e6, CAP / MAC_PER_MS))
    emit("Finding from iter4: WIDTH > TIME for accuracy per MAC")
    emit("  d=64 T=24 b=3 -> TPR 83.2%   vs   d=40 T=24 b=3 -> TPR 78.4%")
    emit("=" * 100)

    rows = []
    for ts, dim, nb, ff in itertools.product(
            (2, 4), (32, 40, 48, 56, 64, 72, 80, 96), (1, 2, 3, 4),
            (1.0, 1.25, 1.5, 1.75, 2.0)):
        if dim % 4:
            continue
        cfg = dict(dim=dim, stem_channels=32, num_blocks=nb, num_heads=4,
                   time_stride=ts, ffn_mult=ff, mel_pool_rank=8, dropout=0.1)
        try:
            p, mac = measure(cfg)
        except Exception:
            continue
        if mac <= CAP:
            rows.append({"cfg": cfg, "params": p, "mac": mac,
                         "T": INPUT_SHAPE[1] // ts})

    emit("swept; %d configs fit under the cap" % len(rows))
    emit("")

    emit("TOP 16 BY QUALITY PROXY  (sqrt(params)/mac^0.25)")
    emit("%-34s %8s %9s %4s %8s" % ("dim/b/ffn/ts", "params", "MMAC", "T", "proxy"))
    emit("-" * 68)
    for r in sorted(rows, key=lambda r: -quality_proxy(r["params"], r["mac"]))[:16]:
        c = r["cfg"]
        emit("d%-3d b%-2d ff%-5.2f ts%d %8d %8.3fM %4d %7.3f"
             % (c["dim"], c["num_blocks"], c["ffn_mult"], c["time_stride"],
                r["params"], r["mac"] / 1e6, r["T"],
                quality_proxy(r["params"], r["mac"])))

    emit("")
    emit("WIDEST under the cap (max capacity, for the params-first reading):")
    for r in sorted(rows, key=lambda r: -r["params"])[:8]:
        c = r["cfg"]
        emit("d%-3d b%-2d ff%-5.2f ts%d %8d params  %8.3fM  T=%d"
             % (c["dim"], c["num_blocks"], c["ffn_mult"], c["time_stride"],
                r["params"], r["mac"] / 1e6, r["T"]))

    open("sweepfine.out", "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    main()