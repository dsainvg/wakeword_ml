"""
1.5 MMAC architecture sweep -- the hard time budget
--------------------------------------------------
The brief: get inference as cheap as possible because the 10% CPU duty limit is
narrow. At 100 ms hop the budget is 11.11 ms, and the arithmetic floor is
MAC/3,840,000, so the budget itself is never the binding constraint -- the risk is
that a cheap-but-blind model fails its accuracy target. This sweep therefore
looks for the most accurate architecture AT OR UNDER 1.5 MMAC (0.391 ms floor),
rather than the cheapest architecture overall.

Design axes explored:
  time_stride  1/2/4   -- the dominant MAC lever (T=49/24/12)
  dim          32..96  -- width; params scale with dim^2, MACs with dim^2 * T
  num_blocks   2..5    -- depth; MACs linear in blocks
  ffn_mult     2/3/4   -- FFN expansion; cheap in params, expensive in MACs

Selection criterion: highest expected accuracy per MAC inside the 1.5 MMAC cap,
where deeper-and-narrow is preferred over wider-and-shallower because the audit
found the model was never capacity-limited but data-limited.

    py sweep_15m.py
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


def main():
    emit("=" * 104)
    emit("CAP: %.2f MMAC  ->  floor %.3f ms  (budget 11.11 ms at 100 ms hop)"
         % (CAP / 1e6, CAP / MAC_PER_MS))
    emit("v3 reference: 84,865 params | 11.24 MMAC | 2.93 ms floor")
    emit("=" * 104)

    rows = []
    for ts, dim, nb, ff in itertools.product(
            (2, 4), (32, 40, 48, 56, 64, 80, 96), (2, 3, 4, 5), (2.0, 3.0, 4.0)):
        cfg = dict(dim=dim, stem_channels=32, num_blocks=nb, num_heads=4,
                   time_stride=ts, ffn_mult=ff, mel_pool_rank=8, dropout=0.1)
        try:
            p, mac = measure(cfg)
        except Exception:
            continue
        rows.append({"cfg": cfg, "params": p, "mac": mac,
                     "T": INPUT_SHAPE[0] // ts})

    fits = [r for r in rows if r["mac"] <= CAP]
    emit("swept %d configs; %d fit under %.2f MMAC" % (len(rows), len(fits), CAP / 1e6))
    emit("")

    # Rank by params at fixed MAC: more parameters inside the same compute budget
    # is the whole point of this exercise, and the audit showed capacity is not
    # the bottleneck, so params are a proxy for headroom, not a target itself.
    fits.sort(key=lambda r: (-r["params"], r["mac"]))
    emit("TOP OPTIONS UNDER THE CAP (ranked by params = capacity headroom)")
    emit("%-46s %8s %9s %5s %9s" % ("dim/blocks/ffn/ts", "params", "MMAC", "T",
                                    "floor"))
    emit("-" * 84)
    for r in fits[:18]:
        c = r["cfg"]
        emit("d%-3d b%-2d ff%.0f ts%d%s%-11s %8d %8.3fM %5d %7.3fms"
             % (c["dim"], c["num_blocks"], c["ffn_mult"], c["time_stride"],
                "", "", r["params"], r["mac"] / 1e6, r["T"], r["mac"] / MAC_PER_MS))

    emit("")
    emit("RECOMMENDED TIERS (accuracy-first ordering inside the cap):")
    picks = [
        ("tier_a_best_acc", dict(dim=64, stem_channels=32, num_blocks=3,
                                 num_heads=4, time_stride=4, ffn_mult=2.0,
                                 mel_pool_rank=8, dropout=0.1)),
        ("tier_b_balanced", dict(dim=56, stem_channels=32, num_blocks=4,
                                 num_heads=4, time_stride=2, ffn_mult=2.0,
                                 mel_pool_rank=8, dropout=0.1)),
        ("tier_c_deep", dict(dim=40, stem_channels=32, num_blocks=5,
                            num_heads=4, time_stride=2, ffn_mult=3.0,
                            mel_pool_rank=8, dropout=0.1)),
        ("tier_d_cheapest", dict(dim=48, stem_channels=24, num_blocks=3,
                                 num_heads=4, time_stride=4, ffn_mult=2.0,
                                 mel_pool_rank=8, dropout=0.1)),
    ]
    emit("%-16s %8s %9s %5s %9s %10s" % ("tier", "params", "MMAC", "T", "floor",
                                          "under cap"))
    emit("-" * 66)
    for name, cfg in picks:
        p, mac = measure(cfg)
        emit("%-16s %8d %8.3fM %5d %7.3fms %10s"
             % (name, p, mac / 1e6, INPUT_SHAPE[0] // cfg["time_stride"],
                mac / MAC_PER_MS, "yes" if mac <= CAP else "NO -- exceeds"))

    open("sweep15.out", "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    main()