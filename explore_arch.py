"""
Architecture design-space explorer: high params, low MACs
--------------------------------------------------------
Brief for this round: MORE parameters than bcconformer_v3 (84,865), while FEWER
than 5 MMAC, while FASTER than v3's 2.93 ms arithmetic floor.

Those are not in tension, and the reason is that params and MACs are decoupled
along one axis this project has left unused: TIME.

v3 keeps all 49 frames through 3 blocks at width 48. Every block pays
`T*d*hidden` on the FFN and `h*T^2*dh` on attention, at T=49. Halving the frame
count before the stack therefore roughly HALVES the arithmetic of every block,
while leaving each weight matrix's parameter count essentially untouched -- a
Linear's params depend on d, not T.

So: downsample time in the stem, run the stack at T=25 or T=13, and spend the
freed budget on WIDTH and DEPTH. That buys parameters nearly for free.

Every candidate is MEASURED, not estimated: params from the real module tree, MACs
from models_torch's analytic walker, floor with the audit's own MAC/3.84e6.

    py explore_arch.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_torch as MT                                        # noqa: E402
from models_torch import (DeltaStack, DSBlock2D, MelAttnPool,      # noqa: E402
                          SoftORPool, RelConformerBlock, INPUT_SHAPE)

MAC_PER_MS = 3_840_000.0
V3_PARAMS = 84_865
V3_MMAC = 11.24

OUT = []


def emit(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def to_bctm(x, expected_c: int = 1):
    """Accept (B,T,M,C) or (B,C,T,M) -> (B,C,T,M).

    Same rule as KWSNet._to_bctm, lifted to module scope so it can be called
    without an instance (the staticmethod needs an explicit self otherwise).
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.shape[1] == expected_c:
        return x
    if x.shape[-1] == expected_c:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


class TimedKWSNet(nn.Module):
    """KWSNet plus an optional stride-2 TIME convolution in the stem.

    Everything that made KWSNet correct is kept: DeltaStack (0 params), learned
    attention pooling over MEL bins, relative position bias b[i-j] only (never an
    absolute embedding), SoftORPool (permutation-invariant over time, never a
    softmax time pool), depthwise-separable convs throughout. The single addition
    is the time-stride, which trades temporal resolution for arithmetic.
    """

    def __init__(self, num_classes: int = 2, dim: int = 64, stem_channels: int = 32,
                 num_blocks: int = 3, num_heads: int = 4,
                 kernel_sizes=(3, 7), ffn_mult: float = 2.0,
                 mel_pool_rank: int = 8, dropout: float = 0.1,
                 time_stride: int = 1, max_len: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.delta = DeltaStack()
        self.stem1 = DSBlock2D(3, stem_channels, kernel=(3, 3), stride=(1, 2))
        self.down = (DSBlock2D(stem_channels, stem_channels, kernel=(3, 3),
                               stride=(time_stride, 1)) if time_stride > 1 else None)
        self.stem2 = DSBlock2D(stem_channels, dim, kernel=(3, 3), stride=(1, 2))
        self.mel_gate = MelAttnPool(dim, rank=mel_pool_rank)
        self.blocks = nn.ModuleList([
            RelConformerBlock(dim, num_heads, kernel_sizes=kernel_sizes,
                              ffn_mult=ffn_mult, dropout=dropout, max_len=max_len)
            for _ in range(num_blocks)
        ])
        self.pool = SoftORPool(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(3 * dim, num_classes)
        self.apply(self._init)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = to_bctm(x, 1)
        x = self.delta(x)
        x = self.stem1(x)
        if self.down is not None:
            x = self.down(x)
        x = self.stem2(x)
        seq = self.mel_gate(x).transpose(1, 2).contiguous()
        for blk in self.blocks:
            seq = blk(seq)
        return self.head(self.drop(self.pool(seq)))

    def canonical_input(self, shape):
        return to_bctm(torch.empty(tuple(shape)), 1).shape

    def macs(self, in_shape):
        mac, s = 0.0, self.canonical_input(in_shape)
        for stage in (self.delta, self.stem1):
            m, s = stage.macs(s)
            mac += m
        if self.down is not None:
            m, s = self.down.macs(s)
            mac += m
        for stage in (self.stem2, self.mel_gate):
            m, s = stage.macs(s)
            mac += m
        for blk in self.blocks:
            m, s = blk.macs(s)
            mac += m
        m, s = self.pool.macs(s)
        mac += m
        m, s, _ = MT._walk(self.head, s)
        mac += m
        return float(mac), s

import itertools

def measure(name, cfg):
    m = TimedKWSNet(**cfg)
    params = sum(p.numel() for p in m.parameters())
    mac = MT.count_macs(m)
    with torch.no_grad():
        y = m(torch.zeros(*INPUT_SHAPE))
    ok = tuple(y.shape) == (1, 2)
    return {"name": name, "params": params, "mac": mac,
            "floor": mac / MAC_PER_MS, "ok": ok, "cfg": cfg,
            "int8_kb": params / 1024.0}


def sweep():
    cands = []

    # --- family A: wide & shallow. Max params per MAC. ---------------------
    for dim, nb, ts in itertools.product((64, 96, 128), (2, 3), (1, 2)):
        cands.append(("A_wide_d%d_b%d_ts%d" % (dim, nb, ts),
                      dict(dim=dim, num_blocks=nb, time_stride=ts,
                           stem_channels=32, ffn_mult=2.0, num_heads=4)))

    # --- family B: deep & narrow. More nonlinearity, still time-reduced. ---
    for dim, nb, ts in itertools.product((48, 64), (4, 5, 6), (1, 2)):
        cands.append(("B_deep_d%d_b%d_ts%d" % (dim, nb, ts),
                      dict(dim=dim, num_blocks=nb, time_stride=ts,
                           stem_channels=32, ffn_mult=2.0, num_heads=4)))

    # --- family C: wide FFN (params are cheap in FFN when T is small) ------
    for dim, nb, ff in itertools.product((64, 80), (3, 4), (3.0, 4.0)):
        cands.append(("C_wideffn_d%d_b%d_f%.0f" % (dim, nb, ff),
                      dict(dim=dim, num_blocks=nb, ffn_mult=ff, time_stride=2,
                           stem_channels=32, num_heads=4)))

    return [measure(n, c) for n, c in cands]


def main():
    emit("=" * 96)
    emit("TARGETS: params > %d (v3)   MAC < 5.0 MMAC   floor < %.2f ms (v3)"
         % (V3_PARAMS, V3_MMAC * 1e6 / MAC_PER_MS))
    emit("v3 reference: 84,865 params, %.2f MMAC, %.2f ms floor, 480 KB RAM"
         % (V3_MMAC, V3_MMAC * 1e6 / MAC_PER_MS))
    emit("=" * 96)
    emit("")
    res = sweep()

    ok = [r for r in res if r["ok"]]
    fits = [r for r in ok if r["params"] > V3_PARAMS and r["mac"] < 5.0e6]
    emit("swept %d configs; %d ran; %d satisfy params>%d AND <5 MMAC"
         % (len(res), len(ok), len(fits), V3_PARAMS))
    emit("")

    if fits:
        fits.sort(key=lambda r: r["mac"])
        emit("CANDIDATES THAT MEET BOTH (sorted by MAC, i.e. fastest first)")
        emit("%-24s %8s %9s %8s %9s" % ("name", "params", "MMAC", "floor", "int8 KB"))
        emit("-" * 64)
        for r in fits[:12]:
            emit("%-24s %8d %8.3fM %7.3fms %8.1f"
                 % (r["name"], r["params"], r["mac"] / 1e6, r["floor"], r["int8_kb"]))
    else:
        emit("NO config satisfied both. Widest options under 5 MMAC:")
        under = sorted([r for r in ok if r["mac"] < 5.0e6], key=lambda r: -r["params"])
        emit("%-24s %8s %9s %8s" % ("name", "params", "MMAC", "floor"))
        for r in under[:12]:
            emit("%-24s %8d %8.3fM %7.3fms" % (r["name"], r["params"],
                                               r["mac"] / 1e6, r["floor"]))
        emit("")
        emit("=> v3's 84,865 params are NOT reachable under 5 MMAC with this")
        emit("   block family. The FFN/attention widths that cost params also cost")
        emit("   MACs unless time is cut harder. See the manual table below.")

    emit("")
    emit("MAC vs TIME, holding width fixed (why time is the lever):")
    for ts in (1, 2, 4):
        r = measure("probe_d64_b3_ts%d" % ts,
                    dict(dim=64, num_blocks=3, time_stride=ts, stem_channels=32))
        emit("  time_stride=%d -> T=%2d  params %6d  MAC %6.3fM  floor %.3f ms"
             % (ts, 49 // ts, r["params"], r["mac"] / 1e6, r["floor"]))

    open("explore.out", "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    main()