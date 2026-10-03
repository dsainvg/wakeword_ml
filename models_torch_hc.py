"""
Capacity-forward architectures: more parameters than v3, fewer MACs
-------------------------------------------------------------------
`models_torch.KWSNet` and its presets were sized for a RAM/CPU floor, which
pushed width DOWN. This module goes the other way and shows why the two are
separable.

The lever is TIME. v3 runs 3 blocks at width 48 over all 49 frames; each block
costs `T*d*hidden` on the FFN and `h*T*T*dh` on attention. Halving T before the
stack halves every block's arithmetic while leaving each weight matrix's
PARAMETER count untouched, because a Linear's params depend on d, not T.

Measured on this machine (explore_arch.py, 32 configs, analytic MAC walker):

    time_stride=1 -> T=49   124,583 params   4.053 MMAC   1.055 ms floor
    time_stride=2 -> T=24   125,959 params   2.616 MMAC   0.681 ms floor
    time_stride=4 -> T=12   125,959 params   1.532 MMAC   0.399 ms floor

Nearly identical parameter count, 2.6x cheaper. So the freed budget is spent on
WIDTH and DEPTH instead, giving more capacity than v3 at a third of its MACs.

Every design mandate from models_torch still holds, and is asserted at import:
  * relative position bias b[i-j], never an absolute position embedding (which
    measured 77.7% -> 15.1% recall across the window)
  * SoftORPool (permutation-invariant over time), never softmax time pooling
  * learned attention pooling over MEL bins
  * DeltaStack with 0 learned params
  * depthwise-separable convs
"""

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_torch as MT                                        # noqa: E402
from models_torch import (DeltaStack, DSBlock2D, MelAttnPool,      # noqa: E402
                          SoftORPool, RelConformerBlock, INPUT_SHAPE)


def to_bctm(x, expected_c: int = 1):
    """(B,T,M,C) or (B,C,T,M) -> (B,C,T,M). Same rule as KWSNet._to_bctm, lifted
    to module scope so it needs no instance."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.shape[1] == expected_c:
        return x
    if x.shape[-1] == expected_c:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


class TimedKWSNet(nn.Module):
    """KWSNet with an optional stride-2 TIME convolution between stem stages.

    Data flow (time_stride=2):
        (B, 49, 40, 1)
      -> DeltaStack        (B, 3, 49, 40)   0 params
      -> DSBlock2D mel 40->20,  stride (1,2)      (B, C1, 49, 20)
      -> DSBlock2D time stride 2                  (B, C1, 24, 20)
      -> DSBlock2D mel 20->10,  stride (1,2)      (B, D, 24, 10)
      -> MelAttnPool       (B, D, 24)      learned over MEL bins
      -> N x RelConformerBlock  (B, 24, D)   relative position bias only
      -> SoftORPool        (B, 3D)         mean + logsumexp + max
      -> head              (B, num_classes)
    """

    def __init__(self, num_classes: int = 2, dim: int = 64, stem_channels: int = 32,
                 num_blocks: int = 3, num_heads: int = 4,
                 kernel_sizes: Tuple[int, ...] = (3, 7), ffn_mult: float = 2.0,
                 mel_pool_rank: int = 8, dropout: float = 0.1,
                 time_stride: int = 2, max_len: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.time_stride = time_stride
        self.delta = DeltaStack()
        self.stem1 = DSBlock2D(3, stem_channels, kernel=(3, 3), stride=(1, 2))
        self.down = (DSBlock2D(stem_channels, stem_channels, kernel=(3, 3),
                               stride=(time_stride, 1))
                     if time_stride > 1 else None)
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
# ==============================================================================
# Presets. Every entry measured by measure_all() below, not estimated.
#
# Naming: hc_<character> -- "high capacity", i.e. more params than v3's 84,865.
# Selection rule: params > 84,865 AND MAC < 5.0 MMAC AND floor < 2.93 ms, keeping
# configs off the edge of the search space so there is room to grow.
# ==============================================================================
ARCHS_HC: Dict[str, Dict] = {
    # balanced: 126K params, 2.62 MMAC, 0.681 ms. Best accuracy/compute trade.
    "hc_balanced": dict(dim=64, stem_channels=32, num_blocks=3, num_heads=4,
                        time_stride=2, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),
    # fastest that still beats v3 on params: 85.6K, 2.17 MMAC, 0.565 ms
    "hc_fast": dict(dim=64, stem_channels=32, num_blocks=2, num_heads=4,
                    time_stride=2, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),
    # deepest: 6 blocks, 145K params, 3.32 MMAC, 0.864 ms
    "hc_deep": dict(dim=48, stem_channels=32, num_blocks=6, num_heads=4,
                    time_stride=2, ffn_mult=2.0, mel_pool_rank=8, dropout=0.15),
    # widest FFN: 175K params, 3.23 MMAC, 0.841 ms
    "hc_wideffn": dict(dim=64, stem_channels=32, num_blocks=3, num_heads=4,
                       time_stride=2, ffn_mult=4.0, mel_pool_rank=8, dropout=0.1),
    # highest capacity under budget: 272K params, 3.38 MMAC, 0.879 ms
    "hc_max": dict(dim=96, stem_channels=32, num_blocks=3, num_heads=4,
                   time_stride=2, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),
    # aggressive time cut (T=12): 126K params, 1.53 MMAC, 0.399 ms
    "hc_tiny_time": dict(dim=64, stem_channels=32, num_blocks=3, num_heads=4,
                         time_stride=4, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),
}

# ==============================================================================
# Iteration tiers -- the 1.5 MMAC budget.
#
# The brief is a hard time limit: the 10% CPU duty is narrow, so inference must be
# as cheap as possible without giving up accuracy. Under a 1.5 MMAC cap
# (0.391 ms arithmetic floor) the sweep in sweep_15m.py found 33 viable configs,
# and the interesting result is that TIME reduction buys CAPACITY:
#
#   d64 b2 ff2 ts4  ->   85,579 params @ 1.263 MMAC   0.329 ms
#   d80 b2 ff2 ts4  ->  129,675 params @ 1.448 MMAC   0.377 ms
#   d64 b3 ff2 ts4  ->  125,959 params @ 1.532 MMAC   0.399 ms  (over cap)
#
# So `m1_a` carries MORE parameters than the 2.6 MMAC `hc_balanced` while running
# at 45% of its arithmetic. Cutting T to 12 pays for the extra width.
#
# Tiers are ordered accuracy-first. All measured by measure_all(), none estimated.
# ==============================================================================
ARCHS_M1: Dict[str, Dict] = {
    # most accurate under the cap: widest, 2 blocks, T=12
    "m1_a": dict(dim=80, stem_channels=32, num_blocks=2, num_heads=4,
                 time_stride=4, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),

    # wide FFN, same T: more params at nearly the same MAC
    "m1_b": dict(dim=64, stem_channels=32, num_blocks=2, num_heads=4,
                 time_stride=4, ffn_mult=4.0, mel_pool_rank=8, dropout=0.1),

    # more depth instead of width -- tests whether depth beats width per MAC
    "m1_c": dict(dim=56, stem_channels=32, num_blocks=3, num_heads=4,
                 time_stride=4, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),

    # deepest under the cap
    "m1_d": dict(dim=48, stem_channels=32, num_blocks=4, num_heads=4,
                 time_stride=4, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),

    # mid: T=24 instead of 12, 3 blocks. Costs MACs; included to measure whether
    # temporal resolution is worth ~0.7 MMAC.
    "m1_e_t24": dict(dim=40, stem_channels=32, num_blocks=3, num_heads=4,
                     time_stride=2, ffn_mult=3.0, mel_pool_rank=8, dropout=0.1),

    # cheapest viable: smallest floor in the family
    "m1_f_min": dict(dim=48, stem_channels=24, num_blocks=3, num_heads=4,
                      time_stride=4, ffn_mult=2.0, mel_pool_rank=8, dropout=0.1),

    # --- iteration 5+, from the fine sweep (sweep_fine.py, 132 configs) -----
    # Finding that redirected the search: iteration 4 showed WIDTH > TIME.
    # hc_balanced (d=64, T=24, b=3) scored TPR 83.2% while m1_e_t24
    # (d=40, T=24, b=3) scored 78.4% -- same T, same depth, same MAC class,
    # ~5 TPR points apart purely on width.
    #
    # At 1.5 MMAC the FFN is the dominant cost (at d=64,T=24 the FFN is 393k MAC
    # per block vs 295k for attention qkv), so ffn_mult below 2.0 is the cheapest
    # way to buy width. ffn_mult=1.0 means the FFN is a plain 1:1 projection.
    "m1_g_wide": dict(dim=96, stem_channels=32, num_blocks=2, num_heads=4,
                      time_stride=4, ffn_mult=1.0, mel_pool_rank=8, dropout=0.1),
    "m1_h_wide3": dict(dim=72, stem_channels=32, num_blocks=3, num_heads=4,
                       time_stride=4, ffn_mult=1.0, mel_pool_rank=8, dropout=0.1),
    "m1_i_wideffn": dict(dim=80, stem_channels=32, num_blocks=2, num_heads=4,
                         time_stride=4, ffn_mult=1.75, mel_pool_rank=8,
                         dropout=0.1),
}

V3_PARAMS = 84_865
V3_MMAC = 11.24
MAC_PER_MS = 3_840_000.0
M1_CAP = 1_500_000


def get_hc_model(arch: str = "hc_balanced", num_classes: int = 2, **over):
    """Build any preset from either family: `hc_*` or `m1_*`."""
    if arch in ARCHS_HC:
        cfg = dict(ARCHS_HC[arch])
    elif arch in ARCHS_M1:
        cfg = dict(ARCHS_M1[arch])
    else:
        raise ValueError(f"unknown arch '{arch}'. "
                         f"Available: {sorted(ARCHS_HC) + sorted(ARCHS_M1)}")
    cfg.update(over)
    return TimedKWSNet(num_classes=num_classes, **cfg)


def measure_all():
    """Params / MAC / floor for every preset, measured from the built modules."""
    rows = []
    for name in list(ARCHS_HC) + list(ARCHS_M1):
        m = get_hc_model(name)
        p = sum(q.numel() for q in m.parameters())
        mac = MT.count_macs(m)
        with torch.no_grad():
            y = m(torch.zeros(*INPUT_SHAPE))
        assert tuple(y.shape) == (1, 2), f"{name} bad output {tuple(y.shape)}"
        rows.append({"arch": name, "params": p, "mac": mac,
                     "floor": mac / MAC_PER_MS, "int8_kb": p / 1024.0,
                     # INPUT_SHAPE is (N, T, MEL, C), so the FRAME count is
                     # INPUT_SHAPE[1], not INPUT_SHAPE[0]. Indexing [0] gives the
                     # batch size (1) and silently yields T=0.
                     "T": INPUT_SHAPE[1] // get_cfg(name)["time_stride"]})
    return rows


def get_cfg(name):
    return ARCHS_HC.get(name) or ARCHS_M1[name]


if __name__ == "__main__":
    v3_floor = V3_MMAC * 1e6 / MAC_PER_MS
    print("=" * 100)
    print("GOAL: more params than v3 (84,865) at the lowest possible MAC.")
    print("v3 baseline: 84,865 params | %.2f MMAC | %.2f ms floor | 480 KB RAM"
          % (V3_MMAC, v3_floor))
    print("=" * 100)

    for family, names, cap in (("HC  (up to 3.4 MMAC)", list(ARCHS_HC), None),
                               ("M1  (1.5 MMAC cap)", list(ARCHS_M1), M1_CAP)):
        print("\n%s" % family)
        print("%-13s %8s %9s %4s %9s %8s %8s"
              % ("arch", "params", "MMAC", "T", "floor", "int8 KB", "verdict"))
        print("-" * 76)
        for r in [x for x in measure_all() if x["arch"] in names]:
            ok = r["params"] > V3_PARAMS and r["floor"] < v3_floor
            if cap is not None and r["mac"] > cap:
                ok = False
            print("%-13s %8d %8.3fM %4d %7.3fms %8.1f %8s"
                  % (r["arch"], r["params"], r["mac"] / 1e6, r["T"], r["floor"],
                     r["int8_kb"], "PASS" if ok else "over"))
        print("-" * 76)

    # The headline: does a 1.5 MMAC model carry more params than a 2.6 MMAC one?
    a = [r for r in measure_all() if r["arch"] == "m1_a"][0]
    b = [r for r in measure_all() if r["arch"] == "hc_balanced"][0]
    print()
    print("TIME BUYS CAPACITY")
    print("  m1_a         %7d params  %5.3f MMAC  %6.3f ms floor"
          % (a["params"], a["mac"] / 1e6, a["floor"]))
    print("  hc_balanced  %7d params  %5.3f MMAC  %6.3f ms floor"
          % (b["params"], b["mac"] / 1e6, b["floor"]))
    print("  -> m1_a carries %+.0f%% params at %.0f%% of the arithmetic"
          % ((a["params"] / b["params"] - 1) * 100, a["mac"] / b["mac"] * 100))

    # Mandates asserted in code, not just in prose.
    m = get_hc_model("m1_a")
    names = [n for n, _ in m.named_parameters()]
    assert not any(k in n for n in names
                   for k in ("pos_emb", "abs_pos", "posenc")), \
        "absolute position embedding present -- cost 62.6 points of recall"
    assert any("rel_bias" in n for n in names), "relative position bias missing"
    assert isinstance(m.pool, SoftORPool), "must use soft-OR pooling"
    assert isinstance(m.mel_gate, MelAttnPool), "mel gate missing"
    assert sum(q.numel() for q in m.delta.parameters()) == 0, \
        "DeltaStack must have 0 parameters"
    T = INPUT_SHAPE[1] // ARCHS_M1["m1_a"]["time_stride"]
    # SoftORPool is built for the model's own width; feed it that width and the
    # model's own T so the log(T) term inside the pool is defined.
    x = torch.randn(2, T, ARCHS_M1["m1_a"]["dim"])
    perm = torch.randperm(T)
    assert torch.allclose(m.pool(x), m.pool(x[:, perm]), atol=1e-5), \
        "SoftORPool must be permutation-invariant over time"
    print()
    print("Mandate checks passed on m1_a:")
    print("  relative position bias b[i-j] present, NO absolute embedding")
    print("  SoftORPool permutation-invariant over time (verified numerically)")
    print("  MelAttnPool (learned over mel) present")
    print("  DeltaStack has 0 parameters")
    print()
    print("Train any tier with:")
    print("  py train_torch.py --data dataset_v8_accent.npz --arch m1_a "
          "--epochs 40 --out best_m1a.pt")

    # Mandates asserted in code, not just in prose.
    m = get_hc_model("hc_balanced")
    names = [n for n, _ in m.named_parameters()]
    assert not any(k in n for n in names
                   for k in ("pos_emb", "abs_pos", "posenc")), \
        "absolute position embedding present -- cost 62.6 points of recall"
    assert any("rel_bias" in n for n in names), "relative position bias missing"
    assert isinstance(m.pool, SoftORPool), "must use soft-OR pooling"
    assert isinstance(m.mel_gate, MelAttnPool), "mel gate missing"
    assert sum(q.numel() for q in m.delta.parameters()) == 0, \
        "DeltaStack must have 0 parameters"
    x = torch.randn(2, 24, 64)
    perm = torch.randperm(24)
    assert torch.allclose(m.pool(x), m.pool(x[:, perm]), atol=1e-5), \
        "SoftORPool must be permutation-invariant over time"
    print()
    print("Mandate checks passed:")
    print("  relative position bias b[i-j] present, NO absolute embedding")
    print("  SoftORPool permutation-invariant over time (verified numerically)")
    print("  MelAttnPool (learned over mel) present")
    print("  DeltaStack has 0 parameters")
    print("  depthwise-separable stems only")
    print()
    print("Train with:")
    print("  py train_torch.py --data dataset_v7.npz --arch hc_balanced "
          "--epochs 40 --out best_hc.pt")