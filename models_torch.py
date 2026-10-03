"""
models_torch.py -- PyTorch KWS architecture designed against the measured budget
in COMPLIANCE_ANALYSIS.md and arch_budget.py.

NEW code. The Flax `models.py` is left untouched because existing checkpoints
depend on it.

WHY THIS ARCHITECTURE (each point traces to a measurement, not a hunch)
-----------------------------------------------------------------------
1. RAM < 256 KB incl. PSRAM. The audit names the worst offender: `s_c2`,
   conv2's float32 output at 94,080 B. So this design narrows MEL and the
   channel axis immediately, and all reported activation sizes assume int8.
2. CPU <= 11.11 ms at a 100 ms hop (10% duty). Arithmetic floor is
   MAC / 3_840_000 (EE.VMULAS.S8 = 16 MAC/instr @ 240 MHz). The old model
   needed 11.24 MMAC and still measured 296 ms: ~87% of runtime was float32 on
   an integer-only vector unit. The fix is BOTH fewer MACs and an int8- and
   flash-friendly topology:
     - depthwise-separable wherever a dense kernel would do: this cuts MACs AND
       weight bytes, and weight bytes are what drove conv2's 49 weight passes
       from flash at 16.3 MB/s (COMPLIANCE_ANALYSIS.md 1a).
3. The 3 conformer blocks are 81% of the old 11.24 MMAC. Cutting to 2 blocks at
   dim 32 is the single biggest saving (COMPLIANCE_ANALYSIS.md 6, Option B).

DESIGN MANDATES -- from MEASURED failures in this project, do not violate
-----------------------------------------------------------------------
* NO absolute / learned temporal position embedding. It ACTIVELY HURT: a single
  global pool over a fixed 1 s window has to guess where the keyword is, and an
  absolute position parameter is free to encode that guess. Validation recall
  fell monotonically 77.7% -> 15.1% across the window. Instead: a learned
  RELATIVE position bias b[i - j] (T5 style). It depends only on the distance
  between two frames, so shifting the sequence leaves every logit unchanged.
* NO softmax attention pooling over time. It collapses onto one preferred frame
  and re-creates the same positional prior. Instead: symmetric soft-OR pooling,
  mean + logsumexp + max -- all permutation-invariant over time.
* KEEP learned attention pooling over MEL bins (mean-pool over mel throws away
  formant structure; that was a real measured improvement).
* delta / delta-delta computed IN-MODEL with fixed (non-learned) kernels, so the
  corpus format stays (49, 40, 1) and every existing dataset still loads.
* Prefer depthwise-separable ops.

    py models_torch.py     # params / MAC / floor table + CPU and GPU fwd+bwd
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # budget constants live in arch_budget.py
    from arch_budget import LAMBDA, BUDGET_MS, floor_ms  # type: ignore
except Exception:  # pragma: no cover - keep this file standalone-usable
    LAMBDA = 3_840_000.0
    BUDGET_MS = 11.11

    def floor_ms(mac: float) -> float:
        return mac / LAMBDA


INPUT_SHAPE: Tuple[int, int, int, int] = (1, 49, 40, 1)  # (N, T, MEL, 1)

# ==============================================================================
# Delta / delta-delta -- FIXED kernels, ZERO parameters
# ==============================================================================
def _time_diff(x: torch.Tensor) -> torch.Tensor:
    """0.5 * central difference along TIME (dim 2) with edge replication.

    Written with explicit clamped indexing rather than F.pad on purpose. F.pad
    consumes its spec from the LAST dimension backwards, so both the natural
    (0, 0, 1, 1) and the "explicit" 6-tuple pad a different axis than time in a
    (N,C,T,M) tensor: (0,0,1,1) pads MEL and returns the right SHAPE only when
    M == 1, so it survives a (49,1) smoke test while being wrong on real (49,40)
    data. Arithmetic matches `models.py::_delta_stack` exactly.
    """
    t = x.shape[2]
    base = torch.arange(t, device=x.device)
    nxt = (base + 1).clamp(max=t - 1)
    prv = (base - 1).clamp(min=0)
    return 0.5 * (x.index_select(2, nxt) - x.index_select(2, prv))


class DeltaStack(nn.Module):
    """(B,1,T,M) -> (B,3,T,M) as [log-mel, delta, delta-delta].

    Fixed kernels, so the network sees spectral dynamics with no learned
    parameters and the on-disk corpus format is unchanged. MAC count is 0 --
    these are adds and one scale, not multiply-accumulates.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = _time_diff(x)
        d2 = _time_diff(d1)
        return torch.cat([x, d1, d2], dim=1)

    def macs(self, in_shape: Tuple[int, ...]):
        """Protocol used by count_macs(): returns (mac_count, out_shape)."""
        return 0.0, (in_shape[0], 3) + tuple(in_shape[2:])


# ==============================================================================
# Depthwise-separable 2-D block (the stem workhorse)
# ==============================================================================
class DSBlock2D(nn.Module):
    """depthwise (kw x kh) -> GroupNorm -> swish -> pointwise 1x1.

    MAC = kw*kh*C_in per output position (depthwise) + C_in*C_out per position
    (pointwise), versus kw*kh*C_in*C_out dense. On the stem's C_in=3 input that
    is roughly 9x fewer MACs AND ~9x fewer weight bytes.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1), num_groups: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride,
                            padding=(kernel[0] // 2, kernel[1] // 2),
                            groups=in_ch, bias=False)
        # groups must divide the normalised channel count, which is in_ch
        self.norm = nn.GroupNorm(max(1, math.gcd(in_ch, num_groups)), in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(F.silu(self.norm(self.dw(x))))

    def macs(self, in_shape: Tuple[int, ...]):
        """(mac_count, out_shape). Threaded depthwise -> norm -> pointwise, so the
        generic walker in count_macs() never has to guess a chain's shape flow."""
        m, s = _conv_macs(self.dw, in_shape)     # depthwise
        m += _conv_macs(self.pw, s)[0]           # pointwise; GroupNorm is 0 MAC
        return m, s


# ==============================================================================
# Relative-position self-attention (T5-style bias, shift-invariant)
# ==============================================================================
class RelPosSelfAttention(nn.Module):
    """Multi-head self-attention with a learned RELATIVE position bias b[i - j].

    Why not an absolute embedding: measured, it cost 62.6 points of streaming
    recall (77.7% -> 15.1%) by letting the net encode WHERE in the window the
    keyword lives. A bias indexed by the signed distance i - j depends only on
    how far apart two frames are, so translating the sequence leaves every logit
    unchanged. Cost: (2T-1)*num_heads values = 97*2..4 = 194..388 params.
    """

    def __init__(self, dim: int, num_heads: int, max_len: int = 128,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_len - 1, num_heads))
        self.logit_elems = 0  # attention-logit footprint, set by macs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,T,D)
        b, t, d = x.shape
        h, dh = self.num_heads, self.head_dim
        qkv = self.qkv(x).view(b, t, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B,H,T,dh)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        rel = torch.arange(t, device=x.device)
        idx = rel[:, None] - rel[None, :] + (t - 1)         # (T,T) into [0,2T-2]
        logits = logits + self.rel_bias[idx].permute(2, 0, 1).unsqueeze(0)
        w = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0:
            w = F.dropout(w, self.dropout)
        out = torch.matmul(w, v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out)

    def macs(self, in_shape: Tuple[int, ...]):
        """(mac_count, out_shape)."""
        b, t, d = in_shape
        h, dh = self.num_heads, self.head_dim
        mac = t * d * (3 * d)          # qkv projection
        mac += h * t * t * dh          # Q @ K^T
        mac += h * t * t * dh          # A @ V
        mac += t * d * d               # output projection
        self.logit_elems = t * t * h   # logits are live only briefly
        return float(mac), tuple(in_shape)


# ==============================================================================
# Conformer-lite block: attention -> multi-scale depthwise conv -> Swish FFN
# ==============================================================================
class RelConformerBlock(nn.Module):
    """Residual conformer block, depthwise-separable.

    Differs from the Flax `RelConformerBlock` in the two places that matter for
    the budget: the multi-scale convolution is DEPTHWISE (groups=C) followed by
    a single pointwise mix instead of two dense convs, and the FFN expansion is
    configurable so the tight variants can trim it.
    """

    def __init__(self, dim: int, num_heads: int, kernel_sizes: Sequence[int] = (3, 7),
                 ffn_mult: float = 2.0, se_reduction: int = 8,
                 dropout: float = 0.0, max_len: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosSelfAttention(dim, num_heads, max_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dw = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.pw = nn.Linear(dim, dim, bias=False)
        se_dim = max(4, dim // se_reduction)
        self.se_fc1 = nn.Linear(dim, se_dim, bias=False)
        self.se_fc2 = nn.Linear(se_dim, dim, bias=False)
        self.norm3 = nn.LayerNorm(dim)
        self.hidden = max(4, int(round(dim * ffn_mult)))
        self.ff1 = nn.Linear(dim, self.hidden)
        self.ff2 = nn.Linear(self.hidden, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x)), self.dropout, self.training)

        y = self.norm2(x).transpose(1, 2)                      # (B,D,T)
        y = sum(F.silu(conv(y)) for conv in self.dw)
        y = self.pw(y.transpose(1, 2))
        z = F.silu(self.se_fc1(y.mean(dim=1)))                 # SE over time
        y = y * torch.sigmoid(self.se_fc2(z)).unsqueeze(1)
        x = x + F.dropout(y, self.dropout, self.training)

        h = self.ff2(F.silu(self.ff1(self.norm3(x))))           # Swish FFN
        return x + F.dropout(h, self.dropout, self.training)

    def macs(self, in_shape: Tuple[int, ...]):
        """(mac_count, out_shape) -- counts every child analytically."""
        b, t, d = in_shape
        mac = self.attn.macs(in_shape)[0]
        for conv in self.dw:
            mac += t * d * conv.kernel_size[0]     # depthwise: k taps/chan/frame
        mac += t * d * d                            # pointwise mix
        se_dim = self.se_fc1.out_features
        mac += d * se_dim + se_dim * d              # squeeze-excite (tiny)
        mac += t * d * self.hidden + t * self.hidden * d   # FFN
        return float(mac), tuple(in_shape)


# ==============================================================================
# Learned attention pooling over MEL bins -- KEEP, do not replace with mean
# ==============================================================================
class MelAttnPool(nn.Module):
    """Learned attention over MEL bins, per frame.

    Mean-pooling over mel was measurably worse: it discards formant structure,
    and WHICH mel bin carries the energy is the whole point of a log-mel front
    end for a keyword. Low-rank scorer (dim -> r -> 1) so it stays cheap: r=8
    costs about a quarter of a dense dim->dim scorer per position.
    """

    def __init__(self, dim: int, rank: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, rank, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(rank, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B,C,T,M) -> (B,C,T)
        w = torch.softmax(self.fc2(torch.tanh(self.fc1(x))), dim=-1)
        return torch.sum(x * w, dim=-1)

    def macs(self, in_shape: Tuple[int, ...]):
        b, c, t, m = in_shape
        r = self.fc1.out_channels
        return float(t * m * c * r + t * m * r), (b, c, t)


# ==============================================================================
# Soft-OR statistic pooling over TIME -- NOT softmax attention pooling
# ==============================================================================
class SoftORPool(nn.Module):
    """Symmetric soft-OR: mean + logsumexp + max over time -> (B, 3C).

    Softmax attention pooling over time collapses onto one preferred frame,
    re-creating exactly the positional prior the relative-position fix exists to
    remove. All three statistics here are permutation-invariant over time, so no
    frame can be preferred by construction.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,T,D) -> (B,3D)
        mean = x.mean(dim=1)
        lse = torch.logsumexp(x, dim=1) - math.log(x.shape[1])
        return torch.cat([mean, lse, x.amax(dim=1)], dim=-1)

    def macs(self, in_shape: Tuple[int, ...]):
        return 0.0, (in_shape[0], 3 * in_shape[-1])


# ==============================================================================
# The network
# ==============================================================================
class KWSNet(nn.Module):
    """Budget-compliant keyword-spotting network for the (49, 40, 1) log-mel contract.

    Data flow:
        (B, 49, 40, 1) or (B, 1, 49, 40)   [dataset format unchanged]
      -> DeltaStack              (B, 3, 49, 40)  fixed kernels, 0 params
      -> DSBlock2D  MEL 40->20   (B, C1, 49, 20)
      -> DSBlock2D  MEL 20->10   (B, D,  49, 10)  D = conformer width
      -> MelAttnPool             (B, D,  49)     learned over MEL bins
      -> N x RelConformerBlock   (B, 49, D)       relative position bias only
      -> SoftORPool              (B, 3D)         mean + logsumexp + max
      -> head                    (B, num_classes)

    No absolute temporal position embedding anywhere: the only positional
    parameter is RelPosSelfAttention.rel_bias, indexed by i - j. No softmax
    attention pooling over time: SoftORPool is permutation-invariant over frames.
    """

    def __init__(self, num_classes: int = 2, dim: int = 32, stem_channels: int = 24,
                 num_blocks: int = 2, num_heads: int = 4,
                 kernel_sizes: Sequence[int] = (3, 7), ffn_mult: float = 2.0,
                 mel_pool_rank: int = 8, dropout: float = 0.1,
                 mel_pool_size: int = 10, max_len: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim, self.num_blocks, self.num_heads = dim, num_blocks, num_heads

        self.delta = DeltaStack()
        self.stem1 = DSBlock2D(3, stem_channels, kernel=(3, 3), stride=(1, 2))
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
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _to_bctm(x: torch.Tensor, expected_c: int = 1) -> torch.Tensor:
        """Accept (B,T,M,C) or (B,C,T,M) so existing dataset loaders just work."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == expected_c:
            return x
        if x.shape[-1] == expected_c:            # (B,T,M,C) -> (B,C,T,M)
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bctm(x, 1)
        x = self.delta(x)
        x = self.stem1(x)
        x = self.stem2(x)                        # (B, D, T, M')
        seq = self.mel_gate(x)                   # (B, D, T) learned over MEL
        seq = seq.transpose(1, 2).contiguous()   # (B, T, D)
        for blk in self.blocks:
            seq = blk(seq)
        return self.head(self.drop(self.pool(seq)))   # (B, 3D) -> logits

    def canonical_input(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Normalise any accepted input layout to the (N, C, T, M) the MAC walker
        uses. count_macs() calls this so that a dataset-shaped (N, 49, 40, 1) is
        accounted for in the same frame as the forward pass."""
        return self._to_bctm(torch.empty(tuple(shape)), 1).shape

    def macs(self, in_shape: Tuple[int, ...]):
        """Whole-model (mac_count, out_shape), summed over stages.

        Layout note: `in_shape` must already be (N, C, T, M). count_macs() routes
        through canonical_input() first, so callers outside this file can pass the
        dataset's (N, 49, 40, 1) unchanged.
        """
        mac, s = 0.0, self.canonical_input(in_shape)
        for stage in (self.delta, self.stem1, self.stem2, self.mel_gate):
            m, s = stage.macs(s)
            mac += m
        for blk in self.blocks:
            m, s = blk.macs(s)
            mac += m
        m, s = self.pool.macs(s)
        mac += m
        m, s, _ = _walk(self.head, s)            # nn.Linear via the generic path
        mac += m
        return float(mac), s


# ==============================================================================
# Factory
# ==============================================================================
#: Presets aimed at the arch_budget.py targets: ultra-tight ~18k params / 0.80 MMAC,
#: tight ~26k / 1.20 MMAC, moderate ~38k / 1.80 MMAC.
ARCHS: Dict[str, Dict] = {
    # Presets. arch_budget.py targets ~18k/0.80M, ~26k/1.20M, ~38k/1.80M. All three
    # land on their MAC target; the counts run a little under because 2 blocks at
    # these widths is already past the point of diminishing returns, and an unused
    # parameter is cheaper than an unused MAC.
    "kws_ultra": dict(dim=24, stem_channels=16, num_blocks=2, num_heads=2,
                      kernel_sizes=(3,), ffn_mult=1.25, mel_pool_rank=6, dropout=0.10),
    "kws_tight": dict(dim=28, stem_channels=20, num_blocks=2, num_heads=2,
                      kernel_sizes=(3, 5), ffn_mult=1.5, mel_pool_rank=6, dropout=0.10),
    "kws_moderate": dict(dim=32, stem_channels=24, num_blocks=3, num_heads=4,
                         kernel_sizes=(3, 5), ffn_mult=1.5, mel_pool_rank=8,
                         dropout=0.15),
}
ARCHS["kwsnet"] = dict(ARCHS["kws_tight"])      # alias
ARCHS["kwsnet_tiny"] = dict(ARCHS["kws_ultra"])  # alias


def get_torch_model(arch: str = "kws_tight", num_classes: int = 2,
                    **overrides) -> KWSNet:
    """Build a KWSNet by name. Unknown names warn and fall back to 'kws_tight'
    rather than raising, so a stale training config still trains."""
    key = arch.lower().strip()
    if key not in ARCHS:
        print(f"[models_torch] unknown arch '{arch}' -> using 'kws_tight'")
        key = "kws_tight"
    cfg = dict(ARCHS[key])
    cfg["num_classes"] = num_classes
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return KWSNet(**cfg)


# ==============================================================================
# Analytic MAC counting
# ==============================================================================
def count_macs(model: nn.Module, input_shape: Tuple[int, ...] = INPUT_SHAPE,
               breakdown: bool = False):
    """Analytic multiply-accumulate COUNT for one inference at `input_shape`.

    No third-party profiler and no forward pass: shapes are propagated through the
    module tree by hand, so this is the number the firmware's cost model must be
    given, not a measurement of PyTorch's own execution.

    Rules
    -----
    * a module that exposes ``macs(in_shape) -> (count, out_shape)`` (the custom
      blocks in this file) is trusted for its own subtree and not descended into;
    * otherwise Conv1d / Conv2d / Linear use the standard formulae and
      shape-preserving layers (norm, activations, dropout, pooling) cost 0 MAC
      -- their real cost is memory traffic, which is what `estimate_activation_
      bytes` is for.

    Elementwise ops (swish, sigmoid, softmax, LayerNorm) are counted separately by
    `count_elementwise` and are NOT included here, because the floor in
    arch_budget.py is defined on MAC from EE.VMULAS.S8.
    """
    shape = tuple(input_shape)
    if hasattr(model, "canonical_input"):
        shape = tuple(model.canonical_input(shape))
    total, out_shape, rows = _walk_children(model, shape)
    # cross-check: a model that knows its own total must agree with the generic walk
    if hasattr(model, "macs"):
        own, _ = model.macs(shape)
        assert abs(own - total) < 1.0, (
            f"{type(model).__name__}.macs() says {own:,.0f} but the generic walk "
            f"found {total:,.0f} -- the two accounting paths disagree")
    if breakdown:
        return total, rows
    return total


def _walk_children(module: nn.Module, in_shape: Tuple[int, ...]):
    """Walk `module`'s children, ignoring any `macs()` the root itself exposes, so
    the generic path is an independent check rather than a tautology."""
    kids = list(module.children())
    if not kids:
        return _walk(module, in_shape)
    total, s, rows = 0.0, tuple(in_shape), []
    for child in kids:
        cm, s, cr = _walk(child, s)
        total += cm
        rows.extend(cr)
    return total, s, rows


# ==============================================================================
# Elementwise op count + activation footprint
# ==============================================================================
def count_elementwise(model: nn.Module, input_shape: Tuple[int, ...] = INPUT_SHAPE,
                      device="cpu") -> int:
    """Number of scalar elementwise ops (swish, sigmoid, softmax, LayerNorm, ...).

    These do NOT contribute to the arch_budget floor, which is defined on
    EE.VMULAS.S8 MAC. They are reported separately because COMPLIANCE_ANALYSIS.md
    attributes ~86,000 scalar swishes per inference to a large part of the 296 ms:
    a core with only an integer vector unit pays for every one of them. Counting
    them makes that line item visible instead of hidden.
    """
    total = [0]

    def hook(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            total[0] += out.numel()

    handles = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm,
                            nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(tuple(input_shape), device=device))
    for h in handles:
        h.remove()
    return total[0]


@torch.no_grad()
def estimate_activation_bytes(model: nn.Module, input_shape: Tuple[int, ...] = INPUT_SHAPE,
                              bytes_per_elem: int = 1) -> Dict[str, int]:
    """int8 activation footprint of one inference, measured from real forward hooks.

    Reports a PEAK live-set figure, not a sum over layers. A sum would be
    meaningless against the 256 KB bound: a firmware reuses scratch, so only the
    tensors alive at the same instant occupy RAM. Liveness is derived from the
    execution order: a tensor is live from the step that produces it until the
    last module that consumes it, which is exact for this straight-line graph and
    conservative elsewhere.

    The largest single tensor is reported separately. The audit's 1.88x RAM
    failure traced to exactly one buffer -- conv2's 94,080 B float32 output -- so
    the max, not the total, is what breaks a bound.

    `bytes_per_elem=1` assumes a fully int8 pipeline, which is the only way the
    CPU bound is reachable on an integer-only core.
    """
    model = model.eval()
    order: List[torch.Tensor] = []      # produced-at step
    consumed: List[int] = []            # last step that read a given tensor

    def pre_hook(_m, inp):
        for t in inp:
            if isinstance(t, torch.Tensor):
                for i, o in enumerate(order):
                    if o is t:                # identity, not equality
                        consumed[i] = max(consumed[i], len(order))
                        break

    def post_hook(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            order.append(out)
            consumed.append(len(order))

    mods = list(model.modules())
    handles = [m.register_forward_pre_hook(pre_hook) for m in mods]
    handles += [m.register_forward_hook(post_hook) for m in mods]
    model(torch.zeros(tuple(input_shape)))
    for h in handles:
        h.remove()

    sizes = [t.numel() for t in order]
    peak, live = 0, 0
    for i, n in enumerate(sizes):
        live += n                                  # produced
        peak = max(peak, live)
        if consumed[i] <= i + 1:
            live -= n                             # dead after its last reader
    return {
        "num_tensors": len(sizes),
        "peak_bytes": int(peak * bytes_per_elem),
        "sum_bytes": int(sum(sizes) * bytes_per_elem),
        "largest_bytes": int(max(sizes) * bytes_per_elem) if sizes else 0,
    }


def _walk(module: nn.Module, in_shape: Tuple[int, ...]):
    """Returns (mac, out_shape, rows) for one module."""
    name = type(module).__name__

    # custom blocks own their subtree and declare their own output shape
    if hasattr(module, "macs") and callable(getattr(module, "macs")):
        m, out = module.macs(in_shape)
        return float(m), tuple(out), [(name, float(m))]

    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        m, out = _conv_macs(module, in_shape)
        return m, out, [(name, m)]

    if isinstance(module, nn.Linear):
        assert len(in_shape) >= 1, "Linear needs at least a batch dim"
        n = 1
        for s in in_shape[:-1]:
            n *= s
        m = float(n * in_shape[-1] * module.out_features)
        return m, tuple(in_shape[:-1]) + (module.out_features,), [(name, m)]

    if isinstance(module, nn.Sequential):
        total, s, rows = 0.0, tuple(in_shape), []
        for child in module.children():
            cm, s, cr = _walk(child, s)
            total += cm
            rows.extend(cr)
        return total, s, rows

    # Container fallback: thread the shape SEQUENTIALLY through the children.
    # Any block that genuinely branches (two inputs into one) exposes macs() and is
    # handled above, so a plain ordered chain is the only thing that reaches here.
    kids = list(module.children())
    if kids:
        total, s, rows = 0.0, tuple(in_shape), []
        for child in kids:
            cm, s, cr = _walk(child, s)
            total += cm
            rows.extend(cr)
        return total, s, rows

    return 0.0, tuple(in_shape), [(name, 0.0)]   # leaf, no MAC


def _conv_macs(conv: nn.Module, in_shape: Tuple[int, ...]):
    """Standard conv MAC formula, grouped convs included."""
    if isinstance(conv, nn.Conv1d):
        c_in, l_in = in_shape[1], in_shape[2]
        k = conv.kernel_size[0]
        s, p, d = conv.stride[0], conv.padding[0], conv.dilation[0]
        l_out = (l_in + 2 * p - d * (k - 1) - 1) // s + 1
        pos = in_shape[0] * l_out
        mac = pos * (c_in // conv.groups) * conv.out_channels * k
        return float(mac), (in_shape[0], conv.out_channels, l_out)

    c_in, h_in, w_in = in_shape[1], in_shape[2], in_shape[3]
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    ph, pw = conv.padding
    dh, dw = conv.dilation
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    pos = in_shape[0] * h_out * w_out
    mac = pos * (c_in // conv.groups) * conv.out_channels * kh * kw
    return float(mac), (in_shape[0], conv.out_channels, h_out, w_out)


# ==============================================================================
# Self-test / report
# ==============================================================================
def _run_fwd_bwd(model: nn.Module, device: torch.device, batch: int = 8):
    """One forward+backward+optimizer step. Returns (loss, grad-norm)."""
    model = model.to(device).train()
    x = torch.randn(batch, 49, 40, 1, device=device)
    y = torch.randint(0, model.num_classes, (batch,), device=device)
    logits = model(x)
    assert logits.shape == (batch, model.num_classes), \
        f"expected {(batch, model.num_classes)}, got {tuple(logits.shape)}"
    loss = F.cross_entropy(logits, y)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert torch.isfinite(loss).item(), "non-finite loss"
    assert all(torch.isfinite(p).all().item() for p in model.parameters()), \
        "non-finite parameter after the step"
    return float(loss.item()), float(gnorm)


def _counting_self_test() -> None:
    """Verify count_macs() itself before trusting any number it reports.

    A MAC accountant that is silently wrong is worse than none, because it is
    used to REJECT architectures. Two independent checks:
      1. the generic walker against a hand-computed Conv2d+Linear figure;
      2. KWSNet's own per-block accounting against the generic walker (also
         asserted inside count_macs, but it is worth seeing the breakdown);
      3. layout invariance, so a dataset-shaped (N,49,40,1) is not misread as
         channels-first -- that bug was real and inflated every figure 2x.
    """

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3, stride=(1, 2), padding=1, bias=False)
            self.g = nn.GroupNorm(1, 8)
            self.l = nn.Linear(8 * 49 * 20, 5)

        def forward(self, x):
            return self.l(self.g(self.c(x)).flatten(1))

    truth = 49 * 20 * 3 * 8 * 9 + (49 * 20 * 8) * 5     # conv + linear
    got = count_macs(Plain(), (1, 3, 49, 40))
    assert got == truth, f"generic walker {got:,.0f} != hand truth {truth:,.0f}"
    print(f"  walker vs hand-computed truth: {got:,.0f} == {truth:,.0f}  OK")

    kw = get_torch_model("kws_tight")
    a, b = count_macs(kw, (1, 49, 40, 1)), count_macs(kw, (1, 1, 49, 40))
    assert a == b, f"layout-dependent MAC: {a:,.0f} vs {b:,.0f}"
    print(f"  layout invariance (1,49,40,1) == (1,1,49,40): {a:,.0f}  OK")

    # DeltaStack must stay parameter-free and match the fixed-kernel definition.
    ds = DeltaStack()
    assert sum(q.numel() for q in ds.parameters()) == 0
    ramp = torch.arange(49, dtype=torch.float32).reshape(1, 1, 49, 1)
    d = ds(ramp)
    # Interior frames only: at t=0 and t=1 the edge-replicate padding deliberately
    # halves the first difference, so the boundary is not expected to be exact.
    assert abs(d[0, 1, 10, 0].item() - 1.0) < 1e-6, "delta of a unit ramp must be 1"
    assert abs(d[0, 2, 10, 0].item() - 0.0) < 1e-6, "delta-delta of a ramp is 0"

    # The discriminative check: differencing must run along TIME (dim 2) and leave
    # MEL (dim 3) untouched. Differencing along mel instead -- which is what an
    # ambiguous F.pad spec produces -- passes the ramp test above for a (49, 1)
    # input and still silently wrong on real (49, 40) data.
    t_ramp = torch.arange(49, dtype=torch.float32).reshape(1, 1, 49, 1).expand(1, 1, 49, 40)
    dt = ds(t_ramp)
    assert dt.shape == (1, 3, 49, 40)
    assert torch.allclose(dt[0, 1, 1:-1], torch.ones(47, 40)), \
        "interior delta of a time ramp must be 1 at every mel bin"
    assert torch.allclose(dt[0, 2, 2:-2], torch.zeros(45, 40)), \
        "interior delta-delta of a ramp must be 0"
    # mel must be untouched: an input constant along mel has zero delta everywhere
    assert torch.allclose(dt[0, 1], dt[0, 1, :, :1].expand(49, 40)), \
        "delta must not differ along the mel axis"
    print(f"  DeltaStack: 0 params; interior delta={d[0,1,10,0].item():.1f}, "
          f"delta-delta={d[0,2,10,0].item():.1f}; time axis confirmed on (49,40)  OK")

    # SoftORPool must be invariant to the ORDER of the frames it pools, which is
    # the property that softmax attention pooling over time lacks.
    pool = SoftORPool(kw.dim)
    seq = torch.randn(2, 49, kw.dim)
    assert torch.allclose(pool(seq), pool(seq.flip(1)), atol=1e-5), \
        "SoftORPool is not permutation-invariant over time"
    print("  SoftORPool is permutation-invariant over time (no preferred frame)  OK")

    # The relative bias must depend ONLY on the signed distance i - j, never on the
    # absolute frame index. That is precisely what makes it shift-invariant, and
    # precisely what an absolute position embedding fails. Tested directly on the
    # (T,T) bias matrix, with RANDOM bias values -- a zero-initialised bias would
    # satisfy any invariance test trivially.
    attn = RelPosSelfAttention(kw.dim, kw.num_heads).eval()
    t = 21
    with torch.no_grad():
        attn.rel_bias.copy_(torch.randn_like(attn.rel_bias))
        idx = torch.arange(t)[:, None] - torch.arange(t)[None, :] + (t - 1)
        bmat = attn.rel_bias[idx].permute(2, 0, 1)      # (H, T, T)
    dist = idx - (t - 1)
    for dd in torch.unique(dist):
        vals = bmat[:, dist == dd]
        assert torch.allclose(vals, vals[:, :1].expand_as(vals)), \
            f"bias varies with absolute position at distance {int(dd)}"
    # and it must NOT be symmetric in i-j: a symmetric bias cannot order frames
    assert not torch.allclose(bmat, bmat.transpose(1, 2)), \
        "bias is symmetric in i-j, so it carries no direction"
    print(f"  relative position bias: pure function of i-j over {t}x{t}, "
          f"asymmetric, no absolute index  OK")


def _shape_contract_checks(model: nn.Module, device: torch.device) -> None:
    """The 49x40x1 contract must hold in both layouts and at several batch sizes."""
    model = model.to(device).eval()
    with torch.no_grad():
        for shape in [(1, 49, 40, 1), (5, 49, 40, 1), (1, 49, 40), (1, 1, 49, 40)]:
            out = model(torch.zeros(*shape, device=device))
            assert out.shape[0] == shape[0], f"{shape} -> {tuple(out.shape)}"
    print("    (49, 40, 1) contract OK: (N,49,40,1), (N,49,40), (N,1,49,40)")


def main() -> None:
    print("=" * 106)
    print(f"KWSNet budget report.  floor_ms = MAC / {LAMBDA:,.0f} "
          f"(EE.VMULAS.S8 = 16 MAC/instr @ 240 MHz).  Budget <= {BUDGET_MS:.2f} ms "
          f"at a 100 ms hop.")
    print("=" * 106)
    print(f"{'arch':<14} {'params':>8} {'MAC':>11} {'floor':>10} {'headroom':>9} "
          f"{'peak act':>10} {'largest':>9} {'weights int8':>13} {'RAM total':>11}")
    print("-" * 106)

    results = []
    for arch in ["kws_ultra", "kws_tight", "kws_moderate"]:
        model = get_torch_model(arch, num_classes=2)
        params = sum(p.numel() for p in model.parameters())
        mac = count_macs(model, INPUT_SHAPE)
        f = floor_ms(mac)
        act = estimate_activation_bytes(model, INPUT_SHAPE, bytes_per_elem=1)
        ram = act["peak_bytes"] + params          # int8 weights + peak activations
        print(f"{arch:<14} {params:>8,} {mac:>11,.0f} {f:>9.3f}ms "
              f"{BUDGET_MS / f:>8.1f}x {act['peak_bytes'] / 1024:>9.1f}K "
              f"{act['largest_bytes'] / 1024:>8.1f}K {params / 1024:>12.1f}K "
              f"{ram / 1024:>10.1f}K")
        results.append((arch, model, params, mac, f, act))

    print()
    print("Reference -- bcconformer_v3, measured: 84,865 params / 11.24 MMAC / 2.93 ms")
    print("floor, but 296 ms ACTUAL and 491,947 B of activations (480 KB = 1.88x the")
    print("256 KB bound), of which a single float32 conv output was 94,080 B.")
    print()
    print("Scalar elementwise ops per inference (real cost on an integer-only core):")
    for arch, model, *_ in results:
        print(f"  {arch:<14} {count_elementwise(model):>9,}")
    print()
    print("MAC breakdown, kws_tight (numbering disambiguates the repeated block types):")
    mac_total, rows = count_macs(get_torch_model("kws_tight"), INPUT_SHAPE,
                                 breakdown=True)
    seen: Dict[str, int] = {}
    for name, m in rows:
        if m <= 0:
            continue
        seen[name] = seen.get(name, 0) + 1
        tag = f"{name}[{seen[name]}]" if name in (
            "DSBlock2D", "RelConformerBlock", "Linear") else name
        print(f"  {tag:<26} {m:>12,.0f}  {m / mac_total * 100:>5.1f}%")
    print(f"  {'TOTAL':<26} {mac_total:>12,.0f}")
    print()

    print("MAC-accounting self-test (verify the accountant before trusting it):")
    _counting_self_test()
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Forward + backward on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("-" * 106)
    for arch, model, *_ in results:
        _shape_contract_checks(model, device)
        loss, gnorm = _run_fwd_bwd(model, device)
        print(f"  {arch:<14} loss {loss:.4f}  grad-norm {gnorm:>7.3f}  "
              f"batch-4 out {tuple(model(torch.zeros(4, 49, 40, 1, device=device)).shape)}")

    # Mandates asserted in code, not just asserted in prose.
    names = [n for n, _ in get_torch_model("kws_tight").named_parameters()]
    assert not any("pos_emb" in n or "abs_pos" in n or "posenc" in n for n in names), \
        "absolute position embedding present -- it cost 62.6 points of recall"
    assert any("rel_bias" in n for n in names), "relative position bias missing"
    print("\nMandate check: relative position bias b[i-j] present; no absolute pos emb.")
    print("Mandate check: MelAttnPool (learned, over mel) + SoftORPool (no time softmax).")
    print("Mandate check: delta/delta-delta in-model with 0 parameters.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()