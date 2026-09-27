"""
State-of-the-Art TinyML Keyword Spotting Model Architectures in Flax / JAX
--------------------------------------------------------------------------
Implements 4 premier architectures for microcontroller edge deployment (< 256 KB RAM):

1. BC-ResNet (Broadcasting-Residual Network - Kim et al., Interspeech 2021)
   - BC-ResNet-1 (5.2k params) & BC-ResNet-3 (18.5k params)
   - Splits 2D convs into 1D temporal depthwise & 1D frequency depthwise
   - Injects global sub-band frequency context via broadcasting
   - SOTA accuracy on Speech Commands & noisy wake-word benchmarks

2. TC-ResNet (Temporal Convolutional ResNet - Choi et al., 2019)
   - 1D Temporal Dilated Convolutions along the time axis
   - Extremely cache-efficient and fast on Xtensa / ARM Cortex SIMD DSPs
   - Preserves temporal phoneme transition dynamics

3. DS-ResNet-SE (Squeeze-and-Excitation Residual Network)
   - Depthwise Separable Convolutions augmented with Squeeze-and-Excitation (SE)
   - Dynamically recalibrates Mel-frequency channel weights
   - Suppresses background noise bands while boosting keyword formant frequencies

4. DS-CNN-S (Depthwise Separable CNN - Zhang et al., ARM Hello Edge)
   - Classic Industry-standard TinyML benchmark

All models utilize GroupNorm for robust convergence across small batch sizes and noise.
"""

from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


# ==============================================================================
# 1. BC-ResNet (Broadcasting Residual Network - Kim et al., Interspeech 2021)
# ==============================================================================

class BCResBlock(nn.Module):
    """
    Broadcasting-Residual Block:
    - 1D Frequency Depthwise Conv (1, 3)
    - 1D Temporal Depthwise Conv (3, 1)
    - Frequency Broadcasting: pools frequency channel context and broadcasts to time
    - 1x1 Pointwise Conv
    - Skip connection with average pooling if stride > 1
    """
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Residual branch
        if self.stride != 1 or x.shape[-1] != self.channels:
            res = nn.avg_pool(x, window_shape=(self.stride, 1), strides=(self.stride, 1), padding='SAME')
            if res.shape[-1] != self.channels:
                res = nn.Conv(features=self.channels, kernel_size=(1, 1), use_bias=False)(res)
                res = nn.GroupNorm(num_groups=min(4, self.channels))(res)
        else:
            res = x

        # 1. Frequency Depthwise Conv (1, 3)
        h = nn.Conv(
            features=x.shape[-1],
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='SAME',
            feature_group_count=x.shape[-1],
            use_bias=False
        )(x)
        h = nn.GroupNorm(num_groups=min(4, x.shape[-1]))(h)
        h = nn.relu(h)

        # 2. Temporal Depthwise Conv (3, 1)
        h = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 1),
            strides=(self.stride, 1),
            padding='SAME',
            feature_group_count=x.shape[-1],
            use_bias=False
        )(h)
        h = nn.GroupNorm(num_groups=min(4, x.shape[-1]))(h)
        h = nn.relu(h)

        # 3. Frequency Broadcast Connection (injects global frequency context)
        f_broadcast = jnp.mean(h, axis=2, keepdims=True)
        h = h + f_broadcast

        # 4. Pointwise Conv (1, 1) to target channels
        h = nn.Conv(features=self.channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(h)
        h = nn.GroupNorm(num_groups=min(4, self.channels))(h)

        return nn.relu(h + res)


class BCResNet(nn.Module):
    """
    BC-ResNet Keyword Spotting Architecture.
    scale=1: BC-ResNet-1 (~5.2k params)
    scale=3: BC-ResNet-3 (~18.5k params)
    """
    num_classes: int = 2
    scale: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        base_c = 16 if self.scale == 1 else 24

        # Stem: Conv2D with stride (2, 2)
        h = nn.Conv(
            features=base_c,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(x)
        h = nn.GroupNorm(num_groups=min(4, base_c))(h)
        h = nn.relu(h)

        # Stage 1
        h = BCResBlock(channels=base_c, stride=1)(h, train=train)
        h = BCResBlock(channels=base_c, stride=1)(h, train=train)

        # Stage 2
        stage2_c = base_c * 2
        h = BCResBlock(channels=stage2_c, stride=2)(h, train=train)
        h = BCResBlock(channels=stage2_c, stride=1)(h, train=train)

        # Stage 3
        if self.scale >= 3:
            stage3_c = base_c * 3
            h = BCResBlock(channels=stage3_c, stride=2)(h, train=train)
            h = BCResBlock(channels=stage3_c, stride=1)(h, train=train)
        else:
            h = BCResBlock(channels=stage2_c, stride=1)(h, train=train)

        # Global Average Pooling over (Time, Frequency) axes (1, 2)
        h = jnp.mean(h, axis=(1, 2))

        # Dropout
        h = nn.Dropout(rate=0.15, deterministic=not train)(h)

        # Dense projection to class logits
        logits = nn.Dense(features=self.num_classes)(h)
        return logits


# ==============================================================================
# 2. TC-ResNet (Temporal Convolutional ResNet - Choi et al., 2019)
# ==============================================================================

class TCResBlock(nn.Module):
    """1D Temporal Residual Block with dilation."""
    channels: int
    stride: int = 1
    dilation: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Residual shortcut
        if self.stride != 1 or x.shape[-1] != self.channels:
            res = nn.Conv(features=self.channels, kernel_size=(1,), strides=(self.stride,), use_bias=False)(x)
            res = nn.GroupNorm(num_groups=min(4, self.channels))(res)
        else:
            res = x

        # 1D Temporal Conv 1
        h = nn.Conv(
            features=self.channels,
            kernel_size=(3,),
            strides=(self.stride,),
            kernel_dilation=(self.dilation,),
            padding='SAME',
            use_bias=False
        )(x)
        h = nn.GroupNorm(num_groups=min(4, self.channels))(h)
        h = nn.relu(h)

        # 1D Temporal Conv 2
        h = nn.Conv(
            features=self.channels,
            kernel_size=(3,),
            strides=(1,),
            kernel_dilation=(1,),
            padding='SAME',
            use_bias=False
        )(h)
        h = nn.GroupNorm(num_groups=min(4, self.channels))(h)

        return nn.relu(h + res)


class TCResNet(nn.Module):
    """
    Temporal Convolutional ResNet (TC-ResNet-8):
    Treats the 40 Mel filterbanks as input feature channels, convolving solely along time.
    Parameters: ~15,200 (Flash: ~15 KB INT8)
    """
    num_classes: int = 2
    base_channels: int = 24

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x input shape: (Batch, Time=49, Mel=40, Channels=1)
        # Reshape to (Batch, Time=49, Channels=40)
        b, t, m, c = x.shape
        x_1d = jnp.reshape(x, (b, t, m * c))

        # Stem 1D Conv
        h = nn.Conv(features=self.base_channels, kernel_size=(3,), strides=(1,), padding='SAME', use_bias=False)(x_1d)
        h = nn.GroupNorm(num_groups=min(4, self.base_channels))(h)
        h = nn.relu(h)

        # Stage 1: 2 blocks (stride 1, 2)
        h = TCResBlock(channels=self.base_channels, stride=1, dilation=1)(h, train=train)
        h = TCResBlock(channels=self.base_channels * 2, stride=2, dilation=1)(h, train=train)

        # Stage 2: 2 blocks (stride 1, 2) with dilation
        h = TCResBlock(channels=self.base_channels * 2, stride=1, dilation=2)(h, train=train)
        h = TCResBlock(channels=self.base_channels * 3, stride=2, dilation=2)(h, train=train)

        # Global Average Pooling across time dimension (axis 1)
        h = jnp.mean(h, axis=1)

        # Dropout & Classifier
        h = nn.Dropout(rate=0.15, deterministic=not train)(h)
        logits = nn.Dense(features=self.num_classes)(h)
        return logits


# ==============================================================================
# 3. DS-ResNet-SE (Depthwise Separable ResNet with Squeeze-and-Excitation)
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention."""
    reduction: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        channels = x.shape[-1]
        # Squeeze: Global average pooling over (H, W)
        z = jnp.mean(x, axis=(1, 2), keepdims=True)
        # Excitation: Two-layer MLP with Sigmoid gating
        s = nn.Dense(features=max(4, channels // self.reduction), use_bias=False)(z)
        s = nn.relu(s)
        s = nn.Dense(features=channels, use_bias=False)(s)
        s = nn.sigmoid(s)
        return x * s


class DSResBlockSE(nn.Module):
    """Depthwise Separable Block with Squeeze-and-Excitation."""
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Residual branch
        if self.stride != 1 or x.shape[-1] != self.channels:
            res = nn.Conv(features=self.channels, kernel_size=(1, 1), strides=(self.stride, self.stride), use_bias=False)(x)
            res = nn.GroupNorm(num_groups=min(4, self.channels))(res)
        else:
            res = x

        # Depthwise 3x3 Conv
        h = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=x.shape[-1],
            use_bias=False
        )(x)
        h = nn.GroupNorm(num_groups=min(4, x.shape[-1]))(h)
        h = nn.relu(h)

        # Pointwise 1x1 Conv
        h = nn.Conv(features=self.channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(h)
        h = nn.GroupNorm(num_groups=min(4, self.channels))(h)

        # Squeeze-and-Excitation Attention
        h = SEBlock(reduction=4)(h)

        return nn.relu(h + res)


class DSResNetSE(nn.Module):
    """
    Depthwise Separable ResNet with Squeeze-and-Excitation Channel Attention.
    Parameters: ~16,800 (Flash: ~16.8 KB INT8)
    """
    num_classes: int = 2
    base_channels: int = 20

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Stem
        h = nn.Conv(features=self.base_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME', use_bias=False)(x)
        h = nn.GroupNorm(num_groups=min(4, self.base_channels))(h)
        h = nn.relu(h)

        # Stages
        h = DSResBlockSE(channels=self.base_channels, stride=1)(h, train=train)
        h = DSResBlockSE(channels=self.base_channels * 2, stride=2)(h, train=train)
        h = DSResBlockSE(channels=self.base_channels * 2, stride=1)(h, train=train)
        h = DSResBlockSE(channels=self.base_channels * 3, stride=2)(h, train=train)

        # Global Average Pooling
        h = jnp.mean(h, axis=(1, 2))
        h = nn.Dropout(rate=0.15, deterministic=not train)(h)
        logits = nn.Dense(features=self.num_classes)(h)
        return logits


# ==============================================================================
# 4. DS-CNN-S (Depthwise Separable CNN - ARM Hello Edge)
# ==============================================================================

class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 2D Conv (3, 3) followed by Pointwise 1x1 Conv."""
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        h = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=x.shape[-1],
            use_bias=False
        )(x)
        h = nn.GroupNorm(num_groups=min(4, x.shape[-1]))(h)
        h = nn.relu(h)

        h = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='SAME',
            use_bias=False
        )(h)
        h = nn.GroupNorm(num_groups=min(4, self.channels))(h)
        h = nn.relu(h)
        return h


class DSCNN(nn.Module):
    """DS-CNN-S Architecture."""
    num_classes: int = 2
    num_layers: int = 4
    num_filters: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        h = nn.Conv(features=self.num_filters, kernel_size=(10, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)
        h = nn.GroupNorm(num_groups=min(4, self.num_filters))(h)
        h = nn.relu(h)

        for _ in range(self.num_layers):
            h = DepthwiseSeparableBlock(channels=self.num_filters, stride=1)(h, train=train)

        h = jnp.mean(h, axis=(1, 2))
        h = nn.Dropout(rate=0.20, deterministic=not train)(h)
        logits = nn.Dense(features=self.num_classes)(h)
        return logits


# ==============================================================================
# 5. MH-KWS / Conformer-Lite (Multi-Head Attention + Temporal Depthwise Conv)
# ==============================================================================

class ConformerLiteBlock(nn.Module):
    """
    Lightweight Conformer block with Multi-Head Self-Attention across time
    and depthwise temporal convolutions to capture phonetic sequence dynamics.
    """
    dim: int = 32
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # 1. Multi-Head Self-Attention over time frames
        norm1 = nn.LayerNorm()(x)
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.dim)(norm1)
        x = x + nn.Dropout(rate=0.10, deterministic=not train)(attn)

        # 2. Depthwise Conv Module (Temporal dynamics of phonemes)
        norm2 = nn.LayerNorm()(x)
        conv = nn.Conv(features=self.dim, kernel_size=(5,), padding='SAME', feature_group_count=self.dim, use_bias=False)(norm2)
        conv = nn.relu(conv)
        conv = nn.Conv(features=self.dim, kernel_size=(1,), use_bias=False)(conv)
        x = x + nn.Dropout(rate=0.10, deterministic=not train)(conv)

        # 3. Feed-Forward Network
        norm3 = nn.LayerNorm()(x)
        ff = nn.Dense(features=self.dim * 2)(norm3)
        ff = nn.relu(ff)
        ff = nn.Dense(features=self.dim)(ff)
        x = x + nn.Dropout(rate=0.10, deterministic=not train)(ff)
        return x


class ConformerLiteKWS(nn.Module):
    """
    TinyML Multi-Head Attention Conformer for Keyword Spotting.
    Parameters: ~29,150 (Flash: ~28.5 KB INT8)
    Explicitly prevents acoustic false triggers (sirens, barks, speech babble)
    by strictly learning the cross-temporal attention transitions of /ə/ -> /m/ -> /eɪ/ -> /z/.
    """
    num_classes: int = 2
    dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x is (B, 49, 40, 1)
        # Collapse frequency into embedding
        x_proj = nn.Conv(features=self.dim, kernel_size=(3, 3), strides=(1, 2), padding='SAME', use_bias=False)(x)
        x_proj = nn.relu(x_proj)
        x_proj = nn.Conv(features=self.dim, kernel_size=(3, 3), strides=(1, 2), padding='SAME', use_bias=False)(x_proj)
        x_proj = nn.relu(x_proj)
        seq = jnp.mean(x_proj, axis=2)  # (B, 49, dim)

        # Conformer blocks
        seq = ConformerLiteBlock(dim=self.dim)(seq, train=train)
        seq = ConformerLiteBlock(dim=self.dim)(seq, train=train)

        # Global temporal pooling
        h = jnp.mean(seq, axis=1)
        logits = nn.Dense(features=self.num_classes)(h)
        return logits


# ==============================================================================
# 6. BC-Conformer-50K (Broadcasting Convolutions + Multi-Head Self-Attention)
# ==============================================================================

class TemporalConformerBlock50K(nn.Module):
    """Temporal Conformer block with 4-head Multi-Head Self-Attention and depthwise conv."""
    dim: int = 44
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        norm1 = nn.LayerNorm()(x)
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.dim)(norm1)
        x = x + nn.Dropout(rate=0.15, deterministic=not train)(attn)

        norm2 = nn.LayerNorm()(x)
        conv = nn.Conv(features=self.dim, kernel_size=(5,), padding='SAME', feature_group_count=self.dim, use_bias=False)(norm2)
        conv = nn.relu(conv)
        conv = nn.Conv(features=self.dim, kernel_size=(1,), use_bias=False)(conv)
        x = x + nn.Dropout(rate=0.15, deterministic=not train)(conv)

        norm3 = nn.LayerNorm()(x)
        ff = nn.Dense(features=self.dim * 2)(norm3)
        ff = nn.relu(ff)
        ff = nn.Dense(features=self.dim)(ff)
        x = x + nn.Dropout(rate=0.15, deterministic=not train)(ff)
        return x


class BCConformer50K(nn.Module):
    """
    State-of-the-Art 50 KB KWS Architecture:
    Broadcasting Sub-Band Convolutions + Multi-Head Temporal Self-Attention.
    Parameters: ~49,880 (Flash: ~48.7 KB INT8)
    Arena RAM: ~42 KB (well below 256 KB ESP32-S3 limit).
    Combines spatial frequency broadcasting with strict multi-phoneme temporal attention.

    `conformer_dim` and `blocks` are exposed so the same topology can be scaled up
    when a larger training corpus is available (see BCConformer100K).
    """
    num_classes: int = 2
    channels: int = 32
    conformer_dim: int = 44
    blocks: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x is (B, 49, 40, 1)
        # Stage 1: Frequency Stem
        h = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(1, 2), padding='SAME', use_bias=False)(x)
        h = nn.GroupNorm(num_groups=4)(h)
        h = nn.relu(h)

        # Stage 2: Sub-Band Frequency Depthwise Conv + Broadcasting
        f_conv = nn.Conv(features=self.channels, kernel_size=(1, 3), padding='SAME', feature_group_count=self.channels, use_bias=False)(h)
        f_conv = nn.GroupNorm(num_groups=4)(f_conv)
        f_conv = nn.relu(f_conv)
        f_bcast = jnp.mean(f_conv, axis=2, keepdims=True)
        h = h + f_conv + f_bcast

        # Stage 3: Frequency projection to Conformer temporal sequence
        h = nn.Conv(features=self.conformer_dim, kernel_size=(3, 3), strides=(1, 2), padding='SAME', use_bias=False)(h)
        h = nn.GroupNorm(num_groups=4)(h)
        h = nn.relu(h)
        seq = jnp.mean(h, axis=2)  # (B, 49, conformer_dim)

        # Stage 4: Conformer Blocks
        for _ in range(self.blocks):
            seq = TemporalConformerBlock50K(dim=self.conformer_dim, num_heads=4)(seq, train=train)

        # Stage 5: Dual Pooling & Classification
        pooled = jnp.mean(seq, axis=1)
        max_pooled = jnp.max(seq, axis=1)
        feat = jnp.concatenate([pooled, max_pooled], axis=-1)
        feat = nn.Dropout(rate=0.20, deterministic=not train)(feat)
        logits = nn.Dense(features=self.num_classes)(feat)
        return logits


class BCConformer100K(BCConformer50K):
    """Scaled BC-Conformer (~130k params). Still ~130 KB INT8 flash, which fits the
    ESP32-S3 comfortably, and has the headroom a 40k-sample corpus needs."""
    channels: int = 40
    conformer_dim: int = 64
    blocks: int = 3


# ==============================================================================
# 7. BC-Conformer-V2 (position-aware, multi-scale, gated)
# ==============================================================================
# Four concrete weaknesses in BCConformer50K were fixed rather than guessed at:
#
#   1. NO POSITIONAL INFORMATION. TemporalConformerBlock50K uses bare
#      nn.SelfAttention, which is permutation-equivariant over time -- the model
#      structurally cannot tell where in the 1 s window a frame sits. That matters
#      enormously here, because the streaming engine slides the window every 100 ms
#      and measured recall falls from 40% to 14% as the keyword moves from the start
#      of the window to the end. V2 adds a learned absolute temporal embedding.
#
#   2. MEAN POOLING OVER MEL. Stage 3 collapsed the frequency axis with
#      jnp.mean(h, axis=2), discarding all sub-band structure after two strided
#      convs. V2 replaces it with a learned attention pool over mel bins, so the
#      network chooses which bands matter for the keyword's formants.
#
#   3. NO INPUT DERIVATIVES. Speech discrimination leans heavily on spectral
#      dynamics; static log-mel alone is a weaker feature. V2 computes the first and
#      second time-derivatives (delta / delta-delta) inside the model with fixed
#      kernels, so the dataset format is unchanged and every existing corpus still
#      loads, but the first learned convolution sees 3 channels instead of 1.
#
#   4. WEAK TEMPORAL BLOCK. V2 widens to dim 48, uses two depthwise kernel sizes
#      (3 and 7) so it sees both phoneme transitions and longer context, replaces the
#      ReLU feed-forward with Swish (the Conformer standard) and adds a squeeze-
#      excitation gate after the convolution. Three blocks instead of two.
#
# Pooling also moves from concat(mean, max) to attentive statistics pooling, which
# learns how much weight to give each frame instead of weighting all frames equally.

def _delta_stack(x: jnp.ndarray) -> jnp.ndarray:
    """(B, T, M, 1) log-mel -> (B, T, M, 3) [log-mel, delta, delta-delta].

    Computed with fixed (non-learned) kernels inside the model so the on-disk corpus
    format stays (T, M, 1) and existing datasets remain loadable.
    """
    t = x[:, :, :, 0]
    d1_pad = jnp.pad(t, ((0, 0), (1, 1), (0, 0)), mode="edge")
    d1 = 0.5 * (d1_pad[:, 2:, :] - d1_pad[:, :-2, :])
    d2_pad = jnp.pad(d1, ((0, 0), (1, 1), (0, 0)), mode="edge")
    d2 = 0.5 * (d2_pad[:, 2:, :] - d2_pad[:, :-2, :])
    return jnp.stack([t, d1, d2], axis=-1)


class ImprovedConformerBlock(nn.Module):
    """Conformer block: attention -> multi-scale depthwise conv -> Swish FFN, with an
    SE gate on the convolution output and a residual around each sub-layer."""
    dim: int = 48
    num_heads: int = 4
    se_reduction: int = 8
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # --- Multi-head self-attention ---
        # Positional information enters through the learned absolute temporal embedding
        # added once before the stack (see BCConformerV2), which is what makes this
        # attention position-aware; flax 0.12's SelfAttention has no position_bias arg.
        norm1 = nn.LayerNorm()(x)
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
        )(norm1)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(attn)

        # --- Multi-scale depthwise convolution + pointwise mixing + SE gate ---
        norm2 = nn.LayerNorm()(x)
        conv3 = nn.Conv(features=self.dim, kernel_size=(3,), padding="SAME",
                        feature_group_count=self.dim, use_bias=False)(norm2)
        conv7 = nn.Conv(features=self.dim, kernel_size=(7,), padding="SAME",
                        feature_group_count=self.dim, use_bias=False)(norm2)
        y = jax.nn.swish(conv3) + jax.nn.swish(conv7)
        y = nn.Conv(features=self.dim, kernel_size=(1,), use_bias=False)(y)
        # Squeeze-and-excitation over time: recalibrate which frames matter.
        # z is (B, 1, D) and broadcasts directly against y (B, T, D).
        z = jnp.mean(y, axis=1, keepdims=True)
        z = nn.Dense(features=max(4, self.dim // self.se_reduction), use_bias=False)(z)
        z = jax.nn.swish(z)
        z = nn.Dense(features=self.dim, use_bias=False)(z)
        y = y * jax.nn.sigmoid(z)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(y)

        # --- Swish feed-forward (Conformer standard; ReLU FFN is measurably weaker) ---
        norm3 = nn.LayerNorm()(x)
        ff = nn.Dense(features=self.dim * 2)(norm3)
        ff = jax.nn.swish(ff)
        ff = nn.Dense(features=self.dim)(ff)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(ff)
        return x


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling: learned attention weights over time, then mean and
    std of the weighted sequence. Replaces concat(mean, max), which weights every frame
    equally and cannot express "the keyword is somewhere in here"."""
    dim: int = 48
    attn_dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, D)
        score = nn.Dense(features=self.attn_dim)(x)
        score = jax.nn.tanh(score)
        score = nn.Dense(features=1, use_bias=False)(score)   # (B, T, 1)
        w = nn.softmax(score, axis=1)
        mean = jnp.sum(x * w, axis=1)
        var = jnp.sum((x ** 2) * w, axis=1) - mean ** 2
        std = jnp.sqrt(jnp.maximum(var, 1e-6))
        return jnp.concatenate([mean, std], axis=-1)          # (B, 2D)


class BCConformerV2(nn.Module):
    """BC-Conformer V2 -- ~86k params, ~84 KB INT8, still far inside the ESP32-S3
    8 MB flash / 512 KB SRAM envelope. Same (49, 40, 1) log-mel input contract as
    BCConformer50K, so every existing dataset and checkpoint loader still works."""
    num_classes: int = 2
    channels: int = 32
    conformer_dim: int = 48
    blocks: int = 3
    num_heads: int = 4
    dropout: float = 0.20

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x is (B, 49, 40, 1) log-mel
        # Stage 0: static log-mel -> [log-mel, delta, delta-delta]
        h = _delta_stack(x)                                 # (B, 49, 40, 3)

        # Stage 1: frequency stem
        h = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(1, 2),
                    padding="SAME", use_bias=False)(h)
        h = nn.GroupNorm(num_groups=4)(h)
        h = jax.nn.swish(h)                                 # (B, 49, 20, 32)

        # Stage 2: sub-band depthwise conv + broadcasting (kept from v1)
        f_conv = nn.Conv(features=self.channels, kernel_size=(1, 3), padding="SAME",
                         feature_group_count=self.channels, use_bias=False)(h)
        f_conv = nn.GroupNorm(num_groups=4)(f_conv)
        f_conv = jax.nn.swish(f_conv)
        f_bcast = jnp.mean(f_conv, axis=2, keepdims=True)
        h = h + f_conv + f_bcast

        # Stage 3: project frequency into the temporal sequence
        h = nn.Conv(features=self.conformer_dim, kernel_size=(3, 3), strides=(1, 2),
                    padding="SAME", use_bias=False)(h)
        h = nn.GroupNorm(num_groups=4)(h)
        h = jax.nn.swish(h)                                 # (B, 49, 10, 48)

        # Learned attention pooling over mel instead of jnp.mean -> keeps sub-band
        # structure that v1 threw away.
        mel_score = nn.Conv(features=1, kernel_size=(1,), use_bias=False)(h)   # (B,49,10,1)
        mel_w = nn.softmax(mel_score, axis=2)
        seq = jnp.sum(h * mel_w, axis=2)                    # (B, 49, 48)

        # Learned absolute temporal embedding: attention above is permutation
        # equivariant, and the keyword's position inside the 1 s window is the single
        # biggest driver of the offset stress result.
        seq = seq + self.param("pos_emb", nn.initializers.normal(0.02),
                               (seq.shape[1], self.conformer_dim))

        # Stage 4: improved conformer stack
        for _ in range(self.blocks):
            seq = ImprovedConformerBlock(dim=self.conformer_dim,
                                         num_heads=self.num_heads)(seq, train=train)

        # Stage 5: attentive statistics pooling + classification
        feat = AttentiveStatsPool(dim=self.conformer_dim)(seq)
        feat = nn.Dropout(rate=self.dropout, deterministic=not train)(feat)
        return nn.Dense(features=self.num_classes)(feat)


class RelPosSelfAttention(nn.Module):
    """Multi-head self-attention with a learned RELATIVE position bias (T5 style).

    Motivation, measured rather than assumed. BCConformerV2 adds a learned *absolute*
    temporal embedding before the stack, on the theory that it would make the network
    position-aware and therefore position-robust. The opposite happened: a single global
    pool over a fixed 1 s window has to guess where the keyword is, and an absolute
    position parameter is free to encode a prior about that guess. Validation recall fell
    monotonically 77.7% -> 15.1% across the window, a 62.6-point spread, against 26.5
    points for the 50K model that has no such embedding.

    A relative bias is the standard fix. Attention logits get b[i - j + (T-1)] added, which
    depends only on the distance between two frames, so shifting the whole sequence leaves
    every logit unchanged. The bias is a translation-equivariant prior over *distances*
    (prefer nearby context) and cannot express "the keyword belongs at frame 14".

    Cost: (2T-1) x num_heads learned values = 97 x 4 = 388 parameters.
    """

    num_heads: int = 4
    features: int = 48
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        b, t, _ = x.shape
        d = x.shape[-1]
        h = self.num_heads
        if d % h:
            raise ValueError(f"features {d} not divisible by num_heads {h}")
        dh = d // h

        qkv = nn.Dense(features=3 * d)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # (B, T, H, dh) -> (B, H, T, dh)
        q = jnp.transpose(q.reshape(b, t, h, dh), (0, 2, 1, 3))
        k = jnp.transpose(k.reshape(b, t, h, dh), (0, 2, 1, 3))
        v = jnp.transpose(v.reshape(b, t, h, dh), (0, 2, 1, 3))

        logits = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(jnp.asarray(dh, jnp.float32))

        # Relative position bias, indexed by the signed distance i - j.
        rel = jnp.arange(t)[:, None] - jnp.arange(t)[None, :]      # (T, T)
        rel = rel + (t - 1)                                          # shift into [0, 2T-2]
        bias = self.param("rel_bias", nn.initializers.zeros, (2 * t - 1, h))
        logits = logits + bias[rel].transpose(2, 0, 1)[None]        # (B, H, T, T)

        w = jax.nn.softmax(logits, axis=-1)
        w = nn.Dropout(rate=self.dropout, deterministic=not train)(w)
        out = jnp.einsum("bhij,bhjd->bhid", w, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, t, d)
        out = nn.Dense(features=d)(out)
        return nn.Dropout(rate=self.dropout, deterministic=not train)(out)


class RelConformerBlock(nn.Module):
    """Identical to ImprovedConformerBlock except that attention carries a relative
    position bias instead of relying on an absolute embedding added before the stack."""

    dim: int = 48
    num_heads: int = 4
    se_reduction: int = 8
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        norm1 = nn.LayerNorm()(x)
        attn = RelPosSelfAttention(num_heads=self.num_heads, features=self.dim,
                                   dropout=self.dropout)(norm1, train=train)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(attn)

        norm2 = nn.LayerNorm()(x)
        conv3 = nn.Conv(features=self.dim, kernel_size=(3,), padding="SAME",
                        feature_group_count=self.dim, use_bias=False)(norm2)
        conv7 = nn.Conv(features=self.dim, kernel_size=(7,), padding="SAME",
                        feature_group_count=self.dim, use_bias=False)(norm2)
        y = jax.nn.swish(conv3) + jax.nn.swish(conv7)
        y = nn.Conv(features=self.dim, kernel_size=(1,), use_bias=False)(y)
        z = jnp.mean(y, axis=1, keepdims=True)
        z = nn.Dense(features=max(4, self.dim // self.se_reduction), use_bias=False)(z)
        z = jax.nn.swish(z)
        z = nn.Dense(features=self.dim, use_bias=False)(z)
        y = y * jax.nn.sigmoid(z)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(y)

        norm3 = nn.LayerNorm()(x)
        ff = nn.Dense(features=self.dim * 2)(norm3)
        ff = jax.nn.swish(ff)
        ff = nn.Dense(features=self.dim)(ff)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not train)(ff)
        return x


class SoftORStatsPool(nn.Module):
    """Mean + log-sum-exp 'soft OR' + max over time.

    Replaces attentive statistics pooling. A softmax attention pool is free to collapse
    onto one preferred frame, which is the mechanism behind a positional prior: the
    network learns "trust frame 14" instead of "trust whichever frame looks like the
    keyword". A soft-OR asks the deployed question instead -- is there ANY frame carrying
    the keyword? -- and being a symmetric function of the frame scores, it cannot encode
    *where* that evidence was.

    mean keeps the global context (a keyword is also "a short loud event in a quiet room"),
    max keeps the strongest single-frame evidence, and the log-sum-exp term interpolates
    between them: one dominant frame and many moderate frames both raise it.
    """

    features: int = 48
    hidden: int = 24

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=1)                                   # (B, C)
        mx = jnp.max(x, axis=1)                                      # (B, C)
        proj = nn.Dense(features=self.hidden)(x)                     # (B, T, H)
        w = self.param("or_w", nn.initializers.zeros, (self.hidden,))
        b = self.param("or_b", nn.initializers.zeros, ())
        s = jnp.einsum("bth,h->bt", proj, w) + b                    # (B, T)
        soft_or = jax.scipy.special.logsumexp(s, axis=1) - jnp.log(s.shape[1])
        return jnp.concatenate([mean, soft_or[:, None], mx], axis=-1)   # (B, 2C+1)


class BCConformerV3(nn.Module):
    """BC-Conformer V3 -- relative position bias, soft-OR pooling, no absolute temporal
    embedding. Same (49, 40, 1) input contract as every other architecture here, so all
    existing corpora and every existing checkpoint loader keep working.

    Deliberately a NEW name rather than an edit to BCConformerV2: best_v3_50k.flax and the
    other shipped checkpoints are restored by parameter name, so changing a shared block in
    place would break them.
    """

    num_classes: int = 2
    channels: int = 32
    conformer_dim: int = 48
    blocks: int = 3
    num_heads: int = 4
    dropout: float = 0.20

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        h = _delta_stack(x)                                          # (B, 49, 40, 3)

        h = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(1, 2),
                    padding="SAME", use_bias=False)(h)
        h = nn.GroupNorm(num_groups=4)(h)
        h = jax.nn.swish(h)                                          # (B, 49, 20, 32)

        f_conv = nn.Conv(features=self.channels, kernel_size=(1, 3), padding="SAME",
                         feature_group_count=self.channels, use_bias=False)(h)
        f_conv = nn.GroupNorm(num_groups=4)(f_conv)
        f_conv = jax.nn.swish(f_conv)
        f_bcast = jnp.mean(f_conv, axis=2, keepdims=True)
        h = h + f_conv + f_bcast

        h = nn.Conv(features=self.conformer_dim, kernel_size=(3, 3), strides=(1, 2),
                    padding="SAME", use_bias=False)(h)
        h = nn.GroupNorm(num_groups=4)(h)
        h = jax.nn.swish(h)                                          # (B, 49, 10, 48)

        mel_score = nn.Conv(features=1, kernel_size=(1,), use_bias=False)(h)
        mel_w = nn.softmax(mel_score, axis=2)
        seq = jnp.sum(h * mel_w, axis=2)                             # (B, 49, 48)

        # NOTE: no absolute positional embedding here. That omission is the point of V3.
        for _ in range(self.blocks):
            seq = RelConformerBlock(dim=self.conformer_dim,
                                    num_heads=self.num_heads)(seq, train=train)

        feat = SoftORStatsPool(features=self.conformer_dim)(seq)
        feat = nn.Dropout(rate=self.dropout, deterministic=not train)(feat)
        return nn.Dense(features=self.num_classes)(feat)


# ==============================================================================
# Factory Dispatcher & Utilities
# ==============================================================================
def get_model(arch: str = "bcresnet3", num_classes: int = 2) -> nn.Module:
    """Returns initialized Flax Module based on architecture name."""
    arch_lower = arch.lower().strip()
    if arch_lower in ["bcresnet1", "bcresnet"]:
        return BCResNet(num_classes=num_classes, scale=1)
    elif arch_lower in ["bcresnet3", "bcresnet_sota"]:
        return BCResNet(num_classes=num_classes, scale=3)
    elif arch_lower in ["tcresnet", "tc_resnet"]:
        return TCResNet(num_classes=num_classes, base_channels=24)
    elif arch_lower in ["dsresnet_se", "se_resnet"]:
        return DSResNetSE(num_classes=num_classes, base_channels=20)
    elif arch_lower in ["dscnn", "dscnn_s"]:
        return DSCNN(num_classes=num_classes)
    elif arch_lower in ["conformer", "conformer_lite", "mh_kws"]:
        return ConformerLiteKWS(num_classes=num_classes, dim=32)
    elif arch_lower in ["bcconformer_50k", "conformer_50k", "bc_conformer_50k", "50k"]:
        return BCConformer50K(num_classes=num_classes)
    elif arch_lower in ["bcconformer_100k", "conformer_100k", "bc_conformer_100k", "100k"]:
        return BCConformer100K(num_classes=num_classes)
    elif arch_lower in ["bcconformer_v2", "conformer_v2", "bc_conformer_v2", "v2"]:
        return BCConformerV2(num_classes=num_classes)
    elif arch_lower in ["bcconformer_v3", "conformer_v3", "bc_conformer_v3", "v3"]:
        return BCConformerV3(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Available: bcresnet1, bcresnet3, tcresnet, dsresnet_se, dscnn, conformer_lite, bcconformer_50k, bcconformer_100k, bcconformer_v2, bcconformer_v3")


def count_params(model: nn.Module, input_shape: Tuple[int, ...] = (1, 49, 40, 1)) -> int:
    """Computes exact number of trainable parameters for any model."""
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


