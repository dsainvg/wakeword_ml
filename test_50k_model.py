import jax
import jax.numpy as jnp
from flax import linen as nn

class TemporalConformerBlock(nn.Module):
    dim: int = 56
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

class BCConformerHybrid(nn.Module):
    num_classes: int = 2
    channels: int = 40
    conformer_dim: int = 56

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

        # Stage 4: 2x Conformer Blocks
        seq = TemporalConformerBlock(dim=self.conformer_dim, num_heads=4)(seq, train=train)
        seq = TemporalConformerBlock(dim=self.conformer_dim, num_heads=4)(seq, train=train)

        # Stage 5: Dual Pooling & Classification
        pooled = jnp.mean(seq, axis=1)
        max_pooled = jnp.max(seq, axis=1)
        feat = jnp.concatenate([pooled, max_pooled], axis=-1)
        feat = nn.Dropout(rate=0.20, deterministic=not train)(feat)
        logits = nn.Dense(features=self.num_classes)(feat)
        return logits

if __name__ == "__main__":
    m = BCConformerHybrid()
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((2, 49, 40, 1), dtype=jnp.float32)
    params = m.init(rng, dummy, train=False)['params']
    n = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"BCConformerHybrid Parameters: {n:,} ({n/1024:.2f} KB INT8 Flash footprint)")
    out = m.apply({'params': params}, dummy, train=False)
    print(f"Forward pass output shape: {out.shape}")
