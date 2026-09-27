"""
JAX / Flax State-of-the-Art Keyword Spotting Training Pipeline
------------------------------------------------------------
Trains BC-ResNet-1 or DS-CNN-S on custom keyword audio.
Features:
- Pure JIT-compiled training and evaluation with JAX / Flax
- GroupNorm layers for deterministic inference & small-batch stability
- Optax AdamW with Cosine Annealing Learning Rate Schedule
- Class-weighted Cross-Entropy to handle non-keyword speech dominance
- Full metrics suite: Accuracy, Loss, Recall (TPR), False Alarm Rate (FPR)
- Serializes trained weights to compressed numpy checkpoint (.npz)
"""

import os
import sys
import glob
import time
import wave
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from features import AudioFeatureExtractor
from models import BCResNet, DSCNN, get_model, count_params


class AudioDataset:
    """Loads 16kHz WAV files and precomputes 40-band Log-Mel spectrograms."""
    def __init__(self, data_dir: str = "dataset"):
        self.extractor = AudioFeatureExtractor()
        pos_files = glob.glob(os.path.join(data_dir, "positive", "*.wav"))
        neg_files = glob.glob(os.path.join(data_dir, "negative", "*.wav"))

        self.samples = []
        for p in pos_files:
            self.samples.append((p, 1))
        for n in neg_files:
            self.samples.append((n, 0))

        np.random.shuffle(self.samples)
        print(f"[Dataset] Found {len(self.samples)} files ({len(pos_files)} positive, {len(neg_files)} negative).", flush=True)

    def __len__(self):
        return len(self.samples)

    def _read_wav(self, path: str) -> np.ndarray:
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        else:
            audio = audio[:16000]
        return audio

    def get_features(self):
        """Extracts all spectrograms into memory."""
        X = []
        y = []
        for path, label in self.samples:
            audio = self._read_wav(path)
            spec = self.extractor.compute_spectrogram(audio)
            X.append(spec)
            y.append(label)

        X = np.array(X, dtype=np.float32)  # (N, 49, 40)
        X = np.expand_dims(X, axis=-1)     # (N, 49, 40, 1) [Time, Mel, Channel]
        y = np.array(y, dtype=np.int32)
        return X, y


def create_train_state(rng, model, learning_rate: float, total_steps: int):
    """Initializes model parameters and Optax AdamW optimizer."""
    dummy_input = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']

    # Cosine learning rate decay with linear warmup
    warmup_steps = max(10, int(0.15 * total_steps))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def train_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, rng: jax.random.PRNGKey):
    """JIT-compiled single gradient step with class weighting."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch_x,
            train=True,
            rngs={'dropout': rng}
        )

        # Cross-entropy with class weighting (weight wake word class 1 by 2.0x)
        weights = jnp.array([1.0, 2.0])
        sample_weights = weights[batch_y]
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y)
        loss = jnp.mean(ce_loss * sample_weights)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return new_state, loss, acc


@jax.jit
def eval_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    """JIT-compiled evaluation step."""
    logits = state.apply_fn({'params': state.params}, batch_x, train=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y))
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds, probs[:, 1]


def run_training(
    data_dir: str = "dataset",
    arch: str = "bcresnet",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 3e-3,
    save_path: str = "best_kws_model.npz"
):
    print("=====================================================================", flush=True)
    print(f" JAX / Flax TinyML Keyword Spotting Training", flush=True)
    print(f" Model Architecture   : {arch.upper()}", flush=True)
    print(f" Target Epochs        : {epochs}", flush=True)
    print(f" Batch Size           : {batch_size}", flush=True)
    print(f" Peak Learning Rate   : {lr}", flush=True)
    print("=====================================================================", flush=True)

    ds = AudioDataset(data_dir=data_dir)
    if len(ds) == 0:
        print("[Error] No samples found in dataset directory!", flush=True)
        return

    print("[Preprocessing] Extracting 40-band Log-Mel Spectrogram features...", flush=True)
    t0 = time.time()
    X, y = ds.get_features()
    print(f"[Preprocessing] Features extracted in {time.time() - t0:.2f}s. Tensor shape: {X.shape}", flush=True)

    # 80/20 Train/Validation Split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"[Dataset] Train: {len(X_train)} samples | Validation: {len(X_val)} samples", flush=True)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    model = get_model(arch, num_classes=2)
    steps_per_epoch = len(X_train) // batch_size
    total_steps = steps_per_epoch * epochs

    state = create_train_state(init_rng, model, learning_rate=lr, total_steps=total_steps)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[Model] Total parameters: {param_count:,} (Quantized INT8: ~{param_count / 1024:.2f} KB)\n", flush=True)

    best_val_acc = 0.0
    best_tpr = 0.0
    best_fpr = 1.0

    print("---------------------------------------------------------------------", flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Acc':^8} | {'TPR (Recall)':^12} | {'FPR (False)':^11}", flush=True)
    print("---------------------------------------------------------------------", flush=True)

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        train_loss_list = []
        train_acc_list = []

        for b in range(steps_per_epoch):
            batch_x = X_train_shuffled[b * batch_size:(b + 1) * batch_size]
            batch_y = y_train_shuffled[b * batch_size:(b + 1) * batch_size]

            rng, step_rng = jax.random.split(rng)
            state, step_loss, step_acc = train_step(
                state,
                jnp.array(batch_x),
                jnp.array(batch_y),
                step_rng
            )
            train_loss_list.append(float(step_loss))
            train_acc_list.append(float(step_acc))

        epoch_train_loss = np.mean(train_loss_list)
        epoch_train_acc = np.mean(train_acc_list)

        # Validation Pass
        val_loss, preds, target_probs = eval_step(state, jnp.array(X_val), jnp.array(y_val))
        val_loss = float(val_loss)
        preds = np.array(preds)

        val_acc = np.mean(preds == y_val)

        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)

        tpr = np.mean(preds[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
        fpr = np.mean(preds[neg_mask] == 1) if np.sum(neg_mask) > 0 else 0.0

        print(f" {epoch:02d}/{epochs:02d}  |   {epoch_train_loss:.4f}   |  {epoch_train_acc * 100:5.1f}%  |  {val_acc * 100:5.1f}%  |   {tpr * 100:5.1f}%    |   {fpr * 100:5.2f}%", flush=True)

        # Save checkpoint if accuracy improves
        if val_acc > best_val_acc or (val_acc == best_val_acc and tpr >= best_tpr):
            best_val_acc = val_acc
            best_tpr = tpr
            best_fpr = fpr

            # Official Flax serialization: preserves nested module structure perfectly
            from flax import serialization
            serialized_bytes = serialization.to_bytes({
                "params": state.params,
                "arch": arch,
                "param_count": param_count,
                "val_acc": val_acc,
                "tpr": tpr,
                "fpr": fpr
            })
            if not save_path.endswith(".flax"):
                save_path = os.path.splitext(save_path)[0] + ".flax"
            with open(save_path, "wb") as f:
                f.write(serialized_bytes)

    elapsed = time.time() - t_train_start
    print("---------------------------------------------------------------------", flush=True)
    print(f" Training completed in {elapsed:.2f}s ({elapsed / epochs:.3f}s/epoch)!", flush=True)
    print(f" Best Validation Accuracy : {best_val_acc * 100:.1f}%", flush=True)
    print(f" Best Keyword Recall (TPR): {best_tpr * 100:.1f}%", flush=True)
    print(f" False Alarm Rate (FPR)   : {best_fpr * 100:.2f}%", flush=True)
    print(f" Best model checkpoint    : {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX KWS Training Pipeline")
    parser.add_argument("--data", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--arch", type=str, default="bcresnet", choices=["bcresnet", "dscnn"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--out", type=str, default="best_kws_model.flax", help="Output model path")
    args = parser.parse_args()

    run_training(data_dir=args.data, arch=args.arch, epochs=args.epochs, batch_size=args.batch, lr=args.lr, save_path=args.out)
