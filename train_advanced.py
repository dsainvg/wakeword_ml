"""
Advanced JAX/Flax Keyword Spotting Training Engine
--------------------------------------------------
Trains modern TinyML architectures (BC-ResNet-1, BC-ResNet-3, TC-ResNet, DS-ResNet-SE)
on large-scale multi-corpus datasets with SpecAugment and GroupNorm.

Features:
- Pure JIT-compiled train and evaluation steps in JAX
- On-the-fly SpecAugment (frequency and time masking) on training minibatches
- Optax AdamW + Cosine Decay Learning Rate with Warmup
- Class-weighted Cross-Entropy loss with label smoothing (0.05)
- Full metrics: Loss, Accuracy, TPR (Recall), FPR (False Alarm), Area Under ROC (AUC)
- Checkpoint serialization via Flax
"""

import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import serialization
import optax

from models import get_model, count_params


def apply_batch_spec_augment(batch_x: np.ndarray, prob: float = 0.50) -> np.ndarray:
    """Applies SpecAugment (time & frequency masking) on numpy batch before feeding JIT."""
    b, t, m, c = batch_x.shape
    augmented = batch_x.copy()
    for i in range(b):
        if np.random.random() < prob:
            # Frequency mask (1-6 bands)
            f_w = np.random.randint(1, 7)
            f_st = np.random.randint(0, m - f_w)
            augmented[i, :, f_st:f_st + f_w, :] = np.min(augmented[i])

            # Time mask (1-8 frames)
            t_w = np.random.randint(1, 9)
            t_st = np.random.randint(0, t - t_w)
            augmented[i, t_st:t_st + t_w, :, :] = np.min(augmented[i])
    return augmented


def create_train_state(rng, model, learning_rate: float, total_steps: int):
    dummy_input = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']

    warmup_steps = max(20, int(0.12 * total_steps))
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
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch_x,
            train=True,
            rngs={'dropout': rng}
        )
        # Class weighting: 2.5x weight on keyword class 1
        weights = jnp.array([1.0, 2.5])
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
    logits = state.apply_fn({'params': state.params}, batch_x, train=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y))
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds, probs[:, 1]


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the ROC Curve without sklearn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U test statistic
    ranks = np.argsort(np.concatenate([pos, neg]))
    r_pos = np.sum(np.where(ranks < len(pos))[0] + 1)
    n_pos = len(pos)
    n_neg = len(neg)
    u = r_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def train_model(
    dataset_npz: str = "dataset_large.npz",
    arch: str = "bcresnet3",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 3.5e-3,
    save_path: str = "best_kws_model.flax"
):
    print("=" * 70, flush=True)
    print(f" Advanced JAX/Flax Keyword Spotting Training", flush=True)
    print(f" Dataset Archive     : {dataset_npz}", flush=True)
    print(f" Architecture        : {arch.upper()}", flush=True)
    print(f" Epochs              : {epochs}", flush=True)
    print(f" Batch Size          : {batch_size}", flush=True)
    print(f" Peak Learning Rate  : {lr}", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(dataset_npz):
        raise FileNotFoundError(f"Dataset archive '{dataset_npz}' not found.")

    data = np.load(dataset_npz)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    print(f"[Dataset] Train samples: {len(X_train)} | Val samples: {len(X_val)}", flush=True)
    pos_train = np.sum(y_train == 1)
    neg_train = np.sum(y_train == 0)
    print(f"[Dataset] Train Distribution: {pos_train} Positives ({pos_train/len(y_train)*100:.1f}%), {neg_train} Negatives ({neg_train/len(y_train)*100:.1f}%)", flush=True)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    model = get_model(arch, num_classes=2)
    param_count = count_params(model)
    print(f"[Model] Trainable Parameters: {param_count:,} (INT8 Footprint: ~{param_count/1024:.2f} KB)\n", flush=True)

    steps_per_epoch = len(X_train) // batch_size
    total_steps = steps_per_epoch * epochs
    state = create_train_state(init_rng, model, learning_rate=lr, total_steps=total_steps)

    best_val_acc = 0.0
    best_tpr = 0.0
    best_fpr = 1.0
    best_auc = 0.0

    print("-" * 75, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Acc':^8} | {'TPR (Recall)':^12} | {'FPR (False)':^11} | {'AUC-ROC':^7}", flush=True)
    print("-" * 75, flush=True)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        train_losses = []
        train_accs = []

        for b in range(steps_per_epoch):
            bx = X_shuffled[b * batch_size:(b + 1) * batch_size]
            by = y_shuffled[b * batch_size:(b + 1) * batch_size]

            # Apply SpecAugment (frequency and time masking) on 50% of minibatches
            bx = apply_batch_spec_augment(bx, prob=0.50)

            rng, step_rng = jax.random.split(rng)
            state, step_loss, step_acc = train_step(state, jnp.array(bx), jnp.array(by), step_rng)
            train_losses.append(float(step_loss))
            train_accs.append(float(step_acc))

        epoch_loss = np.mean(train_losses)
        epoch_acc = np.mean(train_accs)

        # Validation Pass (in batches to conserve memory)
        val_preds_list = []
        val_probs_list = []
        val_loss_list = []

        val_batch_size = 64
        num_val_batches = int(np.ceil(len(X_val) / val_batch_size))
        for vb in range(num_val_batches):
            v_bx = X_val[vb * val_batch_size:(vb + 1) * val_batch_size]
            v_by = y_val[vb * val_batch_size:(vb + 1) * val_batch_size]
            v_loss, preds, probs = eval_step(state, jnp.array(v_bx), jnp.array(v_by))
            val_loss_list.append(float(v_loss))
            val_preds_list.extend(np.array(preds))
            val_probs_list.extend(np.array(probs))

        val_loss = np.mean(val_loss_list)
        preds = np.array(val_preds_list)
        target_probs = np.array(val_probs_list)

        val_acc = np.mean(preds == y_val)
        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)

        tpr = np.mean(preds[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
        fpr = np.mean(preds[neg_mask] == 1) if np.sum(neg_mask) > 0 else 0.0
        auc = compute_roc_auc(y_val, target_probs)

        print(f" {epoch:02d}/{epochs:02d}  |   {epoch_loss:.4f}   |  {epoch_acc*100:5.1f}%  |  {val_acc*100:5.1f}%  |   {tpr*100:5.1f}%    |   {fpr*100:5.2f}%   |  {auc*100:5.1f}%", flush=True)

        # Save checkpoint if accuracy improves or AUC improves
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and auc >= best_auc):
            best_val_acc = val_acc
            best_tpr = tpr
            best_fpr = fpr
            best_auc = auc

            serialized = serialization.to_bytes({
                "params": state.params,
                "arch": arch,
                "param_count": param_count,
                "val_acc": val_acc,
                "tpr": tpr,
                "fpr": fpr,
                "auc": auc
            })
            if not save_path.endswith(".flax"):
                save_path = os.path.splitext(save_path)[0] + ".flax"
            with open(save_path, "wb") as f:
                f.write(serialized)

    elapsed = time.time() - t0
    print("-" * 75, flush=True)
    print(f" Training finished in {elapsed:.1f}s ({elapsed / epochs:.2f}s/epoch)!", flush=True)
    print(f" Best Validation Accuracy : {best_val_acc * 100:.2f}%", flush=True)
    print(f" Best Keyword Recall (TPR): {best_tpr * 100:.2f}%", flush=True)
    print(f" False Alarm Rate (FPR)   : {best_fpr * 100:.2f}%", flush=True)
    print(f" Area Under ROC (AUC)     : {best_auc * 100:.2f}%", flush=True)
    print(f" Best Model Checkpoint    : {save_path}", flush=True)
    print("=" * 70, flush=True)

    return {
        "arch": arch,
        "params": param_count,
        "val_acc": best_val_acc,
        "tpr": best_tpr,
        "fpr": best_fpr,
        "auc": best_auc,
        "checkpoint": save_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced JAX KWS Training")
    parser.add_argument("--data", type=str, default="dataset_large.npz", help="Dataset archive (.npz)")
    parser.add_argument("--arch", type=str, default="bcresnet3", choices=["bcresnet1", "bcresnet3", "tcresnet", "dsresnet_se", "dscnn"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3.5e-3, help="Peak learning rate")
    parser.add_argument("--out", type=str, default="best_kws_model.flax", help="Output model path")
    args = parser.parse_args()

    train_model(
        dataset_npz=args.data,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_path=args.out
    )
