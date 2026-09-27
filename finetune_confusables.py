"""
Fine-Tune Existing Checkpoint on Hard Phonetic Confusables
----------------------------------------------------------
Loads best_50k_kws_model.flax (already trained on dataset_mega.npz with 99.28% AUC)
and fine-tunes on the combined mega + hard confusables dataset with:
- Low learning rate (1/10th of original) to preserve learned features
- Reduced focal gamma (1.5 instead of 2.0) to balance recall vs precision
- Positive-weighted loss to combat the 3:1 neg:pos imbalance
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
from sklearn.metrics import roc_auc_score

from models import get_model, count_params


def apply_spec_augment(batch_x: np.ndarray, prob: float = 0.40) -> np.ndarray:
    b, t, m, c = batch_x.shape
    augmented = batch_x.copy()
    for i in range(b):
        if np.random.random() < prob:
            f_w = np.random.randint(1, 5)
            f_st = np.random.randint(0, m - f_w)
            augmented[i, :, f_st:f_st + f_w, :] = np.min(augmented[i])
            t_w = np.random.randint(1, 6)
            t_st = np.random.randint(0, t - t_w)
            augmented[i, t_st:t_st + t_w, :, :] = np.min(augmented[i])
    return augmented


@jax.jit
def train_step(state, batch_x, batch_y, rng, class_weights):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, batch_x, train=True, rngs={'dropout': rng}
        )
        probs = jax.nn.softmax(logits, axis=-1)
        p_t = jnp.take_along_axis(probs, jnp.expand_dims(batch_y, axis=-1), axis=-1)[:, 0]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y)
        # Focal loss with gamma=1.5 (softer than 2.0 to preserve recall)
        focal_factor = jnp.power(jnp.clip(1.0 - p_t, 1e-4, 1.0), 1.5)
        sw = class_weights[batch_y]
        loss = jnp.mean(ce * focal_factor * sw)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return new_state, loss, acc


@jax.jit
def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn({'params': state.params}, batch_x, train=False)
    probs = jax.nn.softmax(logits, axis=-1)[:, 1]
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return acc, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_50k_kws_model.flax")
    parser.add_argument("--data", type=str, default="dataset_ultra.npz")
    parser.add_argument("--arch", type=str, default="bcconformer_50k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="best_50k_finetuned.flax")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" FINE-TUNING EXISTING CHECKPOINT ON HARD CONFUSABLES", flush=True)
    print("=" * 70, flush=True)

    # Load dataset
    data = np.load(args.data)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_val.shape) == 3:
        X_val = np.expand_dims(X_val, axis=-1)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    print(f" Base Checkpoint   : {args.checkpoint}", flush=True)
    print(f" Dataset           : {args.data}", flush=True)
    print(f" Train Split       : {len(X_train):,} ({n_pos:,} pos, {n_neg:,} neg)", flush=True)
    print(f" Val Split         : {len(X_val):,} ({np.sum(y_val==1):,} pos, {np.sum(y_val==0):,} neg)", flush=True)

    # Load model architecture and checkpoint
    model = get_model(args.arch, num_classes=2)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=False)
    dummy_params = variables['params']

    with open(args.checkpoint, "rb") as f:
        raw = f.read()
    template = {
        "params": dummy_params,
        "arch": args.arch,
        "param_count": 0,
        "epoch": 0,
        "val_auc": 0.0,
        "val_acc": 0.0,
        "tpr": 0.0,
        "fpr": 0.0,
    }
    restored = serialization.from_bytes(template, raw)
    loaded_params = restored["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(loaded_params))
    print(f" Loaded Parameters : {param_count:,} ({param_count/1024:.2f} KB INT8)", flush=True)
    print(f" Base AUC          : {restored.get('val_auc', 0)*100:.2f}%", flush=True)
    print(f" Base TPR          : {restored.get('tpr', 0)*100:.2f}%", flush=True)
    print(f" Base FPR          : {restored.get('fpr', 0)*100:.2f}%", flush=True)

    # Class weights: boost positives since they're outnumbered
    # Use inverse frequency with softening
    pos_w = (n_neg / max(1, n_pos)) * 0.5  # ~1.47
    neg_w = 1.0
    cw = jnp.array([neg_w, pos_w])
    print(f" Fine-Tune LR      : {args.lr}", flush=True)
    print(f" Class Weights     : neg={neg_w:.3f}, pos={pos_w:.3f}", flush=True)
    print(f" Focal Gamma       : 1.5 (softer to preserve recall)", flush=True)

    # Create optimizer with low LR for fine-tuning
    steps_per_epoch = len(X_train) // args.batch_size
    total_steps = args.epochs * steps_per_epoch

    # Cosine decay from peak LR down to 1e-6
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1,
        peak_value=args.lr,
        warmup_steps=max(10, steps_per_epoch),
        decay_steps=total_steps,
        end_value=1e-6
    )
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=loaded_params, tx=tx
    )

    best_auc = 0.0
    best_tpr = 0.0
    best_fpr = 1.0
    best_acc = 0.0

    print("-" * 70, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Acc':^9} | {'TPR':^10} | {'FPR':^10} | {'AUC':^8}", flush=True)
    print("-" * 70, flush=True)

    rng = jax.random.PRNGKey(42)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        train_losses, train_accs = [], []
        for step in range(steps_per_epoch):
            bx = X_shuf[step * args.batch_size:(step + 1) * args.batch_size]
            by = y_shuf[step * args.batch_size:(step + 1) * args.batch_size]
            bx = apply_spec_augment(bx, prob=0.40)
            rng, step_rng = jax.random.split(rng)
            state, loss, acc = train_step(state, jnp.array(bx), jnp.array(by), step_rng, cw)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # Validation
        val_probs_all = []
        val_steps = int(np.ceil(len(X_val) / args.batch_size))
        for vs in range(val_steps):
            vbx = X_val[vs * args.batch_size:(vs + 1) * args.batch_size]
            vby = y_val[vs * args.batch_size:(vs + 1) * args.batch_size]
            v_acc, v_probs = eval_step(state, jnp.array(vbx), jnp.array(vby))
            val_probs_all.extend(np.array(v_probs))

        val_probs_all = np.array(val_probs_all)
        val_preds = (val_probs_all >= 0.85).astype(np.int32)
        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)
        tpr = np.mean(val_preds[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
        fpr = np.mean(val_preds[neg_mask] == 1) if np.sum(neg_mask) > 0 else 0.0
        val_acc = np.mean(val_preds == y_val)

        try:
            auc = roc_auc_score(y_val, val_probs_all)
        except Exception:
            auc = 0.50

        mean_loss = np.mean(train_losses)
        mean_acc = np.mean(train_accs)

        marker = ""
        # Save if: AUC improved AND TPR >= 30% (we want recall!)
        if auc > best_auc and tpr >= 0.30:
            best_auc = auc
            best_tpr = tpr
            best_fpr = fpr
            best_acc = val_acc
            marker = " *"

            bytes_out = serialization.to_bytes({
                'params': state.params,
                'arch': args.arch,
                'param_count': param_count,
                'epoch': ep,
                'val_auc': float(auc),
                'val_acc': float(val_acc),
                'tpr': float(tpr),
                'fpr': float(fpr),
            })
            with open(args.save, "wb") as f:
                f.write(bytes_out)

        print(f" {ep:4d}   |  {mean_loss:9.4f}  |   {mean_acc*100:6.2f}%   |  {val_acc*100:6.2f}%  |   {tpr*100:6.2f}%   |   {fpr*100:6.2f}%   |  {auc*100:5.2f}%{marker}", flush=True)

    elapsed = time.time() - t0
    print("-" * 70, flush=True)
    print(f" Fine-Tuning Complete in {elapsed:.1f}s ({elapsed/args.epochs:.2f}s/epoch)", flush=True)
    print(f" Best Checkpoint: {args.save}", flush=True)
    print(f"   AUC: {best_auc*100:.2f}%  TPR: {best_tpr*100:.2f}%  FPR: {best_fpr*100:.2f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
