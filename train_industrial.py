"""
Industrial KWS Training Engine with Focal Loss & SpecAugment
------------------------------------------------------------
Trains TinyML wake-word detection architectures with:
1. Focal Loss (gamma=2.0) to aggressively penalize hard false positives
2. SpecAugment (Time & Frequency dynamic masking)
3. Cosine Annealing Learning Rate with Warmup
4. Support for all model architectures:
   - BC-ResNet-3 (Broadcasted ResNet)
   - Conformer-Lite (Multi-Head Temporal Attention)
   - DS-ResNet-SE (Squeeze-and-Excitation Channel Attention)
   - TC-ResNet-8 (1D Temporal Dilated Convolutions)
   - BC-ResNet-1 (Ultra-compact 5.2k params)
5. Comprehensive industrial evaluation:
   - Baseline Accuracy
   - True Positive Rate (Keyword Recall)
   - False Positive Rate (False Alarm Rate)
   - ROC-AUC
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


def apply_spec_augment(batch_x: np.ndarray, prob: float = 0.50) -> np.ndarray:
    """Applies dynamic time and frequency masking on numpy minibatches."""
    b, t, m, c = batch_x.shape
    augmented = batch_x.copy()
    for i in range(b):
        if np.random.random() < prob:
            # Frequency masking (1 to 6 bins)
            f_w = np.random.randint(1, 7)
            f_st = np.random.randint(0, m - f_w)
            augmented[i, :, f_st:f_st + f_w, :] = np.min(augmented[i])

            # Time masking (1 to 8 frames)
            t_w = np.random.randint(1, 9)
            t_st = np.random.randint(0, t - t_w)
            augmented[i, t_st:t_st + t_w, :, :] = np.min(augmented[i])
    return augmented


def create_train_state(rng, model, learning_rate: float, total_steps: int):
    dummy_input = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']

    warmup_steps = max(20, int(0.10 * total_steps))
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
def train_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, rng: jax.random.PRNGKey, gamma: float = 2.0, class_weights: jnp.ndarray = jnp.array([1.0, 1.0])):

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch_x,
            train=True,
            rngs={'dropout': rng}
        )
        # Softmax probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        p_t = jnp.take_along_axis(probs, jnp.expand_dims(batch_y, axis=-1), axis=-1)[:, 0]
        # Cross entropy
        ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y)
        # Focal modulating factor (1 - p_t)^gamma
        focal_factor = jnp.power(jnp.clip(1.0 - p_t, 1e-4, 1.0), gamma)
        # Auto-balanced class weights passed in from dataset ratio
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
def eval_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
    logits = state.apply_fn({'params': state.params}, batch_x, train=False)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_y)
    loss = jnp.mean(ce)
    probs = jax.nn.softmax(logits, axis=-1)[:, 1]
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return loss, acc, probs


def run_training(
    data_path: str = "dataset_super.npz",
    arch: str = "bcresnet3",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "best_kws_model.flax"
):
    print("=" * 70)
    print(f" INDUSTRIAL TRAINING SUITE: ARCHITECTURE '{arch.upper()}'")
    print("=" * 70)
    if not os.path.exists(data_path):
        print(f"[Error] Dataset '{data_path}' not found!")
        return

    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_val.shape) == 3:
        X_val = np.expand_dims(X_val, axis=-1)

    print(f" Dataset Archive   : {data_path}")
    print(f" Train Split       : {len(X_train):,} samples ({np.sum(y_train==1):,} pos, {np.sum(y_train==0):,} neg)")
    print(f" Val Split         : {len(X_val):,} samples ({np.sum(y_val==1):,} pos, {np.sum(y_val==0):,} neg)")

    model = get_model(arch, num_classes=2)
    param_count = count_params(model)
    flash_kb = param_count / 1024.0
    print(f" Model Parameters  : {param_count:,} ({flash_kb:.2f} KB INT8 Flash footprint)")
    # Auto-compute balanced class weights from dataset ratio
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    # neg_weight = ratio * 0.6 so positives still get decent gradient
    neg_w = min(2.5, (n_neg / max(1, n_pos)) * 0.6)
    pos_w = 1.0
    cw_jax = jnp.array([neg_w, pos_w])
    print(f" Training Config   : Epochs={epochs}, BatchSize={batch_size}, PeakLR={lr}, Loss=FocalLoss(gamma=2.0)")
    print(f" Class Weights     : neg={neg_w:.3f}, pos={pos_w:.3f}  (pos:neg ratio = 1:{n_neg/max(1,n_pos):.1f})")

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    steps_per_epoch = len(X_train) // batch_size
    total_steps = epochs * steps_per_epoch
    state = create_train_state(init_rng, model, learning_rate=lr, total_steps=total_steps)

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_fpr = 1.0

    print("-" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Acc':^9} | {'TPR (Recall)':^12} | {'FPR (False)':^12} | {'AUC':^8}")
    print("-" * 70)

    t0_all = time.time()
    for ep in range(1, epochs + 1):
        # Shuffle training set
        perm = np.random.permutation(len(X_train))
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]

        train_losses = []
        train_accs = []

        for step in range(steps_per_epoch):
            bx = X_train_shuf[step * batch_size:(step + 1) * batch_size]
            by = y_train_shuf[step * batch_size:(step + 1) * batch_size]

            # Dynamic SpecAugment
            bx_aug = apply_spec_augment(bx, prob=0.50)

            rng, step_rng = jax.random.split(rng)
            state, t_loss, t_acc = train_step(state, jnp.array(bx_aug), jnp.array(by), step_rng, gamma=2.0, class_weights=cw_jax)
            train_losses.append(float(t_loss))
            train_accs.append(float(t_acc))

        # Evaluate on Validation Set
        val_probs_all = []
        val_accs = []
        val_losses = []
        val_steps = int(np.ceil(len(X_val) / batch_size))

        for vstep in range(val_steps):
            vbx = X_val[vstep * batch_size:(vstep + 1) * batch_size]
            vby = y_val[vstep * batch_size:(vstep + 1) * batch_size]
            v_loss, v_acc, v_probs = eval_step(state, jnp.array(vbx), jnp.array(vby))
            val_losses.append(float(v_loss))
            val_accs.append(float(v_acc))
            val_probs_all.extend(np.array(v_probs))

        val_probs_all = np.array(val_probs_all)
        val_preds_85 = (val_probs_all >= 0.85).astype(np.int32)
        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)

        tpr = np.mean(val_preds_85[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
        fpr = np.mean(val_preds_85[neg_mask] == 1) if np.sum(neg_mask) > 0 else 0.0
        val_acc = np.mean(val_preds_85 == y_val)

        try:
            auc = roc_auc_score(y_val, val_probs_all)
        except Exception:
            auc = 0.50

        mean_tr_loss = np.mean(train_losses)
        mean_tr_acc = np.mean(train_accs)

        marker = ""
        # Checkpointing criteria: high AUC + low FPR + meaningful TPR
        # After epoch 15, require TPR >= 20% so a 0% recall model can't win
        tpr_ok = (ep < 15) or (tpr >= 0.20)
        if tpr_ok and (auc > best_val_auc or (auc >= best_val_auc - 0.005 and fpr < best_val_fpr)) and ep >= 10:
            best_val_auc = auc
            best_val_acc = val_acc
            best_val_fpr = fpr
            marker = " *"

            # Serialize model
            bytes_output = serialization.to_bytes({
                'params': state.params,
                'arch': arch,
                'param_count': param_count,
                'epoch': ep,
                'val_auc': float(auc),
                'val_acc': float(val_acc),
                'tpr': float(tpr),
                'fpr': float(fpr)
            })
            with open(save_path, "wb") as f:
                f.write(bytes_output)

        if ep % 2 == 0 or ep == epochs or marker:
            print(f" {ep:4d}   |  {mean_tr_loss:9.4f}  |   {mean_tr_acc*100:6.2f}%   |  {val_acc*100:6.2f}%  |    {tpr*100:6.2f}%    |    {fpr*100:6.2f}%    |  {auc*100:5.2f}%{marker}", flush=True)

    total_time = time.time() - t0_all
    print("-" * 70, flush=True)
    print(f" Training Complete in {total_time:.1f}s ({total_time / epochs:.2f}s/epoch)", flush=True)
    print(f" Best Checkpoint Saved to : {save_path}", flush=True)
    print(f"   Validation AUC         : {best_val_auc*100:.2f}%", flush=True)
    print(f"   Validation Accuracy    : {best_val_acc*100:.2f}%", flush=True)
    print(f"   False Alarm Rate (FPR) : {best_val_fpr*100:.2f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset_super.npz")
    parser.add_argument("--arch", type=str, default="bcresnet3", choices=["bcresnet1", "bcresnet3", "tcresnet", "dsresnet_se", "dscnn", "conformer_lite", "bcconformer_50k", "conformer_50k"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default="best_kws_model.flax")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save
    )
