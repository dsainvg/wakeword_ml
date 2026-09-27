"""
Merge mega-corpus with hard phonetic confusables to produce dataset_ultra.npz
"""
import numpy as np

def main():
    mega = np.load("dataset_mega.npz")
    hard = np.load("dataset_hard_confusables.npz")

    X_train_mega = mega["X_train"]
    y_train_mega = mega["y_train"]
    X_val_mega = mega["X_val"]
    y_val_mega = mega["y_val"]

    X_hard = hard["X"]
    if len(X_hard.shape) == 3:
        X_hard = np.expand_dims(X_hard, axis=-1)
    y_hard = hard["y"]

    np.random.seed(42)
    indices = np.random.permutation(len(X_hard))
    split = int(0.80 * len(X_hard))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train_ultra = np.concatenate([X_train_mega, X_hard[train_idx]], axis=0)
    y_train_ultra = np.concatenate([y_train_mega, y_hard[train_idx]], axis=0)

    X_val_ultra = np.concatenate([X_val_mega, X_hard[val_idx]], axis=0)
    y_val_ultra = np.concatenate([y_val_mega, y_hard[val_idx]], axis=0)

    # Shuffle training set
    tr_perm = np.random.permutation(len(X_train_ultra))
    X_train_ultra = X_train_ultra[tr_perm]
    y_train_ultra = y_train_ultra[tr_perm]

    np.savez_compressed(
        "dataset_ultra.npz",
        X_train=X_train_ultra,
        y_train=y_train_ultra,
        X_val=X_val_ultra,
        y_val=y_val_ultra
    )

    print(f"Created dataset_ultra.npz:")
    print(f"  Train: {len(X_train_ultra)} (Pos: {np.sum(y_train_ultra==1)}, Neg: {np.sum(y_train_ultra==0)})")
    print(f"  Val  : {len(X_val_ultra)} (Pos: {np.sum(y_val_ultra==1)}, Neg: {np.sum(y_val_ultra==0)})")

if __name__ == "__main__":
    main()
