def resolve_arch(name: str):
    """Build a model for `name`, accepting every preset family.

    `kws_*` -> models_torch (RAM/CPU-floor presets)
    `hc_*`, `m1_*` -> models_torch_hc (high-capacity and 1.5 MMAC tiers)

    Note the fallback behaviour: models_torch.get_torch_model warns and returns
    'kws_tight' for an unknown name. Silently training the wrong architecture
    under the requested name is worse than failing, so unknown names raise here
    instead. That bug already cost one run (m1_a trained as kws_tight).
    """
    if name.startswith("m1_") or name.startswith("hc_"):
        from models_torch_hc import ARCHS_HC, ARCHS_M1, get_hc_model
        if name not in ARCHS_HC and name not in ARCHS_M1:
            raise ValueError(f"unknown arch '{name}'. "
                             f"Available: {sorted(ARCHS_HC) + sorted(ARCHS_M1)}")
        return get_hc_model(name)
    from models_torch import get_torch_model, ARCHS
    if name not in ARCHS and name not in ("kwsnet", "kwsnet_tiny"):
        raise ValueError(f"unknown arch '{name}'. Available: {sorted(ARCHS)}")
    return get_torch_model(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v6.npz")
    ap.add_argument("--arch", default="kws_tight")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=1.5, help="focal gamma")
    ap.add_argument("--pos_bias", type=float, default=1.15)
    ap.add_argument("--fpr_budget", type=float, default=0.005)
    ap.add_argument("--hard_neg_k", type=int, default=64)
    ap.add_argument("--spec_aug_prob", type=float, default=0.35)
    ap.add_argument("--gain_prob", type=float, default=0.6)
    ap.add_argument("--roll_prob", type=float, default=0.5,
                    help="probability of a random TIME ROLL per training sample. "
                         "The direct attack on offset fragility: makes every "
                         "position in the 1 s window equally likely to carry the "
                         "keyword, which is what the deployed case looks like.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_memmap", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78, flush=True)
    print(f"data {args.data}  arch {args.arch}  epochs {args.epochs} "
          f"batch {args.batch}  device {device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB",
              flush=True)
    print("=" * 78, flush=True)

    Xtr, Xva, data = load_corpus(args.data, use_memmap=not args.no_memmap)
    ytr = np.asarray(data["y_train"])
    yva = np.asarray(data["y_val"])
    print(f"train {Xtr.shape} {Xtr.dtype}  pos {int((ytr==1).sum()):,} "
          f"neg {int((ytr==0).sum()):,}", flush=True)
    print(f"val   {Xva.shape} {Xva.dtype}  pos {int((yva==1).sum()):,} "
          f"neg {int((yva==0).sum()):,}", flush=True)

    btr = data["b_train"] if "b_train" in data.files else None
    if btr is not None:
        print(f"buckets: {len(np.unique(btr))} distinct", flush=True)

    model = resolve_arch(args.arch).to(device)
    mac = count_macs(model)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {nparam:,} params  {mac:,} MAC  "
          f"floor {mac/3_840_000:.3f} ms  "
          f"({11.11/(mac/3_840_000):.1f}x headroom in 11.11 ms)", flush=True)
    print(flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    nsteps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=nsteps, pct_start=0.15)

    sw = torch.tensor([args.pos_bias, 1.0], device=device, dtype=torch.float32)
    bw_tr = None
    if btr is not None:
        bw_tr = torch.tensor(
            [BUCKET_LOSS_WEIGHT.get(str(b), 1.0) for b in btr],
            device=device, dtype=torch.float32)