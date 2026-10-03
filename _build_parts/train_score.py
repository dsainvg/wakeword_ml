    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device)
        p = torch.softmax(model(xb), -1)[:, 1]
        probs[i:i + bs] = p.float().cpu().numpy()
    model.train()
    return probs

torch.manual_seed(args.seed)
