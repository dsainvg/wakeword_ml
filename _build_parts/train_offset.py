def offset_stress_probs(model, xb, device, n_offsets=8, bs=512):
    """Recall when the same clip is rolled to different positions in the window.

    This is the metric that separates a position-robust detector from one that
    learned a prior. v1/v2 lost 62.6 points of recall across the window because
    of an absolute position embedding; v3's relative bias fixed it.
    """
    probs = []
    model.eval()
    t = xb.shape[1]
    with torch.no_grad():
        for off in range(0, t, max(1, t // n_offsets)):
            rolled = torch.roll(xb, shifts=off, dims=1).contiguous()
            for i in range(0, rolled.shape[0], bs):
                p = torch.softmax(model(rolled[i:i + bs].to(device)), -1)[:, 1]
                probs.append(p.float().cpu().numpy())
    model.train()
    return np.concatenate(probs) if probs else np.array([])