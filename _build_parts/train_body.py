    best = {"tpr": -1.0}
    hist = []
    t_start = time.time()

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(Xtr[idx], dtype=np.float32)).to(device)
            yb = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
            xb = augment(xb, rng, args.spec_aug_prob, args.gain_prob,
                           roll_prob=args.roll_prob)

            logits = model(xb)
            ce = F.cross_entropy(logits, yb, reduction="none",
                                 label_smoothing=0.02)
            pt = torch.softmax(logits, -1).gather(1, yb[:, None]).squeeze(1)
            focal = (1.0 - pt).clamp_min(1e-4).pow(args.gamma)
            per = ce * focal * sw[yb]
            if bw_tr is not None:
                per = per * bw_tr[torch.from_numpy(idx).to(device)]
            if args.hard_neg_k > 0:
                is_neg = yb == 0
                k = min(args.hard_neg_k, int(is_neg.sum()))
                if k > 0:
                    negv = torch.where(is_neg, per,
                                      torch.full_like(per, float("inf")))
                    cut = torch.topk(negv, k, largest=False).values[-1]
                    keep = (yb == 1) | (per >= cut)
                    per = per * keep.float()
            loss = per.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1

        pva = score_all(model, Xva, yva, device)
        pos_p, neg_p = pva[yva == 1], pva[yva == 0]
        auc = auc_score(pos_p, neg_p)
        tpr, thr, fpr = tpr_at_fpr_budget(pos_p, neg_p, args.fpr_budget)
        acc = float(((pva >= thr).astype(np.int32) == yva).mean())

        # position robustness: the same positives rolled across the window
        xpos = torch.from_numpy(np.asarray(Xva[yva == 1][:512], dtype=np.float32))
        op = offset_stress_probs(model, xpos, device)
        os_tpr = float(np.mean(op >= thr)) if len(op) else float("nan")

        hist.append({"epoch": ep, "loss": tot / max(1, nb), "auc": auc,
                     "tpr": tpr, "thr": thr, "fpr": fpr, "acc": acc,
                     "offset_tpr": os_tpr})
        print("ep %2d  loss %.4f  auc %.4f  tpr@%.3f%% %.4f  thr %.4f  "
              "acc %.4f  offset-recall %.4f"
              % (ep, tot / max(1, nb), auc, args.fpr_budget * 100, tpr, thr,
                 acc, os_tpr), flush=True)

        if tpr > best["tpr"]:
            best = {"tpr": tpr, "epoch": ep, "thr": thr, "auc": auc,
                    "fpr": fpr, "offset_tpr": os_tpr, "acc": acc}
            if args.out:
                torch.save({"arch": args.arch, "state_dict": model.state_dict(),
                            "threshold": thr, "epoch": ep, "tpr": tpr,
                            "auc": auc, "mac": mac, "params": nparam}, args.out)

    print(flush=True)
    print("=" * 78)
    print("BEST (by TPR at %.3f%% FPR budget)" % (args.fpr_budget * 100))
    print("  epoch          %d" % best["epoch"])
    print("  TPR            %.4f" % best["tpr"])
    print("  FPR            %.4f" % best["fpr"])
    print("  threshold      %.4f" % best["thr"])
    print("  AUC            %.4f" % best["auc"])
    print("  accuracy@thr   %.4f" % best["acc"])
    if best["offset_tpr"] == best["offset_tpr"]:
        print("  offset recall  %.4f  <- position robustness" % best["offset_tpr"])
    print("  MAC            %d (%.3f ms floor)" % (mac, mac / 3_840_000))
    print("  params         %d" % nparam)
    print("  wall clock     %.1f min" % ((time.time() - t_start) / 60))
    if args.out:
        print("  checkpoint     %s" % args.out)
    print("=" * 78)

    with open("train_result.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "best": best, "history": hist,
                   "mac": mac, "params": nparam,
                   "floor_ms": mac / 3_840_000}, fh, indent=2)
    print("history -> train_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())