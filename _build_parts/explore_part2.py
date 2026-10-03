def measure(name, cfg):
    m = TimedKWSNet(**cfg)
    params = sum(p.numel() for p in m.parameters())
    mac = MT.count_macs(m)
    with torch.no_grad():
        y = m(torch.zeros(1, *INPUT_SHAPE))
    ok = tuple(y.shape) == (1, 2)
    return {"name": name, "params": params, "mac": mac,
            "floor": mac / MAC_PER_MS, "ok": ok, "cfg": cfg,
            "int8_kb": params / 1024.0}


def sweep():
    cands = []

    # --- family A: wide & shallow. Max params per MAC. ---------------------
    for dim, nb, ts in itertools.product((64, 96, 128), (2, 3), (1, 2)):
        cands.append(("A_wide_d%d_b%d_ts%d" % (dim, nb, ts),
                      dict(dim=dim, num_blocks=nb, time_stride=ts,
                           stem_channels=32, ffn_mult=2.0, num_heads=4)))

    # --- family B: deep & narrow. More nonlinearity, still time-reduced. ---
    for dim, nb, ts in itertools.product((48, 64), (4, 5, 6), (1, 2)):
        cands.append(("B_deep_d%d_b%d_ts%d" % (dim, nb, ts),
                      dict(dim=dim, num_blocks=nb, time_stride=ts,
                           stem_channels=32, ffn_mult=2.0, num_heads=4)))

    # --- family C: wide FFN (params are cheap in FFN when T is small) ------
    for dim, nb, ff in itertools.product((64, 80), (3, 4), (3.0, 4.0)):
        cands.append(("C_wideffn_d%d_b%d_f%.0f" % (dim, nb, ff),
                      dict(dim=dim, num_blocks=nb, ffn_mult=ff, time_stride=2,
                           stem_channels=32, num_heads=4)))

    return [measure(n, c) for n, c in cands]


def main():
    emit("=" * 96)
    emit("TARGETS: params > %d (v3)   MAC < 5.0 MMAC   floor < %.2f ms (v3)"
         % (V3_PARAMS, V3_MMAC * 1e6 / MAC_PER_MS))
    emit("v3 reference: 84,865 params, %.2f MMAC, %.2f ms floor, 480 KB RAM"
         % (V3_MMAC, V3_MMAC * 1e6 / MAC_PER_MS))
    emit("=" * 96)
    emit("")
    res = sweep()

    ok = [r for r in res if r["ok"]]
    fits = [r for r in ok if r["params"] > V3_PARAMS and r["mac"] < 5.0e6]
    emit("swept %d configs; %d ran; %d satisfy params>%d AND <5 MMAC"
         % (len(res), len(ok), len(fits), V3_PARAMS))
    emit("")

    if fits:
        fits.sort(key=lambda r: r["mac"])
        emit("CANDIDATES THAT MEET BOTH (sorted by MAC, i.e. fastest first)")
        emit("%-24s %8s %9s %8s %9s" % ("name", "params", "MMAC", "floor", "int8 KB"))
        emit("-" * 64)
        for r in fits[:12]:
            emit("%-24s %8d %8.3fM %7.3fms %8.1f"
                 % (r["name"], r["params"], r["mac"] / 1e6, r["floor"], r["int8_kb"]))
    else:
        emit("NO config satisfied both. Widest options under 5 MMAC:")
        under = sorted([r for r in ok if r["mac"] < 5.0e6], key=lambda r: -r["params"])
        emit("%-24s %8s %9s %8s" % ("name", "params", "MMAC", "floor"))
        for r in under[:12]:
            emit("%-24s %8d %8.3fM %7.3fms" % (r["name"], r["params"],
                                               r["mac"] / 1e6, r["floor"]))
        emit("")
        emit("=> v3's 84,865 params are NOT reachable under 5 MMAC with this")
        emit("   block family. The FFN/attention widths that cost params also cost")
        emit("   MACs unless time is cut harder. See the manual table below.")

    emit("")
    emit("MAC vs TIME, holding width fixed (why time is the lever):")
    for ts in (1, 2, 4):
        r = measure("probe_d64_b3_ts%d" % ts,
                    dict(dim=64, num_blocks=3, time_stride=ts, stem_channels=32))
        emit("  time_stride=%d -> T=%2d  params %6d  MAC %6.3fM  floor %.3f ms"
             % (ts, 49 // ts, r["params"], r["mac"] / 1e6, r["floor"]))

    open("explore.out", "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    main()