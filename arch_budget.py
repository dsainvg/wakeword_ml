"""
Architecture budget calculator for the Amaze KWS models
-----------------------------------------------------
Derived from COMPLIANCE_ANALYSIS.md 1: `EE.VMULAS.S8` retires 16 MAC per
instruction, and 240 MHz is 240,000 cycles per millisecond, so
16 * 240,000 = 3,840,000 MAC per millisecond. The arithmetic floor of an int8,
properly vectorised implementation is therefore

    floor_ms = MAC / 3_840_000        (MAC = multiply-accumulate COUNT, not MMAC)

Everything a real firmware pays on top of that floor -- flash reads, per-channel
rescaling, LayerNorm statistics, non-16-multiple tails -- is implementation
overhead. The audit measured 296 ms against a 2.93 ms floor for the current
11.24 MMAC model (~101x overhead), attributing most of it to float32 execution on
an integer-only vector unit.

This script makes the target explicit BEFORE an architecture is chosen, so a
candidate gets rejected on arithmetic instead of discovered on hardware.

    py arch_budget.py
"""

# --- the constraint being designed against (COMPLIANCE_ANALYSIS.md 1, 3a) ---
HOP_MS = 100.0                        # the cadence the model was validated at
DUTY = 0.10                # < 10% CPU while listening
BUDGET_MS = HOP_MS * DUTY / (1 - DUTY)  # infer <= hop/9  -> 11.11 ms
RAM_BUDGET_KB = 256                    # the brief's RAM bound
LAMBDA = 3_840_000.0                # MAC -> ms (16 MAC/instr @ 240 MHz)


def floor_ms(mac: float) -> float:
    """Arithmetic floor in ms. `mac` is a raw multiply-accumulate COUNT."""
    return mac / LAMBDA


def budget_report(name: str, params: int, mac: float, act_bytes: int) -> str:
    f = floor_ms(mac)
    ram_kb = act_bytes / 1024.0
    return (f"{name:<26} params {params:>7,}  {mac / 1e6:>6.3f} MMAC  "
            f"floor {f:>6.3f} ms  headroom {BUDGET_MS / f:>5.1f}x  "
            f"act {ram_kb:>6.1f} KB ({ram_kb / RAM_BUDGET_KB * 100:>5.1f}% of 256 KB)")


if __name__ == "__main__":
    print("=" * 104)
    print(f"Target: inference <= {BUDGET_MS:.2f} ms at a {HOP_MS:.0f} ms hop "
          f"(10% duty).   floor_ms = MAC / {LAMBDA:.0f}")
    print("=" * 104)
    # the current production model, as measured in COMPLIANCE_ANALYSIS.md
    print(budget_report("bcconformer_v3 (current)", 84_865, 11.24e6, 491_947))
    print()
    print("Its 2.93 ms floor already fits the 11.11 ms budget with 3.8x to spare,")
    print("so the 296 ms is NOT the model being too big -- it is ~101x of")
    print("implementation overhead, mostly float32 on an integer-only core.")
    print("A redesign must therefore fix BOTH: fewer MACs AND an int8-friendly,")
    print("flash-friendly topology. Target: floor <= ~0.5 ms, leaving ~20x for")
    print("overhead, which is what a hand-written kernel realistically costs.")
    print()
    print(f"{'candidate':<26} {'params':>8}  {'MMAC':>7}  {'floor':>9}  {'headroom':>9}")
    print("-" * 104)
    for nm, p, m in [
        ("bcresnet1 (v1 baseline)", 5_266, 0.30e6),
        ("bcconformer_50k", 49_882, 3.20e6),
        ("target A: ultra-tight", 18_000, 0.80e6),
        ("target B: tight", 26_000, 1.20e6),
        ("target C: moderate", 38_000, 1.80e6),
    ]:
        print(f"{nm:<26} {p:>8,}  {m / 1e6:>6.3f}M  {floor_ms(m):>8.3f}ms"
              f"  {BUDGET_MS / floor_ms(m):>8.1f}x")
    print()
    print("Per-layer MAC map of the current v3 (11.24 MMAC total), which is what")
    print("the redesign has to beat -- the conformer blocks dominate, not the stem:")
    print()
    stem = 1.05e6
    blocks = 9.10e6
    pool_head = 1.09e6
    tot = stem + blocks + pool_head
    for nm, v in [("delta stack", 0.0), ("stem conv1+conv2", stem),
                  ("3x conformer blocks", blocks), ("soft-OR pool + head", pool_head)]:
        print(f"  {nm:<24} {v / 1e6:>6.3f} MMAC  ({v / tot * 100:>5.1f}%)")
    print(f"  {'TOTAL':<24} {tot / 1e6:>6.3f} MMAC")
    print()
    print("=> the 3 conformer blocks are ~81% of the arithmetic. Cutting to 2 blocks")
    print("   at dim 32 (from 3 at dim 48) is the single biggest available saving,")
    print("   and COMPLIANCE_ANALYSIS.md 6 Option B names exactly that.")