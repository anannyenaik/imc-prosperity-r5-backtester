"""Inspect Microchip mid-price levels and pair-spread statistics by day."""

import os, sys, math, statistics
sys.path.insert(0, os.path.dirname(__file__))
from r5_microchip_analysis import load_day, MICROCHIPS, aligned_timestamps

PAIRS = (
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"),
    ("MICROCHIP_CIRCLE", "MICROCHIP_RECTANGLE"),
    ("MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"),
    ("MICROCHIP_OVAL", "MICROCHIP_CIRCLE"),
)

def basic_stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "stdev": statistics.pstdev(values),
    }

print("=== Per-day Microchip mid level (mean, stdev, range) ===")
for d in (2, 3, 4):
    day = load_day(d)
    print(f"-- Day {d}")
    for p in MICROCHIPS:
        mids = [s.mid for s in day[p] if s.mid is not None]
        st = basic_stats(mids)
        if st:
            print(f"  {p:30s} n={st['n']:5d}  mean={st['mean']:10.1f}  stdev={st['stdev']:7.2f}  range=[{st['min']:.0f},{st['max']:.0f}]")

print("\n=== Pair raw-spread stats per day ===")
for d in (2, 3, 4):
    day = load_day(d)
    ts = aligned_timestamps(day)
    idx = {p: {s.timestamp: s for s in day[p]} for p in MICROCHIPS}
    print(f"-- Day {d} (n_aligned={len(ts)})")
    for A, B in PAIRS:
        spreads = []
        log_resid = []
        for t in ts:
            sa = idx[A].get(t); sb = idx[B].get(t)
            if sa and sb and sa.mid is not None and sb.mid is not None:
                spreads.append(sa.mid - sb.mid)
                log_resid.append((math.log(sa.mid) - math.log(sb.mid)) * 1000)
        if not spreads:
            continue
        sp = basic_stats(spreads)
        lr = basic_stats(log_resid)
        print(
            f"  {A}/{B}: spread mean={sp['mean']:8.2f} stdev={sp['stdev']:6.2f} range=[{sp['min']:.0f},{sp['max']:.0f}] | log*1000 mean={lr['mean']:8.2f} stdev={lr['stdev']:6.2f}"
        )

print("\n=== Cross-day stability: spread mean shift between days ===")
for A, B in PAIRS:
    means = []
    for d in (2, 3, 4):
        day = load_day(d)
        ts = aligned_timestamps(day)
        idx = {p: {s.timestamp: s for s in day[p]} for p in MICROCHIPS}
        spreads = []
        for t in ts:
            sa = idx[A].get(t); sb = idx[B].get(t)
            if sa and sb and sa.mid is not None and sb.mid is not None:
                spreads.append(sa.mid - sb.mid)
        means.append(sum(spreads) / len(spreads) if spreads else 0)
    print(f"  {A}/{B}: D2_mean={means[0]:.2f} D3_mean={means[1]:.2f} D4_mean={means[2]:.2f} delta_max={max(means)-min(means):.2f}")
