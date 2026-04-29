"""Stress-test leading Microchip pair candidates: fill stress, rolling worst slice, day balance."""

from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))
from r5_microchip_analysis import (
    simulate_pair,
    simulate_basket,
    SimResult,
    load_day,
    aligned_timestamps,
    MICROCHIPS,
    RollingStats,
    POSITION_LIMIT,
    TARGET_SIZE,
    MAX_ORDER_SIZE,
)


def fmt(x: float) -> str:
    return f"{x:>12,.0f}"


CANDIDATES = [
    ("OT raw w=1000 z=1.50", ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "raw", 1000, 1.50),
    ("OT raw w=750 z=1.75", ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "raw", 750, 1.75),
    ("OT raw w=1000 z=1.75", ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "raw", 1000, 1.75),
    ("OT log w=1000 z=1.50", ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "log", 1000, 1.50),
]


def neighbourhood(pair, residual_kind, base_w, base_ez, w_grid, ez_grid):
    print(f"\n--- Neighbourhood: pair={pair[0]}/{pair[1]} resid={residual_kind} base w={base_w} z={base_ez} ---")
    print("window  entry_z       total       D2       D3       D4   trades  drawdown")
    for w in w_grid:
        for ez in ez_grid:
            r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez)
            print(
                f"{w:>6} {ez:>6.2f} {fmt(r.total)} {fmt(r.by_day.get(2,0))} {fmt(r.by_day.get(3,0))} {fmt(r.by_day.get(4,0))} {r.trades:>7} {fmt(r.max_drawdown)}"
            )


def stress_test(label, pair, residual_kind, w, ez):
    print(f"\n=== Stress: {label} ===")
    print("offset       total       D2       D3       D4   trades  drawdown")
    for off in (0, 1, 3, 5, 10):
        r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez, fill_offset=off)
        print(
            f"+{off:<5d} {fmt(r.total)} {fmt(r.by_day.get(2,0))} {fmt(r.by_day.get(3,0))} {fmt(r.by_day.get(4,0))} {r.trades:>7} {fmt(r.max_drawdown)}"
        )


def rolling_pnl_curve(pair, residual_kind, w, ez):
    """Rebuild simulation but record per-tick MTM PnL (across all 3 days, concatenated)."""
    A, B = pair
    stats = RollingStats(w)
    target = {A: 0, B: 0}
    position = {A: 0, B: 0}
    cash = {A: 0.0, B: 0.0}
    pnl_curve: list[tuple[int, int, float]] = []  # (day, idx, pnl)

    for day in (2, 3, 4):
        day_data = load_day(day)
        ts_list = aligned_timestamps(day_data)
        idx_a = {s.timestamp: s for s in day_data[A]}
        idx_b = {s.timestamp: s for s in day_data[B]}
        # reset per day
        stats = RollingStats(w)
        target = {A: 0, B: 0}
        position = {A: 0, B: 0}
        cash = {A: 0.0, B: 0.0}

        for i, ts in enumerate(ts_list):
            sa = idx_a.get(ts)
            sb = idx_b.get(ts)
            if not sa or not sb or sa.mid is None or sb.mid is None:
                continue
            if residual_kind == "raw":
                resid = sa.mid - sb.mid
            else:
                resid = (math.log(sa.mid) - math.log(sb.mid)) * 1000.0
            z = stats.z(resid, w, 0.1)
            if z is None:
                next_ta = 0 if stats._n < w else target[A]
                next_tb = 0 if stats._n < w else target[B]
            else:
                if z > ez:
                    next_ta, next_tb = -TARGET_SIZE, +TARGET_SIZE
                elif z < -ez:
                    next_ta, next_tb = +TARGET_SIZE, -TARGET_SIZE
                else:
                    next_ta, next_tb = target[A], target[B]
            for product, snap, t_target in ((A, sa, next_ta), (B, sb, next_tb)):
                desired = t_target - position[product]
                if desired == 0:
                    continue
                if desired > 0:
                    if snap.best_ask is None:
                        continue
                    qty = min(desired, max(0, snap.ask_volume), max(0, POSITION_LIMIT - position[product]), MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[product] -= qty * snap.best_ask
                    position[product] += qty
                else:
                    if snap.best_bid is None:
                        continue
                    qty = min(-desired, max(0, snap.bid_volume), max(0, POSITION_LIMIT + position[product]), MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[product] += qty * snap.best_bid
                    position[product] -= qty
            target[A], target[B] = next_ta, next_tb
            stats.push(resid)
            mtm = cash[A] + position[A] * sa.mid + cash[B] + position[B] * sb.mid
            pnl_curve.append((day, i, mtm))

    return pnl_curve


def worst_slice(pnl_curve: list[tuple[int, int, float]], slice_len: int):
    """Find worst rolling slice across the *concatenated* PnL series."""
    if not pnl_curve:
        return None
    # Within-day curves only (so we don't mix end-of-day jump with start-of-day reset)
    by_day: dict[int, list[float]] = {}
    for d, _, p in pnl_curve:
        by_day.setdefault(d, []).append(p)
    worst = float("inf")
    where = None
    for d, vals in by_day.items():
        if len(vals) < slice_len:
            continue
        for i in range(len(vals) - slice_len + 1):
            delta = vals[i + slice_len - 1] - vals[i]
            if delta < worst:
                worst = delta
                where = (d, i, i + slice_len - 1)
    return worst, where


def main():
    # Step 1: stress + neighbourhood for the leading candidate
    leader = CANDIDATES[0]
    label, pair, kind, w, ez = leader
    stress_test(label, pair, kind, w, ez)

    print("\n--- Stress for runner-up candidates ---")
    for c in CANDIDATES[1:]:
        stress_test(c[0], c[1], c[2], c[3], c[4])

    # Neighbourhood around (w=1000, z=1.50): w in {750, 1000, 1250}, ez in {1.25, 1.50, 1.75}
    neighbourhood(("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "raw", 1000, 1.50, [750, 1000, 1250], [1.25, 1.50, 1.75])

    # Rolling slice for leader
    print("\n--- Rolling-slice diagnostics for leader (within-day) ---")
    curve = rolling_pnl_curve(pair, kind, w, ez)
    for slc in (1000, 2000):
        worst, where = worst_slice(curve, slc)
        print(f"  slice_len={slc}: worst delta = {worst:,.0f} where={where}")

    # Forced flatten: replay full sim, then forcibly close at last tick best bid/ask (worse-side cross)
    # Already mark-to-mid at end of day in by_day. Flatten close at best opposite-side: subtract spread cost.
    print("\n--- Forced flatten cost vs. mark-to-mid ---")
    for c in CANDIDATES[:3]:
        label, pair, kind, w, ez = c
        r = simulate_pair(pair, residual_kind=kind, window=w, min_history=w, entry_z=ez)
        # crude flatten cost: position * 0.5 spread per product on last tick
        flatten_cost = 0.0
        for d in (2, 3, 4):
            day_data = load_day(d)
            for p in pair:
                last = day_data[p][-1]
                if last.best_bid is None or last.best_ask is None:
                    continue
                pos = r.final_position.get(p, 0) if d == 4 else 0
                if pos > 0:
                    flatten_cost += pos * (last.mid - last.best_bid)
                elif pos < 0:
                    flatten_cost += (-pos) * (last.best_ask - last.mid)
        print(f"  {label}: total={r.total:,.0f} flatten_cost~={flatten_cost:,.0f} flat_total={r.total - flatten_cost:,.0f}")


if __name__ == "__main__":
    main()
