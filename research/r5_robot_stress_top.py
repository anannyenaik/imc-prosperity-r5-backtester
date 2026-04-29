"""Stress + neighbourhood + attribution diagnostics for top LAUNDRY/VACUUMING candidates."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "research")

from research.r5_robot_analysis import (  # type: ignore
    simulate_pair,
    DAYS,
)


def fmt(x: float) -> str:
    return f"{x:>10,.0f}"


CANDIDATES = [
    ("LV raw w=2250 z=2.00", ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw", 2250, 2.00),
    ("LV raw w=2500 z=1.75", ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw", 2500, 1.75),
    ("LV raw w=2000 z=2.25", ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw", 2000, 2.25),
    ("LV raw w=2750 z=1.50", ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw", 2750, 1.50),
    ("LV raw w=1750 z=2.50", ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw", 1750, 2.50),
]


def stress_block(label, pair, kind, w, ez):
    print(f"\n=== Stress: {label} ===")
    print("offset      total        D2        D3        D4   trades       DD  flat_total  pos_A pos_B units")
    for off in (0, 1, 3, 5, 10):
        r = simulate_pair(pair, residual_kind=kind, window=w, min_history=w, entry_z=ez, fill_offset=off)
        rf = simulate_pair(pair, residual_kind=kind, window=w, min_history=w, entry_z=ez, fill_offset=off, forced_flatten=True)
        d2 = r.by_day.get(2, 0)
        d3 = r.by_day.get(3, 0)
        d4 = r.by_day.get(4, 0)
        print(
            f"+{off:<5d} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)} {fmt(rf.total)} "
            f"{r.final_position.get(pair[0],0):>5} {r.final_position.get(pair[1],0):>5} {r.units_traded:>6}"
        )


def attribution(label, pair, kind, w, ez):
    print(f"\n=== Attribution: {label} ===")
    r = simulate_pair(pair, residual_kind=kind, window=w, min_history=w, entry_z=ez)
    for p, v in r.by_product.items():
        print(f"  {p:<22} {v:>12,.0f}")
    print(f"  trades={r.trades} units={r.units_traded} pos_dwell_max={r.pos_max_dwell}")
    print(f"  final positions: {r.final_position}")


def neighbourhood(label, pair, kind, w0, z0):
    print(f"\n=== Parameter neighbourhood: {label} (around w={w0}, z={z0}) ===")
    print("window  entry_z       total       D2       D3       D4")
    for w in (w0 - 250, w0 - 125, w0, w0 + 125, w0 + 250):
        for ez in (z0 - 0.25, z0 - 0.125, z0, z0 + 0.125, z0 + 0.25):
            r = simulate_pair(pair, residual_kind=kind, window=w, min_history=w, entry_z=ez)
            print(f"{w:>6} {ez:>7.3f} {fmt(r.total)} {fmt(r.by_day.get(2,0))} {fmt(r.by_day.get(3,0))} {fmt(r.by_day.get(4,0))}")


def rolling_slice(label, pair, kind, w, ez, slc=(1000, 2000)):
    """Within-day worst slice diagnostic."""
    A, B = pair
    from research.r5_robot_analysis import RollingStats, load_day, aligned_timestamps, POSITION_LIMIT, MAX_ORDER_SIZE, TARGET_SIZE  # type: ignore
    import math
    print(f"\n=== Rolling slice (within-day, no fill_offset): {label} ===")
    for slen in slc:
        worst = float("inf")
        where = None
        for day in DAYS:
            day_data = load_day(day)
            ts_list = aligned_timestamps(day_data, (A, B))
            idx_a = {s.timestamp: s for s in day_data[A]}
            idx_b = {s.timestamp: s for s in day_data[B]}
            stats = RollingStats(w)
            target = {A: 0, B: 0}
            position = {A: 0, B: 0}
            cash = {A: 0.0, B: 0.0}
            curve: list[float] = []
            for ts in ts_list:
                sa, sb = idx_a.get(ts), idx_b.get(ts)
                if not sa or not sb or sa.mid is None or sb.mid is None:
                    continue
                if kind == "raw":
                    resid = sa.mid - sb.mid
                else:
                    resid = (math.log(sa.mid) - math.log(sb.mid)) * 1000.0
                z = stats.z(resid, w, 0.1)
                if z is None:
                    nta = 0 if stats._n < w else target[A]
                    ntb = 0 if stats._n < w else target[B]
                else:
                    if z > ez:
                        nta, ntb = -TARGET_SIZE, TARGET_SIZE
                    elif z < -ez:
                        nta, ntb = TARGET_SIZE, -TARGET_SIZE
                    else:
                        nta, ntb = target[A], target[B]
                for prod, snap, t in ((A, sa, nta), (B, sb, ntb)):
                    d = t - position[prod]
                    if d > 0 and snap.best_ask is not None:
                        q = min(d, max(0, snap.ask_volume), max(0, POSITION_LIMIT - position[prod]), MAX_ORDER_SIZE)
                        if q > 0:
                            cash[prod] -= q * snap.best_ask
                            position[prod] += q
                    elif d < 0 and snap.best_bid is not None:
                        q = min(-d, max(0, snap.bid_volume), max(0, POSITION_LIMIT + position[prod]), MAX_ORDER_SIZE)
                        if q > 0:
                            cash[prod] += q * snap.best_bid
                            position[prod] -= q
                target[A], target[B] = nta, ntb
                stats.push(resid)
                curve.append(cash[A] + position[A] * sa.mid + cash[B] + position[B] * sb.mid)
            for i in range(len(curve) - slen + 1):
                delta = curve[i + slen - 1] - curve[i]
                if delta < worst:
                    worst = delta
                    where = (day, i, i + slen - 1)
        print(f"  slice={slen}: worst delta = {worst:,.0f}  where={where}")


def main():
    for label, pair, kind, w, ez in CANDIDATES:
        stress_block(label, pair, kind, w, ez)
        attribution(label, pair, kind, w, ez)

    # Neighbourhood + rolling slice for the leader only (we'll pick the leader after stress).
    leader = CANDIDATES[2]  # LV w=2000 z=2.25 has best day balance — and we'll see vs others
    neighbourhood(*leader)
    rolling_slice(*leader)

    # Also for the user's preferred local result
    primary = CANDIDATES[0]  # LV w=2250 z=2.00
    neighbourhood(*primary)
    rolling_slice(*primary)


if __name__ == "__main__":
    main()
