"""Partial-fill and delayed-fill sensitivity for the leading Microchip pair candidate.

- partial_fill_cap: cap each fill at K units regardless of visible volume.
- delayed_fill: orders are sent but only fill on the *next* tick at then-best price.
"""

from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
from r5_microchip_analysis import (
    load_day, aligned_timestamps, RollingStats, POSITION_LIMIT, TARGET_SIZE, MAX_ORDER_SIZE
)


def simulate_pair_partial_delay(
    pair, *, window=1000, min_history=1000, entry_z=1.5,
    partial_cap=None, delayed=False, days=(2,3,4)
):
    A, B = pair
    target = {A: 0, B: 0}
    position = {A: 0, B: 0}
    cash = {A: 0.0, B: 0.0}
    by_day = {}
    pending = []  # list of (product, side, qty)

    for day in days:
        day_data = load_day(day)
        ts_list = aligned_timestamps(day_data)
        idx_a = {s.timestamp: s for s in day_data[A]}
        idx_b = {s.timestamp: s for s in day_data[B]}
        stats = RollingStats(window)
        target = {A: 0, B: 0}
        position = {A: 0, B: 0}
        cash = {A: 0.0, B: 0.0}
        pending = []

        for ts in ts_list:
            sa = idx_a.get(ts); sb = idx_b.get(ts)
            if not sa or not sb or sa.mid is None or sb.mid is None:
                continue

            # Process delayed orders first (fill against current tick)
            if delayed and pending:
                still_pending = []
                for product, side, qty in pending:
                    snap = sa if product == A else sb
                    if side == 'buy':
                        if snap.best_ask is None: continue
                        avail = max(0, snap.ask_volume)
                        if partial_cap is not None:
                            avail = min(avail, partial_cap)
                        room = max(0, POSITION_LIMIT - position[product])
                        q = min(qty, avail, room)
                        if q > 0:
                            cash[product] -= q * snap.best_ask
                            position[product] += q
                    else:
                        if snap.best_bid is None: continue
                        avail = max(0, snap.bid_volume)
                        if partial_cap is not None:
                            avail = min(avail, partial_cap)
                        room = max(0, POSITION_LIMIT + position[product])
                        q = min(qty, avail, room)
                        if q > 0:
                            cash[product] += q * snap.best_bid
                            position[product] -= q
                    # Discard rest (no requeue)
                pending = []

            current_resid = (sa.mid - sb.mid)
            current_resid = int(round(current_resid * 10))  # match production scale
            z = stats.z(current_resid, min_history, 1.0 * 10)

            if z is None:
                next_ta = 0 if stats._n < min_history else target[A]
                next_tb = 0 if stats._n < min_history else target[B]
            elif z > entry_z:
                next_ta, next_tb = -TARGET_SIZE, +TARGET_SIZE
            elif z < -entry_z:
                next_ta, next_tb = +TARGET_SIZE, -TARGET_SIZE
            else:
                next_ta, next_tb = target[A], target[B]

            for product, snap, t_target in ((A, sa, next_ta), (B, sb, next_tb)):
                desired = t_target - position[product]
                if desired == 0:
                    continue
                if delayed:
                    # Queue for next tick
                    pending.append((product, 'buy' if desired > 0 else 'sell', abs(desired)))
                    continue
                if desired > 0:
                    if snap.best_ask is None: continue
                    avail = max(0, snap.ask_volume)
                    if partial_cap is not None:
                        avail = min(avail, partial_cap)
                    room = max(0, POSITION_LIMIT - position[product])
                    q = min(desired, avail, room, MAX_ORDER_SIZE)
                    if q > 0:
                        cash[product] -= q * snap.best_ask
                        position[product] += q
                else:
                    if snap.best_bid is None: continue
                    avail = max(0, snap.bid_volume)
                    if partial_cap is not None:
                        avail = min(avail, partial_cap)
                    room = max(0, POSITION_LIMIT + position[product])
                    q = min(-desired, avail, room, MAX_ORDER_SIZE)
                    if q > 0:
                        cash[product] += q * snap.best_bid
                        position[product] -= q

            target[A], target[B] = next_ta, next_tb
            stats.push(current_resid)

        last_a = day_data[A][-1]; last_b = day_data[B][-1]
        eod = cash[A] + position[A] * (last_a.mid or 0) + cash[B] + position[B] * (last_b.mid or 0)
        by_day[day] = eod

    total = sum(by_day.values())
    return total, by_day


def main():
    pair = ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE")
    print("=== Partial-fill / delayed-fill sensitivity (OT raw w=1000 z=1.50) ===")
    base, bd = simulate_pair_partial_delay(pair)
    print(f"baseline:                      total={base:>10,.0f} D2={bd[2]:>10,.0f} D3={bd[3]:>10,.0f} D4={bd[4]:>10,.0f}")
    for cap in (5, 3, 1):
        t, bd = simulate_pair_partial_delay(pair, partial_cap=cap)
        print(f"partial_cap={cap:<2d}                total={t:>10,.0f} D2={bd[2]:>10,.0f} D3={bd[3]:>10,.0f} D4={bd[4]:>10,.0f}")
    t, bd = simulate_pair_partial_delay(pair, delayed=True)
    print(f"delayed_fill_1tick (drop rest): total={t:>10,.0f} D2={bd[2]:>10,.0f} D3={bd[3]:>10,.0f} D4={bd[4]:>10,.0f}")


if __name__ == "__main__":
    main()
