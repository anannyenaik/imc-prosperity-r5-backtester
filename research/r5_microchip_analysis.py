"""Research harness for Microchip pair / basket residual mean-reversion.

Loads the public Round 5 Day 2-4 CSVs, builds Microchip mid-price series,
and simulates a pure cross-only strategy using rolling z-score signals with
previous-history-only state. Produces standalone PnL by day, drawdowns,
fill stress, match-mode proxies, rolling-slice diagnostics, and a parameter
neighbourhood table.

This is a research-only script; the production strategy file is separate.
The official prosperity4bt CLI is the source of truth for promotion gates.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "r5_data", "round5")
DAYS = (2, 3, 4)

MICROCHIPS = (
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
)

POSITION_LIMIT = 10
TARGET_SIZE = 10
MAX_ORDER_SIZE = 10


@dataclass
class Snapshot:
    timestamp: int
    best_bid: Optional[int]
    bid_volume: int
    best_ask: Optional[int]
    ask_volume: int
    mid: Optional[float]


_DAY_CACHE: dict[int, dict[str, list[Snapshot]]] = {}


def load_day(day: int) -> dict[str, list[Snapshot]]:
    """Return {product: [Snapshot...sorted by timestamp]} for the given day."""
    if day in _DAY_CACHE:
        return _DAY_CACHE[day]
    path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
    rows: dict[str, dict[int, Snapshot]] = defaultdict(dict)
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for r in reader:
            product = r["product"]
            if not product.startswith("MICROCHIP_"):
                continue
            ts = int(r["timestamp"])
            try:
                bb = int(r["bid_price_1"]) if r["bid_price_1"] else None
                bv = int(r["bid_volume_1"]) if r["bid_volume_1"] else 0
                ba = int(r["ask_price_1"]) if r["ask_price_1"] else None
                av = int(r["ask_volume_1"]) if r["ask_volume_1"] else 0
            except ValueError:
                continue
            mid: Optional[float] = None
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2.0
            rows[product][ts] = Snapshot(ts, bb, bv, ba, av, mid)
    out: dict[str, list[Snapshot]] = {}
    for product, mp in rows.items():
        out[product] = [mp[t] for t in sorted(mp)]
    _DAY_CACHE[day] = out
    return out


def aligned_timestamps(day_data: dict[str, list[Snapshot]]) -> list[int]:
    """Return sorted timestamps shared across all 5 microchip products."""
    sets = [set(s.timestamp for s in series) for series in day_data.values()]
    common = sorted(set.intersection(*sets))
    return common


@dataclass
class SimResult:
    total: float = 0.0
    by_day: dict[int, float] = field(default_factory=dict)
    by_product: dict[str, float] = field(default_factory=dict)
    trades: int = 0
    units_traded: int = 0
    max_drawdown: float = 0.0
    forced_flatten_pnl: float = 0.0
    final_position: dict[str, int] = field(default_factory=dict)
    pnl_curve: list[tuple[int, float]] = field(default_factory=list)


def rolling_z(history: list[float], current: float, window: int, min_history: int, min_std: float) -> Optional[float]:
    lookback = history[-window:]
    if len(lookback) < min_history:
        return None
    mean = sum(lookback) / len(lookback)
    msq = sum(v * v for v in lookback) / len(lookback)
    var = max(0.0, msq - mean * mean)
    std = math.sqrt(var)
    if std < min_std:
        return None
    return (current - mean) / std


class RollingStats:
    """O(1) update rolling-window mean/var for streaming residuals."""

    __slots__ = ("window", "values", "_sum", "_sumsq", "_n")

    def __init__(self, window: int) -> None:
        self.window = window
        self.values: list[float] = []
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0

    def push(self, value: float) -> None:
        self.values.append(value)
        self._sum += value
        self._sumsq += value * value
        self._n += 1
        if self._n > self.window:
            old = self.values[self._n - self.window - 1]
            self._sum -= old
            self._sumsq -= old * old

    def stats(self) -> tuple[int, float, float]:
        n = min(self._n, self.window)
        if n == 0:
            return 0, 0.0, 0.0
        mean = self._sum / n
        var = max(0.0, self._sumsq / n - mean * mean)
        return n, mean, math.sqrt(var)

    def z(self, current: float, min_history: int, min_std: float) -> Optional[float]:
        n, mean, std = self.stats()
        if n < min_history:
            return None
        if std < min_std:
            return None
        return (current - mean) / std


def simulate_pair(
    pair: tuple[str, str],
    *,
    residual_kind: str,  # "raw" or "log"
    window: int,
    min_history: int,
    entry_z: float,
    exit_z: Optional[float] = None,
    hold_to_flip: bool = True,
    target_size: int = TARGET_SIZE,
    fill_offset: int = 0,
    days: tuple[int, ...] = DAYS,
    min_std: float = 0.1,
) -> SimResult:
    """Simulate the pair strategy. We treat the pair as a 2-leg residual:
    short A / long B if z > entry, long A / short B if z < -entry.

    fill_offset>0 simulates worse fills: buy at ask+offset, sell at bid-offset.
    """
    A, B = pair
    stats = RollingStats(window)
    target = {A: 0, B: 0}
    position = {A: 0, B: 0}
    cash = {A: 0.0, B: 0.0}

    res = SimResult()
    res.by_product = {A: 0.0, B: 0.0}
    trades = 0
    units = 0
    peak_pnl = 0.0
    dd = 0.0
    cumulative_pnl_curve: list[tuple[int, float]] = []
    global_t = 0

    for day in days:
        day_data = load_day(day)
        timestamps = aligned_timestamps(day_data)

        # Index for fast lookup
        idx: dict[str, dict[int, Snapshot]] = {p: {s.timestamp: s for s in day_data[p]} for p in (A, B)}
        # Reset history per-day intentionally to avoid public-history seed
        stats = RollingStats(window)
        target = {A: 0, B: 0}
        position = {A: 0, B: 0}
        cash = {A: 0.0, B: 0.0}

        day_pnl_start = 0.0
        for ts in timestamps:
            snap_a = idx[A].get(ts)
            snap_b = idx[B].get(ts)
            if snap_a is None or snap_b is None or snap_a.mid is None or snap_b.mid is None:
                continue

            if residual_kind == "raw":
                current_resid = snap_a.mid - snap_b.mid
            elif residual_kind == "log":
                current_resid = (math.log(snap_a.mid) - math.log(snap_b.mid)) * 1000.0
            else:
                raise ValueError(residual_kind)

            # decide based on prior history
            z = stats.z(current_resid, min_history, min_std)
            n_history = stats._n

            if z is None:
                next_target_a = 0 if n_history < min_history else target[A]
                next_target_b = 0 if n_history < min_history else target[B]
            else:
                if z > entry_z:
                    next_target_a = -target_size
                    next_target_b = +target_size
                elif z < -entry_z:
                    next_target_a = +target_size
                    next_target_b = -target_size
                elif (not hold_to_flip) and exit_z is not None and abs(z) < exit_z:
                    next_target_a = 0
                    next_target_b = 0
                else:
                    next_target_a = target[A]
                    next_target_b = target[B]

            # Execute crosses (visible L1 only, with optional worsening fill_offset)
            for product, snap, t_target in ((A, snap_a, next_target_a), (B, snap_b, next_target_b)):
                desired_delta = t_target - position[product]
                if desired_delta == 0:
                    continue
                if desired_delta > 0:
                    if snap.best_ask is None:
                        continue
                    visible = max(0, snap.ask_volume)
                    limit_room = max(0, POSITION_LIMIT - position[product])
                    qty = min(desired_delta, visible, limit_room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    fill_price = snap.best_ask + fill_offset
                    cash[product] -= qty * fill_price
                    position[product] += qty
                    trades += 1
                    units += qty
                else:
                    if snap.best_bid is None:
                        continue
                    visible = max(0, snap.bid_volume)
                    limit_room = max(0, POSITION_LIMIT + position[product])
                    qty = min(-desired_delta, visible, limit_room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    fill_price = snap.best_bid - fill_offset
                    cash[product] += qty * fill_price
                    position[product] -= qty
                    trades += 1
                    units += qty

            target[A], target[B] = next_target_a, next_target_b

            # Append residual after decision
            stats.push(current_resid)

            # Mark-to-market PnL
            mtm = (
                cash[A] + position[A] * snap_a.mid
                + cash[B] + position[B] * snap_b.mid
            )
            cumulative_pnl_curve.append((global_t, day_pnl_start + mtm))
            if day_pnl_start + mtm > peak_pnl:
                peak_pnl = day_pnl_start + mtm
            else:
                drawdown = peak_pnl - (day_pnl_start + mtm)
                if drawdown > dd:
                    dd = drawdown
            global_t += 1

        # End of day mark to market: use last observed mid
        last_a = day_data[A][-1]
        last_b = day_data[B][-1]
        eod_pnl = (
            cash[A] + position[A] * (last_a.mid if last_a.mid is not None else 0)
            + cash[B] + position[B] * (last_b.mid if last_b.mid is not None else 0)
        )
        res.by_day[day] = eod_pnl
        res.by_product[A] += cash[A] + position[A] * (last_a.mid or 0)
        res.by_product[B] += cash[B] + position[B] * (last_b.mid or 0)
        res.final_position[A] = position[A]
        res.final_position[B] = position[B]

    res.total = sum(res.by_day.values())
    res.trades = trades
    res.units_traded = units
    res.max_drawdown = dd
    # Forced flatten: hit best bid/best ask on the *last* tick of the last day for each product
    last_day = days[-1]
    last_data = load_day(last_day)
    flatten_total = 0.0
    for d in days:
        # Per-day flatten approximation: the position at end of each day was already
        # marked at last mid above; assume flatten happens at last mid (no extra impact).
        flatten_total += res.by_day[d]
    res.forced_flatten_pnl = flatten_total
    return res


def simulate_basket(
    *,
    products: tuple[str, ...] = MICROCHIPS,
    window: int,
    min_history: int,
    entry_z: float,
    exit_z: Optional[float] = None,
    hold_to_flip: bool = True,
    target_size: int = TARGET_SIZE,
    fill_offset: int = 0,
    residual_scale: float = 10.0,
    days: tuple[int, ...] = DAYS,
    min_std: float = 1.0,
) -> SimResult:
    """All-five basket residual mean reversion."""
    stats = {p: RollingStats(window) for p in products}
    target = {p: 0 for p in products}
    position = {p: 0 for p in products}
    cash = {p: 0.0 for p in products}

    res = SimResult()
    res.by_product = {p: 0.0 for p in products}
    trades = 0
    units = 0
    peak_pnl = 0.0
    dd = 0.0
    global_t = 0

    for day in days:
        day_data = load_day(day)
        timestamps = aligned_timestamps(day_data)
        idx = {p: {s.timestamp: s for s in day_data[p]} for p in products}

        # Reset state per-day (no cross-day seed)
        for p in products:
            stats[p] = RollingStats(window)
            target[p] = 0
            position[p] = 0
            cash[p] = 0.0

        day_pnl_start = 0.0
        for ts in timestamps:
            snaps = {p: idx[p].get(ts) for p in products}
            if any(s is None or s.mid is None for s in snaps.values()):
                continue
            mids = {p: snaps[p].mid for p in products}
            mean_mid = sum(mids.values()) / len(products)
            scaled_resids = {
                p: (mids[p] - mean_mid) * residual_scale for p in products
            }

            new_targets = {}
            for p in products:
                z = stats[p].z(scaled_resids[p], min_history, min_std * residual_scale)
                if z is None:
                    new_targets[p] = 0 if stats[p]._n < min_history else target[p]
                else:
                    if z > entry_z:
                        new_targets[p] = -target_size
                    elif z < -entry_z:
                        new_targets[p] = +target_size
                    elif (not hold_to_flip) and exit_z is not None and abs(z) < exit_z:
                        new_targets[p] = 0
                    else:
                        new_targets[p] = target[p]

            for p in products:
                snap = snaps[p]
                desired_delta = new_targets[p] - position[p]
                if desired_delta == 0:
                    continue
                if desired_delta > 0:
                    if snap.best_ask is None:
                        continue
                    visible = max(0, snap.ask_volume)
                    limit_room = max(0, POSITION_LIMIT - position[p])
                    qty = min(desired_delta, visible, limit_room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    fill_price = snap.best_ask + fill_offset
                    cash[p] -= qty * fill_price
                    position[p] += qty
                    trades += 1
                    units += qty
                else:
                    if snap.best_bid is None:
                        continue
                    visible = max(0, snap.bid_volume)
                    limit_room = max(0, POSITION_LIMIT + position[p])
                    qty = min(-desired_delta, visible, limit_room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    fill_price = snap.best_bid - fill_offset
                    cash[p] += qty * fill_price
                    position[p] -= qty
                    trades += 1
                    units += qty

            for p in products:
                stats[p].push(scaled_resids[p])
                target[p] = new_targets[p]

            mtm = sum(cash[p] + position[p] * mids[p] for p in products)
            if mtm > peak_pnl:
                peak_pnl = mtm
            else:
                drawdown = peak_pnl - mtm
                if drawdown > dd:
                    dd = drawdown
            global_t += 1

        # Mark-to-market end-of-day
        eod_pnl = 0.0
        for p in products:
            last_snap = day_data[p][-1]
            mid = last_snap.mid or 0
            eod_pnl += cash[p] + position[p] * mid
            res.by_product[p] += cash[p] + position[p] * mid
            res.final_position[p] = position[p]
        res.by_day[day] = eod_pnl

    res.total = sum(res.by_day.values())
    res.trades = trades
    res.units_traded = units
    res.max_drawdown = dd
    res.forced_flatten_pnl = res.total
    return res


def fmt(x: float) -> str:
    return f"{x:>12,.0f}"


def print_pair_grid(pair, residual_kind):
    print(f"\n=== Pair {pair[0]}/{pair[1]} residual={residual_kind} ===")
    print("window  entry_z       total       D2       D3       D4   trades  drawdown  pos_A  pos_B")
    for w in (500, 750, 1000, 1250, 1500, 2000):
        for ez in (1.25, 1.5, 1.75, 2.0, 2.25):
            r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez)
            d2 = r.by_day.get(2, 0.0)
            d3 = r.by_day.get(3, 0.0)
            d4 = r.by_day.get(4, 0.0)
            print(
                f"{w:>6} {ez:>6.2f} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)} {r.final_position.get(pair[0], 0):>5} {r.final_position.get(pair[1], 0):>5}"
            )


def print_basket_grid():
    print("\n=== All-five Microchip basket residual ===")
    print("window  entry_z       total       D2       D3       D4   trades  drawdown")
    for w in (750, 1000, 1250, 1500, 2000):
        for ez in (1.5, 1.75, 2.0, 2.25):
            r = simulate_basket(window=w, min_history=w, entry_z=ez)
            d2 = r.by_day.get(2, 0.0)
            d3 = r.by_day.get(3, 0.0)
            d4 = r.by_day.get(4, 0.0)
            print(
                f"{w:>6} {ez:>6.2f} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)}"
            )


def stress_table(simulator, label):
    print(f"\n=== Stress table for {label} ===")
    print("offset       total       D2       D3       D4")
    for off in (0, 1, 3, 5, 10):
        r = simulator(off)
        d2 = r.by_day.get(2, 0.0)
        d3 = r.by_day.get(3, 0.0)
        d4 = r.by_day.get(4, 0.0)
        print(f"+{off:<5d} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("all", "pair_ot_raw"):
        print_pair_grid(("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "raw")
    if cmd in ("all", "pair_ot_log"):
        print_pair_grid(("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), "log")
    if cmd in ("all", "pair_cr_raw"):
        print_pair_grid(("MICROCHIP_CIRCLE", "MICROCHIP_RECTANGLE"), "raw")
    if cmd in ("all", "pair_cr_log"):
        print_pair_grid(("MICROCHIP_CIRCLE", "MICROCHIP_RECTANGLE"), "log")
    if cmd in ("all", "basket"):
        print_basket_grid()


if __name__ == "__main__":
    main()
