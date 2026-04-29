"""Research harness for Robot pair / basket residual mean-reversion.

Mirrors research/r5_microchip_analysis.py but for ROBOT_* products. Loads
public Round 5 Day 2-4 CSVs, simulates cross-only L1 strategies with rolling
z-score signals using previous-history-only state, and produces standalone
PnL by day / drawdowns / fill stress / parameter neighbourhood / forced
flatten / rolling-slice diagnostics / product attribution.

Research-only. The official prosperity4bt CLI on strategies/archive/r5_robot_candidate.py
is the source of truth for promotion gates.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "r5_data", "round5")
DAYS = (2, 3, 4)

ROBOTS = (
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
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
    if day in _DAY_CACHE:
        return _DAY_CACHE[day]
    path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
    rows: dict[str, dict[int, Snapshot]] = defaultdict(dict)
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for r in reader:
            product = r["product"]
            if not product.startswith("ROBOT_"):
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


def aligned_timestamps(day_data: dict[str, list[Snapshot]], products: tuple[str, ...]) -> list[int]:
    sets = [set(s.timestamp for s in day_data[p]) for p in products]
    return sorted(set.intersection(*sets))


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
    pos_max_dwell: int = 0  # longest run at +/- target


class RollingStats:
    __slots__ = ("window", "values", "_sum", "_sumsq", "_n")

    def __init__(self, window: int) -> None:
        self.window = window
        self.values: list[float] = []
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0

    def push(self, v: float) -> None:
        self.values.append(v)
        self._sum += v
        self._sumsq += v * v
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
    residual_kind: str = "raw",
    window: int,
    min_history: int,
    entry_z: float,
    exit_z: Optional[float] = None,
    hold_to_flip: bool = True,
    target_size: int = TARGET_SIZE,
    fill_offset: int = 0,
    days: tuple[int, ...] = DAYS,
    min_std: float = 0.1,
    forced_flatten: bool = False,
) -> SimResult:
    A, B = pair
    res = SimResult(by_product={A: 0.0, B: 0.0})
    trades = 0
    units = 0
    peak = 0.0
    dd = 0.0
    pos_dwell_max = 0

    for day in days:
        day_data = load_day(day)
        ts_list = aligned_timestamps(day_data, (A, B))
        idx_a = {s.timestamp: s for s in day_data[A]}
        idx_b = {s.timestamp: s for s in day_data[B]}

        stats = RollingStats(window)
        target = {A: 0, B: 0}
        position = {A: 0, B: 0}
        cash = {A: 0.0, B: 0.0}
        run_at_cap = 0

        for ts in ts_list:
            sa = idx_a.get(ts)
            sb = idx_b.get(ts)
            if not sa or not sb or sa.mid is None or sb.mid is None:
                continue
            if residual_kind == "raw":
                resid = sa.mid - sb.mid
            elif residual_kind in ("log", "logratio"):
                resid = (math.log(sa.mid) - math.log(sb.mid)) * 1000.0
            else:
                raise ValueError(residual_kind)

            z = stats.z(resid, min_history, min_std)
            if z is None:
                next_ta = 0 if stats._n < min_history else target[A]
                next_tb = 0 if stats._n < min_history else target[B]
            else:
                if z > entry_z:
                    next_ta, next_tb = -target_size, +target_size
                elif z < -entry_z:
                    next_ta, next_tb = +target_size, -target_size
                elif (not hold_to_flip) and exit_z is not None and abs(z) < exit_z:
                    next_ta, next_tb = 0, 0
                else:
                    next_ta, next_tb = target[A], target[B]

            for product, snap, t_target in ((A, sa, next_ta), (B, sb, next_tb)):
                desired = t_target - position[product]
                if desired == 0:
                    continue
                if desired > 0:
                    if snap.best_ask is None:
                        continue
                    visible = max(0, snap.ask_volume)
                    room = max(0, POSITION_LIMIT - position[product])
                    qty = min(desired, visible, room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[product] -= qty * (snap.best_ask + fill_offset)
                    position[product] += qty
                    trades += 1
                    units += qty
                else:
                    if snap.best_bid is None:
                        continue
                    visible = max(0, snap.bid_volume)
                    room = max(0, POSITION_LIMIT + position[product])
                    qty = min(-desired, visible, room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[product] += qty * (snap.best_bid - fill_offset)
                    position[product] -= qty
                    trades += 1
                    units += qty

            target[A], target[B] = next_ta, next_tb
            stats.push(resid)
            mtm = cash[A] + position[A] * sa.mid + cash[B] + position[B] * sb.mid
            if mtm > peak:
                peak = mtm
            else:
                drawdown = peak - mtm
                if drawdown > dd:
                    dd = drawdown
            if abs(position[A]) == POSITION_LIMIT and abs(position[B]) == POSITION_LIMIT:
                run_at_cap += 1
                if run_at_cap > pos_dwell_max:
                    pos_dwell_max = run_at_cap
            else:
                run_at_cap = 0

        last_a = day_data[A][-1]
        last_b = day_data[B][-1]
        if forced_flatten:
            # close at worse opposite side
            ca, cb = cash[A], cash[B]
            if position[A] > 0 and last_a.best_bid is not None:
                ca += position[A] * last_a.best_bid
            elif position[A] < 0 and last_a.best_ask is not None:
                ca += position[A] * last_a.best_ask
            else:
                ca += position[A] * (last_a.mid or 0)
            if position[B] > 0 and last_b.best_bid is not None:
                cb += position[B] * last_b.best_bid
            elif position[B] < 0 and last_b.best_ask is not None:
                cb += position[B] * last_b.best_ask
            else:
                cb += position[B] * (last_b.mid or 0)
            eod = ca + cb
        else:
            eod = (
                cash[A] + position[A] * (last_a.mid or 0)
                + cash[B] + position[B] * (last_b.mid or 0)
            )
        res.by_day[day] = eod
        res.by_product[A] += cash[A] + position[A] * (last_a.mid or 0)
        res.by_product[B] += cash[B] + position[B] * (last_b.mid or 0)
        res.final_position[A] = position[A]
        res.final_position[B] = position[B]

    res.total = sum(res.by_day.values())
    res.trades = trades
    res.units_traded = units
    res.max_drawdown = dd
    res.pos_max_dwell = pos_dwell_max
    return res


def simulate_basket(
    *,
    products: tuple[str, ...] = ROBOTS,
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
    res = SimResult(by_product={p: 0.0 for p in products})
    trades = 0
    units = 0
    peak = 0.0
    dd = 0.0

    for day in days:
        day_data = load_day(day)
        ts_list = aligned_timestamps(day_data, products)
        idx = {p: {s.timestamp: s for s in day_data[p]} for p in products}

        stats = {p: RollingStats(window) for p in products}
        target = {p: 0 for p in products}
        position = {p: 0 for p in products}
        cash = {p: 0.0 for p in products}

        for ts in ts_list:
            snaps = {p: idx[p].get(ts) for p in products}
            if any(s is None or s.mid is None for s in snaps.values()):
                continue
            mids = {p: snaps[p].mid for p in products}
            mean_mid = sum(mids.values()) / len(products)
            scaled_resids = {p: (mids[p] - mean_mid) * residual_scale for p in products}

            new_targets: dict[str, int] = {}
            for p in products:
                z = stats[p].z(scaled_resids[p], min_history, min_std * residual_scale)
                if z is None:
                    new_targets[p] = 0 if stats[p]._n < min_history else target[p]
                else:
                    if z > entry_z:
                        new_targets[p] = -target_size
                    elif z < -entry_z:
                        new_targets[p] = target_size
                    elif (not hold_to_flip) and exit_z is not None and abs(z) < exit_z:
                        new_targets[p] = 0
                    else:
                        new_targets[p] = target[p]

            for p in products:
                snap = snaps[p]
                desired = new_targets[p] - position[p]
                if desired == 0:
                    continue
                if desired > 0:
                    if snap.best_ask is None:
                        continue
                    visible = max(0, snap.ask_volume)
                    room = max(0, POSITION_LIMIT - position[p])
                    qty = min(desired, visible, room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[p] -= qty * (snap.best_ask + fill_offset)
                    position[p] += qty
                    trades += 1
                    units += qty
                else:
                    if snap.best_bid is None:
                        continue
                    visible = max(0, snap.bid_volume)
                    room = max(0, POSITION_LIMIT + position[p])
                    qty = min(-desired, visible, room, MAX_ORDER_SIZE)
                    if qty <= 0:
                        continue
                    cash[p] += qty * (snap.best_bid - fill_offset)
                    position[p] -= qty
                    trades += 1
                    units += qty

            for p in products:
                stats[p].push(scaled_resids[p])
                target[p] = new_targets[p]

            mtm = sum(cash[p] + position[p] * mids[p] for p in products)
            if mtm > peak:
                peak = mtm
            else:
                drawdown = peak - mtm
                if drawdown > dd:
                    dd = drawdown

        eod = 0.0
        for p in products:
            last = day_data[p][-1]
            eod += cash[p] + position[p] * (last.mid or 0)
            res.by_product[p] += cash[p] + position[p] * (last.mid or 0)
            res.final_position[p] = position[p]
        res.by_day[day] = eod

    res.total = sum(res.by_day.values())
    res.trades = trades
    res.units_traded = units
    res.max_drawdown = dd
    return res


def fmt(x: float) -> str:
    return f"{x:>12,.0f}"


def print_pair_grid(pair, residual_kind, windows, zs):
    print(f"\n=== {pair[0]}/{pair[1]} residual={residual_kind} ===")
    print("window  entry_z       total       D2       D3       D4   trades     DD  pos_A pos_B")
    for w in windows:
        for ez in zs:
            r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez)
            d2, d3, d4 = r.by_day.get(2, 0), r.by_day.get(3, 0), r.by_day.get(4, 0)
            print(
                f"{w:>6} {ez:>6.2f} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)} {r.final_position.get(pair[0], 0):>5} {r.final_position.get(pair[1], 0):>5}"
            )


def print_basket_grid(windows, zs):
    print("\n=== All-five Robot basket residual ===")
    print("window  entry_z       total       D2       D3       D4   trades     DD")
    for w in windows:
        for ez in zs:
            r = simulate_basket(window=w, min_history=w, entry_z=ez)
            d2, d3, d4 = r.by_day.get(2, 0), r.by_day.get(3, 0), r.by_day.get(4, 0)
            print(f"{w:>6} {ez:>6.2f} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)}")


def stress_table(label, pair, residual_kind, w, ez):
    print(f"\n=== Stress: {label} ({pair[0]}/{pair[1]} resid={residual_kind} w={w} z={ez}) ===")
    print("offset       total       D2       D3       D4   trades     DD")
    for off in (0, 1, 3, 5, 10):
        r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez, fill_offset=off)
        d2, d3, d4 = r.by_day.get(2, 0), r.by_day.get(3, 0), r.by_day.get(4, 0)
        print(f"+{off:<5d} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)}")


def basket_stress_table(label, w, ez):
    print(f"\n=== Stress: {label} (basket w={w} z={ez}) ===")
    print("offset       total       D2       D3       D4   trades     DD")
    for off in (0, 1, 3, 5, 10):
        r = simulate_basket(window=w, min_history=w, entry_z=ez, fill_offset=off)
        d2, d3, d4 = r.by_day.get(2, 0), r.by_day.get(3, 0), r.by_day.get(4, 0)
        print(f"+{off:<5d} {fmt(r.total)} {fmt(d2)} {fmt(d3)} {fmt(d4)} {r.trades:>7} {fmt(r.max_drawdown)}")


def attribution(label, pair, residual_kind, w, ez):
    r = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez)
    print(f"\n=== Attribution: {label} ===")
    for p, v in r.by_product.items():
        print(f"  {p:<22} {v:>12,.0f}")
    print(f"  trades={r.trades}  units={r.units_traded}  pos_dwell={r.pos_max_dwell}  finalA={r.final_position.get(pair[0],0)}  finalB={r.final_position.get(pair[1],0)}")
    rf = simulate_pair(pair, residual_kind=residual_kind, window=w, min_history=w, entry_z=ez, forced_flatten=True)
    print(f"  forced_flatten total={rf.total:,.0f}  delta={rf.total - r.total:,.0f}")


def rolling_slice(label, pair, residual_kind, w, ez, slice_lens=(1000, 2000)):
    """Within-day worst slice (rebuild the curve)."""
    A, B = pair
    print(f"\n=== Rolling slice: {label} ===")
    for slc in slice_lens:
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
                if residual_kind == "raw":
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
            for i in range(len(curve) - slc + 1):
                delta = curve[i + slc - 1] - curve[i]
                if delta < worst:
                    worst = delta
                    where = (day, i, i + slc - 1)
        print(f"  slice={slc}: worst delta = {worst:,.0f}  where={where}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("all", "lv"):
        print_pair_grid(("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "raw",
                        windows=(1500, 1750, 2000, 2250, 2500, 2750, 3000),
                        zs=(1.50, 1.75, 2.00, 2.25, 2.50))
        print_pair_grid(("ROBOT_LAUNDRY", "ROBOT_VACUUMING"), "log",
                        windows=(1500, 1750, 2000, 2250, 2500, 2750, 3000),
                        zs=(1.50, 1.75, 2.00, 2.25, 2.50))
    if cmd in ("all", "di"):
        print_pair_grid(("ROBOT_DISHES", "ROBOT_IRONING"), "raw",
                        windows=(750, 1000, 1250, 1500, 1750, 2000),
                        zs=(1.25, 1.50, 1.75, 2.00, 2.25))
        print_pair_grid(("ROBOT_DISHES", "ROBOT_IRONING"), "log",
                        windows=(750, 1000, 1250, 1500, 1750, 2000),
                        zs=(1.25, 1.50, 1.75, 2.00, 2.25))
    if cmd in ("all", "discovery"):
        for pair in [
            ("ROBOT_DISHES", "ROBOT_VACUUMING"),
            ("ROBOT_MOPPING", "ROBOT_VACUUMING"),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY"),
            ("ROBOT_MOPPING", "ROBOT_LAUNDRY"),
            ("ROBOT_LAUNDRY", "ROBOT_IRONING"),
            ("ROBOT_VACUUMING", "ROBOT_IRONING"),
            ("ROBOT_DISHES", "ROBOT_MOPPING"),
            ("ROBOT_MOPPING", "ROBOT_IRONING"),
        ]:
            print_pair_grid(pair, "raw", windows=(1500, 2000, 2500), zs=(1.75, 2.00, 2.25))
    if cmd in ("all", "basket"):
        print_basket_grid(windows=(1000, 1500, 2000, 2500), zs=(1.50, 1.75, 2.00, 2.25, 2.50))


if __name__ == "__main__":
    main()
