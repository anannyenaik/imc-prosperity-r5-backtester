"""
Round 5 Pebbles research harness.

A fast, deterministic simulator that mirrors the archived standalone Pebbles
code path in strategies/archive/r5_pebbles.py, plus optional crash-control
variants behind explicit feature flags. Used only as research / sweep tooling.

Outputs per config:
    - merged PnL, day split, traded units, +1/+3/+5 stress
    - max drawdown, terminal inventory value
    - cap dwell (per product)
    - PnL path metrics (peak / trough / final / peak-to-trough drawdown)
    - day-4 first-1,000-tick "website-slice" metrics (peak / trough / final / drawdown)
    - per-product final positions and PnL attribution
    - forced-final-flatten diagnostic PnL
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prosperity4bt.data import BacktestData, read_day_data
from prosperity4bt.file_reader import FileSystemReader

ROOT = Path(__file__).resolve().parents[1]
PEBBLES = ("PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL")
LIMIT = 10
RESIDUAL_SCALE = 10
DEFAULT_DAYS = (2, 3, 4)
WEBSITE_SLICE_TICKS = 1000  # day 4 first-1,000-tick slice


@dataclass(frozen=True)
class Variant:
    """Strategy configuration. Each instance is one cell of the sweep."""

    window: int = 500
    entry_z: float = 2.25
    exit_z: float | None = None
    target_size: int = 10
    min_history: int = 125
    history_limit: int = 700
    min_std_scaled: float = 10.0  # std (in scaled-residual units) below which we don't trade

    # crash-control flags (default off)
    target_per_product: dict[str, int] | None = None  # override target by product
    extreme_z_throttle: float | None = None  # if abs(z) > this, reduce target to throttle_target
    extreme_z_target: int = 8

    # product-specific entry offsets (added to base entry_z); kept simple
    entry_offsets: dict[str, float] = field(default_factory=dict)

    spread_gate: int | None = None  # only trade if best_ask - best_bid <= spread_gate
    reentry_delay: int = 0  # ticks to wait after a flip before another flip
    partial_step_in_z: float | None = None  # if set, target=5 below this z, full target above

    @property
    def hold_to_flip(self) -> bool:
        return self.exit_z is None

    def label(self) -> str:
        bits = [f"w{self.window}", f"z{self.entry_z:g}"]
        bits.append("hold" if self.exit_z is None else f"exit{self.exit_z:g}")
        bits.append(f"t{self.target_size}")
        if self.target_per_product:
            bits.append(
                "tp" + "_".join(f"{p[8]}{self.target_per_product.get(p, self.target_size)}" for p in PEBBLES)
            )
        if self.extreme_z_throttle is not None:
            bits.append(f"xz{self.extreme_z_throttle:g}t{self.extreme_z_target}")
        if self.entry_offsets:
            offset_bits = "_".join(f"{p[8]}{self.entry_offsets[p]:+g}" for p in PEBBLES if p in self.entry_offsets)
            bits.append(f"off{offset_bits}")
        if self.spread_gate is not None:
            bits.append(f"sg{self.spread_gate}")
        if self.reentry_delay:
            bits.append(f"rd{self.reentry_delay}")
        if self.partial_step_in_z is not None:
            bits.append(f"step{self.partial_step_in_z:g}")
        if self.min_history != 125:
            bits.append(f"mh{self.min_history}")
        if self.history_limit != 700:
            bits.append(f"hl{self.history_limit}")
        return "_".join(bits)


def load_cached_data(days: tuple[int, ...]) -> dict[int, BacktestData]:
    reader = FileSystemReader(ROOT / "r5_data")
    out: dict[int, BacktestData] = {}
    for day in days:
        data = read_day_data(reader, 5, day, no_names=False)
        out[day] = data
    return out


def z_score(history: list[int], current_residual: int, variant: Variant) -> float | None:
    lookback = history[-variant.window :]
    if len(lookback) < variant.min_history:
        return None
    mean = sum(lookback) / len(lookback)
    mean_square = sum(value * value for value in lookback) / len(lookback)
    variance = max(0.0, mean_square - mean * mean)
    std = math.sqrt(variance)
    if std < variant.min_std_scaled:
        return None
    return (current_residual - mean) / std


def desired_target(
    product: str,
    previous_target: int,
    z: float | None,
    has_min_history: bool,
    variant: Variant,
) -> int:
    if not has_min_history:
        return 0
    if z is None:
        return previous_target

    base_target = (
        variant.target_per_product.get(product, variant.target_size)
        if variant.target_per_product
        else variant.target_size
    )

    entry_threshold = variant.entry_z + variant.entry_offsets.get(product, 0.0)
    abs_z = abs(z)

    if abs_z < entry_threshold:
        if not variant.hold_to_flip and variant.exit_z is not None and abs_z < variant.exit_z:
            return 0
        return previous_target

    # crash-control: extreme z throttle
    if variant.extreme_z_throttle is not None and abs_z > variant.extreme_z_throttle:
        size = min(variant.extreme_z_target, base_target)
    elif variant.partial_step_in_z is not None and abs_z < variant.partial_step_in_z:
        size = min(5, base_target)
    else:
        size = base_target

    return -size if z > 0 else size


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass
class DayResult:
    day: int
    pnl_path: list[float]
    final_product_pnl: dict[str, float]
    per_product_traded_units: dict[str, int]
    per_product_trade_count: dict[str, int]
    cap_dwell: dict[str, int]
    final_positions: dict[str, int]
    terminal_inventory_value: dict[str, float]
    last_mids: dict[str, float]
    forced_flatten_pnl: float
    total_timestamps: int


def simulate_day(template: BacktestData, variant: Variant) -> DayResult:
    histories = {p: [] for p in PEBBLES}
    targets = {p: 0 for p in PEBBLES}
    positions = {p: 0 for p in PEBBLES}
    cash = {p: 0.0 for p in PEBBLES}
    trade_count = {p: 0 for p in PEBBLES}
    traded_units = {p: 0 for p in PEBBLES}
    cap_dwell = {p: 0 for p in PEBBLES}
    flip_cooldown = {p: 0 for p in PEBBLES}

    pnl_path: list[float] = []
    final_pnl = {p: 0.0 for p in PEBBLES}
    last_mids = {p: 0.0 for p in PEBBLES}

    for timestamp in sorted(template.prices.keys()):
        rows = template.prices[timestamp]
        if any(p not in rows for p in PEBBLES):
            continue

        mids: dict[str, float] = {}
        bids: dict[str, int | None] = {}
        asks: dict[str, int | None] = {}
        bid_volumes: dict[str, int] = {}
        ask_volumes: dict[str, int] = {}
        complete_book = True
        for p in PEBBLES:
            row = rows[p]
            if not row.bid_prices or not row.ask_prices:
                complete_book = False
                break
            bids[p] = row.bid_prices[0]
            asks[p] = row.ask_prices[0]
            bid_volumes[p] = row.bid_volumes[0] if row.bid_volumes else 0
            ask_volumes[p] = row.ask_volumes[0] if row.ask_volumes else 0
            mids[p] = (bids[p] + asks[p]) / 2.0
        if not complete_book:
            pnl_path.append(sum(cash[p] + positions[p] * mids.get(p, 0.0) for p in PEBBLES))
            continue

        last_mids = dict(mids)
        group_mean = sum(mids.values()) / len(PEBBLES)
        scaled_residuals = {p: int(round((mids[p] - group_mean) * RESIDUAL_SCALE)) for p in PEBBLES}

        next_targets: dict[str, int] = {}
        for p in PEBBLES:
            z = z_score(histories[p], scaled_residuals[p], variant)
            has_min_history = len(histories[p]) >= variant.min_history
            new_target = desired_target(p, targets[p], z, has_min_history, variant)
            if (
                variant.reentry_delay
                and new_target != targets[p]
                and targets[p] != 0
                and new_target * targets[p] <= 0
                and flip_cooldown[p] > 0
            ):
                new_target = targets[p]
            next_targets[p] = new_target

        for p in PEBBLES:
            final_pnl[p] = cash[p] + positions[p] * mids[p]
        pnl_path.append(sum(final_pnl.values()))

        for p in PEBBLES:
            target_position = clamp(next_targets[p], -LIMIT, LIMIT)
            current_position = positions[p]
            desired_delta = target_position - current_position

            if variant.spread_gate is not None and desired_delta != 0:
                if asks[p] - bids[p] > variant.spread_gate:
                    desired_delta = 0

            if desired_delta > 0:
                visible_volume = ask_volumes[p]
                limit_room = max(0, LIMIT - current_position)
                quantity = min(desired_delta, visible_volume, limit_room, variant.target_size)
                if quantity > 0:
                    positions[p] += quantity
                    cash[p] -= asks[p] * quantity
                    trade_count[p] += 1
                    traded_units[p] += quantity
            elif desired_delta < 0:
                visible_volume = bid_volumes[p]
                limit_room = max(0, LIMIT + current_position)
                quantity = min(-desired_delta, visible_volume, limit_room, variant.target_size)
                if quantity > 0:
                    positions[p] -= quantity
                    cash[p] += bids[p] * quantity
                    trade_count[p] += 1
                    traded_units[p] += quantity

        for p in PEBBLES:
            histories[p].append(scaled_residuals[p])
            if len(histories[p]) > variant.history_limit:
                del histories[p][: len(histories[p]) - variant.history_limit]
            if next_targets[p] != targets[p] and next_targets[p] * targets[p] <= 0:
                flip_cooldown[p] = variant.reentry_delay
            elif flip_cooldown[p] > 0:
                flip_cooldown[p] -= 1
            targets[p] = next_targets[p]
            if abs(positions[p]) >= LIMIT:
                cap_dwell[p] += 1

    terminal_inventory = {p: positions[p] * last_mids.get(p, 0.0) for p in PEBBLES}
    forced_flatten_pnl = sum(
        cash[p] + positions[p] * (bids[p] if positions[p] > 0 else asks[p] if positions[p] < 0 else last_mids.get(p, 0.0))
        for p in PEBBLES
    )

    return DayResult(
        day=template.day_num,
        pnl_path=pnl_path,
        final_product_pnl=dict(final_pnl),
        per_product_traded_units=traded_units,
        per_product_trade_count=trade_count,
        cap_dwell=cap_dwell,
        final_positions=dict(positions),
        terminal_inventory_value=terminal_inventory,
        last_mids=last_mids,
        forced_flatten_pnl=forced_flatten_pnl,
        total_timestamps=len(pnl_path),
    )


def path_metrics(path: list[float]) -> dict[str, float]:
    if not path:
        return {"peak": 0.0, "trough": 0.0, "final": 0.0, "p2t_dd": 0.0}
    high = -math.inf
    p2t_dd = 0.0
    for value in path:
        high = max(high, value)
        p2t_dd = max(p2t_dd, high - value)
    return {
        "peak": max(path),
        "trough": min(path),
        "final": path[-1],
        "p2t_dd": p2t_dd,
    }


def run_variant(variant: Variant, days: tuple[int, ...], cached: dict[int, BacktestData]) -> dict[str, Any]:
    day_results = [simulate_day(cached[day], variant) for day in days]

    day_pnl = {f"day_{r.day}": sum(r.final_product_pnl.values()) for r in day_results}
    total_pnl = sum(day_pnl.values())

    per_product = {p: 0.0 for p in PEBBLES}
    traded_units_by_product = {p: 0 for p in PEBBLES}
    cap_dwell_data = {p: 0 for p in PEBBLES}
    terminal_inventory = {p: 0.0 for p in PEBBLES}
    final_positions: dict[str, dict[str, int]] = {}
    forced_flatten_pnl = 0.0
    stitched: list[float] = []
    offset = 0.0

    for r in day_results:
        for p in PEBBLES:
            per_product[p] += r.final_product_pnl[p]
            traded_units_by_product[p] += r.per_product_traded_units[p]
            cap_dwell_data[p] += r.cap_dwell[p]
            terminal_inventory[p] += r.terminal_inventory_value[p]
        final_positions[f"day_{r.day}"] = r.final_positions
        forced_flatten_pnl += r.forced_flatten_pnl
        stitched.extend(offset + v for v in r.pnl_path)
        if r.pnl_path:
            offset += r.pnl_path[-1]

    cap_dwell_data["TOTAL_TIMESTAMPS"] = sum(r.total_timestamps for r in day_results)
    traded_units = sum(traded_units_by_product.values())
    full_path = path_metrics(stitched)

    # day-4 website-slice (first 1000 ticks)
    day4_result = next((r for r in day_results if r.day == 4), None)
    if day4_result is not None:
        slice_path = day4_result.pnl_path[:WEBSITE_SLICE_TICKS]
        slice_metrics = path_metrics(slice_path)
    else:
        slice_metrics = {"peak": 0.0, "trough": 0.0, "final": 0.0, "p2t_dd": 0.0}

    return {
        "label": variant.label(),
        "window": variant.window,
        "entry_z": variant.entry_z,
        "exit": "hold" if variant.exit_z is None else variant.exit_z,
        "target_size": variant.target_size,
        "min_history": variant.min_history,
        "history_limit": variant.history_limit,
        "merged_pnl": round(total_pnl, 2),
        "day_2": round(day_pnl.get("day_2", 0.0), 2),
        "day_3": round(day_pnl.get("day_3", 0.0), 2),
        "day_4": round(day_pnl.get("day_4", 0.0), 2),
        "traded_units": traded_units,
        "stress_1": round(total_pnl - traded_units, 2),
        "stress_3": round(total_pnl - 3 * traded_units, 2),
        "stress_5": round(total_pnl - 5 * traded_units, 2),
        "max_drawdown": round(full_path["p2t_dd"], 2),
        "path_peak": round(full_path["peak"], 2),
        "path_trough": round(full_path["trough"], 2),
        "path_final": round(full_path["final"], 2),
        "slice_peak": round(slice_metrics["peak"], 2),
        "slice_trough": round(slice_metrics["trough"], 2),
        "slice_final": round(slice_metrics["final"], 2),
        "slice_p2t_dd": round(slice_metrics["p2t_dd"], 2),
        "forced_flatten_pnl": round(forced_flatten_pnl, 2),
        "per_product_pnl": json.dumps(
            {p: round(per_product[p], 2) for p in PEBBLES}, separators=(",", ":")
        ),
        "traded_units_by_product": json.dumps(traded_units_by_product, separators=(",", ":")),
        "final_positions": json.dumps(final_positions, separators=(",", ":")),
        "cap_dwell": json.dumps({p: cap_dwell_data[p] for p in PEBBLES}, separators=(",", ":")),
        "terminal_inventory_value": json.dumps(
            {p: round(terminal_inventory[p], 2) for p in PEBBLES}, separators=(",", ":")
        ),
        "extreme_z_throttle": variant.extreme_z_throttle if variant.extreme_z_throttle is not None else "",
        "extreme_z_target": variant.extreme_z_target,
        "spread_gate": variant.spread_gate if variant.spread_gate is not None else "",
        "reentry_delay": variant.reentry_delay,
        "partial_step_in_z": variant.partial_step_in_z if variant.partial_step_in_z is not None else "",
        "target_per_product": json.dumps(variant.target_per_product) if variant.target_per_product else "",
        "entry_offsets": json.dumps(variant.entry_offsets) if variant.entry_offsets else "",
    }


def primary_grid() -> list[Variant]:
    out: list[Variant] = []
    for window in (350, 400, 450, 500, 550, 600):
        for entry_z in (2.10, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50, 2.60):
            for target in (8, 9, 10):
                out.append(
                    Variant(window=window, entry_z=entry_z, exit_z=None, target_size=target)
                )
    return out


def exit_z_controls() -> list[Variant]:
    out: list[Variant] = []
    for window in (450, 500, 550):
        for entry_z in (2.25, 2.35):
            for exit_z in (0.25, 0.50, 0.75):
                out.append(Variant(window=window, entry_z=entry_z, exit_z=exit_z, target_size=10))
    return out


def history_controls() -> list[Variant]:
    out: list[Variant] = []
    for min_history in (80, 100, 125, 150, 200, 250):
        out.append(Variant(window=500, entry_z=2.35, target_size=10, min_history=min_history))
    for history_limit in (500, 550, 600, 700):
        out.append(Variant(window=500, entry_z=2.35, target_size=10, history_limit=history_limit))
    return out


def crash_control_variants() -> list[Variant]:
    base_w_z = [(500, 2.35), (500, 2.25), (450, 2.35)]
    out: list[Variant] = []

    # 1. uniform target shrink
    for w, z in base_w_z:
        for t in (8, 9):
            out.append(Variant(window=w, entry_z=z, target_size=t))

    # 1b. selective target shrink
    for w, z in base_w_z:
        out.append(
            Variant(
                window=w,
                entry_z=z,
                target_size=10,
                target_per_product={"PEBBLES_XL": 8, "PEBBLES_M": 10, "PEBBLES_S": 10, "PEBBLES_L": 10, "PEBBLES_XS": 10},
            )
        )
        out.append(
            Variant(
                window=w,
                entry_z=z,
                target_size=10,
                target_per_product={"PEBBLES_XL": 8, "PEBBLES_M": 8, "PEBBLES_S": 8, "PEBBLES_L": 10, "PEBBLES_XS": 10},
            )
        )

    # 2. extreme z throttle
    for w, z in base_w_z:
        for thr in (3.5, 4.0, 4.5):
            out.append(Variant(window=w, entry_z=z, target_size=10, extreme_z_throttle=thr, extreme_z_target=8))

    # 4. product-specific entry offsets (XL only, small offsets)
    for w, z in base_w_z:
        for off in (0.10, 0.15):
            out.append(Variant(window=w, entry_z=z, target_size=10, entry_offsets={"PEBBLES_XL": off}))

    # 6. spread gate
    for w, z in base_w_z:
        for sg in (2, 3, 5):
            out.append(Variant(window=w, entry_z=z, target_size=10, spread_gate=sg))

    # 7. re-entry delay
    for w, z in base_w_z:
        for rd in (1, 2, 5):
            out.append(Variant(window=w, entry_z=z, target_size=10, reentry_delay=rd))

    # 8. partial step-in
    for w, z in base_w_z:
        for step in (2.75, 3.0):
            out.append(Variant(window=w, entry_z=z, target_size=10, partial_step_in_z=step))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backtests/r5_pebbles_research.csv")
    parser.add_argument(
        "--grid",
        choices=("primary", "exit", "history", "crash", "all"),
        default="primary",
    )
    parser.add_argument("--days", nargs="*", type=int, default=list(DEFAULT_DAYS))
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = tuple(args.days)
    cached = load_cached_data(days)

    grids: list[Variant] = []
    if args.grid in ("primary", "all"):
        grids += primary_grid()
    if args.grid in ("exit", "all"):
        grids += exit_z_controls()
    if args.grid in ("history", "all"):
        grids += history_controls()
    if args.grid in ("crash", "all"):
        grids += crash_control_variants()

    if args.max_configs is not None:
        grids = grids[: args.max_configs]

    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(grids, start=1):
        row = run_variant(variant, days, cached)
        rows.append(row)
        print(
            f"{index:03d}/{len(grids):03d} {row['label']} "
            f"pnl={row['merged_pnl']:.0f} stress5={row['stress_5']:.0f} "
            f"days=({row['day_2']:.0f},{row['day_3']:.0f},{row['day_4']:.0f}) "
            f"slice_dd={row['slice_p2t_dd']:.0f} maxdd={row['max_drawdown']:.0f}",
            flush=True,
        )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
