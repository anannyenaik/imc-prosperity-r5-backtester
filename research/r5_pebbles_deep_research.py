"""Deep Round 5 Pebbles research harness.

This is research-only code. It mirrors the archived standalone Pebbles
crossing strategy by default, then runs causal variants and rejection stresses
without pandas or numpy. The production Trader stays in strategies/r5_trader.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosperity4bt.data import BacktestData, read_day_data
from prosperity4bt.file_reader import FileSystemReader

PEBBLES = ("PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL")
DEFAULT_DAYS = (2, 3, 4)
LIMIT = 10
RESIDUAL_SCALE = 10
MIN_STD_SCALED = 10.0
WEBSITE_DAY = 4
WEBSITE_TICKS = 1000


@dataclass
class Series:
    values: list[float] = field(default_factory=list)
    sums: list[float] = field(default_factory=lambda: [0.0])
    sum_squares: list[float] = field(default_factory=lambda: [0.0])

    def append(self, value: float) -> None:
        self.values.append(value)
        self.sums.append(self.sums[-1] + value)
        self.sum_squares.append(self.sum_squares[-1] + value * value)

    def __len__(self) -> int:
        return len(self.values)

    def last(self, n: int) -> list[float]:
        return self.values[-n:]

    def mean_std(self, window: int) -> tuple[float, float] | None:
        count = min(window, len(self.values))
        if count <= 0:
            return None
        end = len(self.values)
        start = end - count
        total = self.sums[end] - self.sums[start]
        total_square = self.sum_squares[end] - self.sum_squares[start]
        mean = total / count
        variance = max(0.0, total_square / count - mean * mean)
        return mean, math.sqrt(variance)


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    window: int = 500
    entry_z: float = 2.35
    target_size: int = 10
    max_order_size: int = 10
    min_history: int = 125
    history_limit: int = 500
    residual_kind: str = "group_mean"
    target_rule: str = "independent"
    ensemble_windows: tuple[int, ...] = ()
    consensus_count: int = 0
    fast_slow: bool = False
    adaptive_window: bool = False
    product_caps: dict[str, int] = field(default_factory=dict)
    entry_offsets: dict[str, float] = field(default_factory=dict)
    two_stage_z: float | None = None
    proportional: bool = False
    soft_net_limit: int | None = None
    strong_net_z: float = 3.5
    extreme_z_cap: tuple[float, int] | None = None
    vol_ratio_cap: tuple[float, int] | None = None
    z_slope_cap: tuple[float, int] | None = None
    basket_sum_gate: tuple[float, int] | None = None
    family_min_signals: int | None = None
    use_l2: bool = False


@dataclass(frozen=True)
class StressConfig:
    name: str = "base"
    tick_cost: int = 0
    partial_fill_cap: int | None = None
    delayed_fill: bool = False
    missed_first_flip: bool = False
    residual_shift_product: str | None = None
    residual_shift: int = 0
    disabled_product: str | None = None
    carry_state_across_days: bool = False


@dataclass
class SimState:
    residuals: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in PEBBLES})
    raw_group_residuals: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in PEBBLES})
    mids: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in PEBBLES})
    basket_sum: Series = field(default_factory=Series)
    targets: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PEBBLES})
    pending_targets: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PEBBLES})
    skipped_first_flip: dict[str, bool] = field(default_factory=lambda: {p: False for p in PEBBLES})


@dataclass
class DayResult:
    day: int
    timestamps: list[int]
    pnl_path: list[float]
    product_paths: dict[str, list[float]]
    final_product_pnl: dict[str, float]
    traded_units: dict[str, int]
    trade_count: dict[str, int]
    cap_dwell: dict[str, int]
    final_positions: dict[str, int]
    forced_flatten_pnl: float
    last_mid: dict[str, float]
    last_bid: dict[str, int]
    last_ask: dict[str, int]


def load_data(days: tuple[int, ...]) -> dict[int, BacktestData]:
    reader = FileSystemReader(ROOT / "r5_data")
    return {day: read_day_data(reader, 5, day, no_names=False) for day in days}


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def z_score(series: Series, current_value: float, window: int, min_history: int) -> float | None:
    if len(series) < min_history:
        return None
    stats = series.mean_std(window)
    if stats is None:
        return None
    mean, std = stats
    if std < MIN_STD_SCALED:
        return None
    return (current_value - mean) / std


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def path_drawdown(path: list[float], timestamps: list[int] | None = None) -> dict[str, float | int]:
    if not path:
        return {
            "peak": 0.0,
            "trough": 0.0,
            "final": 0.0,
            "max_drawdown": 0.0,
            "peak_index": 0,
            "trough_index": 0,
            "peak_timestamp": 0,
            "trough_timestamp": 0,
        }

    high = path[0]
    high_index = 0
    max_dd = 0.0
    dd_peak_index = 0
    dd_trough_index = 0
    for index, value in enumerate(path):
        if value > high:
            high = value
            high_index = index
        drawdown = high - value
        if drawdown > max_dd:
            max_dd = drawdown
            dd_peak_index = high_index
            dd_trough_index = index

    peak_timestamp = timestamps[dd_peak_index] if timestamps and dd_peak_index < len(timestamps) else dd_peak_index
    trough_timestamp = timestamps[dd_trough_index] if timestamps and dd_trough_index < len(timestamps) else dd_trough_index
    return {
        "peak": max(path),
        "trough": min(path),
        "final": path[-1],
        "max_drawdown": max_dd,
        "peak_index": dd_peak_index,
        "trough_index": dd_trough_index,
        "peak_timestamp": peak_timestamp,
        "trough_timestamp": trough_timestamp,
    }


def drawdown_attribution(day: DayResult, start: int | None = None, end: int | None = None) -> dict[str, float]:
    path = day.pnl_path[slice(start, end)]
    timestamps = day.timestamps[slice(start, end)]
    metrics = path_drawdown(path, timestamps)
    peak_index = int(metrics["peak_index"]) + (start or 0)
    trough_index = int(metrics["trough_index"]) + (start or 0)
    attribution: dict[str, float] = {}
    for product in PEBBLES:
        product_path = day.product_paths[product]
        if peak_index < len(product_path) and trough_index < len(product_path):
            attribution[product] = product_path[trough_index] - product_path[peak_index]
        else:
            attribution[product] = 0.0
    return attribution


def rolling_slice_stats(
    days: list[DayResult],
    window: int,
    step: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    pnls: list[float] = []
    drawdowns: list[float] = []
    worst: dict[str, Any] = {"pnl": math.inf}

    for day in days:
        path = day.pnl_path
        for start in range(0, max(0, len(path) - window + 1), step):
            end = start + window
            base = path[start - 1] if start > 0 else 0.0
            slice_pnl = path[end - 1] - base
            metrics = path_drawdown(path[start:end], day.timestamps[start:end])
            pnls.append(slice_pnl)
            drawdowns.append(float(metrics["max_drawdown"]))
            if slice_pnl < worst["pnl"]:
                worst = {
                    "day": day.day,
                    "start_timestamp": day.timestamps[start],
                    "end_timestamp": day.timestamps[end - 1],
                    "pnl": slice_pnl,
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "attribution": {
                        product: day.product_paths[product][end - 1]
                        - (day.product_paths[product][start - 1] if start > 0 else 0.0)
                        for product in PEBBLES
                    },
                }

    if not pnls:
        empty = {"count": 0, "p01": 0.0, "p05": 0.0, "median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "worst_dd": 0.0}
        return empty, worst

    stats = {
        "count": len(pnls),
        "p01": percentile(pnls, 0.01),
        "p05": percentile(pnls, 0.05),
        "median": statistics.median(pnls),
        "mean": sum(pnls) / len(pnls),
        "min": min(pnls),
        "max": max(pnls),
        "worst_dd": max(drawdowns) if drawdowns else 0.0,
    }
    return stats, worst


def regression_residual(product: str, mids: dict[str, float], state: SimState, window: int) -> float | None:
    length = min(len(state.mids[p]) for p in PEBBLES)
    if length < 20:
        return None
    count = min(window, length)
    y_values = state.mids[product].last(count)
    x_values: list[float] = []
    for offset in range(length - count, length):
        x_values.append(sum(state.mids[p].values[offset] for p in PEBBLES if p != product) / (len(PEBBLES) - 1))

    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    var_x = sum((x - mean_x) * (x - mean_x) for x in x_values)
    if var_x <= 1e-9:
        return None
    cov_xy = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(count))
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    current_x = sum(mids[p] for p in PEBBLES if p != product) / (len(PEBBLES) - 1)
    return mids[product] - (alpha + beta * current_x)


def factor_residual(product: str, mids: dict[str, float], state: SimState, window: int) -> float | None:
    length = min(len(state.mids[p]) for p in PEBBLES)
    if length < 20:
        return None
    count = min(window, length)
    y_values = state.mids[product].last(count)
    x_values: list[float] = []
    for offset in range(length - count, length):
        x_values.append(sum(state.mids[p].values[offset] for p in PEBBLES) / len(PEBBLES))

    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    var_x = sum((x - mean_x) * (x - mean_x) for x in x_values)
    if var_x <= 1e-9:
        return None
    cov_xy = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(count))
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    current_x = sum(mids.values()) / len(PEBBLES)
    return mids[product] - (alpha + beta * current_x)


def residuals_for_config(
    mids: dict[str, float],
    bids: dict[str, int],
    asks: dict[str, int],
    bid_volumes: dict[str, int],
    ask_volumes: dict[str, int],
    state: SimState,
    config: StrategyConfig,
) -> dict[str, float]:
    values = [mids[p] for p in PEBBLES]
    group_mean = sum(values) / len(values)
    base_group = {p: (mids[p] - group_mean) * RESIDUAL_SCALE for p in PEBBLES}

    if config.residual_kind == "group_mean":
        return {p: round(base_group[p]) for p in PEBBLES}

    if config.residual_kind == "leave_one_out":
        total = sum(values)
        return {p: round((mids[p] - (total - mids[p]) / (len(values) - 1)) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "basket_constant":
        return {p: round((mids[p] - 10000.0) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "robust_median":
        median = statistics.median(values)
        return {p: round((mids[p] - median) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "trimmed_mean":
        trimmed = sorted(values)[1:-1]
        mean = sum(trimmed) / len(trimmed)
        return {p: round((mids[p] - mean) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "liquidity_weighted":
        weights = {p: max(1, bid_volumes[p] + ask_volumes[p]) for p in PEBBLES}
        total_weight = sum(weights.values())
        weighted_mean = sum(mids[p] * weights[p] for p in PEBBLES) / total_weight
        return {p: round((mids[p] - weighted_mean) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "inverse_spread_weighted":
        weights = {p: 1.0 / max(1, asks[p] - bids[p]) for p in PEBBLES}
        total_weight = sum(weights.values())
        weighted_mean = sum(mids[p] * weights[p] for p in PEBBLES) / total_weight
        return {p: round((mids[p] - weighted_mean) * RESIDUAL_SCALE) for p in PEBBLES}

    if config.residual_kind == "variance_normalised":
        out: dict[str, float] = {}
        for p in PEBBLES:
            stats = state.raw_group_residuals[p].mean_std(config.window)
            std = stats[1] if stats else RESIDUAL_SCALE
            out[p] = round(base_group[p] / max(std, 1.0) * RESIDUAL_SCALE)
        return out

    if config.residual_kind == "rolling_regression":
        out = {}
        for p in PEBBLES:
            residual = regression_residual(p, mids, state, config.window)
            out[p] = round((residual if residual is not None else mids[p] - group_mean) * RESIDUAL_SCALE)
        return out

    if config.residual_kind == "rolling_factor":
        out = {}
        for p in PEBBLES:
            residual = factor_residual(p, mids, state, config.window)
            out[p] = round((residual if residual is not None else mids[p] - group_mean) * RESIDUAL_SCALE)
        return out

    raise ValueError(f"Unknown residual kind: {config.residual_kind}")


def selected_windows(config: StrategyConfig, residual_history: Series) -> tuple[int, ...]:
    if config.ensemble_windows:
        return config.ensemble_windows
    if not config.adaptive_window:
        return (config.window,)

    short_stats = residual_history.mean_std(300)
    long_stats = residual_history.mean_std(700)
    if short_stats is None or long_stats is None or long_stats[1] <= 0:
        return (config.window,)
    return (300,) if short_stats[1] > 1.25 * long_stats[1] else (600,)


def signal_z(product: str, residual: float, state: SimState, config: StrategyConfig) -> float | None:
    history = state.residuals[product]
    windows = selected_windows(config, history)
    z_values = [z_score(history, residual, window, config.min_history) for window in windows]
    usable = [z for z in z_values if z is not None]
    if not usable:
        return None

    if config.fast_slow and len(usable) == len(windows):
        positive = all(z > config.entry_z for z in usable)
        negative = all(z < -config.entry_z for z in usable)
        if not positive and not negative:
            return sum(usable) / len(usable)

    return sum(usable) / len(usable)


def target_size_for_signal(product: str, z_value: float, state: SimState, config: StrategyConfig) -> int:
    cap = config.product_caps.get(product, config.target_size)

    if config.extreme_z_cap is not None:
        threshold, capped_size = config.extreme_z_cap
        if abs(z_value) > threshold:
            cap = min(cap, capped_size)

    if config.vol_ratio_cap is not None:
        ratio_threshold, capped_size = config.vol_ratio_cap
        short_stats = state.residuals[product].mean_std(100)
        long_stats = state.residuals[product].mean_std(config.window)
        if short_stats is not None and long_stats is not None and long_stats[1] > 0:
            if short_stats[1] / long_stats[1] > ratio_threshold:
                cap = min(cap, capped_size)

    if config.basket_sum_gate is not None:
        basket_z_threshold, capped_size = config.basket_sum_gate
        basket_z = z_score(state.basket_sum, state.basket_sum.values[-1] if state.basket_sum.values else 0.0, config.window, config.min_history)
        if basket_z is not None and abs(basket_z) > basket_z_threshold:
            cap = min(cap, capped_size)

    if config.two_stage_z is not None and abs(z_value) < config.two_stage_z:
        cap = min(cap, 5)

    if config.proportional:
        scaled = int(round(min(cap, max(1.0, abs(z_value) / config.entry_z * 5.0))))
        cap = max(1, scaled)

    return clamp(cap, 0, LIMIT)


def independent_targets(
    residuals: dict[str, float],
    z_values: dict[str, float | None],
    state: SimState,
    config: StrategyConfig,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    signal_count = sum(1 for z in z_values.values() if z is not None and abs(z) > config.entry_z)

    for product in PEBBLES:
        previous = state.targets[product]
        if product in config.entry_offsets:
            entry = config.entry_z + config.entry_offsets[product]
        else:
            entry = config.entry_z

        z_value = z_values[product]
        if len(state.residuals[product]) < config.min_history or z_value is None:
            targets[product] = 0 if len(state.residuals[product]) < config.min_history else previous
            continue

        if config.consensus_count and config.ensemble_windows:
            signs = []
            for window in config.ensemble_windows:
                z_for_window = z_score(state.residuals[product], residuals[product], window, config.min_history)
                if z_for_window is not None and abs(z_for_window) > entry:
                    signs.append(1 if z_for_window > 0 else -1)
            if len(signs) < config.consensus_count or abs(sum(signs)) < config.consensus_count:
                targets[product] = previous
                continue

        if config.family_min_signals is not None and signal_count < config.family_min_signals:
            targets[product] = previous
            continue

        size = target_size_for_signal(product, z_value, state, config)
        if z_value > entry:
            targets[product] = -size
        elif z_value < -entry:
            targets[product] = size
        else:
            targets[product] = previous

    return targets


def flatten_weakest_same_sign(targets: dict[str, int], z_values: dict[str, float | None], net_limit: int) -> dict[str, int]:
    adjusted = dict(targets)
    while abs(sum(adjusted.values())) > net_limit:
        net_sign = sign(sum(adjusted.values()))
        candidates = [p for p in PEBBLES if sign(adjusted[p]) == net_sign and adjusted[p] != 0]
        if not candidates:
            break
        weakest = min(candidates, key=lambda p: abs(z_values[p]) if z_values[p] is not None else 0.0)
        adjusted[weakest] = 0
    return adjusted


def apply_target_rule(
    residuals: dict[str, float],
    z_values: dict[str, float | None],
    state: SimState,
    config: StrategyConfig,
) -> dict[str, int]:
    if config.target_rule in {"independent", "basket_neutral", "soft_basket"}:
        targets = independent_targets(residuals, z_values, state, config)
        if config.target_rule == "basket_neutral":
            return flatten_weakest_same_sign(targets, z_values, 0)
        if config.target_rule == "soft_basket":
            return flatten_weakest_same_sign(targets, z_values, config.soft_net_limit or 10)
        return targets

    complete = all(z_values[p] is not None and len(state.residuals[p]) >= config.min_history for p in PEBBLES)
    if not complete:
        return {p: 0 for p in PEBBLES}

    ordered = sorted(PEBBLES, key=lambda p: z_values[p] or 0.0)
    targets = {p: 0 for p in PEBBLES}

    if config.target_rule == "rank_pair":
        low = ordered[0]
        high = ordered[-1]
        if (z_values[low] or 0.0) < -config.entry_z and (z_values[high] or 0.0) > config.entry_z:
            targets[low] = target_size_for_signal(low, z_values[low] or 0.0, state, config)
            targets[high] = -target_size_for_signal(high, z_values[high] or 0.0, state, config)
        return targets

    if config.target_rule == "rank2":
        for low in ordered[:2]:
            if (z_values[low] or 0.0) < -config.entry_z:
                targets[low] = target_size_for_signal(low, z_values[low] or 0.0, state, config)
        for high in ordered[-2:]:
            if (z_values[high] or 0.0) > config.entry_z:
                targets[high] = -target_size_for_signal(high, z_values[high] or 0.0, state, config)
        return targets

    raise ValueError(f"Unknown target rule: {config.target_rule}")


def execute_to_target(
    product: str,
    target: int,
    position: dict[str, int],
    cash: dict[str, float],
    bid_prices: list[int],
    bid_volumes: list[int],
    ask_prices: list[int],
    ask_volumes: list[int],
    config: StrategyConfig,
    stress: StressConfig,
) -> tuple[int, int]:
    current_position = position[product]
    desired_delta = clamp(target, -LIMIT, LIMIT) - current_position
    if desired_delta == 0:
        return 0, 0

    max_order_size = config.max_order_size
    if stress.partial_fill_cap is not None:
        max_order_size = min(max_order_size, stress.partial_fill_cap)
    remaining = min(abs(desired_delta), max_order_size)
    levels = len(ask_prices) if desired_delta > 0 else len(bid_prices)
    if not config.use_l2:
        levels = min(levels, 1)

    traded_units = 0
    trade_count = 0
    if desired_delta > 0:
        for index in range(levels):
            if remaining <= 0:
                break
            visible = ask_volumes[index]
            room = max(0, LIMIT - position[product])
            quantity = min(remaining, visible, room)
            if quantity <= 0:
                continue
            position[product] += quantity
            cash[product] -= ask_prices[index] * quantity
            remaining -= quantity
            traded_units += quantity
            trade_count += 1
    else:
        for index in range(levels):
            if remaining <= 0:
                break
            visible = bid_volumes[index]
            room = max(0, LIMIT + position[product])
            quantity = min(remaining, visible, room)
            if quantity <= 0:
                continue
            position[product] -= quantity
            cash[product] += bid_prices[index] * quantity
            remaining -= quantity
            traded_units += quantity
            trade_count += 1

    return traded_units, trade_count


def simulate_day(
    data: BacktestData,
    config: StrategyConfig,
    stress: StressConfig = StressConfig(),
    incoming_state: SimState | None = None,
) -> tuple[DayResult, SimState]:
    state = incoming_state if incoming_state is not None else SimState()
    positions = {p: 0 for p in PEBBLES}
    cash = {p: 0.0 for p in PEBBLES}
    traded_units = {p: 0 for p in PEBBLES}
    trade_count = {p: 0 for p in PEBBLES}
    cap_dwell = {p: 0 for p in PEBBLES}
    pnl_path: list[float] = []
    product_paths = {p: [] for p in PEBBLES}
    timestamps: list[int] = []
    last_mid = {p: 0.0 for p in PEBBLES}
    last_bid = {p: 0 for p in PEBBLES}
    last_ask = {p: 0 for p in PEBBLES}

    for timestamp in sorted(data.prices.keys()):
        rows = data.prices[timestamp]
        if any(product not in rows for product in PEBBLES):
            continue

        bids: dict[str, int] = {}
        asks: dict[str, int] = {}
        bid_volumes: dict[str, int] = {}
        ask_volumes: dict[str, int] = {}
        mids: dict[str, float] = {}
        complete = True

        for product in PEBBLES:
            row = rows[product]
            if not row.bid_prices or not row.ask_prices:
                complete = False
                break
            bids[product] = row.bid_prices[0]
            asks[product] = row.ask_prices[0]
            bid_volumes[product] = row.bid_volumes[0] if row.bid_volumes else 0
            ask_volumes[product] = row.ask_volumes[0] if row.ask_volumes else 0
            mids[product] = (bids[product] + asks[product]) / 2.0

        if not complete:
            continue

        last_mid = dict(mids)
        last_bid = dict(bids)
        last_ask = dict(asks)
        timestamps.append(timestamp)

        product_pnl = {product: cash[product] + positions[product] * mids[product] for product in PEBBLES}
        for product in PEBBLES:
            product_paths[product].append(product_pnl[product])
        pnl_path.append(sum(product_pnl.values()))

        raw_group_mean = sum(mids.values()) / len(PEBBLES)
        for product in PEBBLES:
            state.raw_group_residuals[product].append((mids[product] - raw_group_mean) * RESIDUAL_SCALE)
            state.mids[product].append(mids[product])
        state.basket_sum.append(sum(mids.values()))

        residuals = residuals_for_config(mids, bids, asks, bid_volumes, ask_volumes, state, config)
        if stress.residual_shift_product is not None:
            residuals[stress.residual_shift_product] += stress.residual_shift
        if stress.disabled_product is not None:
            residuals[stress.disabled_product] = 0.0

        z_values = {product: signal_z(product, residuals[product], state, config) for product in PEBBLES}
        next_targets = apply_target_rule(residuals, z_values, state, config)
        if stress.disabled_product is not None:
            next_targets[stress.disabled_product] = 0

        if config.z_slope_cap is not None:
            slope_threshold, capped_size = config.z_slope_cap
            for product in PEBBLES:
                if len(state.residuals[product]) == 0:
                    continue
                previous_residual = state.residuals[product].values[-1]
                residual_change = residuals[product] - previous_residual
                adverse = positions[product] > 0 and residual_change < -slope_threshold
                adverse = adverse or (positions[product] < 0 and residual_change > slope_threshold)
                if adverse and abs(next_targets[product]) > capped_size:
                    next_targets[product] = sign(next_targets[product]) * capped_size

        if stress.missed_first_flip:
            for product in PEBBLES:
                previous_target = state.targets[product]
                new_target = next_targets[product]
                is_flip = previous_target != 0 and new_target != 0 and sign(previous_target) != sign(new_target)
                if is_flip and not state.skipped_first_flip[product]:
                    next_targets[product] = previous_target
                    state.skipped_first_flip[product] = True

        execution_targets = state.pending_targets if stress.delayed_fill else next_targets
        for product in PEBBLES:
            row = rows[product]
            units, count = execute_to_target(
                product,
                execution_targets[product],
                positions,
                cash,
                row.bid_prices,
                row.bid_volumes,
                row.ask_prices,
                row.ask_volumes,
                config,
                stress,
            )
            traded_units[product] += units
            trade_count[product] += count

        for product in PEBBLES:
            state.residuals[product].append(residuals[product])
            state.targets[product] = next_targets[product]
            if abs(positions[product]) >= LIMIT:
                cap_dwell[product] += 1
        if stress.delayed_fill:
            state.pending_targets = dict(next_targets)

    final_product_pnl = {
        product: (product_paths[product][-1] if product_paths[product] else 0.0) for product in PEBBLES
    }
    forced_flatten_pnl = 0.0
    for product in PEBBLES:
        flatten_price = last_bid[product] if positions[product] > 0 else last_ask[product] if positions[product] < 0 else last_mid[product]
        forced_flatten_pnl += cash[product] + positions[product] * flatten_price

    return (
        DayResult(
            day=data.day_num,
            timestamps=timestamps,
            pnl_path=pnl_path,
            product_paths=product_paths,
            final_product_pnl=final_product_pnl,
            traded_units=traded_units,
            trade_count=trade_count,
            cap_dwell=cap_dwell,
            final_positions=dict(positions),
            forced_flatten_pnl=forced_flatten_pnl,
            last_mid=last_mid,
            last_bid=last_bid,
            last_ask=last_ask,
        ),
        state,
    )


def simulate_config(
    config: StrategyConfig,
    cached: dict[int, BacktestData],
    days: tuple[int, ...] = DEFAULT_DAYS,
    stress: StressConfig = StressConfig(),
) -> dict[str, Any]:
    day_results: list[DayResult] = []
    carried_state: SimState | None = None
    for day in days:
        incoming = carried_state if stress.carry_state_across_days else None
        result, carried_state = simulate_day(cached[day], config, stress, incoming)
        day_results.append(result)

    day_pnl = {f"day_{result.day}": sum(result.final_product_pnl.values()) for result in day_results}
    merged_pnl = sum(day_pnl.values())
    per_product_pnl = {product: sum(result.final_product_pnl[product] for result in day_results) for product in PEBBLES}
    traded_units = {product: sum(result.traded_units[product] for result in day_results) for product in PEBBLES}
    trade_count = {product: sum(result.trade_count[product] for result in day_results) for product in PEBBLES}
    cap_dwell = {product: sum(result.cap_dwell[product] for result in day_results) for product in PEBBLES}
    total_units = sum(traded_units.values())
    total_trade_count = sum(trade_count.values())

    stitched_path: list[float] = []
    stitched_timestamps: list[int] = []
    offset = 0.0
    timestamp_offset = 0
    for result in day_results:
        stitched_path.extend(offset + value for value in result.pnl_path)
        stitched_timestamps.extend(timestamp_offset + timestamp for timestamp in result.timestamps)
        if result.pnl_path:
            offset += result.pnl_path[-1]
            timestamp_offset = stitched_timestamps[-1] + 100

    full_drawdown = path_drawdown(stitched_path, stitched_timestamps)
    roll1000, worst1000 = rolling_slice_stats(day_results, 1000, 500)
    roll2000, worst2000 = rolling_slice_stats(day_results, 2000, 1000)

    day4 = next((result for result in day_results if result.day == WEBSITE_DAY), None)
    if day4 is not None:
        website_path = day4.pnl_path[:WEBSITE_TICKS]
        website_timestamps = day4.timestamps[:WEBSITE_TICKS]
        website = path_drawdown(website_path, website_timestamps)
        website_attribution = drawdown_attribution(day4, 0, min(WEBSITE_TICKS, len(day4.pnl_path)))
        first_1000 = website_path[-1] if website_path else 0.0
    else:
        website = path_drawdown([])
        website_attribution = {product: 0.0 for product in PEBBLES}
        first_1000 = 0.0

    forced_flatten_pnl = sum(result.forced_flatten_pnl for result in day_results)
    forced_flatten_cost = merged_pnl - forced_flatten_pnl
    final_positions = {f"day_{result.day}": result.final_positions for result in day_results}
    terminal_sensitivity = {}
    for shock in (-10, -5, -1, 1, 5, 10):
        terminal_sensitivity[str(shock)] = {
            f"day_{result.day}": sum(result.final_positions[p] * shock for p in PEBBLES) for result in day_results
        }

    row = {
        "label": config.label,
        "stress": stress.name,
        "merged_pnl": round(merged_pnl - stress.tick_cost * total_units, 2),
        "raw_merged_pnl": round(merged_pnl, 2),
        "day_2": round(day_pnl.get("day_2", 0.0), 2),
        "day_3": round(day_pnl.get("day_3", 0.0), 2),
        "day_4": round(day_pnl.get("day_4", 0.0), 2),
        "first_1000_day4": round(first_1000, 2),
        "traded_units": total_units,
        "trade_count": total_trade_count,
        "stress_1": round(merged_pnl - total_units, 2),
        "stress_3": round(merged_pnl - 3 * total_units, 2),
        "stress_5": round(merged_pnl - 5 * total_units, 2),
        "stress_10": round(merged_pnl - 10 * total_units, 2),
        "max_drawdown": round(float(full_drawdown["max_drawdown"]), 2),
        "path_peak": round(float(full_drawdown["peak"]), 2),
        "path_trough": round(float(full_drawdown["trough"]), 2),
        "path_final": round(float(full_drawdown["final"]), 2),
        "website_peak": round(float(website["peak"]), 2),
        "website_trough": round(float(website["trough"]), 2),
        "website_final": round(float(website["final"]), 2),
        "website_max_drawdown": round(float(website["max_drawdown"]), 2),
        "website_peak_timestamp": int(website["peak_timestamp"]),
        "website_trough_timestamp": int(website["trough_timestamp"]),
        "roll1000_count": roll1000["count"],
        "roll1000_p01": round(roll1000["p01"], 2),
        "roll1000_p05": round(roll1000["p05"], 2),
        "roll1000_median": round(roll1000["median"], 2),
        "roll1000_mean": round(roll1000["mean"], 2),
        "roll1000_min": round(roll1000["min"], 2),
        "roll1000_max": round(roll1000["max"], 2),
        "roll1000_worst_dd": round(roll1000["worst_dd"], 2),
        "roll2000_count": roll2000["count"],
        "roll2000_p01": round(roll2000["p01"], 2),
        "roll2000_p05": round(roll2000["p05"], 2),
        "roll2000_median": round(roll2000["median"], 2),
        "roll2000_mean": round(roll2000["mean"], 2),
        "roll2000_min": round(roll2000["min"], 2),
        "roll2000_max": round(roll2000["max"], 2),
        "roll2000_worst_dd": round(roll2000["worst_dd"], 2),
        "forced_flatten_pnl": round(forced_flatten_pnl, 2),
        "forced_flatten_cost": round(forced_flatten_cost, 2),
        "per_product_pnl": json.dumps({p: round(per_product_pnl[p], 2) for p in PEBBLES}, separators=(",", ":")),
        "traded_units_by_product": json.dumps(traded_units, separators=(",", ":")),
        "trade_count_by_product": json.dumps(trade_count, separators=(",", ":")),
        "final_positions": json.dumps(final_positions, separators=(",", ":")),
        "cap_dwell": json.dumps(cap_dwell, separators=(",", ":")),
        "terminal_sensitivity": json.dumps(terminal_sensitivity, separators=(",", ":")),
        "website_drawdown_attribution": json.dumps(
            {p: round(website_attribution[p], 2) for p in PEBBLES}, separators=(",", ":")
        ),
        "worst1000": json.dumps(worst1000, separators=(",", ":")),
        "worst2000": json.dumps(worst2000, separators=(",", ":")),
        "window": config.window,
        "entry_z": config.entry_z,
        "target_size": config.target_size,
        "min_history": config.min_history,
        "history_limit": config.history_limit,
        "residual_kind": config.residual_kind,
        "target_rule": config.target_rule,
        "ensemble_windows": json.dumps(config.ensemble_windows, separators=(",", ":")),
        "product_caps": json.dumps(config.product_caps, separators=(",", ":")),
        "entry_offsets": json.dumps(config.entry_offsets, separators=(",", ":")),
        "use_l2": config.use_l2,
    }
    return row


def baseline_config() -> StrategyConfig:
    return StrategyConfig(label="baseline_w500_z2.35")


def variant_grid(name: str) -> list[StrategyConfig]:
    baseline = baseline_config()
    variants: list[StrategyConfig] = [baseline]

    if name in {"history", "all"}:
        for history_limit in (500, 550, 600, 700):
            variants.append(replace(baseline, label=f"history_limit_{history_limit}", history_limit=history_limit))

    if name in {"residual", "all"}:
        for residual_kind in (
            "leave_one_out",
            "basket_constant",
            "rolling_regression",
            "rolling_factor",
            "robust_median",
            "trimmed_mean",
            "liquidity_weighted",
            "inverse_spread_weighted",
            "variance_normalised",
        ):
            variants.append(replace(baseline, label=f"residual_{residual_kind}", residual_kind=residual_kind))

    if name in {"window", "all"}:
        for window in (300, 350, 400, 450, 500, 550, 600, 700):
            for entry_z in (2.10, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50, 2.60):
                variants.append(replace(baseline, label=f"w{window}_z{entry_z:g}", window=window, entry_z=entry_z))
        variants.extend(
            [
                replace(baseline, label="ensemble_400_500", ensemble_windows=(400, 500)),
                replace(baseline, label="ensemble_450_500", ensemble_windows=(450, 500)),
                replace(baseline, label="ensemble_400_500_600", ensemble_windows=(400, 500, 600)),
                replace(baseline, label="consensus_400_500", ensemble_windows=(400, 500), consensus_count=2),
                replace(baseline, label="fast_slow_400_600", ensemble_windows=(400, 600), fast_slow=True),
                replace(baseline, label="adaptive_window", adaptive_window=True),
            ]
        )

    if name in {"target", "all"}:
        variants.extend(
            [
                replace(baseline, label="basket_neutral", target_rule="basket_neutral"),
                replace(baseline, label="soft_basket_net10", target_rule="soft_basket", soft_net_limit=10),
                replace(baseline, label="rank_pair", target_rule="rank_pair"),
                replace(baseline, label="rank2", target_rule="rank2"),
                replace(baseline, label="proportional", proportional=True),
                replace(baseline, label="two_stage_z3", two_stage_z=3.0),
                replace(baseline, label="family_min3", family_min_signals=3),
            ]
        )

    if name in {"risk", "all"}:
        variants.extend(
            [
                replace(baseline, label="xl_cap8", product_caps={"PEBBLES_XL": 8}),
                replace(
                    baseline,
                    label="xl_m_s_cap8",
                    product_caps={"PEBBLES_XL": 8, "PEBBLES_M": 8, "PEBBLES_S": 8},
                ),
                replace(baseline, label="all_cap9", target_size=9, max_order_size=9),
                replace(baseline, label="all_cap8", target_size=8, max_order_size=8),
                replace(baseline, label="extreme_z3.5_cap8", extreme_z_cap=(3.5, 8)),
                replace(baseline, label="extreme_z4_cap8", extreme_z_cap=(4.0, 8)),
                replace(baseline, label="extreme_z4.5_cap8", extreme_z_cap=(4.5, 8)),
                replace(baseline, label="vol_ratio1.25_cap8", vol_ratio_cap=(1.25, 8)),
                replace(baseline, label="vol_ratio1.5_cap8", vol_ratio_cap=(1.5, 8)),
                replace(baseline, label="z_slope20_cap8", z_slope_cap=(20.0, 8)),
                replace(baseline, label="z_slope30_cap8", z_slope_cap=(30.0, 8)),
                replace(baseline, label="basket_sum_z2.5_cap8", basket_sum_gate=(2.5, 8)),
                replace(baseline, label="basket_sum_z3_cap8", basket_sum_gate=(3.0, 8)),
            ]
        )

    if name in {"execution", "all"}:
        variants.append(replace(baseline, label="cross_l2_if_needed", use_l2=True))

    if name in {"product", "all"}:
        for offset in (0.05, 0.10, 0.15):
            variants.append(replace(baseline, label=f"xl_entry_plus_{offset:g}", entry_offsets={"PEBBLES_XL": offset}))
            variants.append(
                replace(
                    baseline,
                    label=f"ms_entry_plus_{offset:g}",
                    entry_offsets={"PEBBLES_M": offset, "PEBBLES_S": offset},
                )
            )

    if name == "finalists":
        variants = [
            baseline,
            replace(baseline, label="consensus_400_500", ensemble_windows=(400, 500), consensus_count=2),
            replace(baseline, label="xl_entry_plus_0.05", entry_offsets={"PEBBLES_XL": 0.05}),
            replace(baseline, label="w500_z2.4", entry_z=2.40),
            replace(baseline, label="w450_z2.45", window=450, entry_z=2.45),
            replace(baseline, label="w450_z2.4", window=450, entry_z=2.40),
            replace(baseline, label="all_cap9", target_size=9, max_order_size=9),
            replace(baseline, label="xl_cap8", product_caps={"PEBBLES_XL": 8}),
            replace(baseline, label="history_limit_500", history_limit=500),
        ]

    unique: dict[str, StrategyConfig] = {}
    for variant in variants:
        unique[variant.label] = variant
    return list(unique.values())


def stress_suite() -> list[StressConfig]:
    stresses = [StressConfig(name="base")]
    for tick_cost in (1, 3, 5, 10):
        stresses.append(StressConfig(name=f"tick_cost_{tick_cost}", tick_cost=tick_cost))
    stresses.extend(
        [
            StressConfig(name="partial_fill_cap5", partial_fill_cap=5),
            StressConfig(name="delayed_fill_1", delayed_fill=True),
            StressConfig(name="missed_first_flip", missed_first_flip=True),
            StressConfig(name="carry_history_across_days", carry_state_across_days=True),
            StressConfig(name="residual_shift_xl_plus50", residual_shift_product="PEBBLES_XL", residual_shift=50),
            StressConfig(name="residual_shift_xl_minus50", residual_shift_product="PEBBLES_XL", residual_shift=-50),
        ]
    )
    for product in PEBBLES:
        stresses.append(StressConfig(name=f"disabled_{product}", disabled_product=product))
    return stresses


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], limit: int = 20) -> None:
    ranked = sorted(rows, key=lambda row: float(row["merged_pnl"]), reverse=True)
    print("\nTop by merged PnL:")
    for row in ranked[:limit]:
        worst_day = min(float(row["day_2"]), float(row["day_3"]), float(row["day_4"]))
        print(
            f"{row['label'][:32]:32s} pnl={float(row['merged_pnl']):9.0f} "
            f"worst={worst_day:8.0f} +5={float(row['stress_5']):9.0f} "
            f"wdd={float(row['website_max_drawdown']):8.0f} "
            f"p05={float(row['roll1000_p05']):8.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=("finalists", "history", "residual", "window", "target", "risk", "execution", "product", "all"), default="finalists")
    parser.add_argument("--out", default="backtests/r5_pebbles_deep_research.csv")
    parser.add_argument("--stress-out", default="backtests/r5_pebbles_deep_stress.csv")
    parser.add_argument("--stress-suite", action="store_true")
    parser.add_argument("--days", nargs="*", type=int, default=list(DEFAULT_DAYS))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = tuple(args.days)
    cached = load_data(days)
    variants = variant_grid(args.grid)
    rows: list[dict[str, Any]] = []

    for index, config in enumerate(variants, start=1):
        row = simulate_config(config, cached, days)
        rows.append(row)
        if not args.quiet:
            print(
                f"{index:03d}/{len(variants):03d} {config.label} "
                f"pnl={float(row['merged_pnl']):.0f} "
                f"days=({float(row['day_2']):.0f},{float(row['day_3']):.0f},{float(row['day_4']):.0f}) "
                f"wdd={float(row['website_max_drawdown']):.0f} "
                f"p05={float(row['roll1000_p05']):.0f}",
                flush=True,
            )

    out_path = ROOT / args.out
    write_csv(out_path, rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    print_summary(rows)

    if args.stress_suite:
        stress_rows: list[dict[str, Any]] = []
        for config in variants:
            for stress in stress_suite():
                stress_rows.append(simulate_config(config, cached, days, stress))
        stress_path = ROOT / args.stress_out
        write_csv(stress_path, stress_rows)
        print(f"Wrote {stress_path} ({len(stress_rows)} rows)")


if __name__ == "__main__":
    main()
