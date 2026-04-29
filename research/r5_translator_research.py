"""Round 5 Translator research harness.

Research-only code. The submission strategy lives in strategies/r5_trader.py.
This simulator mirrors L1 crossing and keeps all signals causal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosperity4bt.data import BacktestData, read_day_data
from prosperity4bt.file_reader import FileSystemReader

TRANSLATORS = (
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
)
PAIR_LEFT = "TRANSLATOR_ECLIPSE_CHARCOAL"
PAIR_RIGHT = "TRANSLATOR_VOID_BLUE"
PAIR_KEY = "ECL_VOID_PAIR"
PAIR_PRODUCTS = tuple(combinations(TRANSLATORS, 2))
DEFAULT_DAYS = (2, 3, 4)
LIMIT = 10
RESIDUAL_SCALE = 10
MIN_STD_SCALED = 10.0


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
    mode: str = "group"
    window: int = 1000
    entry_z: float = 1.5
    min_history: int = 250
    history_limit: int = 1000
    hold_to_flip: bool = True
    exit_z: float | None = None
    exit_rule: str = "hold"
    residual_kind: str = "group_mean"
    active_products: tuple[str, ...] = TRANSLATORS
    target_size: int = 10
    product_target_caps: tuple[tuple[str, int], ...] = ()
    max_order_size: int = 10
    spread_gate: bool = False
    spread_rule: str = "none"
    edge_gate_buffer: float | None = None
    confirm_ticks: int = 1
    reentry_delay: int = 0
    flip_buffer: float = 0.0
    pair_products: tuple[str, str] = (PAIR_LEFT, PAIR_RIGHT)
    pair_window: int = 2000
    pair_entry_z: float = 1.5
    pair_min_history: int = 500
    pair_exit_z: float | None = 0.3
    pair_hold_to_flip: bool = False


@dataclass
class SimState:
    residuals: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in TRANSLATORS})
    raw_group_residuals: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in TRANSLATORS})
    mids: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in TRANSLATORS})
    spreads: dict[str, Series] = field(default_factory=lambda: {p: Series() for p in TRANSLATORS})
    pair_residuals: Series = field(default_factory=Series)
    targets: dict[str, int] = field(default_factory=lambda: {p: 0 for p in TRANSLATORS})
    pending_direction: dict[str, int] = field(default_factory=lambda: {p: 0 for p in TRANSLATORS})
    pending_count: dict[str, int] = field(default_factory=lambda: {p: 0 for p in TRANSLATORS})
    cooldown: dict[str, int] = field(default_factory=lambda: {p: 0 for p in TRANSLATORS})


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
    cap_dwell_long: dict[str, int]
    cap_dwell_short: dict[str, int]
    spread_paid: dict[str, float]
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


def target_cap(config: StrategyConfig, product: str) -> int:
    caps = dict(config.product_target_caps)
    return clamp(caps.get(product, config.target_size), 0, LIMIT)


def spread_percentile(series: Series, window: int, pct: float) -> float | None:
    values = series.values[-window:]
    if len(values) < max(20, min(300, window // 4)):
        return None
    return percentile(values, pct)


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
    max_drawdown = 0.0
    peak_index = 0
    trough_index = 0
    for index, value in enumerate(path):
        if value > high:
            high = value
            high_index = index
        drawdown = high - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            peak_index = high_index
            trough_index = index

    return {
        "peak": max(path),
        "trough": min(path),
        "final": path[-1],
        "max_drawdown": max_drawdown,
        "peak_index": peak_index,
        "trough_index": trough_index,
        "peak_timestamp": timestamps[peak_index] if timestamps else peak_index,
        "trough_timestamp": timestamps[trough_index] if timestamps else trough_index,
    }


def rolling_slice_stats(days: list[DayResult], window: int, step: int) -> tuple[dict[str, float], dict[str, Any]]:
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
                        for product in TRANSLATORS
                    },
                }

    if not pnls:
        empty = {"count": 0, "p01": 0.0, "p05": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "worst_dd": 0.0}
        return empty, worst

    return (
        {
            "count": len(pnls),
            "p01": percentile(pnls, 0.01),
            "p05": percentile(pnls, 0.05),
            "median": statistics.median(pnls),
            "min": min(pnls),
            "max": max(pnls),
            "worst_dd": max(drawdowns),
        },
        worst,
    )


def rolling_regression_residual(product: str, mids: dict[str, float], state: SimState, config: StrategyConfig) -> float | None:
    history_length = min(len(state.mids[p]) for p in config.active_products)
    count = min(config.window, history_length)
    if count < max(20, config.min_history):
        return None

    start = history_length - count
    x_values: list[float] = []
    y_values = state.mids[product].values[start:history_length]
    for index in range(start, history_length):
        x_values.append(
            sum(state.mids[p].values[index] for p in config.active_products if p != product)
            / (len(config.active_products) - 1)
        )

    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    var_x = sum((value - mean_x) * (value - mean_x) for value in x_values)
    if var_x <= 1e-9:
        return None
    cov_xy = sum((x_values[index] - mean_x) * (y_values[index] - mean_y) for index in range(count))
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    current_x = sum(mids[p] for p in config.active_products if p != product) / (len(config.active_products) - 1)
    return (mids[product] - (alpha + beta * current_x)) * RESIDUAL_SCALE


def group_residuals(mids: dict[str, float], state: SimState, config: StrategyConfig) -> dict[str, float]:
    products = config.active_products
    group_mean = sum(mids[p] for p in products) / len(products)
    base = {p: (mids[p] - group_mean) * RESIDUAL_SCALE for p in products}

    if config.residual_kind == "group_mean":
        return {p: round(base[p]) for p in products}

    if config.residual_kind == "leave_one_out":
        total = sum(mids[p] for p in products)
        return {p: round((mids[p] - (total - mids[p]) / (len(products) - 1)) * RESIDUAL_SCALE) for p in products}

    if config.residual_kind == "product_demeaned":
        out: dict[str, float] = {}
        for product in products:
            stats = state.raw_group_residuals[product].mean_std(config.window)
            mean = stats[0] if stats is not None else 0.0
            out[product] = round(base[product] - mean)
        return out

    if config.residual_kind == "vol_normalised":
        out = {}
        for product in products:
            stats = state.raw_group_residuals[product].mean_std(config.window)
            std = stats[1] if stats is not None else RESIDUAL_SCALE
            out[product] = round(base[product] / max(std, 1.0) * RESIDUAL_SCALE)
        return out

    if config.residual_kind == "rolling_regression":
        out = {}
        for product in products:
            residual = rolling_regression_residual(product, mids, state, config)
            out[product] = round(residual if residual is not None else base[product])
        return out

    if config.residual_kind == "rank":
        ordered = sorted(products, key=lambda p: base[p])
        centre = (len(products) - 1) / 2.0
        return {product: round((rank - centre) * RESIDUAL_SCALE) for rank, product in enumerate(ordered)}

    raise ValueError(f"Unknown residual kind: {config.residual_kind}")


def size_for_signal(abs_z: float, config: StrategyConfig, product: str) -> int:
    cap = target_cap(config, product)
    if config.exit_rule == "two_stage":
        return min(5, cap) if abs_z < config.entry_z + 0.5 else cap
    if config.exit_rule == "proportional":
        raw_size = int(math.floor((abs_z - config.entry_z) * 3.0)) + 1
        return clamp(raw_size, 1, cap)
    return cap


def signal_direction(z_value: float, previous_target: int, config: StrategyConfig) -> int:
    short_trigger = config.entry_z
    long_trigger = config.entry_z
    previous_direction = sign(previous_target)
    if previous_direction > 0:
        short_trigger += config.flip_buffer
    elif previous_direction < 0:
        long_trigger += config.flip_buffer

    if z_value > short_trigger:
        return -1
    if z_value < -long_trigger:
        return 1
    return 0


def confirmed_target(
    product: str,
    desired_direction: int,
    desired_size: int,
    state: SimState,
    config: StrategyConfig,
) -> int:
    previous = state.targets[product]
    previous_direction = sign(previous)

    if desired_direction == 0:
        state.pending_direction[product] = 0
        state.pending_count[product] = 0
        return previous

    if (
        config.reentry_delay > 0
        and state.cooldown[product] > 0
        and previous_direction != 0
        and desired_direction != previous_direction
    ):
        return previous

    if config.confirm_ticks > 1 and desired_direction != previous_direction:
        if state.pending_direction[product] == desired_direction:
            state.pending_count[product] += 1
        else:
            state.pending_direction[product] = desired_direction
            state.pending_count[product] = 1

        if state.pending_count[product] < config.confirm_ticks:
            return previous

    state.pending_direction[product] = 0
    state.pending_count[product] = 0
    next_target = desired_direction * desired_size
    if previous_direction != 0 and sign(next_target) != previous_direction and config.reentry_delay > 0:
        state.cooldown[product] = config.reentry_delay
    return next_target


def apply_group_targets(
    residuals: dict[str, float],
    state: SimState,
    config: StrategyConfig,
) -> dict[str, int]:
    targets = {product: 0 for product in TRANSLATORS}

    for product in config.active_products:
        previous = state.targets[product]
        z_value = z_score(state.residuals[product], residuals[product], config.window, config.min_history)
        if len(state.residuals[product]) < config.min_history or z_value is None:
            targets[product] = 0 if len(state.residuals[product]) < config.min_history else previous
            continue

        direction = signal_direction(z_value, previous, config)
        if direction != 0:
            size = size_for_signal(abs(z_value), config, product)
            targets[product] = confirmed_target(product, direction, size, state, config)
        elif config.exit_rule == "two_stage" and abs(z_value) >= max(0.5, config.entry_z - 0.5):
            direction = -1 if z_value > 0 else 1
            targets[product] = confirmed_target(product, direction, min(5, target_cap(config, product)), state, config)
        elif config.hold_to_flip:
            targets[product] = previous
        elif config.exit_rule == "flatten" and config.exit_z is not None and abs(z_value) < config.exit_z:
            targets[product] = 0
        elif config.exit_rule == "half" and abs(z_value) < 0.5:
            targets[product] = sign(previous) * min(5, abs(previous))
        elif config.exit_rule == "proportional":
            targets[product] = 0
        else:
            targets[product] = previous

    return targets


def apply_pair_targets(mids: dict[str, float], state: SimState, config: StrategyConfig) -> tuple[dict[str, int], float, float | None]:
    targets = {product: 0 for product in TRANSLATORS}
    left, right = config.pair_products
    residual = (mids[left] - mids[right]) * RESIDUAL_SCALE
    window = config.pair_window if config.mode == "combo" else config.window
    entry = config.pair_entry_z if config.mode == "combo" else config.entry_z
    min_history = config.pair_min_history if config.mode == "combo" else config.min_history
    exit_z = config.pair_exit_z if config.mode == "combo" else config.exit_z
    hold_to_flip = config.pair_hold_to_flip if config.mode == "combo" else config.hold_to_flip

    z_value = z_score(state.pair_residuals, residual, window, min_history)
    if len(state.pair_residuals) < min_history or z_value is None:
        return targets, residual, z_value

    previous_left = state.targets[left]
    previous_right = state.targets[right]
    left_size = target_cap(config, left)
    right_size = target_cap(config, right)
    if z_value > entry:
        targets[left] = -left_size
        targets[right] = right_size
    elif z_value < -entry:
        targets[left] = left_size
        targets[right] = -right_size
    elif hold_to_flip:
        targets[left] = previous_left
        targets[right] = previous_right
    elif exit_z is not None and abs(z_value) < exit_z:
        targets[left] = 0
        targets[right] = 0
    else:
        targets[left] = previous_left
        targets[right] = previous_right

    return targets, residual, z_value


def combine_targets(group_targets: dict[str, int], pair_targets: dict[str, int], config: StrategyConfig) -> dict[str, int]:
    combined = dict(group_targets)
    for product in config.pair_products:
        pair_target = pair_targets[product]
        if pair_target == 0:
            continue
        group_target = combined[product]
        if group_target == 0 or sign(group_target) == sign(pair_target):
            combined[product] = pair_target if abs(pair_target) >= abs(group_target) else group_target
    return combined


def spread_allowed(product: str, bids: dict[str, int], asks: dict[str, int], state: SimState, config: StrategyConfig) -> bool:
    rule = "p60" if config.spread_gate and config.spread_rule == "none" else config.spread_rule
    if rule == "none":
        return True
    spread = asks[product] - bids[product]
    pct_by_rule = {
        "p50": 0.50,
        "p60": 0.60,
        "p75": 0.75,
        "skip_p90": 0.90,
        "reduce_p75": 0.75,
    }
    threshold_pct = pct_by_rule.get(rule)
    if threshold_pct is None:
        return True
    threshold = spread_percentile(state.spreads[product], config.window, threshold_pct)
    if threshold is None:
        return True
    if rule == "reduce_p75":
        return True
    return spread <= threshold


def reduce_for_spread(product: str, target: int, bids: dict[str, int], asks: dict[str, int], state: SimState, config: StrategyConfig) -> int:
    if config.spread_rule != "reduce_p75" or target == 0:
        return target
    threshold = spread_percentile(state.spreads[product], config.window, 0.75)
    if threshold is None:
        return target
    spread = asks[product] - bids[product]
    if spread <= threshold:
        return target
    return sign(target) * min(abs(target), 5)


def edge_allowed(
    product: str,
    target: int,
    residuals: dict[str, float],
    bids: dict[str, int],
    asks: dict[str, int],
    state: SimState,
    config: StrategyConfig,
) -> bool:
    if config.edge_gate_buffer is None or target == state.targets[product]:
        return True
    stats = state.residuals[product].mean_std(config.window)
    if stats is None:
        return True
    mean, std = stats
    edge_ticks = abs(residuals.get(product, mean) - mean) / RESIDUAL_SCALE
    half_spread = (asks[product] - bids[product]) / 2.0
    return edge_ticks >= half_spread + config.edge_gate_buffer


def execute_to_target(
    product: str,
    target: int,
    positions: dict[str, int],
    cash: dict[str, float],
    bid: int,
    bid_volume: int,
    ask: int,
    ask_volume: int,
    config: StrategyConfig,
) -> tuple[int, int]:
    desired_delta = clamp(target, -LIMIT, LIMIT) - positions[product]
    if desired_delta == 0:
        return 0, 0

    if desired_delta > 0:
        room = max(0, LIMIT - positions[product])
        quantity = min(desired_delta, ask_volume, room, config.max_order_size)
        if quantity <= 0:
            return 0, 0
        positions[product] += quantity
        cash[product] -= ask * quantity
        return quantity, 1

    room = max(0, LIMIT + positions[product])
    quantity = min(-desired_delta, bid_volume, room, config.max_order_size)
    if quantity <= 0:
        return 0, 0
    positions[product] -= quantity
    cash[product] += bid * quantity
    return quantity, 1


def simulate_day(data: BacktestData, config: StrategyConfig) -> DayResult:
    state = SimState()
    positions = {product: 0 for product in TRANSLATORS}
    cash = {product: 0.0 for product in TRANSLATORS}
    traded_units = {product: 0 for product in TRANSLATORS}
    trade_count = {product: 0 for product in TRANSLATORS}
    cap_dwell = {product: 0 for product in TRANSLATORS}
    cap_dwell_long = {product: 0 for product in TRANSLATORS}
    cap_dwell_short = {product: 0 for product in TRANSLATORS}
    spread_paid = {product: 0.0 for product in TRANSLATORS}
    product_paths = {product: [] for product in TRANSLATORS}
    pnl_path: list[float] = []
    timestamps: list[int] = []
    last_mid = {product: 0.0 for product in TRANSLATORS}
    last_bid = {product: 0 for product in TRANSLATORS}
    last_ask = {product: 0 for product in TRANSLATORS}

    for timestamp in sorted(data.prices.keys()):
        rows = data.prices[timestamp]
        if any(product not in rows for product in TRANSLATORS):
            continue

        bids: dict[str, int] = {}
        asks: dict[str, int] = {}
        bid_volumes: dict[str, int] = {}
        ask_volumes: dict[str, int] = {}
        mids: dict[str, float] = {}
        complete = True
        for product in TRANSLATORS:
            row = rows[product]
            if not row.bid_prices or not row.ask_prices:
                complete = False
                break
            bids[product] = row.bid_prices[0]
            asks[product] = row.ask_prices[0]
            bid_volumes[product] = row.bid_volumes[0]
            ask_volumes[product] = row.ask_volumes[0]
            mids[product] = (bids[product] + asks[product]) / 2.0
        if not complete:
            continue

        last_mid = dict(mids)
        last_bid = dict(bids)
        last_ask = dict(asks)
        timestamps.append(timestamp)

        product_pnl = {product: cash[product] + positions[product] * mids[product] for product in TRANSLATORS}
        for product in TRANSLATORS:
            product_paths[product].append(product_pnl[product])
        pnl_path.append(sum(product_pnl.values()))

        raw_group_mean = sum(mids.values()) / len(TRANSLATORS)
        raw_group = {product: (mids[product] - raw_group_mean) * RESIDUAL_SCALE for product in TRANSLATORS}

        residuals: dict[str, float] = {}
        pair_left, pair_right = config.pair_products
        pair_residual = (mids[pair_left] - mids[pair_right]) * RESIDUAL_SCALE
        if config.mode in {"group", "combo"}:
            residuals = group_residuals(mids, state, config)
            next_targets = apply_group_targets(residuals, state, config)
        else:
            next_targets = {product: 0 for product in TRANSLATORS}

        if config.mode in {"pair", "combo"}:
            pair_targets, pair_residual, _ = apply_pair_targets(mids, state, config)
            next_targets = combine_targets(next_targets, pair_targets, config) if config.mode == "combo" else pair_targets

        for product in TRANSLATORS:
            if product not in config.active_products and config.mode == "group":
                next_targets[product] = 0
            if not spread_allowed(product, bids, asks, state, config):
                next_targets[product] = state.targets[product]
            next_targets[product] = reduce_for_spread(product, next_targets[product], bids, asks, state, config)
            if not edge_allowed(product, next_targets[product], residuals, bids, asks, state, config):
                next_targets[product] = state.targets[product]

        for product in TRANSLATORS:
            units, count = execute_to_target(
                product,
                next_targets[product],
                positions,
                cash,
                bids[product],
                bid_volumes[product],
                asks[product],
                ask_volumes[product],
                config,
            )
            traded_units[product] += units
            trade_count[product] += count
            spread_paid[product] += units * (asks[product] - bids[product])

        for product in TRANSLATORS:
            if state.cooldown[product] > 0:
                state.cooldown[product] -= 1
            state.mids[product].append(mids[product])
            state.spreads[product].append(asks[product] - bids[product])
            state.raw_group_residuals[product].append(raw_group[product])
            if product in residuals:
                state.residuals[product].append(residuals[product])
            if abs(positions[product]) >= LIMIT:
                cap_dwell[product] += 1
            if positions[product] >= LIMIT:
                cap_dwell_long[product] += 1
            elif positions[product] <= -LIMIT:
                cap_dwell_short[product] += 1
        state.pair_residuals.append(pair_residual)
        state.targets = dict(next_targets)

    final_product_pnl = {product: product_paths[product][-1] if product_paths[product] else 0.0 for product in TRANSLATORS}
    forced_flatten_pnl = 0.0
    for product in TRANSLATORS:
        flatten_price = last_bid[product] if positions[product] > 0 else last_ask[product] if positions[product] < 0 else last_mid[product]
        forced_flatten_pnl += cash[product] + positions[product] * flatten_price

    return DayResult(
        day=data.day_num,
        timestamps=timestamps,
        pnl_path=pnl_path,
        product_paths=product_paths,
        final_product_pnl=final_product_pnl,
        traded_units=traded_units,
        trade_count=trade_count,
        cap_dwell=cap_dwell,
        cap_dwell_long=cap_dwell_long,
        cap_dwell_short=cap_dwell_short,
        spread_paid=spread_paid,
        final_positions=dict(positions),
        forced_flatten_pnl=forced_flatten_pnl,
        last_mid=last_mid,
        last_bid=last_bid,
        last_ask=last_ask,
    )


def simulate_config(config: StrategyConfig, cached: dict[int, BacktestData], days: tuple[int, ...] = DEFAULT_DAYS) -> dict[str, Any]:
    day_results = [simulate_day(cached[day], config) for day in days]
    day_pnl = {f"day_{result.day}": sum(result.final_product_pnl.values()) for result in day_results}
    day_units = {f"day_{result.day}": sum(result.traded_units.values()) for result in day_results}
    merged_pnl = sum(day_pnl.values())
    per_product_pnl = {product: sum(result.final_product_pnl[product] for result in day_results) for product in TRANSLATORS}
    traded_units = {product: sum(result.traded_units[product] for result in day_results) for product in TRANSLATORS}
    trade_count = {product: sum(result.trade_count[product] for result in day_results) for product in TRANSLATORS}
    cap_dwell = {product: sum(result.cap_dwell[product] for result in day_results) for product in TRANSLATORS}
    cap_dwell_long = {product: sum(result.cap_dwell_long[product] for result in day_results) for product in TRANSLATORS}
    cap_dwell_short = {product: sum(result.cap_dwell_short[product] for result in day_results) for product in TRANSLATORS}
    spread_paid = {product: sum(result.spread_paid[product] for result in day_results) for product in TRANSLATORS}
    avg_spread_paid = {
        product: (spread_paid[product] / traded_units[product] if traded_units[product] else 0.0)
        for product in TRANSLATORS
    }
    total_units = sum(traded_units.values())
    total_trades = sum(trade_count.values())

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

    drawdown = path_drawdown(stitched_path, stitched_timestamps)
    roll1000, worst1000 = rolling_slice_stats(day_results, 1000, 500)
    roll2000, worst2000 = rolling_slice_stats(day_results, 2000, 1000)
    forced_flatten_pnl = sum(result.forced_flatten_pnl for result in day_results)
    final_positions = {f"day_{result.day}": result.final_positions for result in day_results}
    tick_stress_by_day = {
        str(cost): {day: round(day_pnl[day] - cost * day_units[day], 2) for day in day_pnl}
        for cost in (1, 3, 5, 10)
    }

    return {
        "label": config.label,
        "mode": config.mode,
        "merged_pnl": round(merged_pnl, 2),
        "day_2": round(day_pnl.get("day_2", 0.0), 2),
        "day_3": round(day_pnl.get("day_3", 0.0), 2),
        "day_4": round(day_pnl.get("day_4", 0.0), 2),
        "traded_units": total_units,
        "trade_count": total_trades,
        "stress_1": round(merged_pnl - total_units, 2),
        "stress_3": round(merged_pnl - 3 * total_units, 2),
        "stress_5": round(merged_pnl - 5 * total_units, 2),
        "stress_10": round(merged_pnl - 10 * total_units, 2),
        "stress_by_day": json.dumps(tick_stress_by_day, separators=(",", ":")),
        "max_drawdown": round(float(drawdown["max_drawdown"]), 2),
        "path_trough": round(float(drawdown["trough"]), 2),
        "roll1000_count": roll1000["count"],
        "roll1000_p01": round(roll1000["p01"], 2),
        "roll1000_p05": round(roll1000["p05"], 2),
        "roll1000_median": round(roll1000["median"], 2),
        "roll1000_min": round(roll1000["min"], 2),
        "roll1000_max": round(roll1000["max"], 2),
        "roll1000_worst_dd": round(roll1000["worst_dd"], 2),
        "roll2000_count": roll2000["count"],
        "roll2000_p01": round(roll2000["p01"], 2),
        "roll2000_p05": round(roll2000["p05"], 2),
        "roll2000_median": round(roll2000["median"], 2),
        "roll2000_min": round(roll2000["min"], 2),
        "roll2000_max": round(roll2000["max"], 2),
        "roll2000_worst_dd": round(roll2000["worst_dd"], 2),
        "forced_flatten_pnl": round(forced_flatten_pnl, 2),
        "forced_flatten_cost": round(merged_pnl - forced_flatten_pnl, 2),
        "per_product_pnl": json.dumps({p: round(per_product_pnl[p], 2) for p in TRANSLATORS}, separators=(",", ":")),
        "traded_units_by_product": json.dumps(traded_units, separators=(",", ":")),
        "trade_count_by_product": json.dumps(trade_count, separators=(",", ":")),
        "final_positions": json.dumps(final_positions, separators=(",", ":")),
        "cap_dwell": json.dumps(cap_dwell, separators=(",", ":")),
        "cap_dwell_long": json.dumps(cap_dwell_long, separators=(",", ":")),
        "cap_dwell_short": json.dumps(cap_dwell_short, separators=(",", ":")),
        "avg_spread_paid": json.dumps({p: round(avg_spread_paid[p], 4) for p in TRANSLATORS}, separators=(",", ":")),
        "worst1000": json.dumps(worst1000, separators=(",", ":")),
        "worst2000": json.dumps(worst2000, separators=(",", ":")),
        "window": config.window,
        "entry_z": config.entry_z,
        "min_history": config.min_history,
        "hold_to_flip": config.hold_to_flip,
        "exit_z": config.exit_z,
        "exit_rule": config.exit_rule,
        "residual_kind": config.residual_kind,
        "active_products": json.dumps(config.active_products, separators=(",", ":")),
        "target_size": config.target_size,
        "product_target_caps": json.dumps(dict(config.product_target_caps), separators=(",", ":")),
        "spread_gate": config.spread_gate,
        "spread_rule": config.spread_rule,
        "edge_gate_buffer": config.edge_gate_buffer,
        "confirm_ticks": config.confirm_ticks,
        "reentry_delay": config.reentry_delay,
        "flip_buffer": config.flip_buffer,
        "pair_products": json.dumps(config.pair_products, separators=(",", ":")),
        "pair_window": config.pair_window,
        "pair_entry_z": config.pair_entry_z,
        "pair_exit_z": config.pair_exit_z,
        "pair_hold_to_flip": config.pair_hold_to_flip,
    }


def baseline_config(label: str = "group_w1200_z1.75_mh300_hf") -> StrategyConfig:
    return StrategyConfig(label=label, window=1200, entry_z=1.75, min_history=300, history_limit=1200)


def candidate_configs() -> list[StrategyConfig]:
    base = baseline_config()
    return [
        base,
        StrategyConfig(label="group_w1000_z1.75_mh300_hf", window=1000, entry_z=1.75, min_history=300, history_limit=1000),
        StrategyConfig(label="group_w1400_z1.75_mh350_hf", window=1400, entry_z=1.75, min_history=350, history_limit=1400),
        StrategyConfig(label="group_w1600_z1.75_mh400_hf", window=1600, entry_z=1.75, min_history=400, history_limit=1600),
        StrategyConfig(label="group_w1200_z2_mh300_hf", window=1200, entry_z=2.0, min_history=300, history_limit=1200),
        replace(base, label="group_w1200_z1.75_target9", target_size=9),
        replace(base, label="group_w1200_z1.75_target8", target_size=8),
        replace(base, label="group_w1200_z1.75_cap_sg_vb5", product_target_caps=(("TRANSLATOR_SPACE_GRAY", 5), ("TRANSLATOR_VOID_BLUE", 5))),
        replace(base, label="group_w1200_z1.75_resid_leave_one_out", residual_kind="leave_one_out"),
        replace(base, label="group_w1200_z1.75_spread_p75", spread_rule="p75"),
        replace(base, label="group_w1200_z1.75_confirm2", confirm_ticks=2),
    ]


def neighbourhood_configs() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []
    for window in (500, 600, 800, 1000, 1200, 1400, 1600, 2000):
        min_histories = {window // 4, window // 3, 300}
        for entry_z in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
            for min_history in sorted(min_histories):
                configs.append(
                    StrategyConfig(
                        label=f"group_w{window}_z{entry_z:g}_mh{min_history}_hf",
                        window=window,
                        entry_z=entry_z,
                        min_history=min_history,
                        history_limit=window,
                    )
                )
    return configs


def ablation_configs(base: StrategyConfig) -> list[StrategyConfig]:
    configs = [
        replace(base, label=f"{base.label}_all5"),
        replace(base, label=f"{base.label}_no_space_gray", active_products=tuple(p for p in TRANSLATORS if p != "TRANSLATOR_SPACE_GRAY")),
        replace(base, label=f"{base.label}_no_void_blue", active_products=tuple(p for p in TRANSLATORS if p != "TRANSLATOR_VOID_BLUE")),
        replace(
            base,
            label=f"{base.label}_no_space_gray_void_blue",
            active_products=tuple(p for p in TRANSLATORS if p not in {"TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"}),
        ),
        replace(base, label=f"{base.label}_cap_space_gray5", product_target_caps=(("TRANSLATOR_SPACE_GRAY", 5),)),
        replace(base, label=f"{base.label}_cap_void_blue5", product_target_caps=(("TRANSLATOR_VOID_BLUE", 5),)),
        replace(
            base,
            label=f"{base.label}_cap_low_attr5",
            product_target_caps=(("TRANSLATOR_SPACE_GRAY", 5), ("TRANSLATOR_VOID_BLUE", 5)),
        ),
        replace(base, label=f"{base.label}_target8", target_size=8),
        replace(base, label=f"{base.label}_target9", target_size=9),
    ]
    for product in TRANSLATORS:
        active = tuple(p for p in TRANSLATORS if p != product)
        configs.append(replace(base, label=f"{base.label}_no_{product}", active_products=active))
    return configs


def variant_configs(base: StrategyConfig) -> list[StrategyConfig]:
    configs = [
        replace(base, label=f"{base.label}_resid_leave_one_out", residual_kind="leave_one_out"),
        replace(base, label=f"{base.label}_resid_product_demeaned", residual_kind="product_demeaned"),
        replace(base, label=f"{base.label}_resid_vol_normalised", residual_kind="vol_normalised"),
        replace(base, label=f"{base.label}_resid_rolling_regression", residual_kind="rolling_regression"),
        replace(base, label=f"{base.label}_resid_rank", residual_kind="rank"),
        replace(base, label=f"{base.label}_exit_flat025", hold_to_flip=False, exit_z=0.25, exit_rule="flatten"),
        replace(base, label=f"{base.label}_exit_flat05", hold_to_flip=False, exit_z=0.5, exit_rule="flatten"),
        replace(base, label=f"{base.label}_exit_flat075", hold_to_flip=False, exit_z=0.75, exit_rule="flatten"),
        replace(base, label=f"{base.label}_exit_half05", hold_to_flip=False, exit_z=0.5, exit_rule="half"),
        replace(base, label=f"{base.label}_two_stage", hold_to_flip=False, exit_rule="two_stage"),
        replace(base, label=f"{base.label}_proportional", hold_to_flip=False, exit_rule="proportional"),
        replace(base, label=f"{base.label}_spread_p50", spread_rule="p50"),
        replace(base, label=f"{base.label}_spread_p60", spread_rule="p60"),
        replace(base, label=f"{base.label}_spread_p75", spread_rule="p75"),
        replace(base, label=f"{base.label}_spread_skip_p90", spread_rule="skip_p90"),
        replace(base, label=f"{base.label}_spread_reduce_p75", spread_rule="reduce_p75"),
        replace(base, label=f"{base.label}_edge_half_spread", edge_gate_buffer=0.0),
        replace(base, label=f"{base.label}_edge_half_spread_plus1", edge_gate_buffer=1.0),
        replace(base, label=f"{base.label}_confirm2", confirm_ticks=2),
        replace(base, label=f"{base.label}_confirm3", confirm_ticks=3),
        replace(base, label=f"{base.label}_flip_buffer025", flip_buffer=0.25),
        replace(base, label=f"{base.label}_flip_buffer05", flip_buffer=0.5),
        replace(base, label=f"{base.label}_confirm2_flip_buffer025", confirm_ticks=2, flip_buffer=0.25),
        replace(base, label=f"{base.label}_reentry_delay2", reentry_delay=2),
    ]
    return configs


def pair_neighbourhood_configs() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []
    for left, right in PAIR_PRODUCTS:
        pair_name = f"{left.removeprefix('TRANSLATOR_')}_{right.removeprefix('TRANSLATOR_')}"
        for window, entry_z, exit_z, hold in (
            (1200, 2.0, None, True),
            (1200, 2.0, 0.3, False),
            (2000, 1.5, None, True),
            (2000, 1.5, 0.3, False),
        ):
            configs.append(
                StrategyConfig(
                    label=f"pair_{pair_name}_w{window}_z{entry_z:g}_{'hf' if hold else 'exit03'}",
                    mode="pair",
                    window=window,
                    entry_z=entry_z,
                    min_history=max(300, window // 4),
                    history_limit=window,
                    hold_to_flip=hold,
                    exit_z=exit_z,
                    pair_products=(left, right),
                )
            )
    return configs


def serious_followup_configs() -> list[StrategyConfig]:
    base = baseline_config()
    return [
        base,
        replace(base, label="serious_leave_one_out", residual_kind="leave_one_out"),
        replace(base, label="serious_target9", target_size=9),
        replace(base, label="serious_target8", target_size=8),
        replace(base, label="serious_cap_low_attr5", product_target_caps=(("TRANSLATOR_SPACE_GRAY", 5), ("TRANSLATOR_VOID_BLUE", 5))),
        replace(base, label="serious_spread_p75", spread_rule="p75"),
        replace(base, label="serious_confirm2", confirm_ticks=2),
        StrategyConfig(label="serious_w1000_z1.75_mh300", window=1000, entry_z=1.75, min_history=300, history_limit=1000),
        StrategyConfig(label="serious_w1400_z1.75_mh350", window=1400, entry_z=1.75, min_history=350, history_limit=1400),
        StrategyConfig(label="serious_w1600_z1.75_mh400", window=1600, entry_z=1.75, min_history=400, history_limit=1600),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_top(name: str, rows: list[dict[str, Any]], limit: int = 12) -> None:
    print(f"\n{name}")
    for row in sorted(rows, key=lambda item: float(item["stress_5"]), reverse=True)[:limit]:
        print(
            f"{row['label'][:42]:42s} pnl={float(row['merged_pnl']):9.0f} "
            f"days=({float(row['day_2']):7.0f},{float(row['day_3']):7.0f},{float(row['day_4']):7.0f}) "
            f"+5={float(row['stress_5']):9.0f} +10={float(row['stress_10']):9.0f} "
            f"dd={float(row['max_drawdown']):8.0f} p05={float(row['roll1000_p05']):8.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="research_outputs")
    parser.add_argument("--days", nargs="*", type=int, default=list(DEFAULT_DAYS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = tuple(args.days)
    cached = load_data(days)
    out_dir = ROOT / args.out_dir

    primary_rows = [simulate_config(config, cached, days) for config in candidate_configs()]
    write_csv(out_dir / "r5_translator_primary.csv", primary_rows)
    print_top("Primary candidates", primary_rows)

    neighbourhood_rows = [simulate_config(config, cached, days) for config in neighbourhood_configs()]
    write_csv(out_dir / "r5_translator_neighbourhood.csv", neighbourhood_rows)
    print_top("Group neighbourhood", neighbourhood_rows)

    pair_rows = [simulate_config(config, cached, days) for config in pair_neighbourhood_configs()]
    write_csv(out_dir / "r5_translator_pair_neighbourhood.csv", pair_rows)
    print_top("Pair neighbourhood", pair_rows)

    base = baseline_config()
    ablation_rows = [simulate_config(config, cached, days) for config in ablation_configs(base)]
    write_csv(out_dir / "r5_translator_ablations.csv", ablation_rows)
    print_top("Product ablations", ablation_rows)

    variant_rows = [simulate_config(config, cached, days) for config in variant_configs(base)]
    write_csv(out_dir / "r5_translator_variants.csv", variant_rows)
    print_top("Residual, exit, execution variants", variant_rows)

    serious_rows = [simulate_config(config, cached, days) for config in serious_followup_configs()]
    write_csv(out_dir / "r5_translator_serious.csv", serious_rows)
    print_top("Serious follow-up candidates", serious_rows)

    summary = {
        "primary": primary_rows,
        "best_neighbourhood_by_stress5": sorted(neighbourhood_rows, key=lambda row: float(row["stress_5"]), reverse=True)[:20],
        "best_pair_by_stress5": sorted(pair_rows, key=lambda row: float(row["stress_5"]), reverse=True)[:20],
        "ablations": ablation_rows,
        "variants": variant_rows,
        "serious": serious_rows,
    }
    (out_dir / "r5_translator_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote Translator research outputs to {out_dir}")


if __name__ == "__main__":
    main()
