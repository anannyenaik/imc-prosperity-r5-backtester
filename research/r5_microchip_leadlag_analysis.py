from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "r5_data" / "round5"
OUTPUT_DIR = ROOT / "research_outputs"

DAYS = (2, 3, 4)
MICROCHIPS = (
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_TRIANGLE",
)
FOLLOWER_LAGS = {
    "MICROCHIP_OVAL": 50,
    "MICROCHIP_SQUARE": 100,
    "MICROCHIP_RECTANGLE": 150,
    "MICROCHIP_TRIANGLE": 200,
}
RAW_THRESHOLDS = (15.0, 20.0, 24.0, 26.0, 28.0, 30.0, 32.0, 35.0)
Z_THRESHOLDS = (2.0, 2.5, 3.0, 3.5)
RAW_K_VALUES = (1, 2, 3, 5, 10)
Z_WINDOWS = (250, 500, 1000)
OFFSET_DELTAS = (-1, 0, 1)
HOLD_TICKS = (1, 2, 3, 5, 10)
POSITION_LIMIT = 10
TARGET_SIZE = 10
MAX_ORDER_SIZE = 10


@dataclass(frozen=True)
class Snapshot:
    timestamp: int
    best_bid: Optional[int]
    bid_volume: int
    best_ask: Optional[int]
    ask_volume: int
    mid: Optional[float]


@dataclass(frozen=True)
class DaySeries:
    day: int
    timestamps: list[int]
    snapshots: dict[str, list[Snapshot]]
    mids: dict[str, list[float]]
    raw_returns: dict[str, list[float]]
    log_returns: dict[str, list[float]]
    residual_returns: dict[str, list[float]]


@dataclass(frozen=True)
class SimConfig:
    label: str
    products: tuple[str, ...]
    off_delta: int
    signal_kind: str
    k: int
    z_window: int
    threshold: float
    mode: str
    hold_ticks: int


@dataclass
class SimResult:
    label: str
    products: tuple[str, ...]
    total: float
    by_day: dict[int, float]
    by_product: dict[str, float]
    max_drawdown: float
    trades: int
    units: int
    units_by_day: dict[int, int]
    forced_flatten: float
    final_positions: dict[int, dict[str, int]]
    roll1000_min: float
    roll2000_min: float
    config: SimConfig


_DAY_CACHE: dict[int, DaySeries] = {}
_ROLLING_STD_CACHE: dict[int, dict[int, list[Optional[float]]]] = {}


def _to_int(value: str) -> Optional[int]:
    if value == "":
        return None
    return int(value)


def _corr(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 1e-12 or var_right <= 1e-12:
        return 0.0
    covariance = sum((left[index] - mean_left) * (right[index] - mean_right) for index in range(len(left)))
    return covariance / math.sqrt(var_left * var_right)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _safe_round(value: float, places: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    return round(value, places)


def load_day(day: int) -> DaySeries:
    if day in _DAY_CACHE:
        return _DAY_CACHE[day]

    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    rows: dict[str, dict[int, Snapshot]] = {product: {} for product in MICROCHIPS}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            product = row["product"]
            if product not in rows:
                continue
            best_bid = _to_int(row["bid_price_1"])
            best_ask = _to_int(row["ask_price_1"])
            bid_volume = int(row["bid_volume_1"]) if row["bid_volume_1"] else 0
            ask_volume = int(row["ask_volume_1"]) if row["ask_volume_1"] else 0
            mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
            rows[product][int(row["timestamp"])] = Snapshot(
                timestamp=int(row["timestamp"]),
                best_bid=best_bid,
                bid_volume=bid_volume,
                best_ask=best_ask,
                ask_volume=ask_volume,
                mid=mid,
            )

    common_timestamps = sorted(set.intersection(*(set(rows[product]) for product in MICROCHIPS)))
    snapshots: dict[str, list[Snapshot]] = {product: [] for product in MICROCHIPS}
    mids: dict[str, list[float]] = {product: [] for product in MICROCHIPS}
    kept_timestamps: list[int] = []
    for timestamp in common_timestamps:
        current = {product: rows[product][timestamp] for product in MICROCHIPS}
        if any(snapshot.mid is None for snapshot in current.values()):
            continue
        kept_timestamps.append(timestamp)
        for product, snapshot in current.items():
            snapshots[product].append(snapshot)
            mids[product].append(float(snapshot.mid))

    raw_returns = {
        product: [values[index] - values[index - 1] for index in range(1, len(values))]
        for product, values in mids.items()
    }
    log_returns = {
        product: [math.log(values[index]) - math.log(values[index - 1]) for index in range(1, len(values))]
        for product, values in mids.items()
    }
    residuals = {product: [] for product in MICROCHIPS}
    for index in range(len(kept_timestamps)):
        group_mean = sum(mids[product][index] for product in MICROCHIPS) / len(MICROCHIPS)
        for product in MICROCHIPS:
            residuals[product].append(mids[product][index] - group_mean)
    residual_returns = {
        product: [values[index] - values[index - 1] for index in range(1, len(values))]
        for product, values in residuals.items()
    }

    series = DaySeries(
        day=day,
        timestamps=kept_timestamps,
        snapshots=snapshots,
        mids=mids,
        raw_returns=raw_returns,
        log_returns=log_returns,
        residual_returns=residual_returns,
    )
    _DAY_CACHE[day] = series
    return series


def _returns_for_kind(series: DaySeries, kind: str) -> dict[str, list[float]]:
    if kind == "raw":
        return series.raw_returns
    if kind == "log":
        return series.log_returns
    if kind == "residual":
        return series.residual_returns
    raise ValueError(kind)


def lag_corr(
    circle_returns: list[float],
    target_returns: list[float],
    lag: int,
    *,
    trim_outliers: bool = False,
) -> float:
    if lag < 0 or len(circle_returns) <= lag or len(target_returns) <= lag:
        return 0.0
    left = circle_returns[: len(circle_returns) - lag]
    right = target_returns[lag:]
    if trim_outliers:
        x_cut = _percentile([abs(value) for value in left], 0.995)
        y_cut = _percentile([abs(value) for value in right], 0.995)
        pairs = [(x, y) for x, y in zip(left, right) if abs(x) <= x_cut and abs(y) <= y_cut]
        left = [x for x, _ in pairs]
        right = [y for _, y in pairs]
    return _corr(left, right)


def _lag_window(values: dict[int, float], centre: int, radius: int) -> dict[int, float]:
    return {lag: values.get(lag, 0.0) for lag in range(centre - radius, centre + radius + 1)}


def _window_summary(values: dict[int, float], expected_lag: int) -> dict[str, Any]:
    plus_minus_5 = _lag_window(values, expected_lag, 5)
    plus_minus_10 = _lag_window(values, expected_lag, 10)
    best_lag, best_corr = max(plus_minus_10.items(), key=lambda item: abs(item[1]))
    same_sign_5 = sum(1 for value in plus_minus_5.values() if value * values[expected_lag] > 0)
    same_sign_10 = sum(1 for value in plus_minus_10.values() if value * values[expected_lag] > 0)
    return {
        "expected_corr": values[expected_lag],
        "best_lag_pm10": best_lag,
        "best_corr_pm10": best_corr,
        "same_sign_pm5": same_sign_5,
        "same_sign_pm10": same_sign_10,
        "pm5": {str(lag): _safe_round(corr) for lag, corr in plus_minus_5.items()},
        "pm10": {str(lag): _safe_round(corr) for lag, corr in plus_minus_10.items()},
    }


def _pooled_returns(kind: str, days: Iterable[int]) -> dict[str, list[float]]:
    pooled = {product: [] for product in MICROCHIPS}
    for day in days:
        returns = _returns_for_kind(load_day(day), kind)
        for product in MICROCHIPS:
            pooled[product].extend(returns[product])
    return pooled


def verify_lead_lag() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("raw", "log", "residual"):
        kind_result: dict[str, Any] = {}
        day_returns = {day: _returns_for_kind(load_day(day), kind) for day in DAYS}
        pooled = _pooled_returns(kind, DAYS)
        for target, expected_lag in FOLLOWER_LAGS.items():
            target_result: dict[str, Any] = {}
            for day in DAYS:
                lag_values = {
                    lag: lag_corr(day_returns[day]["MICROCHIP_CIRCLE"], day_returns[day][target], lag)
                    for lag in range(max(0, expected_lag - 20), expected_lag + 21)
                }
                target_result[f"day_{day}"] = _window_summary(lag_values, expected_lag)

                half_length = len(day_returns[day]["MICROCHIP_CIRCLE"]) // 2
                first_circle = day_returns[day]["MICROCHIP_CIRCLE"][:half_length]
                first_target = day_returns[day][target][:half_length]
                second_circle = day_returns[day]["MICROCHIP_CIRCLE"][half_length:]
                second_target = day_returns[day][target][half_length:]
                target_result[f"day_{day}"]["halves"] = {
                    "first": _safe_round(lag_corr(first_circle, first_target, expected_lag)),
                    "second": _safe_round(lag_corr(second_circle, second_target, expected_lag)),
                }
                target_result[f"day_{day}"]["trimmed_expected_corr"] = _safe_round(
                    lag_corr(
                        day_returns[day]["MICROCHIP_CIRCLE"],
                        day_returns[day][target],
                        expected_lag,
                        trim_outliers=True,
                    )
                )

            pooled_values = {
                lag: lag_corr(pooled["MICROCHIP_CIRCLE"], pooled[target], lag)
                for lag in range(max(0, expected_lag - 20), expected_lag + 21)
            }
            target_result["pooled"] = _window_summary(pooled_values, expected_lag)
            target_result["pooled"]["trimmed_expected_corr"] = _safe_round(
                lag_corr(
                    pooled["MICROCHIP_CIRCLE"],
                    pooled[target],
                    expected_lag,
                    trim_outliers=True,
                )
            )
            kind_result[target] = target_result
        out[kind] = kind_result
    return out


class RollingStd:
    def __init__(self, window: int) -> None:
        self.window = window
        self.values: deque[float] = deque()
        self.total = 0.0
        self.total_square = 0.0

    def push(self, value: float) -> None:
        self.values.append(value)
        self.total += value
        self.total_square += value * value
        if len(self.values) > self.window:
            old = self.values.popleft()
            self.total -= old
            self.total_square -= old * old

    def std(self) -> Optional[float]:
        if len(self.values) < self.window:
            return None
        mean = self.total / len(self.values)
        variance = max(0.0, self.total_square / len(self.values) - mean * mean)
        std = math.sqrt(variance)
        return std if std > 1e-12 else None


def _circle_signal(
    series: DaySeries,
    current_index: int,
    product: str,
    config: SimConfig,
    rolling_stds: dict[int, list[Optional[float]]],
) -> float:
    base_lag = FOLLOWER_LAGS[product]
    offset = base_lag + config.off_delta
    event_index = current_index - offset
    if event_index - config.k < 0:
        return 0.0
    signal = series.mids["MICROCHIP_CIRCLE"][event_index] - series.mids["MICROCHIP_CIRCLE"][event_index - config.k]
    if config.signal_kind == "raw":
        return signal
    if config.signal_kind == "z":
        std = rolling_stds[config.z_window][event_index]
        if std is None:
            return 0.0
        return signal / (std * math.sqrt(config.k))
    raise ValueError(config.signal_kind)


def _rolling_std_by_index(series: DaySeries, windows: Iterable[int]) -> dict[int, list[Optional[float]]]:
    cached = _ROLLING_STD_CACHE.get(series.day)
    if cached is not None and all(window in cached for window in windows):
        return {window: cached[window] for window in windows}

    returns_by_end_index = [0.0] * len(series.timestamps)
    mids = series.mids["MICROCHIP_CIRCLE"]
    for index in range(1, len(mids)):
        returns_by_end_index[index] = mids[index] - mids[index - 1]

    out: dict[int, list[Optional[float]]] = {}
    for window in windows:
        stats = RollingStd(window)
        values: list[Optional[float]] = [None] * len(series.timestamps)
        for index, value in enumerate(returns_by_end_index):
            values[index] = stats.std()
            if index > 0:
                stats.push(value)
        out[window] = values
    _ROLLING_STD_CACHE.setdefault(series.day, {}).update(out)
    return out


def _order_to_target(
    snapshot: Snapshot,
    position: int,
    target: int,
    *,
    fill_offset: int,
) -> tuple[int, float, int, int]:
    target = max(-POSITION_LIMIT, min(POSITION_LIMIT, target))
    desired = target - position
    if desired == 0:
        return position, 0.0, 0, 0
    if desired > 0:
        if snapshot.best_ask is None:
            return position, 0.0, 0, 0
        quantity = min(desired, max(0, snapshot.ask_volume), max(0, POSITION_LIMIT - position), MAX_ORDER_SIZE)
        if quantity <= 0:
            return position, 0.0, 0, 0
        return position + quantity, -quantity * (snapshot.best_ask + fill_offset), 1, quantity

    if snapshot.best_bid is None:
        return position, 0.0, 0, 0
    quantity = min(-desired, max(0, snapshot.bid_volume), max(0, POSITION_LIMIT + position), MAX_ORDER_SIZE)
    if quantity <= 0:
        return position, 0.0, 0, 0
    return position - quantity, quantity * (snapshot.best_bid - fill_offset), 1, quantity


def _worst_rolling_delta(levels: list[float], window: int) -> float:
    if len(levels) <= window:
        return 0.0
    worst = 0.0
    for index in range(0, len(levels) - window):
        delta = levels[index + window] - levels[index]
        if delta < worst:
            worst = delta
    return worst


def simulate_config(config: SimConfig, *, days: tuple[int, ...] = DAYS, fill_offset: int = 0) -> SimResult:
    by_day: dict[int, float] = {}
    by_product = {product: 0.0 for product in config.products}
    units_by_day = {day: 0 for day in days}
    final_positions: dict[int, dict[str, int]] = {}
    total_trades = 0
    total_units = 0
    forced_flatten_cost = 0.0
    equity_levels: list[float] = []
    equity_offset = 0.0
    high_water = 0.0
    max_drawdown = 0.0

    for day in days:
        series = load_day(day)
        rolling_stds = _rolling_std_by_index(series, Z_WINDOWS)
        positions = {product: 0 for product in config.products}
        targets = {product: 0 for product in config.products}
        cash = {product: 0.0 for product in config.products}
        expiry = {product: -1 for product in config.products}

        for index in range(len(series.timestamps)):
            next_targets = dict(targets)
            for product in config.products:
                signal = _circle_signal(series, index, product, config, rolling_stds)
                event_target = 0
                if signal > config.threshold:
                    event_target = TARGET_SIZE
                elif signal < -config.threshold:
                    event_target = -TARGET_SIZE

                if config.mode == "hold_to_next_signal":
                    if event_target and event_target != targets[product]:
                        next_targets[product] = event_target
                else:
                    if event_target:
                        next_targets[product] = event_target
                        expiry[product] = index + config.hold_ticks
                    elif index >= expiry[product]:
                        next_targets[product] = 0

            for product in config.products:
                snapshot = series.snapshots[product][index]
                new_position, cash_delta, trade_count, unit_count = _order_to_target(
                    snapshot,
                    positions[product],
                    next_targets[product],
                    fill_offset=fill_offset,
                )
                if trade_count:
                    positions[product] = new_position
                    cash[product] += cash_delta
                    total_trades += trade_count
                    total_units += unit_count
                    units_by_day[day] += unit_count
                targets[product] = next_targets[product]

            mtm = sum(cash[product] + positions[product] * series.mids[product][index] for product in config.products)
            stitched = equity_offset + mtm
            equity_levels.append(stitched)
            high_water = max(high_water, stitched)
            max_drawdown = max(max_drawdown, high_water - stitched)

        day_pnl = 0.0
        for product in config.products:
            last_index = len(series.timestamps) - 1
            product_pnl = cash[product] + positions[product] * series.mids[product][last_index]
            by_product[product] += product_pnl
            day_pnl += product_pnl

            snapshot = series.snapshots[product][last_index]
            if positions[product] > 0 and snapshot.best_bid is not None:
                forced_flatten_cost += (series.mids[product][last_index] - snapshot.best_bid) * positions[product]
            elif positions[product] < 0 and snapshot.best_ask is not None:
                forced_flatten_cost += (snapshot.best_ask - series.mids[product][last_index]) * (-positions[product])

        by_day[day] = day_pnl
        final_positions[day] = dict(positions)
        equity_offset += day_pnl

    total = sum(by_day.values())
    return SimResult(
        label=config.label,
        products=config.products,
        total=total,
        by_day=by_day,
        by_product=by_product,
        max_drawdown=max_drawdown,
        trades=total_trades,
        units=total_units,
        units_by_day=units_by_day,
        forced_flatten=total - forced_flatten_cost,
        final_positions=final_positions,
        roll1000_min=_worst_rolling_delta(equity_levels, 1000),
        roll2000_min=_worst_rolling_delta(equity_levels, 2000),
        config=config,
    )


def _config_label(
    products: tuple[str, ...],
    off_delta: int,
    signal_kind: str,
    k: int,
    z_window: int,
    threshold: float,
    mode: str,
    hold_ticks: int,
) -> str:
    product_part = "+".join(product.replace("MICROCHIP_", "") for product in products)
    offset_part = "Lm1" if off_delta == -1 else "L" if off_delta == 0 else "Lp1"
    if signal_kind == "raw":
        signal_part = f"raw{k:g}"
    else:
        signal_part = f"z{k:g}w{z_window}"
    mode_part = mode if mode == "hold_to_next_signal" else f"h{hold_ticks}"
    return f"{product_part}_{offset_part}_{signal_part}_thr{threshold:g}_{mode_part}"


def iter_configs(products: tuple[str, ...]) -> Iterable[SimConfig]:
    for off_delta in OFFSET_DELTAS:
        for k in RAW_K_VALUES:
            for threshold in RAW_THRESHOLDS:
                for hold_ticks in HOLD_TICKS:
                    label = _config_label(products, off_delta, "raw", k, 0, threshold, "fixed_hold", hold_ticks)
                    yield SimConfig(label, products, off_delta, "raw", k, 0, threshold, "fixed_hold", hold_ticks)
                label = _config_label(products, off_delta, "raw", k, 0, threshold, "hold_to_next_signal", 0)
                yield SimConfig(label, products, off_delta, "raw", k, 0, threshold, "hold_to_next_signal", 0)

        for z_window in Z_WINDOWS:
            for threshold in Z_THRESHOLDS:
                for hold_ticks in HOLD_TICKS:
                    label = _config_label(products, off_delta, "z", 1, z_window, threshold, "fixed_hold", hold_ticks)
                    yield SimConfig(label, products, off_delta, "z", 1, z_window, threshold, "fixed_hold", hold_ticks)
                label = _config_label(products, off_delta, "z", 1, z_window, threshold, "hold_to_next_signal", 0)
                yield SimConfig(label, products, off_delta, "z", 1, z_window, threshold, "hold_to_next_signal", 0)


def _result_row(result: SimResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "products": "+".join(result.products),
        "total": round(result.total, 2),
        "day_2": round(result.by_day.get(2, 0.0), 2),
        "day_3": round(result.by_day.get(3, 0.0), 2),
        "day_4": round(result.by_day.get(4, 0.0), 2),
        "max_drawdown": round(result.max_drawdown, 2),
        "trades": result.trades,
        "units": result.units,
        "stress_1": round(result.total - result.units, 2),
        "stress_3": round(result.total - 3 * result.units, 2),
        "stress_5": round(result.total - 5 * result.units, 2),
        "stress_10": round(result.total - 10 * result.units, 2),
        "forced_flatten": round(result.forced_flatten, 2),
        "roll1000_min": round(result.roll1000_min, 2),
        "roll2000_min": round(result.roll2000_min, 2),
        "off_delta": result.config.off_delta,
        "signal_kind": result.config.signal_kind,
        "k": result.config.k,
        "z_window": result.config.z_window,
        "threshold": result.config.threshold,
        "mode": result.config.mode,
        "hold_ticks": result.config.hold_ticks,
    }


def _promotion_prescreen(result: SimResult) -> bool:
    return (
        result.total > 0
        and all(result.by_day.get(day, 0.0) > 0 for day in DAYS)
        and result.total - 5 * result.units > 0
        and result.total - 10 * result.units > 0
        and max(result.by_day.values()) <= 0.60 * result.total
    )


def sweep_groups() -> dict[str, Any]:
    groups: dict[str, tuple[str, ...]] = {
        "standalone_oval": ("MICROCHIP_OVAL",),
        "standalone_square": ("MICROCHIP_SQUARE",),
        "standalone_rectangle": ("MICROCHIP_RECTANGLE",),
        "standalone_triangle": ("MICROCHIP_TRIANGLE",),
        "standalone_all_followers": tuple(FOLLOWER_LAGS),
        "unused_square": ("MICROCHIP_SQUARE",),
        "unused_rectangle": ("MICROCHIP_RECTANGLE",),
        "unused_square_rectangle": ("MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE"),
        "overlay_oval_triangle": ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"),
    }

    rows: list[dict[str, Any]] = []
    best_by_group: dict[str, dict[str, Any]] = {}
    passed_by_group: dict[str, list[dict[str, Any]]] = {}
    result_cache: dict[tuple[str, ...], list[SimResult]] = {}

    for group_name, products in groups.items():
        if products not in result_cache:
            result_cache[products] = [simulate_config(config) for config in iter_configs(products)]

        results = result_cache[products]
        best_result = max(results, key=lambda item: item.total) if results else None
        passed = [result for result in results if _promotion_prescreen(result)]
        for result in results:
            rows.append({"group": group_name, **_result_row(result)})

        if best_result is not None:
            best_by_group[group_name] = _result_row(best_result)
        passed_by_group[group_name] = [_result_row(result) for result in sorted(passed, key=lambda item: item.total, reverse=True)[:10]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "r5_microchip_leadlag_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "best_by_group": best_by_group,
        "passed_by_group": passed_by_group,
        "rows_written": len(rows),
    }


def neighbourhood_for_config(config_row: dict[str, Any]) -> list[dict[str, Any]]:
    products = tuple(config_row["products"].split("+"))
    base_threshold = float(config_row["threshold"])
    thresholds = sorted(set([base_threshold, *RAW_THRESHOLDS] if config_row["signal_kind"] == "raw" else [base_threshold, *Z_THRESHOLDS]))
    rows: list[dict[str, Any]] = []
    for off_delta in OFFSET_DELTAS:
        for threshold in thresholds:
            config = SimConfig(
                label=_config_label(
                    products,
                    off_delta,
                    str(config_row["signal_kind"]),
                    int(config_row["k"]),
                    int(config_row["z_window"]),
                    threshold,
                    str(config_row["mode"]),
                    int(config_row["hold_ticks"]),
                ),
                products=products,
                off_delta=off_delta,
                signal_kind=str(config_row["signal_kind"]),
                k=int(config_row["k"]),
                z_window=int(config_row["z_window"]),
                threshold=threshold,
                mode=str(config_row["mode"]),
                hold_ticks=int(config_row["hold_ticks"]),
            )
            rows.append(_result_row(simulate_config(config)))
    return rows


def _serialise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    if isinstance(value, tuple):
        return [_serialise(item) for item in value]
    if isinstance(value, float):
        return _safe_round(value)
    return value


def main() -> None:
    start = time.perf_counter()
    verification = verify_lead_lag()
    sweep = sweep_groups()
    neighbourhood: dict[str, list[dict[str, Any]]] = {}
    for group in ("unused_square", "unused_rectangle", "unused_square_rectangle", "standalone_all_followers", "overlay_oval_triangle"):
        if group in sweep["best_by_group"]:
            neighbourhood[group] = neighbourhood_for_config(sweep["best_by_group"][group])

    summary = {
        "verification": verification,
        "sweep": sweep,
        "neighbourhood": neighbourhood,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "r5_microchip_leadlag_summary.json"
    output_path.write_text(json.dumps(_serialise(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"runtime_seconds": summary["runtime_seconds"], "sweep": sweep}, indent=2, sort_keys=True))
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print("Wrote research_outputs/r5_microchip_leadlag_sweep.csv")


if __name__ == "__main__":
    main()
