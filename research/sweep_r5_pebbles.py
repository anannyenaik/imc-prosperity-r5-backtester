from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from contextlib import redirect_stdout
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from prosperity4bt import datamodel
from prosperity4bt.data import BacktestData, read_day_data
from prosperity4bt.datamodel import Observation, Trade, TradingState
from prosperity4bt.file_reader import FileSystemReader
from prosperity4bt.metrics import portfolio_pnl_by_timestamp
from prosperity4bt.models import BacktestResult, SandboxLogRow, TradeMatchingMode
from prosperity4bt.runner import create_activity_logs, enforce_limits, match_orders, prepare_state, type_check_orders


ROOT = Path(__file__).resolve().parents[1]
PEBBLES = ("PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL")
LIMITS = {product: 10 for product in PEBBLES}
DEFAULT_DAYS = (2, 3, 4)


@dataclass(frozen=True)
class Config:
    window: int
    entry_z: float
    exit_z: float | None
    target_size: int

    @property
    def hold_to_flip(self) -> bool:
        return self.exit_z is None

    @property
    def label(self) -> str:
        exit_label = "hold" if self.exit_z is None else f"exit{self.exit_z:g}"
        return f"w{self.window}_z{self.entry_z:g}_{exit_label}_t{self.target_size}"


def import_trader_module() -> Any:
    sys.modules["datamodel"] = datamodel
    strategy_dir = ROOT / "strategies" / "archive"
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    return importlib.import_module("r5_pebbles")


def apply_config(trader_class: Any, config: Config) -> None:
    trader_class.WINDOW = config.window
    trader_class.ENTRY_Z = config.entry_z
    trader_class.EXIT_Z = config.exit_z
    trader_class.HOLD_TO_FLIP = config.hold_to_flip
    trader_class.TARGET_SIZE = config.target_size
    trader_class.MAX_ORDER_SIZE = min(10, config.target_size)
    trader_class.MIN_HISTORY = max(80, config.window // 4)


def clone_trade(trade: Trade) -> Trade:
    return Trade(
        symbol=trade.symbol,
        price=trade.price,
        quantity=trade.quantity,
        buyer=trade.buyer,
        seller=trade.seller,
        timestamp=trade.timestamp,
    )


def filter_pebbles_data(data: BacktestData) -> BacktestData:
    prices = {
        timestamp: {product: row for product, row in rows.items() if product in PEBBLES}
        for timestamp, rows in data.prices.items()
    }

    trades: dict[int, dict[str, list[Trade]]] = defaultdict(lambda: defaultdict(list))
    for timestamp, by_product in data.trades.items():
        for product, product_trades in by_product.items():
            if product in PEBBLES:
                trades[timestamp][product] = [clone_trade(trade) for trade in product_trades]

    return BacktestData(
        round_num=data.round_num,
        day_num=data.day_num,
        prices=prices,
        trades=trades,
        observations=data.observations,
        products=sorted(PEBBLES),
        profit_loss={product: 0.0 for product in PEBBLES},
    )


def clone_data_for_run(template: BacktestData) -> BacktestData:
    trades: dict[int, dict[str, list[Trade]]] = defaultdict(lambda: defaultdict(list))
    for timestamp, by_product in template.trades.items():
        for product, product_trades in by_product.items():
            trades[timestamp][product] = [clone_trade(trade) for trade in product_trades]

    return BacktestData(
        round_num=template.round_num,
        day_num=template.day_num,
        prices=template.prices,
        trades=trades,
        observations=template.observations,
        products=template.products,
        profit_loss={product: 0.0 for product in template.products},
    )


def load_cached_data(days: tuple[int, ...], pebbles_only: bool) -> dict[int, BacktestData]:
    reader = FileSystemReader(ROOT / "r5_data")
    cached: dict[int, BacktestData] = {}
    for day in days:
        data = read_day_data(reader, 5, day, no_names=False)
        cached[day] = filter_pebbles_data(data) if pebbles_only else data
    return cached


def run_backtest_cached(
    trader: Any,
    template: BacktestData,
    trade_matching_mode: TradeMatchingMode,
) -> BacktestResult:
    data = clone_data_for_run(template)

    os.environ["PROSPERITY4BT_ROUND"] = str(data.round_num)
    os.environ["PROSPERITY4BT_DAY"] = str(data.day_num)

    trader_data = ""
    state = TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths={},
        own_trades={},
        market_trades={},
        position={},
        observations=Observation({}, {}),
    )

    result = BacktestResult(
        round_num=data.round_num,
        day_num=data.day_num,
        sandbox_logs=[],
        activity_logs=[],
        trades=[],
    )

    for timestamp in sorted(data.prices.keys()):
        state.timestamp = timestamp
        state.traderData = trader_data

        prepare_state(state, data)

        stdout = StringIO()
        with redirect_stdout(stdout):
            orders, conversions, trader_data = trader.run(state)
        _ = conversions

        sandbox_row = SandboxLogRow(
            timestamp=timestamp,
            sandbox_log="",
            lambda_log=stdout.getvalue().rstrip(),
        )

        result.sandbox_logs.append(sandbox_row)
        type_check_orders(orders)
        create_activity_logs(state, data, result)
        enforce_limits(state, data, orders, sandbox_row, LIMITS)
        match_orders(state, data, orders, result, trade_matching_mode, LIMITS)

    return result


def final_product_pnl(result: BacktestResult) -> dict[str, float]:
    last_timestamp = result.activity_logs[-1].timestamp
    out: dict[str, float] = {}
    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break
        product = row.columns[2]
        if product in PEBBLES:
            out[product] = float(row.columns[-1])
    return out


def final_mid_prices(result: BacktestResult) -> dict[str, float]:
    last_timestamp = result.activity_logs[-1].timestamp
    out: dict[str, float] = {}
    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break
        product = row.columns[2]
        if product in PEBBLES:
            out[product] = float(row.columns[15])
    return out


def own_trade_rows(result: BacktestResult) -> list[Any]:
    rows = []
    for row in result.trades:
        trade = row.trade
        if trade.symbol in PEBBLES and (trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION"):
            rows.append(row)
    return rows


def positions_for_result(result: BacktestResult) -> dict[str, int]:
    positions = {product: 0 for product in PEBBLES}
    for row in own_trade_rows(result):
        trade = row.trade
        if trade.buyer == "SUBMISSION":
            positions[trade.symbol] += trade.quantity
        elif trade.seller == "SUBMISSION":
            positions[trade.symbol] -= trade.quantity
    return positions


def positions_by_day(results: list[BacktestResult]) -> dict[str, dict[str, int]]:
    return {f"day_{result.day_num}": positions_for_result(result) for result in results}


def trade_stats(results: list[BacktestResult]) -> tuple[int, dict[str, int], dict[str, int]]:
    traded_units = 0
    trade_count = {product: 0 for product in PEBBLES}
    traded_units_by_product = {product: 0 for product in PEBBLES}
    for result in results:
        for row in own_trade_rows(result):
            trade = row.trade
            traded_units += trade.quantity
            trade_count[trade.symbol] += 1
            traded_units_by_product[trade.symbol] += trade.quantity
    return traded_units, trade_count, traded_units_by_product


def cap_dwell(results: list[BacktestResult]) -> dict[str, int]:
    dwell = {product: 0 for product in PEBBLES}
    total_timestamps = 0

    for result in results:
        positions = {product: 0 for product in PEBBLES}
        trades_by_timestamp: dict[int, list[Any]] = defaultdict(list)
        for row in own_trade_rows(result):
            trades_by_timestamp[row.trade.timestamp].append(row.trade)

        timestamps = sorted({row.timestamp for row in result.activity_logs})
        total_timestamps += len(timestamps)
        for timestamp in timestamps:
            for trade in trades_by_timestamp.get(timestamp, []):
                if trade.buyer == "SUBMISSION":
                    positions[trade.symbol] += trade.quantity
                elif trade.seller == "SUBMISSION":
                    positions[trade.symbol] -= trade.quantity

            for product in PEBBLES:
                if abs(positions[product]) >= LIMITS[product]:
                    dwell[product] += 1

    dwell["TOTAL_TIMESTAMPS"] = total_timestamps
    return dwell


def max_drawdown(results: list[BacktestResult]) -> float:
    stitched: list[float] = []
    offset = 0.0
    for result in results:
        levels = [value for _, value in portfolio_pnl_by_timestamp(result.activity_logs)]
        if not levels:
            continue
        shifted = [offset + value for value in levels]
        stitched.extend(shifted)
        offset = shifted[-1]

    high_water_mark = -math.inf
    drawdown = 0.0
    for value in stitched:
        high_water_mark = max(high_water_mark, value)
        drawdown = max(drawdown, high_water_mark - value)
    return drawdown


def terminal_inventory_value(results: list[BacktestResult]) -> dict[str, float]:
    out = {product: 0.0 for product in PEBBLES}
    for result in results:
        positions = positions_for_result(result)
        last_mids = final_mid_prices(result)
        for product in PEBBLES:
            out[product] += positions[product] * last_mids.get(product, 0.0)
    return out


def summarise_results(results: list[BacktestResult]) -> dict[str, Any]:
    day_pnl = {f"day_{result.day_num}": sum(final_product_pnl(result).values()) for result in results}
    per_product = {product: 0.0 for product in PEBBLES}
    for result in results:
        for product, pnl in final_product_pnl(result).items():
            per_product[product] += pnl

    total_pnl = sum(day_pnl.values())
    traded_units, trade_count, traded_units_by_product = trade_stats(results)
    final_positions = positions_by_day(results)
    dwell = cap_dwell(results)

    return {
        "total_pnl": total_pnl,
        "day_pnl": day_pnl,
        "per_product_pnl": per_product,
        "traded_units": traded_units,
        "traded_units_by_product": traded_units_by_product,
        "trade_count": trade_count,
        "final_positions": final_positions,
        "cap_dwell": dwell,
        "max_drawdown": max_drawdown(results),
        "terminal_inventory_value": terminal_inventory_value(results),
        "stress_1": total_pnl - traded_units,
        "stress_3": total_pnl - 3 * traded_units,
        "stress_5": total_pnl - 5 * traded_units,
    }


def run_config(
    module: Any,
    config: Config,
    mode: TradeMatchingMode,
    days: tuple[int, ...],
    cached_data: dict[int, BacktestData],
) -> tuple[list[BacktestResult], dict[str, Any]]:
    apply_config(module.Trader, config)
    results = [
        run_backtest_cached(
            module.Trader(),
            cached_data[day],
            mode,
        )
        for day in days
    ]
    return results, summarise_results(results)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def fast_z_score(history: list[int], current_residual: int, config: Config) -> float | None:
    min_history = max(80, config.window // 4)
    lookback = history[-config.window :]
    if len(lookback) < min_history:
        return None

    mean = sum(lookback) / len(lookback)
    mean_square = sum(value * value for value in lookback) / len(lookback)
    variance = max(0.0, mean_square - mean * mean)
    std = math.sqrt(variance)
    if std < 10.0:
        return None
    return (current_residual - mean) / std


def fast_target(previous_target: int, z_score: float | None, history_len: int, config: Config) -> int:
    if history_len < max(80, config.window // 4):
        return 0
    if z_score is None:
        return previous_target
    if z_score > config.entry_z:
        return -config.target_size
    if z_score < -config.entry_z:
        return config.target_size
    if config.exit_z is not None and abs(z_score) < config.exit_z:
        return 0
    return previous_target


def fast_max_drawdown(levels: list[float]) -> float:
    high_water_mark = -math.inf
    drawdown = 0.0
    for value in levels:
        high_water_mark = max(high_water_mark, value)
        drawdown = max(drawdown, high_water_mark - value)
    return drawdown


def simulate_day_fast(template: BacktestData, config: Config) -> dict[str, Any]:
    histories = {product: [] for product in PEBBLES}
    targets = {product: 0 for product in PEBBLES}
    positions = {product: 0 for product in PEBBLES}
    cash = {product: 0.0 for product in PEBBLES}
    trade_count = {product: 0 for product in PEBBLES}
    traded_units_by_product = {product: 0 for product in PEBBLES}
    cap_dwell_counts = {product: 0 for product in PEBBLES}

    pnl_path: list[float] = []
    final_product_pnl = {product: 0.0 for product in PEBBLES}
    terminal_inventory = {product: 0.0 for product in PEBBLES}

    for timestamp in sorted(template.prices.keys()):
        _ = timestamp
        rows = template.prices[timestamp]
        if any(product not in rows for product in PEBBLES):
            continue

        mids: dict[str, float] = {}
        complete_book = True
        for product in PEBBLES:
            row = rows[product]
            if not row.bid_prices or not row.ask_prices:
                complete_book = False
                break
            mids[product] = (row.bid_prices[0] + row.ask_prices[0]) / 2.0
        if not complete_book:
            pnl_path.append(sum(cash[p] + positions[p] * mids.get(p, 0.0) for p in PEBBLES))
            continue

        group_mean = sum(mids.values()) / len(PEBBLES)
        scaled_residuals = {product: int(round((mids[product] - group_mean) * 10)) for product in PEBBLES}

        next_targets: dict[str, int] = {}
        for product in PEBBLES:
            z_score = fast_z_score(histories[product], scaled_residuals[product], config)
            next_targets[product] = fast_target(targets[product], z_score, len(histories[product]), config)

        final_product_pnl = {product: cash[product] + positions[product] * mids[product] for product in PEBBLES}
        pnl_path.append(sum(final_product_pnl.values()))

        for product in PEBBLES:
            row = rows[product]
            current_position = positions[product]
            target_position = clamp(next_targets[product], -10, 10)
            desired_delta = target_position - current_position

            if desired_delta > 0:
                visible_volume = row.ask_volumes[0] if row.ask_volumes else 0
                limit_room = max(0, 10 - current_position)
                quantity = min(desired_delta, visible_volume, limit_room, min(10, config.target_size))
                if quantity > 0:
                    positions[product] += quantity
                    cash[product] -= row.ask_prices[0] * quantity
                    trade_count[product] += 1
                    traded_units_by_product[product] += quantity
            elif desired_delta < 0:
                visible_volume = row.bid_volumes[0] if row.bid_volumes else 0
                limit_room = max(0, 10 + current_position)
                quantity = min(-desired_delta, visible_volume, limit_room, min(10, config.target_size))
                if quantity > 0:
                    positions[product] -= quantity
                    cash[product] += row.bid_prices[0] * quantity
                    trade_count[product] += 1
                    traded_units_by_product[product] += quantity

        for product in PEBBLES:
            histories[product].append(scaled_residuals[product])
            if len(histories[product]) > 700:
                del histories[product][: len(histories[product]) - 700]
            targets[product] = next_targets[product]
            if abs(positions[product]) >= 10:
                cap_dwell_counts[product] += 1
            terminal_inventory[product] = positions[product] * mids[product]

    return {
        "day": template.day_num,
        "total_pnl": sum(final_product_pnl.values()),
        "pnl_path": pnl_path,
        "per_product_pnl": final_product_pnl,
        "traded_units": sum(traded_units_by_product.values()),
        "traded_units_by_product": traded_units_by_product,
        "trade_count": trade_count,
        "final_positions": positions,
        "cap_dwell": cap_dwell_counts,
        "total_timestamps": len(pnl_path),
        "terminal_inventory_value": terminal_inventory,
    }


def run_config_fast(config: Config, days: tuple[int, ...], cached_data: dict[int, BacktestData]) -> dict[str, Any]:
    day_results = [simulate_day_fast(cached_data[day], config) for day in days]

    day_pnl = {f"day_{day_result['day']}": day_result["total_pnl"] for day_result in day_results}
    total_pnl = sum(day_pnl.values())

    per_product = {product: 0.0 for product in PEBBLES}
    traded_units_by_product = {product: 0 for product in PEBBLES}
    trade_count = {product: 0 for product in PEBBLES}
    cap_dwell_data = {product: 0 for product in PEBBLES}
    terminal_inventory = {product: 0.0 for product in PEBBLES}
    final_positions: dict[str, dict[str, int]] = {}
    stitched_levels: list[float] = []
    offset = 0.0

    for day_result in day_results:
        for product in PEBBLES:
            per_product[product] += day_result["per_product_pnl"][product]
            traded_units_by_product[product] += day_result["traded_units_by_product"][product]
            trade_count[product] += day_result["trade_count"][product]
            cap_dwell_data[product] += day_result["cap_dwell"][product]
            terminal_inventory[product] += day_result["terminal_inventory_value"][product]
        final_positions[f"day_{day_result['day']}"] = day_result["final_positions"]
        stitched_levels.extend(offset + value for value in day_result["pnl_path"])
        if day_result["pnl_path"]:
            offset += day_result["pnl_path"][-1]

    cap_dwell_data["TOTAL_TIMESTAMPS"] = sum(day_result["total_timestamps"] for day_result in day_results)
    traded_units = sum(traded_units_by_product.values())

    return {
        "total_pnl": total_pnl,
        "day_pnl": day_pnl,
        "per_product_pnl": per_product,
        "traded_units": traded_units,
        "traded_units_by_product": traded_units_by_product,
        "trade_count": trade_count,
        "final_positions": final_positions,
        "cap_dwell": cap_dwell_data,
        "max_drawdown": fast_max_drawdown(stitched_levels),
        "terminal_inventory_value": terminal_inventory,
        "stress_1": total_pnl - traded_units,
        "stress_3": total_pnl - 3 * traded_units,
        "stress_5": total_pnl - 5 * traded_units,
    }


def iter_configs() -> list[Config]:
    configs: list[Config] = []
    for window in (300, 400, 500, 600):
        for entry_z in (2.0, 2.25, 2.5, 2.75):
            for exit_z in (None, 0.25, 0.5):
                for target_size in (6, 8, 10):
                    configs.append(Config(window, entry_z, exit_z, target_size))
    return configs


def row_from_summary(config: Config, mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    cap_dwell_data = summary["cap_dwell"]
    total_timestamps = cap_dwell_data.get("TOTAL_TIMESTAMPS", 0)
    cap_dwell_without_total = {product: cap_dwell_data.get(product, 0) for product in PEBBLES}
    return {
        "config": config.label,
        "window": config.window,
        "entry_z": config.entry_z,
        "exit": "hold" if config.exit_z is None else config.exit_z,
        "target_size": config.target_size,
        "mode": mode,
        "merged_pnl": round(summary["total_pnl"], 2),
        "day_2": round(summary["day_pnl"].get("day_2", 0.0), 2),
        "day_3": round(summary["day_pnl"].get("day_3", 0.0), 2),
        "day_4": round(summary["day_pnl"].get("day_4", 0.0), 2),
        "traded_units": summary["traded_units"],
        "stress_1": round(summary["stress_1"], 2),
        "stress_3": round(summary["stress_3"], 2),
        "stress_5": round(summary["stress_5"], 2),
        "max_drawdown": round(summary["max_drawdown"], 2),
        "final_positions": json.dumps(summary["final_positions"], sort_keys=True, separators=(",", ":")),
        "cap_dwell": json.dumps(cap_dwell_without_total, sort_keys=True, separators=(",", ":")),
        "cap_dwell_total_timestamps": total_timestamps,
        "per_product_pnl": json.dumps(
            {product: round(value, 2) for product, value in summary["per_product_pnl"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "terminal_inventory_value": json.dumps(
            {product: round(value, 2) for product, value in summary["terminal_inventory_value"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round 5 Pebbles parameter diagnostics.")
    parser.add_argument("--out", default="backtests/r5_pebbles_sweep.csv", help="CSV path for sweep output.")
    parser.add_argument("--top", type=int, default=8, help="Number of worse-mode configs to retest under all/none.")
    parser.add_argument("--days", nargs="*", type=int, default=list(DEFAULT_DAYS), help="Round 5 days to test.")
    parser.add_argument("--max-configs", type=int, default=None, help="Optional smoke-test cap on worse-mode configs.")
    parser.add_argument(
        "--engine",
        choices=("fast", "backtester"),
        default="fast",
        help="Use the fast Pebbles simulator or the public runner functions.",
    )
    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Keep all products in the cached data. Default filters to Pebbles because the trader never reads others.",
    )
    args = parser.parse_args()

    module = import_trader_module() if args.engine == "backtester" else None
    days = tuple(args.days)
    cached_data = load_cached_data(days, pebbles_only=not args.all_products)

    rows: list[dict[str, Any]] = []
    worse_rows: list[dict[str, Any]] = []
    configs = iter_configs()
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    for index, config in enumerate(configs, start=1):
        if args.engine == "fast":
            summary = run_config_fast(config, days, cached_data)
        else:
            _, summary = run_config(module, config, TradeMatchingMode.worse, days, cached_data)
        row = row_from_summary(config, "worse", summary)
        worse_rows.append(row)
        rows.append(row)
        print(
            f"{index:03d}/{len(configs):03d} {config.label} worse pnl={row['merged_pnl']:.0f} "
            f"stress3={row['stress_3']:.0f} days=({row['day_2']:.0f},{row['day_3']:.0f},{row['day_4']:.0f})",
            flush=True,
        )

    ranked = sorted(
        worse_rows,
        key=lambda row: (
            min(row["day_2"], row["day_3"], row["day_4"]),
            row["stress_3"],
            row["merged_pnl"],
        ),
        reverse=True,
    )

    config_by_label = {config.label: config for config in iter_configs()}
    for row in ranked[: args.top]:
        config = config_by_label[row["config"]]
        for mode in (TradeMatchingMode.all, TradeMatchingMode.none):
            if args.engine == "fast":
                summary = run_config_fast(config, days, cached_data)
            else:
                _, summary = run_config(module, config, mode, days, cached_data)
            rows.append(row_from_summary(config, mode.value, summary))

    write_csv(ROOT / args.out, rows)

    print("\nTop worse-mode configs:")
    for row in ranked[: args.top]:
        print(
            f"{row['config']}: pnl={row['merged_pnl']:.0f}, "
            f"days=({row['day_2']:.0f},{row['day_3']:.0f},{row['day_4']:.0f}), "
            f"stress3={row['stress_3']:.0f}, units={row['traded_units']}"
        )
    print(f"\nWrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
