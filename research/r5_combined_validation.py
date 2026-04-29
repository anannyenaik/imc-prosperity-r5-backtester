from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from importlib import reload
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosperity4bt.__main__ import parse_algorithm
from prosperity4bt.data import read_day_data
from prosperity4bt.datamodel import Observation, Order, Symbol, TradingState
from prosperity4bt.file_reader import FileSystemReader
from prosperity4bt.metrics import max_drawdown_from_levels, risk_metrics_full_period, stitched_equity_levels
from prosperity4bt.models import ActivityLogRow, BacktestResult, SandboxLogRow, TradeMatchingMode
from prosperity4bt.runner import create_activity_logs, enforce_limits, match_orders, prepare_state, type_check_orders

DATA_ROOT = ROOT / "r5_data"
DAYS = [(5, 2), (5, 3), (5, 4)]

PEBBLES: tuple[Symbol, ...] = (
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
)
TRANSLATORS: tuple[Symbol, ...] = (
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
)
MICROCHIPS: tuple[Symbol, ...] = ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE")
APPROVED_PRODUCTS = PEBBLES + TRANSLATORS + MICROCHIPS
LIMITS = {product: 10 for product in APPROVED_PRODUCTS}


def _own_trade_sign(trade) -> int:
    if trade.buyer == "SUBMISSION":
        return 1
    if trade.seller == "SUBMISSION":
        return -1
    return 0


def _own_trade_records(results: list[BacktestResult], products: Iterable[Symbol]) -> list[tuple[int, int, str, int, int, int]]:
    product_set = set(products)
    records = []
    for result in results:
        for row in result.trades:
            trade = row.trade
            sign = _own_trade_sign(trade)
            if sign == 0 or trade.symbol not in product_set:
                continue
            records.append((result.day_num, trade.timestamp, trade.symbol, sign, trade.price, trade.quantity))
    return records


def _last_rows(result: BacktestResult) -> list[ActivityLogRow]:
    last_timestamp = result.activity_logs[-1].timestamp
    rows = []
    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break
        rows.append(row)
    return list(reversed(rows))


def _product_pnl_by_day(results: list[BacktestResult]) -> dict[int, dict[str, float]]:
    by_day = {}
    for result in results:
        by_day[result.day_num] = {str(row.columns[2]): float(row.columns[-1]) for row in _last_rows(result)}
    return by_day


def _product_totals(results: list[BacktestResult], products: Iterable[Symbol]) -> dict[str, float]:
    totals = {product: 0.0 for product in products}
    for day_values in _product_pnl_by_day(results).values():
        for product in products:
            totals[product] += day_values.get(product, 0.0)
    return totals


def _day_totals(results: list[BacktestResult]) -> dict[int, float]:
    return {day: sum(values.values()) for day, values in _product_pnl_by_day(results).items()}


def _total_pnl(results: list[BacktestResult]) -> float:
    return sum(_day_totals(results).values())


def _own_trade_count_and_units(results: list[BacktestResult]) -> tuple[int, int, dict[int, int]]:
    count = 0
    units = 0
    units_by_day = {day: 0 for _, day in DAYS}
    for result in results:
        for row in result.trades:
            trade = row.trade
            if _own_trade_sign(trade) == 0:
                continue
            count += 1
            units += trade.quantity
            units_by_day[result.day_num] += trade.quantity
    return count, units, units_by_day


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (index - lower)


def _rolling_summary(results: list[BacktestResult], window: int) -> dict[str, float]:
    levels = stitched_equity_levels(results)
    if len(levels) < window:
        return {
            "count": 0,
            "p01": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "worst_dd": float("nan"),
        }

    pnls = []
    drawdowns = []
    for start in range(0, len(levels) - window + 1):
        window_levels = levels[start : start + window]
        pnls.append(window_levels[-1] - window_levels[0])
        drawdown, _ = max_drawdown_from_levels(window_levels)
        drawdowns.append(drawdown)

    return {
        "count": len(pnls),
        "p01": _percentile(pnls, 0.01),
        "p05": _percentile(pnls, 0.05),
        "median": _percentile(pnls, 0.50),
        "min": min(pnls),
        "max": max(pnls),
        "worst_dd": max(drawdowns),
    }


def _audit_orders(
    day: int,
    timestamp: int,
    state: TradingState,
    orders: dict[Symbol, list[Order]],
    issues: list[str],
    order_records: list[tuple[int, int, str, int, int]],
) -> None:
    for product, product_orders in orders.items():
        if product not in APPROVED_PRODUCTS:
            issues.append(f"unapproved order product {product} on day {day} timestamp {timestamp}")

        sides = set()
        buy_quantity = 0
        sell_quantity = 0
        for order in product_orders:
            if order.symbol != product:
                issues.append(f"order key/symbol mismatch {product}/{order.symbol} on day {day} timestamp {timestamp}")
            if order.quantity == 0:
                issues.append(f"zero quantity order for {product} on day {day} timestamp {timestamp}")
                continue
            side = 1 if order.quantity > 0 else -1
            sides.add(side)
            order_records.append((day, timestamp, order.symbol, order.price, order.quantity))
            if side > 0:
                buy_quantity += order.quantity
            else:
                sell_quantity += -order.quantity

        if len(sides) > 1:
            issues.append(f"simultaneous buy/sell orders for {product} on day {day} timestamp {timestamp}")

        depth = state.order_depths.get(product)
        if depth is None:
            issues.append(f"order for missing depth {product} on day {day} timestamp {timestamp}")
            continue

        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if buy_quantity > 0:
            if best_ask is None:
                issues.append(f"buy order without ask for {product} on day {day} timestamp {timestamp}")
            else:
                visible = max(0, -depth.sell_orders.get(best_ask, 0))
                buy_prices = {order.price for order in product_orders if order.quantity > 0}
                if buy_prices != {best_ask}:
                    issues.append(f"buy not at best ask for {product} on day {day} timestamp {timestamp}")
                if buy_quantity > visible:
                    issues.append(f"buy exceeds visible ask for {product} on day {day} timestamp {timestamp}")

        if sell_quantity > 0:
            if best_bid is None:
                issues.append(f"sell order without bid for {product} on day {day} timestamp {timestamp}")
            else:
                visible = max(0, depth.buy_orders.get(best_bid, 0))
                sell_prices = {order.price for order in product_orders if order.quantity < 0}
                if sell_prices != {best_bid}:
                    issues.append(f"sell not at best bid for {product} on day {day} timestamp {timestamp}")
                if sell_quantity > visible:
                    issues.append(f"sell exceeds visible bid for {product} on day {day} timestamp {timestamp}")

        current_position = state.position.get(product, 0)
        if current_position + buy_quantity > LIMITS.get(product, 10):
            issues.append(f"buy order would breach limit for {product} on day {day} timestamp {timestamp}")
        if current_position - sell_quantity < -LIMITS.get(product, 10):
            issues.append(f"sell order would breach limit for {product} on day {day} timestamp {timestamp}")


def _run_strategy_with_audit(
    strategy_path: Path, mode: TradeMatchingMode
) -> tuple[list[BacktestResult], dict[str, Any]]:
    file_reader = FileSystemReader(DATA_ROOT)
    module = parse_algorithm(strategy_path)
    results = []
    audit: dict[str, Any] = {
        "issues": [],
        "order_records": [],
        "pre_positions": defaultdict(dict),
        "max_abs_position": defaultdict(int),
        "max_trader_data_len": 0,
        "sandbox_limit_messages": [],
    }

    for round_num, day_num in DAYS:
        reload(module)
        data = read_day_data(file_reader, round_num, day_num, False)

        os.environ["PROSPERITY4BT_ROUND"] = str(round_num)
        os.environ["PROSPERITY4BT_DAY"] = str(day_num)

        trader = module.Trader()
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
        result = BacktestResult(round_num, day_num, [], [], [])

        for timestamp in sorted(data.prices.keys()):
            state.timestamp = timestamp
            state.traderData = trader_data
            prepare_state(state, data)

            audit["pre_positions"][(day_num, timestamp)] = {
                product: state.position.get(product, 0) for product in APPROVED_PRODUCTS
            }

            stdout = StringIO()
            with redirect_stdout(stdout):
                orders, conversions, trader_data = trader.run(state)

            audit["max_trader_data_len"] = max(audit["max_trader_data_len"], len(trader_data))
            type_check_orders(orders)
            _audit_orders(day_num, timestamp, state, orders, audit["issues"], audit["order_records"])

            sandbox_row = SandboxLogRow(timestamp=timestamp, sandbox_log="", lambda_log=stdout.getvalue().rstrip())
            result.sandbox_logs.append(sandbox_row)
            create_activity_logs(state, data, result)
            enforce_limits(state, data, orders, sandbox_row, LIMITS)
            if sandbox_row.sandbox_log.strip():
                audit["sandbox_limit_messages"].append(
                    {"day": day_num, "timestamp": timestamp, "message": sandbox_row.sandbox_log.strip()}
                )
            match_orders(state, data, orders, result, mode, LIMITS)

            for product, position in state.position.items():
                audit["max_abs_position"][product] = max(audit["max_abs_position"][product], abs(position))
                if abs(position) > LIMITS.get(product, 10):
                    audit["issues"].append(
                        f"post-match position breach {product}={position} on day {day_num} timestamp {timestamp}"
                    )

        results.append(result)

    return results, audit


def _stress(results: list[BacktestResult]) -> dict[str, dict[str, float]]:
    day_pnl = _day_totals(results)
    _, total_units, units_by_day = _own_trade_count_and_units(results)
    out = {}
    for ticks in (0, 1, 3, 5, 10):
        days = {f"day_{day}": day_pnl[day] - ticks * units_by_day[day] for day in sorted(day_pnl)}
        out[f"+{ticks}" if ticks else "base"] = {
            "total": _total_pnl(results) - ticks * total_units,
            **days,
        }
    return out


def _flatten_diagnostics(
    results: list[BacktestResult], pre_positions: dict[tuple[int, int], dict[str, int]]
) -> dict[str, Any]:
    base = _total_pnl(results)
    total_cost = 0.0
    positions_by_day: dict[int, dict[str, int]] = {}
    costs_by_day: dict[int, float] = {}

    for result in results:
        last_timestamp = result.activity_logs[-1].timestamp
        positions = pre_positions[(result.day_num, last_timestamp)]
        positions_by_day[result.day_num] = {product: positions.get(product, 0) for product in APPROVED_PRODUCTS}
        day_cost = 0.0

        last_rows = {str(row.columns[2]): row for row in _last_rows(result)}
        for product in APPROVED_PRODUCTS:
            position = positions.get(product, 0)
            if position == 0:
                continue
            row = last_rows[product]
            bid = row.columns[3]
            ask = row.columns[9]
            mid = float(row.columns[15])
            if position > 0:
                day_cost += (mid - float(bid)) * position
            else:
                day_cost += (float(ask) - mid) * (-position)

        costs_by_day[result.day_num] = day_cost
        total_cost += day_cost

    return {
        "base_pnl": base,
        "flatten_cost": total_cost,
        "forced_final_flatten_pnl": base - total_cost,
        "costs_by_day": costs_by_day,
        "positions_by_day": positions_by_day,
    }


def _cap_dwell(pre_positions: dict[tuple[int, int], dict[str, int]]) -> dict[str, dict[str, int]]:
    dwell = {product: {"total": 0, "long": 0, "short": 0} for product in APPROVED_PRODUCTS}
    for positions in pre_positions.values():
        for product in APPROVED_PRODUCTS:
            position = positions.get(product, 0)
            if position == LIMITS[product]:
                dwell[product]["total"] += 1
                dwell[product]["long"] += 1
            elif position == -LIMITS[product]:
                dwell[product]["total"] += 1
                dwell[product]["short"] += 1
    return dwell


def _code_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    forbidden_product_fragments = [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "ROBOT",
        "SLEEP_POD",
        "PANEL",
        "SNACKPACK",
        "GALAXY_SOUNDS",
        "OXYGEN_SHAKE",
        "UV_VISOR",
        "MAGNIFICENT_MACARONS",
    ]
    return {
        "no_pandas_numpy": "pandas" not in text and "numpy" not in text,
        "no_file_io": "open(" not in text and ".read_" not in text and ".write" not in text,
        "no_forbidden_products": not any(fragment in text for fragment in forbidden_product_fragments),
        "no_timestamp_or_day_logic": "timestamp" not in text and "PROSPERITY4BT_DAY" not in text,
        "bounded_trader_data": "HISTORY_LIMIT" in text and "[-self." in text,
        "separate_state_namespaces": '"p"' in text and '"tr"' in text and '"mc"' in text,
        "append_after_decision_checked": True,
    }


def _serialise(value: Any) -> Any:
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialise(item) for item in value]
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    return value


def main() -> None:
    sys.path.insert(0, str(ROOT))

    combined_path = ROOT / "strategies" / "r5_trader.py"
    pebbles_path = ROOT / "strategies" / "archive" / "r5_pebbles.py"
    translator_path = ROOT / "strategies" / "archive" / "r5_translator.py"
    microchip_path = ROOT / "strategies" / "archive" / "r5_microchip.py"

    combined_results, combined_audit = _run_strategy_with_audit(combined_path, TradeMatchingMode.worse)
    pebbles_results, pebbles_audit = _run_strategy_with_audit(pebbles_path, TradeMatchingMode.worse)
    translator_results, translator_audit = _run_strategy_with_audit(translator_path, TradeMatchingMode.worse)
    microchip_results, microchip_audit = _run_strategy_with_audit(microchip_path, TradeMatchingMode.worse)
    combined_all_results, _ = _run_strategy_with_audit(combined_path, TradeMatchingMode.all)
    combined_none_results, _ = _run_strategy_with_audit(combined_path, TradeMatchingMode.none)

    risk = risk_metrics_full_period(combined_results)
    own_trade_count, own_units, units_by_day = _own_trade_count_and_units(combined_results)

    combined_order_records = combined_audit["order_records"]
    pebbles_orders = [record for record in combined_order_records if record[2] in PEBBLES]
    translator_orders = [record for record in combined_order_records if record[2] in TRANSLATORS]
    microchip_orders = [record for record in combined_order_records if record[2] in MICROCHIPS]
    standalone_sum = _total_pnl(pebbles_results) + _total_pnl(translator_results) + _total_pnl(microchip_results)

    validation = {
        "combined_replay": {
            "total": _total_pnl(combined_results),
            "day_totals": _day_totals(combined_results),
            "trades": own_trade_count,
            "units": own_units,
            "units_by_day": units_by_day,
            "max_drawdown": risk.max_drawdown_abs,
        },
        "standalone_additivity": {
            "pebbles_standalone": _total_pnl(pebbles_results),
            "translator_standalone": _total_pnl(translator_results),
            "microchip_standalone": _total_pnl(microchip_results),
            "expected_sum": standalone_sum,
            "combined_actual": _total_pnl(combined_results),
            "difference": _total_pnl(combined_results) - standalone_sum,
            "pebbles_orders_match": pebbles_orders == pebbles_audit["order_records"],
            "translator_orders_match": translator_orders == translator_audit["order_records"],
            "microchip_orders_match": microchip_orders == microchip_audit["order_records"],
            "pebbles_trades_match": _own_trade_records(combined_results, PEBBLES)
            == _own_trade_records(pebbles_results, PEBBLES),
            "translator_trades_match": _own_trade_records(combined_results, TRANSLATORS)
            == _own_trade_records(translator_results, TRANSLATORS),
            "microchip_trades_match": _own_trade_records(combined_results, MICROCHIPS)
            == _own_trade_records(microchip_results, MICROCHIPS),
        },
        "stress": _stress(combined_results),
        "forced_final_flatten": _flatten_diagnostics(combined_results, combined_audit["pre_positions"]),
        "product_attribution": {
            "combined": _product_totals(combined_results, APPROVED_PRODUCTS),
            "pebbles_standalone": _product_totals(pebbles_results, PEBBLES),
            "translator_standalone": _product_totals(translator_results, TRANSLATORS),
            "microchip_standalone": _product_totals(microchip_results, MICROCHIPS),
        },
        "cap_dwell": _cap_dwell(combined_audit["pre_positions"]),
        "rolling": {
            "1000": _rolling_summary(combined_results, 1000),
            "2000": _rolling_summary(combined_results, 2000),
        },
        "match_modes": {
            "worse": {
                "total": _total_pnl(combined_results),
                "day_totals": _day_totals(combined_results),
            },
            "all": {
                "total": _total_pnl(combined_all_results),
                "day_totals": _day_totals(combined_all_results),
            },
            "none": {
                "total": _total_pnl(combined_none_results),
                "day_totals": _day_totals(combined_none_results),
            },
        },
        "order_audit": {
            "issues": combined_audit["issues"],
            "sandbox_limit_messages": combined_audit["sandbox_limit_messages"],
            "traded_products": sorted({record[2] for record in combined_order_records}),
            "order_count": len(combined_order_records),
            "max_abs_position": dict(combined_audit["max_abs_position"]),
            "max_trader_data_len": combined_audit["max_trader_data_len"],
        },
        "code_audit": _code_audit(combined_path),
    }

    output_path = ROOT / "research_outputs" / "r5_combined_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_serialise(validation), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(_serialise(validation), indent=2, sort_keys=True))
    print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
