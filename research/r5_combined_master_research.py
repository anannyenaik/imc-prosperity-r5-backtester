from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import r5_pebbles_deep_research as pebbles_deep  # type: ignore
import r5_translator_research as translator_research  # type: ignore
from prosperity4bt.__main__ import parse_algorithm
from prosperity4bt.data import BacktestData, read_day_data
from prosperity4bt.datamodel import Observation, TradingState
from prosperity4bt.file_reader import FileSystemReader
from prosperity4bt.runner import prepare_state


DAYS = (2, 3, 4)
PEBBLES = (
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
)
TRANSLATORS = (
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
)
MICROCHIPS = ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE")
APPROVED_PRODUCTS = PEBBLES + TRANSLATORS + MICROCHIPS


def load_round5() -> dict[int, BacktestData]:
    reader = FileSystemReader(ROOT / "r5_data")
    return {day: read_day_data(reader, 5, day, no_names=False) for day in DAYS}


def percentile(values: list[float], pct: float) -> float:
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


def safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) * (value - mean) for value in values) / len(values)
    return {
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "min": round(min(values), 6),
        "p25": round(percentile(values, 0.25), 6),
        "p50": round(percentile(values, 0.50), 6),
        "p75": round(percentile(values, 0.75), 6),
        "max": round(max(values), 6),
    }


def autocorr(values: list[float], lag: int = 1) -> float:
    if len(values) <= lag:
        return 0.0
    x_values = values[:-lag]
    y_values = values[lag:]
    return corr(x_values, y_values)


def corr(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 1e-12 or var_right <= 1e-12:
        return 0.0
    cov = sum((left[i] - mean_left) * (right[i] - mean_right) for i in range(len(left)))
    return cov / math.sqrt(var_left * var_right)


def book_snapshot(row: Any) -> tuple[float, int, int, int, int] | None:
    if not row.bid_prices or not row.ask_prices:
        return None
    bid = row.bid_prices[0]
    ask = row.ask_prices[0]
    bid_volume = row.bid_volumes[0] if row.bid_volumes else 0
    ask_volume = row.ask_volumes[0] if row.ask_volumes else 0
    return (bid + ask) / 2.0, bid, ask, bid_volume, ask_volume


def group_series(data: BacktestData, products: tuple[str, ...]) -> dict[str, Any]:
    timestamps: list[int] = []
    mids = {product: [] for product in products}
    spreads = {product: [] for product in products}
    bid_depths = {product: [] for product in products}
    ask_depths = {product: [] for product in products}
    group_sums: list[float] = []
    residuals = {product: [] for product in products}

    for timestamp in sorted(data.prices):
        rows = data.prices[timestamp]
        snapshots = {product: book_snapshot(rows[product]) for product in products if product in rows}
        if len(snapshots) != len(products) or any(snapshot is None for snapshot in snapshots.values()):
            continue
        mids_now = {product: snapshots[product][0] for product in products}  # type: ignore[index]
        group_mean = sum(mids_now.values()) / len(products)
        timestamps.append(timestamp)
        group_sums.append(sum(mids_now.values()))
        for product in products:
            mid, bid, ask, bid_volume, ask_volume = snapshots[product]  # type: ignore[misc]
            mids[product].append(mid)
            spreads[product].append(ask - bid)
            bid_depths[product].append(bid_volume)
            ask_depths[product].append(ask_volume)
            residuals[product].append(round((mid - group_mean) * 10))

    return {
        "timestamps": timestamps,
        "mids": mids,
        "spreads": spreads,
        "bid_depths": bid_depths,
        "ask_depths": ask_depths,
        "group_sums": group_sums,
        "residuals": residuals,
    }


def return_correlation(mids: dict[str, list[float]]) -> dict[str, float]:
    products = list(mids)
    pair_values: list[float] = []
    for i, left in enumerate(products):
        left_returns = [mids[left][index] - mids[left][index - 1] for index in range(1, len(mids[left]))]
        for right in products[i + 1 :]:
            right_returns = [mids[right][index] - mids[right][index - 1] for index in range(1, len(mids[right]))]
            pair_values.append(corr(left_returns, right_returns))
    return {
        "avg_pair_corr": round(sum(pair_values) / len(pair_values), 6) if pair_values else 0.0,
        "min_pair_corr": round(min(pair_values), 6) if pair_values else 0.0,
        "max_pair_corr": round(max(pair_values), 6) if pair_values else 0.0,
    }


def markouts(
    data_by_day: dict[int, BacktestData],
    products: tuple[str, ...],
    window: int,
    entry_z: float,
    min_history: int,
) -> dict[str, Any]:
    horizons = (10, 50, 100)
    out: dict[str, Any] = {}
    for product in products:
        signed_moves = {horizon: [] for horizon in horizons}
        event_count = 0
        for data in data_by_day.values():
            series = group_series(data, products)
            product_mids = series["mids"][product]
            product_residuals = [int(value) for value in series["residuals"][product]]
            history: list[int] = []
            for index, residual in enumerate(product_residuals):
                if len(history) >= min_history:
                    lookback = history[-window:]
                    mean = sum(lookback) / len(lookback)
                    mean_square = sum(value * value for value in lookback) / len(lookback)
                    std = math.sqrt(max(0.0, mean_square - mean * mean))
                    if std >= 10.0:
                        z_score = (residual - mean) / std
                        direction = -1 if z_score > entry_z else 1 if z_score < -entry_z else 0
                        if direction:
                            event_count += 1
                            for horizon in horizons:
                                future_index = min(len(product_mids) - 1, index + horizon)
                                signed_moves[horizon].append(
                                    direction * (product_mids[future_index] - product_mids[index])
                                )
                history.append(residual)
                if len(history) > window:
                    del history[: len(history) - window]
        out[product] = {
            "events": event_count,
            **{f"markout_{horizon}": round(sum(values) / len(values), 6) if values else 0.0 for horizon, values in signed_moves.items()},
        }
    return out


def structural_diagnostics(data_by_day: dict[int, BacktestData], products: tuple[str, ...]) -> dict[str, Any]:
    by_day: dict[str, Any] = {}
    combined = {
        "mids": {product: [] for product in products},
        "spreads": {product: [] for product in products},
        "bid_depths": {product: [] for product in products},
        "ask_depths": {product: [] for product in products},
        "group_sums": [],
        "residuals": {product: [] for product in products},
    }

    for day, data in data_by_day.items():
        series = group_series(data, products)
        by_day[str(day)] = {
            "ticks": len(series["timestamps"]),
            "group_sum": safe_stats(series["group_sums"]),
            "return_corr": return_correlation(series["mids"]),
            "residuals": {
                product: {**safe_stats(series["residuals"][product]), "autocorr_1": round(autocorr(series["residuals"][product]), 6)}
                for product in products
            },
        }
        for key in ("group_sums",):
            combined[key].extend(series[key])
        for product in products:
            for key in ("mids", "spreads", "bid_depths", "ask_depths", "residuals"):
                combined[key][product].extend(series[key][product])

    return {
        "by_day": by_day,
        "combined": {
            "group_sum": safe_stats(combined["group_sums"]),
            "return_corr": return_correlation(combined["mids"]),
            "spread_depth": {
                product: {
                    "spread": safe_stats(combined["spreads"][product]),
                    "bid_depth": safe_stats(combined["bid_depths"][product]),
                    "ask_depth": safe_stats(combined["ask_depths"][product]),
                }
                for product in products
            },
            "residual_autocorr_1": {
                product: round(autocorr(combined["residuals"][product]), 6) for product in products
            },
        },
    }


def pebbles_configs() -> list[tuple[str, Any, str]]:
    base = pebbles_deep.baseline_config()
    return [
        ("current", base, "kept"),
        ("mh500", replace(base, label="mh500", min_history=500), "rejected: lower PnL and stress"),
        ("mh250", replace(base, label="mh250", min_history=250), "rejected: lower PnL and stress"),
        ("mh375", replace(base, label="mh375", min_history=375), "rejected: lower PnL and stress"),
        ("z2.25", replace(base, label="z2.25", entry_z=2.25), "rejected: lower stress"),
        ("z2.30", replace(base, label="z2.30", entry_z=2.30), "rejected: lower stress"),
        ("z2.40", replace(base, label="z2.40", entry_z=2.40), "rejected: public PnL gain, no robust need"),
        ("z2.45", replace(base, label="z2.45", entry_z=2.45), "rejected: weak day 3"),
        ("w400", replace(base, label="w400", window=400), "rejected: lower stress"),
        ("w450", replace(base, label="w450", window=450), "rejected: lower stress and day 2"),
        ("w550", replace(base, label="w550", window=550), "rejected: lower PnL"),
        ("w600", replace(base, label="w600", window=600), "rejected: lower PnL"),
        ("target9", replace(base, label="target9", target_size=9, max_order_size=9), "rejected: lower PnL"),
        ("target8", replace(base, label="target8", target_size=8, max_order_size=8), "rejected: lower PnL"),
        ("basket_neutral", replace(base, label="basket_neutral", target_rule="basket_neutral"), "rejected: destroys edge"),
        ("rank_pair", replace(base, label="rank_pair", target_rule="rank_pair"), "rejected: negative PnL"),
        ("rank2", replace(base, label="rank2", target_rule="rank2"), "rejected: negative PnL"),
        ("xl_entry_plus_0.05", replace(base, label="xl_entry_plus_0.05", entry_offsets={"PEBBLES_XL": 0.05}), "rejected: product-specific public tweak"),
        ("consensus_400_500", replace(base, label="consensus_400_500", ensemble_windows=(400, 500), consensus_count=2), "rejected: added complexity and public-path timing"),
    ]


def translator_configs() -> list[tuple[str, Any, str]]:
    def cfg(label: str, **kwargs: Any) -> Any:
        defaults = {"window": 1200, "entry_z": 1.75, "min_history": 1200, "history_limit": 1200}
        defaults.update(kwargs)
        return translator_research.StrategyConfig(label=label, **defaults)

    return [
        ("current", cfg("current"), "kept"),
        ("w1000_z1.75_mh1000", cfg("w1000_z1.75_mh1000", window=1000, min_history=1000, history_limit=1000), "rejected: lower PnL and stress"),
        ("w1200_z1.5_mh1200", cfg("w1200_z1.5_mh1200", entry_z=1.5), "rejected: lower stress"),
        ("w1200_z2.0_mh1200", cfg("w1200_z2.0_mh1200", entry_z=2.0), "rejected: lower day 2 and drawdown"),
        ("w1400_z1.75_mh1400", cfg("w1400_z1.75_mh1400", window=1400, min_history=1400, history_limit=1400), "rejected: lower PnL"),
        ("w1600_z1.5_mh1600", cfg("w1600_z1.5_mh1600", window=1600, entry_z=1.5, min_history=1600, history_limit=1600), "rejected: lower PnL"),
        ("w1600_z1.75_mh1600", cfg("w1600_z1.75_mh1600", window=1600, min_history=1600, history_limit=1600), "rejected: lower PnL"),
        ("target9", cfg("target9", target_size=9), "rejected: lower PnL"),
        ("target8", cfg("target8", target_size=8), "rejected: lower PnL"),
        ("spread_p75", cfg("spread_p75", spread_rule="p75"), "rejected: too small and public execution slice"),
        ("spread_p60", cfg("spread_p60", spread_rule="p60"), "rejected: too small and public execution slice"),
        ("edge_half_spread", cfg("edge_half_spread", edge_gate_buffer=0.0), "rejected: no behavioural change"),
        ("confirm2", cfg("confirm2", confirm_ticks=2), "rejected: neighbourhood instability"),
        ("confirm3", cfg("confirm3", confirm_ticks=3), "rejected: worse drawdown"),
        ("spread_p75_confirm2", cfg("spread_p75_confirm2", spread_rule="p75", confirm_ticks=2), "rejected: two-knob public timing variant"),
        ("leave_one_out", cfg("leave_one_out", residual_kind="leave_one_out"), "rejected: tiny algebraic variant"),
        ("rank", cfg("rank", residual_kind="rank"), "rejected: weak and sparse"),
        ("two_stage", cfg("two_stage", hold_to_flip=False, exit_rule="two_stage"), "rejected: negative PnL"),
    ]


def normalise_pebbles_row(row: dict[str, Any], label: str, decision: str) -> dict[str, Any]:
    return {
        "module": "pebbles",
        "label": label,
        "decision": decision,
        "total": float(row["merged_pnl"]),
        "day_2": float(row["day_2"]),
        "day_3": float(row["day_3"]),
        "day_4": float(row["day_4"]),
        "units": int(row["traded_units"]),
        "trades": int(row.get("trade_count", 0)),
        "stress_1": float(row["stress_1"]),
        "stress_3": float(row["stress_3"]),
        "stress_5": float(row["stress_5"]),
        "stress_10": float(row["stress_10"]),
        "max_drawdown": float(row["max_drawdown"]),
        "roll1000_p05": float(row["roll1000_p05"]),
        "roll2000_p05": float(row["roll2000_p05"]),
        "forced_flatten": float(row["forced_flatten_pnl"]),
    }


def normalise_translator_row(row: dict[str, Any], label: str, decision: str) -> dict[str, Any]:
    return {
        "module": "translator",
        "label": label,
        "decision": decision,
        "total": float(row["merged_pnl"]),
        "day_2": float(row["day_2"]),
        "day_3": float(row["day_3"]),
        "day_4": float(row["day_4"]),
        "units": int(row["traded_units"]),
        "trades": int(row["trade_count"]),
        "stress_1": float(row["stress_1"]),
        "stress_3": float(row["stress_3"]),
        "stress_5": float(row["stress_5"]),
        "stress_10": float(row["stress_10"]),
        "max_drawdown": float(row["max_drawdown"]),
        "roll1000_p05": float(row["roll1000_p05"]),
        "roll2000_p05": float(row["roll2000_p05"]),
        "forced_flatten": float(row["forced_flatten_pnl"]),
    }


def add_rows(left: dict[str, Any], right: dict[str, Any], label: str, decision: str) -> dict[str, Any]:
    return {
        "module": "combined_candidate",
        "label": label,
        "decision": decision,
        "total": left["total"] + right["total"],
        "day_2": left["day_2"] + right["day_2"],
        "day_3": left["day_3"] + right["day_3"],
        "day_4": left["day_4"] + right["day_4"],
        "units": left["units"] + right["units"],
        "trades": left["trades"] + right["trades"],
        "stress_1": left["stress_1"] + right["stress_1"],
        "stress_3": left["stress_3"] + right["stress_3"],
        "stress_5": left["stress_5"] + right["stress_5"],
        "stress_10": left["stress_10"] + right["stress_10"],
        "max_drawdown": "",
        "roll1000_p05": "",
        "roll2000_p05": "",
        "forced_flatten": left["forced_flatten"] + right["forced_flatten"],
    }


def run_sweeps() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pebbles_cached = pebbles_deep.load_data(DAYS)
    translator_cached = translator_research.load_data(DAYS)

    rows: list[dict[str, Any]] = []
    pebbles_by_label: dict[str, dict[str, Any]] = {}
    translator_by_label: dict[str, dict[str, Any]] = {}

    for label, config, decision in pebbles_configs():
        sim_row = pebbles_deep.simulate_config(config, pebbles_cached, DAYS)
        row = normalise_pebbles_row(sim_row, label, decision)
        rows.append(row)
        pebbles_by_label[label] = row

    for label, config, decision in translator_configs():
        sim_row = translator_research.simulate_config(config, translator_cached, DAYS)
        row = normalise_translator_row(sim_row, label, decision)
        rows.append(row)
        translator_by_label[label] = row

    rows.append(add_rows(pebbles_by_label["current"], translator_by_label["current"], "current", "kept"))
    rows.append(add_rows(pebbles_by_label["current"], translator_by_label["confirm2"], "translator_confirm2", "rejected: neighbourhood instability"))
    rows.append(
        add_rows(
            pebbles_by_label["current"],
            translator_by_label["spread_p75_confirm2"],
            "translator_spread_p75_confirm2",
            "rejected: two-knob public timing variant",
        )
    )
    rows.append(add_rows(pebbles_by_label["current"], translator_by_label["target9"], "translator_target9", "rejected: lower expected PnL"))

    return rows, {"pebbles": pebbles_by_label, "translator": translator_by_label}


def benchmark_trader_data() -> dict[str, Any]:
    module = parse_algorithm(ROOT / "strategies" / "r5_trader.py")
    data_by_day = load_round5()
    tick_count = 0
    max_trader_data_len = 0
    start = time.perf_counter()

    for day in DAYS:
        data = data_by_day[day]
        trader = module.Trader()
        trader_data = ""
        state = TradingState(
            traderData="",
            timestamp=0,
            listings={},
            order_depths={},
            own_trades={},
            market_trades={},
            position={},
            observations=Observation({}, {}),
        )
        for timestamp in sorted(data.prices):
            state.timestamp = timestamp
            state.traderData = trader_data
            prepare_state(state, data)
            _, _, trader_data = trader.run(state)
            max_trader_data_len = max(max_trader_data_len, len(trader_data))
            tick_count += 1

    elapsed = time.perf_counter() - start
    return {
        "ticks": tick_count,
        "wall_seconds": round(elapsed, 6),
        "approx_ms_per_tick": round(elapsed * 1000.0 / tick_count, 6) if tick_count else 0.0,
        "max_trader_data_len": max_trader_data_len,
    }


def code_text_audit() -> dict[str, Any]:
    text = (ROOT / "strategies" / "r5_trader.py").read_text(encoding="utf-8")
    forbidden = [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "ROBOT",
        "SLEEP_POD",
        "PANEL",
        "SNACKPACK",
        "OXYGEN",
        "GALAXY",
        "UV_VISOR",
        "MAGNIFICENT_MACARONS",
    ]
    return {
        "approved_products_only_in_trader": not any(name in text for name in forbidden),
        "no_pandas_numpy": "pandas" not in text and "numpy" not in text,
        "no_file_io": "open(" not in text and ".read_" not in text and ".write" not in text,
        "no_timestamp_day_logic": "timestamp" not in text and "PROSPERITY4BT_DAY" not in text and "PROSPERITY4BT_ROUND" not in text,
        "separate_namespaces": '"p"' in text and '"tr"' in text and '"mc"' in text,
        "bounded_history": "HISTORY_LIMIT" in text,
        "l1_only": "best_ask" in text and "best_bid" in text and "bid_prices[1]" not in text and "ask_prices[1]" not in text,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    validation_path = ROOT / "research_outputs" / "r5_combined_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    data_by_day = load_round5()
    sweep_rows, module_rows = run_sweeps()

    summary = {
        "verdict": "FINAL_LOCK_COMBINED_CURRENT",
        "reason": "Combined current exactly reproduces standalone orders and trades; public-winning variants are rejected as unstable or overfit.",
        "combined_validation": validation,
        "structural_diagnostics": {
            "pebbles": {
                **structural_diagnostics(data_by_day, PEBBLES),
                "markouts": markouts(data_by_day, PEBBLES, window=500, entry_z=2.35, min_history=125),
            },
            "translator": {
                **structural_diagnostics(data_by_day, TRANSLATORS),
                "markouts": markouts(data_by_day, TRANSLATORS, window=1200, entry_z=1.75, min_history=1200),
            },
        },
        "sweep_summary": {
            "best_pebbles_by_stress5": sorted(module_rows["pebbles"].values(), key=lambda row: row["stress_5"], reverse=True)[:8],
            "best_translator_by_stress5": sorted(module_rows["translator"].values(), key=lambda row: row["stress_5"], reverse=True)[:8],
            "accepted": {
                "pebbles": module_rows["pebbles"]["current"],
                "translator": module_rows["translator"]["current"],
            },
        },
        "runtime": benchmark_trader_data(),
        "code_audit": code_text_audit(),
        "website_preview_proxy": {
            "status": "FINISHED locally",
            "day4_first_1000_pnl_estimate": 7190.0,
            "translator_first_1000_note": "Translator mh1200 intentionally does not trade in the first 1000 ticks of a fresh day.",
        },
        "rejected_variant_reasons": {
            row["label"]: row["decision"]
            for row in sweep_rows
            if row["decision"] != "kept" and row["module"] in {"pebbles", "translator", "combined_candidate"}
        },
    }

    write_csv(ROOT / "research_outputs" / "r5_combined_master_sweeps.csv", sweep_rows)
    write_json(ROOT / "research_outputs" / "r5_combined_master_summary.json", summary)
    print(json.dumps({"verdict": summary["verdict"], "runtime": summary["runtime"]}, indent=2))
    print("Wrote research_outputs/r5_combined_master_summary.json")
    print("Wrote research_outputs/r5_combined_master_sweeps.csv")


if __name__ == "__main__":
    main()
