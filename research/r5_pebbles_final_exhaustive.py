"""Final Round 5 Pebbles research harness.

This script is research-only. It reuses the local deterministic Pebbles
simulator in r5_pebbles_deep_research.py, adds the final causal diagnostics,
and writes the tables needed for the submission decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import r5_pebbles_deep_research as deep  # type: ignore

PEBBLES = deep.PEBBLES
DEFAULT_DAYS = deep.DEFAULT_DAYS
RESIDUAL_SCALE = deep.RESIDUAL_SCALE

ORIGINAL_RESIDUALS_FOR_CONFIG = deep.residuals_for_config
ORIGINAL_APPLY_TARGET_RULE = deep.apply_target_rule


def mean_std(values: list[float], window: int) -> tuple[float, float] | None:
    if not values:
        return None
    lookback = values[-window:]
    mean = sum(lookback) / len(lookback)
    mean_square = sum(value * value for value in lookback) / len(lookback)
    variance = max(0.0, mean_square - mean * mean)
    return mean, math.sqrt(variance)


def ensure_cross_sums(state: deep.SimState) -> dict[str, list[float]]:
    cross_sums = getattr(state, "mid_basket_cross_sums", None)
    if cross_sums is None:
        cross_sums = {product: [0.0] for product in PEBBLES}
        state.mid_basket_cross_sums = cross_sums

    basket_len = len(state.basket_sum.values)
    for product in PEBBLES:
        product_cross = cross_sums[product]
        while len(product_cross) <= basket_len:
            index = len(product_cross) - 1
            product_cross.append(
                product_cross[-1] + state.mids[product].values[index] * state.basket_sum.values[index]
            )
    return cross_sums


def causal_regression_residual(
    product: str,
    mids: dict[str, float],
    state: deep.SimState,
    window: int,
    ridge: bool,
    factor: str,
) -> float | None:
    current_index = len(state.mids[product].values) - 1
    if current_index < 20:
        return None

    count = min(window, current_index)
    start = current_index - count
    cross_sums = ensure_cross_sums(state)

    y_series = state.mids[product]
    basket_series = state.basket_sum
    sum_y = y_series.sums[current_index] - y_series.sums[start]
    sum_y_square = y_series.sum_squares[current_index] - y_series.sum_squares[start]
    sum_basket = basket_series.sums[current_index] - basket_series.sums[start]
    sum_basket_square = basket_series.sum_squares[current_index] - basket_series.sum_squares[start]
    sum_y_basket = cross_sums[product][current_index] - cross_sums[product][start]

    if factor == "other_four":
        sum_x = (sum_basket - sum_y) / (len(PEBBLES) - 1)
        sum_x_square = (sum_basket_square - 2.0 * sum_y_basket + sum_y_square) / ((len(PEBBLES) - 1) ** 2)
        sum_xy = (sum_y_basket - sum_y_square) / (len(PEBBLES) - 1)
    else:
        sum_x = sum_basket / len(PEBBLES)
        sum_x_square = sum_basket_square / (len(PEBBLES) ** 2)
        sum_xy = sum_y_basket / len(PEBBLES)

    mean_x = sum_x / count
    mean_y = sum_y / count
    var_x = sum_x_square - count * mean_x * mean_x
    if var_x <= 1e-9:
        return None

    cov_xy = sum_xy - count * mean_x * mean_y
    penalty = 0.25 * var_x if ridge else 0.0
    beta = cov_xy / (var_x + penalty)
    alpha = mean_y - beta * mean_x

    if factor == "other_four":
        current_x = sum(mids[p] for p in PEBBLES if p != product) / (len(PEBBLES) - 1)
    else:
        current_x = sum(mids.values()) / len(PEBBLES)
    return mids[product] - (alpha + beta * current_x)


def extended_residuals_for_config(
    mids: dict[str, float],
    bids: dict[str, int],
    asks: dict[str, int],
    bid_volumes: dict[str, int],
    ask_volumes: dict[str, int],
    state: deep.SimState,
    config: deep.StrategyConfig,
) -> dict[str, float]:
    kind = config.residual_kind
    values = [mids[p] for p in PEBBLES]
    group_mean = sum(values) / len(values)
    base_group = {p: (mids[p] - group_mean) * RESIDUAL_SCALE for p in PEBBLES}

    if kind in {
        "group_mean",
        "leave_one_out",
        "basket_constant",
        "robust_median",
        "trimmed_mean",
        "liquidity_weighted",
        "inverse_spread_weighted",
    }:
        return ORIGINAL_RESIDUALS_FOR_CONFIG(mids, bids, asks, bid_volumes, ask_volumes, state, config)

    if kind in {"rolling_regression", "rolling_ridge"}:
        out = {}
        for product in PEBBLES:
            residual = causal_regression_residual(
                product,
                mids,
                state,
                config.window,
                ridge=(kind == "rolling_ridge"),
                factor="other_four",
            )
            out[product] = round((residual if residual is not None else mids[product] - group_mean) * RESIDUAL_SCALE)
        return out

    if kind in {"rolling_factor", "rolling_pca_factor"}:
        out = {}
        for product in PEBBLES:
            residual = causal_regression_residual(
                product,
                mids,
                state,
                config.window,
                ridge=False,
                factor="group_mean",
            )
            out[product] = round((residual if residual is not None else mids[product] - group_mean) * RESIDUAL_SCALE)
        return out

    if kind == "product_vol_norm":
        out = {}
        for product in PEBBLES:
            previous = state.raw_group_residuals[product].values[:-1]
            stats = mean_std(previous, config.window)
            std = stats[1] if stats else RESIDUAL_SCALE
            out[product] = round(base_group[product] / max(std, 1.0) * RESIDUAL_SCALE)
        return out

    if kind == "group_vol_norm":
        stds = []
        for product in PEBBLES:
            stats = mean_std(state.raw_group_residuals[product].values[:-1], config.window)
            if stats is not None:
                stds.append(stats[1])
        group_std = sum(stds) / len(stds) if stds else RESIDUAL_SCALE
        return {product: round(base_group[product] / max(group_std, 1.0) * RESIDUAL_SCALE) for product in PEBBLES}

    if kind.startswith("return_lag_"):
        lag = int(kind.rsplit("_", 1)[1])
        deltas: dict[str, float] = {}
        for product in PEBBLES:
            series = state.mids[product].values
            if len(series) <= lag:
                return {p: round(base_group[p]) for p in PEBBLES}
            deltas[product] = series[-1] - series[-lag - 1]
        delta_mean = sum(deltas.values()) / len(deltas)
        return {product: round((deltas[product] - delta_mean) * RESIDUAL_SCALE) for product in PEBBLES}

    raise ValueError(f"Unknown residual kind: {kind}")


def extended_apply_target_rule(
    residuals: dict[str, float],
    z_values: dict[str, float | None],
    state: deep.SimState,
    config: deep.StrategyConfig,
) -> dict[str, int]:
    if config.target_rule == "wide_residual_spread":
        usable = [z for z in z_values.values() if z is not None]
        if len(usable) < len(PEBBLES) or max(usable) - min(usable) < 4.7:
            return dict(state.targets)
        return ORIGINAL_APPLY_TARGET_RULE(residuals, z_values, state, replace(config, target_rule="independent"))

    if config.target_rule == "two_tick_confirm":
        counts = getattr(state, "confirm_counts", {product: 0 for product in PEBBLES})
        signs = getattr(state, "confirm_signs", {product: 0 for product in PEBBLES})
        targets: dict[str, int] = {}

        for product in PEBBLES:
            previous = state.targets[product]
            z_value = z_values[product]
            if len(state.residuals[product]) < config.min_history or z_value is None:
                counts[product] = 0
                signs[product] = 0
                targets[product] = 0 if len(state.residuals[product]) < config.min_history else previous
                continue

            signal = 0
            if z_value > config.entry_z:
                signal = -1
            elif z_value < -config.entry_z:
                signal = 1

            if signal == 0:
                counts[product] = 0
                signs[product] = 0
                targets[product] = previous
                continue

            if signs[product] == signal:
                counts[product] += 1
            else:
                signs[product] = signal
                counts[product] = 1

            targets[product] = signal * config.target_size if counts[product] >= 2 else previous

        state.confirm_counts = counts
        state.confirm_signs = signs
        return targets

    return ORIGINAL_APPLY_TARGET_RULE(residuals, z_values, state, config)


deep.residuals_for_config = extended_residuals_for_config
deep.apply_target_rule = extended_apply_target_rule


def state_size_estimate_bytes() -> int:
    state = {
        "h": {product: [0] * 500 for product in PEBBLES},
        "t": {product: 0 for product in PEBBLES},
    }
    return len(json.dumps(state, separators=(",", ":")))


def add_common_fields(row: dict[str, Any], family: str) -> dict[str, Any]:
    out = dict(row)
    out["family"] = family
    out["state_size_estimate_bytes"] = state_size_estimate_bytes()
    return out


def unique_configs(configs: list[tuple[str, deep.StrategyConfig]]) -> list[tuple[str, deep.StrategyConfig]]:
    seen: set[str] = set()
    out: list[tuple[str, deep.StrategyConfig]] = []
    for family, config in configs:
        if config.label in seen:
            continue
        seen.add(config.label)
        out.append((family, config))
    return out


def final_configs() -> list[tuple[str, deep.StrategyConfig]]:
    base = deep.baseline_config()
    configs: list[tuple[str, deep.StrategyConfig]] = [("benchmark", base)]

    for residual_kind in (
        "leave_one_out",
        "basket_constant",
        "rolling_regression",
        "rolling_ridge",
        "rolling_factor",
        "rolling_pca_factor",
        "liquidity_weighted",
        "inverse_spread_weighted",
        "trimmed_mean",
        "product_vol_norm",
        "group_vol_norm",
        "return_lag_1",
        "return_lag_3",
        "return_lag_5",
        "return_lag_10",
    ):
        configs.append(("residual", replace(base, label=f"residual_{residual_kind}", residual_kind=residual_kind)))

    for window in (300, 350, 400, 450, 500, 550, 600, 700):
        for entry_z in (2.10, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50, 2.60):
            configs.append(("window", replace(base, label=f"w{window}_z{entry_z:g}", window=window, entry_z=entry_z)))

    for windows in ((400, 500), (450, 500), (400, 500, 600)):
        label = "ensemble_" + "_".join(str(window) for window in windows)
        configs.append(("ensemble", replace(base, label=label, ensemble_windows=windows)))

    configs.extend(
        [
            ("ensemble", replace(base, label="consensus_400_500", ensemble_windows=(400, 500), consensus_count=2)),
            ("ensemble", replace(base, label="consensus_450_500", ensemble_windows=(450, 500), consensus_count=2)),
            (
                "ensemble",
                replace(base, label="consensus_400_500_600", ensemble_windows=(400, 500, 600), consensus_count=3),
            ),
            ("ensemble", replace(base, label="fast_slow_400_600", ensemble_windows=(400, 600), fast_slow=True)),
            ("ensemble", replace(base, label="fast_slow_350_550", ensemble_windows=(350, 550), fast_slow=True)),
        ]
    )

    configs.extend(
        [
            ("target", replace(base, label="uniform_target_9", target_size=9, max_order_size=9)),
            ("target", replace(base, label="uniform_target_8", target_size=8, max_order_size=8)),
            ("target", replace(base, label="xl_target_8", product_caps={"PEBBLES_XL": 8})),
            (
                "target",
                replace(
                    base,
                    label="xl_m_s_target_8",
                    product_caps={"PEBBLES_XL": 8, "PEBBLES_M": 8, "PEBBLES_S": 8},
                ),
            ),
            ("target", replace(base, label="basket_neutral", target_rule="basket_neutral")),
            ("target", replace(base, label="soft_basket_net10", target_rule="soft_basket", soft_net_limit=10)),
            ("target", replace(base, label="rank_pair", target_rule="rank_pair")),
            ("target", replace(base, label="rank2", target_rule="rank2")),
            ("target", replace(base, label="proportional", proportional=True)),
            ("target", replace(base, label="two_stage_z3", two_stage_z=3.0)),
            ("target", replace(base, label="family_min3", family_min_signals=3)),
            ("target", replace(base, label="wide_residual_spread", target_rule="wide_residual_spread")),
            ("target", replace(base, label="two_tick_confirm", target_rule="two_tick_confirm")),
        ]
    )

    configs.extend(
        [
            ("crash", replace(base, label="extreme_z3.5_cap8", extreme_z_cap=(3.5, 8))),
            ("crash", replace(base, label="extreme_z4_cap8", extreme_z_cap=(4.0, 8))),
            ("crash", replace(base, label="extreme_z4.5_cap8", extreme_z_cap=(4.5, 8))),
            ("crash", replace(base, label="vol_ratio1.25_cap8", vol_ratio_cap=(1.25, 8))),
            ("crash", replace(base, label="vol_ratio1.5_cap8", vol_ratio_cap=(1.5, 8))),
            ("crash", replace(base, label="z_slope10_cap8", z_slope_cap=(10.0, 8))),
            ("crash", replace(base, label="z_slope20_cap8", z_slope_cap=(20.0, 8))),
            ("crash", replace(base, label="z_slope30_cap8", z_slope_cap=(30.0, 8))),
            ("crash", replace(base, label="basket_sum_z2.5_cap8", basket_sum_gate=(2.5, 8))),
            ("crash", replace(base, label="basket_sum_z3_cap8", basket_sum_gate=(3.0, 8))),
            ("crash", replace(base, label="xl_entry_plus_0.05", entry_offsets={"PEBBLES_XL": 0.05})),
            ("crash", replace(base, label="xl_entry_plus_0.10", entry_offsets={"PEBBLES_XL": 0.10})),
            ("crash", replace(base, label="ms_entry_plus_0.05", entry_offsets={"PEBBLES_M": 0.05, "PEBBLES_S": 0.05})),
        ]
    )

    configs.append(("execution", replace(base, label="cross_l2_if_needed", use_l2=True)))

    for min_history in (80, 100, 125, 150, 200):
        configs.append(("sensitivity", replace(base, label=f"min_history_{min_history}", min_history=min_history)))
    for history_limit in (400, 500, 600, 700):
        configs.append(("sensitivity", replace(base, label=f"history_limit_{history_limit}", history_limit=history_limit)))

    return unique_configs(configs)


def finalist_configs() -> list[deep.StrategyConfig]:
    base = deep.baseline_config()
    return [
        base,
        replace(base, label="consensus_400_500", ensemble_windows=(400, 500), consensus_count=2),
        replace(base, label="consensus_450_500", ensemble_windows=(450, 500), consensus_count=2),
        replace(base, label="w500_z2.4", entry_z=2.40),
        replace(base, label="xl_entry_plus_0.05", entry_offsets={"PEBBLES_XL": 0.05}),
        replace(base, label="w450_z2.45", window=450, entry_z=2.45),
        replace(base, label="xl_target_8", product_caps={"PEBBLES_XL": 8}),
    ]


def horizon_rows(cached: dict[int, deep.BacktestData], configs: list[deep.StrategyConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizons = (1000, 2000, 5000)
    for config in configs:
        for day in DEFAULT_DAYS:
            result, _ = deep.simulate_day(cached[day], config)
            path = result.pnl_path
            row: dict[str, Any] = {"label": config.label, "day": day, "full": round(path[-1] if path else 0.0, 2)}
            for horizon in horizons:
                row[f"first_{horizon}"] = round(path[horizon - 1], 2) if len(path) >= horizon else ""
            if len(path) >= 5000:
                row["last_5000"] = round(path[-1] - path[-5001], 2)
            else:
                row["last_5000"] = ""
            slice_pnls = []
            for end in range(1000, len(path) + 1, 1000):
                start = end - 1000
                base = path[start - 1] if start > 0 else 0.0
                slice_pnls.append(path[end - 1] - base)
            row["forced_flatten_every_1000_proxy_min"] = round(min(slice_pnls), 2) if slice_pnls else 0.0
            row["forced_flatten_every_1000_proxy_median"] = (
                round(sorted(slice_pnls)[len(slice_pnls) // 2], 2) if slice_pnls else 0.0
            )
            rows.append(row)
    return rows


def diagnostic_rows(data: deep.BacktestData, config: deep.StrategyConfig) -> list[dict[str, Any]]:
    centres = {
        "website_peak_region": 83000,
        "post_peak_drawdown_region": 93200,
        "final_recovery_region": 99800,
    }
    wanted: dict[int, str] = {}
    for region, centre in centres.items():
        for timestamp in range(centre - 1000, centre + 1001, 200):
            wanted[timestamp] = region

    state = deep.SimState()
    positions = {product: 0 for product in PEBBLES}
    cash = {product: 0.0 for product in PEBBLES}
    rows_out: list[dict[str, Any]] = []

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

        raw_group_mean = sum(mids.values()) / len(PEBBLES)
        for product in PEBBLES:
            state.raw_group_residuals[product].append((mids[product] - raw_group_mean) * RESIDUAL_SCALE)
            state.mids[product].append(mids[product])
        state.basket_sum.append(sum(mids.values()))

        residuals = deep.residuals_for_config(mids, bids, asks, bid_volumes, ask_volumes, state, config)
        z_values = {product: deep.signal_z(product, residuals[product], state, config) for product in PEBBLES}
        next_targets = deep.apply_target_rule(residuals, z_values, state, config)

        previous_residuals = {
            product: state.residuals[product].values[-1] if state.residuals[product].values else None
            for product in PEBBLES
        }

        if timestamp in wanted:
            for product in PEBBLES:
                mtm = cash[product] + positions[product] * mids[product]
                previous_residual = previous_residuals[product]
                rows_out.append(
                    {
                        "label": config.label,
                        "region": wanted[timestamp],
                        "timestamp": timestamp,
                        "product": product,
                        "mid": mids[product],
                        "residual": residuals[product],
                        "z": round(z_values[product], 6) if z_values[product] is not None else "",
                        "z_slope": residuals[product] - previous_residual if previous_residual is not None else "",
                        "target": next_targets[product],
                        "position": positions[product],
                        "product_mtm": round(mtm, 2),
                        "basket_sum": round(sum(mids.values()), 2),
                    }
                )

        for product in PEBBLES:
            row = rows[product]
            deep.execute_to_target(
                product,
                next_targets[product],
                positions,
                cash,
                row.bid_prices,
                row.bid_volumes,
                row.ask_prices,
                row.ask_volumes,
                config,
                deep.StressConfig(),
            )

        for product in PEBBLES:
            state.residuals[product].append(residuals[product])
            state.targets[product] = next_targets[product]

    return rows_out


def tick_count_rows(cached: dict[int, deep.BacktestData]) -> list[dict[str, Any]]:
    return [{"day": day, "ticks": len(cached[day].prices)} for day in DEFAULT_DAYS]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backtests/r5_pebbles_final_exhaustive.csv")
    parser.add_argument("--stress-out", default="backtests/r5_pebbles_final_exhaustive_stress.csv")
    parser.add_argument("--horizon-out", default="backtests/r5_pebbles_final_exhaustive_horizons.csv")
    parser.add_argument("--diagnostic-out", default="backtests/r5_pebbles_final_exhaustive_diagnostics.csv")
    parser.add_argument("--summary-out", default="backtests/r5_pebbles_final_exhaustive_summary.json")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached = deep.load_data(DEFAULT_DAYS)
    configs = final_configs()
    if args.quick:
        configs = configs[:12]

    rows: list[dict[str, Any]] = []
    for index, (family, config) in enumerate(configs, start=1):
        row = deep.simulate_config(config, cached, DEFAULT_DAYS)
        rows.append(add_common_fields(row, family))
        print(
            f"{index:03d}/{len(configs):03d} {config.label} "
            f"pnl={float(row['merged_pnl']):.0f} "
            f"days=({float(row['day_2']):.0f},{float(row['day_3']):.0f},{float(row['day_4']):.0f}) "
            f"p05={float(row['roll1000_p05']):.0f} "
            f"wdd={float(row['website_max_drawdown']):.0f}",
            flush=True,
        )

    write_csv(ROOT / args.out, rows)

    finalists = finalist_configs()
    stress_rows: list[dict[str, Any]] = []
    for config in finalists:
        for stress in deep.stress_suite():
            stress_rows.append(add_common_fields(deep.simulate_config(config, cached, DEFAULT_DAYS, stress), "stress"))
    write_csv(ROOT / args.stress_out, stress_rows)

    write_csv(ROOT / args.horizon_out, horizon_rows(cached, finalists))

    diagnostics: list[dict[str, Any]] = []
    for config in finalists[:2]:
        diagnostics.extend(diagnostic_rows(cached[4], config))
    write_csv(ROOT / args.diagnostic_out, diagnostics)

    ranked = sorted(rows, key=lambda row: float(row["merged_pnl"]), reverse=True)
    summary = {
        "tick_counts": tick_count_rows(cached),
        "top_20": ranked[:20],
        "finalists": [row for row in rows if row["label"] in {config.label for config in finalists}],
        "state_size_estimate_bytes": state_size_estimate_bytes(),
        "website_zip_present": bool(list(ROOT.glob("**/*.zip"))),
    }
    write_json(ROOT / args.summary_out, summary)

    print(f"Wrote {ROOT / args.out} ({len(rows)} rows)")
    print(f"Wrote {ROOT / args.stress_out} ({len(stress_rows)} rows)")
    print(f"Wrote {ROOT / args.horizon_out}")
    print(f"Wrote {ROOT / args.diagnostic_out}")
    print(f"Wrote {ROOT / args.summary_out}")

    print("\nTop by merged PnL:")
    for row in ranked[:10]:
        worst_day = min(float(row["day_2"]), float(row["day_3"]), float(row["day_4"]))
        print(
            f"{row['label'][:34]:34s} pnl={float(row['merged_pnl']):9.0f} "
            f"worst={worst_day:8.0f} +5={float(row['stress_5']):9.0f} "
            f"p05={float(row['roll1000_p05']):8.0f} wdd={float(row['website_max_drawdown']):8.0f}"
        )


if __name__ == "__main__":
    main()
