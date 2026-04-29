"""Analyse the research-sweep CSVs and surface plateau / risk-controlled candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "backtests/r5_pebbles_research_primary.csv"
CRASH = ROOT / "backtests/r5_pebbles_research_crash.csv"
HISTORY = ROOT / "backtests/r5_pebbles_research_history.csv"
EXIT = ROOT / "backtests/r5_pebbles_research_exit.csv"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def to_int(s: str) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def normalise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flat = dict(row)
        for k in (
            "merged_pnl",
            "day_2",
            "day_3",
            "day_4",
            "stress_1",
            "stress_3",
            "stress_5",
            "max_drawdown",
            "path_peak",
            "path_trough",
            "path_final",
            "slice_peak",
            "slice_trough",
            "slice_final",
            "slice_p2t_dd",
            "forced_flatten_pnl",
            "entry_z",
        ):
            flat[k] = to_float(row.get(k, "0"))
        flat["traded_units"] = to_int(row.get("traded_units", "0"))
        flat["window"] = to_int(row.get("window", "0"))
        flat["target_size"] = to_int(row.get("target_size", "0"))
        out.append(flat)
    return out


def show_top(rows: list[dict[str, Any]], key: str, n: int, header: str) -> None:
    print(f"\n=== {header} ===")
    ranked = sorted(rows, key=lambda r: r.get(key, 0), reverse=True)[:n]
    for row in ranked:
        worst_day = min(row["day_2"], row["day_3"], row["day_4"])
        print(
            f"{row['label'][:55]:55s} pnl={row['merged_pnl']:7.0f} "
            f"+5={row['stress_5']:7.0f} worst={worst_day:7.0f} "
            f"slice_dd={row['slice_p2t_dd']:6.0f} maxdd={row['max_drawdown']:6.0f} "
            f"units={row['traded_units']:5d}"
        )


def show_plateau(rows: list[dict[str, Any]], baseline_label: str) -> None:
    """Find plateau around the baseline by neighbours in window/entry_z space."""
    baseline = next((r for r in rows if r["label"] == baseline_label), None)
    if baseline is None:
        print(f"baseline {baseline_label} not found")
        return
    print(f"\n=== Plateau around {baseline_label} (target_size={baseline['target_size']}) ===")
    print(f"baseline: pnl={baseline['merged_pnl']:.0f} +5={baseline['stress_5']:.0f}")
    target_size = baseline["target_size"]
    window = baseline["window"]
    entry_z = baseline["entry_z"]
    for row in rows:
        if row["target_size"] != target_size:
            continue
        if abs(row["window"] - window) > 100:
            continue
        if abs(row["entry_z"] - entry_z) > 0.20:
            continue
        worst_day = min(row["day_2"], row["day_3"], row["day_4"])
        print(
            f"  {row['label'][:50]:50s} pnl={row['merged_pnl']:7.0f} "
            f"+5={row['stress_5']:7.0f} worst={worst_day:7.0f} "
            f"slice_dd={row['slice_p2t_dd']:6.0f} maxdd={row['max_drawdown']:6.0f}"
        )


def filter_pass(rows: list[dict[str, Any]], min_worst_day: float = 50000.0, min_stress5: float = 280000.0) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        worst_day = min(row["day_2"], row["day_3"], row["day_4"])
        if worst_day < min_worst_day:
            continue
        if row["stress_5"] < min_stress5:
            continue
        out.append(row)
    return out


def crash_compare(crash_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    print(f"\n=== Crash-control vs baseline {baseline['label']} ===")
    print(f"baseline: pnl={baseline['merged_pnl']:.0f} slice_dd={baseline['slice_p2t_dd']:.0f} maxdd={baseline['max_drawdown']:.0f}")
    sorted_rows = sorted(crash_rows, key=lambda r: r["slice_p2t_dd"])
    print(f"\n{'label':<70s} {'pnl':>8s} {'pnl%':>6s} {'slice_dd':>9s} {'maxdd':>7s} {'worst':>7s} {'units':>6s}")
    for row in sorted_rows:
        worst_day = min(row["day_2"], row["day_3"], row["day_4"])
        pnl_pct = 100.0 * row["merged_pnl"] / baseline["merged_pnl"] if baseline["merged_pnl"] else 0.0
        print(
            f"{row['label'][:70]:<70s} {row['merged_pnl']:8.0f} {pnl_pct:6.1f} "
            f"{row['slice_p2t_dd']:9.0f} {row['max_drawdown']:7.0f} {worst_day:7.0f} "
            f"{row['traded_units']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="w500_z2.35_hold_t10")
    args = parser.parse_args()

    primary = normalise(read_rows(PRIMARY))
    crash = normalise(read_rows(CRASH))
    history = normalise(read_rows(HISTORY))
    exit_rows = normalise(read_rows(EXIT))

    print(f"primary={len(primary)}  crash={len(crash)}  history={len(history)}  exit={len(exit_rows)}")

    if primary:
        passing = filter_pass(primary, min_worst_day=50000.0, min_stress5=280000.0)
        print(f"\nprimary passing min_worst_day>=50k & stress5>=280k: {len(passing)}/{len(primary)}")
        show_top(primary, "merged_pnl", 20, "Top 20 by merged PnL (primary)")
        show_top(primary, "stress_5", 20, "Top 20 by stress_5 (primary)")
        show_top(passing, "merged_pnl", 20, "Top 20 by merged PnL among passing (worst_day>=50k, +5>=280k)")
        # robustness: rank by worst_day then +5 then merged
        ranked = sorted(
            primary,
            key=lambda r: (
                min(r["day_2"], r["day_3"], r["day_4"]),
                r["stress_5"],
                r["merged_pnl"],
            ),
            reverse=True,
        )
        print("\n=== Top 20 by (worst_day, stress_5, merged_pnl) ===")
        for row in ranked[:20]:
            worst = min(row["day_2"], row["day_3"], row["day_4"])
            print(
                f"  {row['label'][:55]:55s} pnl={row['merged_pnl']:7.0f} "
                f"+5={row['stress_5']:7.0f} worst={worst:7.0f} "
                f"slice_dd={row['slice_p2t_dd']:6.0f} units={row['traded_units']:5d}"
            )
        show_plateau(primary, args.baseline)
        show_plateau(primary, "w500_z2.25_hold_t10")
        show_plateau(primary, "w450_z2.35_hold_t10")

    baseline = next((r for r in primary if r["label"] == args.baseline), None)
    if crash and baseline is not None:
        crash_compare(crash, baseline)

    if history:
        show_top(history, "merged_pnl", 30, "All history-control configs by merged PnL")

    if exit_rows:
        show_top(exit_rows, "merged_pnl", 30, "All exit-control configs by merged PnL")


if __name__ == "__main__":
    main()
