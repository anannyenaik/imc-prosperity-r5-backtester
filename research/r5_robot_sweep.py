"""Grid-sweep harness for Robot pair candidates.

Runs strategies/archive/r5_robot_candidate.py through prosperity4bt for every (pair,
residual, window, entry_z) cell. Captures total + per-day PnL by parsing the
backtest stdout (no log-file scraping). Also supports stress modes.

Usage:
  python research/r5_robot_sweep.py [grid_name]

Grid names:
  laundry_vac    – LAUNDRY/VACUUMING raw + logratio
  dishes_iron    – DISHES/IRONING raw + logratio
  discovery      – DISHES/VAC, MOP/VAC, DISHES/LAUNDRY, MOP/LAUNDRY, etc.
  basket         – all-five basket residual
  stress         – +1/+3/+5/+10 stress on a single config (BEST_*)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYBIN = str(ROOT / ".venv" / "Scripts" / "python.exe")
STRATEGY = str(ROOT / "strategies" / "archive" / "r5_robot_candidate.py")
DATA = str(ROOT / "r5_data")

ROBOT_LIMITS = [
    "--limit", "ROBOT_VACUUMING:10",
    "--limit", "ROBOT_MOPPING:10",
    "--limit", "ROBOT_DISHES:10",
    "--limit", "ROBOT_LAUNDRY:10",
    "--limit", "ROBOT_IRONING:10",
]

PROD_LIMITS = [
    "--limit", "PEBBLES_XS:10",
    "--limit", "PEBBLES_S:10",
    "--limit", "PEBBLES_M:10",
    "--limit", "PEBBLES_L:10",
    "--limit", "PEBBLES_XL:10",
    "--limit", "TRANSLATOR_SPACE_GRAY:10",
    "--limit", "TRANSLATOR_ASTRO_BLACK:10",
    "--limit", "TRANSLATOR_ECLIPSE_CHARCOAL:10",
    "--limit", "TRANSLATOR_GRAPHITE_MIST:10",
    "--limit", "TRANSLATOR_VOID_BLUE:10",
    "--limit", "MICROCHIP_OVAL:10",
    "--limit", "MICROCHIP_TRIANGLE:10",
] + ROBOT_LIMITS


DAY_RE = re.compile(r"Round 5 day (\d+):\s*([-\d,]+)")
TOT_RE = re.compile(r"^Total profit:\s*([-\d,]+)\s*$", re.M)
DD_RE = re.compile(r"max_drawdown_abs:\s*([-\d,\.]+)")


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def run_backtest(env: dict, strategy: str, days, limits, extra: list[str] | None = None) -> dict:
    cmd = [PYBIN, "-m", "prosperity4bt", "cli", strategy, *days, "--data", DATA,
           "--match-trades", "worse", "--merge-pnl", "--no-progress"] + limits
    if extra:
        cmd += extra
    full_env = os.environ.copy()
    full_env.update({k: str(v) for k, v in env.items()})
    out = subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=600)
    text = out.stdout + out.stderr
    days_pnl = {int(d): _to_int(p) for d, p in DAY_RE.findall(text)}
    tot_matches = TOT_RE.findall(text)
    total = _to_int(tot_matches[-1]) if tot_matches else 0
    dd_match = DD_RE.search(text)
    max_dd = float(dd_match.group(1).replace(",", "")) if dd_match else float("nan")
    return {
        "total": total,
        "d2": days_pnl.get(2, 0),
        "d3": days_pnl.get(3, 0),
        "d4": days_pnl.get(4, 0),
        "max_dd": max_dd,
        "rc": out.returncode,
    }


def fmt_row(name: str, params: dict, res: dict) -> str:
    return (
        f"{name:<32} w={params.get('w',''):>5} z={params.get('z',''):<5} "
        f"res={params.get('res','raw'):<8} "
        f"total={res['total']:>8,} D2={res['d2']:>7,} D3={res['d3']:>7,} D4={res['d4']:>7,} "
        f"DD={res['max_dd']:>8,.0f}"
    )


def grid_pair(pair_a: str, pair_b: str, residuals, windows, zs):
    rows = []
    for res, w, z in product(residuals, windows, zs):
        env = {
            "ROBOT_PAIR_A": pair_a,
            "ROBOT_PAIR_B": pair_b,
            "ROBOT_RESIDUAL": res,
            "ROBOT_WINDOW": w,
            "ROBOT_MIN_HISTORY": w,
            "ROBOT_ENTRY_Z": z,
            "ROBOT_TARGET": 10,
            "ROBOT_BASKET": "0",
        }
        r = run_backtest(env, STRATEGY, ["5-2", "5-3", "5-4"], ROBOT_LIMITS)
        params = {"w": w, "z": z, "res": res}
        line = fmt_row(f"{pair_a}/{pair_b}", params, r)
        print(line, flush=True)
        rows.append((pair_a, pair_b, res, w, z, r))
    return rows


def grid_basket(windows, zs):
    rows = []
    for w, z in product(windows, zs):
        env = {
            "ROBOT_BASKET": "1",
            "ROBOT_WINDOW": w,
            "ROBOT_MIN_HISTORY": w,
            "ROBOT_ENTRY_Z": z,
            "ROBOT_TARGET": 10,
        }
        r = run_backtest(env, STRATEGY, ["5-2", "5-3", "5-4"], ROBOT_LIMITS)
        params = {"w": w, "z": z, "res": "basket"}
        line = fmt_row("BASKET_5_residual", params, r)
        print(line, flush=True)
        rows.append((w, z, r))
    return rows


def main(argv: list[str]) -> None:
    grid = argv[1] if len(argv) > 1 else "laundry_vac"

    if grid == "laundry_vac":
        windows = [1500, 1750, 2000, 2250, 2500, 2750, 3000]
        zs = [1.50, 1.75, 2.00, 2.25, 2.50]
        print("# LAUNDRY/VACUUMING grid")
        grid_pair("ROBOT_LAUNDRY", "ROBOT_VACUUMING", ["raw", "logratio"], windows, zs)

    elif grid == "dishes_iron":
        windows = [750, 1000, 1250, 1500, 1750, 2000]
        zs = [1.25, 1.50, 1.75, 2.00, 2.25]
        print("# DISHES/IRONING grid")
        grid_pair("ROBOT_DISHES", "ROBOT_IRONING", ["raw", "logratio"], windows, zs)

    elif grid == "discovery":
        # Coarser grid for discovery
        windows = [1500, 2000, 2500]
        zs = [1.75, 2.00, 2.25]
        pairs = [
            ("ROBOT_DISHES", "ROBOT_VACUUMING"),
            ("ROBOT_MOPPING", "ROBOT_VACUUMING"),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY"),
            ("ROBOT_MOPPING", "ROBOT_LAUNDRY"),
            ("ROBOT_LAUNDRY", "ROBOT_IRONING"),
            ("ROBOT_VACUUMING", "ROBOT_IRONING"),
            ("ROBOT_DISHES", "ROBOT_MOPPING"),
            ("ROBOT_MOPPING", "ROBOT_IRONING"),
        ]
        for a, b in pairs:
            print(f"# discovery {a}/{b}")
            grid_pair(a, b, ["raw"], windows, zs)

    elif grid == "basket":
        windows = [1000, 1500, 2000, 2500]
        zs = [1.50, 1.75, 2.00, 2.25, 2.50]
        print("# all-5 basket residual grid")
        grid_basket(windows, zs)

    elif grid == "neighbour_lv":
        # neighbourhood around best LV cell. set BEST_W, BEST_Z env, default 2250/2.00
        bw = int(os.environ.get("BEST_W", 2250))
        bz = float(os.environ.get("BEST_Z", 2.00))
        windows = [bw - 250, bw - 125, bw, bw + 125, bw + 250]
        zs = [round(bz - 0.25, 2), round(bz - 0.125, 3), bz, round(bz + 0.125, 3), round(bz + 0.25, 2)]
        print(f"# LAUNDRY/VAC neighbourhood around w={bw} z={bz}")
        grid_pair("ROBOT_LAUNDRY", "ROBOT_VACUUMING", ["raw"], windows, zs)

    else:
        print(f"unknown grid: {grid}")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
