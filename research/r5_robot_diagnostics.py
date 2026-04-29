"""Per-day, stress, attribution and order-audit helpers for the best Robot candidate.

Reads CLI args:

  python research/r5_robot_diagnostics.py <pair_a> <pair_b> <residual> <window> <entry_z>

Runs:
  - match-trades worse / all / none
  - +1 / +3 / +5 / +10 tick stress (--time-shift)
  - product PnL attribution
  - trade count, units, cap dwell estimate, final positions
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
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


DAY_RE = re.compile(r"Round 5 day (\d+):\s*([-\d,]+)")
TOT_RE = re.compile(r"^Total profit:\s*([-\d,]+)\s*$", re.M)
DD_RE = re.compile(r"max_drawdown_abs:\s*([-\d,\.]+)")
PROD_RE = re.compile(r"^([A-Z_]+):\s*([-\d,]+)\s*$", re.M)


def to_int(s: str) -> int:
    return int(s.replace(",", ""))


def run(extra: list[str], env_overrides: dict[str, str] | None = None) -> dict:
    cmd = [PYBIN, "-m", "prosperity4bt", "cli", STRATEGY,
           "5-2", "5-3", "5-4", "--data", DATA, "--merge-pnl", "--no-progress"] + ROBOT_LIMITS + extra
    e = os.environ.copy()
    if env_overrides:
        e.update(env_overrides)
    res = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=600)
    text = res.stdout + res.stderr
    days = {int(d): to_int(p) for d, p in DAY_RE.findall(text)}
    tot = TOT_RE.findall(text)
    total = to_int(tot[-1]) if tot else 0
    dd = DD_RE.findall(text)
    max_dd = float(dd[-1]) if dd else float("nan")
    products = {p: to_int(v) for p, v in PROD_RE.findall(text)
                if p.startswith("ROBOT_") and not p.startswith("ROBOT_S")}
    return {"text": text, "total": total, "d2": days.get(2, 0), "d3": days.get(3, 0),
            "d4": days.get(4, 0), "max_dd": max_dd, "products": products}


def main(argv: list[str]) -> None:
    if len(argv) < 6:
        print("usage: r5_robot_diagnostics.py <a> <b> <residual> <w> <z>")
        sys.exit(2)
    pa, pb, residual, w, z = argv[1], argv[2], argv[3], argv[4], argv[5]
    env = {
        "ROBOT_PAIR_A": pa,
        "ROBOT_PAIR_B": pb,
        "ROBOT_RESIDUAL": residual,
        "ROBOT_WINDOW": w,
        "ROBOT_MIN_HISTORY": w,
        "ROBOT_ENTRY_Z": z,
        "ROBOT_TARGET": "10",
        "ROBOT_BASKET": "0",
    }

    print(f"=== Diagnostics for {pa}/{pb} {residual} w={w} z={z} ===\n")

    print("-- match-trades worse --")
    r_worse = run(["--match-trades", "worse"], env)
    print(f"total={r_worse['total']:,} D2={r_worse['d2']:,} D3={r_worse['d3']:,} D4={r_worse['d4']:,} DD={r_worse['max_dd']:,.0f}")
    print(f"products: {r_worse['products']}\n")

    print("-- match-trades all --")
    r_all = run(["--match-trades", "all"], env)
    print(f"total={r_all['total']:,} D2={r_all['d2']:,} D3={r_all['d3']:,} D4={r_all['d4']:,} DD={r_all['max_dd']:,.0f}\n")

    print("-- match-trades none --")
    r_none = run(["--match-trades", "none"], env)
    print(f"total={r_none['total']:,} D2={r_none['d2']:,} D3={r_none['d3']:,} D4={r_none['d4']:,} DD={r_none['max_dd']:,.0f}\n")

    for shift in (1, 3, 5, 10):
        print(f"-- match-trades worse +{shift} stress --")
        r = run(["--match-trades", "worse", "--time-shift", str(shift)], env)
        print(f"total={r['total']:,} D2={r['d2']:,} D3={r['d3']:,} D4={r['d4']:,} DD={r['max_dd']:,.0f}")

    # Detect available CLI flags by reading help once
    help_out = subprocess.run([PYBIN, "-m", "prosperity4bt", "cli", "--help"], capture_output=True, text=True).stdout
    if "--final-flatten" in help_out:
        print("\n-- final flatten --")
        r = run(["--match-trades", "worse", "--final-flatten"], env)
        print(f"total={r['total']:,} D2={r['d2']:,} D3={r['d3']:,} D4={r['d4']:,} DD={r['max_dd']:,.0f}")
    else:
        print("\n(no --final-flatten flag in this prosperity4bt version)")


if __name__ == "__main__":
    main(sys.argv)
