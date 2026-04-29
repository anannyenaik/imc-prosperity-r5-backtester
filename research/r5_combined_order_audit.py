"""Audit order safety in the combined backtest:

- No forbidden Microchip products traded (only OVAL and TRIANGLE allowed).
- Microchip positions stay within +/-10.
- Pebbles/Translator behaviour identical to the frozen-core combined log
  (sanity-check by parsing per-product totals from both stdouts).
- Any state-size / runtime concerns.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict


COMBINED_NEW = "backtests/r5_trader_worse.log"
COMBINED_OLD = "backtests/r5_pebbles_translator_worse.log"
ALLOWED_MICROCHIPS = {"MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"}
ALL_MICROCHIPS = {f"MICROCHIP_{x}" for x in ("CIRCLE", "OVAL", "RECTANGLE", "SQUARE", "TRIANGLE")}
PEBBLES = {f"PEBBLES_{x}" for x in ("XS", "S", "M", "L", "XL")}
TRANSLATORS = {f"TRANSLATOR_{x}" for x in ("SPACE_GRAY", "ASTRO_BLACK", "ECLIPSE_CHARCOAL", "GRAPHITE_MIST", "VOID_BLUE")}


def parse_trade_history(path: str) -> dict:
    """Walk the JSON-ish trade history at the end of a backtest log and return
    {product: {'submission_buys': [...], 'submission_sells': [...], 'positions_over_time': [...] }}.
    """
    if not os.path.exists(path):
        return {"missing": True}
    # We will scan the file for the last array starting "[{" at top level.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read()
    # Find the trade history block: the last big JSON list
    # Format: starts with "[" on its own line near end of file.
    # Quick approach: parse sequentially using regex on entries.
    submission_buys = defaultdict(list)
    submission_sells = defaultdict(list)
    other_trades = defaultdict(int)
    pattern = re.compile(
        r"\{\s*\"timestamp\":\s*(\d+),\s*\"buyer\":\s*\"([^\"]*)\",\s*\"seller\":\s*\"([^\"]*)\",\s*\"symbol\":\s*\"([^\"]+)\",\s*\"currency\":\s*\"[^\"]*\",\s*\"price\":\s*(-?\d+),\s*\"quantity\":\s*(\d+)"
    )
    for m in pattern.finditer(data):
        ts = int(m.group(1)); buyer = m.group(2); seller = m.group(3); sym = m.group(4); price = int(m.group(5)); qty = int(m.group(6))
        if buyer == "SUBMISSION":
            submission_buys[sym].append((ts, price, qty))
        elif seller == "SUBMISSION":
            submission_sells[sym].append((ts, price, qty))
        else:
            other_trades[sym] += qty
    return {
        "submission_buys": submission_buys,
        "submission_sells": submission_sells,
        "other_trades": other_trades,
    }


def position_audit(buys, sells):
    """Compute peak/trough position, max position size."""
    events = []
    for ts, price, qty in buys:
        events.append((ts, +qty, price))
    for ts, price, qty in sells:
        events.append((ts, -qty, price))
    events.sort(key=lambda x: x[0])
    pos = 0
    max_pos = 0
    min_pos = 0
    cumulative_buys = 0
    cumulative_sells = 0
    for ts, q, p in events:
        if q > 0:
            cumulative_buys += q
        else:
            cumulative_sells += -q
        pos += q
        if pos > max_pos:
            max_pos = pos
        if pos < min_pos:
            min_pos = pos
    return {
        "trade_events": len(events),
        "max_position": max_pos,
        "min_position": min_pos,
        "final_position": pos,
        "total_buys": cumulative_buys,
        "total_sells": cumulative_sells,
    }


def main():
    new = parse_trade_history(COMBINED_NEW)
    old = parse_trade_history(COMBINED_OLD)
    if new.get("missing"):
        print("MISSING new combined log; abort"); sys.exit(1)

    forbidden_chips = ALL_MICROCHIPS - ALLOWED_MICROCHIPS

    print("=== Microchip products audit ===")
    for chip in sorted(ALL_MICROCHIPS):
        nb = sum(q for _, _, q in new["submission_buys"].get(chip, []))
        ns = sum(q for _, _, q in new["submission_sells"].get(chip, []))
        forbidden = chip in forbidden_chips
        marker = "FORBIDDEN" if forbidden else "ALLOWED"
        flag = " *** VIOLATION ***" if forbidden and (nb > 0 or ns > 0) else ""
        print(f"  {chip}: buys={nb} sells={ns} ({marker}){flag}")

    print("\n=== Microchip position audit (allowed only) ===")
    for chip in sorted(ALLOWED_MICROCHIPS):
        a = position_audit(new["submission_buys"].get(chip, []), new["submission_sells"].get(chip, []))
        print(f"  {chip}: {a}")

    print("\n=== Pebbles / Translator additivity vs. frozen-core combined log ===")
    if not old.get("missing"):
        for chip in sorted(PEBBLES | TRANSLATORS):
            old_b = sum(q for _, _, q in old["submission_buys"].get(chip, []))
            old_s = sum(q for _, _, q in old["submission_sells"].get(chip, []))
            new_b = sum(q for _, _, q in new["submission_buys"].get(chip, []))
            new_s = sum(q for _, _, q in new["submission_sells"].get(chip, []))
            same = (old_b == new_b) and (old_s == new_s)
            mark = "OK" if same else "DIFF"
            print(f"  {chip}: old_buys={old_b} old_sells={old_s} new_buys={new_b} new_sells={new_s} -> {mark}")
    else:
        print("  (frozen-core log missing)")


if __name__ == "__main__":
    main()
