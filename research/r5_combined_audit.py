"""Parse a prosperity4bt combined backtest stdout to extract product/day PnLs and
report additivity vs. frozen core + standalone Microchip references."""

import os, re, sys, json
from collections import defaultdict


def parse_stdout(path):
    """Returns (overall_total, by_day, by_product) from the *final* sections."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    # Find the *final* "Profit summary:" block (after merge-pnl)
    # Pattern: "Round 5 day N: <num>" followed by "Total profit: <num>"
    # But the file lists per-day product PnLs preceding each day's Total profit too.
    # The merged summary is the last block.
    days = {}
    overall_total = None
    by_product_per_day = defaultdict(dict)  # day -> {product: pnl}
    # Per-day blocks structure:
    # ... block header ... lines like "PRODUCT: 1,234"
    # then "Total profit: <num>" then "Profit summary:" appears once, with the
    # round 5 day lines.
    # Simpler heuristic: the final lines before "Profit summary:" are the merged-day per-product PnLs.
    # We'll scan for Round-day section markers if printed.
    # Capture "Round 5 day N: VAL" lines
    for line in lines:
        m = re.match(r"^Round 5 day (\d+): ([-\d,]+)$", line)
        if m:
            days[int(m.group(1))] = int(m.group(2).replace(",", ""))
    for line in reversed(lines):
        m = re.match(r"^Total profit: ([-\d,]+)$", line)
        if m:
            overall_total = int(m.group(1).replace(",", ""))
            break
    # Per-product PnL: scan blocks separated by blank lines or 'Round 5 day' headers.
    current_day = None
    current_prod_block = {}
    for line in lines:
        # Day separator
        m_day = re.match(r"^Round 5 day (\d+):.*$", line)
        if m_day:
            # Close existing block: assign to current_day if we have one
            if current_day is not None and current_prod_block:
                by_product_per_day[current_day] = dict(current_prod_block)
            current_day = int(m_day.group(1))
            current_prod_block = {}
            continue
        m_prod = re.match(r"^([A-Z_0-9]+): ([-\d,]+)$", line)
        if m_prod and current_day is None:
            # Per-product line for the in-progress day before any 'Round 5 day' header
            # We'll record into a pre-day block keyed as -1
            current_prod_block[m_prod.group(1)] = int(m_prod.group(2).replace(",", ""))
        elif m_prod:
            current_prod_block[m_prod.group(1)] = int(m_prod.group(2).replace(",", ""))
    return overall_total, days, by_product_per_day


def main():
    files = {
        "final_worse": "backtests/r5_trader_worse.log",
        "final_all": "backtests/r5_trader_all.log",
        "final_none": "backtests/r5_trader_none.log",
        "microchip_worse": "backtests/r5_microchip_worse.log",
        "pebbles_worse": "backtests/r5_pebbles_worse.log",
        "translator_worse": "backtests/r5_translator_worse.log",
    }
    for label, path in files.items():
        if not os.path.exists(path):
            print(f"{label}: MISSING {path}")
            continue
        try:
            tot, days, _ = parse_stdout(path)
        except Exception as e:
            print(f"{label}: parse error {e}")
            continue
        print(f"{label}: total={tot:,} days={ {d: f'{v:,}' for d,v in sorted(days.items())} }")


if __name__ == "__main__":
    main()
