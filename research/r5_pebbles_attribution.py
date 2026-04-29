"""Print per-product attribution for baseline and challenger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from r5_pebbles_research import Variant, load_cached_data, run_variant  # type: ignore


def main() -> None:
    cached = load_cached_data((2, 3, 4))
    for variant in (
        Variant(window=500, entry_z=2.25, target_size=10),
        Variant(window=500, entry_z=2.35, target_size=10),
    ):
        row = run_variant(variant, (2, 3, 4), cached)
        print(f"\n=== {row['label']} ===")
        print(f"merged={row['merged_pnl']} days=({row['day_2']},{row['day_3']},{row['day_4']})")
        print(f"units={row['traded_units']} +1={row['stress_1']} +3={row['stress_3']} +5={row['stress_5']}")
        print(f"max_drawdown={row['max_drawdown']}")
        print(f"slice peak={row['slice_peak']} trough={row['slice_trough']} final={row['slice_final']} dd={row['slice_p2t_dd']}")
        print(f"forced_flatten={row['forced_flatten_pnl']}")
        print(f"per_product: {row['per_product_pnl']}")
        print(f"traded_units_by_product: {row['traded_units_by_product']}")
        print(f"final_positions: {row['final_positions']}")
        print(f"cap_dwell: {row['cap_dwell']}  (out of {3*10000} ticks)")
        print(f"terminal_inventory_value: {row['terminal_inventory_value']}")


if __name__ == "__main__":
    main()
