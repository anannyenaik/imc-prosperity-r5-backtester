"""Quick reproducibility check vs the user's expected baseline / challenger diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from r5_pebbles_research import Variant, load_cached_data, run_variant  # type: ignore


def main() -> None:
    days = (2, 3, 4)
    cached = load_cached_data(days)

    targets = [
        Variant(window=500, entry_z=2.25, target_size=10),
        Variant(window=500, entry_z=2.35, target_size=10),
    ]
    for variant in targets:
        row = run_variant(variant, days, cached)
        print(
            f"{row['label']}: merged={row['merged_pnl']:.0f} "
            f"days=({row['day_2']:.0f},{row['day_3']:.0f},{row['day_4']:.0f}) "
            f"units={row['traded_units']} stress5={row['stress_5']:.0f} "
            f"slice_final={row['slice_final']:.0f} slice_p2t_dd={row['slice_p2t_dd']:.0f} "
            f"maxdd={row['max_drawdown']:.0f} forced_flat={row['forced_flatten_pnl']:.0f}"
        )


if __name__ == "__main__":
    main()
