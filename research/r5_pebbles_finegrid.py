"""Phase-7 robustness: fine grid around the locked candidate w500 z2.35 t10."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from r5_pebbles_research import Variant, load_cached_data, run_variant  # type: ignore


def main() -> None:
    cached = load_cached_data((2, 3, 4))

    print("\n=== fine entry_z sweep (w=500, t=10, hold) ===")
    rows = []
    for z in (2.28, 2.30, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.40, 2.42):
        row = run_variant(Variant(window=500, entry_z=z, target_size=10), (2, 3, 4), cached)
        rows.append(row)
        worst = min(row["day_2"], row["day_3"], row["day_4"])
        print(
            f"  z={z:.2f}  pnl={row['merged_pnl']:7.0f}  +5={row['stress_5']:7.0f}  "
            f"worst={worst:7.0f}  units={row['traded_units']:5d}  slice_dd={row['slice_p2t_dd']:6.0f}  maxdd={row['max_drawdown']:6.0f}"
        )

    print("\n=== fine window sweep (z=2.35, t=10, hold) ===")
    for w in (470, 480, 490, 500, 510, 520, 530, 540):
        row = run_variant(Variant(window=w, entry_z=2.35, target_size=10), (2, 3, 4), cached)
        worst = min(row["day_2"], row["day_3"], row["day_4"])
        print(
            f"  w={w}  pnl={row['merged_pnl']:7.0f}  +5={row['stress_5']:7.0f}  "
            f"worst={worst:7.0f}  units={row['traded_units']:5d}  slice_dd={row['slice_p2t_dd']:6.0f}  maxdd={row['max_drawdown']:6.0f}"
        )

    print("\n=== day-order permutation test (w=500, z=2.35, t=10) ===")
    base = run_variant(Variant(window=500, entry_z=2.35, target_size=10), (2, 3, 4), cached)
    print(f"  order 2,3,4: pnl={base['merged_pnl']:.0f} units={base['traded_units']}")
    for order in [(4, 3, 2), (3, 2, 4), (2, 4, 3), (4, 2, 3), (3, 4, 2)]:
        row = run_variant(Variant(window=500, entry_z=2.35, target_size=10), order, cached)
        print(
            f"  order {','.join(str(d) for d in order)}: pnl={row['merged_pnl']:7.0f}  "
            f"days=({row['day_2']:.0f},{row['day_3']:.0f},{row['day_4']:.0f})  units={row['traded_units']}"
        )


if __name__ == "__main__":
    main()
