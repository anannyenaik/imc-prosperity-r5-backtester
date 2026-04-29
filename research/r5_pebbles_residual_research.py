"""Quick Phase-5 alpha tests for alternative residual definitions.

Tests:
  - leave_one_out: residual = mid - mean(other 4 mids), instead of mean(all 5)
  - basket_constrained: residual = mid - share where share = (50000/5)
    (uses the constant 50,000 sum approximately; falsifiable assumption)
  - robust_median: residual = mid - median(all 5 mids)

Each is run with WINDOW=500, ENTRY_Z=2.35, TARGET=10, hold-to-flip; no other
changes vs the production trader. Acceptance bar (per task): >=10k robust uplift
without breaking any day, or else REJECT.
"""

from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from r5_pebbles_research import LIMIT, PEBBLES, RESIDUAL_SCALE, load_cached_data, path_metrics  # type: ignore


@dataclass
class Cfg:
    window: int = 500
    entry_z: float = 2.35
    target_size: int = 10
    min_history: int = 125
    history_limit: int = 700
    min_std_scaled: float = 10.0
    residual_kind: str = "group_mean"  # "group_mean", "leave_one_out", "basket_constrained", "robust_median"


def z_score(history, current_residual, cfg: Cfg):
    lookback = history[-cfg.window:]
    if len(lookback) < cfg.min_history:
        return None
    mean = sum(lookback) / len(lookback)
    ms = sum(v * v for v in lookback) / len(lookback)
    var = max(0.0, ms - mean * mean)
    std = math.sqrt(var)
    if std < cfg.min_std_scaled:
        return None
    return (current_residual - mean) / std


def residuals_for_kind(mids: dict, kind: str) -> dict:
    products = list(PEBBLES)
    values = [mids[p] for p in products]
    if kind == "group_mean":
        gm = sum(values) / len(values)
        return {p: int(round((mids[p] - gm) * RESIDUAL_SCALE)) for p in products}
    if kind == "leave_one_out":
        out = {}
        total = sum(values)
        n = len(values)
        for p in products:
            mu = (total - mids[p]) / (n - 1)
            out[p] = int(round((mids[p] - mu) * RESIDUAL_SCALE))
        return out
    if kind == "basket_constrained":
        per_share = 50000.0 / len(values)
        return {p: int(round((mids[p] - per_share) * RESIDUAL_SCALE)) for p in products}
    if kind == "robust_median":
        med = statistics.median(values)
        return {p: int(round((mids[p] - med) * RESIDUAL_SCALE)) for p in products}
    raise ValueError(f"unknown kind {kind}")


def simulate_day(template, cfg: Cfg):
    histories = {p: [] for p in PEBBLES}
    targets = {p: 0 for p in PEBBLES}
    positions = {p: 0 for p in PEBBLES}
    cash = {p: 0.0 for p in PEBBLES}
    units = {p: 0 for p in PEBBLES}
    cap_dwell = {p: 0 for p in PEBBLES}
    pnl_path = []
    final_pnl = {p: 0.0 for p in PEBBLES}
    last_mids = {p: 0.0 for p in PEBBLES}

    for ts in sorted(template.prices.keys()):
        rows = template.prices[ts]
        if any(p not in rows for p in PEBBLES):
            continue
        bids = {}
        asks = {}
        bvols = {}
        avols = {}
        mids = {}
        complete = True
        for p in PEBBLES:
            row = rows[p]
            if not row.bid_prices or not row.ask_prices:
                complete = False
                break
            bids[p] = row.bid_prices[0]
            asks[p] = row.ask_prices[0]
            bvols[p] = row.bid_volumes[0] if row.bid_volumes else 0
            avols[p] = row.ask_volumes[0] if row.ask_volumes else 0
            mids[p] = (bids[p] + asks[p]) / 2.0
        if not complete:
            pnl_path.append(sum(cash[p] + positions[p] * mids.get(p, 0.0) for p in PEBBLES))
            continue

        last_mids = dict(mids)
        residuals = residuals_for_kind(mids, cfg.residual_kind)
        next_targets = {}
        for p in PEBBLES:
            z = z_score(histories[p], residuals[p], cfg)
            if len(histories[p]) < cfg.min_history:
                next_targets[p] = 0
            elif z is None:
                next_targets[p] = targets[p]
            elif z > cfg.entry_z:
                next_targets[p] = -cfg.target_size
            elif z < -cfg.entry_z:
                next_targets[p] = cfg.target_size
            else:
                next_targets[p] = targets[p]

        for p in PEBBLES:
            final_pnl[p] = cash[p] + positions[p] * mids[p]
        pnl_path.append(sum(final_pnl.values()))

        for p in PEBBLES:
            target = max(-LIMIT, min(LIMIT, next_targets[p]))
            cur = positions[p]
            delta = target - cur
            if delta > 0:
                qty = min(delta, avols[p], max(0, LIMIT - cur), cfg.target_size)
                if qty > 0:
                    positions[p] += qty
                    cash[p] -= asks[p] * qty
                    units[p] += qty
            elif delta < 0:
                qty = min(-delta, bvols[p], max(0, LIMIT + cur), cfg.target_size)
                if qty > 0:
                    positions[p] -= qty
                    cash[p] += bids[p] * qty
                    units[p] += qty

        for p in PEBBLES:
            histories[p].append(residuals[p])
            if len(histories[p]) > cfg.history_limit:
                del histories[p][:len(histories[p]) - cfg.history_limit]
            targets[p] = next_targets[p]
            if abs(positions[p]) >= LIMIT:
                cap_dwell[p] += 1

    return {
        "day": template.day_num,
        "pnl_path": pnl_path,
        "final_pnl": dict(final_pnl),
        "units": dict(units),
        "cap_dwell": dict(cap_dwell),
        "final_positions": dict(positions),
    }


def run_kind(kind: str, cached):
    cfg = Cfg(residual_kind=kind)
    days = sorted(cached.keys())
    day_results = [simulate_day(cached[d], cfg) for d in days]
    day_pnl = {f"day_{r['day']}": sum(r["final_pnl"].values()) for r in day_results}
    total = sum(day_pnl.values())
    units = sum(sum(r["units"].values()) for r in day_results)
    stitched = []
    offset = 0.0
    for r in day_results:
        stitched.extend(offset + v for v in r["pnl_path"])
        if r["pnl_path"]:
            offset += r["pnl_path"][-1]
    pm = path_metrics(stitched)
    day4 = next((r for r in day_results if r["day"] == 4), None)
    slice_metrics = path_metrics(day4["pnl_path"][:1000]) if day4 else {}
    return {
        "kind": kind,
        "total": total,
        "day_pnl": day_pnl,
        "units": units,
        "stress5": total - 5 * units,
        "max_dd": pm["p2t_dd"],
        "slice_dd": slice_metrics.get("p2t_dd", 0),
        "slice_final": slice_metrics.get("final", 0),
    }


def main():
    cached = load_cached_data((2, 3, 4))
    for kind in ("group_mean", "leave_one_out", "basket_constrained", "robust_median"):
        r = run_kind(kind, cached)
        print(
            f"{r['kind']:24s} total={r['total']:8.0f} "
            f"days=({r['day_pnl']['day_2']:.0f},{r['day_pnl']['day_3']:.0f},{r['day_pnl']['day_4']:.0f}) "
            f"units={r['units']} +5={r['stress5']:.0f} "
            f"maxdd={r['max_dd']:.0f} slice_dd={r['slice_dd']:.0f} slice_final={r['slice_final']:.0f}"
        )


if __name__ == "__main__":
    main()
