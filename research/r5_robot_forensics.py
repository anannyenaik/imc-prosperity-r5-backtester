"""Robot product forensics for IMC Prosperity Round 5.

Reads price + trade CSVs for days 2/3/4 and reports:
- product mid stats per day,
- pair spread stats and AR(1) half-life,
- same-tick + lagged correlations,
- group/basket residual diagnostics,
- regime stability (first half vs second half, day vs day).

This script is research-only and never used at trade time.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "r5_data" / "round5"
DAYS = (2, 3, 4)
ROBOTS = (
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
)


def load_prices() -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for day in DAYS:
        df = pd.read_csv(DATA / f"prices_round_5_day_{day}.csv", sep=";")
        df = df[df["product"].isin(ROBOTS)]
        wide = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
        wide = wide.sort_index()
        out[day] = wide
    return out


def product_stats(prices: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for day, wide in prices.items():
        for prod in ROBOTS:
            s = wide[prod].dropna()
            if s.empty:
                continue
            ret = s.diff().dropna()
            rows.append(
                {
                    "day": day,
                    "product": prod,
                    "n": len(s),
                    "mean": round(s.mean(), 2),
                    "std": round(s.std(), 2),
                    "min": s.min(),
                    "max": s.max(),
                    "abs_drift": round(s.iloc[-1] - s.iloc[0], 2),
                    "ret_std": round(ret.std(), 3),
                    "ret_abs_mean": round(ret.abs().mean(), 3),
                }
            )
    return pd.DataFrame(rows)


def half_life(series: pd.Series) -> float:
    s = series.dropna().to_numpy()
    if len(s) < 50:
        return float("nan")
    x = s[:-1]
    y = s[1:] - s[:-1]
    # OLS slope of dy on (x - mean(x))
    xc = x - x.mean()
    denom = (xc * xc).sum()
    if denom == 0:
        return float("nan")
    rho = (xc * y).sum() / denom
    if rho >= 0:
        return float("inf")
    return -math.log(2) / math.log(1 + rho) if (1 + rho) > 0 else float("nan")


def autocorr(series: pd.Series, lag: int) -> float:
    s = series.dropna()
    if len(s) <= lag + 5:
        return float("nan")
    return s.autocorr(lag)


def pair_stats(prices: dict[int, pd.DataFrame], pair: tuple[str, str]) -> pd.DataFrame:
    a, b = pair
    rows: list[dict] = []
    for day, wide in prices.items():
        if a not in wide.columns or b not in wide.columns:
            continue
        spread = (wide[a] - wide[b]).dropna()
        log_ratio = (wide[a].apply(math.log) - wide[b].apply(math.log)).dropna()
        rows.append(
            {
                "day": day,
                "pair": f"{a}-{b}",
                "n": len(spread),
                "spread_mean": round(spread.mean(), 2),
                "spread_std": round(spread.std(), 2),
                "spread_min": spread.min(),
                "spread_max": spread.max(),
                "drift": round(spread.iloc[-1] - spread.iloc[0], 2),
                "ar1": round(autocorr(spread, 1), 4),
                "ar10": round(autocorr(spread, 10), 4),
                "ar100": round(autocorr(spread, 100), 4),
                "half_life": round(half_life(spread), 1),
                "logratio_std": round(log_ratio.std(), 5),
            }
        )
    return pd.DataFrame(rows)


def same_tick_corr(prices: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Same-tick mid-return correlation per day."""
    out: dict[int, pd.DataFrame] = {}
    for day, wide in prices.items():
        rets = wide[list(ROBOTS)].diff().dropna()
        out[day] = rets.corr().round(3)
    return out


def lagged_corr_table(prices: dict[int, pd.DataFrame], lags: tuple[int, ...]) -> pd.DataFrame:
    """For each (predictor, target) pair, lagged correlation: ret_pred(t) vs ret_target(t+lag)."""
    rows: list[dict] = []
    for day, wide in prices.items():
        rets = wide[list(ROBOTS)].diff().dropna()
        for pred in ROBOTS:
            for tgt in ROBOTS:
                if pred == tgt:
                    continue
                rp = rets[pred]
                rt = rets[tgt]
                row = {"day": day, "predictor": pred, "target": tgt}
                for lag in lags:
                    if lag <= 0:
                        continue
                    s = rp.iloc[: -lag].reset_index(drop=True)
                    t = rt.iloc[lag:].reset_index(drop=True)
                    n = min(len(s), len(t))
                    if n < 100:
                        row[f"L{lag}"] = float("nan")
                        continue
                    a = s.iloc[:n]
                    b = t.iloc[:n]
                    sa = a.std()
                    sb = b.std()
                    if sa == 0 or sb == 0:
                        row[f"L{lag}"] = float("nan")
                        continue
                    row[f"L{lag}"] = round(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb), 4)
                rows.append(row)
    return pd.DataFrame(rows)


def basket_residual_stats(prices: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for day, wide in prices.items():
        df = wide[list(ROBOTS)].dropna()
        if df.empty:
            continue
        group_mean = df.mean(axis=1)
        for prod in ROBOTS:
            res = df[prod] - group_mean
            rows.append(
                {
                    "day": day,
                    "product": prod,
                    "res_mean": round(res.mean(), 3),
                    "res_std": round(res.std(), 3),
                    "res_drift": round(res.iloc[-1] - res.iloc[0], 3),
                    "ar1": round(autocorr(res, 1), 4),
                    "half_life": round(half_life(res), 1),
                }
            )
    return pd.DataFrame(rows)


def regime_split_stats(prices: dict[int, pd.DataFrame], pair: tuple[str, str]) -> pd.DataFrame:
    a, b = pair
    rows: list[dict] = []
    for day, wide in prices.items():
        if a not in wide.columns or b not in wide.columns:
            continue
        s = (wide[a] - wide[b]).dropna()
        half = len(s) // 2
        h1 = s.iloc[:half]
        h2 = s.iloc[half:]
        rows.append(
            {
                "day": day,
                "pair": f"{a}-{b}",
                "h1_mean": round(h1.mean(), 2),
                "h1_std": round(h1.std(), 2),
                "h2_mean": round(h2.mean(), 2),
                "h2_std": round(h2.std(), 2),
                "h1_ar1": round(autocorr(h1, 1), 4),
                "h2_ar1": round(autocorr(h2, 1), 4),
            }
        )
    return pd.DataFrame(rows)


PAIRS_OF_INTEREST = [
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"),
    ("ROBOT_DISHES", "ROBOT_IRONING"),
    ("ROBOT_DISHES", "ROBOT_VACUUMING"),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING"),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY"),
    ("ROBOT_MOPPING", "ROBOT_LAUNDRY"),
    ("ROBOT_LAUNDRY", "ROBOT_IRONING"),
    ("ROBOT_VACUUMING", "ROBOT_IRONING"),
    ("ROBOT_DISHES", "ROBOT_MOPPING"),
    ("ROBOT_MOPPING", "ROBOT_IRONING"),
]


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    prices = load_prices()

    print("=" * 80)
    print("PRODUCT MID STATS")
    print("=" * 80)
    print(product_stats(prices).to_string(index=False))

    print()
    print("=" * 80)
    print("PAIR SPREAD STATS")
    print("=" * 80)
    for pair in PAIRS_OF_INTEREST:
        df = pair_stats(prices, pair)
        if df.empty:
            continue
        print(df.to_string(index=False))
        print()

    print("=" * 80)
    print("SAME-TICK MID-RETURN CORRELATIONS (per day)")
    print("=" * 80)
    for day, m in same_tick_corr(prices).items():
        print(f"-- day {day} --")
        print(m.to_string())
        print()

    print("=" * 80)
    print("LAGGED CROSS-CORRELATIONS (predictor t -> target t+lag)")
    print("=" * 80)
    lc = lagged_corr_table(prices, (1, 5, 20, 50, 100, 200, 300))
    # show only |max lag corr| > 0.04 to flag interesting ones
    lc["max_abs"] = lc[[c for c in lc.columns if c.startswith("L")]].abs().max(axis=1)
    print(lc.sort_values(["day", "max_abs"], ascending=[True, False]).head(40).to_string(index=False))

    print()
    print("=" * 80)
    print("ALL-FIVE BASKET RESIDUAL STATS")
    print("=" * 80)
    print(basket_residual_stats(prices).to_string(index=False))

    print()
    print("=" * 80)
    print("REGIME SPLIT (first half vs second half) FOR PAIRS OF INTEREST")
    print("=" * 80)
    for pair in PAIRS_OF_INTEREST[:4]:
        df = regime_split_stats(prices, pair)
        if df.empty:
            continue
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
