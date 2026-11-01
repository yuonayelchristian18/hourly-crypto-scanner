"""Walk-forward test: tune on older data, judge ONLY on newer data.

For each coin the history is cut into rolling folds. In every fold the best
settings are chosen using the training window, then applied to the unseen
test window that follows. Only the test windows count toward the results.

Needs scanner.py and signals.py in the same folder. Takes a few minutes.

Usage:
    python walkforward.py
    python walkforward.py --max-coins 8 --bars 6000 --train 2000 --test 500
"""

import argparse
import csv
import itertools
import time

import pandas as pd

import scanner as sc
from signals import SignalConfig, add_indicators

MIN_TRAIN_TRADES = 5


def get_history(session, base, symbol, total):
    """Page backwards through Binance klines to collect `total` hourly candles."""
    frames, end, remaining = [], None, total
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        if end is not None:
            params["endTime"] = end
        response = session.get(f"{base}/klines", params=params, timeout=20)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=sc.COLUMNS))
        end = int(rows[0][0]) - 1
        remaining -= len(rows)
        if len(rows) < limit:
            break
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame(columns=sc.COLUMNS)
    df = pd.concat(frames[::-1], ignore_index=True)
    return df.iloc[:-1].reset_index(drop=True)  # drop the still-forming candle


def build_grid():
    grid = []
    for (fast, slow), (lo, hi), trend in itertools.product(
        [(5, 13), (9, 21), (12, 26)], [(50, 65), (45, 70)], [None, 100]
    ):
        grid.append(SignalConfig(
            fast=fast, slow=slow, rsi_low=lo, rsi_high=hi,
            trend_period=trend, min_vol_ratio=1.0 if trend else 0.0,
        ))
    return grid


def run_slice(raw, start, end, cfg):
    """Backtest cfg over raw[start:end], with warmup taken from earlier rows."""
    warm = max(cfg.trend_period or 0, cfg.slow, cfg.rsi_period, cfg.vol_period) + 2
    lo = max(0, start - warm)
    df = add_indicators(raw.iloc[lo:end].reset_index(drop=True), cfg)
    return sc.backtest(df, cfg)


def describe(cfg):
    trend = f"trend{cfg.trend_period}" if cfg.trend_period else "no-trend"
    return f"EMA{cfg.fast}/{cfg.slow} RSI{cfg.rsi_low:.0f}-{cfg.rsi_high:.0f} {trend}"


def walk_forward(raw, train, test, grid, default_cfg):
    folds, start = [], train
    while start + test <= len(raw):
        best = None
        for cfg in grid:
            result = run_slice(raw, start - train, start, cfg)
            if result["trades"] < MIN_TRAIN_TRADES:
                continue
            if best is None or result["profit"] > best[1]["profit"]:
                best = (cfg, result)

        baseline = run_slice(raw, start, start + test, default_cfg)
        fold = {
            "default_ret": baseline["profit"] / sc.STARTING_BALANCE,
            "bh": baseline["buy_hold"],
            "tuned_ret": 0.0,      # no qualifying settings = stay in cash
            "trades": 0,
            "settings": "none (stayed flat)",
        }
        if best is not None:
            oos = run_slice(raw, start, start + test, best[0])
            fold.update(
                tuned_ret=oos["profit"] / sc.STARTING_BALANCE,
                trades=oos["trades"],
                settings=describe(best[0]),
            )
        folds.append(fold)
        start += test
    return folds


def compound(returns):
    total = 1.0
    for r in returns:
        total *= 1 + r
    return total - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--categories", default=sc.DEFAULT_CATEGORIES)
    parser.add_argument("--max-coins", type=int, default=12)
    parser.add_argument("--bars", type=int, default=4000, help="hourly candles per coin")
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--csv", default="walkforward_results.csv")
    args = parser.parse_args()

    session = sc.make_session()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    universe = sc.get_universe(session, categories, 100, 1.0, 1_000_000, 20_000_000)
    base, binance_symbols = sc.pick_binance(session)

    grid, default_cfg = build_grid(), SignalConfig()
    print(f"{len(universe)} candidate coins, {len(grid)} setting combinations, "
          f"train {args.train} / test {args.test} bars\n")

    rows, seen, all_folds = [], set(), []

    for coin in universe:
        if len(rows) >= args.max_coins:
            break
        name = coin["symbol"].upper()
        symbol = name + "USDT"
        if name in sc.STABLES or symbol in seen or symbol not in binance_symbols:
            continue
        seen.add(symbol)

        try:
            raw = get_history(session, base, symbol, args.bars)
            if len(raw) < args.train + args.test:
                print(f"{symbol}: only {len(raw)} candles, skipped")
                continue

            gap = abs(float(raw.iloc[-1]["close"]) / coin["current_price"] - 1)
            if gap > sc.PRICE_TOLERANCE:
                print(f"{symbol}: ticker clash suspected ({gap * 100:.0f}% price gap), skipped")
                continue

            folds = walk_forward(raw, args.train, args.test, grid, default_cfg)
            all_folds.extend(folds)

            row = {
                "symbol": symbol,
                "folds": len(folds),
                "tuned_oos": compound(f["tuned_ret"] for f in folds),
                "default_oos": compound(f["default_ret"] for f in folds),
                "buy_hold": compound(f["bh"] for f in folds),
                "beat_bh": sum(f["tuned_ret"] > f["bh"] for f in folds),
                "trades": sum(f["trades"] for f in folds),
            }
            rows.append(row)
            print(f"{symbol:<12} folds {row['folds']}  "
                  f"tuned {row['tuned_oos'] * 100:>7.1f}%  "
                  f"default {row['default_oos'] * 100:>7.1f}%  "
                  f"buy&hold {row['buy_hold'] * 100:>7.1f}%  "
                  f"beat B&H in {row['beat_bh']}/{row['folds']}  trades {row['trades']}")
        except Exception as exc:
            print(f"{symbol}: error - {exc}")

    if not rows:
        print("No results.")
        return

    n = len(rows)
    avg = lambda key: sum(r[key] for r in rows) / n
    total_folds = len(all_folds)
    beat = sum(f["tuned_ret"] > f["bh"] for f in all_folds)
    profitable = sum(f["tuned_ret"] > 0 for f in all_folds)
    flat = sum(f["trades"] == 0 for f in all_folds)

    print("\n=== OUT-OF-SAMPLE SUMMARY (test windows only) ===")
    print(f"Coins: {n}   Folds: {total_folds}")
    print(f"Average tuned strategy:  {avg('tuned_oos') * 100:+.1f}%")
    print(f"Average default settings: {avg('default_oos') * 100:+.1f}%")
    print(f"Average buy & hold:       {avg('buy_hold') * 100:+.1f}%")
    print(f"Folds profitable: {profitable}/{total_folds}   "
          f"beat buy & hold: {beat}/{total_folds}   no trades: {flat}/{total_folds}")

    if avg("tuned_oos") > avg("buy_hold") and beat / total_folds >= 0.6:
        print("Verdict: some evidence of an edge. Confirm it with paper trading "
              "before using real money.")
    elif avg("tuned_oos") > 0:
        print("Verdict: profitable but not clearly better than just holding. "
              "Treat as unproven.")
    else:
        print("Verdict: no out-of-sample edge found. Do not trade this with real money.")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()