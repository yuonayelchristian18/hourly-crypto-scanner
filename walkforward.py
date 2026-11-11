"""walkforward.py - honest test of the course strategies on real Coinbase history.

  python walkforward.py                       # 180 days of 1h candles, default rules
  python walkforward.py --mode more --days 365
  python walkforward.py --cost 0.001          # lighter costs per side

Rules are FIXED (course defaults, no tuning). The last 30% of each pair's history is
reported as the "test" split. Entry = next candle open; stop assumed first if a candle
hits both stop and target. Costs are charged per side (default 0.25% = fee 0.10% + slip 0.15%).
"""
import argparse

import pandas as pd

from data import PRODUCTS, fetch_candles
from gwvsa import Params, backtest, summarize


def fmt(s):
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    return (f"trades {s['trades']:4d} | win {s['win_rate'] * 100:5.1f}% | avg {s['avg_R']:+.3f}R | "
            f"net {s['net_R']:+8.2f}R | PF {pf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--mode", default="original", choices=["original", "more", "max"])
    ap.add_argument("--cost", type=float, default=0.0025)
    ap.add_argument("--split", type=float, default=0.70)
    ap.add_argument("--partial", action="store_true", help="take half at 1R")
    ap.add_argument("--products", nargs="*", default=None)
    args = ap.parse_args()

    p = Params(mode=args.mode)
    out, used = [], 0
    for pr in (args.products or PRODUCTS):
        df = fetch_candles(pr, args.interval, args.days)
        if df is None or len(df) < 400:
            print(f"skip {pr}"); continue
        used += 1
        tr = backtest(df, p, cost_per_side=args.cost, partial_1r=args.partial)
        if len(tr):
            cut = df.index[int(len(df) * args.split)]
            tr["product"] = pr
            tr["split"] = ["test" if t >= cut else "train" for t in tr["entry_time"]]
            out.append(tr)
        print(f"{pr:<14} bars {len(df):5d} trades {len(tr):3d}")
    if not out:
        print(f"\n{used} pairs tested, 0 trades. Try --mode more."); return
    t = pd.concat(out, ignore_index=True)
    t.to_csv("walkforward_trades.csv", index=False)

    print(f"\n=== {used} pairs | {args.days}d {args.interval} | mode {args.mode} | cost {args.cost * 100:.2f}%/side ===")
    print("ALL   ", fmt(summarize(t)))
    print("TRAIN ", fmt(summarize(t[t.split == "train"])))
    te = summarize(t[t.split == "test"])
    print("TEST  ", fmt(te))
    print("\nBy pattern (all trades):")
    for name, g in t.groupby("pattern"):
        print(f"  {name:<26}", fmt(summarize(g)))
    print("\nBy exit reason:")
    for name, g in t.groupby("reason"):
        print(f"  {name:<12}", fmt(summarize(g)))

    if te["trades"] < 30:
        v = "too few test trades to judge (try --mode more or --days 365)"
    elif te["profit_factor"] >= 1.3 and te["avg_R"] > 0:
        v = "worth forward-testing on paper (still not proof)"
    else:
        v = "no proven edge"
    print("\nVERDICT:", v)


if __name__ == "__main__":
    main()
