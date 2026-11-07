import itertools
import numpy as np
import pandas as pd
import trend_bot as tb

DAYS = 365
TRAIN_D, TEST_D = 120, 30
MIN_TRAIN_TRADES = 20
GRID = {
    "brk": [24, 48, 96],
    "atr_mult": [2.0, 3.0, 4.0],
    "vol_mult": [1.0, 1.5, 2.0],
    "ema_len": [100, 200],
}


def in_window(t, a, b):
    return t[(t["time"] >= a) & (t["time"] < b)]


def main():
    btc, data = tb.load_all(DAYS)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"\nTesting {len(combos)} setting combos...")
    results = []
    for k, p in enumerate(combos, 1):
        results.append((p, tb.run_all(data, btc, p)))
        if k % 10 == 0:
            print(f"  {k}/{len(combos)} done")
    fixed_t = tb.run_all(data, btc, tb.DEFAULT)

    start, end = btc.index[210], btc.index[-1]
    train_d, test_d = pd.Timedelta(days=TRAIN_D), pd.Timedelta(days=TEST_D)
    oos, fixed = [], []
    ts = start
    print("\nWINDOWS (settings chosen on past data only):")
    while ts + train_d + test_d <= end:
        a, b, c = ts, ts + train_d, ts + train_d + test_d
        best = None
        for p, t in results:
            tr = in_window(t, a, b)
            if len(tr) < MIN_TRAIN_TRADES:
                continue
            score = tr["r"].mean()
            if best is None or score > best[0]:
                best = (score, p, t)
        fixed.append(in_window(fixed_t, b, c))
        if best is None:
            print(f"{b.date()}: no setting qualified, skipped")
        else:
            te = in_window(best[2], b, c)
            oos.append(te)
            print(f"{b.date()}: {best[1]} -> test trades {len(te)}, "
                  f"net R {te['r'].sum():.1f}")
        ts += test_d

    if not oos:
        print("No out-of-sample trades.")
        return
    oos = pd.concat(oos)
    fixed = pd.concat(fixed)
    oos.to_csv("walkforward_trades.csv", index=False)

    s = tb.stats(oos)
    print("\nOUT-OF-SAMPLE (stitched):", s)
    print("DEFAULT SETTINGS, same periods:", tb.stats(fixed))

    rng = np.random.default_rng(0)
    r = oos["r"].values
    sims = np.array([rng.choice(r, len(r)).sum() for _ in range(5000)])
    p5 = np.percentile(sims, 5)
    print(f"Bootstrap: 5th percentile net R = {p5:.1f}, "
          f"chance of losing = {(sims < 0).mean() * 100:.0f}%")

    bh = btc["close"].iloc[-1] / btc["close"].iloc[0] - 1
    print(f"BTC buy-and-hold over {DAYS} days: {bh * 100:.1f}%")

    ok = s["trades"] >= 50 and s["profit_factor"] >= 1.2 and p5 > 0
    print("\nVERDICT:", "possible edge, keep paper trading" if ok
          else "no proven edge yet, do not use real money")


if __name__ == "__main__":
    main()
