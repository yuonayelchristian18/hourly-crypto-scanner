import csv
import itertools
import json
import math
import os
import time

import requests

COINS = ["bitcoin", "ethereum", "solana", "ripple", "cardano"]

DAYS = 90
STARTING_CASH = 1000
FEE = 0.001            # per side
SLIPPAGE = 0.0005      # per side, adverse
TRAIN_FRACTION = 0.7
MIN_TRAIN_TRADES = 3

DEFAULT_PAIR = (6, 24)
SHORT_GRID = [3, 6, 9, 12]
LONG_GRID = [24, 48, 72, 120]

CACHE_DIR = "cache"
CACHE_TTL = 3600       # seconds


def get_prices(coin, retries=5):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{coin}_{DAYS}d.json")

    if os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_TTL:
        with open(path) as f:
            return json.load(f)

    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": DAYS}

    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        response.raise_for_status()
        prices = response.json().get("prices", [])
        with open(path, "w") as f:
            json.dump(prices, f)
        return prices

    raise RuntimeError("Rate limited by CoinGecko after several retries")


def sma(values, window):
    out = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        if i >= window - 1:
            out[i] = running / window
    return out


def simulate(values, short, long_):
    """Signal on bar i's close, fill at bar i+1's price (no look-ahead)."""
    fast, slow = sma(values, short), sma(values, long_)
    cash, qty, cash_in = STARTING_CASH, 0.0, 0.0
    equity, trades, bars_in = [], [], 0
    pending = None

    for i in range(long_, len(values)):
        price = values[i]

        if pending == "buy" and qty == 0:
            fill = price * (1 + SLIPPAGE)
            cash_in = cash
            qty = cash * (1 - FEE) / fill
            cash = 0.0
        elif pending == "sell" and qty > 0:
            fill = price * (1 - SLIPPAGE)
            cash = qty * fill * (1 - FEE)
            qty = 0.0
            trades.append(cash / cash_in - 1)
        pending = None

        equity.append(cash + qty * price)
        if qty > 0:
            bars_in += 1

        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            pending = "buy"
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            pending = "sell"

    return equity, trades, bars_in


def metrics(equity, trades, bars_in):
    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = min(max_dd, e / peak - 1)

    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
    sharpe = mean / sd * math.sqrt(24 * 365) if sd else 0.0

    wins = sum(1 for t in trades if t > 0)
    return {
        "return": equity[-1] / STARTING_CASH - 1,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "exposure": bars_in / len(equity),
    }


def run_coin(coin):
    prices = get_prices(coin)
    if len(prices) < 500:
        print(f"{coin}: not enough data ({len(prices)} points)")
        return None

    gaps = sorted(
        (prices[i][0] - prices[i - 1][0]) / 60000 for i in range(1, len(prices))
    )
    median_gap = gaps[len(gaps) // 2]
    if median_gap > 90:
        print(f"{coin}: data is not hourly (median gap {median_gap:.0f} min), skipped")
        return None

    values = [p[1] for p in prices]
    split = int(len(values) * TRAIN_FRACTION)
    train = values[:split]

    # Parameter sweep on the training window only.
    best = None
    for s, l in itertools.product(SHORT_GRID, LONG_GRID):
        if s >= l:
            continue
        m = metrics(*simulate(train, s, l))
        if m["trades"] < MIN_TRAIN_TRADES:
            continue
        if best is None or m["sharpe"] > best[0]["sharpe"]:
            best = (m, s, l)

    def test(s, l):
        # Start l bars early so the averages are warm when the test begins.
        return metrics(*simulate(values[split - l:], s, l))

    buy_hold = (values[-1] / values[split]) * (1 - FEE) ** 2 - 1
    default_test = test(*DEFAULT_PAIR)

    print(f"\n{coin.upper()}  (test window: {len(values) - split} hourly bars)")
    print(f"  Buy & hold:           {buy_hold * 100:7.2f}%")
    print(
        f"  Default {DEFAULT_PAIR}:     {default_test['return'] * 100:7.2f}%  "
        f"DD {default_test['max_dd'] * 100:6.2f}%  trades {default_test['trades']}"
    )

    row = {"coin": coin, "buy_hold": buy_hold, "default_return": default_test["return"]}

    if best is None:
        print("  Tuned: no parameter pair produced enough training trades")
    else:
        _, s, l = best
        t = test(s, l)
        print(
            f"  Tuned ({s},{l}):        {t['return'] * 100:7.2f}%  "
            f"DD {t['max_dd'] * 100:6.2f}%  Sharpe {t['sharpe']:5.2f}  "
            f"trades {t['trades']}  win {t['win_rate'] * 100:.0f}%  "
            f"in-market {t['exposure'] * 100:.0f}%"
        )
        row.update(
            short=s, long=l, tuned_return=t["return"], max_dd=t["max_dd"],
            sharpe=t["sharpe"], trades=t["trades"], win_rate=t["win_rate"],
        )
    return row


def main():
    rows = []
    for coin in COINS:
        try:
            row = run_coin(coin)
            if row:
                rows.append(row)
        except Exception as error:
            print(f"{coin}: Error - {error}")
        time.sleep(3)

    if rows:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "coin", k))
        with open("results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("\nSaved results.csv")


if __name__ == "__main__":
    main()

 