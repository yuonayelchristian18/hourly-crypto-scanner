
import sys
import time

import numpy as np
import pandas as pd
import requests

API = "https://api.coingecko.com/api/v3"
CATEGORIES = ["artificial-intelligence", "meme-token"]
MAX_PRICE = 1.0
MAX_COINS = 15
ALERT_SCORE = 70
HORIZON = 24  # hours ahead to measure the result


def get(path, params=None, retries=4):
    for i in range(retries):
        try:
            r = requests.get(
                f"{API}{path}",
                params=params,
                timeout=30,
                headers={"User-Agent": "crypto-scanner"},
            )

            if r.status_code == 429:
                print("Rate limited, waiting...")
                time.sleep(20 * (i + 1))
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            print(f"Request failed ({path}): {e}")
            time.sleep(5 * (i + 1))

    return None


def pick_coins():
    coins = {}

    for cat in CATEGORIES:
        data = get(
            "/coins/markets",
            {
                "vs_currency": "usd",
                "category": cat,
                "order": "volume_desc",
                "per_page": 100,
                "page": 1,
            },
        )

        if not data:
            continue

        for c in data:
            price = c.get("current_price")

            if price is not None and price < MAX_PRICE:
                coins[c["id"]] = c["symbol"].upper()

        time.sleep(3)

    return list(coins.items())[:MAX_COINS]


def history(coin_id):
    d = get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": 90},
    )

    if not d or not d.get("prices") or not d.get("total_volumes"):
        return None

    n = min(len(d["prices"]), len(d["total_volumes"]))

    if n == 0:
        return None

    df = pd.DataFrame(
        {
            "ts": [x[0] for x in d["prices"][:n]],
            "price": [x[1] for x in d["prices"][:n]],
            "volume": [x[1] for x in d["total_volumes"][:n]],
        }
    )

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    return df.set_index("ts")


def add_score(df):
    df = df.copy()

    p = df["price"]

    mom = p.pct_change(24)

    mom_rank = mom.rolling(
        24 * 30,
        min_periods=24 * 7
    ).rank(pct=True)

    vol_ratio = df["volume"] / df["volume"].rolling(
        24 * 7,
        min_periods=24
    ).mean()

    vol_rank = vol_ratio.rolling(
        24 * 30,
        min_periods=24 * 7
    ).rank(pct=True)

    trend = (
        p > p.rolling(24 * 7, min_periods=24).mean()
    ).astype(float)

    df["score"] = (
        50 * mom_rank
        + 30 * vol_rank
        + 20 * trend
    )

    df["fwd"] = p.shift(-HORIZON) / p - 1

    return df


def main():
    coins = pick_coins()

    if not coins:
        sys.exit(
            "No coins returned from CoinGecko. "
            "See request errors above."
        )

    print(
        f"Testing {len(coins)} coins: "
        f"{', '.join(s for _, s in coins)}"
    )

    frames = []
    latest = {}

    for cid, sym in coins:

        print(f"\nDownloading {sym} history...")

        df = history(cid)

        time.sleep(7)

        # FIX 1: Check that enough historical data exists.
        if df is None or len(df) < 24 * 14:
            print(f"Skipping {sym}: insufficient data.")
            continue

        df = add_score(df)

        valid = df.dropna(subset=["score", "fwd"])

        if valid.empty:
            print(f"Skipping {sym}: no valid scores.")
            continue

        latest[sym] = {
            "score": float(df["score"].iloc[-1]),
            "price": float(df["price"].iloc[-1]),
        }

        valid = valid.copy()
        valid["symbol"] = sym

        frames.append(valid)

        print(
            f"{sym}: latest score "
            f"{latest[sym]['score']:.2f}"
        )

    if not frames:
        sys.exit("No valid historical data available.")

    combined = pd.concat(frames)

    # Historical signals
    signals = combined[
        combined["score"] >= ALERT_SCORE
    ].copy()

    if signals.empty:
        print("\nNo signals reached the alert threshold.")
        return

    # FIX 2: Avoid evaluating an empty signal dataset.
    if len(signals) == 0:
        print("No signals available for evaluation.")
        return

    signals["win"] = signals["fwd"] > 0

    total = len(signals)
    wins = int(signals["win"].sum())

    win_rate = (wins / total) * 100

    average_return = signals["fwd"].mean() * 100

    median_return = signals["fwd"].median() * 100

    print("\n========== WALK-FORWARD RESULTS ==========")

    print(f"Total historical signals: {total}")
    print(f"Winning signals: {wins}")
    print(f"Losing signals: {total - wins}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average forward return: {average_return:.2f}%")
    print(f"Median forward return: {median_return:.2f}%")

    print("\n========== LATEST COIN SCORES ==========")

    ranked = sorted(
        latest.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    for sym, data in ranked:
        status = (
            "BUY SIGNAL"
            if data["score"] >= ALERT_SCORE
            else "WATCH"
        )

        print(
            f"{sym}: Score {data['score']:.2f} | "
            f"Price ${data['price']:.8f} | {status}"
        )

    print("\n========== TOP HISTORICAL SIGNALS ==========")

    top = signals.sort_values(
        "score",
        ascending=False
    ).head(10)

    for _, row in top.iterrows():
        print(
            f"{row['symbol']} | "
            f"Score: {row['score']:.2f} | "
            f"Forward return: {row['fwd'] * 100:.2f}%"
        )

    print("\nWalk-forward analysis completed.")


if __name__ == "__main__":
    main()
