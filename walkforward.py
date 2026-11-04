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
                print("rate limited, waiting...")
                time.sleep(20 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"request failed ({path}): {e}")
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
    d = get(f"/coins/{coin_id}/market_chart", {"vs_currency": "usd", "days": 90})
    if not d or not d.get("prices"):
        return None
    n = min(len(d["prices"]), len(d["total_volumes"]))
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
    p = df["price"]
    mom = p.pct_change(24)
    mom_rank = mom.rolling(24 * 30, min_periods=24 * 7).rank(pct=True)
    vol_ratio = df["volume"] / df["volume"].rolling(24 * 7, min_periods=24).mean()
    vol_rank = vol_ratio.rolling(24 * 30, min_periods=24 * 7).rank(pct=True)
    trend = (p > p.rolling(24 * 7, min_periods=24).mean()).astype(float)
    df["score"] = 50 * mom_rank + 30 * vol_rank + 20 * trend
    df["fwd"] = p.shift(-HORIZON) / p - 1
    return df


def main():
    coins = pick_coins()
    if not coins:
        sys.exit("No coins returned from CoinGecko. See request errors above.")
    print(f"Testing {len(coins)} coins: {', '.join(s for _, s in coins)}")

    frames = []
    latest = {}
    for cid, sym in coins:
        df = history(cid)
        time.sleep(7)  # stay under the free rate limit
        if df is None or len(df) 
