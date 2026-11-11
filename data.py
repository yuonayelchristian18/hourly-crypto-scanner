"""data.py - Coinbase public candles (no API key needed).

fetch_candles() returns a DataFrame (UTC DatetimeIndex, columns open/high/low/close/volume,
oldest first) containing CLOSED candles only, or None if the pair does not exist / fails.

Set GW_DEMO=1 to get synthetic data instead (offline smoke test, no internet).
"""
import hashlib
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = "https://api.exchange.coinbase.com"
GRAN = {"15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}

# AI + meme + your focus coins. Pairs Coinbase does not list are skipped automatically.
PRODUCTS = [
    # meme
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "POPCAT-USD",
    "MOG-USD", "TURBO-USD", "BRETT-USD", "PNUT-USD", "FARTCOIN-USD", "MEW-USD", "GIGA-USD",
    # your focus
    "XRP-USD", "ZK-USD", "AURORA-USD",
    # AI
    "FET-USD", "RENDER-USD", "GRT-USD", "TAO-USD", "WLD-USD", "AKT-USD", "VIRTUAL-USD",
    "ARKM-USD", "NEAR-USD", "IMX-USD", "AIOZ-USD", "OCEAN-USD",
    # liquid cheap coins
    "ADA-USD", "XLM-USD", "HBAR-USD", "VET-USD", "ALGO-USD", "SUI-USD", "SEI-USD",
    "ARB-USD", "OP-USD", "POL-USD",
]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _demo(product: str, interval: str, days: int) -> pd.DataFrame:
    import gwvsa
    g = GRAN[interval]
    n = int(days * 86400 / g)
    seed = int(hashlib.md5(product.encode()).hexdigest()[:8], 16)
    df = gwvsa._synthetic(seed, n)
    last = pd.Timestamp.now(tz="UTC").floor(f"{g}s") - pd.Timedelta(seconds=g)
    df.index = pd.date_range(end=last, periods=n, freq=f"{g}s")
    return df


def fetch_candles(product: str, interval: str = "1h", days: int = 21, retries: int = 3):
    if os.getenv("GW_DEMO") == "1":
        return _demo(product, interval, days)
    g = GRAN[interval]
    now = int(time.time())
    floor_limit = now - days * 86400
    end = now
    rows = []
    while end > floor_limit:
        start = max(floor_limit, end - 300 * g)
        data = None
        for attempt in range(retries):
            try:
                r = requests.get(f"{BASE}/products/{product}/candles",
                                 params={"granularity": g, "start": _iso(start), "end": _iso(end)},
                                 timeout=20)
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1)); continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # network hiccup
                if attempt == retries - 1:
                    print(f"  ! {product}: {e}")
                    return None
                time.sleep(1.0)
        if not data:
            break
        rows.extend(data)
        end = start
        time.sleep(0.12)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.drop_duplicates("time").sort_values("time")
    df = df[df["time"] + g <= now]                      # drop the still-forming candle
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df if len(df) else None
