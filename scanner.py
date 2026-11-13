"""Hourly crypto scanner (paper/alerts only, no trading).

Scores each Coinbase pair 0-100 on 1h candles and prints alerts at 70+.
Requires: requests
"""
import sys
import time

import requests

PAIRS = [
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD",
    "XRP-USD", "ZK-USD", "AURORA-USD", "FET-USD", "RNDR-USD", "TAO-USD",
    "ARB-USD", "OP-USD", "ADA-USD", "XLM-USD", "HBAR-USD", "SUI-USD",
    "SEI-USD", "APT-USD", "NEAR-USD", "INJ-USD", "TIA-USD", "JASMY-USD",
    "VET-USD", "ALGO-USD", "CRV-USD", "GALA-USD", "MANA-USD", "SAND-USD",
]
ALERT_SCORE = 70
URL = "https://api.exchange.coinbase.com/products/{}/candles"


def get_candles(pair):
    """Return list of (close, volume), oldest first, or None on failure."""
    try:
        r = requests.get(URL.format(pair), params={"granularity": 3600}, timeout=15)
        if r.status_code != 200:
            return None
        rows = sorted(r.json(), key=lambda x: x[0])
        return [(row[4], row[5]) for row in rows]
    except Exception:
        return None


def ema(values, n):
    k = 2 / (n + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi(closes, n=14):
    gains, losses = [], []
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains) / n
    avg_l = sum(losses) / n
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def score(candles):
    closes = [c for c, _ in candles]
    vols = [v for _, v in candles]
    price = closes[-1]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    pts = 0
    if price > e20:
        pts += 20
    if e20 > e50:
        pts += 20
    r = rsi(closes)
    if 50 <= r <= 70:
        pts += 20
    avg_vol = sum(vols[-21:-1]) / 20
    if avg_vol > 0 and vols[-1] > 1.5 * avg_vol:
        pts += 20
    if len(closes) >= 25 and price > closes[-25]:
        pts += 20
    return pts, price, r


def main():
    alerts = []
    for pair in PAIRS:
        candles = get_candles(pair)
        time.sleep(0.2)
        if not candles or len(candles) < 60:
            print(f"{pair}: skipped (no data)")
            continue
        pts, price, r = score(candles)
        line = f"{pair}: score {pts} price {price:.6f} rsi {r:.1f}"
        print(line)
        if pts >= ALERT_SCORE and price < 1:
            alerts.append(line)

    print("")
    if alerts:
        print(f"ALERTS ({len(alerts)}):")
        for a in alerts:
            print(a)
    else:
        print("No alerts this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
