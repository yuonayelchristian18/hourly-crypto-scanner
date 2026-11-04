"""Scan AI and meme coins under $1 using Kraken data and alert on strong setups.

Needs signals.py in the same folder.

Usage:
    python scanner.py                       # AI + meme coins under $1, alert at score 70+
    python scanner.py --alert 80            # stricter alerts
    python scanner.py --categories artificial-intelligence,meme-token,<other-id>
    python scanner.py --no-filters --no-stop   # plain strategy, for comparison

The setup score (0-100) ranks how strong a coin's current chart setup looks.
It is NOT a probability that the price will rise.
"""

import argparse
import csv
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from signals import (
    BUY, SELL, SignalConfig, add_indicators, latest_signal, setup_score, signal_at,
)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
KRAKEN_URL = "https://api.kraken.com/0/public"

DEFAULT_CATEGORIES = "artificial-intelligence,meme-token"
LABELS = {"artificial-intelligence": "AI", "meme-token": "MEME"}

STARTING_BALANCE = 200.00
FEE_RATE = 0.001       # per side
SLIPPAGE = 0.0015      # per side; higher than large caps because these trade thinner
MAX_CANDLES = 720      # Kraken returns at most 720 candles per request
PRICE_TOLERANCE = 0.15  # max gap between CoinGecko and Kraken price (ticker-clash guard)

KRAKEN_PAIRS = {}

STABLES = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD",
    "USDS", "BUSD", "USD1", "USDD",
}

COLUMNS = [
    "time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "buy_volume", "buy_quote_volume", "ignore",
]


def make_session():
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_universe(session, categories, per_category, max_price, min_volume, min_mcap):
    """CoinGecko coins in the given categories, filtered for price and liquidity."""
    coins = {}
    for category in categories:
        response = session.get(
            COINGECKO_URL,
            params={
                "vs_currency": "usd",
                "category": category,
                "order": "market_cap_desc",
                "per_page": per_category,
                "page": 1,
            },
            timeout=20,
        )
        response.raise_for_status()
        label = LABELS.get(category, category)
        for coin in response.json():
            entry = coins.setdefault(coin["id"], {**coin, "labels": []})
            if label not in entry["labels"]:
                entry["labels"].append(label)
        time.sleep(2)  # CoinGecko's free tier is strict

    kept = []
    for coin in coins.values():
        price = coin.get("current_price")
        if price is None or price >= max_price:
            continue
        if (coin.get("total_volume") or 0) < min_volume:
            continue
        if (coin.get("market_cap") or 0) < min_mcap:
            continue
        kept.append(coin)

    # Biggest market cap first, so a ticker clash keeps the larger coin.
    kept.sort(key=lambda c: c.get("market_cap") or 0, reverse=True)
    return kept


def pick_binance(session):
    """Kept under the old name so other scripts still work.

    Returns (base_url, {our SYMBOL: kraken pair name}) for USD spot pairs.
    """
    response = session.get(f"{KRAKEN_URL}/AssetPairs", timeout=20)
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")

    pairs = {}
    for info in body["result"].values():
        if info.get("quote") not in ("ZUSD", "USD"):
            continue
        if info.get("status", "online") != "online":
            continue
        wsname = info.get("wsname", "")


        