"""data.py - Coinbase public candles (no API key needed), with Gate.io fallback.

fetch_candles() returns a DataFrame (UTC DatetimeIndex, columns open/high/low/close/volume,
oldest first) containing CLOSED candles only, or None if the pair does not exist / fails.
Coinbase is tried first; if it has no data for a pair, Gate.io (USDT pair) is tried.

Set GW_DEMO=1 to get synthetic data instead (offline smoke test, no internet).
"""
import hashlib
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = "https://api.exchange.coinbase.com"
GATE = "https://api.gateio.ws/api/v4/spot/candlesticks"
GRAN = {"15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
GATE_GRAN = {"15m": "15m", "1h": "1h", "1d": "1d"}  # Gate has no 6h

# AI + meme + your focus coins. Pairs with no data on either exchange are skipped automatically.
PRODUCTS = [
    # meme
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "POPCAT-USD",
    "MOG-USD", "TURBO-USD", "BRETT-USD", "PNUT-USD",
