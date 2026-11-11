"""scanner.py - scan the coin list on closed candles, print a table, log/alert signals."""
import argparse
import os
import sys
import time

import pandas as pd
import requests

from data import PRODUCTS, fetch_candles
from signals import analyze, make_params

ALERTS = "alerts.csv"


def load_seen():
    try:
        old = pd.read_csv(ALERTS)
        return set(zip(old["product"], old["time"].astype(str)))
    except Exception:
        return set()


def get_candles(pr, interval, days, tries=3):
    for i in range(tries):
        try:
            return fetch_candles(pr, interval, days)
        except Exception as e:
            print(f"{pr}: fetch error ({e}) try {i + 1}/{tries}")
            time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--mode", default="original", choices=["original", "more", "max"])
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--products", nargs="*", default=None)
    args = ap.parse_args()

    p = make_params(args.mode)
    prods = args.products or PRODUCTS
    seen = load_seen()

    scanned, errors, alerts = 0, 0, []
    print(f"{'pair':<14}{'close':>14}  {'trend':<6} signal")
    for pr in prods:
        try:
            df = get_candles(pr, args.interval, args.days)
            if df is None or len(df) < 150:
                print(f"{pr:
