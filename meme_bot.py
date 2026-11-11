"""
meme_bot.py - PAPER TRADING ONLY. No real orders, no API keys.

Run once an hour. Each run:
  1. Replays 5-minute candles since each open paper trade was bought and closes it on
     a tight trailing take-profit, a hard stop, or a max-hold timeout.
  2. Scans the meme coins and opens paper buys on the top BUY_PER_RUN by score.
Results go to meme_trades.csv, open trades to meme_state.json.
"""
import csv
import json
import os
import time

import pandas as pd
import requests

API = "https://api.exchange.coinbase.com"
COINS = ["DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD",
         "PENGU-USD", "TRUMP-USD", "FARTCOIN-USD", "MOODENG-USD", "POPCAT-USD"]

BUY_PER_RUN = 2        # forced: buys the top 2 every run even if setups are weak
ARM_PCT = 1.0          # trailing take-profit turns on after price is up this much
TRAIL_PCT = 0.4        # then sells when price falls this much from its peak
STOP_PCT = 1.5         # hard stop below entry (before the trail is armed)
MAX_HOLD_HOURS = 6     # sell at market if still open after this long
FEE_PCT = 0.25         # per side - SET THIS TO YOUR REAL EXCHANGE FEE
SLIP_PCT = 0.10        # assumed slippage per side

STATE_FILE = "meme_state.json"
LOG_FILE = "meme_trades.csv"


def get_candles(pid):
    r = requests.get(f"{API}/products/{pid}/candles",
                     params={"granularity": 300}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["time", "low", "high", "open", "close", "volume"])
    return df.sort_values("time").reset_index(drop=True)


def score_coin(df):
    """Higher = stronger recent momentum with rising volume, above its 4h average."""
    if len(df) < 70:
        return None
    close = df["close"]
    ret_1h = (close.iloc[-1] / close.iloc[-13] - 1) * 100
    recent_vol = df["volume"].iloc[-12:].mean()
    base_vol = df["volume"].iloc[-60:-12].mean()
    vol_ratio = recent_vol / base_vol if base_vol > 0 else 0
    ema = close.ewm(span=48, adjust=False).mean().iloc[-1]
    trend_bonus = 1.0 if close.iloc[-1] > ema else 0.0
    return 0.6 * ret_1h + 0.4 * min(vol_ratio, 3) + trend_bonus


def check_exit(pos, df):
    """Replay candles after entry. Returns (exit_price, reason, peak) or (None, None, peak)."""
    entry, peak = pos["entry"], pos["peak"]
    for c in df[df["time"] >= pos["time"]].itertuples():
        armed = peak >= entry * (1 + ARM_PCT / 100)
        # check exits using the peak from BEFORE this candle (conservative)
        if armed and c.low <= peak * (1 - TRAIL_PCT / 100):
            return peak * (1 - TRAIL_PCT / 100), "trail", peak
        if not armed and c.low <= entry * (1 - STOP_PCT / 100):
            return entry * (1 - STOP_PCT / 100), "stop", peak
        peak = max(peak, c.high)
    if time.time() - pos["time"] > MAX_HOLD_HOURS * 3600:
        return df["close"].iloc[-1], "timeout", peak
    return None, None, peak


def net_pct(entry, exit_price):
    return ((exit_price * (1 - SLIP_PCT / 100)) / entry - 1) * 100 - 2 * FEE_PCT


def log_trade(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["closed_at", "pair", "entry", "exit", "reason", "net_pct"])
        w.writerow(row)


def summary():
    if not os.path.exists(LOG_FILE):
        print("No closed trades yet.")
        return
    t = pd.read_csv(LOG_FILE)
    wins = t[t["net_pct"] > 0]["net_pct"].sum()
    losses = -t[t["net_pct"] <= 0]["net_pct"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    print(f"Closed trades: {len(t)} | win rate: {(t['net_pct'] > 0).mean() * 100:.1f}% "
          f"| net: {t['net_pct'].sum():.2f}% | profit factor: {pf:.2f}")


def main():
    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {"open": []}
    still_open = []

    for pos in state["open"]:
        try:
            df = get_candles(pos["pair"])
        except Exception as e:
            print(f"{pos['pair']}: fetch failed ({e}), keeping open")
            still_open.append(pos)
            continue
        exit_price, reason, peak = check_exit(pos, df)
        if exit_price is None:
            pos["peak"] = peak
            still_open.append(pos)
        else:
            pnl = net_pct(pos["entry"], exit_price)
            log_trade([time.strftime("%Y-%m-%d %H:%M:%S"), pos["pair"],
                       pos["entry"], round(exit_price, 10), reason, round(pnl, 3)])
            print(f"CLOSED {pos['pair']} ({reason}) net {pnl:+.2f}%")
        time.sleep(0.3)

    held = {p["pair"] for p in still_open}
    scored = []
    for pid in COINS:
        if pid in held:
            continue
        try:
            df = get_candles(pid)
            s = score_coin(df)
            if s is not None:
                scored.append((s, pid, float(df["close"].iloc[-1])))
        except Exception as e:
            print(f"{pid}: skipped ({e})")
        time.sleep(0.3)

    scored.sort(reverse=True)
    for s, pid, price in scored[:BUY_PER_RUN]:
        entry = price * (1 + SLIP_PCT / 100)
        still_open.append({"pair": pid, "entry": entry, "peak": entry, "time": time.time()})
        print(f"PAPER BUY {pid} at {price} (score {s:.2f})")

    state["open"] = still_open
    json.dump(state, open(STATE_FILE, "w"), indent=2)
    summary()


if __name__ == "__main__":
    main()
