"""
walkforward.py — multi-coin historical simulation of the scanner's signal logic.

Running `python walkforward.py` with NO options does the big test:
15m candles, 180 days, up to 60 Coinbase USD pairs (auto-discovered),
downloaded in parallel, with a train/test split at the end.

Uses closed candles, next-candle entries, gap-aware stops, fees, slippage
and a $200 reference balance. No live orders are ever placed.

Notes:
- Single-timeframe test: it does NOT include the scanner's 1h/4h/5m filters.
- Keep settings fixed after you look at the TEST result, or the split means nothing.
- Kraken source is limited to the most recent 720 candles (API limit).
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from signals import SignalConfig, add_indicators, signal_at, BUY, _sequential_setup

KRAKEN_URL = "https://api.kraken.com/0/public"
COINBASE_URL = "https://api.exchange.coinbase.com"

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0015
START_BALANCE = 200.0
RISK_DOLLARS = 2.0
TRAIN_FRACTION = 0.7

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDS",
           "BUSD", "USD1", "USDD", "EURC", "GUSD", "PAX", "WBTC", "CBBTC", "WETH"}

# Tried first (AI / meme / liquid alts). Anything Coinbase doesn't list is skipped.
PRIORITY = [
    "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "FET", "RENDER", "TAO", "NEAR",
    "ARB", "INJ", "IMX", "LINK", "AVAX", "ADA", "XRP", "POL", "HBAR", "SUI",
    "APT", "OP", "ATOM", "DOT", "UNI", "AAVE", "LTC", "BCH", "FIL", "ICP",
    "ONDO", "JASMY", "GRT", "AKT", "VIRTUAL", "TRUMP", "POPCAT", "MOG", "TURBO",
    "BTC", "ETH", "SOL",
]

_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.12  # seconds between Coinbase calls across all threads


def get_json(url, params=None, retries=6):
    """GET with global rate limiting and retry/backoff on 429/5xx/network errors."""
    delay = 1.0
    for attempt in range(retries):
        with _rate_lock:
            wait = MIN_INTERVAL - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20)


def discover_pairs(max_pairs):
    """All online Coinbase USD pairs, priority coins first, stablecoins excluded."""
    products = get_json(f"{COINBASE_URL}/products")
    live = {}
    for p in products:
        if p.get("quote_currency") != "USD":
            continue
        if p.get("trading_disabled") or p.get("status", "online") != "online":
            continue
        base = str(p.get("base_currency", "")).upper()
        if not base or base in STABLES:
            continue
        live[base] = p["id"]

    ordered = [live[b] for b in PRIORITY if b in live]
    rest = sorted(pid for b, pid in live.items() if pid not in ordered)
    return (ordered + rest)[:max_pairs]


def fetch_coinbase(product, days=180, granularity=900):
    """Paginated Coinbase history, 300 candles max per request."""
    end = pd.Timestamp.now(tz="UTC").floor(f"{granularity // 60}min")
    cur = end - pd.Timedelta(days=days)
    step = pd.Timedelta(seconds=granularity * 290)
    frames = []

    while cur < end:
        nxt = min(cur + step, end)
        rows = get_json(
            f"{COINBASE_URL}/products/{product}/candles",
            params={"granularity": granularity,
                    "start": cur.isoformat(), "end": nxt.isoformat()},
        )
        if rows:
            frames.append(pd.DataFrame(
                rows, columns=["time", "low", "high", "open", "close", "volume"]))
        cur = nxt

    if not frames:
        raise RuntimeError("no candles returned")

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.drop_duplicates("time").sort_values("time")
        .loc[lambda d: d["time"] < end]  # drop still-forming candle
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .reset_index(drop=True)
    )


def fetch_kraken(pair, interval):
    """Kraken: at most the 720 most recent candles."""
    body = get_json(f"{KRAKEN_URL}/OHLC", params={"pair": pair, "interval": interval})
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body["result"]
    key = next(k for k in result if k != "last")
    df = pd.DataFrame(result[key][:-1],
                      columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def candidate_mask(data, cfg):
    """
    Cheap vectorized prefilter: only these candles can possibly be BUY signals,
    so the slow sequential check runs on a tiny fraction of candles.
    """
    m = (
        (data["ema_fast"] > data["ema_slow"])
        & data["rsi"].between(cfg.rsi_low, cfg.rsi_high)
        & (data["vol_ratio"] >= cfg.min_vol_ratio)
        & (data["atr"] > 0)
        & ((data["high"] - data["low"]) <= cfg.max_signal_bar_atr * data["atr"])
    )
    if cfg.trend_period:
        m &= data["close"] > data["ema_trend"]
    return m.fillna(False).to_numpy()


def run_backtest(df, cfg, starting_balance=START_BALANCE):
    data = add_indicators(df, cfg)
    cand = candidate_mask(data, cfg)

    opens = data["open"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    times = data["time"].to_numpy()
    atrs = data["atr"].to_numpy(float)

    balance = float(starting_balance)
    trades = []
    start = max(cfg.trend_period or cfg.slow, 220)
    n = len(data)
    i = start

    while i < n - 2:
        if not cand[i] or signal_at(data, i, cfg) != BUY:
            i += 1
            continue

        entry_i = i + 1
        entry = opens[entry_i] * (1 + SLIPPAGE_PER_SIDE)

        setup = _sequential_setup(data, i, cfg)
        if setup is None:
            i += 1
            continue

        atr = atrs[i]
        stop = min(float(setup["sweep_low"]), entry - cfg.atr_stop_mult * atr) \
            - cfg.stop_buffer_atr * atr
        if not (0 < stop < entry):
            i += 1
            continue

        risk_per_unit = (entry - stop) + entry * 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
        if risk_per_unit <= 0:
            i += 1
            continue

        qty = min(RISK_DOLLARS / risk_per_unit,
                  balance / (entry * (1 + FEE_PER_SIDE)))
        if qty <= 0:
            i += 1
            continue

        target = entry + cfg.reward_risk * (entry - stop)

        exit_price = exit_i = outcome = None
        for j in range(entry_i, n):
            hit_stop = lows[j] <= stop
            hit_target = highs[j] >= target
            if hit_stop:
                # Stop wins ties; gaps below the stop fill at the open.
                exit_price = min(stop, opens[j])
                outcome = "STOP_BOTH_TOUCHED" if hit_target else "STOP"
                exit_i = j
                break
            if hit_target:
                exit_price, outcome, exit_i = target, "TARGET", j
                break

        if exit_price is None:
            break  # trade still open at end of data; not counted

        exit_fill = exit_price * (1 - SLIPPAGE_PER_SIDE)
        gross = (exit_fill - entry) * qty
        fees = (entry * qty + exit_fill * qty) * FEE_PER_SIDE
        net = gross - fees
        balance += net

        trades.append({
            "entry_time": pd.Timestamp(times[entry_i]),
            "exit_time": pd.Timestamp(times[exit_i]),
            "entry": entry, "stop": stop, "target": target, "exit": exit_fill,
            "quantity": qty, "outcome": outcome, "net_pnl": net,
            "balance": balance, "R_multiple": net / RISK_DOLLARS,
        })
        i = exit_i + 1  # no overlapping trades on the same pair

    return pd.DataFrame(trades)


def summarize(t):
    if t.empty:
        return {"trades": 0}
    w = t.loc[t["net_pnl"] > 0, "net_pnl"].sum()
    l = abs(t.loc[t["net_pnl"] < 0, "net_pnl"].sum())
    return {
        "trades": int(len(t)),
        "win_rate": round(float((t["net_pnl"] > 0).mean() * 100), 1),
        "net_R": round(float(t["R_multiple"].sum()), 2),
        "avg_R": round(float(t["R_multiple"].mean()), 3),
        "profit_factor": round(float(w / l), 2) if l > 0 else float("inf"),
    }


def process_pair(pair, args, cfg):
    """Download + backtest one pair. Returns (pair, trades_df, note)."""
    if args.source == "coinbase":
        df = fetch_coinbase(pair, days=args.days, granularity=args.interval * 60)
    else:
        df = fetch_kraken(pair, args.interval)

    if len(df) < 300:
        return pair, pd.DataFrame(), f"skipped (only {len(df)} candles)"
    if args.max_price and float(df["close"].iloc[-1]) >= args.max_price:
        return pair, pd.DataFrame(), f"skipped (price >= ${args.max_price})"

    trades = run_backtest(df, cfg)
    if not trades.empty:
        trades.insert(0, "pair", pair)
    return pair, trades, f"{len(df)} candles"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["coinbase", "kraken"], default="coinbase")
    p.add_argument("--pairs", default="auto",
                   help="Comma-separated list, or 'auto' to discover many USD pairs.")
    p.add_argument("--max-pairs", type=int, default=60, help="Cap for auto discovery.")
    p.add_argument("--days", type=int, default=180, help="History length (coinbase only).")
    p.add_argument("--interval", type=int, default=15, help="Candle minutes: 15 or 60.")
    p.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    p.add_argument("--max-price", type=float, default=0.0,
                   help="Only test coins priced below this (0 = no limit). Use 1 to mimic the scanner.")
    p.add_argument("--csv", default="walkforward_trades.csv")
    args = p.parse_args()

    if args.pairs == "auto":
        if args.source != "coinbase":
            p.error("--pairs auto needs --source coinbase; pass --pairs for kraken")
        pairs = discover_pairs(args.max_pairs)
    else:
        pairs = [x.strip() for x in args.pairs.split(",") if x.strip()]

    print(f"Testing {len(pairs)} pairs | {args.interval}m | {args.days} days | {args.source}")
    cfg = SignalConfig()
    all_trades, failed = [], []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_pair, pr, args, cfg): pr for pr in pairs}
        for done, fut in enumerate(as_completed(futures), 1):
            pr = futures[fut]
            try:
                pair, trades, note = fut.result()
                print(f"[{done}/{len(pairs)}] {pair}: {len(trades)} trades ({note})", flush=True)
                if not trades.empty:
                    all_trades.append(trades)
            except Exception as exc:
                failed.append(pr)
                print(f"[{done}/{len(pairs)}] {pr}: ERROR {exc}", flush=True)

    print(f"\nFinished in {time.time() - t_start:.0f}s. Failed pairs: {len(failed)}")

    if not all_trades:
        print("No completed trades.")
        return

    combined = pd.concat(all_trades, ignore_index=True).sort_values("entry_time")
    combined.to_csv(args.csv, index=False)
    print(f"Saved {len(combined)} trades to {args.csv}")

    print("\nPER-PAIR (sorted by net R):")
    rows = []
    for pair, g in combined.groupby("pair"):
        s = summarize(g)
        rows.append({"pair": pair, **s})
    table = pd.DataFrame(rows).sort_values("net_R", ascending=False)
    print(table.to_string(index=False))

    total_r = combined["R_multiple"].sum()
    best = table.iloc[0]
    if total_r > 0 and best["net_R"] > 0.5 * total_r:
        print(f"\nWARNING: {best['pair']} supplies over half of all profit; "
              "the edge may rest on one coin.")

    t0, t1 = combined["entry_time"].min(), combined["entry_time"].max()
    cut = t0 + TRAIN_FRACTION * (t1 - t0)
    print("\nALL TRADES:", summarize(combined))
    print("TRAIN (first 70%):", summarize(combined[combined["entry_time"] <= cut]))
    print("TEST  (last 30%): ", summarize(combined[combined["entry_time"] > cut]))

    if len(combined) < 100:
        print("\nNote: under 100 trades; treat these results as inconclusive.")


if __name__ == "__main__":
    main()
