"""
walkforward.py — multi-coin historical simulation. No live orders, ever.

STRATEGIES (--strategy):
  sweep    current scanner logic (sweep -> break -> retest), fixed 2R target
  turtle   Donchian breakout trend following (Dennis/Seykota/Tudor Jones 200-MA rule)
  meanrev  buy sharp dips inside an uptrend, quick small target, time stop
           (built for a HIGHER WIN RATE; judge it by avg_R, not win rate alone)

COIN SETS (--pairs):
  auto     up to --max-pairs Coinbase USD pairs (default)
  target   meme coins + XRP + AURORA + ZK
  A,B,C    your own list, e.g. DOGE-USD,XRP-USD

EXAMPLES:
  python walkforward.py --strategy meanrev --pairs target --interval 60 --days 365 --sweep
  python walkforward.py --strategy meanrev --pairs target --interval 60 --days 365
  python walkforward.py --interval 60 --days 365 --fee 0.0025     (stress-test costs)

RULES OF THE GAME:
- Only the TEST half (last 30% of time) is honest. Never tune on it.
- --sweep picks settings on TRAIN only and shows how they did on TEST.
- Costs default to optimistic values; try --fee 0.0025 --slippage 0.002 for a harsher check.
"""

import argparse
import itertools
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

PRIORITY = [
    "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "FET", "RENDER", "TAO", "NEAR",
    "ARB", "INJ", "IMX", "LINK", "AVAX", "ADA", "XRP", "POL", "HBAR", "SUI",
    "APT", "OP", "ATOM", "DOT", "UNI", "AAVE", "LTC", "BCH", "FIL", "ICP",
    "ONDO", "JASMY", "GRT", "AKT", "VIRTUAL", "TRUMP", "POPCAT", "MOG", "TURBO",
    "BTC", "ETH", "SOL",
]

# Meme coins + XRP + Aurora + zkSync. Pairs Coinbase doesn't list are skipped with a note.
TARGET_PAIRS = [
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD",
    "POPCAT-USD", "MOG-USD", "TURBO-USD", "TRUMP-USD",
    "XRP-USD", "AURORA-USD", "ZK-USD",
]

_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.12


# --------------------------------------------------------------------------- data

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
    rest = sorted(pid for pid in live.values() if pid not in ordered)
    return (ordered + rest)[:max_pairs]


def fetch_coinbase(product, days=180, granularity=900):
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
        raise RuntimeError("no candles returned (pair may not be listed)")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (
        df.drop_duplicates("time").sort_values("time")
        .loc[lambda d: d["time"] < end]
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .reset_index(drop=True)
    )


def fetch_kraken(pair, interval):
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


def load_df(pair, args):
    if args.source == "coinbase":
        return fetch_coinbase(pair, days=args.days, granularity=args.interval * 60)
    return fetch_kraken(pair, args.interval)


# ----------------------------------------------------------------- shared helpers

def _finish_trade(trades, balance, entry, qty, exit_price, entry_i, exit_i, times,
                  stop, target, outcome):
    exit_fill = exit_price * (1 - SLIPPAGE_PER_SIDE)
    net = (exit_fill - entry) * qty - (entry * qty + exit_fill * qty) * FEE_PER_SIDE
    balance += net
    trades.append({
        "entry_time": pd.Timestamp(times[entry_i]),
        "exit_time": pd.Timestamp(times[exit_i]),
        "entry": entry, "stop": stop, "target": target, "exit": exit_fill,
        "quantity": qty, "outcome": outcome, "net_pnl": net,
        "balance": balance, "R_multiple": net / RISK_DOLLARS,
    })
    return balance


def _size(entry, stop, balance):
    risk_per_unit = (entry - stop) + entry * 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
    if risk_per_unit <= 0:
        return 0.0
    return min(RISK_DOLLARS / risk_per_unit, balance / (entry * (1 + FEE_PER_SIDE)))


def summarize(t):
    if t is None or t.empty:
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


# ------------------------------------------------------------ strategy 1: sweep

def candidate_mask(data, cfg):
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


def run_backtest(df, cfg):
    data = add_indicators(df, cfg)
    cand = candidate_mask(data, cfg)
    opens = data["open"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    times = data["time"].to_numpy()
    atrs = data["atr"].to_numpy(float)
    balance, trades = START_BALANCE, []
    n, i = len(data), max(cfg.trend_period or cfg.slow, 220)

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
        qty = _size(entry, stop, balance)
        if qty <= 0:
            i += 1
            continue
        target = entry + cfg.reward_risk * (entry - stop)

        exit_price = exit_i = outcome = None
        for j in range(entry_i, n):
            hit_stop, hit_target = lows[j] <= stop, highs[j] >= target
            if hit_stop:
                exit_price = min(stop, opens[j])
                outcome = "STOP_BOTH_TOUCHED" if hit_target else "STOP"
                exit_i = j
                break
            if hit_target:
                exit_price, outcome, exit_i = target, "TARGET", j
                break
        if exit_price is None:
            break
        balance = _finish_trade(trades, balance, entry, qty, exit_price,
                                entry_i, exit_i, times, stop, target, outcome)
        i = exit_i + 1
    return pd.DataFrame(trades)


# ------------------------------------------------------------ strategy 2: turtle

def run_turtle(df, cfg, entry_n=20, exit_n=10):
    data = add_indicators(df, cfg)
    don_hi = data["high"].shift(1).rolling(entry_n).max().to_numpy(float)
    exit_lo = data["low"].shift(1).rolling(exit_n).min().to_numpy(float)
    ema_t = data["ema_trend"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    opens = data["open"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    atrs = data["atr"].to_numpy(float)
    times = data["time"].to_numpy()
    balance, trades = START_BALANCE, []
    n, i = len(data), max(cfg.trend_period or cfg.slow, 220)

    while i < n - 2:
        ok = closes[i] > don_hi[i] and atrs[i] > 0
        if cfg.trend_period:
            ok = ok and closes[i] > ema_t[i]
        if not ok:
            i += 1
            continue
        entry_i = i + 1
        entry = opens[entry_i] * (1 + SLIPPAGE_PER_SIDE)
        stop = entry - cfg.atr_stop_mult * atrs[i]
        if not (0 < stop < entry):
            i += 1
            continue
        qty = _size(entry, stop, balance)
        if qty <= 0:
            i += 1
            continue
        exit_price = exit_i = outcome = None
        for j in range(entry_i, n):
            if lows[j] <= stop:
                exit_price, outcome, exit_i = min(stop, opens[j]), "STOP", j
                break
            if j + 1 < n and closes[j] < exit_lo[j]:
                exit_price, outcome, exit_i = opens[j + 1], "TRAIL_EXIT", j + 1
                break
        if exit_price is None:
            break
        balance = _finish_trade(trades, balance, entry, qty, exit_price,
                                entry_i, exit_i, times, stop, float("nan"), outcome)
        i = exit_i + 1
    return pd.DataFrame(trades)


# --------------------------------------------------------- strategy 3: meanrev

def _rsi(close, n):
    ch = close.diff()
    ag = ch.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = (-ch.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    return rsi.mask((al == 0) & (ag > 0), 100.0)


def prep_meanrev(df, cfg, rsi_len=3):
    """Compute indicator arrays once so a sweep can reuse them."""
    data = add_indicators(df, cfg)
    return {
        "open": data["open"].to_numpy(float),
        "high": data["high"].to_numpy(float),
        "low": data["low"].to_numpy(float),
        "close": data["close"].to_numpy(float),
        "atr": data["atr"].to_numpy(float),
        "ema_t": data["ema_trend"].to_numpy(float),
        "rsi": _rsi(data["close"], rsi_len).to_numpy(float),
        "time": data["time"].to_numpy(),
        "start": max(cfg.trend_period or cfg.slow, 220),
        "use_trend": bool(cfg.trend_period),
    }


def run_meanrev(pr, rsi_max=20.0, target_atr=1.0, stop_atr=2.0, max_hold=24):
    """
    Buy a sharp oversold dip (RSI(3) < rsi_max) while price is above the 200 EMA.
    Enter next open. Stop = stop_atr * ATR below. Target = target_atr * ATR above.
    Exit at the close after max_hold bars if neither is hit.
    A target smaller than the stop raises the win rate, but only avg_R > 0 means it pays.
    """
    with np.errstate(invalid="ignore"):
        cand = (pr["rsi"] < rsi_max) & (pr["atr"] > 0)
        if pr["use_trend"]:
            cand &= pr["close"] > pr["ema_t"]
    opens, highs, lows, closes, atrs, times = (
        pr["open"], pr["high"], pr["low"], pr["close"], pr["atr"], pr["time"])
    balance, trades = START_BALANCE, []
    n, i = len(opens), pr["start"]

    while i < n - 2:
        if not cand[i]:
            i += 1
            continue
        entry_i = i + 1
        entry = opens[entry_i] * (1 + SLIPPAGE_PER_SIDE)
        stop = entry - stop_atr * atrs[i]
        target = entry + target_atr * atrs[i]
        if not (0 < stop < entry):
            i += 1
            continue
        qty = _size(entry, stop, balance)
        if qty <= 0:
            i += 1
            continue
        exit_price = exit_i = outcome = None
        for j in range(entry_i, n):
            hit_stop, hit_target = lows[j] <= stop, highs[j] >= target
            if hit_stop:
                exit_price = min(stop, opens[j])
                outcome = "STOP_BOTH_TOUCHED" if hit_target else "STOP"
                exit_i = j
                break
            if hit_target:
                exit_price, outcome, exit_i = target, "TARGET", j
                break
            if j - entry_i >= max_hold:
                exit_price, outcome, exit_i = closes[j], "TIME", j
                break
        if exit_price is None:
            break
        balance = _finish_trade(trades, balance, entry, qty, exit_price,
                                entry_i, exit_i, times, stop, target, outcome)
        i = exit_i + 1
    return pd.DataFrame(trades)


def run_sweep(preps, cut, min_train=100):
    """Try a grid of settings. Rank on TRAIN only, then show how each did on TEST."""
    grid = list(itertools.product([10, 20, 30], [1.0, 1.5, 2.0], [1.5, 2.5], [12, 24]))
    rows = []
    for rsi_max, tgt, stp, hold in grid:
        frames = []
        for pair, pr in preps.items():
            t = run_meanrev(pr, rsi_max, tgt, stp, hold)
            if not t.empty:
                frames.append(t)
        if not frames:
            continue
        c = pd.concat(frames, ignore_index=True)
        tr, te = summarize(c[c["entry_time"] <= cut]), summarize(c[c["entry_time"] > cut])
        rows.append({
            "rsi<": rsi_max, "tgt_atr": tgt, "stop_atr": stp, "hold": hold,
            "tr_n": tr.get("trades", 0), "tr_win%": tr.get("win_rate", np.nan),
            "tr_avgR": tr.get("avg_R", np.nan),
            "te_n": te.get("trades", 0), "te_win%": te.get("win_rate", np.nan),
            "te_avgR": te.get("avg_R", np.nan), "te_PF": te.get("profit_factor", np.nan),
        })
    if not rows:
        print("Sweep produced no trades.")
        return
    table = pd.DataFrame(rows)
    ok = table[table["tr_n"] >= min_train]
    note = ""
    if ok.empty:
        ok, note = table, f" (no setting reached {min_train} train trades; shown anyway)"
    ok = ok.sort_values("tr_avgR", ascending=False)
    print(f"\nSWEEP: {len(grid)} settings tried, ranked by TRAIN avg_R{note}")
    print(ok.head(10).to_string(index=False))
    best = ok.iloc[0]
    print(f"\nBest on TRAIN: rsi<{best['rsi<']}  target {best['tgt_atr']} ATR  "
          f"stop {best['stop_atr']} ATR  hold {int(best['hold'])} bars")
    print(f"  TRAIN: win {best['tr_win%']}%  avg_R {best['tr_avgR']}  ({int(best['tr_n'])} trades)")
    print(f"  TEST : win {best['te_win%']}%  avg_R {best['te_avgR']}  PF {best['te_PF']}  "
          f"({int(best['te_n'])} trades)  <-- the honest number")
    print("\nThe best of many tries always looks better on TRAIN than it really is. "
          "Only trust it if TEST avg_R is also above 0 with a decent trade count.")
    print("Re-run without --sweep, passing the winning values, to see the per-coin table:")
    print(f"  --rsi-max {best['rsi<']} --target-atr {best['tgt_atr']} "
          f"--stop-atr {best['stop_atr']} --max-hold {int(best['hold'])}")


# ---------------------------------------------------------------------- reporting

def report(combined):
    combined = combined.sort_values("entry_time")
    print("\nPER-PAIR (sorted by net R):")
    rows = [{"pair": pair, **summarize(g)} for pair, g in combined.groupby("pair")]
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


def download_all(pairs, args):
    """Parallel download. Returns {pair: df}. Unlisted or failed pairs are reported."""
    dfs, failed = {}, []

    def job(pair):
        df = load_df(pair, args)
        if len(df) < 300:
            raise RuntimeError(f"only {len(df)} candles")
        if args.max_price and float(df["close"].iloc[-1]) >= args.max_price:
            raise RuntimeError(f"price >= ${args.max_price}")
        return df

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(job, p): p for p in pairs}
        for done, fut in enumerate(as_completed(futures), 1):
            pair = futures[fut]
            try:
                dfs[pair] = fut.result()
                print(f"[{done}/{len(pairs)}] {pair}: {len(dfs[pair])} candles", flush=True)
            except Exception as exc:
                failed.append(pair)
                print(f"[{done}/{len(pairs)}] {pair}: SKIPPED ({exc})", flush=True)
    if failed:
        print(f"\nSkipped pairs: {', '.join(sorted(failed))}")
    return dfs


def main():
    global FEE_PER_SIDE, SLIPPAGE_PER_SIDE
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["coinbase", "kraken"], default="coinbase")
    p.add_argument("--pairs", default="auto",
                   help="'auto', 'target' (memes+XRP+AURORA+ZK), or a comma-separated list.")
    p.add_argument("--max-pairs", type=int, default=60)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--interval", type=int, default=15, help="Candle minutes: 15 or 60.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-price", type=float, default=0.0,
                   help="Only test coins priced below this (0 = no limit).")
    p.add_argument("--strategy", choices=["sweep", "turtle", "meanrev"], default="sweep")
    p.add_argument("--donchian-entry", type=int, default=20)
    p.add_argument("--donchian-exit", type=int, default=10)
    p.add_argument("--rsi-max", type=float, default=20.0, help="meanrev: buy when RSI(3) below this.")
    p.add_argument("--target-atr", type=float, default=1.0, help="meanrev: target in ATRs.")
    p.add_argument("--stop-atr", type=float, default=2.0, help="meanrev: stop in ATRs.")
    p.add_argument("--max-hold", type=int, default=24, help="meanrev: max bars to hold.")
    p.add_argument("--sweep", action="store_true", help="meanrev: try many settings (train/test).")
    p.add_argument("--fee", type=float, default=None, help="Fee per side, e.g. 0.0025 = 0.25%%.")
    p.add_argument("--slippage", type=float, default=None, help="Slippage per side, e.g. 0.002.")
    p.add_argument("--csv", default="walkforward_trades.csv")
    args = p.parse_args()

    if args.fee is not None:
        FEE_PER_SIDE = args.fee
    if args.slippage is not None:
        SLIPPAGE_PER_SIDE = args.slippage

    if args.pairs == "auto":
        if args.source != "coinbase":
            p.error("--pairs auto needs --source coinbase; pass --pairs for kraken")
        pairs = discover_pairs(args.max_pairs)
    elif args.pairs == "target":
        pairs = list(TARGET_PAIRS)
    else:
        pairs = [x.strip() for x in args.pairs.split(",") if x.strip()]

    print(f"Testing {len(pairs)} pairs | {args.interval}m | {args.days} days | {args.source} | "
          f"strategy={args.strategy} | fee {FEE_PER_SIDE:.4f} + slip {SLIPPAGE_PER_SIDE:.4f} per side")
    cfg = SignalConfig()
    t_start = time.time()
    dfs = download_all(pairs, args)
    if not dfs:
        print("No data downloaded.")
        return

    if args.strategy == "meanrev":
        preps = {pair: prep_meanrev(df, cfg) for pair, df in dfs.items()}
        if args.sweep:
            t0 = min(df["time"].iloc[0] for df in dfs.values())
            t1 = max(df["time"].iloc[-1] for df in dfs.values())
            run_sweep(preps, t0 + TRAIN_FRACTION * (t1 - t0))
            print(f"\nFinished in {time.time() - t_start:.0f}s.")
            return
        frames = []
        for pair, pr in preps.items():
            t = run_meanrev(pr, args.rsi_max, args.target_atr, args.stop_atr, args.max_hold)
            if not t.empty:
                t.insert(0, "pair", pair)
                frames.append(t)
    else:
        frames = []
        for pair, df in dfs.items():
            t = (run_turtle(df, cfg, args.donchian_entry, args.donchian_exit)
                 if args.strategy == "turtle" else run_backtest(df, cfg))
            if not t.empty:
                t.insert(0, "pair", pair)
                frames.append(t)

    print(f"\nFinished in {time.time() - t_start:.0f}s.")
    if not frames:
        print("No completed trades.")
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.csv, index=False)
    print(f"Saved {len(combined)} trades to {args.csv}")
    report(combined)


if __name__ == "__main__":
    main()
