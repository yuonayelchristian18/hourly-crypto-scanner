"""paper.py - hourly PAPER trader (no real money, no exchange keys).

Each run (best: right after each hourly candle closes):
  1. fetch closed candles for every pair
  2. check exits on open paper positions (stop, 2R target, Gann-flip, opposite signal)
  3. open new positions from fresh signals (long only), best score first

  python paper.py            # run once
  python paper.py --report   # print results so far

State: paper_state.json (equity + open positions). Closed trades: paper_trades.csv.
Sizing: $2 risk per trade on a $200 account, max 3 open positions, each capped at 1/3 of
equity (so a wide stop can mean less than $2 of real risk). Costs 0.25%/side (fee+slippage).
Entry is modelled at the signal candle's close (a live fill would be the next open).
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import gwvsa
from data import PRODUCTS, fetch_candles
from gwvsa import position_size
from signals import analyze, make_params

STATE_FILE = "paper_state.json"
LOG_FILE = "paper_trades.csv"
START_EQUITY = 200.0
RISK_USD = 2.0
MAX_OPEN = 3
COST = 0.0025


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"equity": START_EQUITY, "open": [], "last_signal": {}}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def check_exit(df, pos, p):
    """Walk the candles AFTER the signal candle. Same rules as gwvsa.backtest (stop first)."""
    sigs, A, _ = gwvsa._run(df, p)
    o, h, l, c, trend = A["o"], A["h"], A["l"], A["c"], A["trend"]
    opp = {s.idx for s in sigs if s.direction == "short"}
    start = df.index.searchsorted(pd.Timestamp(pos["signal_time"]), side="right")
    for j in range(start, len(df)):
        t = str(df.index[j])
        if o[j] <= pos["stop"]:
            return dict(price=float(o[j]), reason="stop", time=t)
        if l[j] <= pos["stop"]:
            return dict(price=pos["stop"], reason="stop", time=t)
        if h[j] >= pos["target"]:
            return dict(price=pos["target"], reason="target", time=t)
        if j > start and trend[j] == -1 and trend[j - 1] != -1:
            return dict(price=float(c[j]), reason="gann_flip", time=t)
        if j > start and j in opp:
            return dict(price=float(c[j]), reason="opposite", time=t)
    return None


def log_trade(st, pos, ex):
    gross = pos["qty"] * (ex["price"] - pos["entry"])
    fees = COST * pos["qty"] * (pos["entry"] + ex["price"])
    pnl = gross - fees
    st["equity"] = round(st["equity"] + pnl, 4)
    row = dict(product=pos["product"], pattern=pos["pattern"], entry_type=pos["entry_type"],
               final_score=pos["final_score"], signal_time=pos["signal_time"], exit_time=ex["time"],
               entry=pos["entry"], stop=pos["stop"], exit=ex["price"], reason=ex["reason"],
               qty=pos["qty"], pnl_usd=round(pnl, 4), R=round(pnl / pos["risk_usd"], 3),
               equity_after=st["equity"])
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    print(f"  CLOSED {pos['product']} {ex['reason']} pnl ${pnl:+.2f} ({row['R']:+.2f}R) equity ${st['equity']:.2f}")


def report():
    st = load_state()
    print(f"equity ${st['equity']:.2f} (start ${START_EQUITY:.0f}) | open positions: {len(st['open'])}")
    for pos in st["open"]:
        print(f"  OPEN {pos['product']} {pos['pattern']} entry {pos['entry']:.8g} stop {pos['stop']:.8g} target {pos['target']:.8g}")
    if not os.path.exists(LOG_FILE):
        print("no closed trades yet"); return
    t = pd.read_csv(LOG_FILE)
    r = t["R"]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    print(f"closed {len(t)} | win rate {(r > 0).mean() * 100:.1f}% | net {r.sum():+.2f}R | "
          f"PF {gp / gl if gl > 0 else float('inf'):.2f} | net ${t['pnl_usd'].sum():+.2f}")
    print(t.groupby("pattern")["R"].agg(["count", "sum"]).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--mode", default="original", choices=["original", "more", "max"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(); return

    p = make_params(args.mode)
    st = load_state()
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | equity ${st['equity']:.2f} | open {len(st['open'])}")

    dfs = {}
    for pr in PRODUCTS:
        df = fetch_candles(pr, args.interval, args.days)
        if df is not None and len(df) >= 150:
            dfs[pr] = df
    print(f"data for {len(dfs)}/{len(PRODUCTS)} pairs")

    # 1) exits
    still_open = []
    for pos in st["open"]:
        df = dfs.get(pos["product"])
        ex = check_exit(df, pos, p) if df is not None else None
        if ex:
            log_trade(st, pos, ex)
        else:
            still_open.append(pos)
    st["open"] = still_open

    # 2) entries (best score first)
    cands = []
    held = {x["product"] for x in st["open"]}
    for pr, df in dfs.items():
        if pr in held:
            continue
        s = analyze(df, p)["signal"]
        if s is not None and st["last_signal"].get(pr) != str(s.time):
            cands.append((pr, s))
    cands.sort(key=lambda x: (-x[1].final_score, -x[1].setup_score))
    for pr, s in cands:
        if len(st["open"]) >= MAX_OPEN:
            break
        qty = position_size(s.entry_ref, s.stop, RISK_USD, COST, equity=st["equity"], exposure_cap=1.0 / MAX_OPEN)
        st["last_signal"][pr] = str(s.time)
        if qty <= 0:
            print(f"  skip {pr}: size too small"); continue
        risk_usd = qty * (abs(s.entry_ref - s.stop) + s.entry_ref * 2 * COST)
        st["open"].append(dict(product=pr, pattern=s.pattern, entry_type=s.entry_type,
                               final_score=s.final_score, signal_time=str(s.time),
                               entry=s.entry_ref, stop=s.stop, target=s.target_2r,
                               qty=qty, risk_usd=round(risk_usd, 4)))
        print(f"  OPEN {pr} {s.pattern} [{s.entry_type}] score {s.scanner_score} entry {s.entry_ref:.8g} "
              f"stop {s.stop:.8g} target {s.target_2r:.8g} qty {qty:.6g} risk ${risk_usd:.2f}")
    save_state(st)
    print(f"open positions now: {len(st['open'])}")


if __name__ == "__main__":
    main()
