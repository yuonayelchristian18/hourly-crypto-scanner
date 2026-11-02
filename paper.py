"""Paper trading: follow the scanner's signals with fake money and keep a log.

Run it once an hour (by hand or with cron). Each run:
  1. checks open paper positions against their stops and sell signals,
  2. opens new positions for coins with a BUY signal and a high setup score,
  3. saves state to paper_state.json and every event to paper_log.csv.

Needs scanner.py and signals.py in the same folder.

Usage:
    python paper.py
    python paper.py --alert 75 --risk 0.5 --max-positions 3
    python paper.py --reset            # start over with a fresh $200
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import scanner as sc
from signals import BUY, SELL, SignalConfig, add_indicators, latest_signal, setup_score, signal_at

STATE_FILE = "paper_state.json"
LOG_FILE = "paper_log.csv"
LOG_FIELDS = ["time", "event", "symbol", "price", "size_usd", "pnl_usd", "reason", "score"]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": sc.STARTING_BALANCE, "open": [], "closed": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log(**fields):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"time": now(), **fields})


def manage_open(session, base, state, cfg):
    """Close positions that hit their stop or got a sell signal. Returns last prices."""
    last_prices = {}
    for pos in list(state["open"]):
        df = add_indicators(sc.get_candles(session, base, pos["symbol"], 200), cfg)
        last_prices[pos["symbol"]] = float(df.iloc[-1]["close"])

        exit_price = reason = None
        newer = df[df["time"].astype("int64") > pos["entry_ms"]]
        for _, row in newer.iterrows():
            if float(row["low"]) <= pos["stop"]:
                exit_price, reason = min(float(row["open"]), pos["stop"]), "stop hit"
                break
        if exit_price is None and signal_at(df, len(df) - 1, cfg) == SELL:
            exit_price, reason = last_prices[pos["symbol"]], "sell signal"

        if exit_price is not None:
            proceeds = (pos["qty"] * exit_price
                        * (1 - sc.SLIPPAGE) * (1 - sc.FEE_RATE))
            pnl = proceeds - pos["cost"]
            state["cash"] += proceeds
            state["open"].remove(pos)
            state["closed"].append({"symbol": pos["symbol"], "pnl": pnl,
                                    "reason": reason, "closed": now()})
            log(event="SELL", symbol=pos["symbol"], price=f"{exit_price:.6g}",
                size_usd=f"{proceeds:.2f}", pnl_usd=f"{pnl:.2f}", reason=reason)
            print(f"CLOSED {pos['symbol']} at {exit_price:.6g} ({reason}), P/L ${pnl:+.2f}")
        time.sleep(0.15)
    return last_prices


def find_entries(session, base, universe, binance_symbols, cfg, args, held):
    candidates, seen = [], set()
    for coin in universe:
        name = coin["symbol"].upper()
        symbol = name + "USDT"
        if (name in sc.STABLES or symbol in seen or symbol in held
                or symbol not in binance_symbols):
            continue
        seen.add(symbol)
        try:
            df = add_indicators(sc.get_candles(session, base, symbol, 400), cfg)
            if len(df) < 200:
                continue
            live = latest_signal(df, cfg)
            if abs(live["price"] / coin["current_price"] - 1) > sc.PRICE_TOLERANCE:
                continue
            scored = setup_score(df, cfg)
            if live["signal"] == BUY and scored["score"] >= args.alert and live["suggested_stop"]:
                candidates.append({
                    "symbol": symbol, "price": live["price"],
                    "stop": live["suggested_stop"], "score": scored["score"],
                    "entry_ms": int(df.iloc[-1]["close_time"]),
                })
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
        time.sleep(0.15)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def equity(state, last_prices):
    total = state["cash"]
    for pos in state["open"]:
        price = last_prices.get(pos["symbol"], pos["entry_price"])
        total += pos["qty"] * price
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--categories", default=sc.DEFAULT_CATEGORIES)
    parser.add_argument("--alert", type=int, default=70)
    parser.add_argument("--risk", type=float, default=1.0,
                        help="percent of current equity risked per trade")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for path in (STATE_FILE, LOG_FILE):
            if os.path.exists(path):
                os.remove(path)
        print("Paper account reset.")

    cfg = SignalConfig()
    state = load_state()
    session = sc.make_session()
    base, binance_symbols = sc.pick_binance(session)

    last_prices = manage_open(session, base, state, cfg)

    if len(state["open"]) < args.max_positions:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        universe = sc.get_universe(session, categories, 100, 1.0, 1_000_000, 20_000_000)
        held = {p["symbol"] for p in state["open"]}
        for cand in find_entries(session, base, universe, binance_symbols, cfg, args, held):
            if len(state["open"]) >= args.max_positions:
                break
            total = equity(state, last_prices)
            risk_dollars = total * args.risk / 100
            size = min(risk_dollars / ((cand["price"] - cand["stop"]) / cand["price"]),
                       state["cash"])
            if size < 10:
                continue
            price = cand["price"] * (1 + sc.SLIPPAGE)
            qty = size * (1 - sc.FEE_RATE) / price
            state["cash"] -= size
            state["open"].append({
                "symbol": cand["symbol"], "entry_price": price, "qty": qty,
                "cost": size, "stop": cand["stop"],
                "entry_ms": cand["entry_ms"], "score": cand["score"],
            })
            last_prices[cand["symbol"]] = cand["price"]
            log(event="BUY", symbol=cand["symbol"], price=f"{price:.6g}",
                size_usd=f"{size:.2f}", reason=f"stop {cand['stop']:.6g}",
                score=cand["score"])
            print(f"OPENED {cand['symbol']} at {price:.6g}, ${size:.2f}, "
                  f"stop {cand['stop']:.6g}, score {cand['score']}")

    save_state(state)

    total = equity(state, last_prices)
    closed = state["closed"]
    wins = sum(1 for c in closed if c["pnl"] > 0)
    print("\n--- PAPER ACCOUNT ---")
    print(f"Equity: ${total:.2f}  ({(total / sc.STARTING_BALANCE - 1) * 100:+.1f}% "
          f"vs ${sc.STARTING_BALANCE:.0f} start)   Cash: ${state['cash']:.2f}")
    for pos in state["open"]:
        price = last_prices.get(pos["symbol"], pos["entry_price"])
        move = (price / pos["entry_price"] - 1) * 100
        print(f"  open {pos['symbol']:<12} entry {pos['entry_price']:.6g}  "
              f"now {price:.6g} ({move:+.1f}%)  stop {pos['stop']:.6g}")
    if not state["open"]:
        print("  no open positions")
    if closed:
        print(f"Closed trades: {len(closed)}  wins: {wins} "
              f"({100 * wins / len(closed):.0f}%)  "
              f"total P/L: ${sum(c['pnl'] for c in closed):+.2f}")
    print("Judge it after 30+ closed trades, not before.")


if __name__ == "__main__":
    main()
