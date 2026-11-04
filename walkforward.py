"""
walkforward.py — historical 15m single-timeframe execution simulation.

This tests the sequential 15m entry rules with realistic next-candle entries,
stop/target checks, fees and slippage. It DOES NOT claim to reproduce the
scanner's 5m/1h/4h filters; use it as a conservative first validation only.
Kraken OHLC is limited to 720 candles per request.
"""
import argparse
import time
import requests
import pandas as pd

from signals import SignalConfig, add_indicators, signal_at, BUY, _sequential_setup

KRAKEN_URL = "https://api.kraken.com/0/public"
FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0015
START_BALANCE = 200.0
RISK_DOLLARS = 2.0


def fetch_ohlc(pair, interval=15):
    r = requests.get(f"{KRAKEN_URL}/OHLC", params={"pair": pair, "interval": interval}, timeout=25)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body["result"]
    key = next(k for k in result if k != "last")
    rows = result[key][:-1]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def run_backtest(df, cfg, starting_balance=START_BALANCE):
    data = add_indicators(df, cfg)
    balance = float(starting_balance)
    trades = []
    i = max(cfg.trend_period or cfg.slow, 220)
    while i < len(data) - 2:
        if signal_at(data, i, cfg) != BUY:
            i += 1
            continue

        # Signal forms on closed candle i; simulate entry at next candle open.
        entry_i = i + 1
        entry = float(data.iloc[entry_i]["open"]) * (1 + SLIPPAGE_PER_SIDE)
        # Use the actual sweep low from the same sequential setup logic as signals.py.
        setup = _sequential_setup(data, i, cfg)
        if setup is None:
            i += 1
            continue
        atr = float(data.iloc[i]["atr"])
        sweep_low = float(setup["sweep_low"])
        stop = min(sweep_low, entry - cfg.atr_stop_mult * atr) - cfg.stop_buffer_atr * atr
        if not (0 < stop < entry):
            i += 1
            continue
        risk_per_unit = (entry - stop) + entry * 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
        qty = min(RISK_DOLLARS / risk_per_unit, balance / entry)
        if qty <= 0:
            i += 1
            continue
        target = entry + cfg.reward_risk * (entry - stop)
        exit_price = None
        outcome = "OPEN"
        exit_i = None

        for j in range(entry_i, len(data)):
            candle = data.iloc[j]
            hit_stop = float(candle["low"]) <= stop
            hit_target = float(candle["high"]) >= target
            if hit_stop and hit_target:
                exit_price, outcome, exit_i = stop, "STOP (both touched; conservative)", j
                break
            if hit_stop:
                exit_price, outcome, exit_i = stop, "STOP", j
                break
            if hit_target:
                exit_price, outcome, exit_i = target, "TARGET", j
                break

        if exit_price is None:
            # Do not count an unfinished trade as a win/loss.
            break

        exit_fill = exit_price * (1 - SLIPPAGE_PER_SIDE)
        gross = (exit_fill - entry) * qty
        fees = (entry * qty + exit_fill * qty) * FEE_PER_SIDE
        net = gross - fees
        balance += net
        trades.append({
            "entry_time": data.iloc[entry_i]["time"] if "time" in data else entry_i,
            "exit_time": data.iloc[exit_i]["time"] if "time" in data else exit_i,
            "entry": entry, "stop": stop, "target": target, "exit": exit_fill,
            "quantity": qty, "outcome": outcome, "net_pnl": net, "balance": balance,
            "R_multiple": net / RISK_DOLLARS
        })
        i = exit_i + 1

    results = pd.DataFrame(trades)
    if results.empty:
        return results, {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0, "ending_balance": balance, "max_drawdown_pct": 0.0}

    wins = results[results["net_pnl"] > 0]["net_pnl"].sum()
    losses = abs(results[results["net_pnl"] < 0]["net_pnl"].sum())
    equity = pd.Series([starting_balance] + results["balance"].tolist())
    peak = equity.cummax()
    dd = ((peak - equity) / peak).max() * 100
    stats = {
        "trades": int(len(results)),
        "win_rate": round((results["net_pnl"] > 0).mean() * 100, 2),
        "net_pnl": round(float(results["net_pnl"].sum()), 2),
        "profit_factor": round(float(wins / losses), 3) if losses else float("inf"),
        "ending_balance": round(float(balance), 2),
        "max_drawdown_pct": round(float(dd), 2),
        "average_net_trade": round(float(results["net_pnl"].mean()), 3)
    }
    return results, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="XBTUSD,ETHUSD,SOLUSD")
    p.add_argument("--interval", type=int, default=15)
    p.add_argument("--csv", default="walkforward_trades.csv")
    args = p.parse_args()
    cfg = SignalConfig()
    all_trades = []
    for pair in [x.strip() for x in args.pairs.split(",") if x.strip()]:
        try:
            print(f"\nDownloading {pair} {args.interval}m candles...")
            df = fetch_ohlc(pair, args.interval)
            if len(df) < max(250, cfg.trend_period or cfg.slow + 5):
                print(f"{pair}: insufficient history ({len(df)} candles)")
                continue
            trades, stats = run_backtest(df, cfg)
            print(pair, stats)
            if not trades.empty:
                trades.insert(0, "pair", pair)
                all_trades.append(trades)
        except Exception as exc:
            print(f"{pair}: ERROR: {exc}")
        time.sleep(1)
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(args.csv, index=False)
        print(f"\nSaved {len(combined)} trades to {args.csv}")
    else:
        print("\nNo completed trades. This is not evidence of profitability.")


if __name__ == "__main__":
    main()