"""
walkforward.py — 15m historical trading simulation.

Reports individual entry filters plus a setup funnel so we can
identify why trades are or are not being generated.

Uses closed candles, next-candle entries, stop/target execution,
fees, slippage and a $200 starting balance.

DATA SOURCES:
- coinbase (default): paginated history, any number of days.
- kraken: at most 720 recent candles (API limit, cannot page back).

IMPORTANT:
- This tests 15m signals only, not the scanner's 5m/1h/4h filters.
- No live orders are placed.
"""

import argparse
import time
from collections import Counter

import requests
import pandas as pd

from signals import (
    SignalConfig,
    add_indicators,
    signal_at,
    BUY,
    _sequential_setup,
)

KRAKEN_URL = "https://api.kraken.com/0/public"
COINBASE_URL = "https://api.exchange.coinbase.com"

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0015

START_BALANCE = 200.0
RISK_DOLLARS = 2.0

DEFAULT_PAIRS = {
    "coinbase": "BTC-USD,ETH-USD,SOL-USD",
    "kraken": "XBTUSD,ETHUSD,SOLUSD",
}


def fetch_ohlc(pair, interval=15):
    """Kraken: max 720 most recent candles."""
    url = f"{KRAKEN_URL}/OHLC"

    response = requests.get(
        url,
        params={"pair": pair, "interval": interval},
        timeout=30,
    )

    response.raise_for_status()
    body = response.json()

    if body.get("error"):
        raise RuntimeError(body["error"])

    result = body["result"]
    key = next(k for k in result if k != "last")

    # Exclude the currently forming candle.
    rows = result[key][:-1]

    columns = [
        "time", "open", "high", "low",
        "close", "vwap", "volume", "count"
    ]

    df = pd.DataFrame(rows, columns=columns)

    for column in [
        "open", "high", "low", "close", "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    return (
        df.dropna(
            subset=["open", "high", "low", "close", "volume"]
        )
        .reset_index(drop=True)
    )


def fetch_coinbase(product, days=60, granularity=900):
    """Coinbase: paginated history, 300 candles max per request."""
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    cur = end - pd.Timedelta(days=days)
    step = pd.Timedelta(seconds=granularity * 290)
    frames = []

    while cur < end:
        nxt = min(cur + step, end)

        response = requests.get(
            f"{COINBASE_URL}/products/{product}/candles",
            params={
                "granularity": granularity,
                "start": cur.isoformat(),
                "end": nxt.isoformat(),
            },
            timeout=30,
        )
        response.raise_for_status()

        rows = response.json()

        if rows:
            frames.append(pd.DataFrame(
                rows,
                columns=[
                    "time", "low", "high",
                    "open", "close", "volume",
                ],
            ))

        cur = nxt
        time.sleep(0.4)

    if not frames:
        raise RuntimeError(f"No candles returned for {product}")

    df = pd.concat(frames, ignore_index=True)

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    for column in [
        "open", "high", "low", "close", "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    return (
        df.drop_duplicates("time")
        .sort_values("time")
        # Drop any still-forming candle.
        .loc[lambda d: d["time"] < end]
        .dropna(
            subset=["open", "high", "low", "close", "volume"]
        )
        .reset_index(drop=True)
    )


def diagnose(data, cfg, start):
    """
    Count individual filters and complete BUY signals.
    Individual filter counts are not necessarily simultaneous.
    """

    stats = {
        "candles_checked": 0,
        "sequential_setups": 0,
        "ema_fast_above_slow": 0,
        "above_200_ema": 0,
        "rsi_in_range": 0,
        "volume_pass": 0,
        "signal_bar_pass": 0,
        "complete_buy_signals": 0,
    }

    for i in range(start, len(data) - 2):

        stats["candles_checked"] += 1
        row = data.iloc[i]

        setup = _sequential_setup(data, i, cfg)

        if setup is not None:
            stats["sequential_setups"] += 1

        if (
            pd.notna(row["ema_fast"])
            and pd.notna(row["ema_slow"])
            and row["ema_fast"] > row["ema_slow"]
        ):
            stats["ema_fast_above_slow"] += 1

        if (
            cfg.trend_period
            and pd.notna(row["ema_trend"])
            and row["close"] > row["ema_trend"]
        ):
            stats["above_200_ema"] += 1

        if (
            pd.notna(row["rsi"])
            and cfg.rsi_low <= row["rsi"] <= cfg.rsi_high
        ):
            stats["rsi_in_range"] += 1

        if (
            pd.notna(row["vol_ratio"])
            and row["vol_ratio"] >= cfg.min_vol_ratio
        ):
            stats["volume_pass"] += 1

        atr = row["atr"]

        if (
            pd.notna(atr)
            and atr > 0
            and row["high"] - row["low"]
            <= cfg.max_signal_bar_atr * atr
        ):
            stats["signal_bar_pass"] += 1

        if signal_at(data, i, cfg) == BUY:
            stats["complete_buy_signals"] += 1

    return stats


def setup_funnel(data, cfg, start):
    """
    For each bar with a sequential setup, record the FIRST filter
    that rejects it. Returns (passed_all_filters, blocked_by_counts).
    """
    blocked = Counter()
    passed = 0

    for i in range(start, len(data) - 2):

        if _sequential_setup(data, i, cfg) is None:
            continue

        row = data.iloc[i]
        atr = row["atr"]

        checks = [
            ("ema", row["ema_fast"] > row["ema_slow"]),
            (
                "above_200",
                (not cfg.trend_period)
                or row["close"] > row["ema_trend"],
            ),
            (
                "rsi",
                cfg.rsi_low <= row["rsi"] <= cfg.rsi_high,
            ),
            ("volume", row["vol_ratio"] >= cfg.min_vol_ratio),
            (
                "signal_bar",
                atr > 0
                and row["high"] - row["low"]
                <= cfg.max_signal_bar_atr * atr,
            ),
        ]

        failed = next(
            (name for name, ok in checks if not ok), None
        )

        if failed:
            blocked[failed] += 1
        else:
            passed += 1

    return passed, dict(blocked)


def run_backtest(df, cfg, starting_balance=START_BALANCE):

    data = add_indicators(df, cfg)

    balance = float(starting_balance)
    trades = []

    start = max(cfg.trend_period or cfg.slow, 220)

    diagnostics = diagnose(data, cfg, start)

    print(
        "SETUP FUNNEL (passed, blocked_by):",
        setup_funnel(data, cfg, start),
    )

    i = start

    while i < len(data) - 2:

        if signal_at(data, i, cfg) != BUY:
            i += 1
            continue

        # Signal on closed candle; enter next candle.
        entry_i = i + 1

        entry = (
            float(data.iloc[entry_i]["open"])
            * (1 + SLIPPAGE_PER_SIDE)
        )

        setup = _sequential_setup(data, i, cfg)

        if setup is None:
            i += 1
            continue

        atr = float(data.iloc[i]["atr"])
        sweep_low = float(setup["sweep_low"])

        stop = (
            min(
                sweep_low,
                entry - cfg.atr_stop_mult * atr
            )
            - cfg.stop_buffer_atr * atr
        )

        if not (0 < stop < entry):
            i += 1
            continue

        risk_per_unit = (
            (entry - stop)
            + entry * 2 * (
                FEE_PER_SIDE + SLIPPAGE_PER_SIDE
            )
        )

        if risk_per_unit <= 0:
            i += 1
            continue

        # Limit position size to $2 planned risk and available cash.
        risk_qty = RISK_DOLLARS / risk_per_unit

        cash_qty = balance / (
            entry * (1 + FEE_PER_SIDE)
        )

        qty = min(risk_qty, cash_qty)

        if qty <= 0:
            i += 1
            continue

        target = (
            entry
            + cfg.reward_risk * (entry - stop)
        )

        exit_price = None
        outcome = "OPEN"
        exit_i = None

        for j in range(entry_i, len(data)):

            candle = data.iloc[j]

            hit_stop = float(candle["low"]) <= stop
            hit_target = float(candle["high"]) >= target

            # If both are touched in one candle,
            # assume the stop was hit first.
            if hit_stop and hit_target:
                exit_price = stop
                outcome = "STOP_BOTH_TOUCHED"
                exit_i = j
                break

            if hit_stop:
                exit_price = stop
                outcome = "STOP"
                exit_i = j
                break

            if hit_target:
                exit_price = target
                outcome = "TARGET"
                exit_i = j
                break

        if exit_price is None:
            # Incomplete trade is not counted.
            break

        exit_fill = (
            exit_price * (1 - SLIPPAGE_PER_SIDE)
        )

        gross = (exit_fill - entry) * qty

        fees = (
            entry * qty + exit_fill * qty
        ) * FEE_PER_SIDE

        net = gross - fees

        balance += net

        trades.append({
            "entry_time": data.iloc[entry_i]["time"],
            "exit_time": data.iloc[exit_i]["time"],
            "entry": entry,
            "stop": stop,
            "target": target,
            "exit": exit_fill,
            "quantity": qty,
            "outcome": outcome,
            "net_pnl": net,
            "balance": balance,
            "R_multiple": net / RISK_DOLLARS,
        })

        # Do not overlap trades on the same pair.
        i = exit_i + 1

    results = pd.DataFrame(trades)

    if results.empty:

        summary = {
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "ending_balance": round(balance, 2),
            "max_drawdown_pct": 0.0,
        }

        return results, summary, diagnostics

    wins = results.loc[
        results["net_pnl"] > 0, "net_pnl"
    ].sum()

    losses = abs(
        results.loc[
            results["net_pnl"] < 0, "net_pnl"
        ].sum()
    )

    equity = pd.Series(
        [starting_balance] + results["balance"].tolist()
    )

    peak = equity.cummax()

    drawdown = (
        (peak - equity) / peak
    ).max() * 100

    summary = {
        "trades": int(len(results)),
        "win_rate": round(
            (results["net_pnl"] > 0).mean() * 100, 2
        ),
        "net_pnl": round(
            float(results["net_pnl"].sum()), 2
        ),
        "profit_factor": (
            round(float(wins / losses), 3)
            if losses > 0 else float("inf")
        ),
        "ending_balance": round(float(balance), 2),
        "max_drawdown_pct": round(float(drawdown), 2),
        "average_net_trade": round(
            float(results["net_pnl"].mean()), 3
        ),
    }

    return results, summary, diagnostics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["coinbase", "kraken"],
        default="coinbase",
    )

    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated. Defaults depend on --source.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="History length (coinbase only).",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=15
    )

    parser.add_argument(
        "--csv",
        default="walkforward_trades.csv"
    )

    args = parser.parse_args()

    pairs_arg = args.pairs or DEFAULT_PAIRS[args.source]

    cfg = SignalConfig()

    all_trades = []

    for pair in [
        x.strip()
        for x in pairs_arg.split(",")
        if x.strip()
    ]:

        try:

            print(
                f"\nDownloading {pair} {args.interval}m "
                f"candles from {args.source}..."
            )

            if args.source == "coinbase":
                df = fetch_coinbase(
                    pair,
                    days=args.days,
                    granularity=args.interval * 60,
                )
            else:
                df = fetch_ohlc(pair, args.interval)

            print(f"Completed candles received: {len(df)}")

            if len(df) < 250:
                print(
                    f"{pair}: insufficient history "
                    f"({len(df)} candles)"
                )
                continue

            trades, summary, diagnostics = run_backtest(
                df, cfg
            )

            print("\nENTRY DIAGNOSTICS")

            for name, value in diagnostics.items():
                print(f"{name}: {value}")

            print("\nBACKTEST RESULTS")
            print(summary)

            if not trades.empty:

                trades.insert(0, "pair", pair)
                all_trades.append(trades)

        except Exception as exc:

            print(f"{pair}: ERROR: {exc}")

        time.sleep(1)

    if all_trades:

        combined = pd.concat(
            all_trades,
            ignore_index=True
        )

        combined.to_csv(args.csv, index=False)

        print(
            f"\nSaved {len(combined)} trades to {args.csv}"
        )

    else:

        print(
            "\nNo completed trades. "
            "Check ENTRY DIAGNOSTICS and SETUP FUNNEL above."
        )


if __name__ == "__main__":
    main()
