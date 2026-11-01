import time

import pandas as pd
import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
# If api.binance.com is blocked in your region (HTTP 451), try
# "https://data-api.binance.vision/api/v3" for market data.
BINANCE_URL = "https://api.binance.com/api/v3"

STARTING_BALANCE = 200.00
FEE_RATE = 0.001       # per side
SLIPPAGE = 0.0005      # per side, adverse
CANDLES = 1000         # Binance max per request (~41 days of 1h)
WARMUP = 22            # bars needed before signals are valid
RSI_LOW, RSI_HIGH = 50, 65

STABLES = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD",
    "USDS", "BUSD", "USD1", "USDD",
}

session = requests.Session()


def get_top_coins():
    response = session.get(
        COINGECKO_URL,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def get_binance_symbols():
    response = session.get(f"{BINANCE_URL}/exchangeInfo", timeout=20)
    response.raise_for_status()
    return {
        item["symbol"]
        for item in response.json()["symbols"]
        if item["status"] == "TRADING"
        and item["quoteAsset"] == "USDT"
        and item["isSpotTradingAllowed"]
    }


def get_candles(symbol):
    response = session.get(
        f"{BINANCE_URL}/klines",
        params={"symbol": symbol, "interval": "1h", "limit": CANDLES},
        timeout=20,
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json(), columns=[
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "buy_volume", "buy_quote_volume", "ignore",
    ])
    for col in ("open", "close", "volume"):
        df[col] = pd.to_numeric(df[col])

    # Drop the current, still-forming candle.
    if len(df) > 1:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def add_indicators(df):
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    change = df["close"].diff()
    avg_gain = change.clip(lower=0).ewm(
        alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = (-change.clip(upper=0)).ewm(
        alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # No losses at all means RSI is 100, not NaN.
    df["rsi"] = rsi.where(avg_loss != 0, 100.0)
    return df


def signal_at(df, i):
    row, prev = df.iloc[i], df.iloc[i - 1]

    crossed_up = prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]
    crossed_down = prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]

    if crossed_up and RSI_LOW <= row["rsi"] <= RSI_HIGH:
        return "BUY"
    if crossed_down:
        return "SELL"
    return "HOLD"


def backtest(df):
    """Signal on a candle's close, fill at the NEXT candle's open."""
    cash, qty, cash_in = STARTING_BALANCE, 0.0, 0.0
    trades = wins = 0
    pending = None
    equity = []

    for i in range(WARMUP, len(df)):
        open_price = float(df.iloc[i]["open"])
        close_price = float(df.iloc[i]["close"])

        if pending == "BUY" and qty == 0:
            fill = open_price * (1 + SLIPPAGE)
            cash_in = cash
            qty = cash * (1 - FEE_RATE) / fill
            cash = 0.0
        elif pending == "SELL" and qty > 0:
            fill = open_price * (1 - SLIPPAGE)
            cash = qty * fill * (1 - FEE_RATE)
            qty = 0.0
            trades += 1
            if cash > cash_in:   # win is judged after fees
                wins += 1

        equity.append(cash + qty * close_price)

        signal = signal_at(df, i)
        pending = signal if signal != "HOLD" else None

    last_close = float(df.iloc[-1]["close"])
    final = cash + qty * last_close * (1 - SLIPPAGE) * (1 - FEE_RATE)

    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = min(max_dd, e / peak - 1)

    start_price = float(df.iloc[WARMUP]["close"])
    buy_hold = (last_close / start_price) * (1 - FEE_RATE) ** 2 - 1

    return {
        "profit": final - STARTING_BALANCE,
        "trades": trades,
        "wins": wins,
        "max_dd": max_dd,
        "buy_hold": buy_hold,
        "in_position": qty > 0,
    }


def main():
    print("Loading top 50 coins...")
    coins = get_top_coins()
    binance_symbols = get_binance_symbols()

    results = []

    for coin in coins:
        base = coin["symbol"].upper()
        symbol = base + "USDT"

        if base in STABLES or symbol not in binance_symbols:
            continue

        try:
            df = add_indicators(get_candles(symbol))
            if len(df) < 50:
                continue

            last = len(df) - 1
            signal = signal_at(df, last)
            r = backtest(df)
            r.update(
                symbol=symbol,
                price=float(df.iloc[last]["close"]),
                rsi=float(df.iloc[last]["rsi"]),
                signal=signal,
            )
            results.append(r)

            print(
                f"{symbol:<12} Price: {r['price']:.6g}  RSI: {r['rsi']:.1f}  "
                f"Signal: {signal:<4}  P/L: ${r['profit']:.2f}  "
                f"B&H: {r['buy_hold'] * 100:.1f}%  "
                f"DD: {r['max_dd'] * 100:.1f}%  Trades: {r['trades']}"
            )
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")

        time.sleep(0.15)

    if not results:
        print("No results.")
        return

    total_profit = sum(r["profit"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    bh_profit = sum(STARTING_BALANCE * r["buy_hold"] for r in results)
    avg_dd = sum(r["max_dd"] for r in results) / len(results)
    beat = sum(1 for r in results if r["profit"] / STARTING_BALANCE > r["buy_hold"])

    print("\n--- HISTORICAL TEST SUMMARY ---")
    print(f"Coins tested: {len(results)}")
    print(f"Starting balance per coin: ${STARTING_BALANCE:.2f}")
    print(f"Strategy combined P/L:  ${total_profit:.2f}")
    print(f"Buy & hold combined P/L: ${bh_profit:.2f}")
    print(f"Beat buy & hold on {beat} of {len(results)} coins")
    print(f"Average max drawdown: {avg_dd * 100:.1f}%")
    print(f"Closed trades: {total_trades}")
    if total_trades:
        print(f"Win rate (after fees): {100 * total_wins / total_trades:.1f}%")
    else:
        print("Win rate: no closed trades")


if __name__ == "__main__":
    main()

        