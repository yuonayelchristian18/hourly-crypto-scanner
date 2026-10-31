
import time
import requests
import pandas as pd

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_URL = "https://api.binance.com/api/v3"

STARTING_BALANCE = 200.00
FEE_RATE = 0.001
CANDLES = 300

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
    response = session.get(
        f"{BINANCE_URL}/exchangeInfo", timeout=20
    )
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
        params={
            "symbol": symbol,
            "interval": "1h",
            "limit": CANDLES,
        },
        timeout=20,
    )
    response.raise_for_status()

    rows = response.json()
    columns = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "buy_volume", "buy_quote_volume", "ignore"
    ]

    df = pd.DataFrame(rows, columns=columns)
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    # Exclude the unfinished hourly candle.
    return df.iloc[:-1].copy().reset_index(drop=True)


def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    change = df["close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def signal_at(df, i):
    row = df.iloc[i]
    previous = df.iloc[i - 1]

    crossed_up = (
        previous["ema9"] <= previous["ema21"]
        and row["ema9"] > row["ema21"]
    )

    crossed_down = (
        previous["ema9"] >= previous["ema21"]
        and row["ema9"] < row["ema21"]
    )

    if crossed_up and 50 <= row["rsi"] <= 65:
        return "BUY"

    if crossed_down:
        return "SELL"

    return "HOLD"


def backtest(df):
    balance = STARTING_BALANCE
    quantity = 0.0
    entry_price = None
    trades = 0
    wins = 0

    for i in range(22, len(df)):
        price = float(df.iloc[i]["close"])
        signal = signal_at(df, i)

        if signal == "BUY" and quantity == 0:
            quantity = balance * (1 - FEE_RATE) / price
            entry_price = price
            balance = 0.0

        elif signal == "SELL" and quantity > 0:
            balance = quantity * price * (1 - FEE_RATE)
            trades += 1

            if price > entry_price:
                wins += 1

            quantity = 0.0
            entry_price = None

    if quantity > 0:
        balance = quantity * float(df.iloc[-1]["close"])
        balance *= 1 - FEE_RATE

    profit = balance - STARTING_BALANCE
    return balance, profit, trades, wins


def main():
    print("Loading top 50 cryptocurrencies...")
    coins = get_top_coins()
    binance_symbols = get_binance_symbols()

    results = []

    for coin in coins:
        symbol = coin["symbol"].upper() + "USDT"

        if symbol not in binance_symbols:
            continue

        try:
            df = add_indicators(get_candles(symbol))

            if len(df) < 50:
                continue

            last = len(df) - 1
            signal = signal_at(df, last)

            balance, profit, trades, wins = backtest(df)

            results.append({
                "symbol": symbol,
                "profit": profit,
                "trades": trades,
                "wins": wins,
            })

            print(
                f"{symbol:<14} "
                f"Price: {df.iloc[last]['close']:.6g} "
                f"RSI: {df.iloc[last]['rsi']:.1f} "
                f"Signal: {signal:<4} "
                f"Backtest P/L: ${profit:.2f} "
                f"Trades: {trades}"
            )

            time.sleep(0.15)

        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"Skipping {symbol}: {error}")

    if results:
        total_profit = sum(item["profit"] for item in results)
        total_trades = sum(item["trades"] for item in results)
        total_wins = sum(item["wins"] for item in results)

        print("\n--- HISTORICAL TEST SUMMARY ---")
        print(f"Coins tested: {len(results)}")
        print(f"Starting balance per coin: ${STARTING_BALANCE:.2f}")
        print(f"Combined simulated P/L: ${total_profit:.2f}")
        print(f"Completed trades: {total_trades}")

        if total_trades:
            print(f"Win rate: {100 * total_wins / total_trades:.1f}%")
        else:
            print("Win rate: No completed trades")


if __name__ == "__main__":
    main()
