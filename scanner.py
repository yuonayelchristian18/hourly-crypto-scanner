
"""
Kraken AI + meme coin scanner.

- Discovers AI and meme coins using CoinGecko categories.
- Uses Kraken USD spot pairs, prices and OHLC candles.
- Uses signals.py for trading signals.
- Sends Telegram alerts for qualifying setups.
- Saves scan results to scanner_results.csv.

This version sends alerts only. It does not place live trades.
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from signals import (
    BUY,
    SELL,
    SignalConfig,
    add_indicators,
    latest_signal,
    setup_score,
    signal_at,
)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
KRAKEN_URL = "https://api.kraken.com/0/public"

DEFAULT_CATEGORIES = "artificial-intelligence,meme-token"

LABELS = {
    "artificial-intelligence": "AI",
    "meme-token": "MEME",
}

STARTING_BALANCE = 200.00
RISK_PER_TRADE = 2.00

FEE_RATE = 0.001
SLIPPAGE = 0.0015

MAX_CANDLES = 720
PRICE_TOLERANCE = 0.15

MIN_CANDLES = 210

STABLES = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD",
    "USDE", "PYUSD", "USDS", "BUSD", "USD1",
    "USDD",
}

CSV_FILE = "scanner_results.csv"


def make_session():
    session = requests.Session()

    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry),
    )

    return session


def kraken_request(session, endpoint, params=None):
    response = session.get(
        f"{KRAKEN_URL}/{endpoint}",
        params=params,
        timeout=25,
    )

    response.raise_for_status()

    body = response.json()

    if body.get("error"):
        raise RuntimeError(
            f"Kraken error: {body['error']}"
        )

    return body["result"]


def get_universe(
    session,
    categories,
    per_category,
    max_price,
    min_volume,
    min_mcap,
):
    """Discover coins using CoinGecko category information."""

    coins = {}

    for category in categories:
        response = session.get(
            COINGECKO_URL,
            params={
                "vs_currency": "usd",
                "category": category,
                "order": "market_cap_desc",
                "per_page": per_category,
                "page": 1,
            },
            timeout=25,
        )

        response.raise_for_status()

        label = LABELS.get(category, category)

        for coin in response.json():
            symbol = str(coin.get("symbol", "")).upper()

            if not symbol or symbol in STABLES:
                continue

            entry = coins.setdefault(
                coin["id"],
                {**coin, "labels": []},
            )

            if label not in entry["labels"]:
                entry["labels"].append(label)

        # CoinGecko free API rate limiting.
        time.sleep(2)

    kept = []

    for coin in coins.values():
        price = coin.get("current_price")

        if price is None or price <= 0:
            continue

        if price >= max_price:
            continue

        if (coin.get("total_volume") or 0) < min_volume:
            continue

        if (coin.get("market_cap") or 0) < min_mcap:
            continue

        kept.append(coin)

    kept.sort(
        key=lambda item: item.get("market_cap") or 0,
        reverse=True,
    )

    return kept


def get_kraken_pairs(session):
    """Return Kraken USD spot pairs indexed by base currency."""

    result = kraken_request(session, "AssetPairs")

    pairs = {}

    for pair_name, info in result.items():
        if info.get("quote") not in ("ZUSD", "USD"):
            continue

        if info.get("status", "online") != "online":
            continue

        wsname = info.get("wsname", "")

        if not wsname or "/" not in wsname:
            continue

        base = wsname.split("/")[0].upper()

        if base in STABLES:
            continue

        pairs.setdefault(
            base,
            {
                "pair": pair_name,
                "altname": info.get("altname", pair_name),
                "wsname": wsname,
            },
        )

    # Kraken sometimes uses XBT for Bitcoin.
    if "XBT" in pairs:
        pairs.setdefault("BTC", pairs["XBT"])

    return pairs


def get_kraken_candles(session, pair, interval):
    """
    Fetch Kraken OHLC candles.

    Kraken returns at most 720 candles. Its final candle is
    still forming, so it is excluded from calculations.
    """

    result = kraken_request(
        session,
        "OHLC",
        {
            "pair": pair,
            "interval": interval,
        },
    )

    candle_key = next(
        (key for key in result if key != "last"),
        None,
    )

    if candle_key is None:
        return pd.DataFrame()

    rows = result[candle_key]

    if len(rows) < 3:
        return pd.DataFrame()

    # Remove the currently forming candle.
    rows = rows[:-1]

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "count",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True,
    )

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    return df.reset_index(drop=True)


def get_kraken_price(session, pair):
    """Get the latest Kraken ticker price."""

    result = kraken_request(
        session,
        "Ticker",
        {"pair": pair},
    )

    if not result:
        return None

    ticker = next(iter(result.values()))

    return float(ticker["c"][0])


def add_signal_indicators(df, config):
    """Support the signals.py indicator function."""

    try:
        return add_indicators(df.copy(), config)
    except TypeError:
        return add_indicators(df.copy())


def calculate_fallback_score(df):
    """
    Backup score if setup_score() has a different signature.

    This is a setup-strength score, not a win probability.
    """

    if df.empty:
        return 0.0

    row = df.iloc[-1]

    score = 0.0

    close = float(row.get("close", 0) or 0)

    ema8 = row.get("ema_fast")
    ema40 = row.get("ema_slow")
    ema200 = row.get("ema_trend")

    if pd.notna(ema8) and pd.notna(ema40):
        if ema8 > ema40:
            score += 20

    if pd.notna(ema200) and close > ema200:
        score += 20

    rsi = row.get("rsi")

    if pd.notna(rsi):
        if 45 <= float(rsi) <= 65:
            score += 15
        elif 35 <= float(rsi) < 45:
            score += 8

    volume_ratio = row.get("volume_ratio")

    if pd.notna(volume_ratio):
        if float(volume_ratio) >= 1.5:
            score += 15
        elif float(volume_ratio) >= 1.0:
            score += 8

    for column, points in [
        ("liquidity_sweep", 10),
        ("structure_break", 10),
        ("retest", 5),
        ("bullish_engulfing", 5),
    ]:
        if bool(row.get(column, False)):
            score += points

    return round(min(score, 100), 2)


def get_setup_score(df):
    """Use the score function from signals.py when possible."""

    for argument in (df, df.iloc[-1]):
        try:
            result = setup_score(argument)

            if isinstance(result, dict):
                result = result.get(
                    "score",
                    result.get("setup_score", 0),
                )

            score = float(result)

            if 0 <= score <= 100:
                return round(score, 2)

        except (TypeError, ValueError, KeyError, AttributeError):
            continue

    return calculate_fallback_score(df)


def normalize_signal(result):
    """Convert different signal return formats into BUY/SELL/HOLD."""

    if isinstance(result, dict):
        result = result.get(
            "signal",
            result.get(
                "action",
                result.get("side", "HOLD"),
            ),
        )

    elif isinstance(result, (tuple, list)):
        result = result[0] if result else "HOLD"

    if result is None:
        return "HOLD"

    result = str(result).upper().strip()

    if result in ("BUY", "LONG"):
        return BUY

    if result in ("SELL", "SHORT"):
        return SELL

    return "HOLD"


def get_signal(df, df_1h, df_4h, config):
    """Ask signals.py for the latest trading signal."""

    result = None

    # Preferred call: full multi-timeframe analysis.
    try:
        result = latest_signal(
            df,
            df_1h=df_1h,
            df_4h=df_4h,
            config=config,
            risk_amount=RISK_PER_TRADE,
        )

    except TypeError:
        # Compatibility with simpler latest_signal versions.
        try:
            result = latest_signal(df)
        except Exception:
            result = None

    except Exception:
        result = None

    signal = normalize_signal(result)

    if signal != "HOLD":
        return signal, result

    # Fallback: inspect the most recent completed candle.
    try:
        enriched = add_signal_indicators(df, config)

        result = signal_at(
            enriched,
            len(enriched) - 1,
            config,
        )

        signal = normalize_signal(result)

    except Exception:
        signal = "HOLD"

    return signal, result


def extract_stop(result, df):
    """Extract a stop price if signals.py provides one."""

    if isinstance(result, dict):
        for key in (
            "stop",
            "stop_price",
            "stop_loss",
            "sl",
        ):
            value = result.get(key)

            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return None


def send_telegram(message):
    """Send a Telegram message if credentials are configured."""

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram credentials not configured.")
        return False

    url = (
        f"https://api.telegram.org/bot{token}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": message,
            },
            timeout=20,
        )

        response.raise_for_status()

        body = response.json()

        if not body.get("ok"):
            print("Telegram rejected the message:", body)
            return False

        return True

    except requests.RequestException as error:
        print("Telegram error:", error)
        return False


def save_result(row):
    """Append a scan result to CSV."""

    file_exists = os.path.isfile(CSV_FILE)

    fields = [
        "scan_time",
        "coin",
        "symbol",
        "categories",
        "price",
        "signal",
        "score",
        "stop",
        "risk_amount",
        "position_size",
    ]

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def scan_coin(session, coin, pair_info, args, config):
    """Scan one coin using Kraken market data."""

    symbol = str(coin["symbol"]).upper()
    name = coin.get("name", symbol)

    pair = pair_info["pair"]

    try:
        price = get_kraken_price(session, pair)

        if price is None or price <= 0:
            print(f"{symbol}: no valid Kraken price")
            return

        # Compare Kraken with CoinGecko only as a ticker check.
        cg_price = coin.get("current_price")

        if cg_price and cg_price > 0:
            difference = abs(price - cg_price) / cg_price

            if difference > PRICE_TOLERANCE:
                print(
                    f"{symbol}: skipped; Kraken/CoinGecko "
                    f"price difference is {difference:.1%}"
                )
                return

        # Kraken intervals: 15 minutes, 1 hour, 4 hours.
        df_15m = get_kraken_candles(
            session,
            pair,
            15,
        )

        time.sleep(0.25)

        df_1h = get_kraken_candles(
            session,
            pair,
            60,
        )

        time.sleep(0.25)

        df_4h = get_kraken_candles(
            session,
            pair,
            240,
        )

        if min(
            len(df_15m),
            len(df_1h),
            len(df_4h),
        ) < MIN_CANDLES:
            print(f"{symbol}: not enough candle history")
            return

        config = config

        # Calculate the score using 15-minute candles.
        enriched = add_signal_indicators(
            df_15m,
            config,
        )

        score = get_setup_score(enriched)

        signal, signal_result = get_signal(
            df_15m,
            df_1h,
            df_4h,
            config,
        )

        stop = extract_stop(
            signal_result,
            df_15m,
        )

        # Position sizing based on planned dollar risk.
        # This is an estimate, not an exchange order quantity.
        position_size = None

        if stop is not None and stop < price:
            risk_per_coin = price - stop

            if risk_per_coin > 0:
                position_size = RISK_PER_TRADE / risk_per_coin

        categories = ",".join(coin.get("labels", []))

        print(
            f"{symbol:12} "
            f"Price: ${price:.8f} "
            f"Signal: {signal:5} "
            f"Score: {score:5.1f}"
        )

        row = {
            "scan_time": datetime.now(
                timezone.utc
            ).isoformat(),
            "coin": name,
            "symbol": symbol,
            "categories": categories,
            "price": round(price, 10),
            "signal": signal,
            "score": score,
            "stop": stop,
            "risk_amount": RISK_PER_TRADE,
            "position_size": position_size,
        }

        save_result(row)

        if args.no_filters:
            qualifies = signal == BUY
        else:
            qualifies = (
                signal == BUY
                and score >= args.alert
            )

        if not qualifies:
            return

        message = (
            f"🚨 KRAKEN BUY SETUP\n\n"
            f"Coin: {name} ({symbol})\n"
            f"Pair: {pair_info['wsname']}\n"
            f"Price: ${price:.8f}\n"
            f"Setup score: {score}/100\n"
            f"Categories: {categories}\n\n"
            f"Planned risk: ${RISK_PER_TRADE:.2f}\n"
        )

        if stop is not None:
            message += f"Technical stop: ${stop:.8f}\n"

        if position_size is not None:
            message += (
                f"Estimated coin quantity: "
                f"{position_size:.8f}\n"
            )

        message += (
            "\nThis is an alert, not an executed trade. "
            "Check fees, slippage and the chart before entering."
        )

        send_telegram(message)

    except Exception as error:
        print(f"{symbol}: scan failed: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Kraken AI and meme coin scanner"
    )

    parser.add_argument(
        "--categories",
        default=DEFAULT_CATEGORIES,
    )

    parser.add_argument(
        "--per-category",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-price",
        type=float,
        default=1.00,
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=100000,
    )

    parser.add_argument(
        "--min-mcap",
        type=float,
        default=1000000,
    )

    parser.add_argument(
        "--alert",
        type=float,
        default=70,
    )

    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Alert on BUY signals regardless of score",
    )

    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Do not display a calculated stop in alerts",
    )

    args = parser.parse_args()

    if args.max_price <= 0:
        parser.error("--max-price must be positive")

    if not 0 <= args.alert <= 100:
        parser.error("--alert must be between 0 and 100")

    categories = [
        item.strip()
        for item in args.categories.split(",")
        if item.strip()
    ]

    print("=" * 60)
    print("KRAKEN AI + MEME COIN SCANNER")
    print("=" * 60)

    print("Starting balance reference:", STARTING_BALANCE)
    print("Planned risk per trade:", RISK_PER_TRADE)
    print("Alert score:", args.alert)
    print("Categories:", categories)

    session = make_session()

    config = SignalConfig()

    try:
        print("\nLoading Kraken trading pairs...")

        pairs = get_kraken_pairs(session)

        print(f"Kraken USD pairs found: {len(pairs)}")

        print("\nDiscovering coins...")

        coins = get_universe(
            session,
            categories,
            args.per_category,
            args.max_price,
            args.min_volume,
            args.min_mcap,
        )

        print(f"Coins discovered: {len(coins)}")

        matched = []

        for coin in coins:
            symbol = str(
                coin.get("symbol", "")
            ).upper()

            pair_info = pairs.get(symbol)

            if pair_info is None and symbol == "BTC":
                pair_info = pairs.get("XBT")

            if pair_info is None:
                continue

            matched.append((coin, pair_info))

        print(
            f"Coins with Kraken USD pairs: {len(matched)}"
        )

        if not matched:
            print(
                "No matching Kraken USD pairs found. "
                "Check the category list and coin availability."
            )
            return

        print("\nStarting scan...\n")

        for coin, pair_info in matched:
            scan_coin(
                session,
                coin,
                pair_info,
                args,
                config,
            )

            # Keep requests spaced out.
            time.sleep(0.5)

        print("\nScan complete.")
        print("Results saved to:", CSV_FILE)

    except Exception as error:
        print("\nSCANNER ERROR:", error)
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()
