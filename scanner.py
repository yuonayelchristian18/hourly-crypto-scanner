"""
scanner.py — Kraken AI/meme coin scanner (alerts only; no live orders).

Requires: pip install pandas requests
GitHub secrets: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
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

from signals import BUY, SignalConfig, add_indicators, latest_signal, setup_score

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
KRAKEN_URL = "https://api.kraken.com/0/public"
DEFAULT_CATEGORIES = "artificial-intelligence,meme-token"
LABELS = {"artificial-intelligence": "AI", "meme-token": "MEME"}
STARTING_BALANCE = 200.0
RISK_PER_TRADE = 2.0
FEE_RATE = 0.001       # assumed fee per side; check your Kraken fee tier
SLIPPAGE_RATE = 0.0015 # conservative assumed slippage per side
MIN_CANDLES = 210
MAX_PRICE_DEFAULT = 1.0
MAX_OPEN_POSITIONS = 2  # alerting reference only; no position tracking/orders
CSV_FILE = "scanner_results.csv"
STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDS", "BUSD", "USD1", "USDD"}


def make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def kraken_request(session, endpoint, params=None):
    r = session.get(f"{KRAKEN_URL}/{endpoint}", params=params, timeout=25)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")
    return body["result"]


def get_universe(session, categories, per_category, max_price, min_volume, min_mcap):
    coins = {}
    for category in categories:
        try:
            r = session.get(COINGECKO_URL, params={
                "vs_currency": "usd", "category": category, "order": "market_cap_desc",
                "per_page": per_category, "page": 1
            }, timeout=25)
            r.raise_for_status()
            label = LABELS.get(category, category)
            for coin in r.json():
                symbol = str(coin.get("symbol", "")).upper()
                if not symbol or symbol in STABLES:
                    continue
                entry = coins.setdefault(coin["id"], {**coin, "labels": []})
                if label not in entry["labels"]:
                    entry["labels"].append(label)
        except requests.RequestException as exc:
            print(f"CoinGecko category {category} failed: {exc}")
        time.sleep(3)
    kept = []
    for coin in coins.values():
        price = coin.get("current_price")
        if price is None or price <= 0 or price >= max_price:
            continue
        if (coin.get("total_volume") or 0) < min_volume:
            continue
        if (coin.get("market_cap") or 0) < min_mcap:
            continue
        kept.append(coin)
    kept.sort(key=lambda x: x.get("market_cap") or 0, reverse=True)
    return kept


def get_kraken_pairs(session):
    result = kraken_request(session, "AssetPairs")
    pairs = {}
    for pair_name, info in result.items():
        if info.get("quote") not in ("ZUSD", "USD") or info.get("status", "online") != "online":
            continue
        wsname = info.get("wsname", "")
        if "/" not in wsname:
            continue
        base = wsname.split("/")[0].upper()
        if base in STABLES:
            continue
        pairs.setdefault(base, {"pair": pair_name, "altname": info.get("altname", pair_name), "wsname": wsname})
    # Common symbol aliases.
    aliases = {"BTC": "XBT"}
    for wanted, actual in aliases.items():
        if actual in pairs:
            pairs.setdefault(wanted, pairs[actual])
    return pairs


def get_kraken_candles(session, pair, interval):
    result = kraken_request(session, "OHLC", {"pair": pair, "interval": interval})
    key = next((k for k in result if k != "last"), None)
    if key is None or len(result[key]) < 3:
        return pd.DataFrame()
    rows = result[key][:-1]  # discard still-forming candle
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for c in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def get_kraken_price(session, pair):
    result = kraken_request(session, "Ticker", {"pair": pair})
    ticker = next(iter(result.values()))
    return float(ticker["c"][0])


def send_telegram(message):
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not configured; alert printed only.")
        print(message)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": message}, timeout=20)
        r.raise_for_status()
        body = r.json()
        return bool(body.get("ok"))
    except requests.RequestException as exc:
        print("Telegram error:", exc)
        return False


def save_result(row):
    fields = ["scan_time", "coin", "symbol", "pair", "price", "signal", "score",
              "stop", "target", "risk_amount", "position_size", "estimated_cost"]
    exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scan_coin(session, coin, pair_info, args, cfg):
    symbol = str(coin["symbol"]).upper()
    name = coin.get("name", symbol)
    pair = pair_info["pair"]
    try:
        ticker_price = get_kraken_price(session, pair)
        cg_price = coin.get("current_price")
        if cg_price and cg_price > 0:
            diff = abs(ticker_price - cg_price) / cg_price
            if diff > 0.15:
                print(f"{symbol}: skipped; price feeds differ by {diff:.1%}")
                return

        # Fetch entry and confirmation timeframes; all candles returned are closed.
        frames = {}
        for interval in (5, 15, 60, 240):
            frames[interval] = get_kraken_candles(session, pair, interval)
            time.sleep(0.3)
        df5, df15, df1h, df4h = frames[5], frames[15], frames[60], frames[240]
        required = cfg.trend_period + 5 if cfg.trend_period else cfg.slow + 5
        if any(len(x) < max(MIN_CANDLES, required) for x in (df5, df15, df1h, df4h)):
            print(f"{symbol}: skipped; insufficient closed-candle history")
            return

        result = latest_signal(df15, cfg=cfg, df_1h=df1h, df_4h=df4h, df_5m=df5, risk_amount=RISK_PER_TRADE)
        enriched = add_indicators(df15, cfg)
        score_data = setup_score(enriched, cfg)
        score = float(score_data.get("score", 0))

        entry = float(result["price"]) if result.get("price") else None
        stop = result.get("suggested_stop")
        target = result.get("suggested_target")

        # Hard safety gates: never alert BUY without a complete plan.
        qualifies = (
            result.get("signal") == BUY and entry and stop and target and
            0 < float(stop) < entry < float(target) and score >= args.alert and
            not score_data.get("flags")
        )
        position_size = None
        estimated_cost = None
        if qualifies:
            # Approximate round-trip fees + slippage in per-unit risk.
            unit_risk = (entry - float(stop)) + entry * 2 * (FEE_RATE + SLIPPAGE_RATE)
            raw_size = RISK_PER_TRADE / unit_risk
            # Never allocate more than the reference account balance.
            position_size = min(raw_size, STARTING_BALANCE / entry)
            estimated_cost = position_size * entry * 2 * (FEE_RATE + SLIPPAGE_RATE)
            if position_size * entry > STARTING_BALANCE:
                qualifies = False

        row = {
            "scan_time": datetime.now(timezone.utc).isoformat(),
            "coin": name, "symbol": symbol, "pair": pair_info["wsname"],
            "price": entry, "signal": result.get("signal"), "score": score,
            "stop": stop, "target": target, "risk_amount": RISK_PER_TRADE if qualifies else "",
            "position_size": round(position_size, 8) if position_size else "",
            "estimated_cost": round(estimated_cost, 4) if estimated_cost is not None else ""
        }
        save_result(row)
        print(f"{symbol:12} ${ticker_price:.8f} | {result.get('signal')} | score {score:.1f} | {', '.join(result.get('reasons', []))}")

        if not qualifies:
            return
        message = (
            f"KRAKEN PAPER-TRADE SETUP\n\n{ name } ({symbol}) — {pair_info['wsname']}\n"
            f"Closed-candle entry reference: ${entry:.8f}\nStop: ${float(stop):.8f}\n"
            f"Target (2R+): ${float(target):.8f}\nScore: {score:.1f}/100 (not win probability)\n"
            f"Estimated quantity: {position_size:.8f}\nPlanned risk: ${RISK_PER_TRADE:.2f}\n"
            f"Estimated round-trip costs: ${estimated_cost:.4f}\n"
            f"Reasons: {', '.join(result.get('reasons', []))}\n\n"
            "Alert only — no order has been placed. Confirm current price, fees and liquidity before trading."
        )
        send_telegram(message)
    except Exception as exc:
        print(f"{symbol}: scan failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Kraken AI/meme coin alert scanner")
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--max-price", type=float, default=MAX_PRICE_DEFAULT)
    parser.add_argument("--min-volume", type=float, default=100000)
    parser.add_argument("--min-mcap", type=float, default=1000000)
    parser.add_argument("--alert", type=float, default=80)
    args = parser.parse_args()
    if args.max_price <= 0 or not 0 <= args.alert <= 100:
        parser.error("Invalid max-price or alert threshold")

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    cfg = SignalConfig()
    session = make_session()
    try:
        print("KRAKEN AI + MEME SCANNER — ALERTS ONLY")
        pairs = get_kraken_pairs(session)
        print(f"Kraken USD pairs: {len(pairs)}")
        coins = get_universe(session, categories, args.per_category, args.max_price, args.min_volume, args.min_mcap)
        print(f"Eligible CoinGecko coins: {len(coins)}")
        matched = []
        for coin in coins:
            sym = str(coin.get("symbol", "")).upper()
            pair_info = pairs.get(sym)
            if pair_info:
                matched.append((coin, pair_info))
        print(f"Matched Kraken pairs: {len(matched)}")
        for coin, pair_info in matched:
            scan_coin(session, coin, pair_info, args, cfg)
            time.sleep(0.5)
        print("Scan complete. CSV:", CSV_FILE)
    finally:
        session.close()


if __name__ == "__main__":
    main()