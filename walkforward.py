import time
import requests
import pandas as pd
import numpy as np

API = "https://api.exchange.coinbase.com"
DAYS = 180
MIN_USD_VOL = 2_000_000   # 24h volume filter
MAX_PAIRS = 40
COST = 2 * (0.0010 + 0.0015)   # round-trip fee + slippage
ATR_MULT = 3.0
SKIP = {"USDT", "USDC", "DAI", "EURC", "PYUSD", "GUSD", "WBTC"}


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    return None


def liquid_pairs():
    prods = get(API + "/products") or []
    out = []
    for p in prods:
        if p.get("quote_currency") != "USD" or p.get("status") != "online":
            continue
        if p.get("trading_disabled") or p.get("base_currency") in SKIP:
            continue
        s = get(f"{API}/products/{p['id']}/stats")
        time.sleep(0.12)
        try:
            usd = float(s["volume"]) * float(s["last"])
        except Exception:
            continue
        if usd >= MIN_USD_VOL:
            out.append((usd, p["id"]))
    out.sort(reverse=True)
    ids = [i for _, i in out][:MAX_PAIRS]
    if "BTC-USD" not in ids:
        ids.append("BTC-USD")
    return ids


def candles(pid):
    end = pd.Timestamp.now(tz="UTC").floor("h")
    stop = end - pd.Timedelta(days=DAYS + 10)
    rows = []
    while end > stop:
        start = end - pd.Timedelta(hours=299)
        data = get(f"{API}/products/{pid}/candles", {
            "start": start.isoformat(), "end": end.isoformat(),
            "granularity": 3600})
        time.sleep(0.15)
        if not data:
            break
        rows += data
        end = start - pd.Timedelta(hours=1)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop_duplicates("t").sort_values("t").set_index("t").astype(float)
    return df if len(df) > 400 else None


def indicators(df):
    c = df["close"]
    df["ema100"] = c.ewm(span=100, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["hh"] = df["high"].rolling(48).max().shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vavg"] = df["volume"].rolling(24).mean().shift(1)
    return df


def simulate(pid, df, regime):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr, hh, e100, va, vol = (df[k].values for k in ("atr", "hh", "ema100", "vavg", "volume"))
    reg = regime.reindex(df.index).ffill().fillna(False).values
    idx = df.index
    n = len(df)
    trades = []
    i = 210
    while i < n - 1:
        ok = (reg[i] and c[i] > hh[i] and c[i] > e100[i]
              and vol[i] > 1.5 * va[i] and atr[i] > 0)
        if not ok:
            i += 1
            continue
        entry = o[i + 1]
        risk = ATR_MULT * atr[i]
        if risk / entry < 5 * COST:   # stop must be wide vs. costs
            i += 1
            continue
        trail = entry - risk
        j = i + 1
        exit_p = None
        while j < n:
            if l[j] <= trail:
                exit_p = min(trail, o[j])
                break
            trail = max(trail, c[j] - ATR_MULT * atr[j])
            j += 1
        if exit_p is None:
            exit_p, j = c[n - 1], n - 1
        r = (exit_p - entry - entry * COST) / risk
        trades.append({"pair": pid, "time": idx[i + 1], "r": r})
        i = j + 1
    return trades


def stats(t):
    if len(t) == 0:
        return {"trades": 0}
    r = t["r"]
    w, lo = r[r > 0].sum(), -r[r <= 0].sum()
    return {"trades": len(t), "win_rate": round((r > 0).mean() * 100, 1),
            "net_R": round(r.sum(), 1), "avg_R": round(r.mean(), 3),
            "profit_factor": round(w / lo, 2) if lo > 0 else float("inf")}


def main():
    print("Finding liquid pairs...")
    pairs = liquid_pairs()
    print(f"{len(pairs)} pairs pass the volume check")
    btc = candles("BTC-USD")
    btc = indicators(btc)
    regime = btc["close"] > btc["ema200"]
    all_trades = []
    for k, pid in enumerate(pairs, 1):
        df = candles(pid)
        if df is None:
            print(f"[{k}/{len(pairs)}] {pid}: not enough data")
            continue
        df = indicators(df)
        t = simulate(pid, df, regime)
        all_trades += t
        print(f"[{k}/{len(pairs)}] {pid}: {len(t)} trades")
    if not all_trades:
        print("No trades found.")
        return
    t = pd.DataFrame(all_trades).sort_values("time")
    t.to_csv("trend_trades.csv", index=False)
    cut = t["time"].min() + 0.7 * (t["time"].max() - t["time"].min())
    train, test = t[t["time"] <= cut], t[t["time"] > cut]
    print("\nALL  :", stats(t))
    print("TRAIN:", stats(train))
    print("TEST :", stats(test))
    bh = btc["close"].iloc[-1] / btc["close"].iloc[-DAYS * 24] - 1
    print(f"BTC buy-and-hold over {DAYS} days: {bh*100:.1f}%")
    s = stats(test)
    ok = s.get("trades", 0) >= 50 and s.get("profit_factor", 0) >= 1.2
    print("VERDICT:", "possible edge, keep paper trading" if ok
          else "no proven edge yet, do not use real money")


if __name__ == "__main__":
    main()
