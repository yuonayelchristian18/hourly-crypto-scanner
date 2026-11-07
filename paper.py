import json
import os
import pandas as pd
import trend_bot as tb

RISK_USD = 2.0
STATE = "paper_state.json"
LOG = "paper_log.csv"
P = tb.DEFAULT


def load():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"positions": [], "last_bar": None}


def save(s):
    with open(STATE, "w") as f:
        json.dump(s, f, indent=1)


def log_trade(row):
    new = not os.path.exists(LOG)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=new, index=False)


def main():
    s = load()
    now = pd.Timestamp.now(tz="UTC").floor("h")
    pairs = tb.liquid_pairs()
    btc = tb.candles("BTC-USD", 12)
    if btc is None:
        raise SystemExit("No BTC data")
    btc = tb.add_base(btc)
    btc = btc[btc.index < now]
    reg = tb.regime(btc, P["ema_len"])
    last_bar = btc.index[-1]

    data = {}
    for pid in set(pairs) | {p["pair"] for p in s["positions"]}:
        df = tb.candles(pid, 12)
        if df is None:
            continue
        df = tb.add_base(df)
        data[pid] = df[df.index < now]

    still = []
    for pos in s["positions"]:
        df = data.get(pos["pair"])
        if df is None:
            still.append(pos)
            continue
        if pos["status"] == "pending":
            et = pd.Timestamp(pos["signal_time"]) + pd.Timedelta(hours=1)
            if et not in df.index:
                still.append(pos)
                continue
            pos["entry"] = float(df.loc[et, "open"])
            pos["trail"] = pos["entry"] - pos["risk"]
            pos["status"] = "open"
            pos["checked"] = (et - pd.Timedelta(hours=1)).isoformat()
            print("ENTERED", pos["pair"], "at", pos["entry"])
        closed = False
        start = pd.Timestamp(pos["checked"])
        for t, bar in df[df.index > start].iterrows():
            if bar["low"] <= pos["trail"]:
                exit_p = min(pos["trail"], bar["open"])
                r = (exit_p - pos["entry"] - pos["entry"] * tb.COST) / pos["risk"]
                log_trade({"pair": pos["pair"], "signal_time": pos["signal_time"],
                           "exit_time": t.isoformat(), "entry": pos["entry"],
                           "exit": exit_p, "r": round(r, 3),
                           "pnl_usd": round(r * RISK_USD, 2)})
                print("CLOSED", pos["pair"], f"R={r:.2f}", f"${r * RISK_USD:.2f}")
                closed = True
                break
            pos["trail"] = max(pos["trail"], bar["close"] - P["atr_mult"] * bar["atr"])
            pos["checked"] = t.isoformat()
        if not closed:
            still.append(pos)

    if s.get("last_bar") != last_bar.isoformat():
        held = {p["pair"] for p in still}
        btc_ok = bool(reg.iloc[-1])
        for pid, df in data.items():
            if pid in held or pid not in pairs or len(df) < 210:
                continue
            if df.index[-1] != last_bar:
                continue
            hh = df["high"].rolling(P["brk"]).max().shift(1)
            row = df.iloc[-1]
            ok = (btc_ok and row["close"] > hh.iloc[-1]
                  and row["close"] > row["ema100"]
                  and row["volume"] > P["vol_mult"] * row["vavg"]
                  and row["atr"] > 0)
            if not ok:
                continue
            risk = P["atr_mult"] * row["atr"]
            if risk / row["close"] < 5 * tb.COST:
                continue
            still.append({"pair": pid, "status": "pending",
                          "signal_time": last_bar.isoformat(), "risk": float(risk)})
            print("SIGNAL", pid, "at", row["close"])
        s["last_bar"] = last_bar.isoformat()

    s["positions"] = still
    save(s)

    print(f"\nBTC uptrend filter on: {bool(reg.iloc[-1])}")
    print(f"Open or pending positions: {len(still)}")
    if os.path.exists(LOG):
        lg = pd.read_csv(LOG)
        print(f"Closed trades: {len(lg)} | net R {lg['r'].sum():.1f} "
              f"| paper P&L ${lg['pnl_usd'].sum():.2f}")


if __name__ == "__main__":
    main()
