"""Indicator and signal logic for scanner.py.

Pure functions only: no network calls, no printing, so everything here
can be tested on its own. `add_indicators` and `signal_at` keep the same
names as before, so scanner.py can import them directly.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


@dataclass(frozen=True)
class SignalConfig:
    fast: int = 9
    slow: int = 21
    rsi_period: int = 14
    rsi_low: float = 50.0
    rsi_high: float = 65.0
    trend_period: Optional[int] = 100    # None disables the trend filter
    vol_period: int = 20
    min_vol_ratio: float = 1.0           # 0 disables the volume filter
    atr_period: int = 14
    atr_stop_mult: float = 2.0

    def __post_init__(self):
        if self.fast >= self.slow:
            raise ValueError("fast period must be shorter than slow period")
        if not 0 <= self.rsi_low < self.rsi_high <= 100:
            raise ValueError("RSI band must satisfy 0 <= low < high <= 100")
        if self.trend_period is not None and self.trend_period <= self.slow:
            raise ValueError("trend_period must be longer than slow period")


DEFAULT_CONFIG = SignalConfig()


def _as_numeric(df):
    df = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_indicators(df, cfg=DEFAULT_CONFIG):
    """Return a copy of df with indicator columns added."""
    if df is None or df.empty:
        raise ValueError("empty candle data")
    for col in ("close", "volume"):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    df = _as_numeric(df)
    close = df["close"]

    df["ema_fast"] = close.ewm(span=cfg.fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=cfg.slow, adjust=False).mean()

    if cfg.trend_period:
        df["ema_trend"] = close.ewm(
            span=cfg.trend_period, adjust=False, min_periods=cfg.trend_period
        ).mean()

    # RSI (Wilder smoothing)
    change = close.diff()
    avg_gain = change.clip(lower=0).ewm(
        alpha=1 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False
    ).mean()
    avg_loss = (-change.clip(upper=0)).ewm(
        alpha=1 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False
    ).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # No down moves at all means RSI is 100, not NaN.
    df["rsi"] = rsi.where(avg_loss != 0, 100.0)

    # Volume relative to its recent average
    avg_vol = df["volume"].rolling(cfg.vol_period).mean()
    df["vol_ratio"] = df["volume"] / avg_vol.replace(0, float("nan"))

    # ATR (needs high and low; skipped if they are missing)
    if {"high", "low"}.issubset(df.columns):
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = true_range.ewm(
            alpha=1 / cfg.atr_period, min_periods=cfg.atr_period, adjust=False
        ).mean()
    else:
        df["atr"] = float("nan")

    return df


def signal_at(df, i, cfg=DEFAULT_CONFIG):
    """Signal using only data up to and including row i (no look-ahead)."""
    if i < 1 or i >= len(df):
        return HOLD

    row, prev = df.iloc[i], df.iloc[i - 1]

    needed = [row["ema_fast"], row["ema_slow"], prev["ema_fast"],
              prev["ema_slow"], row["rsi"]]
    if any(pd.isna(v) for v in needed):
        return HOLD

    crossed_up = (prev["ema_fast"] <= prev["ema_slow"]
                  and row["ema_fast"] > row["ema_slow"])
    crossed_down = (prev["ema_fast"] >= prev["ema_slow"]
                    and row["ema_fast"] < row["ema_slow"])

    if crossed_down:
        return SELL

    if crossed_up and cfg.rsi_low <= row["rsi"] <= cfg.rsi_high:
        if cfg.trend_period:
            trend = row.get("ema_trend")
            if pd.isna(trend) or row["close"] <= trend:
                return HOLD
        if cfg.min_vol_ratio > 0:
            ratio = row.get("vol_ratio")
            if pd.isna(ratio) or ratio < cfg.min_vol_ratio:
                return HOLD
        return BUY

    return HOLD


def latest_signal(df, cfg=DEFAULT_CONFIG):
    """Summary of the most recent closed candle, for display or alerts."""
    i = len(df) - 1
    row = df.iloc[i]
    atr = row.get("atr")
    stop = None
    if atr is not None and not pd.isna(atr):
        stop = float(row["close"] - cfg.atr_stop_mult * atr)

    return {
        "signal": signal_at(df, i, cfg),
        "price": float(row["close"]),
        "rsi": None if pd.isna(row["rsi"]) else float(row["rsi"]),
        "vol_ratio": None if pd.isna(row["vol_ratio"]) else float(row["vol_ratio"]),
        "suggested_stop": stop,
    }


def setup_score(df, cfg=DEFAULT_CONFIG):
    """Explainable 0-100 score for how strong the current setup looks.

    This is a ranking tool, NOT a probability that the price will rise.
    Components: trend 25, RSI zone 20, volume surge 20, breakout 15,
    24h momentum 10, volatility 10. Penalties for overbought or
    already-pumped coins.
    """
    last = len(df) - 1
    if last < 30:
        return {"score": 0, "reasons": ["not enough data"], "flags": [],
                "ret_24h": None}

    row = df.iloc[last]
    close = float(row["close"])
    score, reasons, flags = 0, [], []

    # Trend (25)
    if pd.notna(row["ema_fast"]) and pd.notna(row["ema_slow"]) \
            and row["ema_fast"] > row["ema_slow"]:
        score += 10
        reasons.append("fast EMA above slow EMA")
    trend = row.get("ema_trend")
    if trend is not None and pd.notna(trend) and close > trend:
        score += 15
        reasons.append("price above trend EMA")

    # RSI zone (20)
    rsi = row["rsi"]
    if pd.notna(rsi):
        if 55 <= rsi <= 70:
            score += 20
            reasons.append(f"RSI {rsi:.0f} in momentum zone")
        elif 50 <= rsi < 55 or 70 < rsi <= 75:
            score += 10
        if rsi > 78:
            score -= 15
            flags.append(f"overbought (RSI {rsi:.0f})")

    # Volume surge, averaged over the last 3 candles (20)
    vol = df["vol_ratio"].iloc[-3:].mean()
    if pd.notna(vol):
        if vol >= 2:
            score += 20
            reasons.append(f"volume {vol:.1f}x average")
        elif vol >= 1.5:
            score += 14
            reasons.append(f"volume {vol:.1f}x average")
        elif vol >= 1:
            score += 8

    # Breakout: close to the 3-day high (15)
    highs = df["high"] if "high" in df.columns else df["close"]
    high72 = float(highs.iloc[-72:].max())
    dist = close / high72 - 1
    if dist >= -0.03:
        score += 15
        reasons.append("near 3-day high")
    elif dist >= -0.06:
        score += 8

    # 24h momentum, penalising coins that already ran (10)
    ret24 = None
    if last >= 25:
        ret24 = close / float(df["close"].iloc[-25]) - 1
        if 0.03 <= ret24 <= 0.20:
            score += 10
            reasons.append(f"+{ret24 * 100:.0f}% in 24h")
        elif 0 <= ret24 < 0.03:
            score += 5
        if ret24 > 0.30:
            score -= 20
            flags.append(f"already pumped +{ret24 * 100:.0f}% in 24h")

    # Volatility: enough movement to matter, not chaos (10)
    atr = row.get("atr")
    if atr is not None and pd.notna(atr):
        atr_pct = float(atr) / close
        score += 10 if 0.008 <= atr_pct <= 0.04 else 5
        if atr_pct > 0.04:
            flags.append(f"very volatile (ATR {atr_pct * 100:.1f}%)")

    return {"score": max(0, min(100, score)), "reasons": reasons,
            "flags": flags, "ret_24h": ret24}


if __name__ == "__main__":
    # Self-check on synthetic data (no network needed).
    import random

    random.seed(3)
    price, rows = 100.0, []
    for _ in range(600):
        open_ = price
        price *= 1 + random.gauss(0.0003, 0.01)
        high = max(open_, price) * 1.003
        low = min(open_, price) * 0.997
        rows.append([open_, high, low, price, random.uniform(50, 150)])

    data = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    data = add_indicators(data)

    counts = {BUY: 0, SELL: 0, HOLD: 0}
    for k in range(len(data)):
        counts[signal_at(data, k)] += 1

    print("Signal counts:", counts)
    print("Latest:", latest_signal(data))
    print("Score:", setup_score(data))
