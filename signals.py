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

