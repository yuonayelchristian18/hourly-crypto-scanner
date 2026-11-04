
"""Kraken trading signals: trend, momentum, liquidity sweeps and risk.

Pure functions only: no network calls, no order placement.
Signals use closed candles. Scores are rankings, NOT probabilities.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


@dataclass(frozen=True)
class SignalConfig:
    fast: int = 8
    slow: int = 40
    trend_period: Optional[int] = 200

    rsi_period: int = 14
    rsi_low: float = 50.0
    rsi_high: float = 68.0

    vol_period: int = 20
    min_vol_ratio: float = 1.2

    atr_period: int = 14
    atr_stop_mult: float = 2.0
    reward_risk: float = 2.0

    sweep_lookback: int = 20
    structure_lookback: int = 10
    retest_tolerance_atr: float = 0.5

    def __post_init__(self):
        if self.fast < 1 or self.slow <= self.fast:
            raise ValueError("Require 1 <= fast < slow")
        if self.trend_period is not None and self.trend_period <= self.slow:
            raise ValueError("trend_period must exceed slow")
        if not 0 <= self.rsi_low < self.rsi_high <= 100:
            raise ValueError("Invalid RSI band")
        if self.vol_period < 1 or self.atr_period < 1:
            raise ValueError("Indicator periods must be positive")
        if self.min_vol_ratio < 0:
            raise ValueError("min_vol_ratio cannot be negative")
        if self.atr_stop_mult <= 0 or self.reward_risk <= 0:
            raise ValueError("Stop multiplier and reward/risk must be positive")
        if self.sweep_lookback < 3 or self.structure_lookback < 3:
            raise ValueError("Lookback periods must be at least 3")


DEFAULT_CONFIG = SignalConfig()


def _as_numeric(df):
    df = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_indicators(df, cfg=DEFAULT_CONFIG):
    """Add EMA, RSI, volume ratio, ATR and setup features."""
    if df is None or df.empty:
        raise ValueError("Empty candle data")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    df = _as_numeric(df)

    if df[list(required)].isna().any().any():
        # Preserve rows; individual signals will reject incomplete candles.
        pass

    close = df["close"]

    df["ema_fast"] = close.ewm(
        span=cfg.fast, adjust=False, min_periods=cfg.fast
    ).mean()
    df["ema_slow"] = close.ewm(
        span=cfg.slow, adjust=False, min_periods=cfg.slow
    ).mean()

    if cfg.trend_period is not None:
        df["ema_trend"] = close.ewm(
            span=cfg.trend_period,
            adjust=False,
            min_periods=cfg.trend_period,
        ).mean()
    else:
        df["ema_trend"] = float("nan")

    # Wilder RSI, including explicit handling for flat prices.
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / cfg.rsi_period,
        min_periods=cfg.rsi_period,
        adjust=False,
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / cfg.rsi_period,
        min_periods=cfg.rsi_period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    df["rsi"] = rsi

    avg_vol = df["volume"].rolling(cfg.vol_period).mean()
    df["vol_ratio"] = df["volume"] / avg_vol.replace(0, float("nan"))

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / cfg.atr_period,
        min_periods=cfg.atr_period,
        adjust=False,
    ).mean()

    # Use previous candles only for reference highs and lows.
    # This avoids using the current candle to define its own sweep level.
    df["prior_swing_low"] = (
        df["low"].shift(1).rolling(cfg.sweep_lookback).min()
    )
    df["prior_swing_high"] = (
        df["high"].shift(1).rolling(cfg.structure_lookback).max()
    )

    df["liquidity_sweep"] = (
        (df["low"] < df["prior_swing_low"])
        & (close > df["prior_swing_low"])
    )

    df["structure_break"] = close > df["prior_swing_high"]

    # Approximate retest: price revisits the prior structure level
    # within an ATR tolerance and closes back above it.
    tolerance = cfg.retest_tolerance_atr * df["atr"]
    df["retest"] = (
        (df["low"] <= df["prior_swing_high"] + tolerance)
        & (close > df["prior_swing_high"])
    )

    candle_range = (df["high"] - df["low"]).replace(0, float("nan"))
    body = (close - df["open"]).abs()
    df["bullish_engulfing"] = (
        (close > df["open"])
        & (df["close"].shift(1) < df["open"].shift(1))
        & (df["open"] <= df["close"].shift(1))
        & (close >= df["open"].shift(1))
    )
    df["strong_bullish_close"] = (
        (close > df["open"])
        & ((close - df["low"]) / candle_range >= 0.70)
        & (body / candle_range >= 0.40)
    )

    # Momentum trend filter.
    df["trend_bullish"] = (
        (df["ema_fast"] > df["ema_slow"])
        & (
            (close > df["ema_trend"])
            if cfg.trend_period is not None
            else True
        )
    )

    return df


def _valid(row, names):
    return all(name in row.index and pd.notna(row[name]) for name in names)


def _bullish_context(df, i, cfg):
    """Check trend and entry confirmations at candle i."""
    if i < 1 or i >= len(df):
        return False, []

    row = df.iloc[i]
    required = [
        "open", "high", "low", "close", "volume",
        "ema_fast", "ema_slow", "rsi", "vol_ratio", "atr",
        "prior_swing_low", "prior_swing_high",
    ]

    if cfg.trend_period is not None:
        required.append("ema_trend")

    if not _valid(row, required):
        return False, []

    close = float(row["close"])
    atr = float(row["atr"])
    rsi = float(row["rsi"])
    volume_ratio = float(row["vol_ratio"])

    if close <= 0 or atr <= 0:
        return False, []

    if row["ema_fast"] <= row["ema_slow"]:
        return False, []

    if cfg.trend_period is not None and close <= row["ema_trend"]:
        return False, []

    if not cfg.rsi_low <= rsi <= cfg.rsi_high:
        return False, []

    if cfg.min_vol_ratio > 0 and volume_ratio < cfg.min_vol_ratio:
        return False, []

    reasons = ["bullish EMA trend", "RSI confirmation", "volume confirmation"]

    # Avoid buying a candle that has already moved too far from its open.
    if (float(row["high"]) - float(row["low"])) > 3.0 * atr:
        return False, []

    swept = bool(row["liquidity_sweep"])
    broken = bool(row["structure_break"])
    retested = bool(row["retest"])
    engulfing = bool(row["bullish_engulfing"])
    strong_close = bool(row["strong_bullish_close"])

    # Require a sweep followed by a bullish structure break,
    # or a structure break with retest / bullish candle confirmation.
    recent_sweep = (
        df["liquidity_sweep"]
        .iloc[max(0, i - 3):i + 1]
        .fillna(False)
        .any()
    )

    sweep_setup = recent_sweep and (broken or strong_close or engulfing)
    breakout_setup = broken and (retested or engulfing or strong_close)

    if not (sweep_setup or breakout_setup):
        return False, []

    if recent_sweep:
        reasons.append("liquidity sweep")
    if broken:
        reasons.append("bullish structure break")
    if retested:
        reasons.append("retest confirmation")
    if engulfing:
        reasons.append("bullish engulfing")
    elif strong_close:
        reasons.append("strong bullish candle")

    return True, reasons


def signal_at(df, i, cfg=DEFAULT_CONFIG):
    """Return BUY, SELL or HOLD using candles up to row i only."""
    if df is None or i < 1 or i >= len(df):
        return HOLD

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if not _valid(row, ["close", "ema_fast", "ema_slow", "rsi"]):
        return HOLD

    if not _valid(prev, ["ema_fast", "ema_slow"]):
        return HOLD

    crossed_down = (
        prev["ema_fast"] >= prev["ema_slow"]
        and row["ema_fast"] < row["ema_slow"]
    )

    if crossed_down:
        return SELL

    valid, _ = _bullish_context(df, i, cfg)
    return BUY if valid else HOLD


def _trend_is_bullish(htf, cfg):
    """Check the latest supplied higher-timeframe candle."""
    if htf is None or htf.empty:
        return False

    try:
        data = add_indicators(htf, cfg)
    except (ValueError, KeyError, TypeError):
        return False

    row = data.iloc[-1]
    required = ["close", "ema_fast", "ema_slow"]

    if cfg.trend_period is not None:
        required.append("ema_trend")

    if not _valid(row, required):
        return False

    if row["ema_fast"] <= row["ema_slow"]:
        return False

    return cfg.trend_period is None or row["close"] > row["ema_trend"]


def latest_signal(
    df,
    cfg=DEFAULT_CONFIG,
    df_1h=None,
    df_4h=None,
    risk_amount=2.0,
):
    """Summarize the latest entry candle and optional HTF confirmations.

    df must contain CLOSED entry-timeframe candles.
    df_1h and df_4h are optional higher-timeframe candles.
    risk_amount is a planned cash risk, not a guaranteed maximum loss.
    """
    if df is None or df.empty:
        return {
            "signal": HOLD,
            "price": None,
            "rsi": None,
            "vol_ratio": None,
            "suggested_stop": None,
            "suggested_target": None,
            "position_size": None,
            "reasons": ["no candle data"],
        }

    data = add_indicators(df, cfg)
    i = len(data) - 1
    row = data.iloc[i]

    signal = signal_at(data, i, cfg)
    valid, reasons = _bullish_context(data, i, cfg)

    # Optional higher-timeframe filters. If supplied, both must agree.
    if signal == BUY and df_1h is not None and not _trend_is_bullish(df_1h, cfg):
        signal = HOLD
        reasons.append("1H trend not confirmed")

    if signal == BUY and df_4h is not None and not _trend_is_bullish(df_4h, cfg):
        signal = HOLD
        reasons.append("4H trend not confirmed")

    price = float(row["close"])
    atr = row.get("atr")
    stop = None
    target = None
    position_size = None

    if pd.notna(atr) and float(atr) > 0:
        stop = price - cfg.atr_stop_mult * float(atr)
        risk_per_unit = price - stop

        if stop > 0 and risk_per_unit > 0:
            target = price + cfg.reward_risk * risk_per_unit
            if risk_amount > 0:
                position_size = risk_amount / risk_per_unit

    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else None
    vol = float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else None

    return {
        "signal": signal,
        "price": price,
        "rsi": rsi,
        "vol_ratio": vol,
        "suggested_stop": stop,
        "suggested_target": target,
        "position_size": position_size,
        "reasons": reasons if valid else [],
    }


def setup_score(df, cfg=DEFAULT_CONFIG):
    """Explainable 0-100 setup ranking; never a win probability."""
    if df is None or len(df) < 30:
        return {
            "score": 0,
            "reasons": ["not enough data"],
            "flags": [],
            "ret_24h": None,
        }

    data = add_indicators(df, cfg)
    row = data.iloc[-1]
    score = 0
    reasons = []
    flags = []

    required = ["close", "ema_fast", "ema_slow", "rsi", "vol_ratio", "atr"]
    if not _valid(row, required):
        return {
            "score": 0,
            "reasons": ["indicators not ready"],
            "flags": [],
            "ret_24h": None,
        }

    close = float(row["close"])
    if close <= 0:
        return {
            "score": 0,
            "reasons": ["invalid price"],
            "flags": [],
            "ret_24h": None,
        }

    # Trend: 25 points.
    if row["ema_fast"] > row["ema_slow"]:
        score += 10
        reasons.append("fast EMA above slow EMA")

    if cfg.trend_period is None:
        score += 15
    elif pd.notna(row["ema_trend"]) and close > row["ema_trend"]:
        score += 15
        reasons.append("price above 200-period trend EMA")

    # RSI: 20 points.
    rsi = float(row["rsi"])
    if 52 <= rsi <= 65:
        score += 20
        reasons.append("RSI in preferred momentum zone")
    elif 48 <= rsi < 52 or 65 < rsi <= 70:
        score += 10

    if rsi > 75:
        score -= 15
        flags.append(f"overbought RSI ({rsi:.1f})")

    # Volume: 20 points.
    vol = float(data["vol_ratio"].iloc[-3:].mean())
    if pd.notna(vol):
        if vol >= 2.0:
            score += 20
            reasons.append(f"volume {vol:.1f}x average")
        elif vol >= 1.5:
            score += 14
        elif vol >= cfg.min_vol_ratio:
            score += 8

    # Structure / breakout: 15 points.
    if bool(row.get("liquidity_sweep", False)):
        score += 8
        reasons.append("liquidity sweep")
    if bool(row.get("structure_break", False)):
        score += 7
        reasons.append("structure break")
    elif bool(row.get("retest", False)):
        score += 5
        reasons.append("retest")

    # Approximate 24-hour return only if candle spacing is known.
    ret24 = None
    if len(data) >= 25:
        old_price = float(data["close"].iloc[-25])
        if old_price > 0:
            ret24 = close / old_price - 1
            if 0.01 <= ret24 <= 0.15:
                score += 10
                reasons.append(f"+{ret24 * 100:.1f}% over 24 prior bars")
            elif 0 <= ret24 < 0.01:
                score += 5
            if ret24 > 0.25:
                score -= 20
                flags.append("price may already be overextended")

    # Volatility: 10 points.
    atr_pct = float(row["atr"]) / close
    if 0.005 <= atr_pct <= 0.04:
        score += 10
    elif atr_pct <= 0.06:
        score += 5
    else:
        flags.append(f"high volatility (ATR {atr_pct * 100:.1f}%)")

    score = max(0, min(100, score))
    return {
        "score": score,
        "reasons": reasons,
        "flags": flags,
        "ret_24h": ret24,
    }


if __name__ == "__main__":
    import random

    random.seed(3)
    price = 100.0
    rows = []

    for _ in range(1200):
        open_ = price
        price *= 1 + random.gauss(0.0002, 0.008)
        high = max(open_, price) * 1.003
        low = min(open_, price) * 0.997
        rows.append([
            open_, high, low, price, random.uniform(50, 150)
        ])

    columns = ["open", "high", "low", "close", "volume"]
    candles = pd.DataFrame(rows, columns=columns)
    indicators = add_indicators(candles)

    counts = {BUY: 0, SELL: 0, HOLD: 0}
    for i in range(len(indicators)):
        counts[signal_at(indicators, i)] += 1

    print("Signal counts:", counts)
    print("Latest:", latest_signal(candles))
    print("Score:", setup_score(candles))
