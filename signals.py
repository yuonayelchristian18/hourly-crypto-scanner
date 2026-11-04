"""
signals.py — Kraken spot trading signals.

Closed-candle calculations only. No network access and no order placement.
BUY signals require a sequential liquidity sweep -> structure break -> retest,
bullish trend/momentum, volume, and (when supplied) higher-timeframe confirmation.
Scores rank setups; they are NOT win probabilities.
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
    sequence_window: int = 12
    retest_tolerance_atr: float = 0.35
    stop_buffer_atr: float = 0.15
    max_signal_bar_atr: float = 3.0

    def __post_init__(self):
        if self.fast < 1 or self.slow <= self.fast:
            raise ValueError("Require 1 <= fast < slow")
        if self.trend_period is not None and self.trend_period <= self.slow:
            raise ValueError("trend_period must exceed slow")
        if not 0 <= self.rsi_low < self.rsi_high <= 100:
            raise ValueError("Invalid RSI range")
        if min(self.rsi_period, self.vol_period, self.atr_period) < 1:
            raise ValueError("Indicator periods must be positive")
        if self.min_vol_ratio < 0 or self.atr_stop_mult <= 0:
            raise ValueError("Invalid volume or ATR setting")
        if self.reward_risk < 2:
            raise ValueError("This strategy requires at least 2:1 reward/risk")
        if min(self.sweep_lookback, self.structure_lookback, self.sequence_window) < 3:
            raise ValueError("Lookback values must be at least 3")


DEFAULT_CONFIG = SignalConfig()


def add_indicators(df, cfg=DEFAULT_CONFIG):
    if df is None or df.empty:
        raise ValueError("Empty candle data")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    data = df.copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.reset_index(drop=True)
    close = data["close"]

    data["ema_fast"] = close.ewm(span=cfg.fast, adjust=False, min_periods=cfg.fast).mean()
    data["ema_slow"] = close.ewm(span=cfg.slow, adjust=False, min_periods=cfg.slow).mean()
    data["ema_trend"] = (
        close.ewm(span=cfg.trend_period, adjust=False, min_periods=cfg.trend_period).mean()
        if cfg.trend_period else float("nan")
    )

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    data["rsi"] = rsi

    avg_volume = data["volume"].rolling(cfg.vol_period, min_periods=cfg.vol_period).mean()
    data["vol_ratio"] = data["volume"] / avg_volume.replace(0, float("nan"))

    prev_close = close.shift(1)
    tr = pd.concat([
        data["high"] - data["low"],
        (data["high"] - prev_close).abs(),
        (data["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    data["atr"] = tr.ewm(alpha=1 / cfg.atr_period, min_periods=cfg.atr_period, adjust=False).mean()

    data["prior_swing_low"] = data["low"].shift(1).rolling(cfg.sweep_lookback).min()
    data["prior_swing_high"] = data["high"].shift(1).rolling(cfg.structure_lookback).max()

    data["liquidity_sweep"] = (
        (data["low"] < data["prior_swing_low"]) &
        (data["close"] > data["prior_swing_low"])
    ).fillna(False)

    rng = (data["high"] - data["low"]).replace(0, float("nan"))
    body = (data["close"] - data["open"]).abs()
    data["bullish_engulfing"] = (
        (data["close"] > data["open"]) &
        (data["close"].shift(1) < data["open"].shift(1)) &
        (data["open"] <= data["close"].shift(1)) &
        (data["close"] >= data["open"].shift(1))
    ).fillna(False)
    data["strong_bullish_close"] = (
        (data["close"] > data["open"]) &
        (((data["close"] - data["low"]) / rng) >= 0.70) &
        ((body / rng) >= 0.40)
    ).fillna(False)
    return data


def _row_valid(row, columns):
    return all(c in row.index and pd.notna(row[c]) for c in columns)


def _sequential_setup(data, i, cfg):
    """Find sweep, then later structure break, then a later retest at candle i."""
    if i < 3:
        return None
    start = max(1, i - cfg.sequence_window)
    # Retest candle i must be later than both sweep and break candles.
    for sweep_i in range(start, i - 1):
        sweep = data.iloc[sweep_i]
        if not bool(sweep.get("liquidity_sweep", False)):
            continue
        sweep_low = float(sweep["low"])
        for break_i in range(sweep_i + 1, i):
            br = data.iloc[break_i]
            level = br.get("prior_swing_high")
            if pd.isna(level) or float(br["close"]) <= float(level):
                continue
            # Break candle must occur after the sweep. Retest is current candle only.
            current = data.iloc[i]
            atr = current.get("atr")
            if pd.isna(atr) or float(atr) <= 0:
                continue
            tolerance = cfg.retest_tolerance_atr * float(atr)
            touched = float(current["low"]) <= float(level) + tolerance
            held = float(current["close"]) > float(level)
            bullish = (
                float(current["close"]) > float(current["open"]) and
                (bool(current.get("bullish_engulfing", False)) or
                 bool(current.get("strong_bullish_close", False)))
            )
            if touched and held and bullish:
                return {
                    "sweep_index": sweep_i,
                    "break_index": break_i,
                    "level": float(level),
                    "sweep_low": sweep_low,
                }
    return None


def _bullish_context(data, i, cfg):
    if i < 1 or i >= len(data):
        return False, [], None
    row = data.iloc[i]
    required = ["open", "high", "low", "close", "ema_fast", "ema_slow",
                "rsi", "vol_ratio", "atr", "prior_swing_low", "prior_swing_high"]
    if cfg.trend_period:
        required.append("ema_trend")
    if not _row_valid(row, required):
        return False, [], None

    close, atr = float(row["close"]), float(row["atr"])
    if close <= 0 or atr <= 0:
        return False, [], None
    if float(row["ema_fast"]) <= float(row["ema_slow"]):
        return False, [], None
    if cfg.trend_period and close <= float(row["ema_trend"]):
        return False, [], None
    if not cfg.rsi_low <= float(row["rsi"]) <= cfg.rsi_high:
        return False, [], None
    if float(row["vol_ratio"]) < cfg.min_vol_ratio:
        return False, [], None
    if float(row["high"]) - float(row["low"]) > cfg.max_signal_bar_atr * atr:
        return False, [], None

    setup = _sequential_setup(data, i, cfg)
    if setup is None:
        return False, [], None
    reasons = [
        "EMA trend aligned", "RSI confirmation", "volume confirmation",
        "liquidity sweep", "later bullish structure break",
        "separate-candle retest and bullish confirmation"
    ]
    return True, reasons, setup


def signal_at(df, i, cfg=DEFAULT_CONFIG):
    if df is None or i < 1 or i >= len(df):
        return HOLD
    row, prev = df.iloc[i], df.iloc[i - 1]
    if not _row_valid(row, ["close", "ema_fast", "ema_slow"]) or not _row_valid(prev, ["ema_fast", "ema_slow"]):
        return HOLD
    if prev["ema_fast"] >= prev["ema_slow"] and row["ema_fast"] < row["ema_slow"]:
        return SELL
    valid, _, _ = _bullish_context(df, i, cfg)
    return BUY if valid else HOLD


def _htf_bullish(htf, cfg):
    if htf is None or htf.empty:
        return False
    try:
        data = add_indicators(htf, cfg)
    except (ValueError, TypeError, KeyError):
        return False
    row = data.iloc[-1]
    required = ["close", "ema_fast", "ema_slow"]
    if cfg.trend_period:
        required.append("ema_trend")
    if not _row_valid(row, required):
        return False
    return (
        float(row["ema_fast"]) > float(row["ema_slow"]) and
        (not cfg.trend_period or float(row["close"]) > float(row["ema_trend"]))
    )


def _ltf_5m_bullish(df_5m, cfg):
    if df_5m is None:
        return True, "5m confirmation not supplied"
    try:
        data = add_indicators(df_5m, cfg)
    except (ValueError, TypeError, KeyError):
        return False, "5m data invalid"
    if len(data) < cfg.trend_period + 5 if cfg.trend_period else len(data) < cfg.slow + 5:
        return False, "not enough 5m history"
    row = data.iloc[-1]
    if not _row_valid(row, ["close", "ema_fast", "ema_slow", "vol_ratio"]):
        return False, "5m indicators not ready"
    ok = (
        float(row["ema_fast"]) > float(row["ema_slow"]) and
        float(row["close"]) > float(row["ema_fast"]) and
        float(row["vol_ratio"]) >= 1.0
    )
    return ok, "5m momentum and volume confirmed" if ok else "5m confirmation failed"


def latest_signal(df, cfg=DEFAULT_CONFIG, df_1h=None, df_4h=None, df_5m=None, risk_amount=2.0):
    empty = {
        "signal": HOLD, "price": None, "suggested_stop": None,
        "suggested_target": None, "position_size": None,
        "reasons": ["no candle data"]
    }
    if df is None or df.empty:
        return empty
    data = add_indicators(df, cfg)
    i = len(data) - 1
    row = data.iloc[i]
    valid, reasons, setup = _bullish_context(data, i, cfg)
    signal = BUY if valid else HOLD

    if signal == BUY and df_1h is not None and not _htf_bullish(df_1h, cfg):
        signal, reasons = HOLD, reasons + ["1h trend confirmation failed"]
    if signal == BUY and df_4h is not None and not _htf_bullish(df_4h, cfg):
        signal, reasons = HOLD, reasons + ["4h trend confirmation failed"]
    if signal == BUY and df_5m is not None:
        ok, why = _ltf_5m_bullish(df_5m, cfg)
        if not ok:
            signal, reasons = HOLD, reasons + [why]
        else:
            reasons.append(why)

    price = float(row["close"]) if pd.notna(row["close"]) else None
    atr = float(row["atr"]) if pd.notna(row["atr"]) else None
    stop = target = size = None
    if price and atr and atr > 0 and setup:
        # Stop sits below both the sweep low and an ATR-based safety distance.
        stop = min(setup["sweep_low"], price - cfg.atr_stop_mult * atr) - cfg.stop_buffer_atr * atr
        if 0 < stop < price:
            distance = price - stop
            target = price + cfg.reward_risk * distance
            if risk_amount > 0:
                size = risk_amount / distance
        else:
            stop = None
    if signal == BUY and (stop is None or target is None):
        signal = HOLD
        reasons.append("valid stop/target could not be calculated")

    return {
        "signal": signal, "price": price, "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
        "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else None,
        "suggested_stop": stop, "suggested_target": target,
        "position_size": size, "sweep_low": setup["sweep_low"] if setup else None,
        "break_level": setup["level"] if setup else None,
        "reasons": reasons
    }


def setup_score(df, cfg=DEFAULT_CONFIG):
    if df is None or len(df) < max(30, (cfg.trend_period or cfg.slow) + 5):
        return {"score": 0.0, "reasons": ["not enough candle history"], "flags": []}
    data = add_indicators(df, cfg)
    row = data.iloc[-1]
    required = ["close", "ema_fast", "ema_slow", "rsi", "vol_ratio", "atr"]
    if cfg.trend_period:
        required.append("ema_trend")
    if not _row_valid(row, required):
        return {"score": 0.0, "reasons": ["indicators not ready"], "flags": []}

    score, reasons, flags = 0.0, [], []
    if row["ema_fast"] > row["ema_slow"]:
        score += 15; reasons.append("fast EMA above slow EMA")
    if not cfg.trend_period or row["close"] > row["ema_trend"]:
        score += 15; reasons.append("above trend EMA")
    rsi = float(row["rsi"])
    if 52 <= rsi <= 65:
        score += 15; reasons.append("RSI momentum zone")
    elif 50 <= rsi < 52 or 65 < rsi <= 68:
        score += 8
    if rsi > 72:
        flags.append("RSI overbought")
    vol = float(row["vol_ratio"])
    if vol >= 2:
        score += 15; reasons.append("strong volume")
    elif vol >= 1.5:
        score += 10
    elif vol >= cfg.min_vol_ratio:
        score += 5
    setup = _sequential_setup(data, len(data)-1, cfg)
    if setup:
        score += 25; reasons.append("sequential sweep-break-retest")
    atr_pct = float(row["atr"]) / float(row["close"])
    if 0.003 <= atr_pct <= 0.05:
        score += 10
    elif atr_pct > 0.08:
        flags.append("extreme ATR volatility")
    if len(data) >= 25:
        old = float(data["close"].iloc[-25])
        ret = float(row["close"]) / old - 1 if old > 0 else 0
        if 0 <= ret <= 0.15:
            score += 10
        elif ret > 0.25:
            flags.append("possible overextension")
    return {"score": round(min(score, 100), 2), "reasons": reasons, "flags": flags}


if __name__ == "__main__":
    print("signals.py loaded successfully")