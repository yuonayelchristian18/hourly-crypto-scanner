"""signals.py - strategy layer. All strategy logic lives in gwvsa.py (the course strategies).

analyze(df) -> dict(trend, close, signal)  where signal is a gwvsa.Signal fired on the
last closed candle (long-only unless allow_short) or None.
"""
from gwvsa import Params, _run

TREND_NAME = {1: "up", -1: "down", 0: "mixed"}


def make_params(mode: str = "original", **kw) -> Params:
    """mode: 'original' (course default, very selective) | 'more' | 'max'."""
    return Params(mode=mode, **kw)


def analyze(df, params: Params = None) -> dict:
    p = params or make_params()
    sigs, arrays, _ = _run(df, p)
    last = len(df) - 1
    sig = None
    for s in reversed(sigs):
        if s.idx == last and s.tradable:
            sig = s
            break
        if s.idx < last:
            break
    return {"trend": TREND_NAME[int(arrays["trend"][-1])],
            "close": float(df["close"].iloc[-1]),
            "signal": sig}