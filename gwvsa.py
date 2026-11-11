"""
gwvsa.py - Gann / Wyckoff / VSA strategies from MAIN_TRADING_COURSE.pdf
(GWVSA v7.0.1 "Visible Volume", plain-English edition), for the crypto scanner.

WHAT IS IN HERE
  * 13 named VSA patterns (7 bullish, 6 bearish) with the exact thresholds from
    course sections 11.5 / 11.7 / 11.8
  * confirmed-swing structure (HH/HL/LH/LL trend), Gann eighths, Fibonacci levels,
    50% reaction, completed-higher-timeframe check, background memory, phase proxy
  * the 13-point setup score, 15-point final score, Grade A/B/C
  * stored candidate: GREEN/RED trigger, fixed stop (0.5 ATR buffer), 5-bar expiry
  * entry = first close-confirmed breakout OR a qualified retest ("no chase")
  * checks: stop distance 0.25-4 ATR, participation, room >= 1R
  * 2R target, optional 1R partial, Gann-flip and opposite-signal exits
  * position sizing with the 0.05%/side fee reserve
  * frequency modes: "original" / "more" / "max"

HONEST LIMITS
  The PDF describes the Pine script in prose; it does not include the source code.
  Pieces the PDF gives exactly are used as written. Pieces it leaves open are my
  reconstruction and are marked  # ASSUMPTION  so you can find and change them.
  The course itself says it does not establish a profitable strategy or win rate.
  Run walkforward.py on this before trusting any signal.

QUICK USE
    import gwvsa
    sigs = gwvsa.scan(df)                    # df: open/high/low/close/volume, oldest first
    s = gwvsa.latest_signal(df)              # signal that fired on the last closed bar, or None
    trades = gwvsa.backtest(df)              # simulated trades (entry = next bar open)
    print(gwvsa.summarize(trades))
    qty = gwvsa.position_size(entry, stop, cash_risk=2.0, equity=200)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
import pandas as pd

PATTERNS_BULL = [
    "Shakeout", "Selling Climax", "Stopping Volume", "Possible Absorption",
    "Spring / Trap-Down", "No Supply", "Test in a Rising Market",
]
PATTERNS_BEAR = [
    "Upthrust", "Buying Climax", "Effort-Result Warning",
    "Trap-Up / Bull Trap", "No Demand", "Test in a Falling Market",
]
QUIET = {"No Supply", "Test in a Rising Market", "No Demand", "Test in a Falling Market"}


# --------------------------------------------------------------------------- #
# Settings (course 11.5 and 11.6 defaults)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # measurements
    vol_len: int = 20            # RMA, includes current bar
    rng_len: int = 20            # SMA, includes current bar
    low_vol: float = 0.80
    high_vol: float = 1.20
    ultra_vol: float = 1.80
    narrow: float = 0.80
    wide: float = 1.20
    bg_memory: int = 30
    phase_range: int = 60
    # scoring
    setup_min: int = 7
    final_min: int = 9
    # entry
    breakout_vol_min: float = 1.00
    retest_vol_max: float = 1.10
    retest_touch_atr: float = 0.15
    expiry_bars: int = 5
    # risk
    atr_len: int = 14
    stop_buffer_atr: float = 0.50
    stop_min_atr: float = 0.25
    stop_max_atr: float = 4.00
    target_r: float = 2.0
    room_min_r: float = 1.0
    fee_per_side: float = 0.0005
    upthrust_vol_min: float = 1.20
    # switches
    use_supplemental: bool = True    # Spring, Trap-Up, No Supply, No Demand
    allow_reversal: bool = False     # balanced default = confirmed trend only
    allow_short: bool = False        # spot crypto: long only. Shorts still detected (for exits)
    mode: str = "original"           # "original" | "more" | "max"
    htf_rule: str = "4h"             # needs a DatetimeIndex; otherwise HTF is skipped
    # mandatory gates (original mode keeps them all on)
    gate_htf: bool = True
    gate_phase: bool = True
    gate_bg: bool = True
    gate_loc: bool = True
    gate_part: bool = True
    gate_room: bool = True
    # ASSUMPTIONS (course does not give numbers)
    loc_atr: float = 1.0             # ASSUMPTION: "near swing low/high" = within 1 ATR
    recent_ext_bars: int = 10        # ASSUMPTION: "recent low/high" lookback
    room_lookback: int = 50          # ASSUMPTION: room reference lookback


def _apply_mode(p: Params) -> Params:
    """Course 6.8: More trades relaxes HTF/phase/room. Max trades also relaxes
    background/location/participation and lowers thresholds to 6/7."""
    if p.mode == "more":
        return replace(p, gate_htf=False, gate_phase=False, gate_room=False)
    if p.mode == "max":
        return replace(p, gate_htf=False, gate_phase=False, gate_room=False,
                       gate_bg=False, gate_loc=False, gate_part=False,
                       setup_min=6, final_min=7)
    return p


@dataclass
class Signal:
    idx: int
    time: object
    direction: str          # "long" | "short"
    pattern: str
    entry_type: str         # "breakout" | "retest"
    setup_score: int        # 0-13
    final_score: int        # 0-15
    grade: str              # A / B / C from setup score
    trigger: float
    entry_ref: float        # close of the event bar (real fill = next open)
    stop: float
    target_1r: float
    target_2r: float
    atr: float
    tradable: bool          # False for shorts when allow_short=False
    scanner_score: int      # final score mapped to 0-100 (final/15*100)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def prepare(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    d["rng"] = d["high"] - d["low"]
    d["vol_rma"] = _rma(d["volume"], p.vol_len)
    d["rng_sma"] = d["rng"].rolling(p.rng_len, min_periods=p.rng_len).mean()
    d["vr"] = d["volume"] / d["vol_rma"]
    d["sr"] = d["rng"] / d["rng_sma"]
    d["cp"] = ((d["close"] - d["low"]) / d["rng"]).where(d["rng"] > 0, 0.5)
    pc = d["close"].shift(1)
    d["down"] = d["close"] < pc
    d["up"] = d["close"] > pc
    d["inside"] = (d["high"] <= d["high"].shift(1)) & (d["low"] >= d["low"].shift(1))
    tr = pd.concat([d["rng"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = _rma(tr, p.atr_len)
    return d


# --------------------------------------------------------------------------- #
# Confirmed-swing engine (Gann one-bar swings)
# ASSUMPTION: the PDF names the "one-bar engine" but not its code. This is the
# classic Gann one-bar swing: an up-swing turns down on a bar with a lower high
# AND lower low (confirming the running high); the mirror turns it up. Inside
# bars are ignored. Outside bars resolve by close vs open.
# A pivot only becomes usable on its confirmation bar (no hindsight).
# --------------------------------------------------------------------------- #
def _swings(o, h, l, c):
    n = len(c)
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    psh = np.full(n, np.nan); psl = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    hh_n = np.zeros(n, dtype=int); ll_n = np.zeros(n, dtype=int)
    highs: List[float] = []; lows: List[float] = []
    direction = 0
    run_hi = h[0]; run_lo = l[0]
    hh = 0; ll = 0
    for i in range(1, n):
        up_bar = h[i] > h[i - 1] and l[i] > l[i - 1]
        dn_bar = h[i] < h[i - 1] and l[i] < l[i - 1]
        if h[i] > h[i - 1] and l[i] < l[i - 1]:          # outside bar
            up_bar, dn_bar = c[i] >= o[i], c[i] < o[i]
        if direction == 0:
            if up_bar:
                direction = 1; run_hi = h[i]; lows.append(float(np.min(l[:i])))
            elif dn_bar:
                direction = -1; run_lo = l[i]; highs.append(float(np.max(h[:i])))
        elif direction == 1:
            if h[i] > run_hi:
                run_hi = h[i]
            if dn_bar:
                highs.append(float(run_hi)); direction = -1; run_lo = l[i]
                if len(highs) >= 2:
                    hh = hh + 1 if highs[-1] > highs[-2] else 0
        else:
            if l[i] < run_lo:
                run_lo = l[i]
            if up_bar:
                lows.append(float(run_lo)); direction = 1; run_hi = h[i]
                if len(lows) >= 2:
                    ll = ll + 1 if lows[-1] < lows[-2] else 0
        if highs:
            sh[i] = highs[-1]
            if len(highs) >= 2: psh[i] = highs[-2]
        if lows:
            sl[i] = lows[-1]
            if len(lows) >= 2: psl[i] = lows[-2]
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                trend[i] = 1
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                trend[i] = -1
        hh_n[i] = hh; ll_n[i] = ll
    return dict(sh=sh, sl=sl, psh=psh, psl=psl, trend=trend, hh_n=hh_n, ll_n=ll_n)


def _htf_trend(df: pd.DataFrame, p: Params) -> Optional[np.ndarray]:
    """Trend of a COMPLETED higher-timeframe bar (course 2.7). Needs DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        agg = df.resample(p.htf_rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if len(agg) < 10:
            return None
        t = _swings(agg["open"].values, agg["high"].values, agg["low"].values, agg["close"].values)["trend"]
        s = pd.Series(t, index=agg.index).shift(1).fillna(0)   # only completed bars
        out = s.reindex(df.index, method="ffill").fillna(0).astype(int).values
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main engine
# --------------------------------------------------------------------------- #
def _run(df: pd.DataFrame, params: Optional[Params] = None):
    p = _apply_mode(params or Params())
    d = prepare(df, p)
    n = len(d)
    idx_time = df.index
    o = d["open"].values; h = d["high"].values; l = d["low"].values; c = d["close"].values
    v = d["volume"].values
    vr = d["vr"].values; sr = d["sr"].values; cp = d["cp"].values; atr = d["atr"].values
    down = d["down"].values; up = d["up"].values; inside = d["inside"].values

    sw = _swings(o, h, l, c)
    SH, SL, TREND = sw["sh"], sw["sl"], sw["trend"]
    htf = _htf_trend(df, p) if p.gate_htf or True else None

    sigs: List[Signal] = []
    last_strength = -10**9
    last_weak = -10**9
    cand = None
    start = max(p.rng_len, p.atr_len, p.phase_range // 3) + 2

    def bg_bull(i):  # supportive strength recent and more recent than opposing weakness
        alive = i - last_strength <= p.bg_memory
        opp_alive = i - last_weak <= p.bg_memory
        return alive and (not opp_alive or last_strength > last_weak)

    def bg_bear(i):
        alive = i - last_weak <= p.bg_memory
        opp_alive = i - last_strength <= p.bg_memory
        return alive and (not opp_alive or last_weak > last_strength)

    for i in range(start, n):
        if not (np.isfinite(vr[i]) and np.isfinite(sr[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            continue
        tol = p.retest_touch_atr * atr[i]

        # ---------- 1. manage the stored candidate (events happen AFTER the setup bar) ----------
        if cand is not None:
            s = cand["s"]
            if cand["broke"] is None and i - cand["i0"] > p.expiry_bars:
                cand = None
            elif cand["broke"] is not None and i - cand["broke"] > p.expiry_bars:
                cand = None
            if cand is not None and ((s == 1 and l[i] <= cand["stop"]) or (s == -1 and h[i] >= cand["stop"])):
                cand = None                                     # stop touched first -> cancelled
            if cand is not None:
                trig = cand["trig"]
                fire = None
                if cand["broke"] is None:
                    if (s == 1 and c[i] > trig) or (s == -1 and c[i] < trig):
                        cand["vol_pt"] = 1 if vr[i] >= p.breakout_vol_min else 0
                        cand["broke"] = i
                        part_ok = cand["vol_pt"] == 1 or not p.gate_part
                        if part_ok:
                            fire = ("breakout", cand["setup"] + cand["vol_pt"])
                elif i > cand["broke"]:
                    touch = (l[i] <= trig + tol) if s == 1 else (h[i] >= trig - tol)
                    hold = (c[i] > trig) if s == 1 else (c[i] < trig)
                    cpok = (cp[i] >= 0.55) if s == 1 else (cp[i] <= 0.45)
                    if touch and hold and cpok and vr[i] <= p.retest_vol_max:
                        fire = ("retest", cand["setup"] + cand["vol_pt"] + 1)
                if fire is not None:
                    etype, final = fire
                    entry = c[i]
                    dist = abs(entry - cand["stop"])
                    stop_ok = dist > 0 and p.stop_min_atr <= dist / atr[i] <= p.stop_max_atr
                    room_ok = True
                    if p.gate_room and stop_ok:
                        lo_i = max(0, i - p.room_lookback)
                        if s == 1:
                            ref = np.max(h[lo_i:i]); room_ok = ref <= entry or (ref - entry) >= p.room_min_r * dist
                        else:
                            ref = np.min(l[lo_i:i]); room_ok = ref >= entry or (entry - ref) >= p.room_min_r * dist
                    if final >= p.final_min and stop_ok and room_ok:
                        sigs.append(Signal(
                            idx=i, time=idx_time[i], direction="long" if s == 1 else "short",
                            pattern=cand["pattern"], entry_type=etype, setup_score=cand["setup"],
                            final_score=int(final), grade=_grade(cand["setup"]), trigger=float(trig),
                            entry_ref=float(entry), stop=float(cand["stop"]),
                            target_1r=float(entry + s * dist), target_2r=float(entry + s * p.target_r * dist),
                            atr=float(atr[i]), tradable=(s == 1 or p.allow_short),
                            scanner_score=int(round(final / 15 * 100))))
                        cand = None

        # ---------- 2. detect raw patterns on this bar ----------
        sh_i, sl_i = SH[i], SL[i]
        trend_i = TREND[i]
        lo_w = max(0, i - p.phase_range + 1)
        w_hi, w_lo = np.max(h[lo_w:i + 1]), np.min(l[lo_w:i + 1])
        rpos = (c[i] - w_lo) / (w_hi - w_lo) if w_hi > w_lo else 0.5
        near_lo = np.isfinite(sl_i) and l[i] <= sl_i + p.loc_atr * atr[i]
        near_hi = np.isfinite(sh_i) and h[i] >= sh_i - p.loc_atr * atr[i]
        r0 = max(0, i - p.recent_ext_bars)
        rec_lo = l[i] <= np.min(l[r0:i]) if i > r0 else False
        rec_hi = h[i] >= np.max(h[r0:i]) if i > r0 else False
        loc_bull = bool(near_lo or rpos <= 0.5)
        loc_bear = bool(near_hi or rpos >= 0.5)
        bgb, bgs = bg_bull(i), bg_bear(i)
        high = vr[i] >= p.high_vol; low = vr[i] <= p.low_vol
        wide = sr[i] >= p.wide; narrow = sr[i] <= p.narrow
        vdec = v[i] < v[i - 1]
        x = cp[i]

        bull = []; bear = []
        if down[i] and high and wide and x >= 0.70 and loc_bull: bull.append("Shakeout")
        if down[i] and high and wide and 0.50 <= x <= 0.85 and (trend_i == -1 or rec_lo): bull.append("Selling Climax")
        if down[i] and high and wide and 0.35 <= x <= 0.65 and loc_bull: bull.append("Stopping Volume")
        if down[i] and high and narrow and 0.55 <= x <= 0.90 and (trend_i == -1 or rec_lo): bull.append("Possible Absorption")
        if p.use_supplemental and np.isfinite(sl_i) and l[i] < sl_i and c[i] > sl_i and x >= 0.55 and vr[i] >= 1.0:
            bull.append("Spring / Trap-Down")
        if p.use_supplemental and (down[i] or inside[i]) and low and vdec and narrow and x >= 0.45 \
                and trend_i == 1 and near_lo and bgb:
            bull.append("No Supply")
        if down[i] and low and narrow and 0.15 <= x <= 0.55 and trend_i == 1 and near_lo and bgb:
            bull.append("Test in a Rising Market")

        up_min = max(p.upthrust_vol_min, p.high_vol)
        if up[i] and wide and x <= 0.30 and vr[i] >= up_min and loc_bear: bear.append("Upthrust")
        if up[i] and high and wide and 0.15 <= x <= 0.50 and (trend_i == 1 or rec_hi): bear.append("Buying Climax")
        if up[i] and high and narrow and 0.35 <= x <= 0.65 and (trend_i == 1 or rec_hi): bear.append("Effort-Result Warning")
        if p.use_supplemental and np.isfinite(sh_i) and h[i] > sh_i and c[i] < sh_i and x <= 0.45 and vr[i] >= 1.0:
            bear.append("Trap-Up / Bull Trap")
        if p.use_supplemental and (up[i] or inside[i]) and low and vdec and narrow and x <= 0.55 \
                and trend_i == -1 and near_hi and bgs:
            bear.append("No Demand")
        if up[i] and low and narrow and 0.45 <= x <= 0.85 and trend_i == -1 and near_hi and bgs:
            bear.append("Test in a Falling Market")

        bull_pat = bull[0] if bull else None          # ASSUMPTION: naming priority = list order
        bear_pat = bear[0] if bear else None

        # ---------- 3. build a candidate from the pattern ----------
        chosen = None
        if bull_pat and not bear_pat:
            chosen = (1, bull_pat)
        elif bear_pat and not bull_pat:
            chosen = (-1, bear_pat)
        if chosen is not None:
            s, pat = chosen
            cand_new = _build_candidate(
                s, pat, i, p, o, h, l, c, atr, trend_i, sh_i, sl_i, sw, rpos, bgb, bgs, htf,
                last_strength, last_weak)
            if cand_new is not None:
                if cand is None or cand_new["setup"] >= cand["setup"]:
                    cand = cand_new

        # ---------- 4. update background AFTER evaluating this bar ----------
        for nm in (bull_pat, bear_pat):
            if nm in QUIET or nm is None:
                continue
            if nm in PATTERNS_BULL: last_strength = i
            else: last_weak = i

    arrays = dict(trend=TREND, sh=SH, sl=SL, o=o, h=h, l=l, c=c)
    return sigs, arrays, d


def _grade(score: int) -> str:
    return "A" if score >= 10 else "B" if score >= 7 else "C" if score >= 4 else "-"


def _build_candidate(s, pat, i, p, o, h, l, c, atr, trend_i, sh_i, sl_i, sw, rpos, bgb, bgs, htf,
                     last_strength, last_weak):
    cont = trend_i == s
    if not cont and not p.allow_reversal:
        return None                                        # confirmed-trend-only (balanced default)
    if not cont and trend_i == 0:
        return None
    trig = sh_i if s == 1 else sl_i                        # GREEN / RED trigger = last confirmed swing top/bottom
    if not np.isfinite(trig):
        return None
    if (s == 1 and trig <= c[i]) or (s == -1 and trig >= c[i]):
        return None                                        # trigger already crossed -> nothing later to wait for
    # stop (course 11.6)
    if cont:
        ref = min(l[i], sl_i) if s == 1 and np.isfinite(sl_i) else \
              max(h[i], sh_i) if s == -1 and np.isfinite(sh_i) else (l[i] if s == 1 else h[i])
    else:
        ref = l[i] if s == 1 else h[i]
    stop = ref - s * p.stop_buffer_atr * atr[i]

    tol = p.retest_touch_atr * atr[i]
    # --- 13-point setup score ---
    pts = 2 if cont else 1                                             # structure
    pts += 1 if pat in QUIET else 2                                    # event / quiet test
    supportive = bgb if s == 1 else bgs
    prior_event = (i - 1 - (last_strength if s == 1 else last_weak)) >= 0 and supportive
    cur_event = pat not in QUIET
    pts += 2 if prior_event else (1 if cur_event else 0)               # background
    near_ext = False
    lo_w = max(0, i - p.phase_range + 1)
    if s == 1:
        near_ext = (np.isfinite(sl_i) and l[i] <= sl_i + p.loc_atr * atr[i]) or l[i] <= np.min(l[lo_w:i + 1]) + p.loc_atr * atr[i]
    else:
        near_ext = (np.isfinite(sh_i) and h[i] >= sh_i - p.loc_atr * atr[i]) or h[i] >= np.max(h[lo_w:i + 1]) - p.loc_atr * atr[i]
    loc_pt = 1 if near_ext else 0
    pts += loc_pt
    gann_pt = fib_pt = half_pt = 0
    if np.isfinite(sh_i) and np.isfinite(sl_i) and sh_i > sl_i:
        rng = sh_i - sl_i
        ext = l[i] if s == 1 else h[i]
        held = (lambda lv: c[i] >= lv - tol) if s == 1 else (lambda lv: c[i] <= lv + tol)
        gl = [sl_i + k / 8.0 * rng for k in range(1, 8)]
        gann_pt = int(any(abs(ext - lv) <= tol and held(lv) for lv in gl))
        fl = [(sh_i - f * rng) if s == 1 else (sl_i + f * rng) for f in (0.382, 0.5, 0.618, 0.786)]
        fib_pt = int(any(abs(ext - lv) <= tol and held(lv) for lv in fl))
        mid = sl_i + 0.5 * rng
        half_pt = int((l[i] <= mid + tol and c[i] >= mid) if s == 1 else (h[i] >= mid - tol and c[i] <= mid))
    pts += gann_pt + fib_pt + half_pt
    htf_pt = int(htf is not None and htf[i] == s)
    pts += htf_pt
    sec = sw["hh_n"][i] if s == 1 else sw["ll_n"][i]
    sec_pt = int(cont and sec >= 2)                                    # ASSUMPTION: >=2 consecutive sections
    pts += sec_pt
    # two-bar engine is off by default -> 0 points

    # --- mandatory gates (original mode) ---
    if p.gate_htf and htf is not None and htf[i] != s:
        return None
    if p.gate_phase:
        if s == 1 and trend_i == 1 and bgs and rpos > 0.5: return None      # "Distribution"
        if s == -1 and trend_i == -1 and bgb and rpos < 0.5: return None    # "Accumulation"
    if p.gate_bg and ((s == 1 and bgs and not bgb) or (s == -1 and bgb and not bgs)):
        return None
    if p.gate_loc and not loc_pt:
        return None
    if pts < p.setup_min:
        return None
    return dict(s=s, pattern=pat, i0=i, trig=float(trig), stop=float(stop), setup=int(pts),
                broke=None, vol_pt=0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def scan(df: pd.DataFrame, params: Optional[Params] = None) -> List[Signal]:
    """All signals over the whole history (each uses only data up to its own bar)."""
    return _run(df, params)[0]


def latest_signal(df: pd.DataFrame, params: Optional[Params] = None) -> Optional[Signal]:
    """Tradable signal that fired on the LAST bar of df (call with closed candles only)."""
    sigs = scan(df, params)
    for s in reversed(sigs):
        if s.idx == len(df) - 1 and s.tradable:
            return s
        if s.idx < len(df) - 1:
            break
    return None


def label_patterns(df: pd.DataFrame, params: Optional[Params] = None) -> pd.DataFrame:
    """Debug helper: signals as a DataFrame."""
    return pd.DataFrame([s.__dict__ for s in scan(df, params)])


def position_size(entry: float, stop: float, cash_risk: float, fee_per_side: float = 0.0005,
                  equity: Optional[float] = None, exposure_cap: float = 1.0,
                  step: float = 0.0001, min_qty: float = 0.0001) -> float:
    """Course 11.6: risk/unit = |entry-stop| + entry*2*fee; cap by exposure; round DOWN."""
    rpu = abs(entry - stop) + entry * 2 * fee_per_side
    if rpu <= 0:
        return 0.0
    q = cash_risk / rpu
    if equity:
        q = min(q, equity * exposure_cap / entry)
    q = math.floor(q / step) * step
    return q if q >= min_qty else 0.0


def backtest(df: pd.DataFrame, params: Optional[Params] = None, cost_per_side: float = 0.0005,
             partial_1r: bool = False, gann_flip_exit: bool = True, opposite_exit: bool = True) -> pd.DataFrame:
    """Simulate trades. Entry = NEXT bar open after the signal bar. If a bar hits both
    stop and target the stop is assumed first. Costs in R: 2*cost_per_side*entry/risk."""
    p = _apply_mode(params or Params())
    sigs, A, _ = _run(df, p)
    o, h, l, c, trend = A["o"], A["h"], A["l"], A["c"], A["trend"]
    n = len(c)
    opp = {}
    for s in sigs:
        opp.setdefault(s.idx, set()).add(s.direction)
    rows = []
    busy_until = -1
    for sg in sigs:
        if not sg.tradable or sg.idx + 1 >= n or sg.idx <= busy_until:
            continue
        s = 1 if sg.direction == "long" else -1
        e = sg.idx + 1
        entry = o[e]
        risk = (entry - sg.stop) * s
        if risk <= 0:
            continue
        tgt = entry + s * p.target_r * risk
        t1 = entry + s * risk
        got_partial = False
        exit_px, reason, j_exit = c[-1], "eod", n - 1
        for j in range(e, n):
            gap_stop = (s == 1 and o[j] <= sg.stop) or (s == -1 and o[j] >= sg.stop)
            hit_stop = (s == 1 and l[j] <= sg.stop) or (s == -1 and h[j] >= sg.stop)
            if gap_stop:
                exit_px, reason, j_exit = o[j], "stop", j; break
            if hit_stop:
                exit_px, reason, j_exit = sg.stop, "stop", j; break
            if partial_1r and not got_partial and ((s == 1 and h[j] >= t1) or (s == -1 and l[j] <= t1)):
                got_partial = True
            if (s == 1 and h[j] >= tgt) or (s == -1 and l[j] <= tgt):
                exit_px, reason, j_exit = tgt, "target", j; break
            if j > e and gann_flip_exit and trend[j] == -s and trend[j - 1] != -s:
                exit_px, reason, j_exit = c[j], "gann_flip", j; break
            if j > e and opposite_exit and ("short" if s == 1 else "long") in opp.get(j, ()):
                exit_px, reason, j_exit = c[j], "opposite", j; break
        gross_main = (exit_px - entry) * s / risk
        gross = (0.5 * 1.0 + 0.5 * gross_main) if (partial_1r and got_partial) else gross_main
        cost = 2 * cost_per_side * entry / risk
        rows.append(dict(entry_time=df.index[e], exit_time=df.index[j_exit], direction=sg.direction,
                         pattern=sg.pattern, entry_type=sg.entry_type, setup_score=sg.setup_score,
                         final_score=sg.final_score, entry=entry, stop=sg.stop, exit=exit_px,
                         reason=reason, R_gross=gross, R_net=gross - cost, bars=j_exit - e))
        busy_until = j_exit
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return dict(trades=0, win_rate=0.0, avg_R=0.0, net_R=0.0, profit_factor=0.0)
    r = trades["R_net"]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=int(len(r)), win_rate=float((r > 0).mean()), avg_R=float(r.mean()),
                net_R=float(r.sum()), profit_factor=float(gp / gl) if gl > 0 else float("inf"))


# --------------------------------------------------------------------------- #
# Self-test:  python gwvsa.py
# --------------------------------------------------------------------------- #
def _synthetic(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 400) % 2 == 0, 0.0006, -0.0003)
    ret = drift + rng.normal(0, 0.008, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.006, n)) + 0.002
    high = np.maximum(open_, close) * (1 + spread * rng.random(n))
    low = np.minimum(open_, close) * (1 - spread * rng.random(n))
    vol = rng.lognormal(6, 0.5, n) * (1 + 12 * np.abs(ret))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame(dict(open=open_, high=high, low=low, close=close, volume=vol), index=idx)


if __name__ == "__main__":
    total = 0
    for seed in range(5):
        df = _synthetic(seed)
        for mode in ("original", "more", "max"):
            sg = scan(df, Params(mode=mode))
            tr = backtest(df, Params(mode=mode))
            print(f"seed {seed} mode {mode:8s} signals {len(sg):3d}  trades {len(tr):3d}  {summarize(tr)}")
            total += len(sg)
    # no-lookahead check: a signal computed on truncated data must equal the full-run one
    df = _synthetic(1)
    full = {s.idx: (s.direction, s.pattern, s.entry_type, s.final_score) for s in scan(df, Params(mode="max"))}
    bad = 0
    for k in sorted(full)[:15]:
        part = {s.idx: (s.direction, s.pattern, s.entry_type, s.final_score)
                for s in scan(df.iloc[:k + 1], Params(mode="max"))}
        if part.get(k) != full[k]:
            bad += 1
    print("lookahead mismatches:", bad, "| total signals seen:", total)
