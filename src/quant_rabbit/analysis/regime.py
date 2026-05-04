"""4-state regime classifier — Hurst + ADX + Choppiness + ATR-percentile.

Per `docs/research/03-quant-ensemble-regime.md` §1, a defensible 4-state
regime gate without a full HMM is:

    Hurst(rolling 200 bars) + Choppiness(14) + ADX(14) + HV percentile (1y)

Each indicator is cheap, each is robust on FX OHLC, and the ensemble vote
catches the failure mode where (e.g.) ADX>25 alone calls "trend" during a
choppy whipsaw.

Output states:
    TREND_STRONG       — H>0.55 AND ADX>25 AND Chop<38.2
    TREND_WEAK         — partial agreement (Hurst trending OR ADX>25, but
                         not all three)
    RANGE              — H<0.45 AND ADX<20 AND Chop>61.8 AND ATR_pct<40
    BREAKOUT_PENDING   — ATR percentile in bottom 20% AND ADX/Chop neutral
                         (squeeze waiting for expansion)
    TRANSITION         — anything else
    UNKNOWN            — insufficient data for any of the four indicators

Pure-stdlib, no numpy/pandas. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence


# ---------------------------------------------------------------------------
# Hurst regime thresholds — per AGENT_CONTRACT §3.5 (a)/(b)/(c)
# ---------------------------------------------------------------------------
#
# HURST_TREND_THRESHOLD = 0.55, HURST_RANGE_THRESHOLD = 0.45
#   (a) Market reality: a true random walk has Hurst = 0.5. Estimators on
#       finite samples (R/S, DFA) carry an upward small-sample bias, so the
#       0.55/0.45 split (rather than 0.51/0.49) corrects for that and is the
#       split recommended by Macrosynergy and the source research brief.
#   (b) Constant rather than market-derived: the bias correction depends on
#       window length and estimator family, not on the current market. For
#       DFA at window=200 the 0.55/0.45 split is the documented practice.
#   (c) Replace via: pass alternative thresholds to `classify_regime` if a
#       longer DFA window (e.g. 500+) is used, where the bias correction
#       narrows; or move to 0.50/0.50 if validated on the actual estimator.
HURST_TREND_THRESHOLD: float = 0.55
HURST_RANGE_THRESHOLD: float = 0.45

# ADX_TREND_THRESHOLD = 25, ADX_RANGE_THRESHOLD = 20
#   (a) Market reality: Welles Wilder's original ADX(14) prescription —
#       above 25 a trend is statistically distinguishable from noise, below
#       20 directional movement is dominated by noise. These thresholds are
#       the de-facto industry default and survive cross-asset / cross-pair
#       sensitivity studies.
#   (b) Constant rather than market-derived: ADX is already a moving
#       statistic of price; layering a second adaptive threshold over it
#       would double-count the smoothing. Wilder's defaults are the policy.
#   (c) Replace via: kwargs on `classify_regime` if backtest evidence on
#       a specific pair/TF shows the noise floor differs (rarely needed).
ADX_TREND_THRESHOLD: float = 25.0
ADX_RANGE_THRESHOLD: float = 20.0

# CHOP_TREND_THRESHOLD = 38.2, CHOP_RANGE_THRESHOLD = 61.8
#   (a) Market reality: Bill Dreiss' Choppiness Index uses Fibonacci levels
#       (38.2 / 61.8) as the trending vs choppy split. Below 38.2 the price
#       has covered most of the True-Range envelope (=trend); above 61.8 it
#       has covered little (=chop). The 38.2/61.8 split is the canonical
#       documented usage.
#   (b) Constant rather than market-derived: Fibonacci levels are
#       indicator-specification, not market data.
#   (c) Replace via: kwargs on `classify_regime`.
CHOP_TREND_THRESHOLD: float = 38.2
CHOP_RANGE_THRESHOLD: float = 61.8

# ATR_RANGE_PCT_THRESHOLD = 40, ATR_BREAKOUT_PCT_THRESHOLD = 20
#   (a) Market reality: a clean range regime needs *low* volatility — high
#       ATR percentile (top half of the year) means the market is moving,
#       not coiling. Bottom-20% ATR is the canonical "squeeze before
#       breakout" volatility footprint.
#   (b) Constant rather than market-derived: percentile cutoffs over a
#       1-year lookback are the indicator-specification; adapting them per
#       pair would discard the cross-pair comparability the percentile
#       transform was designed to provide.
#   (c) Replace via: kwargs on `classify_regime` if a pair shows persistent
#       low-vol regimes (e.g. pegged rates) where the bottom-20% gate is
#       always firing.
ATR_RANGE_PCT_THRESHOLD: float = 40.0
ATR_BREAKOUT_PCT_THRESHOLD: float = 20.0

# DEFAULT_HURST_WINDOW = 200
#   (a) Market reality: the source research brief (§1) prescribes
#       "rolling 200 bars" as the standard Hurst window for FX intraday.
#       Long enough for DFA scaling regression to span 2-3 octaves of
#       segment lengths; short enough to track a regime that has shifted
#       within the last trading day.
#   (b) Constant rather than market-derived: window length is operator
#       policy (responsiveness vs stability), same as in normalize.py.
#   (c) Replace via: `hurst_window=` kwarg on `classify_regime`. Use 100
#       for very short series, 252 for D1.
DEFAULT_HURST_WINDOW: int = 200

# DEFAULT_BARS_PER_YEAR = 252 * 288 = 72576
#   (a) Market reality: 252 trading days × 288 M5 bars per 24-hour FX day
#       = canonical 1-year lookback for ATR percentile on M5. The research
#       brief calls ATR-percentile-1y "the single highest-leverage cheap
#       regime feature" so the lookback must actually be ~1 year.
#   (b) Constant rather than market-derived: trading-day count is calendar
#       convention; 288 bars/day is M5 specifically. Other timeframes need
#       different multipliers — caller passes their own value.
#   (c) Replace via: `bars_per_year=` kwarg on `classify_regime`. H1 = 252*24
#       = 6048; D1 = 252; M15 = 252*96 = 24192.
DEFAULT_BARS_PER_YEAR: int = 252 * 288


@dataclass(frozen=True)
class RegimeReading:
    """Result of a regime classification.

    Attributes:
        state: One of "TREND_STRONG", "TREND_WEAK", "RANGE",
            "BREAKOUT_PENDING", "TRANSITION", "UNKNOWN".
        hurst: Latest Hurst exponent (DFA), or None if window underfilled.
        adx: Latest ADX(14), or None if window underfilled.
        choppiness: Latest Choppiness(14), or None if window underfilled.
        atr_percentile: ATR(14) percentile (0..100) over `bars_per_year`,
            or None if window underfilled.
        confidence: 0..1, fraction of the four signals that vote with the
            output state. 0 when state == "UNKNOWN".
    """

    state: str
    hurst: float | None
    adx: float | None
    choppiness: float | None
    atr_percentile: float | None
    confidence: float


# ---------------------------------------------------------------------------
# Pure-stdlib helpers (re-implemented here to avoid coupling regime.py to
# indicators.py's full IndicatorSet pipeline; the helpers below are the
# minimum needed for the regime gate alone).
# ---------------------------------------------------------------------------

def _hurst_dfa(series: Sequence[float], *, window: int) -> float | None:
    """Detrended Fluctuation Analysis Hurst exponent over the trailing window.

    Procedure (canonical DFA):
      1. Take the last `window` log-returns.
      2. Build the integrated profile y[i] = sum(returns[:i+1]).
      3. For each segment length n in {4, 8, 16, ...} up to window/4, split
         the profile into non-overlapping segments, fit a linear trend in
         each, and compute the rms of residuals F(n).
      4. Hurst = slope of log F(n) vs log n.

    Returns None if the window is underfilled or the regression is
    degenerate.
    """
    if len(series) < window + 1:
        return None
    # Build log-returns over trailing window bars.
    rets: list[float] = []
    tail = series[-(window + 1):]
    for i in range(1, len(tail)):
        prev, cur = tail[i - 1], tail[i]
        if prev is None or cur is None or prev <= 0 or cur <= 0:
            continue
        rets.append(log(cur / prev))
    if len(rets) < 16:
        return None

    # Profile (cumulative sum of mean-centered returns).
    mean_r = sum(rets) / len(rets)
    profile: list[float] = []
    s = 0.0
    for r in rets:
        s += (r - mean_r)
        profile.append(s)

    # Segment lengths: powers of two from 4 to len/4.
    n = len(profile)
    seg_lengths: list[int] = []
    k = 4
    while k <= n // 4:
        seg_lengths.append(k)
        k *= 2
    if len(seg_lengths) < 3:
        return None

    log_n: list[float] = []
    log_f: list[float] = []
    for seg_len in seg_lengths:
        n_segments = n // seg_len
        if n_segments < 1:
            continue
        rms_sq_sum = 0.0
        for seg_idx in range(n_segments):
            start = seg_idx * seg_len
            seg = profile[start:start + seg_len]
            # Linear detrend: fit y = a*x + b via least squares.
            x_mean = (seg_len - 1) / 2.0
            y_mean = sum(seg) / seg_len
            num = 0.0
            den = 0.0
            for x_i, y_i in enumerate(seg):
                dx = x_i - x_mean
                num += dx * (y_i - y_mean)
                den += dx * dx
            slope = num / den if den != 0.0 else 0.0
            intercept = y_mean - slope * x_mean
            for x_i, y_i in enumerate(seg):
                resid = y_i - (slope * x_i + intercept)
                rms_sq_sum += resid * resid
        f_n = sqrt(rms_sq_sum / (n_segments * seg_len))
        if f_n <= 0.0:
            continue
        log_n.append(log(seg_len))
        log_f.append(log(f_n))

    if len(log_n) < 3:
        return None

    # Linear regression slope of log_f vs log_n = Hurst exponent.
    nx = len(log_n)
    mx = sum(log_n) / nx
    my = sum(log_f) / nx
    num = sum((log_n[i] - mx) * (log_f[i] - my) for i in range(nx))
    den = sum((log_n[i] - mx) ** 2 for i in range(nx))
    if den == 0.0:
        return None
    return num / den


def _atr_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder ATR series. Returns a list aligned with `closes` length."""
    n = len(closes)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out
    # True ranges
    tr: list[float] = [0.0] * n
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    # Wilder smoothing: first ATR = simple mean of TR[1..period], then EMA.
    atr0 = sum(tr[1:period + 1]) / period
    out[period] = atr0
    prev = atr0
    for i in range(period + 1, n):
        cur = (prev * (period - 1) + tr[i]) / period
        out[i] = cur
        prev = cur
    return out


def _adx_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Latest Wilder ADX(period). None if insufficient data."""
    n = len(closes)
    if n < 2 * period + 1:
        return None
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Wilder smoothing for TR, +DM, -DM.
    s_tr = sum(tr[1:period + 1])
    s_plus = sum(plus_dm[1:period + 1])
    s_minus = sum(minus_dm[1:period + 1])
    dx_series: list[float] = []
    for i in range(period + 1, n):
        s_tr = s_tr - (s_tr / period) + tr[i]
        s_plus = s_plus - (s_plus / period) + plus_dm[i]
        s_minus = s_minus - (s_minus / period) + minus_dm[i]
        if s_tr == 0:
            continue
        plus_di = 100.0 * s_plus / s_tr
        minus_di = 100.0 * s_minus / s_tr
        denom = plus_di + minus_di
        if denom == 0:
            dx_series.append(0.0)
        else:
            dx_series.append(100.0 * abs(plus_di - minus_di) / denom)
    if len(dx_series) < period:
        return None
    # ADX = Wilder smoothing of DX.
    adx = sum(dx_series[:period]) / period
    for v in dx_series[period:]:
        adx = (adx * (period - 1) + v) / period
    return adx


def _choppiness_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Latest Choppiness Index (Bill Dreiss). None if insufficient data."""
    n = len(closes)
    if n < period + 1:
        return None
    tr_window: list[float] = []
    for i in range(n - period, n):
        if i <= 0:
            continue
        tr_window.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if not tr_window:
        return None
    atr_sum = sum(tr_window)
    hh = max(highs[n - period: n])
    ll = min(lows[n - period: n])
    rng = hh - ll
    if rng <= 0 or atr_sum <= 0:
        return None
    # Choppiness = 100 * log10(sum(TR)/range) / log10(period)
    from math import log10
    return 100.0 * log10(atr_sum / rng) / log10(period)


def _atr_percentile_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    period: int = 14,
    lookback: int,
) -> float | None:
    """Latest ATR(14) percentile (0..100) over the trailing `lookback` bars."""
    series = _atr_series(highs, lows, closes, period=period)
    cur = series[-1]
    if cur is None:
        return None
    valid = [v for v in series[-lookback:] if v is not None]
    if len(valid) < 2:
        return None
    below_or_eq = sum(1 for v in valid if v <= cur)
    return 100.0 * below_or_eq / len(valid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_regime(
    *,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    hurst_window: int = DEFAULT_HURST_WINDOW,
) -> RegimeReading:
    """Classify the current bar into one of the 5 regime states.

    Args:
        closes / highs / lows: aligned OHLC series, oldest first.
        bars_per_year: ATR-percentile lookback. Defaults to M5 (252*288).
            Pass an explicit value for other timeframes (see constant doc).
        hurst_window: rolling DFA window for Hurst. Default 200 (M5).

    Returns:
        RegimeReading. If any of the four indicators is None, the state is
        "UNKNOWN" with confidence=0; the trader can branch on that loudly
        rather than acting on partial signals.
    """
    if len(closes) < 2 or len(highs) != len(closes) or len(lows) != len(closes):
        return RegimeReading(
            state="UNKNOWN",
            hurst=None,
            adx=None,
            choppiness=None,
            atr_percentile=None,
            confidence=0.0,
        )

    hurst = _hurst_dfa(closes, window=hurst_window)
    adx = _adx_last(highs, lows, closes, period=14)
    chop = _choppiness_last(highs, lows, closes, period=14)
    atr_pct = _atr_percentile_last(
        highs, lows, closes, period=14, lookback=bars_per_year
    )

    # If ANY indicator is missing, refuse to invent a state. Per
    # AGENT_CONTRACT §3.5, "when in doubt, fail loud" — UNKNOWN is the loud
    # signal the trader can branch on.
    if hurst is None or adx is None or chop is None or atr_pct is None:
        return RegimeReading(
            state="UNKNOWN",
            hurst=hurst,
            adx=adx,
            choppiness=chop,
            atr_percentile=atr_pct,
            confidence=0.0,
        )

    trend_votes = 0
    if hurst > HURST_TREND_THRESHOLD:
        trend_votes += 1
    if adx > ADX_TREND_THRESHOLD:
        trend_votes += 1
    if chop < CHOP_TREND_THRESHOLD:
        trend_votes += 1

    range_votes = 0
    if hurst < HURST_RANGE_THRESHOLD:
        range_votes += 1
    if adx < ADX_RANGE_THRESHOLD:
        range_votes += 1
    if chop > CHOP_RANGE_THRESHOLD:
        range_votes += 1
    range_low_vol = atr_pct < ATR_RANGE_PCT_THRESHOLD

    # TREND_STRONG: all three trend indicators agree.
    if trend_votes == 3:
        return RegimeReading(
            state="TREND_STRONG",
            hurst=hurst,
            adx=adx,
            choppiness=chop,
            atr_percentile=atr_pct,
            confidence=1.0,
        )

    # RANGE: all three range indicators agree AND vol is low.
    if range_votes == 3 and range_low_vol:
        return RegimeReading(
            state="RANGE",
            hurst=hurst,
            adx=adx,
            choppiness=chop,
            atr_percentile=atr_pct,
            confidence=1.0,
        )

    # BREAKOUT_PENDING: ATR squeezed AND ADX/Chop neutral (neither pure
    # trend nor pure range — a coil before expansion).
    adx_neutral = ADX_RANGE_THRESHOLD <= adx <= ADX_TREND_THRESHOLD
    chop_neutral = CHOP_TREND_THRESHOLD <= chop <= CHOP_RANGE_THRESHOLD
    if atr_pct < ATR_BREAKOUT_PCT_THRESHOLD and adx_neutral and chop_neutral:
        return RegimeReading(
            state="BREAKOUT_PENDING",
            hurst=hurst,
            adx=adx,
            choppiness=chop,
            atr_percentile=atr_pct,
            confidence=0.75,
        )

    # TREND_WEAK: at least one trend indicator fires, but not all three.
    if trend_votes >= 1 and trend_votes < 3:
        return RegimeReading(
            state="TREND_WEAK",
            hurst=hurst,
            adx=adx,
            choppiness=chop,
            atr_percentile=atr_pct,
            confidence=trend_votes / 3.0,
        )

    # Anything else: TRANSITION (mixed / neutral / disagreeing signals).
    return RegimeReading(
        state="TRANSITION",
        hurst=hurst,
        adx=adx,
        choppiness=chop,
        atr_percentile=atr_pct,
        confidence=0.25,
    )
