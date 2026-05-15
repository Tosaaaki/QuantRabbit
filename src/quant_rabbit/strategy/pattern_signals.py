"""Pattern-recognition signals — the "そろそろ感" detector layer.

User feedback 2026-05-14:
  「そろそろあがるな、そろそろ下がるな、この形だからこうだな
   みたいなのって、考えられてる？」

The existing modules cover:
- regime_classifier: 24h range exhaustion via sigma multiple
- reversal_signal: extreme percentile + close-confirmed structural break
- entry_timing_gate: M5 last-3-candles direction
- chart_reader.structure_events: BOS/CHOCH with close_confirmed flag

This layer covers the discretionary trader's pattern-reading skill:
1. **Failed breakout** (`close_confirmed=False` BOS = wick-only sweep,
   classic trap pattern → fade direction).
2. **RSI extreme + price-band proximity** (RSI≥70 AND price near BB
   upper = overbought at upper rail → exhaustion).
3. **Dealing range edge** (price at swing_high ≥ 95% of range → top;
   ≤ 5% → bottom).
4. **Aroon strong cross** (aroon_up high + aroon_down low → momentum;
   reverse → fade).

Each detected pattern emits a `PatternSignal` with direction +
confidence + bounded bonus. `aggregate_pattern_score` sums signals
matching intent direction (bonus) and subtracts mismatched signals
(half-magnitude penalty), clamped to ±PATTERN_TOTAL_CAP.

The detectors operate on the existing indicator + structure data in
`pair_charts.json`. When `recent_candles` is available they also read
OHLC candle shape directly: engulfing, pin/shooting-star, doji, inside
bar, three-bar reversal, soldiers/crows, inside-bar break, volume spike,
and divergence patterns.

All magnitudes and thresholds are env-tunable. Kill switch:
`QR_DISABLE_PATTERN_SIGNALS=1`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Tunable knobs
PATTERN_TOTAL_CAP = float(os.environ.get("QR_PATTERN_TOTAL_CAP", "30.0"))
FAILED_BREAKOUT_BONUS = float(os.environ.get("QR_FAILED_BREAKOUT_BONUS", "15.0"))
RSI_EXTREME_BONUS = float(os.environ.get("QR_RSI_EXTREME_BONUS", "12.0"))
DEALING_RANGE_BONUS = float(os.environ.get("QR_DEALING_RANGE_BONUS", "10.0"))
AROON_FLIP_BONUS = float(os.environ.get("QR_AROON_FLIP_BONUS", "8.0"))
ENGULFING_BONUS = float(os.environ.get("QR_ENGULFING_BONUS", "13.0"))
PIN_BAR_BONUS = float(os.environ.get("QR_PIN_BAR_BONUS", "11.0"))
DOJI_BONUS = float(os.environ.get("QR_DOJI_BONUS", "6.0"))
INSIDE_BAR_BONUS = float(os.environ.get("QR_INSIDE_BAR_BONUS", "5.0"))
VOLUME_SPIKE_BONUS = float(os.environ.get("QR_VOLUME_SPIKE_BONUS", "10.0"))
TIME_EXHAUSTION_BONUS = float(os.environ.get("QR_TIME_EXHAUSTION_BONUS", "12.0"))
RSI_DIVERGENCE_BONUS = float(os.environ.get("QR_RSI_DIVERGENCE_BONUS", "14.0"))
MACD_DIVERGENCE_BONUS = float(os.environ.get("QR_MACD_DIVERGENCE_BONUS", "12.0"))
THREE_BAR_REVERSAL_BONUS = float(os.environ.get("QR_THREE_BAR_REVERSAL_BONUS", "14.0"))
THREE_SOLDIERS_CROWS_BONUS = float(os.environ.get("QR_THREE_SOLDIERS_CROWS_BONUS", "10.0"))
INSIDE_BAR_BREAK_BONUS = float(os.environ.get("QR_INSIDE_BAR_BREAK_BONUS", "11.0"))
VOLUME_PROFILE_HVN_BONUS = float(os.environ.get("QR_VOLUME_HVN_BONUS", "9.0"))
COT_SHIFT_BONUS = float(os.environ.get("QR_COT_SHIFT_BONUS", "11.0"))
OPTION_SKEW_BONUS = float(os.environ.get("QR_OPTION_SKEW_BONUS", "10.0"))

# Option-skew threshold — RR (risk reversal) magnitude. RR > threshold
# means call skew > put skew (bullish positioning); RR < -threshold
# means put skew > call skew (bearish positioning).
OPTION_SKEW_RR_THRESHOLD = float(os.environ.get("QR_OPTION_SKEW_RR_THRESHOLD", "0.5"))

# Volume profile: HVN (high volume node) detection — the price level
# where the most volume has transacted in recent N bars. Price returns
# to HVN tend to find support/resistance.
VOLUME_PROFILE_BINS = int(os.environ.get("QR_VOL_PROFILE_BINS", "20"))
VOLUME_PROFILE_NEAR_HVN_ATR_MULT = float(os.environ.get("QR_VOL_PROFILE_NEAR_HVN_ATR", "0.5"))

# COT shift: percentile shift of large-spec net positioning vs prior week.
COT_SHIFT_THRESHOLD = float(os.environ.get("QR_COT_SHIFT_THRESHOLD", "0.10"))  # 10% week-over-week

# Swing detection sensitivity for divergence: a local extreme requires
# being higher/lower than the N bars on each side. 2 is reasonable for
# 30-bar windows.
DIVERGENCE_SWING_STRENGTH = int(os.environ.get("QR_DIVERGENCE_SWING_STRENGTH", "2"))
# Minimum bar separation between the two swings used in divergence.
DIVERGENCE_MIN_SEPARATION = int(os.environ.get("QR_DIVERGENCE_MIN_SEPARATION", "5"))

# Volume-spike threshold: current bar volume must be at least this
# multiple of the rolling N-bar average to qualify.
VOLUME_SPIKE_MULT = float(os.environ.get("QR_VOLUME_SPIKE_MULT", "2.0"))
# Time-exhaustion: N consecutive same-color candles
TIME_EXHAUSTION_MIN_RUN = int(os.environ.get("QR_TIME_EXHAUSTION_MIN_RUN", "5"))
# Candle body / wick ratio thresholds for pin bar detection
PIN_BAR_WICK_TO_BODY_MIN = float(os.environ.get("QR_PIN_BAR_WICK_TO_BODY_MIN", "2.0"))
PIN_BAR_OPPOSITE_WICK_MAX_RATIO = float(os.environ.get("QR_PIN_BAR_OPPOSITE_WICK_MAX", "0.5"))
# Doji: body / total-range
DOJI_BODY_RATIO_MAX = float(os.environ.get("QR_DOJI_BODY_RATIO_MAX", "0.1"))

RSI_OVERBOUGHT = float(os.environ.get("QR_PATTERN_RSI_OVERBOUGHT", "70.0"))
RSI_OVERSOLD = float(os.environ.get("QR_PATTERN_RSI_OVERSOLD", "30.0"))
DEALING_RANGE_TOP_FRACTION = float(os.environ.get("QR_PATTERN_DR_TOP_FRACTION", "0.95"))
DEALING_RANGE_BOTTOM_FRACTION = float(os.environ.get("QR_PATTERN_DR_BOTTOM_FRACTION", "0.05"))
AROON_STRONG_THRESHOLD = float(os.environ.get("QR_PATTERN_AROON_STRONG", "70.0"))
AROON_WEAK_THRESHOLD = float(os.environ.get("QR_PATTERN_AROON_WEAK", "30.0"))

PATTERN_TIMEFRAMES = ("M5", "M15", "M30", "H1")


@dataclass(frozen=True)
class PatternSignal:
    name: str
    timeframe: str
    direction: str  # "UP" | "DOWN"
    confidence: float  # 0.0–1.0
    bonus_magnitude: float
    rationale: str


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_PATTERN_SIGNALS", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _detect_failed_breakout(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Recent BOS/CHOCH with close_confirmed=False = wick-only sweep.
    The market PRINTED a break in one direction but the candle closed
    back — classic trap. Fade direction."""
    structure = view.get("structure") or {}
    events = structure.get("structure_events") or []
    if not events:
        return []
    # Inspect last 4 events; pick the most recent failed one.
    for e in reversed(events[-4:]):
        if not isinstance(e, dict):
            continue
        if e.get("close_confirmed"):
            continue
        kind = str(e.get("kind") or "")
        if "UP" in kind:
            fade_dir = "DOWN"
        elif "DOWN" in kind:
            fade_dir = "UP"
        else:
            continue
        return [PatternSignal(
            name="failed_breakout",
            timeframe=tf,
            direction=fade_dir,
            confidence=0.75,
            bonus_magnitude=FAILED_BREAKOUT_BONUS,
            rationale=f"{tf} {kind}@{e.get('broken_pivot_price')} wick-only (close_confirmed=False) → trap fade {fade_dir}",
        )]
    return []


def _detect_rsi_extreme(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """RSI overbought + price near BB upper → reversal SHORT.
    Mirror for oversold + BB lower → reversal LONG.
    Requires BOTH conditions; pure RSI ≥70 in a strong trend isn't a
    reversal signal by itself, but RSI + price-at-rail is."""
    ind = view.get("indicators") or {}
    rsi = _to_float(ind.get("rsi_14"))
    close = _to_float(ind.get("close"))
    bb_upper = _to_float(ind.get("bb_upper"))
    bb_lower = _to_float(ind.get("bb_lower"))
    if None in (rsi, close, bb_upper, bb_lower):
        return []
    if bb_upper <= bb_lower:
        return []
    bb_mid = (bb_upper + bb_lower) / 2.0
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
    if rsi >= RSI_OVERBOUGHT and close >= bb_mid and bb_position >= 0.85:
        return [PatternSignal(
            name="rsi_extreme_top",
            timeframe=tf,
            direction="DOWN",
            confidence=min(1.0, (rsi - RSI_OVERBOUGHT) / 15.0 + 0.5),
            bonus_magnitude=RSI_EXTREME_BONUS,
            rationale=f"{tf} RSI {rsi:.1f}≥{RSI_OVERBOUGHT} + BB pos {bb_position:.2f}≥0.85 → reversal DOWN",
        )]
    if rsi <= RSI_OVERSOLD and close <= bb_mid and bb_position <= 0.15:
        return [PatternSignal(
            name="rsi_extreme_bottom",
            timeframe=tf,
            direction="UP",
            confidence=min(1.0, (RSI_OVERSOLD - rsi) / 15.0 + 0.5),
            bonus_magnitude=RSI_EXTREME_BONUS,
            rationale=f"{tf} RSI {rsi:.1f}≤{RSI_OVERSOLD} + BB pos {bb_position:.2f}≤0.15 → reversal UP",
        )]
    return []


def _detect_dealing_range_extreme(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Price at the top/bottom of the current dealing range = exhaustion."""
    smc = view.get("smc") or {}
    dr = smc.get("dealing_range") or {}
    swing_high = _to_float(dr.get("swing_high"))
    swing_low = _to_float(dr.get("swing_low"))
    ind = view.get("indicators") or {}
    close = _to_float(ind.get("close"))
    if None in (swing_high, swing_low, close):
        return []
    range_size = swing_high - swing_low
    if range_size <= 0:
        return []
    position = (close - swing_low) / range_size
    if position >= DEALING_RANGE_TOP_FRACTION:
        return [PatternSignal(
            name="dealing_range_top",
            timeframe=tf,
            direction="DOWN",
            confidence=0.6,
            bonus_magnitude=DEALING_RANGE_BONUS,
            rationale=f"{tf} close {close:.5f} at {position:.0%} of dealing range → exhaustion DOWN",
        )]
    if position <= DEALING_RANGE_BOTTOM_FRACTION:
        return [PatternSignal(
            name="dealing_range_bottom",
            timeframe=tf,
            direction="UP",
            confidence=0.6,
            bonus_magnitude=DEALING_RANGE_BONUS,
            rationale=f"{tf} close at {position:.0%} of dealing range → exhaustion UP",
        )]
    return []


def _detect_aroon_flip(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Aroon up strong + down weak → strong UP momentum; mirror DOWN."""
    ind = view.get("indicators") or {}
    up = _to_float(ind.get("aroon_up_14"))
    down = _to_float(ind.get("aroon_down_14"))
    if up is None or down is None:
        return []
    if up >= AROON_STRONG_THRESHOLD and down <= AROON_WEAK_THRESHOLD:
        return [PatternSignal(
            name="aroon_strong_up",
            timeframe=tf,
            direction="UP",
            confidence=0.55,
            bonus_magnitude=AROON_FLIP_BONUS,
            rationale=f"{tf} aroon_up={up:.0f}/down={down:.0f} → momentum UP",
        )]
    if down >= AROON_STRONG_THRESHOLD and up <= AROON_WEAK_THRESHOLD:
        return [PatternSignal(
            name="aroon_strong_down",
            timeframe=tf,
            direction="DOWN",
            confidence=0.55,
            bonus_magnitude=AROON_FLIP_BONUS,
            rationale=f"{tf} aroon_down={down:.0f}/up={up:.0f} → momentum DOWN",
        )]
    return []


def _candle_metrics(c: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract OHLCV from a serialised candle dict (keys o/h/l/c/v).
    Returns derived geometry: body, range, upper_wick, lower_wick,
    plus bull/bear sign."""
    try:
        o = float(c["o"]); h = float(c["h"]); l_ = float(c["l"]); cl = float(c["c"])
    except (KeyError, TypeError, ValueError):
        return None
    v = float(c.get("v") or 0)
    rng = h - l_
    if rng <= 0:
        return None
    body = abs(cl - o)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l_
    is_bull = cl > o
    is_bear = cl < o
    return {
        "open": o, "high": h, "low": l_, "close": cl, "volume": v,
        "range": rng, "body": body, "upper_wick": upper_wick,
        "lower_wick": lower_wick, "is_bull": 1.0 if is_bull else 0.0,
        "is_bear": 1.0 if is_bear else 0.0,
    }


def _detect_candlestick_patterns(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Engulfing, pin bar (hammer / shooting star), doji, inside bar."""
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < 2:
        return []
    # We score the LAST CLOSED candle (most recent in the list). Live
    # incomplete bars are excluded by the candle fetcher upstream.
    last = _candle_metrics(candles_raw[-1])
    prev = _candle_metrics(candles_raw[-2])
    if last is None or prev is None:
        return []
    out: List[PatternSignal] = []

    # 1. Engulfing
    # Bullish: prev bear, last bull, last body engulfs prev body.
    if prev["is_bear"] and last["is_bull"]:
        if last["close"] > prev["open"] and last["open"] < prev["close"]:
            out.append(PatternSignal(
                name="bullish_engulfing", timeframe=tf, direction="UP",
                confidence=0.75, bonus_magnitude=ENGULFING_BONUS,
                rationale=f"{tf} bullish engulfing (last bull body engulfs prev bear)",
            ))
    elif prev["is_bull"] and last["is_bear"]:
        if last["close"] < prev["open"] and last["open"] > prev["close"]:
            out.append(PatternSignal(
                name="bearish_engulfing", timeframe=tf, direction="DOWN",
                confidence=0.75, bonus_magnitude=ENGULFING_BONUS,
                rationale=f"{tf} bearish engulfing (last bear body engulfs prev bull)",
            ))

    # 2. Pin bar: long wick on ONE side, small body at the opposite end
    # Hammer (bullish reversal): long lower wick, small body near top
    if last["body"] > 0 and last["range"] > 0:
        lower_to_body = last["lower_wick"] / last["body"] if last["body"] > 0 else 0
        upper_to_body = last["upper_wick"] / last["body"] if last["body"] > 0 else 0
        upper_to_range = last["upper_wick"] / last["range"]
        lower_to_range = last["lower_wick"] / last["range"]
        # Hammer
        if (lower_to_body >= PIN_BAR_WICK_TO_BODY_MIN
            and upper_to_range <= PIN_BAR_OPPOSITE_WICK_MAX_RATIO * 0.5
            and lower_to_range >= 0.5):
            out.append(PatternSignal(
                name="hammer", timeframe=tf, direction="UP",
                confidence=0.65, bonus_magnitude=PIN_BAR_BONUS,
                rationale=f"{tf} hammer (lower wick {lower_to_range:.0%} of range, body near top) → UP",
            ))
        # Shooting star
        elif (upper_to_body >= PIN_BAR_WICK_TO_BODY_MIN
              and lower_to_range <= PIN_BAR_OPPOSITE_WICK_MAX_RATIO * 0.5
              and upper_to_range >= 0.5):
            out.append(PatternSignal(
                name="shooting_star", timeframe=tf, direction="DOWN",
                confidence=0.65, bonus_magnitude=PIN_BAR_BONUS,
                rationale=f"{tf} shooting star (upper wick {upper_to_range:.0%} of range, body near bottom) → DOWN",
            ))

    # 3. Doji: very small body relative to range = indecision / reversal
    # Doji on its own is weak; combined with a prior trending candle
    # implies pause. We emit it neutral-direction (LONG and SHORT both
    # benefit equally from caution) by emitting two signals.
    if last["body"] / last["range"] <= DOJI_BODY_RATIO_MAX:
        # If prev was bullish, the doji warns of pause → slight DOWN tilt
        # If prev was bearish, slight UP tilt
        if prev["is_bull"]:
            out.append(PatternSignal(
                name="doji_after_bull", timeframe=tf, direction="DOWN",
                confidence=0.4, bonus_magnitude=DOJI_BONUS,
                rationale=f"{tf} doji after bull candle (body {last['body']/last['range']:.0%} of range) → pause/reverse",
            ))
        elif prev["is_bear"]:
            out.append(PatternSignal(
                name="doji_after_bear", timeframe=tf, direction="UP",
                confidence=0.4, bonus_magnitude=DOJI_BONUS,
                rationale=f"{tf} doji after bear candle → pause/reverse",
            ))

    # 4. Inside bar: current bar's high ≤ prev high AND low ≥ prev low
    # Inside bar = consolidation, no directional bias yet; we DON'T
    # emit a signal because direction is undefined. Reserved for
    # future "inside-bar break" detector that pairs with breakout dir.

    return out


def _detect_volume_spike(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Last bar volume ≥ VOLUME_SPIKE_MULT × rolling 20-bar average →
    climax / capitulation. Direction inferred from candle direction
    (spike bull = exhaustion top, spike bear = exhaustion bottom)
    — climactic moves often precede reversal, so we FADE direction."""
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < 21:
        return []
    last = _candle_metrics(candles_raw[-1])
    if last is None:
        return []
    if last["volume"] <= 0:
        return []
    vols = [_candle_metrics(c)["volume"] for c in candles_raw[-21:-1] if _candle_metrics(c) is not None]
    if not vols:
        return []
    avg_vol = sum(vols) / len(vols)
    if avg_vol <= 0:
        return []
    ratio = last["volume"] / avg_vol
    if ratio < VOLUME_SPIKE_MULT:
        return []
    # Spike in direction of candle = likely climactic; fade.
    if last["is_bull"]:
        return [PatternSignal(
            name="volume_spike_climax_up", timeframe=tf, direction="DOWN",
            confidence=min(1.0, (ratio - VOLUME_SPIKE_MULT) / 2.0 + 0.5),
            bonus_magnitude=VOLUME_SPIKE_BONUS,
            rationale=f"{tf} volume {ratio:.1f}× avg on bull candle → climax exhaustion DOWN",
        )]
    if last["is_bear"]:
        return [PatternSignal(
            name="volume_spike_climax_down", timeframe=tf, direction="UP",
            confidence=min(1.0, (ratio - VOLUME_SPIKE_MULT) / 2.0 + 0.5),
            bonus_magnitude=VOLUME_SPIKE_BONUS,
            rationale=f"{tf} volume {ratio:.1f}× avg on bear candle → climax exhaustion UP",
        )]
    return []


def _detect_time_exhaustion(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """N consecutive same-color candles WITH shrinking range = trend
    losing steam → fade direction."""
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < TIME_EXHAUSTION_MIN_RUN + 1:
        return []
    metrics_list = [_candle_metrics(c) for c in candles_raw[-TIME_EXHAUSTION_MIN_RUN:]]
    if any(m is None for m in metrics_list):
        return []
    # All same color?
    all_bull = all(m["is_bull"] for m in metrics_list)
    all_bear = all(m["is_bear"] for m in metrics_list)
    if not (all_bull or all_bear):
        return []
    # Range shrinking? Check that the LAST half's avg range < first half's avg range.
    half = TIME_EXHAUSTION_MIN_RUN // 2
    first_half = metrics_list[:half] or metrics_list[:1]
    second_half = metrics_list[-half:] if half > 0 else metrics_list[-1:]
    first_avg = sum(m["range"] for m in first_half) / len(first_half)
    second_avg = sum(m["range"] for m in second_half) / len(second_half)
    if first_avg <= 0 or second_avg >= first_avg:
        return []
    shrink_ratio = 1 - (second_avg / first_avg)
    if shrink_ratio < 0.15:  # need at least 15% shrink
        return []
    fade_dir = "DOWN" if all_bull else "UP"
    return [PatternSignal(
        name="time_exhaustion", timeframe=tf, direction=fade_dir,
        confidence=min(1.0, shrink_ratio + 0.3),
        bonus_magnitude=TIME_EXHAUSTION_BONUS,
        rationale=f"{tf} {TIME_EXHAUSTION_MIN_RUN} consecutive {'bull' if all_bull else 'bear'} candles, range shrinking {shrink_ratio:.0%} → fade {fade_dir}",
    )]


def _find_swings(values: list[float], strength: int = 2, find_highs: bool = True) -> list[int]:
    """Return indices of local extremes.

    A local high at index i requires `values[i]` to be > all values in
    [i-strength, i-1] AND > all values in [i+1, i+strength]. Mirror for
    lows. Endpoints (within `strength` of either edge) cannot be swings.
    """
    if len(values) < 2 * strength + 1:
        return []
    out: list[int] = []
    for i in range(strength, len(values) - strength):
        center = values[i]
        left = values[i - strength: i]
        right = values[i + 1: i + 1 + strength]
        if find_highs:
            if all(center > x for x in left) and all(center > x for x in right):
                out.append(i)
        else:
            if all(center < x for x in left) and all(center < x for x in right):
                out.append(i)
    return out


def _detect_indicator_divergence(view: Dict[str, Any], tf: str, indicator_key: str, bonus: float) -> List[PatternSignal]:
    """True divergence using bar-aligned indicator series against
    price swings in `recent_candles`.

    Returns at most one signal per call (the most recent qualifying
    divergence). Both bearish (price HH + indicator LH) and bullish
    (price LL + indicator HL) classes are evaluated; pick whichever
    last fired.
    """
    candles_raw = view.get("recent_candles") or []
    series_map = view.get("indicator_series") or {}
    indicator_series = series_map.get(indicator_key) or []
    if not isinstance(indicator_series, (list, tuple)):
        return []
    if len(candles_raw) < 2 * DIVERGENCE_SWING_STRENGTH + DIVERGENCE_MIN_SEPARATION:
        return []
    if len(indicator_series) < len(candles_raw):
        # series shorter than candles → align right (most recent
        # values correspond to most recent candles).
        offset = len(candles_raw) - len(indicator_series)
    else:
        offset = 0
    if offset < 0:
        return []

    highs_price = [_candle_metrics(c)["high"] if _candle_metrics(c) else None for c in candles_raw]
    lows_price = [_candle_metrics(c)["low"] if _candle_metrics(c) else None for c in candles_raw]
    if None in highs_price or None in lows_price:
        return []

    high_swings = _find_swings(highs_price, strength=DIVERGENCE_SWING_STRENGTH, find_highs=True)
    low_swings = _find_swings(lows_price, strength=DIVERGENCE_SWING_STRENGTH, find_highs=False)

    def _ind_at(candle_idx: int) -> Optional[float]:
        series_idx = candle_idx - offset
        if series_idx < 0 or series_idx >= len(indicator_series):
            return None
        return float(indicator_series[series_idx])

    # Bearish divergence: take last 2 swing highs, check price HH but indicator LH.
    if len(high_swings) >= 2:
        a, b = high_swings[-2], high_swings[-1]
        if (b - a) >= DIVERGENCE_MIN_SEPARATION:
            ind_a = _ind_at(a); ind_b = _ind_at(b)
            if ind_a is not None and ind_b is not None:
                if highs_price[b] > highs_price[a] and ind_b < ind_a:
                    gap = (highs_price[b] - highs_price[a]) / (highs_price[a] or 1.0)
                    ind_drop = (ind_a - ind_b) / (abs(ind_a) + 1e-9)
                    confidence = min(1.0, 0.4 + abs(ind_drop))
                    return [PatternSignal(
                        name=f"bearish_{indicator_key}_divergence",
                        timeframe=tf, direction="DOWN",
                        confidence=confidence, bonus_magnitude=bonus,
                        rationale=f"{tf} price HH {highs_price[a]:.5f}→{highs_price[b]:.5f} but {indicator_key} LH {ind_a:.2f}→{ind_b:.2f} → bearish divergence",
                    )]
    # Bullish divergence: take last 2 swing lows, check price LL but indicator HL.
    if len(low_swings) >= 2:
        a, b = low_swings[-2], low_swings[-1]
        if (b - a) >= DIVERGENCE_MIN_SEPARATION:
            ind_a = _ind_at(a); ind_b = _ind_at(b)
            if ind_a is not None and ind_b is not None:
                if lows_price[b] < lows_price[a] and ind_b > ind_a:
                    ind_rise = (ind_b - ind_a) / (abs(ind_a) + 1e-9)
                    confidence = min(1.0, 0.4 + abs(ind_rise))
                    return [PatternSignal(
                        name=f"bullish_{indicator_key}_divergence",
                        timeframe=tf, direction="UP",
                        confidence=confidence, bonus_magnitude=bonus,
                        rationale=f"{tf} price LL {lows_price[a]:.5f}→{lows_price[b]:.5f} but {indicator_key} HL {ind_a:.2f}→{ind_b:.2f} → bullish divergence",
                    )]
    return []


def _detect_rsi_divergence(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """True bar-aligned RSI divergence against price swings.

    Replaces the prior `price_percentile_24h` proxy with the proper
    swing-vs-swing comparison. Requires `view.indicator_series.rsi_14`
    (produced by chart_reader 2026-05-14+); falls back to no-fire when
    the series is missing.
    """
    return _detect_indicator_divergence(view, tf, "rsi_14", RSI_DIVERGENCE_BONUS)


def _detect_macd_divergence(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Bar-aligned MACD histogram divergence."""
    return _detect_indicator_divergence(view, tf, "macd_hist", MACD_DIVERGENCE_BONUS)


def _detect_three_bar_patterns(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Morning star, evening star, three white soldiers, three black crows.

    Needs at least 3 recent candles. Each pattern emits at most one
    signal per view. Patterns:

    - **Morning star (bullish reversal)**: bear → small body (doji-ish)
      → bull that closes ≥ midpoint of the first candle's body.
    - **Evening star (bearish reversal)**: bull → small body → bear
      that closes ≤ midpoint of the first candle's body.
    - **Three white soldiers (bullish continuation/reversal)**: 3
      consecutive bull candles, each closing higher than the previous
      and opening within (not below) the previous body.
    - **Three black crows (bearish)**: mirror.
    """
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < 3:
        return []
    c1 = _candle_metrics(candles_raw[-3])
    c2 = _candle_metrics(candles_raw[-2])
    c3 = _candle_metrics(candles_raw[-1])
    if c1 is None or c2 is None or c3 is None:
        return []
    out: List[PatternSignal] = []

    # Morning star
    if (c1["is_bear"]
        and c2["body"] / c2["range"] <= 0.35  # narrow middle
        and c3["is_bull"]
        and c3["close"] >= (c1["open"] + c1["close"]) / 2.0):
        out.append(PatternSignal(
            name="morning_star", timeframe=tf, direction="UP",
            confidence=0.75, bonus_magnitude=THREE_BAR_REVERSAL_BONUS,
            rationale=f"{tf} morning star (bear / narrow / bull closing ≥ first body midpoint) → reversal UP",
        ))
    # Evening star
    if (c1["is_bull"]
        and c2["body"] / c2["range"] <= 0.35
        and c3["is_bear"]
        and c3["close"] <= (c1["open"] + c1["close"]) / 2.0):
        out.append(PatternSignal(
            name="evening_star", timeframe=tf, direction="DOWN",
            confidence=0.75, bonus_magnitude=THREE_BAR_REVERSAL_BONUS,
            rationale=f"{tf} evening star → reversal DOWN",
        ))
    # Three white soldiers
    if (c1["is_bull"] and c2["is_bull"] and c3["is_bull"]
        and c2["close"] > c1["close"] and c3["close"] > c2["close"]
        and c2["open"] >= c1["open"] and c2["open"] <= c1["close"]
        and c3["open"] >= c2["open"] and c3["open"] <= c2["close"]):
        out.append(PatternSignal(
            name="three_white_soldiers", timeframe=tf, direction="UP",
            confidence=0.6, bonus_magnitude=THREE_SOLDIERS_CROWS_BONUS,
            rationale=f"{tf} three white soldiers (3 bulls, each opens within prev body, closes higher) → UP",
        ))
    # Three black crows
    if (c1["is_bear"] and c2["is_bear"] and c3["is_bear"]
        and c2["close"] < c1["close"] and c3["close"] < c2["close"]
        and c2["open"] <= c1["open"] and c2["open"] >= c1["close"]
        and c3["open"] <= c2["open"] and c3["open"] >= c2["close"]):
        out.append(PatternSignal(
            name="three_black_crows", timeframe=tf, direction="DOWN",
            confidence=0.6, bonus_magnitude=THREE_SOLDIERS_CROWS_BONUS,
            rationale=f"{tf} three black crows → DOWN",
        ))
    return out


def _detect_volume_profile_hvn(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Volume profile HVN proximity.

    Builds a histogram of (close, volume) from `recent_candles` and
    identifies the HVN — the bin with the highest cumulative volume.
    When current price is within `VOLUME_PROFILE_NEAR_HVN_ATR_MULT × ATR`
    of an HVN, the level acts as magnet (support OR resistance depending
    on approach direction).

    Direction inference: if current price is BELOW the HVN, it's
    likely to rise toward it (UP); ABOVE → likely to fall toward it (DOWN).
    Confidence scales with the HVN's volume concentration vs total.
    """
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < 10:
        return []
    ind = view.get("indicators") or {}
    atr_pips = _to_float(ind.get("atr_pips"))
    current_close = _to_float(ind.get("close"))
    if not atr_pips or atr_pips <= 0 or current_close is None:
        return []
    # Build price-volume histogram
    pip_factor = 100.0 if "JPY" in str(ind.get("pair") or "") else 10000.0  # fallback; real value comes from caller
    # Determine price range from candles
    highs = []; lows = []; price_vol_pairs = []
    for c in candles_raw:
        m = _candle_metrics(c)
        if m is None:
            continue
        highs.append(m["high"]); lows.append(m["low"])
        mid = (m["high"] + m["low"]) / 2.0
        price_vol_pairs.append((mid, m["volume"]))
    if not highs:
        return []
    p_min, p_max = min(lows), max(highs)
    if p_max <= p_min:
        return []
    bin_size = (p_max - p_min) / VOLUME_PROFILE_BINS
    if bin_size <= 0:
        return []
    bins = [0.0] * VOLUME_PROFILE_BINS
    total_vol = 0.0
    for price, vol in price_vol_pairs:
        idx = min(VOLUME_PROFILE_BINS - 1, int((price - p_min) / bin_size))
        bins[idx] += vol
        total_vol += vol
    if total_vol <= 0:
        return []
    max_bin = max(range(VOLUME_PROFILE_BINS), key=lambda i: bins[i])
    hvn_price = p_min + (max_bin + 0.5) * bin_size
    hvn_concentration = bins[max_bin] / total_vol
    # Need at least 15% concentration to be meaningful
    if hvn_concentration < 0.15:
        return []
    # Pip factor heuristic from price magnitude
    pip_factor_local = 100.0 if (p_max < 1000 and p_max > 50) else 10000.0
    distance = abs(current_close - hvn_price)
    threshold_distance = (VOLUME_PROFILE_NEAR_HVN_ATR_MULT * atr_pips) / pip_factor_local
    if distance > threshold_distance:
        return []
    direction = "UP" if current_close < hvn_price else "DOWN"
    return [PatternSignal(
        name="volume_profile_hvn_magnet",
        timeframe=tf, direction=direction,
        confidence=min(1.0, hvn_concentration * 2 + 0.3),
        bonus_magnitude=VOLUME_PROFILE_HVN_BONUS,
        rationale=f"{tf} HVN at {hvn_price:.5f} ({hvn_concentration:.0%} of vol) — price magnet → {direction}",
    )]


def _detect_option_skew(pair: str, option_skew_payload: Optional[Dict[str, Any]]) -> List[PatternSignal]:
    """Option-skew (Risk Reversal) directional bias.

    Reads `data/option_skew_snapshot.json` readings entries for the
    target pair. RR (25-delta risk reversal):
    - `rr_25d > OPTION_SKEW_RR_THRESHOLD` → calls more expensive than
      puts → bullish positioning → UP bias
    - `rr_25d < -OPTION_SKEW_RR_THRESHOLD` → puts more expensive →
      bearish → DOWN bias

    Returns empty when:
    - No provider configured (rr_25d == None) — common state until
      a paid option-vol feed is wired up
    - RR within ±threshold band (no decisive bias)

    Preference: 1M tenor (most actionable for swing positioning),
    fall back to 1W if 1M missing.
    """
    if not option_skew_payload:
        return []
    readings = option_skew_payload.get("readings") or []
    if not isinstance(readings, list):
        return []
    # Pick 1M tenor first, else 1W
    target_reading = None
    for tenor_pref in ("1M", "1W", "3M"):
        for r in readings:
            if not isinstance(r, dict):
                continue
            if r.get("pair") != pair:
                continue
            if r.get("tenor") != tenor_pref:
                continue
            if r.get("rr_25d") is not None:
                target_reading = r
                break
        if target_reading is not None:
            break
    if target_reading is None:
        return []
    rr = _to_float(target_reading.get("rr_25d"))
    if rr is None:
        return []
    tenor = str(target_reading.get("tenor") or "")
    if abs(rr) < OPTION_SKEW_RR_THRESHOLD:
        return []
    direction = "UP" if rr > 0 else "DOWN"
    return [PatternSignal(
        name="option_skew_rr",
        timeframe=None, direction=direction,
        confidence=min(1.0, abs(rr) / (OPTION_SKEW_RR_THRESHOLD * 3)),
        bonus_magnitude=OPTION_SKEW_BONUS,
        rationale=f"option skew {tenor} RR_25d={rr:+.2f} ({direction} positioning bias)",
    )]


def _detect_cot_positioning_shift(pair: str, cot_payload: Optional[Dict[str, Any]]) -> List[PatternSignal]:
    """COT large-spec positioning shift detector.

    Reads `data/cot_snapshot.json` reports. For the pair's BASE currency,
    inspects `week_change_leveraged_net` — the week-over-week change in
    leveraged (large-spec / hedge-fund) net positioning. A meaningful
    shift (≥ COT_SHIFT_THRESHOLD × open_interest) signals institutional
    intent change.

    Positive `week_change_leveraged_net` for the base currency →
    institutions are increasing LONG → pair UP bias.
    Negative → pair DOWN bias.

    Confidence scales with the magnitude vs OI; lead time ≈ 1 week
    since COT data is weekly release.
    """
    if not cot_payload:
        return []
    reports = cot_payload.get("reports") or []
    if not reports:
        return []
    base, quote = _split_pair_for_cot(pair)
    if not base:
        return []
    target_report = None
    for r in reports:
        if not isinstance(r, dict):
            continue
        if str(r.get("currency", "")).upper() == base.upper():
            target_report = r
            break
    if target_report is None:
        return []
    open_interest = _to_float(target_report.get("open_interest"))
    week_change = _to_float(target_report.get("week_change_leveraged_net"))
    if open_interest is None or week_change is None or open_interest <= 0:
        return []
    shift_ratio = week_change / open_interest
    if abs(shift_ratio) < COT_SHIFT_THRESHOLD:
        return []
    direction = "UP" if shift_ratio > 0 else "DOWN"
    return [PatternSignal(
        name="cot_positioning_shift",
        timeframe=None, direction=direction,
        confidence=min(1.0, abs(shift_ratio) / (COT_SHIFT_THRESHOLD * 3)),
        bonus_magnitude=COT_SHIFT_BONUS,
        rationale=f"COT {base} large-spec week_net {week_change:+.0f} ({shift_ratio:+.1%} of OI) → {direction}",
    )]


def _split_pair_for_cot(pair: str) -> tuple[str, str]:
    parts = pair.split("_")
    if len(parts) != 2:
        return ("", "")
    return parts[0], parts[1]


def _detect_inside_bar_break(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Three-candle pattern: mother → inside → break.

    Inside bar (candle 2): high ≤ mother.high AND low ≥ mother.low.
    Break (candle 3, latest): close > mother.high → UP break;
                              close < mother.low → DOWN break.
    """
    candles_raw = view.get("recent_candles") or []
    if len(candles_raw) < 3:
        return []
    mother = _candle_metrics(candles_raw[-3])
    inside = _candle_metrics(candles_raw[-2])
    breaker = _candle_metrics(candles_raw[-1])
    if mother is None or inside is None or breaker is None:
        return []
    # Confirm inside bar:
    if not (inside["high"] <= mother["high"] and inside["low"] >= mother["low"]):
        return []
    if breaker["close"] > mother["high"]:
        return [PatternSignal(
            name="inside_bar_break_up", timeframe=tf, direction="UP",
            confidence=0.7, bonus_magnitude=INSIDE_BAR_BREAK_BONUS,
            rationale=f"{tf} inside bar break UP (mother.high {mother['high']:.5f} → breaker.close {breaker['close']:.5f})",
        )]
    if breaker["close"] < mother["low"]:
        return [PatternSignal(
            name="inside_bar_break_down", timeframe=tf, direction="DOWN",
            confidence=0.7, bonus_magnitude=INSIDE_BAR_BREAK_BONUS,
            rationale=f"{tf} inside bar break DOWN",
        )]
    return []


def _detect_multi_tf_volume_hvn(pair_chart: Dict[str, Any]) -> List[PatternSignal]:
    """Aggregate volume profile across M1, M5, M15 simultaneously.

    Per-view HVN detection (already done in `_detect_volume_profile_hvn`)
    can miss HVNs that are weaker on any single TF but strong when
    aggregated. This module combines all three short TFs into a single
    price-volume histogram and emits a stronger signal when the
    multi-TF HVN aligns with current price.

    Returns at most ONE signal per pair (the strongest combined HVN).
    """
    aggregate_pv: List[tuple[float, float]] = []
    current_close: Optional[float] = None
    atr_pips: Optional[float] = None
    pair = str(pair_chart.get("pair", ""))
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or "").upper()
        if tf not in ("M1", "M5", "M15"):
            continue
        ind = view.get("indicators") or {}
        if current_close is None:
            current_close = _to_float(ind.get("close"))
        if atr_pips is None:
            atr_pips = _to_float(ind.get("atr_pips"))
        for c in view.get("recent_candles") or []:
            m = _candle_metrics(c)
            if m is None:
                continue
            mid = (m["high"] + m["low"]) / 2.0
            aggregate_pv.append((mid, m["volume"]))
    if not aggregate_pv or current_close is None or atr_pips is None:
        return []

    prices = [p for p, _ in aggregate_pv]
    p_min, p_max = min(prices), max(prices)
    if p_max <= p_min:
        return []
    bin_count = VOLUME_PROFILE_BINS
    bin_size = (p_max - p_min) / bin_count
    if bin_size <= 0:
        return []
    bins = [0.0] * bin_count
    total = 0.0
    for price, vol in aggregate_pv:
        idx = min(bin_count - 1, int((price - p_min) / bin_size))
        bins[idx] += vol
        total += vol
    if total <= 0:
        return []
    max_bin = max(range(bin_count), key=lambda i: bins[i])
    hvn_price = p_min + (max_bin + 0.5) * bin_size
    concentration = bins[max_bin] / total
    if concentration < 0.12:
        return []
    pip_factor = 100.0 if pair.endswith("_JPY") else 10000.0
    distance = abs(current_close - hvn_price)
    threshold_distance = (VOLUME_PROFILE_NEAR_HVN_ATR_MULT * atr_pips) / pip_factor
    if distance > threshold_distance:
        return []
    direction = "UP" if current_close < hvn_price else "DOWN"
    return [PatternSignal(
        name="multi_tf_volume_hvn",
        timeframe="M1+M5+M15", direction=direction,
        confidence=min(1.0, concentration * 2 + 0.4),
        bonus_magnitude=VOLUME_PROFILE_HVN_BONUS * 1.3,  # multi-TF stronger
        rationale=f"multi-TF HVN at {hvn_price:.5f} ({concentration:.0%} of M1+M5+M15 vol) — magnet → {direction}",
    )]


def detect_pattern_signals(
    pair_chart: Optional[Dict[str, Any]],
    *,
    cot_payload: Optional[Dict[str, Any]] = None,
    option_skew_payload: Optional[Dict[str, Any]] = None,
) -> List[PatternSignal]:
    """Run all pattern detectors against every relevant timeframe view.

    `cot_payload`: optional `data/cot_snapshot.json` content.
    `option_skew_payload`: optional `data/option_skew_snapshot.json` content.
    Both are pair-level (not per-view) and skipped silently when missing.
    """
    if _is_disabled():
        return []
    if not pair_chart:
        return []
    out: List[PatternSignal] = []
    pair = str(pair_chart.get("pair") or "")
    if pair and cot_payload:
        out.extend(_detect_cot_positioning_shift(pair, cot_payload))
    if pair and option_skew_payload:
        out.extend(_detect_option_skew(pair, option_skew_payload))
    out.extend(_detect_multi_tf_volume_hvn(pair_chart))
    # Pair-level confluence used by some detectors (divergence). Inject
    # it onto each view so detectors don't need the parent chart.
    pair_confluence = pair_chart.get("confluence") or {}
    for view in pair_chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or "").upper()
        if tf not in PATTERN_TIMEFRAMES:
            continue
        # Inject pair-level confluence as a sibling key the detector reads.
        view_with_confluence = {**view, "confluence": pair_confluence}
        out.extend(_detect_failed_breakout(view, tf))
        out.extend(_detect_rsi_extreme(view, tf))
        out.extend(_detect_dealing_range_extreme(view, tf))
        out.extend(_detect_aroon_flip(view, tf))
        out.extend(_detect_candlestick_patterns(view, tf))
        out.extend(_detect_volume_spike(view, tf))
        out.extend(_detect_time_exhaustion(view, tf))
        out.extend(_detect_rsi_divergence(view, tf))
        out.extend(_detect_macd_divergence(view, tf))
        out.extend(_detect_three_bar_patterns(view, tf))
        out.extend(_detect_inside_bar_break(view, tf))
        out.extend(_detect_volume_profile_hvn(view, tf))
    return out


def aggregate_pattern_score(
    signals: List[PatternSignal],
    intent_direction: str,
) -> tuple[float, List[str]]:
    """Sum aligned signals (full magnitude), subtract opposed signals
    (half magnitude). Clamp the total to ±PATTERN_TOTAL_CAP.

    Aligned-side bias > opposed-side keeps the layer from canceling
    itself out when both UP and DOWN signals fire weakly — the model
    favors directional commitment when the bulk of pattern weight is
    on one side.
    """
    intent_up = intent_direction.upper() == "LONG"
    total = 0.0
    rationales: List[str] = []
    for s in signals:
        signal_up = s.direction.upper() == "UP"
        contribution = s.bonus_magnitude * s.confidence
        if signal_up == intent_up:
            total += contribution
            rationales.append(f"+{contribution:.1f} {s.rationale}")
        else:
            total -= contribution * 0.5
            rationales.append(f"-{contribution * 0.5:.1f} AGAINST {s.rationale}")
    if total > PATTERN_TOTAL_CAP:
        total = PATTERN_TOTAL_CAP
    elif total < -PATTERN_TOTAL_CAP:
        total = -PATTERN_TOTAL_CAP
    return round(total, 2), rationales
