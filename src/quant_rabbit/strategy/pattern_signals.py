"""Pattern-recognition signals — the "そろそろ感" detector layer.

User feedback 2026-05-14:
  「そろそろあがるな、そろそろ下がるな、この形だからこうだな
   みたいなのって、考えられてる？」

The existing modules cover:
- regime_classifier: 24h range exhaustion via sigma multiple
- reversal_signal: extreme percentile + close-confirmed structural break
- entry_timing_gate: M5 last-3-candles direction
- chart_reader.structure_events: BOS/CHOCH with close_confirmed flag

What's still missing — the discretionary trader's pattern-reading skill:
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
`pair_charts.json` — no raw OHLC needed (candlestick patterns like
engulfing/pin-bar would require raw candles and are deferred).

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


def _detect_rsi_divergence(view: Dict[str, Any], tf: str) -> List[PatternSignal]:
    """Bullish divergence: price makes a lower low, but RSI makes a higher low.
    Bearish divergence: price makes a higher high, but RSI doesn't.

    Without an RSI time-series we approximate by comparing the CURRENT
    RSI vs the RSI implied by the most recent swing extreme. Since
    pair_charts doesn't ship RSI history, we use a lightweight proxy:
    compare current RSI to rsi_percentile_100 — if price is at a new
    24h extreme but RSI percentile shows the indicator is NOT also at
    extreme, that's a structural divergence.
    """
    ind = view.get("indicators") or {}
    rsi = _to_float(ind.get("rsi_14"))
    rsi_pctile = _to_float(ind.get("rsi_percentile_100"))
    confluence = (view.get("confluence") if isinstance(view.get("confluence"), dict) else None) or {}
    price_pctile_24h = _to_float(confluence.get("price_percentile_24h"))
    # Fall back: pair-level confluence isn't stored on view; rely on
    # view-level fields if the parent passes them via chart_context.
    if price_pctile_24h is None:
        # No 24h price percentile here; divergence can't be confirmed.
        return []
    if rsi is None or rsi_pctile is None:
        return []
    # Bearish divergence: price at extreme high (≥0.9) but RSI percentile
    # NOT at extreme (≤0.7).
    if price_pctile_24h >= 0.9 and rsi_pctile <= 70:
        gap = (price_pctile_24h * 100) - rsi_pctile
        return [PatternSignal(
            name="bearish_rsi_divergence", timeframe=tf, direction="DOWN",
            confidence=min(1.0, gap / 50.0 + 0.4),
            bonus_magnitude=RSI_DIVERGENCE_BONUS,
            rationale=f"{tf} price pctile {price_pctile_24h:.2f} at extreme but RSI pctile {rsi_pctile:.0f} not confirming → bearish divergence",
        )]
    # Bullish divergence: price at extreme low (≤0.1) but RSI percentile
    # NOT at extreme low (≥30).
    if price_pctile_24h <= 0.1 and rsi_pctile >= 30:
        gap = rsi_pctile - (price_pctile_24h * 100)
        return [PatternSignal(
            name="bullish_rsi_divergence", timeframe=tf, direction="UP",
            confidence=min(1.0, gap / 50.0 + 0.4),
            bonus_magnitude=RSI_DIVERGENCE_BONUS,
            rationale=f"{tf} price pctile {price_pctile_24h:.2f} at extreme low but RSI pctile {rsi_pctile:.0f} not confirming → bullish divergence",
        )]
    return []


def detect_pattern_signals(
    pair_chart: Optional[Dict[str, Any]],
) -> List[PatternSignal]:
    """Run all pattern detectors against every relevant timeframe view."""
    if _is_disabled():
        return []
    if not pair_chart:
        return []
    out: List[PatternSignal] = []
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
        out.extend(_detect_rsi_divergence(view_with_confluence, tf))
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
