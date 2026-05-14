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


def detect_pattern_signals(pair_chart: Optional[Dict[str, Any]]) -> List[PatternSignal]:
    """Run all pattern detectors against every relevant timeframe view."""
    if _is_disabled():
        return []
    if not pair_chart:
        return []
    out: List[PatternSignal] = []
    for view in pair_chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or "").upper()
        if tf not in PATTERN_TIMEFRAMES:
            continue
        out.extend(_detect_failed_breakout(view, tf))
        out.extend(_detect_rsi_extreme(view, tf))
        out.extend(_detect_dealing_range_extreme(view, tf))
        out.extend(_detect_aroon_flip(view, tf))
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
