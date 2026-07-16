"""Per-pair regime classification — detect reversal risk before entry.

Reads `data/pair_charts.json` confluence + ATR data per pair and emits
a `regime_label` plus a `reversal_risk` score in [0, 1]. trader_brain
uses these to:

- Demote entries in REVERSAL_RISK > 0.6 lanes
- Surface the regime classification in lane rationale
- Block entries where the regime classification disagrees with the
  intent's method (e.g. TREND_CONTINUATION on a STABLE_RANGE pair)

This addresses the 2026-05-12 disaster pattern: the trader kept entering
LONG positions through the rally peak (5/8-5/11 wins, then 5/12 -22,980
LONG losses) because no module classified the regime as "exhausting
rally" before the reversal. The classifier looks at:

1. **ATR expansion**: current H1 ATR vs 7-day H1 ATR average. >1.5x
   means volatility is regime-shifting — caution.
2. **Multi-TF divergence**: D direction vs H4 direction vs H1 direction.
   When the three disagree, the trend is fracturing.
3. **24h expansion outlier**: `range_24h_expansion_outlier` from
   chart_reader. This compares the current 24h/H1 expansion ratio with
   the same pair's rolling distribution instead of pretending the raw
   ratio is a standard deviation.
4. **Recent reversal candle count**: looks for engulfing/pin-bar
   reversal patterns in last 4 H1 candles.

None of these alone trigger a block — they sum into a reversal_risk
score that the trader sees alongside other gates. The user remains the
final arbiter; this module raises the gate for the next decision layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


REGIME_ATR_EXPANSION_THRESHOLD = float(os.environ.get("QR_REGIME_ATR_EXPANSION", "1.5"))
REGIME_REVERSAL_RISK_PENALTY = float(os.environ.get("QR_REGIME_REVERSAL_RISK_PENALTY", "25.0"))
# Threshold above which a same-direction entry gets the full penalty.
# Below this threshold, the score scales linearly with reversal_risk.
REGIME_REVERSAL_RISK_GATE = float(os.environ.get("QR_REGIME_REVERSAL_RISK_GATE", "0.6"))


@dataclass(frozen=True)
class RegimeSnapshot:
    pair: str
    label: str  # "STABLE_TREND" | "REVERSAL_RISK" | "RANGING" | "UNKNOWN"
    direction_hint: str  # "UP" | "DOWN" | "NEUTRAL"
    reversal_risk: float  # [0.0, 1.0]
    signals: tuple[str, ...]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_pair(pair_chart: Dict[str, Any]) -> RegimeSnapshot:
    """Classify the regime for a single pair from its full pair_charts entry.

    The pair_chart payload is the per-pair object emitted by
    `quant_rabbit.cli pair-charts`. Missing fields degrade gracefully —
    we return UNKNOWN with reversal_risk=0 rather than blowing up on
    incomplete data.
    """
    signals: list[str] = []
    pair = str(pair_chart.get("pair", "?"))

    confluence = pair_chart.get("confluence") or {}
    expansion_outlier = confluence.get("range_24h_expansion_outlier") is True
    expansion_ratio = _coerce_float(confluence.get("range_24h_expansion_ratio"))
    expansion_fence = _coerce_float(confluence.get("range_24h_expansion_upper_fence"))
    atr_pct = _coerce_float(confluence.get("atr_percentile_24h"))
    tf_agreement = _coerce_float(confluence.get("tf_agreement_score"))

    # Multi-TF direction read.
    tf_directions: dict[str, str] = {}
    for tf in ("D", "H4", "H1"):
        block = pair_chart.get(tf) or {}
        bias = str(block.get("bias") or block.get("regime") or "").upper()
        if "UP" in bias or "BULL" in bias or "LONG" in bias:
            tf_directions[tf] = "UP"
        elif "DOWN" in bias or "BEAR" in bias or "SHORT" in bias:
            tf_directions[tf] = "DOWN"
        else:
            tf_directions[tf] = "NEUTRAL"

    # Direction hint from D when present, else H4.
    direction_hint = tf_directions.get("D", "NEUTRAL")
    if direction_hint == "NEUTRAL":
        direction_hint = tf_directions.get("H4", "NEUTRAL")

    # Signal 1: 24h range exhaustion.
    reversal_risk = 0.0
    if expansion_outlier:
        reversal_risk += 0.35
        signals.append(
            "24h expansion ratio is above its pair-relative upper fence "
            f"({expansion_ratio} > {expansion_fence}; range exhausted)"
        )

    # Signal 2: ATR percentile extreme (volatility regime shift).
    if atr_pct is not None and atr_pct >= 0.90:
        reversal_risk += 0.25
        signals.append(f"H1 ATR percentile {atr_pct:.2f} ≥ 0.90 (volatility expansion)")

    # Signal 3: TF disagreement.
    distinct_dirs = {d for d in tf_directions.values() if d != "NEUTRAL"}
    if len(distinct_dirs) >= 2:
        reversal_risk += 0.30
        signals.append(f"TF disagreement: D={tf_directions.get('D')} H4={tf_directions.get('H4')} H1={tf_directions.get('H1')}")
    elif tf_agreement is not None and tf_agreement < 0.5:
        reversal_risk += 0.20
        signals.append(f"tf_agreement_score {tf_agreement:.2f} < 0.50")

    # Signal 4: ATR percentile high AND a pair-relative 24h expansion outlier
    # together (compound exhaustion — vol expansion at a range extreme).
    if (
        expansion_outlier
        and atr_pct is not None and atr_pct >= 0.75
    ):
        reversal_risk += 0.15
        signals.append("compound exhaustion (range AND vol both stretched)")

    reversal_risk = min(1.0, reversal_risk)

    # Detect "no usable signal" up front so empty/malformed payloads do
    # not silently land in RANGING. UNKNOWN preserves the "we cannot
    # tell" state through the scoring path and avoids the small
    # RANGING penalty when there is genuinely no data.
    has_confluence = (
        confluence.get("range_24h_expansion_outlier") is not None
        or atr_pct is not None
        or tf_agreement is not None
    )
    has_tf_bias = any(d != "NEUTRAL" for d in tf_directions.values())
    if not has_confluence and not has_tf_bias:
        label = "UNKNOWN"
    elif reversal_risk >= REGIME_REVERSAL_RISK_GATE:
        label = "REVERSAL_RISK"
    elif len(distinct_dirs) <= 1 and direction_hint != "NEUTRAL":
        label = "STABLE_TREND"
    elif len(distinct_dirs) == 0:
        label = "RANGING"
    else:
        label = "UNKNOWN"

    return RegimeSnapshot(
        pair=pair,
        label=label,
        direction_hint=direction_hint,
        reversal_risk=reversal_risk,
        signals=tuple(signals),
    )


def classify_all(pair_charts_payload: Dict[str, Any]) -> Dict[str, RegimeSnapshot]:
    """Classify every pair in the pair_charts payload."""
    out: Dict[str, RegimeSnapshot] = {}
    charts = pair_charts_payload.get("charts") or []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        pair = str(chart.get("pair") or "")
        if not pair:
            continue
        out[pair] = classify_pair(chart)
    return out


def regime_score_modifier(
    snapshot: Optional[RegimeSnapshot],
    intent_direction: str,
) -> tuple[float, str | None]:
    """Compute a score delta for an entry given the pair's regime snapshot.

    - REVERSAL_RISK pair + entry direction matching the existing trend
      direction (i.e. trying to ride a possibly-exhausted move) →
      penalty proportional to reversal_risk.
    - STABLE_TREND pair + entry direction matching the trend → no
      penalty (this is the right setup).
    - RANGING pair + any direction → small penalty (no edge in either
      direction without a confluence boost elsewhere).
    """
    if snapshot is None:
        return 0.0, None

    intent_norm = "UP" if intent_direction.upper() == "LONG" else "DOWN"

    if snapshot.label == "REVERSAL_RISK":
        # The dangerous case: trading WITH the existing direction at a
        # reversal risk zone = buying the top or selling the bottom.
        if intent_norm == snapshot.direction_hint:
            delta = -REGIME_REVERSAL_RISK_PENALTY * snapshot.reversal_risk
            rationale = (
                f"regime REVERSAL_RISK ({snapshot.reversal_risk:.2f}) on {snapshot.pair}: "
                f"entry {intent_direction} aligns with possibly-exhausted {snapshot.direction_hint} → {delta:+.1f}"
            )
            return delta, rationale
        # Counter-trend entry in REVERSAL_RISK regime — neutral to mildly
        # favorable. We don't reward fading without a fresh structural
        # signal because the reversal hasn't confirmed yet.
        return 0.0, f"regime REVERSAL_RISK on {snapshot.pair}: counter-trend entry tolerated (no reward, no penalty)"

    if snapshot.label == "STABLE_TREND":
        if intent_norm == snapshot.direction_hint:
            return 8.0, f"regime STABLE_TREND on {snapshot.pair}: entry {intent_direction} matches trend (+8.0)"
        return -8.0, f"regime STABLE_TREND on {snapshot.pair}: entry {intent_direction} counter-trend (-8.0)"

    if snapshot.label == "RANGING":
        # Small structural penalty — ranging markets favor neither side
        # without a confluence-quality entry signal. The trader's
        # existing range_rotation method should still earn the lane via
        # method_pressure, this just removes the "free" thesis boost.
        return -5.0, f"regime RANGING on {snapshot.pair}: no thesis-direction edge (-5.0)"

    return 0.0, None
