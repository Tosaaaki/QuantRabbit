"""Market-derived position-management policy.

AGENT_CONTRACT §3.5 mandates that all R:R / threshold multipliers are
regime-derived, not fixed literals. The trailing-TP and adverse-partial-
close modules originally shipped with fixed multipliers
(TRAILING_TRIGGER=1.0, TRAILING_LOCK_BEHIND=1.5, ADVERSE_TRIGGER=1.5,
PARTIAL_FRACTION=0.5). User feedback 2026-05-14:「市況みれてる？」
correctly identified the anti-pattern.

This module computes those multipliers from the live chart_context
each cycle. The inputs are the same ones used by
`intent_generator._market_derived_reward_risk`:
- `confluence.atr_percentile_24h`
- `h1_adx` / `h4_adx`
- `confluence.range_24h_sigma_multiple`
- `session_current_tag`
- per-position direction-vs-higher-TF alignment

Each helper returns (multiplier, rationale_lines) so callers can
surface the decision in audit logs. All adders are env-overridable
operator knobs; the BASE values match the previous hardcoded
defaults so the dynamic path is a pure additive refinement, not a
breaking policy change.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Base values match the prior hardcoded constants so the dynamic path
# is a refinement, not a regression in low-info conditions.
TRAILING_TRIGGER_BASE = float(os.environ.get("QR_TRAILING_TRIGGER_BASE", "1.0"))
TRAILING_TRIGGER_FLOOR = float(os.environ.get("QR_TRAILING_TRIGGER_FLOOR", "0.5"))
TRAILING_TRIGGER_CEILING = float(os.environ.get("QR_TRAILING_TRIGGER_CEILING", "2.5"))

TRAILING_LOCK_BASE = float(os.environ.get("QR_TRAILING_LOCK_BASE", "1.5"))
TRAILING_LOCK_FLOOR = float(os.environ.get("QR_TRAILING_LOCK_FLOOR", "0.7"))
TRAILING_LOCK_CEILING = float(os.environ.get("QR_TRAILING_LOCK_CEILING", "3.5"))

ADVERSE_TRIGGER_BASE = float(os.environ.get("QR_ADVERSE_TRIGGER_BASE", "1.5"))
ADVERSE_TRIGGER_FLOOR = float(os.environ.get("QR_ADVERSE_TRIGGER_FLOOR", "0.7"))
ADVERSE_TRIGGER_CEILING = float(os.environ.get("QR_ADVERSE_TRIGGER_CEILING", "3.0"))

PARTIAL_FRACTION_BASE = float(os.environ.get("QR_PARTIAL_FRACTION_BASE", "0.5"))
PARTIAL_FRACTION_FLOOR = float(os.environ.get("QR_PARTIAL_FRACTION_FLOOR", "0.25"))
PARTIAL_FRACTION_CEILING = float(os.environ.get("QR_PARTIAL_FRACTION_CEILING", "0.75"))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _higher_tf_alignment(chart_context: Dict[str, Any], intent_direction: str) -> str:
    """Returns 'WITH', 'AGAINST', or 'NEUTRAL' based on higher-TF bias
    in confluence.higher_tf_alignment + higher_tf_regime."""
    if not chart_context:
        return "NEUTRAL"
    confluence = chart_context.get("confluence") or {}
    higher_align = str(confluence.get("higher_tf_alignment") or "").upper()
    higher_regime = str(confluence.get("higher_tf_regime") or "").upper()
    side = intent_direction.upper()
    # Direct alignment field if present.
    if higher_align == "LONG_LEAN" and side == "LONG":
        return "WITH"
    if higher_align == "SHORT_LEAN" and side == "SHORT":
        return "WITH"
    if higher_align == "LONG_LEAN" and side == "SHORT":
        return "AGAINST"
    if higher_align == "SHORT_LEAN" and side == "LONG":
        return "AGAINST"
    # Fallback: regime label.
    if "TREND_UP" in higher_regime and side == "LONG":
        return "WITH"
    if "TREND_DOWN" in higher_regime and side == "SHORT":
        return "WITH"
    if "TREND_UP" in higher_regime and side == "SHORT":
        return "AGAINST"
    if "TREND_DOWN" in higher_regime and side == "LONG":
        return "AGAINST"
    return "NEUTRAL"


def trailing_trigger_mult(chart_context: Optional[Dict[str, Any]]) -> tuple[float, list[str]]:
    """Multiplier on ATR before trailing branch activates.

    Trending regime → trail sooner (smaller multiplier). Choppy → wait
    for proven profit. Deep-liquidity session → trail sooner.
    """
    rationale: list[str] = []
    v = TRAILING_TRIGGER_BASE
    if not chart_context:
        return v, ["chart_context missing → base"]
    confluence = chart_context.get("confluence") or {}
    adx = _to_float(chart_context.get("h1_adx") or chart_context.get("h4_adx"))
    if adx is not None:
        if adx >= 25:
            v -= 0.3
            rationale.append(f"ADX {adx:.1f} ≥ 25 → trail sooner -0.3")
        elif adx <= 18:
            v += 0.5
            rationale.append(f"ADX {adx:.1f} ≤ 18 → wait for proof +0.5")
    session = str(chart_context.get("session_current_tag") or chart_context.get("session_bucket") or "").upper()
    if "LONDON_NY" in session:
        v -= 0.2
        rationale.append(f"{session} → -0.2")
    elif session in ("OFF_HOURS", "JP_HOLIDAY"):
        v += 0.3
        rationale.append(f"{session} → +0.3")
    return _clamp(v, TRAILING_TRIGGER_FLOOR, TRAILING_TRIGGER_CEILING), rationale


def trailing_lock_behind_mult(chart_context: Optional[Dict[str, Any]]) -> tuple[float, list[str]]:
    """Multiplier on ATR for the distance trailing-TP keeps behind
    current price.

    High volatility / strong trend → wider behind (don't get noise-
    stopped). Range exhaustion → tighter (lock in before reversal).
    """
    rationale: list[str] = []
    v = TRAILING_LOCK_BASE
    if not chart_context:
        return v, ["chart_context missing → base"]
    confluence = chart_context.get("confluence") or {}
    atr_pct = _to_float(confluence.get("atr_percentile_24h"))
    if atr_pct is not None:
        if atr_pct >= 0.7:
            v += 0.5
            rationale.append(f"ATR %ile {atr_pct:.2f} ≥ 0.7 → wider lock +0.5")
        elif atr_pct <= 0.3:
            v -= 0.3
            rationale.append(f"ATR %ile {atr_pct:.2f} ≤ 0.3 → tighter -0.3")
    adx = _to_float(chart_context.get("h1_adx") or chart_context.get("h4_adx"))
    if adx is not None and adx >= 25:
        v += 0.5
        rationale.append(f"ADX {adx:.1f} ≥ 25 → wider lock +0.5")
    sigma_24h = _to_float(confluence.get("range_24h_sigma_multiple"))
    if sigma_24h is not None and sigma_24h >= 2.5:
        v -= 0.4
        rationale.append(f"24h σ {sigma_24h:.2f} ≥ 2.5 → lock-in tighter -0.4")
    return _clamp(v, TRAILING_LOCK_FLOOR, TRAILING_LOCK_CEILING), rationale


def adverse_trigger_mult(
    chart_context: Optional[Dict[str, Any]],
    intent_direction: str,
) -> tuple[float, list[str]]:
    """Multiplier on ATR for adverse threshold before partial close.

    Lower (close earlier) when higher-TF regime is against the
    position. Higher (give room) when regime supports it. Choppy
    regime → close earlier (no thesis to wait for).
    """
    rationale: list[str] = []
    v = ADVERSE_TRIGGER_BASE
    if not chart_context:
        return v, ["chart_context missing → base"]
    confluence = chart_context.get("confluence") or {}
    alignment = _higher_tf_alignment(chart_context, intent_direction)
    if alignment == "AGAINST":
        v -= 0.5
        rationale.append(f"higher-TF AGAINST {intent_direction} → close earlier -0.5")
    elif alignment == "WITH":
        v += 0.5
        rationale.append(f"higher-TF WITH {intent_direction} → give room +0.5")
    adx = _to_float(chart_context.get("h1_adx") or chart_context.get("h4_adx"))
    if adx is not None and adx <= 18:
        v -= 0.3
        rationale.append(f"ADX {adx:.1f} ≤ 18 (choppy) → close earlier -0.3")
    sigma_24h = _to_float(confluence.get("range_24h_sigma_multiple"))
    if sigma_24h is not None and sigma_24h >= 2.5:
        v -= 0.2
        rationale.append(f"24h σ {sigma_24h:.2f} ≥ 2.5 → close earlier -0.2")
    return _clamp(v, ADVERSE_TRIGGER_FLOOR, ADVERSE_TRIGGER_CEILING), rationale


def partial_close_fraction(
    chart_context: Optional[Dict[str, Any]],
    intent_direction: str,
) -> tuple[float, list[str]]:
    """Fraction of position to partial-close.

    Higher-TF AGAINST → reduce more (cut bigger chunk).
    Higher-TF WITH → reduce less (keep more thesis on).
    Choppy regime → reduce more (no edge in waiting).
    """
    rationale: list[str] = []
    v = PARTIAL_FRACTION_BASE
    if not chart_context:
        return v, ["chart_context missing → base"]
    alignment = _higher_tf_alignment(chart_context, intent_direction)
    if alignment == "AGAINST":
        v += 0.25
        rationale.append(f"higher-TF AGAINST {intent_direction} → reduce more +0.25")
    elif alignment == "WITH":
        v -= 0.25
        rationale.append(f"higher-TF WITH {intent_direction} → keep more thesis -0.25")
    adx = _to_float(chart_context.get("h1_adx") or chart_context.get("h4_adx"))
    if adx is not None and adx <= 18:
        v += 0.1
        rationale.append(f"ADX {adx:.1f} ≤ 18 (choppy) → reduce more +0.1")
    return _clamp(v, PARTIAL_FRACTION_FLOOR, PARTIAL_FRACTION_CEILING), rationale
