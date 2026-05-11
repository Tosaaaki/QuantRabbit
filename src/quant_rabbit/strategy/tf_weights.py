"""Situation-aware multi-timeframe weighting.

User 2026-05-11「TFの組み合わせってそのときの状況でかわるよね？」.

Fixed TF weights (D 0.20 → M1 0.10) over-fit one regime. Real
discretionary trading shifts the active TF set with:
  - Session (ASIA quiet vs LONDON impulse vs NY overlap)
  - Pair-level dominant regime (TREND_STRONG vs RANGE vs TRANSITION)
  - ADX distribution across timeframes
  - Trade method (scalp vs swing vs range rotation)

This module exposes a single `dynamic_tf_weights(...)` function that
returns a per-TF weight dict for a given situation. Callers (MTF
confluence, PA aggregate, structural TP picker) replace their hardcoded
weight tables with this so the same scoring engine reads the chart
through the appropriate timeframe lens for the moment.

The design is intentionally additive — every TF still gets non-zero
weight (never silently dropping a timeframe). Only the relative
emphasis shifts.
"""

from __future__ import annotations

import re
from typing import Any


# Situation tags surfaced in the rationale so the operator audit can see
# why the weighting shifted on a given cycle.
SITUATION_LABELS = {
    "ASIA_RANGE": "asia low-vol range scalp",
    "ASIA_TREND": "asia trend continuation",
    "LONDON_IMPULSE": "london open impulse",
    "LONDON_TREND": "london sustained trend",
    "NY_OVERLAP": "ny overlap multi-tf storm",
    "NY_TREND": "ny session trend",
    "ROLLOVER_SWING": "rollover / off-hours swing carryover",
    "TRANSITION": "regime transition / mixed",
}


# Baseline weights matching the previous MTF confluence table. Returned
# verbatim when no situation signal is available so behaviour degrades
# gracefully (no silent fallback to a literal — this is the documented
# baseline, not a guess).
BASELINE_WEIGHTS: dict[str, float] = {
    "D": 0.20,
    "H4": 0.18,
    "H1": 0.16,
    "M30": 0.14,
    "M15": 0.12,
    "M5": 0.10,
    "M1": 0.10,
}


# Per-situation weight profiles. Sum to 1.0 each. The shifts are
# discretionary-trader heuristics distilled from the user's directives:
# - Scalp regimes lean on M1/M5/M15 because the move is in those bars.
# - Trend regimes lean on H1/H4/D because the move spans hours+.
# - Range regimes lean on M15/M30 because rotation is bar-frequency.
# - Storm regimes weight evenly (every TF carries new info).
SITUATION_WEIGHTS: dict[str, dict[str, float]] = {
    # Quiet ASIA: M5/M15 dominate (range scalps), D nearly silent
    "ASIA_RANGE": {"D": 0.06, "H4": 0.10, "H1": 0.12, "M30": 0.18, "M15": 0.22, "M5": 0.20, "M1": 0.12},
    # ASIA trend (e.g., USD/JPY moves): M30/H1 dominant
    "ASIA_TREND": {"D": 0.10, "H4": 0.14, "H1": 0.20, "M30": 0.20, "M15": 0.16, "M5": 0.12, "M1": 0.08},
    # LONDON OPEN impulse: M1/M5 catches the burst, M15/M30 frames it
    "LONDON_IMPULSE": {"D": 0.08, "H4": 0.12, "H1": 0.14, "M30": 0.16, "M15": 0.18, "M5": 0.18, "M1": 0.14},
    # LONDON sustained trend: H1/H4 carry, M30 frames continuation
    "LONDON_TREND": {"D": 0.14, "H4": 0.20, "H1": 0.22, "M30": 0.16, "M15": 0.12, "M5": 0.10, "M1": 0.06},
    # NY OVERLAP storm: every TF active, slight bias to medium TFs
    "NY_OVERLAP": {"D": 0.14, "H4": 0.16, "H1": 0.16, "M30": 0.14, "M15": 0.14, "M5": 0.14, "M1": 0.12},
    # NY trend: similar to LONDON_TREND, slightly more H4/D weight
    "NY_TREND": {"D": 0.18, "H4": 0.22, "H1": 0.20, "M30": 0.14, "M15": 0.10, "M5": 0.08, "M1": 0.08},
    # ROLLOVER / off-hours: H1/H4 swing, micro almost noise
    "ROLLOVER_SWING": {"D": 0.20, "H4": 0.24, "H1": 0.22, "M30": 0.14, "M15": 0.10, "M5": 0.06, "M1": 0.04},
    # TRANSITION: equal-ish across mid TFs (where flips show first)
    "TRANSITION": {"D": 0.10, "H4": 0.14, "H1": 0.16, "M30": 0.18, "M15": 0.18, "M5": 0.14, "M1": 0.10},
}


# Method-aware overlay multipliers. Multiplied against the situation
# weights then renormalized. Captures the truism that range methods
# care about M15/M30 even in a trending session, etc.
METHOD_OVERLAY: dict[str, dict[str, float]] = {
    "RANGE_ROTATION": {"M15": 1.4, "M30": 1.3, "M5": 1.1, "H1": 0.9, "H4": 0.7, "D": 0.6, "M1": 0.9},
    "BREAKOUT_FAILURE": {"M5": 1.3, "M15": 1.2, "M30": 1.1, "M1": 1.1, "H1": 1.0, "H4": 0.9, "D": 0.8},
    "TREND_CONTINUATION": {"H4": 1.3, "H1": 1.3, "D": 1.2, "M30": 1.0, "M15": 0.9, "M5": 0.8, "M1": 0.7},
    "EVENT_RISK": {"M1": 1.4, "M5": 1.3, "M15": 1.1, "M30": 1.0, "H1": 0.9, "H4": 0.8, "D": 0.7},
}


_TF_BLOCK_PATTERN = re.compile(r"\b(D|H4|H1|M30|M15|M5|M1)\(([^)]+)\)")
_REGIME_FROM_BODY = re.compile(r"^\s*([A-Z_]+)")
_ADX_PATTERN = re.compile(r"ADX=([\d.]+)")


def _extract_per_tf_state(chart_story: str) -> dict[str, dict[str, float | str]]:
    """Pull (regime, adx) per TF out of the inline chart_story."""
    out: dict[str, dict[str, float | str]] = {}
    for match in _TF_BLOCK_PATTERN.finditer(chart_story or ""):
        tf = match.group(1)
        body = match.group(2)
        regime_match = _REGIME_FROM_BODY.match(body)
        regime = regime_match.group(1) if regime_match else ""
        adx_match = _ADX_PATTERN.search(body)
        adx = float(adx_match.group(1)) if adx_match else 0.0
        out[tf] = {"regime": regime, "adx": adx}
    return out


def classify_situation(
    *,
    session: str | None = None,
    chart_story: str = "",
    dominant_regime: str | None = None,
) -> str:
    """Return one of the SITUATION_WEIGHTS keys for the current state.

    Order of evaluation (specific → general):
    1. ROLLOVER session → ROLLOVER_SWING (regardless of regime)
    2. NY session: TREND_STRONG → NY_TREND, else NY_OVERLAP (storm bias)
    3. LONDON session: detect impulse vs sustained trend by H1/H4 ADX
    4. ASIA session: TREND_STRONG → ASIA_TREND, else ASIA_RANGE
    5. Unknown / TRANSITION → TRANSITION
    """
    sess = (session or "").upper()
    if "ROLLOVER" in sess or "OFF_HOURS" in sess:
        return "ROLLOVER_SWING"

    per_tf = _extract_per_tf_state(chart_story)
    h1 = per_tf.get("H1") or {}
    h4 = per_tf.get("H4") or {}
    m5 = per_tf.get("M5") or {}
    m15 = per_tf.get("M15") or {}

    h1_strong = float(h1.get("adx", 0.0)) >= 25.0 and "TREND" in str(h1.get("regime", ""))
    h4_strong = float(h4.get("adx", 0.0)) >= 25.0 and "TREND" in str(h4.get("regime", ""))
    micro_strong = float(m5.get("adx", 0.0)) >= 28.0 or float(m15.get("adx", 0.0)) >= 28.0

    if "NY" in sess or "NEW_YORK" in sess or "NEWYORK" in sess:
        if h1_strong and h4_strong:
            return "NY_TREND"
        return "NY_OVERLAP"

    if "LONDON" in sess or "LDN" in sess or "EUROPE" in sess:
        if micro_strong and not (h1_strong and h4_strong):
            return "LONDON_IMPULSE"
        if h1_strong or h4_strong:
            return "LONDON_TREND"
        return "LONDON_IMPULSE"  # default for london when no clear backbone

    if "ASIA" in sess or "TOKYO" in sess or "JP" in sess:
        if h1_strong and h4_strong:
            return "ASIA_TREND"
        return "ASIA_RANGE"

    # Use dominant_regime as a fallback signal when session is unknown.
    if dominant_regime:
        dr = dominant_regime.upper()
        if "TREND" in dr:
            return "NY_TREND"  # generic trend bucket
        if "RANGE" in dr or "TRANSITION" in dr or "UNCLEAR" in dr:
            return "TRANSITION"
    return "TRANSITION"


def dynamic_tf_weights(
    *,
    session: str | None = None,
    chart_story: str = "",
    dominant_regime: str | None = None,
    method: str | None = None,
) -> tuple[dict[str, float], str]:
    """Pick a per-TF weight dict for the current situation × method.

    Returns (weights, situation_label). weights sums to 1.0. The label is
    surfaced in scoring rationale so the operator can audit which lens
    was active for a given cycle.
    """
    situation = classify_situation(
        session=session,
        chart_story=chart_story,
        dominant_regime=dominant_regime,
    )
    base = SITUATION_WEIGHTS.get(situation, BASELINE_WEIGHTS).copy()
    overlay = METHOD_OVERLAY.get((method or "").upper(), {}) if method else {}
    if overlay:
        for tf, w in base.items():
            base[tf] = w * overlay.get(tf, 1.0)
        # Renormalize to 1.0
        total = sum(base.values())
        if total > 0:
            for tf in base:
                base[tf] = base[tf] / total
    return base, situation
