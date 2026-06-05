"""Situation-aware multi-timeframe weighting.

User 2026-05-11「TFの組み合わせってそのときの状況でかわるよね？」+
「残研究も進めてみて」.

Fixed TF weights (D 0.20 → M1 0.10) over-fit one regime. Real
discretionary trading shifts the active TF set with:
  - Session (ASIA quiet vs LONDON impulse vs NY overlap)
  - Pair-level dominant regime (TREND_STRONG vs RANGE vs TRANSITION)
  - ADX distribution across timeframes
  - Trade method (scalp vs swing vs range rotation)
  - ATR percentile per TF (where the actual move is)
  - Pair character (USD/JPY slow vs G7 majors fast vs JPY crosses
    intervention-prone)
  - News calendar proximity (force EVENT_RISK overlay near events)

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

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
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


# Pair-specific overlay (Research Extension 1: pair character).
# Captures pair-level realities the situation+method axes miss:
# - USD/JPY: BoJ intervention risk → micro spikes amplified, daily slow
# - JPY crosses: intervention contagion → micro carries more meaning
# - EUR/CHF: SNB stickiness, daily edges dominant
# - AUD/NZD: thin liquidity, M30/H1 carry but micro often noise
# - Major USD pairs: balanced
PAIR_OVERLAY: dict[str, dict[str, float]] = {
    "USD_JPY": {"M1": 1.2, "M5": 1.2, "M15": 1.1, "M30": 1.0, "H1": 1.0, "H4": 0.9, "D": 0.85},
    "EUR_JPY": {"M1": 1.2, "M5": 1.15, "M15": 1.1, "M30": 1.0, "H1": 1.0, "H4": 0.9, "D": 0.85},
    "GBP_JPY": {"M1": 1.25, "M5": 1.2, "M15": 1.1, "M30": 1.0, "H1": 1.0, "H4": 0.9, "D": 0.85},
    "AUD_JPY": {"M1": 1.15, "M5": 1.15, "M15": 1.1, "M30": 1.05, "H1": 1.0, "H4": 0.95, "D": 0.9},
    "EUR_CHF": {"M1": 0.7, "M5": 0.85, "M15": 0.95, "M30": 1.05, "H1": 1.1, "H4": 1.15, "D": 1.2},
    "USD_CHF": {"M1": 0.8, "M5": 0.9, "M15": 1.0, "M30": 1.05, "H1": 1.1, "H4": 1.1, "D": 1.05},
    "AUD_NZD": {"M1": 0.7, "M5": 0.85, "M15": 1.0, "M30": 1.15, "H1": 1.15, "H4": 1.05, "D": 0.95},
    "NZD_CAD": {"M1": 0.75, "M5": 0.85, "M15": 1.0, "M30": 1.1, "H1": 1.15, "H4": 1.05, "D": 0.95},
}


# ATR percentile boost (Research Extension 2: vol where action lives).
# When a TF's atr_percentile is in the top quartile, the move is in
# THAT bar — boost its weight. Bottom quartile = quiet TF, reduce.
ATR_PERCENTILE_HIGH = 70.0
ATR_PERCENTILE_LOW = 20.0
ATR_HIGH_BOOST = 1.20
ATR_LOW_DAMP = 0.85


# News calendar overlay (Research Extension 3: event proximity).
# When a high-impact economic event is within ±MINUTES_BEFORE_AFTER
# of now and the pair is affected, force EVENT_RISK overlay regardless
# of session classification.
NEWS_WINDOW_MINUTES_BEFORE = 30
NEWS_WINDOW_MINUTES_AFTER = 60
HIGH_IMPACT_TOKENS = {"HIGH", "VERY_HIGH", "RED"}
DEFAULT_CALENDAR_PATH = Path("data/economic_calendar.json")


# Strategy mining edge multiplier (Research Extension 4: lean into
# methods that worked here). When a (pair, method) has strong positive
# historical evidence (high positive_evidence_n + positive live_net_jpy),
# boost its method overlay further by EDGE_MULT_HIGH; when it's weak
# or losing, dampen by EDGE_MULT_LOW. This compounds with METHOD_OVERLAY
# so a TREND_CONTINUATION method that has 200+ winning trades on EUR_USD
# gets H1/H4 weighted even more heavily than the method-overlay alone.
EDGE_EVIDENCE_HIGH = 100      # positive_evidence_n threshold for "well-mined"
EDGE_EVIDENCE_LOW = 20        # below this = barely-tested
EDGE_LIVE_NET_HIGH = 500.0    # JPY net edge to qualify as proven
EDGE_MULT_HIGH = 1.15         # boost method overlay strength
EDGE_MULT_LOW = 0.92          # weak / losing edge → dampen overlay
DEFAULT_STRATEGY_PROFILE_PATH = Path("data/strategy_profile.json")


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


_ATR_PCT_PATTERN = re.compile(r"\b(D|H4|H1|M30|M15|M5|M1)\([^)]*atr_percentile=([\d.]+)", re.IGNORECASE)


def _atr_percentile_from_views(pair_chart: dict[str, Any] | None) -> dict[str, float]:
    """Extract atr_percentile per TF from `pair_chart.views[].regime_reading`.

    Returns {tf: atr_pct (0-100)}. Empty dict when pair_chart missing.
    Caller passes the full pair_chart dict (price_action / trader_brain
    have it via _load_full_pair_charts).
    """
    out: dict[str, float] = {}
    if not isinstance(pair_chart, dict):
        return out
    for view in pair_chart.get("views") or []:
        gran = str(view.get("granularity") or "")
        if gran not in BASELINE_WEIGHTS:
            continue
        rr = view.get("regime_reading")
        if not isinstance(rr, dict):
            continue
        try:
            ap = float(rr.get("atr_percentile"))
        except (TypeError, ValueError):
            continue
        out[gran] = ap
    return out


def _high_impact_news_active(
    *,
    pair: str | None = None,
    calendar_path: Path | None = None,
    now_utc: datetime | None = None,
) -> tuple[bool, str]:
    """Check if a high-impact event is in the news window for `pair`.

    Returns (active, reason). Reads economic_calendar.json. Active when
    any HIGH/VERY_HIGH event for one of the pair's currencies is within
    [now - WINDOW_BEFORE, now + WINDOW_AFTER]. `pair` like "EUR_USD"
    matches events tagged USD or EUR.
    """
    path = calendar_path or DEFAULT_CALENDAR_PATH
    if not path.exists():
        return False, ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, ""

    now = now_utc or datetime.now(timezone.utc)
    window_before = NEWS_WINDOW_MINUTES_BEFORE * 60
    window_after = NEWS_WINDOW_MINUTES_AFTER * 60

    pair_currencies: set[str] = set()
    if pair and "_" in pair:
        a, b = pair.split("_", 1)
        pair_currencies = {a.upper(), b.upper()}

    events = payload.get("events") or payload.get("calendar") or []
    if not isinstance(events, list):
        return False, ""
    issues = payload.get("issues") or []
    if not events and any("MISSING_FOREX_FACTORY_FEED" in str(issue) for issue in issues):
        return True, "calendar_unavailable"
    for ev in events:
        if not isinstance(ev, dict):
            continue
        impact = str(ev.get("impact") or ev.get("importance") or "").upper()
        if impact not in HIGH_IMPACT_TOKENS:
            continue
        currency = str(ev.get("currency") or ev.get("country") or "").upper()
        if pair_currencies and currency and currency not in pair_currencies:
            continue
        ts_raw = ev.get("timestamp_utc") or ev.get("time_utc") or ev.get("time") or ev.get("timestamp") or ev.get("date")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        delta = (ts - now).total_seconds()
        if -window_after <= delta <= window_before:
            return True, f"{currency}:{ev.get('title') or ev.get('event') or 'event'} {int(delta/60)}m"
    return False, ""


def _strategy_edge_multiplier(
    pair: str | None,
    method: str | None,
    strategy_profile_path: Path | None,
) -> tuple[float, str]:
    """Look up (pair, method) edge from strategy_profile.json.

    Returns (multiplier, edge_label). Multiplier is applied to the method
    overlay magnitude — well-mined + winning method gets EDGE_MULT_HIGH,
    barely-tested or losing gets EDGE_MULT_LOW. Falls back to 1.0 (no
    boost, no dampen) when data is missing — documented neutral, not a
    silent guess.
    """
    if not pair or not method:
        return 1.0, ""
    path = strategy_profile_path or DEFAULT_STRATEGY_PROFILE_PATH
    if not path.exists():
        return 1.0, ""
    try:
        sp = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 1.0, ""
    method_upper = method.upper()
    pair_upper = pair.upper()
    best: dict[str, Any] | None = None
    for entry in sp.get("profiles") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("pair") or "").upper() != pair_upper:
            continue
        if str(entry.get("method") or "").upper() != method_upper:
            continue
        # Pick the entry with most positive_evidence (across LONG/SHORT)
        ev = int(entry.get("positive_evidence_n") or 0)
        if best is None or ev > int(best.get("positive_evidence_n") or 0):
            best = entry
    if best is None:
        return 1.0, ""
    ev_n = int(best.get("positive_evidence_n") or 0)
    live_net = float(best.get("live_net_jpy") or 0.0)
    if ev_n >= EDGE_EVIDENCE_HIGH and live_net >= EDGE_LIVE_NET_HIGH:
        return EDGE_MULT_HIGH, f"edge:{pair_upper}:{method_upper}:n={ev_n}/live={live_net:.0f}"
    if ev_n <= EDGE_EVIDENCE_LOW or live_net < 0:
        return EDGE_MULT_LOW, f"weak_edge:{pair_upper}:{method_upper}:n={ev_n}/live={live_net:.0f}"
    return 1.0, ""


def dynamic_tf_weights(
    *,
    session: str | None = None,
    chart_story: str = "",
    dominant_regime: str | None = None,
    method: str | None = None,
    pair: str | None = None,
    pair_chart: dict[str, Any] | None = None,
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
    now_utc: datetime | None = None,
) -> tuple[dict[str, float], str]:
    """Pick a per-TF weight dict for the current situation × method.

    Returns (weights, situation_label). weights sums to 1.0. Combines
    in order:
      1. Situation profile (session × dominant_regime × ADX)
      2. Method overlay (RANGE/BREAKOUT/TREND/EVENT)
      3. Pair character overlay (USD_JPY slow vs JPY crosses fast etc)
      4. ATR percentile boost (high-vol TF → +20%, low-vol → -15%)
      5. News calendar override (force EVENT_RISK overlay near events)
    """
    # Step 0: news calendar may override the situation entirely.
    news_active, news_reason = _high_impact_news_active(
        pair=pair, calendar_path=calendar_path, now_utc=now_utc
    )
    if news_active:
        situation = "NY_OVERLAP"  # storm bias for event window
        method_for_overlay = "EVENT_RISK"
    else:
        situation = classify_situation(
            session=session,
            chart_story=chart_story,
            dominant_regime=dominant_regime,
        )
        method_for_overlay = (method or "").upper()

    base = SITUATION_WEIGHTS.get(situation, BASELINE_WEIGHTS).copy()

    # Step 1+2: method overlay (with strategy-mining edge multiplier
    # applied to the overlay strength — well-mined + winning methods
    # have their TF preferences pushed harder)
    overlay = METHOD_OVERLAY.get(method_for_overlay, {}) if method_for_overlay else {}
    edge_mult, edge_label = _strategy_edge_multiplier(pair, method, strategy_profile_path)
    if overlay:
        for tf in base:
            ov = overlay.get(tf, 1.0)
            # Distance from 1.0 is the overlay's "strength"; multiply that
            # distance by edge_mult so a proven method overlays harder
            # while a weak method overlays softer.
            adjusted = 1.0 + (ov - 1.0) * edge_mult
            base[tf] = base[tf] * adjusted

    # Step 3: pair-specific overlay
    pair_key = (pair or "").upper()
    pair_ov = PAIR_OVERLAY.get(pair_key, {})
    if pair_ov:
        for tf in base:
            base[tf] = base[tf] * pair_ov.get(tf, 1.0)

    # Step 4: ATR percentile boost (use actual measured vol per TF)
    atr_pcts = _atr_percentile_from_views(pair_chart)
    for tf, ap in atr_pcts.items():
        if tf not in base:
            continue
        if ap >= ATR_PERCENTILE_HIGH:
            base[tf] = base[tf] * ATR_HIGH_BOOST
        elif ap <= ATR_PERCENTILE_LOW:
            base[tf] = base[tf] * ATR_LOW_DAMP

    # Renormalize to 1.0
    total = sum(base.values())
    if total > 0:
        for tf in base:
            base[tf] = base[tf] / total

    # Build descriptive label including all active overlays
    parts = [situation]
    if pair_ov:
        parts.append(f"pair:{pair_key}")
    if news_active:
        parts.append(f"news:{news_reason}")
    high_atr_tfs = [tf for tf, ap in atr_pcts.items() if ap >= ATR_PERCENTILE_HIGH]
    if high_atr_tfs:
        parts.append(f"atr_hot:{'/'.join(high_atr_tfs)}")
    if edge_label:
        parts.append(edge_label)
    label = " | ".join(parts)
    return base, label
