"""Forward-looking projection signals — predicting what's about to happen.

User feedback 2026-05-14:「未来を予測できてる？相場が動いてから動いちゃだめ」.

The existing 12 pattern_signals + reversal_signal layers all read PAST
events (last N candles, completed structure breaks, recent percentile
extremes). The user is right that this is reactive — by the time a
hammer prints or a divergence completes, the move has started.

This module adds FORWARD-LOOKING signals:

1. **BB squeeze → expansion forecast** — when BB width has been
   compressed for N bars AND ATR percentile is low, the next move is
   statistically likely to be a volatility expansion. Direction
   unknown but volatility is predictable; the trader can pre-position
   straddle-style or wait for the break.

2. **Liquidity sweep target** — equal swing highs / equal swing lows
   are "buy-side / sell-side liquidity pools". Smart money targets
   these BEFORE reversing. When price approaches an equal-extreme
   within a fade-distance ATR multiple, the projected next move is
   sweep-then-reverse.

3. **News catalyst lookahead** — when a high-impact economic event
   for the pair's currency lands within `NEWS_LOOKAHEAD_MIN` minutes,
   the next move IS the news reaction. Pre-position warning to the
   trader: don't open fresh exposure 30 min before NFP / CPI / rate
   decision.

4. **Cross-asset lag** — when a leading instrument (DXY, US10Y, SPX)
   has moved meaningfully but the FX pair hasn't caught up yet, the
   pair has a directional bias for the next 1-2 hours.

5. **Session expansion timing** — London open (~07:00 UTC) and NY
   open (~13:00 UTC) reliably produce volatility expansion. Pre-
   position before, not after.

Each detector emits a `ProjectionSignal` with `lead_time_min` so the
trader knows the EXPECTED arrival of the move (not just "it might
happen"). `ProjectionSignal.direction` is always the forecast / entry
direction consumed by `directional_forecaster` and `trader_brain`, not
an intermediate setup-trigger direction. All bounded magnitudes per
`PROJECTION_TOTAL_CAP`.

Kill switch: `QR_DISABLE_FORWARD_PROJECTION=1`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_rabbit.strategy.news_themes import NEWS_MAX_TOTAL_BIAS, parse_news_themes


PROJECTION_TOTAL_CAP = float(os.environ.get("QR_PROJECTION_TOTAL_CAP", "25.0"))

# BB squeeze
BB_SQUEEZE_WIDTH_PERCENTILE_MAX = float(os.environ.get("QR_BB_SQUEEZE_WIDTH_PCTILE", "0.25"))
BB_SQUEEZE_ATR_PERCENTILE_MAX = float(os.environ.get("QR_BB_SQUEEZE_ATR_PCTILE", "0.30"))
BB_SQUEEZE_BONUS = float(os.environ.get("QR_BB_SQUEEZE_BONUS", "8.0"))

# Liquidity sweep
LIQUIDITY_SWEEP_DISTANCE_ATR_MULT = float(os.environ.get("QR_LIQ_SWEEP_DIST_ATR", "0.5"))
LIQUIDITY_SWEEP_BONUS = float(os.environ.get("QR_LIQ_SWEEP_BONUS", "12.0"))
# Sweep targets inside the current operating noise are not forecasts; they are
# ordinary spread / micro-candle wobble. A quarter of the view ATR keeps the
# sweep target inside the existing 0.5 ATR "near enough to matter" window while
# filtering sub-noise 0.x-pip equal highs/lows that inflated fade confidence.
LIQUIDITY_SWEEP_MIN_DISTANCE_ATR_MULT = float(os.environ.get("QR_LIQ_SWEEP_MIN_DIST_ATR", "0.25"))

# News catalyst
NEWS_LOOKAHEAD_MIN = float(os.environ.get("QR_NEWS_LOOKAHEAD_MIN", "60.0"))  # warn 60 min before
NEWS_BLOCKED_MIN = float(os.environ.get("QR_NEWS_BLOCKED_MIN", "15.0"))  # hard wait 15 min before
NEWS_HIGH_IMPACT_PENALTY = float(os.environ.get("QR_NEWS_HIGH_IMPACT_PENALTY", "15.0"))
# (a) Post-release macro-event follow-through window. After a tier/high-impact
#     event prints actual-vs-forecast, the first several hours are still the
#     same repricing regime, not a stale headline.
# (b) Constant because NFP/CPI/rate-decision reactions routinely drive the
#     London/NY overlap and settle over roughly 2-4 hours; six hours covers
#     the full post-release digestion without carrying it into the next day.
# (c) Replace with per-event projection-ledger calibration once there are
#     enough event-tagged samples.
EVENT_SURPRISE_LOOKBACK_MIN = float(os.environ.get("QR_EVENT_SURPRISE_LOOKBACK_MIN", "360.0"))
# (a) Directional projection magnitude for an actual-vs-forecast surprise.
# (b) Same order as liquidity/cross-asset projection bonuses: large enough to
#     alter forecast direction when charts are ambiguous, but still bounded by
#     PROJECTION_TOTAL_CAP so technical/broker truth can override it.
# (c) Replace with event-tier calibrated effect sizes after measured samples.
EVENT_SURPRISE_BONUS = float(os.environ.get("QR_EVENT_SURPRISE_BONUS", "16.0"))
# (a) Minimum confidence for a high-impact event whose actual differs from
#     consensus. The sign is factual, but magnitude uncertainty remains.
# (b) Baseline sits just above the forecast entry floor so a clean surprise can
#     participate while still needing other market evidence for LIVE_READY.
# (c) Replace with event-name specific surprise-response calibration.
EVENT_SURPRISE_BASE_CONFIDENCE = float(os.environ.get("QR_EVENT_SURPRISE_BASE_CONFIDENCE", "0.58"))
# (a) Additional confidence from surprise size relative to consensus.
# (b) Ratio-based, not pip/JPY based, so NFP, CPI, PMI, and rates can share it.
# (c) Replace with per-event z-score calibration when historical releases are
#     labeled in the projection ledger.
EVENT_SURPRISE_RATIO_CONFIDENCE_GAIN = float(os.environ.get("QR_EVENT_SURPRISE_RATIO_CONFIDENCE_GAIN", "0.32"))
# (a) Post-release news-theme follow-through. This converts the curated news
#     digest / news_items packet into a directional forecast component when
#     calendar actuals are missing or delayed.
# (b) News parser bias is already bounded by NEWS_MAX_TOTAL_BIAS and never
#     bypasses spread, RR, chart-structure, profile, or gateway checks; this
#     projection gives forecast-first lanes a current macro input instead of
#     only a TraderBrain ranking nudge.
# (c) Replace the confidence mapping with event/headline hit-rate calibration
#     once `news_theme_followthrough` has enough projection-ledger samples.
NEWS_THEME_FOLLOWTHROUGH_MIN_ABS_BIAS = float(os.environ.get("QR_NEWS_THEME_FOLLOWTHROUGH_MIN_ABS_BIAS", "6.0"))
NEWS_THEME_FOLLOWTHROUGH_BONUS = float(os.environ.get("QR_NEWS_THEME_FOLLOWTHROUGH_BONUS", "10.0"))
NEWS_THEME_FOLLOWTHROUGH_BASE_CONFIDENCE = float(os.environ.get("QR_NEWS_THEME_FOLLOWTHROUGH_BASE_CONF", "0.56"))
NEWS_THEME_FOLLOWTHROUGH_CONFIDENCE_GAIN = float(os.environ.get("QR_NEWS_THEME_FOLLOWTHROUGH_CONF_GAIN", "0.24"))
# (a) Pre-release US employment nowcast. NFP is not knowable before BLS prints,
#     but the trader can score whether same-week labor-market evidence is
#     leaning toward a USD-positive or USD-negative surprise versus consensus.
# (b) Five calendar days covers the normal NFP evidence stack (JOLTS/ISM/ADP/
#     claims/layoffs) without carrying old month-end commentary into a new
#     event week. News evidence is separately capped to seven days.
# (c) Replace with a learned per-indicator regression once the projection
#     ledger has enough `us_employment_nowcast` samples.
US_EMPLOYMENT_NOWCAST_LOOKAHEAD_MIN = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_LOOKAHEAD_MIN", "7200.0"))
# (a) Maximum age for labor-market headlines used in the same event-week
#     nowcast. Public news/RSS items often arrive several days before NFP.
# (b) Seven days is the release-week envelope, not a trading threshold.
# (c) Replace with source/event timestamp joins when a structured macro data
#     provider is wired into the artifact packet.
US_EMPLOYMENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS = float(
    os.environ.get("QR_US_EMPLOYMENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS", "168.0")
)
# (a) Minimum net evidence score before the nowcast is emitted.
# (b) One generic headline is not enough; this floor requires either one
#     primary leading indicator plus direction wording, or several secondary
#     labor clues in agreement.
# (c) Replace with forecast-error calibrated posterior odds.
US_EMPLOYMENT_NOWCAST_MIN_ABS_SCORE = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_MIN_ABS_SCORE", "2.0"))
# (a) Score at which confidence/bonus saturation begins.
# (b) Three primary indicators or equivalent secondary agreement is enough to
#     call the pre-release labor skew strong without letting news dominate all
#     chart/broker evidence.
# (c) Replace with a standard-deviation scale after structured samples exist.
US_EMPLOYMENT_NOWCAST_SCORE_CAP = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_SCORE_CAP", "6.0"))
# (a) Projection magnitude for a pre-release employment nowcast.
# (b) Lower than actual-vs-forecast surprise because this is probabilistic
#     evidence, but large enough to shape forecast direction in ambiguous charts.
# (c) Replace with realized NFP-week FX response calibration.
US_EMPLOYMENT_NOWCAST_BONUS = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_BONUS", "12.0"))
# (a) Confidence mapping for pre-release nowcast evidence.
# (b) Baseline starts below the same-cycle bootstrap floor; only corroborated
#     evidence reaches high confidence.
# (c) Replace with learned hit-rate calibration by event cluster.
US_EMPLOYMENT_NOWCAST_BASE_CONFIDENCE = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_BASE_CONF", "0.57"))
US_EMPLOYMENT_NOWCAST_CONFIDENCE_GAIN = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_CONF_GAIN", "0.24"))

# (a) Relative evidence weights inside the NFP nowcast. ADP/nonfarm/payroll
#     clues are primary; claims/JOLTS/ISM/layoff/wage clues are supporting.
# (b) These are qualitative information weights, not risk or pip thresholds.
# (c) Replace each with learned per-indicator coefficients.
US_EMPLOYMENT_NOWCAST_PRIMARY_WEIGHT = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_PRIMARY_WEIGHT", "2.0"))
US_EMPLOYMENT_NOWCAST_SECONDARY_WEIGHT = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_SECONDARY_WEIGHT", "1.4"))
US_EMPLOYMENT_NOWCAST_GENERIC_WEIGHT = float(os.environ.get("QR_US_EMPLOYMENT_NOWCAST_GENERIC_WEIGHT", "0.8"))
# (a) Generic pre-event macro nowcast for scheduled high-impact data and
#     central-bank events beyond NFP: CPI/PCE/PPI, rate decisions/minutes,
#     GDP/PMI/ISM, retail, trade/commodity, and non-US employment releases.
# (b) Five calendar days matches the evidence window used by the NFP nowcast:
#     it is an event-week information envelope, not a trade or risk threshold.
# (c) Replace with event-family specific lead windows after projection-ledger
#     samples show which families need shorter/longer pre-release memory.
MACRO_EVENT_NOWCAST_LOOKAHEAD_MIN = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_LOOKAHEAD_MIN", "7200.0"))
# (a) Maximum age for generic pre-event evidence from headlines/digest.
# (b) Public macro previews and analyst notes are often published several days
#     before CPI/FOMC/GDP; seven days keeps them in the event week only.
# (c) Replace with structured source timestamps when vendor macro feeds exist.
MACRO_EVENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS = float(
    os.environ.get("QR_MACRO_EVENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS", "168.0")
)
# (a) Minimum net evidence before emitting a generic macro nowcast.
# (b) One weak headline is not enough; it takes a primary event-family clue or
#     multiple supporting clues in the same direction.
# (c) Replace with family-specific posterior odds after measured samples exist.
MACRO_EVENT_NOWCAST_MIN_ABS_SCORE = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_MIN_ABS_SCORE", "2.0"))
# (a) Score saturation for generic event-family evidence.
# (b) Three corroborating event-family clues or equivalent is strong enough to
#     shape forecast direction while staying below broker/risk authority.
# (c) Replace with learned z-score scaling by event family.
MACRO_EVENT_NOWCAST_SCORE_CAP = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_SCORE_CAP", "6.0"))
# (a) Projection magnitude for generic pre-event macro nowcast.
# (b) Lower than actual-vs-forecast surprise because it is probabilistic, but
#     high enough to matter when charts are otherwise ambiguous.
# (c) Replace with family-specific realized response calibration.
MACRO_EVENT_NOWCAST_BONUS = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_BONUS", "11.0"))
# (a) Confidence mapping for generic macro nowcast.
# (b) Baseline stays below live-entry confidence; only corroborated evidence
#     reaches same-cycle bootstrap territory.
# (c) Replace with projection-ledger hit-rate calibration by family/currency.
MACRO_EVENT_NOWCAST_BASE_CONFIDENCE = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_BASE_CONF", "0.56"))
MACRO_EVENT_NOWCAST_CONFIDENCE_GAIN = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_CONF_GAIN", "0.23"))
# (a) Evidence weights for generic macro nowcast: direct event-family wording
#     weighs more than a broad macro/risk adjective.
# (b) Qualitative information weights, not pip/JPY or risk thresholds.
# (c) Replace with learned coefficients per event family.
MACRO_EVENT_NOWCAST_PRIMARY_WEIGHT = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_PRIMARY_WEIGHT", "2.0"))
MACRO_EVENT_NOWCAST_SECONDARY_WEIGHT = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_SECONDARY_WEIGHT", "1.3"))
MACRO_EVENT_NOWCAST_GENERIC_WEIGHT = float(os.environ.get("QR_MACRO_EVENT_NOWCAST_GENERIC_WEIGHT", "0.7"))

# Cross-asset lag
CROSS_ASSET_MOVE_THRESHOLD_PCT = float(os.environ.get("QR_CROSS_ASSET_THRESHOLD_PCT", "0.3"))
CROSS_ASSET_LAG_BONUS = float(os.environ.get("QR_CROSS_ASSET_LAG_BONUS", "10.0"))

# Session expansion
SESSION_PRE_WINDOW_MIN = 30  # minutes before LDN/NY open
SESSION_EXPANSION_BONUS = float(os.environ.get("QR_SESSION_EXPANSION_BONUS", "6.0"))


@dataclass(frozen=True)
class ProjectionSignal:
    name: str
    timeframe: Optional[str]
    direction: str  # "UP" | "DOWN" | "EITHER" (forecast / entry direction)
    lead_time_min: float  # estimated minutes until the move
    confidence: float
    bonus_magnitude: float
    rationale: str


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_FORWARD_PROJECTION", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _detect_bb_squeeze_expansion(view: Dict[str, Any], tf: str) -> List[ProjectionSignal]:
    """Detect BB width compression + low ATR percentile = expansion soon.

    Direction is EITHER (not directional); the value to the trader is
    "expect volatility expansion in next ~5-10 bars" so scalpers can
    pre-position and trend-followers can ready stops.
    """
    ind = view.get("indicators") or {}
    bb_width_pctile = _to_float(ind.get("bb_width_percentile_100")) or _to_float(ind.get("bb_width_pctile"))
    atr_pctile = _to_float(ind.get("atr_percentile_100"))
    if bb_width_pctile is None or atr_pctile is None:
        return []
    if bb_width_pctile > BB_SQUEEZE_WIDTH_PERCENTILE_MAX:
        return []
    if atr_pctile > BB_SQUEEZE_ATR_PERCENTILE_MAX:
        return []
    # TF-based lead time estimate: M5 squeeze → ~5 M5 bars = 25 min
    tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
    bar_min = tf_minutes.get(tf, 15)
    lead = bar_min * 5  # ~5 bars expected before expansion
    return [ProjectionSignal(
        name="bb_squeeze_expansion_imminent",
        timeframe=tf, direction="EITHER",
        lead_time_min=lead,
        confidence=min(1.0, 0.4 + (BB_SQUEEZE_WIDTH_PERCENTILE_MAX - bb_width_pctile)),
        bonus_magnitude=BB_SQUEEZE_BONUS,
        rationale=f"{tf} BB width pctile {bb_width_pctile:.2f} + ATR pctile {atr_pctile:.2f} = squeeze, expansion ~{lead:.0f}min out",
    )]


def _detect_liquidity_sweep_target(view: Dict[str, Any], tf: str, current_price: Optional[float], pip_factor: float) -> List[ProjectionSignal]:
    """Equal swing highs / lows are liquidity targets.

    If price is within `LIQUIDITY_SWEEP_DISTANCE_ATR_MULT × ATR` of a
    swing high cluster, the next likely move is a sweep upward THEN
    reversal. Same for lows downward THEN reversal.

    The signal direction is the fade / entry direction because the
    forecast stack treats `direction` as executable bias. The sweep
    target itself remains in the rationale for projection-ledger
    verification and predictive LIMIT placement.
    """
    if current_price is None or current_price <= 0:
        return []
    ind = view.get("indicators") or {}
    atr_pips = _to_float(ind.get("atr_pips"))
    if not atr_pips or atr_pips <= 0:
        return []
    pip_size = 1.0 / pip_factor
    threshold_distance = LIQUIDITY_SWEEP_DISTANCE_ATR_MULT * atr_pips * pip_size
    spread_pips = _to_float(ind.get("spread_pips")) or 0.0
    min_distance_pips = max(spread_pips, LIQUIDITY_SWEEP_MIN_DISTANCE_ATR_MULT * atr_pips)
    min_distance = min_distance_pips * pip_size

    structure = view.get("structure") or {}
    liquidity = structure.get("liquidity") or []
    out: List[ProjectionSignal] = []

    # The schema is a list of {'price': float, 'side': 'EQ_HIGH'|'EQ_LOW', 'indices': [...]}
    eq_highs_prices: List[float] = []
    eq_lows_prices: List[float] = []
    if isinstance(liquidity, list):
        for entry in liquidity:
            if not isinstance(entry, dict):
                continue
            side = str(entry.get("side") or "").upper()
            price = _to_float(entry.get("price"))
            if price is None:
                continue
            if "HIGH" in side or "BSL" in side or "BUY" in side:
                eq_highs_prices.append(price)
            elif "LOW" in side or "SSL" in side or "SELL" in side:
                eq_lows_prices.append(price)

    # Nearest equal-high ABOVE current price
    prices_above = [p for p in eq_highs_prices if p > current_price]
    if prices_above:
        nearest = min(prices_above)
        dist = nearest - current_price
        if min_distance <= dist <= threshold_distance:
            pips_to = dist * pip_factor
            out.append(ProjectionSignal(
                name="liquidity_sweep_high",
                timeframe=tf, direction="DOWN",
                lead_time_min=15,
                confidence=min(1.0, 0.5 + (1.0 - dist / threshold_distance)),
                bonus_magnitude=LIQUIDITY_SWEEP_BONUS,
                rationale=f"{tf} equal-highs at {nearest:.5f} ({pips_to:.1f}pip up) — buy-side sweep target, fade SHORT",
            ))
    prices_below = [p for p in eq_lows_prices if p < current_price]
    if prices_below:
        nearest = max(prices_below)
        dist = current_price - nearest
        if min_distance <= dist <= threshold_distance:
            pips_to = dist * pip_factor
            out.append(ProjectionSignal(
                name="liquidity_sweep_low",
                timeframe=tf, direction="UP",
                lead_time_min=15,
                confidence=min(1.0, 0.5 + (1.0 - dist / threshold_distance)),
                bonus_magnitude=LIQUIDITY_SWEEP_BONUS,
                rationale=f"{tf} equal-lows at {nearest:.5f} ({pips_to:.1f}pip down) — sell-side sweep target, fade LONG",
            ))
    return out


def _pair_currencies(pair: str) -> Tuple[str, str]:
    parts = pair.split("_")
    if len(parts) == 2:
        return parts[0], parts[1]
    return ("", "")


def _detect_news_catalyst(
    pair: str,
    calendar_path: Path,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    """High-impact event in next NEWS_LOOKAHEAD_MIN minutes for either
    currency = pre-position warning. Direction is EITHER (the news
    moves price, but unknown direction); the trader should pause new
    entries within NEWS_BLOCKED_MIN of the print."""
    if not calendar_path.exists():
        return []
    now = now or datetime.now(timezone.utc)
    try:
        payload = json.loads(calendar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    events = payload.get("events", []) or []
    base, quote = _pair_currencies(pair)
    if not base or not quote:
        return []
    out: List[ProjectionSignal] = []
    for e in events:
        ts_raw = e.get("timestamp_utc") or e.get("time_utc")
        if not ts_raw:
            continue
        try:
            t = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if t < now:
            continue
        delta_min = (t - now).total_seconds() / 60.0
        if delta_min > NEWS_LOOKAHEAD_MIN:
            continue
        currency = str(e.get("currency") or "").upper()
        if currency not in (base, quote):
            continue
        impact = str(e.get("impact") or "").lower()
        if "high" not in impact:
            continue
        # Penalty magnitude inversely proportional to time remaining
        # (closer = stronger warning).
        # Linear: 1.0 at NEWS_BLOCKED_MIN, 0.0 at NEWS_LOOKAHEAD_MIN
        if delta_min <= NEWS_BLOCKED_MIN:
            urgency = 1.0
        else:
            urgency = max(0.0, 1.0 - (delta_min - NEWS_BLOCKED_MIN) / (NEWS_LOOKAHEAD_MIN - NEWS_BLOCKED_MIN))
        # Direction EITHER + negative bias for new entries via bonus magnitude.
        title = str(e.get("title") or "?")[:60]
        out.append(ProjectionSignal(
            name="news_catalyst_pending",
            timeframe=None, direction="EITHER",
            lead_time_min=delta_min,
            confidence=urgency,
            bonus_magnitude=-NEWS_HIGH_IMPACT_PENALTY,  # negative = discourage entry
            rationale=f"{currency} high-impact in {delta_min:.0f}min: {title} → reduce conviction (urgency {urgency:.2f})",
        ))
    return out


def _detect_event_surprise(
    pair: str,
    calendar_path: Path,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    """Actual-vs-forecast macro surprise after release.

    Pre-release news is directionless. Post-release actual data is different:
    if a high-impact event for one leg of the pair prints a measurable
    surprise, the currency gets a bounded directional follow-through signal.
    """
    if not calendar_path.exists():
        return []
    now = now or datetime.now(timezone.utc)
    try:
        payload = json.loads(calendar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    events = payload.get("events", []) or []
    base, quote = _pair_currencies(pair)
    if not base or not quote:
        return []
    out: List[ProjectionSignal] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        ts_raw = event.get("timestamp_utc") or event.get("time_utc")
        if not ts_raw:
            continue
        try:
            event_time = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if event_time > now:
            continue
        age_min = (now - event_time).total_seconds() / 60.0
        if age_min > EVENT_SURPRISE_LOOKBACK_MIN:
            continue
        currency = str(event.get("currency") or "").upper()
        if currency not in (base, quote):
            continue
        impact = str(event.get("impact") or "").lower()
        if "high" not in impact:
            continue
        title = str(event.get("title") or event.get("event") or "")
        actual = _parse_macro_number(event.get("actual"))
        forecast = _parse_macro_number(event.get("forecast"))
        if actual is None or forecast is None:
            continue
        strength = _event_currency_strength_sign(title=title, actual=actual, forecast=forecast)
        if strength == 0:
            continue
        direction = _currency_strength_to_pair_direction(pair, currency=currency, strength=strength)
        if direction is None:
            continue
        surprise_ratio = abs(actual - forecast) / max(abs(forecast), 1.0)
        confidence = min(1.0, EVENT_SURPRISE_BASE_CONFIDENCE + min(1.0, surprise_ratio) * EVENT_SURPRISE_RATIO_CONFIDENCE_GAIN)
        sign = "beat" if strength > 0 else "miss"
        out.append(ProjectionSignal(
            name="event_surprise_followthrough",
            timeframe=None,
            direction=direction,
            lead_time_min=0.0,
            confidence=confidence,
            bonus_magnitude=EVENT_SURPRISE_BONUS,
            rationale=(
                f"{currency} high-impact {title}: actual {event.get('actual')} vs "
                f"forecast {event.get('forecast')} ({sign}) → {pair} {direction} "
                f"post-event follow-through, age {age_min:.0f}min"
            ),
        ))
    return out


def _parse_macro_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        number = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    suffix = text[match.end():].strip().upper()
    if suffix.startswith("K"):
        number *= 1_000.0
    elif suffix.startswith("M"):
        number *= 1_000_000.0
    elif suffix.startswith("B"):
        number *= 1_000_000_000.0
    return number


def _event_currency_strength_sign(*, title: str, actual: float, forecast: float) -> int:
    title_upper = title.upper()
    if actual == forecast:
        return 0
    lower_is_better = (
        "UNEMPLOYMENT" in title_upper
        or "JOBLESS" in title_upper
        or "CLAIMS" in title_upper
    )
    higher_is_better = any(
        token in title_upper
        for token in (
            "EMPLOYMENT",
            "NON-FARM",
            "NONFARM",
            "PAYROLL",
            "EARNINGS",
            "CPI",
            "PCE",
            "PMI",
            "ISM",
            "GDP",
            "RETAIL SALES",
            "INDUSTRIAL PRODUCTION",
            "PRICES",
        )
    )
    if lower_is_better:
        return 1 if actual < forecast else -1
    if higher_is_better:
        return 1 if actual > forecast else -1
    return 0


def _currency_strength_to_pair_direction(pair: str, *, currency: str, strength: int) -> str | None:
    base, quote = _pair_currencies(pair)
    if currency == base:
        return "UP" if strength > 0 else "DOWN"
    if currency == quote:
        return "DOWN" if strength > 0 else "UP"
    return None


def _detect_news_theme_followthrough(
    pair: str,
    *,
    news_digest_path: Path | None,
    news_items_path: Path | None,
    calendar_path: Path | None,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    digest_path = news_digest_path or Path("__missing_news_digest__")
    if not digest_path.exists() and (news_items_path is None or not news_items_path.exists()):
        return []
    now_utc = now or datetime.now(timezone.utc)
    themes = parse_news_themes(
        digest_path,
        calendar_path=calendar_path,
        news_items_path=news_items_path,
        now_utc=now_utc,
    )
    long_bias = float(themes.biases.get((pair, "LONG"), 0.0))
    short_bias = float(themes.biases.get((pair, "SHORT"), 0.0))
    net = long_bias - short_bias
    if abs(net) < NEWS_THEME_FOLLOWTHROUGH_MIN_ABS_BIAS:
        return []
    direction = "UP" if net > 0 else "DOWN"
    normalized = min(1.0, abs(net) / max(NEWS_MAX_TOTAL_BIAS, 1.0))
    confidence = min(0.90, NEWS_THEME_FOLLOWTHROUGH_BASE_CONFIDENCE + normalized * NEWS_THEME_FOLLOWTHROUGH_CONFIDENCE_GAIN)
    bonus = NEWS_THEME_FOLLOWTHROUGH_BONUS * normalized
    theme_text = ", ".join(themes.detected_themes[:4]) or "news theme"
    side = "LONG" if direction == "UP" else "SHORT"
    return [
        ProjectionSignal(
            name="news_theme_followthrough",
            timeframe=None,
            direction=direction,
            lead_time_min=0.0,
            confidence=confidence,
            bonus_magnitude=bonus,
            rationale=(
                f"{pair} {side} news-theme follow-through: net_bias={net:+.1f} "
                f"(long={long_bias:+.1f}, short={short_bias:+.1f}, themes={theme_text})"
            ),
        )
    ]


_US_EMPLOYMENT_EVENT_RE = re.compile(
    r"\b(NFP|NON[-\s]?FARM|PAYROLL|EMPLOYMENT\s+CHANGE|UNEMPLOYMENT\s+RATE|"
    r"AVERAGE\s+HOURLY\s+EARNINGS)\b",
    re.IGNORECASE,
)
_US_EMPLOYMENT_EVIDENCE_RE = re.compile(
    r"\b(NFP|NON[-\s]?FARM|PAYROLL|ADP|PRIVATE\s+PAYROLL|JOBLESS|CLAIMS|"
    r"JOLTS|JOB\s+OPENINGS|ISM|PMI|EMPLOYMENT|UNEMPLOYMENT|LAYOFF|LAYOFFS|"
    r"JOB\s+CUTS|HIRING|WAGES?|AVERAGE\s+HOURLY\s+EARNINGS|LABO[RU]R\s+MARKET)\b",
    re.IGNORECASE,
)
_US_EMPLOYMENT_STRONG_RE = re.compile(
    r"\b(beat|beats|above|strong|stronger|robust|resilient|hot|tight|"
    r"broad[-\s]?based|accelerat(?:e|es|ed|ing)|jump|jumps|jumped|surge|"
    r"surges|surged|rise|rises|rose|higher|upside|increase|increased|"
    r"expansion)\b",
    re.IGNORECASE,
)
_US_EMPLOYMENT_SOFT_RE = re.compile(
    r"\b(miss|misses|below|weak|weaker|soft|softer|cool|cools|cooled|"
    r"slow|slows|slowed|slowing|crack|cracks|fall|falls|fell|drop|drops|"
    r"dropped|lower|downside|contraction|decline|declined|cut|cuts|"
    r"layoff|layoffs)\b",
    re.IGNORECASE,
)
_US_EMPLOYMENT_PRIMARY_RE = re.compile(r"\b(NFP|NON[-\s]?FARM|PAYROLL|ADP|PRIVATE\s+PAYROLL)\b", re.IGNORECASE)
_US_EMPLOYMENT_CLAIMS_RE = re.compile(r"\b(JOBLESS|CLAIMS|UNEMPLOYMENT\s+INSURANCE)\b", re.IGNORECASE)
_US_EMPLOYMENT_LOWER_IS_BETTER_RE = re.compile(
    r"\b(JOBLESS|CLAIMS|UNEMPLOYMENT\s+RATE|LAYOFF|LAYOFFS|JOB\s+CUTS)\b",
    re.IGNORECASE,
)
_US_EMPLOYMENT_SECONDARY_RE = re.compile(
    r"\b(JOLTS|JOB\s+OPENINGS|ISM|PMI|HIRING|WAGES?|AVERAGE\s+HOURLY\s+EARNINGS|"
    r"LABO[RU]R\s+MARKET)\b",
    re.IGNORECASE,
)


def _detect_us_employment_nowcast(
    pair: str,
    *,
    calendar_path: Path | None,
    news_digest_path: Path | None,
    news_items_path: Path | None,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    """Pre-release NFP/US employment surprise nowcast from current news evidence."""

    if calendar_path is None or not calendar_path.exists():
        return []
    base, quote = _pair_currencies(pair)
    if "USD" not in {base, quote}:
        return []
    now_utc = now or datetime.now(timezone.utc)
    cluster = _next_us_employment_event_cluster(calendar_path, now_utc)
    if cluster is None:
        return []
    event_time, event_titles = cluster
    evidence = _us_employment_evidence_from_news_items(
        news_items_path,
        now_utc=now_utc,
        event_time=event_time,
    )
    evidence.extend(
        _us_employment_evidence_from_digest(
            news_digest_path,
            now_utc=now_utc,
            event_time=event_time,
        )
    )
    if not evidence:
        return []
    score = sum(item["score"] for item in evidence)
    if abs(score) < US_EMPLOYMENT_NOWCAST_MIN_ABS_SCORE:
        return []
    direction = _currency_strength_to_pair_direction(pair, currency="USD", strength=1 if score > 0 else -1)
    if direction is None:
        return []
    normalized = min(1.0, abs(score) / max(US_EMPLOYMENT_NOWCAST_SCORE_CAP, 1.0))
    confidence = min(0.88, US_EMPLOYMENT_NOWCAST_BASE_CONFIDENCE + normalized * US_EMPLOYMENT_NOWCAST_CONFIDENCE_GAIN)
    bonus = US_EMPLOYMENT_NOWCAST_BONUS * normalized
    lead_min = max(0.0, (event_time - now_utc).total_seconds() / 60.0)
    side = "LONG" if direction == "UP" else "SHORT"
    skew = "USD-positive" if score > 0 else "USD-negative"
    top_evidence = sorted(evidence, key=lambda item: abs(item["score"]), reverse=True)[:3]
    evidence_text = "; ".join(str(item["reason"])[:90] for item in top_evidence)
    title_text = ", ".join(event_titles[:3])
    return [
        ProjectionSignal(
            name="us_employment_nowcast",
            timeframe=None,
            direction=direction,
            lead_time_min=lead_min,
            confidence=confidence,
            bonus_magnitude=bonus,
            rationale=(
                f"{pair} {side} pre-NFP nowcast: {skew} labor evidence score={score:+.1f} "
                f"before {event_time.isoformat()} ({title_text}); evidence: {evidence_text}"
            ),
        )
    ]


def _next_us_employment_event_cluster(calendar_path: Path, now_utc: datetime) -> tuple[datetime, list[str]] | None:
    try:
        payload = json.loads(calendar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    candidates: list[tuple[datetime, str]] = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("currency") or "").upper() != "USD":
            continue
        impact = str(event.get("impact") or "").lower()
        if "high" not in impact:
            continue
        title = str(event.get("title") or event.get("event") or "")
        if not _US_EMPLOYMENT_EVENT_RE.search(title):
            continue
        ts_raw = event.get("timestamp_utc") or event.get("time_utc")
        if not ts_raw:
            continue
        try:
            event_time = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if event_time < now_utc:
            continue
        lead_min = (event_time - now_utc).total_seconds() / 60.0
        if lead_min > US_EMPLOYMENT_NOWCAST_LOOKAHEAD_MIN:
            continue
        candidates.append((event_time, title))
    if not candidates:
        return None
    next_time = min(event_time for event_time, _title in candidates)
    titles = sorted({title for event_time, title in candidates if event_time == next_time})
    return next_time, titles


def _us_employment_evidence_from_news_items(
    news_items_path: Path | None,
    *,
    now_utc: datetime,
    event_time: datetime,
) -> list[dict[str, Any]]:
    if news_items_path is None or not news_items_path.exists():
        return []
    try:
        payload = json.loads(news_items_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        published = _parse_iso_utc(item.get("published_at_utc"))
        if published is None:
            continue
        if published > now_utc or published > event_time:
            continue
        age_hours = (now_utc - published).total_seconds() / 3600.0
        if age_hours > US_EMPLOYMENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS:
            continue
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                " ".join(str(topic) for topic in item.get("topics", []) or []),
                " ".join(str(currency) for currency in item.get("currencies", []) or []),
            ]
        )
        scored = _score_us_employment_nowcast_text(text)
        if scored is None:
            continue
        score, reason = scored
        source = str(item.get("source") or "news")
        out.append({"score": score, "reason": f"{source}: {reason}"})
    return out


def _us_employment_evidence_from_digest(
    news_digest_path: Path | None,
    *,
    now_utc: datetime,
    event_time: datetime,
) -> list[dict[str, Any]]:
    if news_digest_path is None or not news_digest_path.exists():
        return []
    try:
        text = news_digest_path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not _US_EMPLOYMENT_EVIDENCE_RE.search(line):
            continue
        scored = _score_us_employment_nowcast_text(line)
        if scored is None:
            continue
        score, reason = scored
        out.append({"score": score, "reason": f"digest: {reason}"})
    if not out:
        return []
    try:
        mtime = datetime.fromtimestamp(news_digest_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return out
    if mtime > now_utc or mtime > event_time:
        return []
    age_hours = (now_utc - mtime).total_seconds() / 3600.0
    if age_hours > US_EMPLOYMENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS:
        return []
    return out


def _score_us_employment_nowcast_text(text: str) -> tuple[float, str] | None:
    if not _US_EMPLOYMENT_EVIDENCE_RE.search(text):
        return None
    score = 0.0
    reasons: list[str] = []
    if _US_EMPLOYMENT_PRIMARY_RE.search(text):
        delta = _higher_is_better_labor_score(text, US_EMPLOYMENT_NOWCAST_PRIMARY_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"primary payroll/ADP {'hot' if delta > 0 else 'soft'}")
    if _US_EMPLOYMENT_CLAIMS_RE.search(text) or _US_EMPLOYMENT_LOWER_IS_BETTER_RE.search(text):
        delta = _lower_is_better_labor_score(text, US_EMPLOYMENT_NOWCAST_SECONDARY_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"claims/unemployment/layoff {'tight' if delta > 0 else 'soft'}")
    if _US_EMPLOYMENT_SECONDARY_RE.search(text):
        delta = _higher_is_better_labor_score(text, US_EMPLOYMENT_NOWCAST_SECONDARY_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"secondary labor {'hot' if delta > 0 else 'soft'}")
    if score == 0.0:
        delta = _higher_is_better_labor_score(text, US_EMPLOYMENT_NOWCAST_GENERIC_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"generic labor {'hot' if delta > 0 else 'soft'}")
    if score == 0.0:
        return None
    clipped = re.sub(r"\s+", " ", text).strip()
    return score, f"{', '.join(reasons)}: {clipped[:120]}"


def _higher_is_better_labor_score(text: str, weight: float) -> float:
    strong = bool(_US_EMPLOYMENT_STRONG_RE.search(text))
    soft = bool(_US_EMPLOYMENT_SOFT_RE.search(text))
    if strong and not soft:
        return weight
    if soft and not strong:
        return -weight
    return 0.0


def _lower_is_better_labor_score(text: str, weight: float) -> float:
    strong = bool(_US_EMPLOYMENT_STRONG_RE.search(text))
    soft = bool(_US_EMPLOYMENT_SOFT_RE.search(text))
    if soft and not strong:
        return weight
    if strong and not soft:
        return -weight
    return 0.0


def _parse_iso_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class _MacroNowcastFamily:
    name: str
    event_re: re.Pattern[str]
    evidence_re: re.Pattern[str]
    primary_re: re.Pattern[str]
    secondary_re: re.Pattern[str]
    strong_re: re.Pattern[str]
    soft_re: re.Pattern[str]
    lower_is_better_re: re.Pattern[str] | None = None


_MACRO_STRONG_RE = re.compile(
    r"\b(beat|beats|above|strong|stronger|robust|resilient|hot|sticky|"
    r"hawkish|hike|hikes|hiking|tighten|tightens|tightening|firm|firmer|"
    r"accelerat(?:e|es|ed|ing)|jump|jumps|jumped|surge|surges|surged|"
    r"rise|rises|rose|rising|higher|upside|increase|increased|expansion|"
    r"improve|improves|improved|rebound|rebounds|rebounded|support(?:ed)?|bid)\b",
    re.IGNORECASE,
)
_MACRO_SOFT_RE = re.compile(
    r"\b(miss|misses|below|weak|weaker|soft|softer|cool|cools|cooled|cooling|"
    r"dovish|cut|cuts|cutting|easing|ease|eases|lower|fall|falls|fell|"
    r"drop|drops|dropped|slow|slows|slowed|slowing|contraction|decline|"
    r"declined|downside|recession|fragile|stall|stalls|stalled|disappoint(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_CENTRAL_BANK_STRONG_RE = re.compile(
    r"\b(beat|beats|above|strong|stronger|robust|resilient|hot|sticky|"
    r"hawkish|hike|hikes|hiking|tighten|tightens|tightening|firm|firmer|"
    r"higher|upside|increase|increased|support(?:ed)?|bid|intervention|"
    r"intervene|rate[\s-]*check|yield\s+rise|yields?\s+higher)\b",
    re.IGNORECASE,
)
_CENTRAL_BANK_SOFT_RE = re.compile(
    r"\b(miss|misses|below|weak|weaker|soft|softer|cool|cools|cooled|cooling|"
    r"dovish|cut|cuts|cutting|easing|ease|eases|lower|fall|falls|fell|"
    r"drop|drops|dropped|slow|slows|slowed|slowing|contraction|decline|"
    r"declined|downside|pause|pausing|hold(?:ing)?\s+fire|rate\s+cuts?)\b",
    re.IGNORECASE,
)
_COMMODITY_STRONG_RE = re.compile(
    r"\b(beat|beats|above|strong|stronger|robust|resilient|firm|firmer|"
    r"jump|jumps|jumped|surge|surges|surged|rise|rises|rose|rising|higher|"
    r"upside|increase|increased|expansion|improve|improves|improved|"
    r"oil\s+rally|oil\s+surge|crude\s+higher|trade\s+surplus)\b",
    re.IGNORECASE,
)
_COMMODITY_SOFT_RE = re.compile(
    r"\b(miss|misses|below|weak|weaker|soft|softer|lower|fall|falls|fell|"
    r"drop|drops|dropped|slow|slows|slowed|slowing|contraction|decline|"
    r"declined|downside|oil\s+drop|oil\s+falls|crude\s+lower|trade\s+deficit)\b",
    re.IGNORECASE,
)
# (a) GDP/trade-drag context where a superficially "strong" word such as
#     "imports surged" is negative for growth or the currency.
# (b) The 120-character proximity window keeps the drag verb inside the same
#     headline sentence/clause without requiring a full NLP dependency parser.
# (c) Replace with structured macro-component parsing when GDP contribution
#     data is wired into the news/economic-calendar packet.
_GROWTH_DRAG_CONTEXT_RE = re.compile(
    r"\b(net\s+trade|imports?|exports?|government\s+spending|public\s+demand|GDP|growth|outlook)\b"
    r"[^.]{0,120}\b(subtract(?:s|ed)?|drag(?:s|ged)?|cloud(?:s|ed)?|weigh(?:s|ed)?|"
    r"flat|fall(?:s|en)?|fell|slow(?:s|ed|ing)?|weaker|below)\b",
    re.IGNORECASE,
)
_RATE_CUT_BETS_FADE_RE = re.compile(
    r"\b(rate\s+)?cut\s+bets?\s+(fade|fades|faded|recede|recedes|receded|"
    r"drop|drops|dropped|fall|falls|fell|ease|eases|eased|trim|trimmed|pared)\b",
    re.IGNORECASE,
)
_RATE_HIKE_BETS_RISE_RE = re.compile(
    r"\b(rate\s+)?hike\s+bets?\s+(rise|rises|rose|rising|increase|increases|"
    r"increased|build|builds|built|firm|firms|firmed)\b",
    re.IGNORECASE,
)
_RATE_CUT_BETS_RISE_RE = re.compile(
    r"\b(rate\s+)?cut\s+bets?\s+(rise|rises|rose|rising|increase|increases|"
    r"increased|build|builds|built|firm|firms|firmed)\b",
    re.IGNORECASE,
)
_RATE_HIKE_BETS_FADE_RE = re.compile(
    r"\b(rate\s+)?hike\s+bets?\s+(fade|fades|faded|recede|recedes|receded|"
    r"drop|drops|dropped|fall|falls|fell|ease|eases|eased|trim|trimmed|pared)\b",
    re.IGNORECASE,
)

_MACRO_NOWCAST_FAMILIES: tuple[_MacroNowcastFamily, ...] = (
    _MacroNowcastFamily(
        name="inflation",
        event_re=re.compile(r"\b(CPI|PCE|PPI|INFLATION|PRICE\s+INDEX|PRICES|AVERAGE\s+HOURLY\s+EARNINGS)\b", re.IGNORECASE),
        evidence_re=re.compile(r"\b(CPI|PCE|PPI|INFLATION|PRICE\s+INDEX|PRICES|WAGES?|EARNINGS|OIL|WTI|BRENT)\b", re.IGNORECASE),
        primary_re=re.compile(r"\b(CPI|PCE|PPI|INFLATION|PRICE\s+INDEX|PRICES)\b", re.IGNORECASE),
        secondary_re=re.compile(r"\b(WAGES?|EARNINGS|OIL|WTI|BRENT|EXPECTATIONS?)\b", re.IGNORECASE),
        strong_re=_MACRO_STRONG_RE,
        soft_re=_MACRO_SOFT_RE,
    ),
    _MacroNowcastFamily(
        name="central_bank",
        event_re=re.compile(
            r"\b(FOMC|FED|ECB|BOJ|BOE|RBA|BOC|RBNZ|SNB|RATE\s+DECISION|INTEREST\s+RATE|"
            r"REFINANCING\s+RATE|CASH\s+RATE|BANK\s+RATE|MONETARY\s+POLICY|"
            r"POLICY\s+STATEMENT|MEETING\s+MINUTES|GOV(?:ERNOR)?\s+SPEAKS|"
            r"CHAIR\s+SPEAKS|PRESS\s+CONFERENCE)\b",
            re.IGNORECASE,
        ),
        evidence_re=re.compile(
            r"\b(FOMC|FED|ECB|BOJ|BOE|RBA|BOC|RBNZ|SNB|RATE|YIELD|BOND|HAWKISH|DOVISH|"
            r"CUT|HIKE|TIGHTEN|EASING|INTERVENTION|RATE[\s-]*CHECK|MOF)\b",
            re.IGNORECASE,
        ),
        primary_re=re.compile(r"\b(FOMC|FED|ECB|BOJ|BOE|RBA|BOC|RBNZ|SNB|RATE|HAWKISH|DOVISH|CUT|HIKE)\b", re.IGNORECASE),
        secondary_re=re.compile(r"\b(YIELD|BOND|TIGHTEN|EASING|INTERVENTION|RATE[\s-]*CHECK|MOF)\b", re.IGNORECASE),
        strong_re=_CENTRAL_BANK_STRONG_RE,
        soft_re=_CENTRAL_BANK_SOFT_RE,
    ),
    _MacroNowcastFamily(
        name="growth",
        event_re=re.compile(r"\b(GDP|PMI|ISM|INDUSTRIAL\s+PRODUCTION|DURABLE\s+GOODS|FACTORY\s+ORDERS)\b", re.IGNORECASE),
        evidence_re=re.compile(r"\b(GDP|PMI|ISM|GROWTH|INDUSTRIAL\s+PRODUCTION|DURABLE\s+GOODS|FACTORY\s+ORDERS|MANUFACTURING|SERVICES)\b", re.IGNORECASE),
        primary_re=re.compile(r"\b(GDP|PMI|ISM|INDUSTRIAL\s+PRODUCTION|DURABLE\s+GOODS|FACTORY\s+ORDERS)\b", re.IGNORECASE),
        secondary_re=re.compile(r"\b(GROWTH|MANUFACTURING|SERVICES|OUTPUT|NEW\s+ORDERS)\b", re.IGNORECASE),
        strong_re=_MACRO_STRONG_RE,
        soft_re=_MACRO_SOFT_RE,
    ),
    _MacroNowcastFamily(
        name="consumption",
        event_re=re.compile(r"\b(RETAIL\s+SALES|CONSUMER\s+SENTIMENT|CONSUMER\s+CONFIDENCE|SPENDING|PERSONAL\s+INCOME|PERSONAL\s+SPENDING)\b", re.IGNORECASE),
        evidence_re=re.compile(r"\b(RETAIL\s+SALES|CONSUMER|SENTIMENT|CONFIDENCE|SPENDING|PERSONAL\s+INCOME|PERSONAL\s+SPENDING)\b", re.IGNORECASE),
        primary_re=re.compile(r"\b(RETAIL\s+SALES|CONSUMER\s+SENTIMENT|CONSUMER\s+CONFIDENCE|SPENDING)\b", re.IGNORECASE),
        secondary_re=re.compile(r"\b(CONSUMER|INCOME|DEMAND|SHOPPERS?)\b", re.IGNORECASE),
        strong_re=_MACRO_STRONG_RE,
        soft_re=_MACRO_SOFT_RE,
    ),
    _MacroNowcastFamily(
        name="employment",
        event_re=re.compile(
            r"\b(NFP|NON[-\s]?FARM|PAYROLL|EMPLOYMENT\s+CHANGE|UNEMPLOYMENT\s+RATE|"
            r"JOBLESS|CLAIMS|JOLTS|AVERAGE\s+HOURLY\s+EARNINGS)\b",
            re.IGNORECASE,
        ),
        evidence_re=_US_EMPLOYMENT_EVIDENCE_RE,
        primary_re=_US_EMPLOYMENT_PRIMARY_RE,
        secondary_re=_US_EMPLOYMENT_SECONDARY_RE,
        strong_re=_US_EMPLOYMENT_STRONG_RE,
        soft_re=_US_EMPLOYMENT_SOFT_RE,
        lower_is_better_re=_US_EMPLOYMENT_LOWER_IS_BETTER_RE,
    ),
    _MacroNowcastFamily(
        name="trade_commodity",
        event_re=re.compile(r"\b(TRADE\s+BALANCE|CURRENT\s+ACCOUNT|CRUDE\s+OIL|OIL\s+INVENTORIES|EXPORTS?|IMPORTS?)\b", re.IGNORECASE),
        evidence_re=re.compile(r"\b(TRADE\s+BALANCE|CURRENT\s+ACCOUNT|CRUDE\s+OIL|OIL|WTI|BRENT|EXPORTS?|IMPORTS?|COMMODITY|COMMODITIES)\b", re.IGNORECASE),
        primary_re=re.compile(r"\b(TRADE\s+BALANCE|CURRENT\s+ACCOUNT|CRUDE\s+OIL|OIL\s+INVENTORIES|EXPORTS?|IMPORTS?)\b", re.IGNORECASE),
        secondary_re=re.compile(r"\b(OIL|WTI|BRENT|COMMODITY|COMMODITIES)\b", re.IGNORECASE),
        strong_re=_COMMODITY_STRONG_RE,
        soft_re=_COMMODITY_SOFT_RE,
    ),
)

_MACRO_CURRENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "USD": ("USD", "US Dollar", "Greenback", "DXY", "Fed", "FOMC"),
    "EUR": ("EUR", "Euro", "ECB"),
    "GBP": ("GBP", "Sterling", "Pound", "Cable", "BOE"),
    "JPY": ("JPY", "Yen", "BOJ", "MOF"),
    "AUD": ("AUD", "Aussie", "Australian Dollar", "RBA"),
    "NZD": ("NZD", "Kiwi", "New Zealand Dollar", "RBNZ"),
    "CAD": ("CAD", "Loonie", "Canadian Dollar", "BOC"),
    "CHF": ("CHF", "Swiss Franc", "Swissie", "SNB"),
}


def _detect_macro_event_nowcast(
    pair: str,
    *,
    calendar_path: Path | None,
    news_digest_path: Path | None,
    news_items_path: Path | None,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    if calendar_path is None or not calendar_path.exists():
        return []
    base, quote = _pair_currencies(pair)
    pair_currencies = {base, quote}
    if not base or not quote:
        return []
    now_utc = now or datetime.now(timezone.utc)
    out: list[ProjectionSignal] = []
    for cluster in _upcoming_macro_event_clusters(calendar_path, now_utc):
        currency = str(cluster["currency"])
        family = cluster["family"]
        event_time = cluster["event_time"]
        titles = list(cluster["titles"])
        if currency not in pair_currencies:
            continue
        if currency == "USD" and family.name == "employment" and any(_US_EMPLOYMENT_EVENT_RE.search(t) for t in titles):
            # The USD employment stack has a dedicated NFP nowcast with labor-
            # specific ADP/claims/JOLTS treatment; avoid double-counting it.
            continue
        evidence = _macro_event_evidence_from_news_items(
            news_items_path,
            currency=currency,
            family=family,
            now_utc=now_utc,
            event_time=event_time,
        )
        evidence.extend(
            _macro_event_evidence_from_digest(
                news_digest_path,
                currency=currency,
                family=family,
                now_utc=now_utc,
                event_time=event_time,
            )
        )
        if not evidence:
            continue
        score = sum(item["score"] for item in evidence)
        if abs(score) < MACRO_EVENT_NOWCAST_MIN_ABS_SCORE:
            continue
        direction = _currency_strength_to_pair_direction(pair, currency=currency, strength=1 if score > 0 else -1)
        if direction is None:
            continue
        normalized = min(1.0, abs(score) / max(MACRO_EVENT_NOWCAST_SCORE_CAP, 1.0))
        confidence = min(0.87, MACRO_EVENT_NOWCAST_BASE_CONFIDENCE + normalized * MACRO_EVENT_NOWCAST_CONFIDENCE_GAIN)
        bonus = MACRO_EVENT_NOWCAST_BONUS * normalized
        lead_min = max(0.0, (event_time - now_utc).total_seconds() / 60.0)
        side = "LONG" if direction == "UP" else "SHORT"
        skew = f"{currency}-positive" if score > 0 else f"{currency}-negative"
        top_evidence = sorted(evidence, key=lambda item: abs(item["score"]), reverse=True)[:3]
        evidence_text = "; ".join(str(item["reason"])[:90] for item in top_evidence)
        title_text = ", ".join(titles[:3])
        out.append(
            ProjectionSignal(
                name=f"macro_event_nowcast_{family.name}",
                timeframe=None,
                direction=direction,
                lead_time_min=lead_min,
                confidence=confidence,
                bonus_magnitude=bonus,
                rationale=(
                    f"{pair} {side} pre-event {family.name} nowcast: {skew} "
                    f"evidence score={score:+.1f} before {event_time.isoformat()} "
                    f"({title_text}); evidence: {evidence_text}"
                ),
            )
        )
    return out


def _upcoming_macro_event_clusters(calendar_path: Path, now_utc: datetime) -> list[dict[str, Any]]:
    try:
        payload = json.loads(calendar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    clusters: dict[tuple[str, str, datetime], dict[str, Any]] = {}
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        currency = str(event.get("currency") or "").upper()
        if currency not in _MACRO_CURRENCY_ALIASES:
            continue
        impact = str(event.get("impact") or "").lower()
        if "high" not in impact and "medium" not in impact:
            continue
        title = str(event.get("title") or event.get("event") or "")
        family = _macro_family_for_event_title(title)
        if family is None:
            continue
        event_time = _parse_iso_utc(event.get("timestamp_utc") or event.get("time_utc"))
        if event_time is None or event_time < now_utc:
            continue
        lead_min = (event_time - now_utc).total_seconds() / 60.0
        if lead_min > MACRO_EVENT_NOWCAST_LOOKAHEAD_MIN:
            continue
        key = (currency, family.name, event_time)
        cluster = clusters.setdefault(
            key,
            {"currency": currency, "family": family, "event_time": event_time, "titles": []},
        )
        if title and title not in cluster["titles"]:
            cluster["titles"].append(title)
    return sorted(clusters.values(), key=lambda item: (item["event_time"], item["currency"], item["family"].name))


def _macro_family_for_event_title(title: str) -> _MacroNowcastFamily | None:
    for family in _MACRO_NOWCAST_FAMILIES:
        if family.event_re.search(title):
            return family
    return None


def _macro_event_evidence_from_news_items(
    news_items_path: Path | None,
    *,
    currency: str,
    family: _MacroNowcastFamily,
    now_utc: datetime,
    event_time: datetime,
) -> list[dict[str, Any]]:
    if news_items_path is None or not news_items_path.exists():
        return []
    try:
        payload = json.loads(news_items_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        published = _parse_iso_utc(item.get("published_at_utc"))
        if published is None or published > now_utc or published > event_time:
            continue
        age_hours = (now_utc - published).total_seconds() / 3600.0
        if age_hours > MACRO_EVENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS:
            continue
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                " ".join(str(topic) for topic in item.get("topics", []) or []),
                " ".join(str(ccy) for ccy in item.get("currencies", []) or []),
                " ".join(str(category) for category in item.get("categories", []) or []),
            ]
        )
        item_currencies = {str(ccy).upper() for ccy in item.get("currencies", []) or []}
        if currency not in item_currencies and not _macro_text_mentions_currency(text, currency):
            continue
        scored = _score_macro_nowcast_text(text, family)
        if scored is None:
            continue
        score, reason = scored
        source = str(item.get("source") or "news")
        out.append({"score": score, "reason": f"{source}: {reason}"})
    return out


def _macro_event_evidence_from_digest(
    news_digest_path: Path | None,
    *,
    currency: str,
    family: _MacroNowcastFamily,
    now_utc: datetime,
    event_time: datetime,
) -> list[dict[str, Any]]:
    if news_digest_path is None or not news_digest_path.exists():
        return []
    try:
        mtime = datetime.fromtimestamp(news_digest_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return []
    if mtime > now_utc or mtime > event_time:
        return []
    age_hours = (now_utc - mtime).total_seconds() / 3600.0
    if age_hours > MACRO_EVENT_NOWCAST_MAX_EVIDENCE_AGE_HOURS:
        return []
    try:
        text = news_digest_path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not family.evidence_re.search(line):
            continue
        if not _macro_text_mentions_currency(line, currency):
            continue
        scored = _score_macro_nowcast_text(line, family)
        if scored is None:
            continue
        score, reason = scored
        out.append({"score": score, "reason": f"digest: {reason}"})
    return out


def _score_macro_nowcast_text(text: str, family: _MacroNowcastFamily) -> tuple[float, str] | None:
    if not family.evidence_re.search(text):
        return None
    score = 0.0
    reasons: list[str] = []
    if family.primary_re.search(text):
        delta = _macro_family_direction_score(text, family, MACRO_EVENT_NOWCAST_PRIMARY_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"primary {family.name} {'hot' if delta > 0 else 'soft'}")
    if family.secondary_re.search(text):
        delta = _macro_family_direction_score(text, family, MACRO_EVENT_NOWCAST_SECONDARY_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"secondary {family.name} {'hot' if delta > 0 else 'soft'}")
    if score == 0.0:
        delta = _macro_family_direction_score(text, family, MACRO_EVENT_NOWCAST_GENERIC_WEIGHT)
        if delta:
            score += delta
            reasons.append(f"generic {family.name} {'hot' if delta > 0 else 'soft'}")
    if score == 0.0:
        return None
    clipped = re.sub(r"\s+", " ", text).strip()
    return score, f"{', '.join(reasons)}: {clipped[:120]}"


def _macro_family_direction_score(text: str, family: _MacroNowcastFamily, weight: float) -> float:
    if family.name == "central_bank":
        if _RATE_CUT_BETS_FADE_RE.search(text) or _RATE_HIKE_BETS_RISE_RE.search(text):
            return weight
        if _RATE_CUT_BETS_RISE_RE.search(text) or _RATE_HIKE_BETS_FADE_RE.search(text):
            return -weight
    if family.name in {"growth", "trade_commodity"} and _GROWTH_DRAG_CONTEXT_RE.search(text):
        return -weight
    if family.lower_is_better_re is not None and family.lower_is_better_re.search(text):
        return _macro_lower_is_better_score(text, family, weight)
    return _macro_higher_is_better_score(text, family, weight)


def _macro_higher_is_better_score(text: str, family: _MacroNowcastFamily, weight: float) -> float:
    strong = bool(family.strong_re.search(text))
    soft = bool(family.soft_re.search(text))
    if strong and not soft:
        return weight
    if soft and not strong:
        return -weight
    return 0.0


def _macro_lower_is_better_score(text: str, family: _MacroNowcastFamily, weight: float) -> float:
    strong = bool(family.strong_re.search(text))
    soft = bool(family.soft_re.search(text))
    if soft and not strong:
        return weight
    if strong and not soft:
        return -weight
    return 0.0


def _macro_text_mentions_currency(text: str, currency: str) -> bool:
    aliases = _MACRO_CURRENCY_ALIASES.get(currency, ())
    for alias in aliases:
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _detect_cross_asset_lag(
    pair: str,
    cross_asset_path: Path,
    pair_chart: Optional[Dict[str, Any]],
) -> List[ProjectionSignal]:
    """When DXY / US10Y / SPX has moved meaningfully but the FX pair
    hasn't caught up yet, project the catch-up direction.

    Rule of thumb:
    - DXY up X% → USD-quote pairs (EUR/USD, GBP/USD, AUD/USD) should
      go DOWN (catch-up SHORT bias).
    - DXY up X% → USD-base pairs (USD/JPY, USD/CAD, USD/CHF) should
      go UP.
    """
    if not cross_asset_path.exists():
        return []
    try:
        payload = json.loads(cross_asset_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    dxy_change_pct = _to_float(((payload.get("synthetic_dxy") or {}).get("change_pct_24h")))
    snapshots = payload.get("snapshots") or payload.get("instruments") or []
    # Older cross-asset snapshots used a flat instrument list. Keep that parser
    # only as compatibility; the current snapshot emits `synthetic_dxy`.
    if dxy_change_pct is None:
        for s in snapshots if isinstance(snapshots, list) else []:
            name = str(s.get("symbol") or s.get("name") or "").upper()
            if name not in ("DXY", "US_DOLLAR_INDEX", "DX_F"):
                continue
            for key in ("change_pct_24h", "change_pct_1h", "change_pct_4h", "change_pct"):
                v = _to_float(s.get(key))
                if v is not None:
                    dxy_change_pct = v
                    break
            break
    if dxy_change_pct is None:
        return []
    if abs(dxy_change_pct) < CROSS_ASSET_MOVE_THRESHOLD_PCT:
        return []
    base, quote = _pair_currencies(pair)
    out: List[ProjectionSignal] = []
    dxy_up = dxy_change_pct > 0
    # USD-quote pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD): DXY up → pair down
    if quote == "USD":
        direction = "DOWN" if dxy_up else "UP"
    elif base == "USD":
        direction = "UP" if dxy_up else "DOWN"
    else:
        return []
    out.append(ProjectionSignal(
        name="cross_asset_dxy_lag",
        timeframe=None, direction=direction,
        lead_time_min=60,
        confidence=min(1.0, abs(dxy_change_pct) / (CROSS_ASSET_MOVE_THRESHOLD_PCT * 3)),
        bonus_magnitude=CROSS_ASSET_LAG_BONUS,
        rationale=f"DXY moved {dxy_change_pct:+.2f}% → {pair} projected {direction} catch-up within ~1h",
    ))
    return out


def _detect_session_expansion(now: Optional[datetime] = None) -> List[ProjectionSignal]:
    """London open (07:00 UTC) and NY open (13:00 UTC) reliably
    produce volatility expansion. Within SESSION_PRE_WINDOW_MIN before
    open = "expansion soon, hold off or pre-position"."""
    now = now or datetime.now(timezone.utc)
    # Anchors in UTC
    london_open = now.replace(hour=7, minute=0, second=0, microsecond=0)
    ny_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
    out: List[ProjectionSignal] = []
    for label, anchor in (("LONDON", london_open), ("NY", ny_open)):
        delta = (anchor - now).total_seconds() / 60.0
        if 0 < delta <= SESSION_PRE_WINDOW_MIN:
            out.append(ProjectionSignal(
                name=f"session_expansion_{label.lower()}",
                timeframe=None, direction="EITHER",
                lead_time_min=delta,
                confidence=1.0 - (delta / SESSION_PRE_WINDOW_MIN),
                bonus_magnitude=SESSION_EXPANSION_BONUS,
                rationale=f"{label} open in {delta:.0f}min — volatility expansion likely",
            ))
    return out


def detect_forward_projections(
    pair_chart: Optional[Dict[str, Any]],
    *,
    pair: str,
    current_price: Optional[float] = None,
    calendar_path: Optional[Path] = None,
    news_digest_path: Optional[Path] = None,
    news_items_path: Optional[Path] = None,
    cross_asset_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> List[ProjectionSignal]:
    """Run all forward projection detectors against a single pair."""
    if _is_disabled():
        return []
    out: List[ProjectionSignal] = []
    if pair_chart:
        pip_factor = 100.0 if pair.endswith("_JPY") else 10000.0
        for view in pair_chart.get("views", []) or []:
            if not isinstance(view, dict):
                continue
            tf = str(view.get("granularity") or "").upper()
            if tf not in ("M5", "M15", "M30", "H1"):
                continue
            out.extend(_detect_bb_squeeze_expansion(view, tf))
            out.extend(_detect_liquidity_sweep_target(view, tf, current_price, pip_factor))
    if calendar_path is not None:
        out.extend(_detect_news_catalyst(pair, calendar_path, now=now))
        out.extend(_detect_event_surprise(pair, calendar_path, now=now))
    if news_digest_path is not None or news_items_path is not None:
        if calendar_path is not None:
            out.extend(
                _detect_us_employment_nowcast(
                    pair,
                    calendar_path=calendar_path,
                    news_digest_path=news_digest_path,
                    news_items_path=news_items_path,
                    now=now,
                )
            )
            out.extend(
                _detect_macro_event_nowcast(
                    pair,
                    calendar_path=calendar_path,
                    news_digest_path=news_digest_path,
                    news_items_path=news_items_path,
                    now=now,
                )
            )
        out.extend(
            _detect_news_theme_followthrough(
                pair,
                news_digest_path=news_digest_path,
                news_items_path=news_items_path,
                calendar_path=calendar_path,
                now=now,
            )
        )
    if cross_asset_path is not None:
        out.extend(_detect_cross_asset_lag(pair, cross_asset_path, pair_chart))
    out.extend(_detect_session_expansion(now=now))
    return out


def aggregate_projection_score(
    signals: List[ProjectionSignal],
    intent_direction: str,
    *,
    hit_rates: Optional[Dict[str, Dict[str, Any]]] = None,
    pair: Optional[str] = None,
    regime: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """Sum aligned projection signals, subtract opposed (half-weight).

    Optional `hit_rates` (per-detector historical hit rate from
    `projection_ledger.compute_hit_rates`) recalibrates each signal's
    `confidence` via `confidence_calibration` — detectors with poor
    track records get dampened; strong ones boosted.

    Confluence multiplier: when ≥ 3 signals agree on intent direction,
    the aligned-side contributions get multiplied by 1.3 (capped via
    PROJECTION_TOTAL_CAP). High-confluence setups are explicitly
    rewarded so the trader's setup-grade A/B/C distinction maps to
    score magnitude.

    `direction == "EITHER"` signals (BB squeeze, news catalyst, session
    expansion) apply their `bonus_magnitude` as-is regardless of intent
    direction. News catalyst is negative (entry penalty).
    """
    from quant_rabbit.strategy.projection_ledger import (
        confidence_calibration,
        select_calibration_signal_name,
    )

    intent_up = intent_direction.upper() == "LONG"
    aligned_count = sum(
        1 for s in signals
        if s.direction != "EITHER" and (s.direction.upper() == "UP") == intent_up
    )
    confluence_mult = 1.3 if aligned_count >= 3 else (1.15 if aligned_count == 2 else 1.0)

    total = 0.0
    rationales: List[str] = []
    for s in signals:
        cal_mult = 1.0
        if hit_rates is not None and pair:
            cal_signal_name = select_calibration_signal_name(
                s.name,
                s.direction,
                pair,
                hit_rates=hit_rates,
                regime=regime,
            )
            cal_mult = confidence_calibration(
                cal_signal_name,
                pair,
                hit_rates=hit_rates,
                regime=regime,
            )
        contribution = s.bonus_magnitude * s.confidence * cal_mult
        cal_note = f" [cal×{cal_mult:.2f}]" if cal_mult != 1.0 else ""
        if s.direction == "EITHER":
            total += contribution
            rationales.append(f"{'+' if contribution >= 0 else ''}{contribution:.1f}{cal_note} {s.rationale}")
        else:
            signal_up = s.direction.upper() == "UP"
            if signal_up == intent_up:
                # Aligned: apply confluence multiplier
                final = contribution * confluence_mult
                total += final
                rationales.append(f"+{final:.1f}{cal_note} {s.rationale}")
            else:
                total -= contribution * 0.5
                rationales.append(f"-{contribution * 0.5:.1f}{cal_note} AGAINST {s.rationale}")
    if total > PROJECTION_TOTAL_CAP:
        total = PROJECTION_TOTAL_CAP
    elif total < -PROJECTION_TOTAL_CAP:
        total = -PROJECTION_TOTAL_CAP
    if confluence_mult > 1.0:
        rationales.insert(0, f"confluence×{confluence_mult:.2f} ({aligned_count} aligned signals)")
    return round(total, 2), rationales
