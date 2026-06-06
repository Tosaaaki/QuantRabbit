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
    snapshots = payload.get("snapshots") or payload.get("instruments") or []
    if not snapshots:
        return []
    # Find DXY snapshot. Schema varies; try common shapes.
    dxy_change_pct = None
    for s in snapshots if isinstance(snapshots, list) else []:
        name = str(s.get("symbol") or s.get("name") or "").upper()
        if name not in ("DXY", "US_DOLLAR_INDEX", "DX_F"):
            continue
        # change_pct_24h, change_pct_1h, change_pct
        for key in ("change_pct_1h", "change_pct_4h", "change_pct"):
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
