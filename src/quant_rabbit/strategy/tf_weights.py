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
import hashlib
import math
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

DYNAMIC_TF_POLICY_CONTRACT = "QR_DYNAMIC_TF_POLICY_INPUTS_V1"
DYNAMIC_TF_POLICY_FIELDS = {
    "contract",
    "situation",
    "requested_method",
    "effective_weight_method",
    "news_event_active",
    "pair",
    "atr_percentile_by_timeframe",
    "strategy_edge_multiplier",
}

DYNAMIC_TF_POLICY_EVIDENCE_CONTRACT = "QR_DYNAMIC_TF_POLICY_RAW_EVIDENCE_V1"
DYNAMIC_TF_POLICY_EVIDENCE_FIELDS = {
    "contract",
    "pair",
    "requested_method",
    "classifier_inputs",
    "derived_situation",
    "atr_percentile_by_timeframe",
    "news_evidence",
    "strategy_profile_evidence",
    "evidence_sha256",
}
CLASSIFIER_TIMEFRAMES = ("H1", "H4", "M5", "M15")
MAX_SOURCE_PATH_CHARS = 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_SESSION_CHARS = 64
MAX_REGIME_CHARS = 64
MAX_NEWS_CANDIDATES = 4
MAX_NEWS_TITLE_CHARS = 120
MAX_NEWS_REASON_CHARS = 320
MAX_STRATEGY_CANDIDATES = 4
# This evidence is embedded inside the 16 KiB forecast technical context,
# alongside a ~10.9 KiB seven-frame/failure-proof body.  Its own ceiling must
# therefore be materially smaller than the parent ceiling.
MAX_DYNAMIC_TF_EVIDENCE_BYTES = 4600
_SOURCE_STATUSES = {
    "OK",
    "MISSING",
    "READ_ERROR",
    "LIMIT_EXCEEDED",
    "INVALID_JSON",
    "INVALID_ROOT",
    "NOT_APPLICABLE",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
ALLOWED_EDGE_MULTIPLIERS = {EDGE_MULT_LOW, 1.0, EDGE_MULT_HIGH}
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


def _normalized_optional_token(value: object, *, limit: int) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    return text[:limit]


def build_dynamic_tf_classifier_inputs(
    *,
    session: str | None,
    chart_story: str,
    dominant_regime: str | None,
) -> dict[str, Any]:
    """Canonicalize only the values that the situation classifier consumes."""

    parsed = _extract_per_tf_state(chart_story)
    timeframes: dict[str, dict[str, Any]] = {}
    for timeframe in CLASSIFIER_TIMEFRAMES:
        row = parsed.get(timeframe)
        if row is None:
            timeframes[timeframe] = {
                "present": False,
                "regime": "",
                "adx": 0.0,
            }
            continue
        regime = str(row.get("regime") or "")[:MAX_REGIME_CHARS]
        adx = float(row.get("adx") or 0.0)
        timeframes[timeframe] = {
            "present": True,
            "regime": regime,
            "adx": adx,
        }
    return {
        "session": _normalized_optional_token(session, limit=MAX_SESSION_CHARS),
        "dominant_regime": _normalized_optional_token(
            dominant_regime,
            limit=MAX_REGIME_CHARS,
        ),
        "timeframes": timeframes,
    }


# Backward-local alias for the existing classifier entry points.  The public
# name is also used by forecast_technical_context to freeze an independently
# hashed parent copy of the exact classifier facts.
_build_classifier_inputs = build_dynamic_tf_classifier_inputs


def classify_situation_from_classifier_inputs(
    classifier_inputs: dict[str, Any],
) -> str:
    """Pure situation classifier over the bounded raw classifier evidence."""

    if set(classifier_inputs) != {"session", "dominant_regime", "timeframes"}:
        raise ValueError("dynamic TF classifier schema invalid")
    session = classifier_inputs.get("session")
    if session is not None and (
        not isinstance(session, str)
        or not session
        or session != session.strip().upper()
        or len(session) > MAX_SESSION_CHARS
    ):
        raise ValueError("dynamic TF classifier session invalid")
    dominant_regime = classifier_inputs.get("dominant_regime")
    if dominant_regime is not None and (
        not isinstance(dominant_regime, str)
        or not dominant_regime
        or dominant_regime != dominant_regime.strip().upper()
        or len(dominant_regime) > MAX_REGIME_CHARS
    ):
        raise ValueError("dynamic TF dominant regime invalid")
    timeframes = classifier_inputs.get("timeframes")
    if not isinstance(timeframes, dict) or set(timeframes) != set(
        CLASSIFIER_TIMEFRAMES
    ):
        raise ValueError("dynamic TF classifier timeframes invalid")
    parsed: dict[str, dict[str, Any]] = {}
    for timeframe in CLASSIFIER_TIMEFRAMES:
        row = timeframes.get(timeframe)
        if not isinstance(row, dict) or set(row) != {"present", "regime", "adx"}:
            raise ValueError("dynamic TF classifier row invalid")
        present = row.get("present")
        regime = row.get("regime")
        adx = row.get("adx")
        if not isinstance(present, bool):
            raise ValueError("dynamic TF classifier presence invalid")
        if (
            not isinstance(regime, str)
            or len(regime) > MAX_REGIME_CHARS
            or regime != regime.upper()
        ):
            raise ValueError("dynamic TF classifier regime invalid")
        if isinstance(adx, bool) or not isinstance(adx, (int, float)):
            raise ValueError("dynamic TF classifier ADX invalid")
        adx_value = float(adx)
        if not math.isfinite(adx_value) or adx_value < 0.0 or adx_value > 10_000.0:
            raise ValueError("dynamic TF classifier ADX invalid")
        if not present and (regime or adx_value != 0.0):
            raise ValueError("dynamic TF missing classifier row invalid")
        parsed[timeframe] = {"regime": regime, "adx": adx_value}

    sess = session or ""
    if "ROLLOVER" in sess or "OFF_HOURS" in sess:
        return "ROLLOVER_SWING"

    h1 = parsed["H1"]
    h4 = parsed["H4"]
    m5 = parsed["M5"]
    m15 = parsed["M15"]
    h1_strong = h1["adx"] >= 25.0 and "TREND" in h1["regime"]
    h4_strong = h4["adx"] >= 25.0 and "TREND" in h4["regime"]
    micro_strong = m5["adx"] >= 28.0 or m15["adx"] >= 28.0

    if "NY" in sess or "NEW_YORK" in sess or "NEWYORK" in sess:
        return "NY_TREND" if h1_strong and h4_strong else "NY_OVERLAP"
    if "LONDON" in sess or "LDN" in sess or "EUROPE" in sess:
        if micro_strong and not (h1_strong and h4_strong):
            return "LONDON_IMPULSE"
        if h1_strong or h4_strong:
            return "LONDON_TREND"
        return "LONDON_IMPULSE"
    if "ASIA" in sess or "TOKYO" in sess or "JP" in sess:
        return "ASIA_TREND" if h1_strong and h4_strong else "ASIA_RANGE"
    if dominant_regime:
        if "TREND" in dominant_regime:
            return "NY_TREND"
        if any(
            token in dominant_regime
            for token in ("RANGE", "TRANSITION", "UNCLEAR")
        ):
            return "TRANSITION"
    return "TRANSITION"


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
    return classify_situation_from_classifier_inputs(
        _build_classifier_inputs(
            session=session,
            chart_story=chart_story,
            dominant_regime=dominant_regime,
        )
    )


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
            rr = {}
        raw_ap = rr.get("atr_percentile")
        if raw_ap is None:
            indicators = view.get("indicators")
            raw_fraction = (
                indicators.get("atr_percentile_100")
                if isinstance(indicators, dict)
                else None
            )
            if isinstance(raw_fraction, bool):
                continue
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                continue
            raw_ap = fraction * 100.0
        if isinstance(raw_ap, bool):
            continue
        try:
            ap = float(raw_ap)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(ap) or ap < 0.0 or ap > 100.0:
            continue
        # ``regime_reading.atr_percentile`` is already emitted on a 0..100
        # scale by chart_reader.  Do not reinterpret a legitimate 0.5th/1st
        # percentile as a 0..1 fraction; doing so turns the quietest regime
        # into neutral/high volatility and selects the wrong TF profile.
        out[gran] = round(ap, 4)
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def dynamic_tf_policy_evidence_sha256(evidence: dict[str, Any]) -> str:
    """Return the canonical digest over an evidence body without its hash."""

    body = dict(evidence)
    body.pop("evidence_sha256", None)
    return _sha256(body)


def _canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _canonical_utc(parsed)


def _source_identity(path: Path, *, applicable: bool = True) -> tuple[dict[str, Any], object | None]:
    path_text = path.expanduser().resolve(strict=False).as_posix()
    if not path_text or len(path_text) > MAX_SOURCE_PATH_CHARS:
        raise ValueError("dynamic TF source path invalid")
    if not applicable:
        return {
            "path": path_text,
            "status": "NOT_APPLICABLE",
            "sha256": None,
            "byte_length": None,
        }, None
    try:
        byte_length = path.stat().st_size
    except FileNotFoundError:
        return {
            "path": path_text,
            "status": "MISSING",
            "sha256": None,
            "byte_length": None,
        }, None
    except OSError:
        return {
            "path": path_text,
            "status": "READ_ERROR",
            "sha256": None,
            "byte_length": None,
        }, None
    if byte_length < 0 or byte_length > MAX_SOURCE_BYTES:
        return {
            "path": path_text,
            "status": "LIMIT_EXCEEDED",
            "sha256": None,
            "byte_length": max(0, int(byte_length)),
        }, None
    try:
        raw = path.read_bytes()
    except OSError:
        return {
            "path": path_text,
            "status": "READ_ERROR",
            "sha256": None,
            "byte_length": int(byte_length),
        }, None
    digest = hashlib.sha256(raw).hexdigest()
    identity = {
        "path": path_text,
        "status": "OK",
        "sha256": digest,
        "byte_length": len(raw),
    }
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        identity["status"] = "INVALID_JSON"
        return identity, None
    if not isinstance(payload, dict):
        identity["status"] = "INVALID_ROOT"
        return identity, None
    return identity, payload


def _validate_source_identity(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "status",
        "sha256",
        "byte_length",
    }:
        return False
    path = value.get("path")
    status = value.get("status")
    digest = value.get("sha256")
    byte_length = value.get("byte_length")
    if (
        not isinstance(path, str)
        or not path
        or len(path) > MAX_SOURCE_PATH_CHARS
        or status not in _SOURCE_STATUSES
    ):
        return False
    if byte_length is not None and (
        isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length < 0
    ):
        return False
    if status in {"OK", "INVALID_JSON", "INVALID_ROOT"}:
        return (
            isinstance(digest, str)
            and _SHA256_PATTERN.fullmatch(digest) is not None
            and isinstance(byte_length, int)
            and 0 <= byte_length <= MAX_SOURCE_BYTES
        )
    if status == "LIMIT_EXCEEDED":
        return (
            (byte_length is None or isinstance(byte_length, int))
            and (
                digest is None
                or (
                    isinstance(digest, str)
                    and _SHA256_PATTERN.fullmatch(digest) is not None
                )
            )
        )
    return digest is None and byte_length is None


def _bounded_news_title(value: object) -> str:
    text = str(value or "event")
    text = " ".join(text.split()) or "event"
    return text[:MAX_NEWS_TITLE_CHARS]


def _build_news_evidence(
    *,
    pair: str,
    calendar_path: Path,
    evaluated_at_utc: datetime,
) -> dict[str, Any]:
    source, payload = _source_identity(calendar_path)
    evaluation_time = (
        evaluated_at_utc.replace(tzinfo=timezone.utc)
        if evaluated_at_utc.tzinfo is None
        else evaluated_at_utc.astimezone(timezone.utc)
    )
    events: object = []
    issues: object = []
    if isinstance(payload, dict):
        events = payload.get("events") or payload.get("calendar") or []
        issues = payload.get("issues") or []
        if not isinstance(events, list):
            source["status"] = "INVALID_ROOT"
            events = []
    event_count = len(events) if isinstance(events, list) else 0
    missing_feed_issue = bool(
        event_count == 0
        and isinstance(issues, list)
        and any("MISSING_FOREX_FACTORY_FEED" in str(issue) for issue in issues)
    )
    pair_currencies: set[str] = set()
    if "_" in pair:
        base, quote = pair.split("_", 1)
        pair_currencies = {base.upper(), quote.upper()}
    candidates: list[dict[str, Any]] = []
    for row in events if isinstance(events, list) else []:
        if not isinstance(row, dict):
            continue
        impact = str(row.get("impact") or row.get("importance") or "").upper()
        if impact not in HIGH_IMPACT_TOKENS:
            continue
        currency = str(
            row.get("currency") or row.get("country") or ""
        ).upper()[:16]
        if pair_currencies and currency and currency not in pair_currencies:
            continue
        timestamp = _parse_utc_text(
            row.get("timestamp_utc")
            or row.get("time_utc")
            or row.get("time")
            or row.get("timestamp")
            or row.get("date")
        )
        if timestamp is None:
            continue
        event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        delta = (event_time - evaluation_time).total_seconds()
        if not (
            -(NEWS_WINDOW_MINUTES_AFTER * 60)
            <= delta
            <= NEWS_WINDOW_MINUTES_BEFORE * 60
        ):
            continue
        candidates.append(
            {
                "impact": impact,
                "currency": currency,
                "timestamp_utc": timestamp,
                "title": _bounded_news_title(
                    row.get("title") or row.get("event") or "event"
                ),
            }
        )
        if len(candidates) > MAX_NEWS_CANDIDATES:
            candidates = []
            source["status"] = "LIMIT_EXCEEDED"
            break
    news: dict[str, Any] = {
        "source": source,
        "evaluated_at_utc": _canonical_utc(evaluation_time),
        "event_count": event_count,
        "missing_feed_issue": missing_feed_issue,
        "candidates": candidates,
        "active": False,
        "reason": "",
    }
    news["active"], news["reason"] = _news_active_from_evidence(pair, news)
    return news


def _news_active_from_evidence(
    pair: str,
    news_evidence: dict[str, Any],
) -> tuple[bool, str]:
    source = news_evidence.get("source") or {}
    if source.get("status") == "LIMIT_EXCEEDED":
        return True, "calendar_unavailable"
    if source.get("status") != "OK":
        return False, ""
    if news_evidence.get("event_count") == 0 and news_evidence.get(
        "missing_feed_issue"
    ):
        return True, "calendar_unavailable"
    evaluated_text = news_evidence.get("evaluated_at_utc")
    try:
        evaluated_at = datetime.fromisoformat(
            str(evaluated_text).replace("Z", "+00:00")
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError("dynamic TF news evaluation time invalid")
    pair_currencies: set[str] = set()
    if "_" in pair:
        base, quote = pair.split("_", 1)
        pair_currencies = {base.upper(), quote.upper()}
    for candidate in news_evidence.get("candidates") or []:
        currency = candidate.get("currency") or ""
        if pair_currencies and currency and currency not in pair_currencies:
            continue
        timestamp = candidate.get("timestamp_utc")
        if timestamp is None:
            continue
        try:
            event_time = datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )
        except (TypeError, ValueError, OverflowError):
            continue
        delta = (event_time - evaluated_at).total_seconds()
        if (
            -(NEWS_WINDOW_MINUTES_AFTER * 60)
            <= delta
            <= NEWS_WINDOW_MINUTES_BEFORE * 60
        ):
            reason = f"{currency}:{candidate.get('title') or 'event'} {int(delta / 60)}m"
            return True, reason[:MAX_NEWS_REASON_CHARS]
    return False, ""


def _build_strategy_profile_evidence(
    *,
    pair: str,
    pair_was_supplied: bool,
    requested_method: str | None,
    strategy_profile_path: Path,
) -> dict[str, Any]:
    source, payload = _source_identity(
        strategy_profile_path,
        applicable=pair_was_supplied and requested_method is not None,
    )
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        profiles = payload.get("profiles") or []
        if not isinstance(profiles, list):
            source["status"] = "INVALID_ROOT"
            profiles = []
        for index, entry in enumerate(profiles):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("pair") or "").strip().upper() != pair:
                continue
            if (
                str(entry.get("method") or "").strip().upper()
                != requested_method
            ):
                continue
            raw_count = entry.get("positive_evidence_n") or 0
            raw_live_net = entry.get("live_net_jpy") or 0.0
            if isinstance(raw_count, bool) or isinstance(raw_live_net, bool):
                continue
            try:
                count = int(raw_count)
                live_net = float(raw_live_net)
            except (TypeError, ValueError, OverflowError):
                continue
            if count < 0 or not math.isfinite(live_net):
                continue
            candidate = {
                "source_index": index,
                "pair": pair,
                "method": requested_method,
                "positive_evidence_n": count,
                "live_net_jpy": live_net,
            }
            candidates.append(candidate)
            if len(candidates) > MAX_STRATEGY_CANDIDATES:
                candidates = []
                source["status"] = "LIMIT_EXCEEDED"
                break
    selected = _select_strategy_candidate(candidates)
    evidence: dict[str, Any] = {
        "source": source,
        "candidates": candidates,
        "selected": selected,
        "multiplier": 1.0,
        "label": "",
    }
    evidence["multiplier"], evidence["label"] = (
        _strategy_multiplier_from_evidence(pair, requested_method, evidence)
    )
    return evidence


def _select_strategy_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Purely select the first highest-evidence exact-scope profile row."""

    selected: dict[str, Any] | None = None
    for candidate in candidates:
        if (
            selected is None
            or int(candidate["positive_evidence_n"])
            > int(selected["positive_evidence_n"])
        ):
            selected = candidate
    return dict(selected) if selected is not None else None


def _strategy_multiplier_from_evidence(
    pair: str,
    requested_method: str | None,
    strategy_evidence: dict[str, Any],
) -> tuple[float, str]:
    selected = strategy_evidence.get("selected")
    if not isinstance(selected, dict):
        return 1.0, ""
    if selected.get("pair") != pair or selected.get("method") != requested_method:
        raise ValueError("dynamic TF strategy selection binding invalid")
    count = int(selected.get("positive_evidence_n"))
    live_net = float(selected.get("live_net_jpy"))
    if count >= EDGE_EVIDENCE_HIGH and live_net >= EDGE_LIVE_NET_HIGH:
        return (
            EDGE_MULT_HIGH,
            f"edge:{pair}:{requested_method}:n={count}/live={live_net:.0f}",
        )
    if count <= EDGE_EVIDENCE_LOW or live_net < 0.0:
        return (
            EDGE_MULT_LOW,
            f"weak_edge:{pair}:{requested_method}:n={count}/live={live_net:.0f}",
        )
    return 1.0, ""


def _compact_oversized_dynamic_tf_evidence(evidence: dict[str, Any]) -> None:
    """Fail closed when valid UTF-8 evidence exceeds the GPT packet budget."""

    news = evidence["news_evidence"]
    news["source"]["status"] = "LIMIT_EXCEEDED"
    news["candidates"] = []
    news["active"], news["reason"] = _news_active_from_evidence(
        str(evidence["pair"]),
        news,
    )

    strategy = evidence["strategy_profile_evidence"]
    strategy["source"]["status"] = "LIMIT_EXCEEDED"
    strategy["candidates"] = []
    strategy["selected"] = None
    strategy["multiplier"] = 1.0
    strategy["label"] = ""


def build_dynamic_tf_policy_evidence(
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
) -> dict[str, Any]:
    """Build bounded point-in-time evidence for the dynamic-TF policy."""

    requested_method = (method or "").strip().upper() or None
    if requested_method not in METHOD_OVERLAY:
        requested_method = None
    pair_input = str(pair or "").strip()
    pair_key = pair_input.upper() or "UNKNOWN"
    if len(pair_key) > 16:
        raise ValueError("dynamic TF pair invalid")
    classifier_inputs = _build_classifier_inputs(
        session=session,
        chart_story=chart_story,
        dominant_regime=dominant_regime,
    )
    atr_values = _atr_percentile_from_views(pair_chart)
    evaluated_at = now_utc or datetime.now(timezone.utc)
    evidence: dict[str, Any] = {
        "contract": DYNAMIC_TF_POLICY_EVIDENCE_CONTRACT,
        "pair": pair_key,
        "requested_method": requested_method,
        "classifier_inputs": classifier_inputs,
        "derived_situation": classify_situation_from_classifier_inputs(
            classifier_inputs
        ),
        "atr_percentile_by_timeframe": {
            timeframe: atr_values.get(timeframe)
            for timeframe in BASELINE_WEIGHTS
        },
        "news_evidence": _build_news_evidence(
            pair=pair_key,
            calendar_path=calendar_path or DEFAULT_CALENDAR_PATH,
            evaluated_at_utc=evaluated_at,
        ),
        "strategy_profile_evidence": _build_strategy_profile_evidence(
            pair=pair_key,
            pair_was_supplied=bool(pair_input),
            requested_method=requested_method,
            strategy_profile_path=(
                strategy_profile_path or DEFAULT_STRATEGY_PROFILE_PATH
            ),
        ),
    }
    evidence["evidence_sha256"] = dynamic_tf_policy_evidence_sha256(evidence)
    if len(_canonical_json_bytes(evidence)) > MAX_DYNAMIC_TF_EVIDENCE_BYTES:
        _compact_oversized_dynamic_tf_evidence(evidence)
        evidence["evidence_sha256"] = dynamic_tf_policy_evidence_sha256(
            evidence
        )
    if len(_canonical_json_bytes(evidence)) > MAX_DYNAMIC_TF_EVIDENCE_BYTES:
        raise ValueError("DYNAMIC_TF_EVIDENCE_LIMIT_EXCEEDED")
    valid, error = verify_dynamic_tf_policy_evidence(evidence)
    if not valid:
        raise ValueError(error or "dynamic TF evidence invalid")
    return evidence


def verify_dynamic_tf_policy_evidence(
    value: object,
) -> tuple[bool, str | None]:
    """Verify evidence shape, bounds, digest, and all stored derivations."""

    try:
        if not isinstance(value, dict) or set(value) != DYNAMIC_TF_POLICY_EVIDENCE_FIELDS:
            return False, "DYNAMIC_TF_EVIDENCE_SCHEMA_INVALID"
        evidence = dict(value)
        if evidence.get("contract") != DYNAMIC_TF_POLICY_EVIDENCE_CONTRACT:
            return False, "DYNAMIC_TF_EVIDENCE_CONTRACT_INVALID"
        if len(_canonical_json_bytes(evidence)) > MAX_DYNAMIC_TF_EVIDENCE_BYTES:
            return False, "DYNAMIC_TF_EVIDENCE_LIMIT_EXCEEDED"
        pair = evidence.get("pair")
        if (
            not isinstance(pair, str)
            or not pair
            or pair != pair.strip().upper()
            or len(pair) > 16
        ):
            return False, "DYNAMIC_TF_EVIDENCE_PAIR_INVALID"
        requested_method = evidence.get("requested_method")
        if requested_method is not None and requested_method not in METHOD_OVERLAY:
            return False, "DYNAMIC_TF_EVIDENCE_METHOD_INVALID"
        classifier_inputs = evidence.get("classifier_inputs")
        if not isinstance(classifier_inputs, dict):
            return False, "DYNAMIC_TF_EVIDENCE_CLASSIFIER_INVALID"
        expected_situation = classify_situation_from_classifier_inputs(
            classifier_inputs
        )
        if evidence.get("derived_situation") != expected_situation:
            return False, "DYNAMIC_TF_EVIDENCE_SITUATION_MISMATCH"

        atr_values = evidence.get("atr_percentile_by_timeframe")
        if not isinstance(atr_values, dict) or set(atr_values) != set(
            BASELINE_WEIGHTS
        ):
            return False, "DYNAMIC_TF_EVIDENCE_ATR_INVALID"
        for timeframe in BASELINE_WEIGHTS:
            atr = atr_values.get(timeframe)
            if atr is None:
                continue
            if isinstance(atr, bool) or not isinstance(atr, (int, float)):
                return False, "DYNAMIC_TF_EVIDENCE_ATR_INVALID"
            parsed_atr = float(atr)
            if (
                not math.isfinite(parsed_atr)
                or parsed_atr < 0.0
                or parsed_atr > 100.0
            ):
                return False, "DYNAMIC_TF_EVIDENCE_ATR_INVALID"

        news = evidence.get("news_evidence")
        if not isinstance(news, dict) or set(news) != {
            "source",
            "evaluated_at_utc",
            "event_count",
            "missing_feed_issue",
            "candidates",
            "active",
            "reason",
        }:
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_INVALID"
        if not _validate_source_identity(news.get("source")):
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_SOURCE_INVALID"
        evaluated_at = news.get("evaluated_at_utc")
        if _parse_utc_text(evaluated_at) != evaluated_at:
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_TIME_INVALID"
        event_count = news.get("event_count")
        if (
            isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or event_count < 0
            or event_count > 1_000_000
            or not isinstance(news.get("missing_feed_issue"), bool)
        ):
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_INVALID"
        candidates = news.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > MAX_NEWS_CANDIDATES:
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_INVALID"
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "impact",
                "currency",
                "timestamp_utc",
                "title",
            }:
                return False, "DYNAMIC_TF_EVIDENCE_NEWS_CANDIDATE_INVALID"
            impact = candidate.get("impact")
            currency = candidate.get("currency")
            timestamp = candidate.get("timestamp_utc")
            title = candidate.get("title")
            if (
                impact not in HIGH_IMPACT_TOKENS
                or not isinstance(currency, str)
                or currency != currency.upper()
                or len(currency) > 16
                or not isinstance(title, str)
                or not title
                or title != " ".join(title.split())
                or len(title) > MAX_NEWS_TITLE_CHARS
                or (timestamp is not None and _parse_utc_text(timestamp) != timestamp)
            ):
                return False, "DYNAMIC_TF_EVIDENCE_NEWS_CANDIDATE_INVALID"
        if not isinstance(news.get("active"), bool):
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_INVALID"
        reason = news.get("reason")
        if not isinstance(reason, str) or len(reason) > MAX_NEWS_REASON_CHARS:
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_INVALID"
        expected_news = _news_active_from_evidence(pair, news)
        if (news.get("active"), reason) != expected_news:
            return False, "DYNAMIC_TF_EVIDENCE_NEWS_MISMATCH"

        strategy = evidence.get("strategy_profile_evidence")
        if not isinstance(strategy, dict) or set(strategy) != {
            "source",
            "candidates",
            "selected",
            "multiplier",
            "label",
        }:
            return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_INVALID"
        if not _validate_source_identity(strategy.get("source")):
            return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SOURCE_INVALID"
        strategy_candidates = strategy.get("candidates")
        if (
            not isinstance(strategy_candidates, list)
            or len(strategy_candidates) > MAX_STRATEGY_CANDIDATES
        ):
            return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_INVALID"
        seen_source_indexes: set[int] = set()
        for candidate in strategy_candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "source_index",
                "pair",
                "method",
                "positive_evidence_n",
                "live_net_jpy",
            }:
                return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_INVALID"
            source_index = candidate.get("source_index")
            count = candidate.get("positive_evidence_n")
            live_net = candidate.get("live_net_jpy")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
                or source_index in seen_source_indexes
                or candidate.get("pair") != pair
                or candidate.get("method") != requested_method
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or isinstance(live_net, bool)
                or not isinstance(live_net, (int, float))
                or not math.isfinite(float(live_net))
            ):
                return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_INVALID"
            seen_source_indexes.add(source_index)
        selected = strategy.get("selected")
        if selected is not None:
            if not isinstance(selected, dict) or set(selected) != {
                "source_index",
                "pair",
                "method",
                "positive_evidence_n",
                "live_net_jpy",
            }:
                return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_INVALID"
            source_index = selected.get("source_index")
            count = selected.get("positive_evidence_n")
            live_net = selected.get("live_net_jpy")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
                or selected.get("pair") != pair
                or selected.get("method") != requested_method
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or isinstance(live_net, bool)
                or not isinstance(live_net, (int, float))
                or not math.isfinite(float(live_net))
            ):
                return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_INVALID"
        if selected != _select_strategy_candidate(strategy_candidates):
            return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_MISMATCH"
        expected_multiplier, expected_label = _strategy_multiplier_from_evidence(
            pair,
            requested_method,
            strategy,
        )
        multiplier = strategy.get("multiplier")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or float(multiplier) != expected_multiplier
            or strategy.get("label") != expected_label
        ):
            return False, "DYNAMIC_TF_EVIDENCE_STRATEGY_MISMATCH"
        digest = evidence.get("evidence_sha256")
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or digest != dynamic_tf_policy_evidence_sha256(evidence)
        ):
            return False, "DYNAMIC_TF_EVIDENCE_SHA_MISMATCH"
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return False, "DYNAMIC_TF_EVIDENCE_INVALID"
    return True, None


def derive_dynamic_tf_policy_from_evidence(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], str]:
    """Purely derive policy inputs, normalized weights, and audit label."""

    valid, error = verify_dynamic_tf_policy_evidence(evidence)
    if not valid:
        raise ValueError(error or "dynamic TF evidence invalid")
    pair = str(evidence["pair"])
    requested_method = evidence.get("requested_method")
    news = evidence["news_evidence"]
    strategy = evidence["strategy_profile_evidence"]
    news_active = news.get("active") is True
    situation = "NY_OVERLAP" if news_active else evidence["derived_situation"]
    effective_method = "EVENT_RISK" if news_active else requested_method
    policy_inputs = {
        "contract": DYNAMIC_TF_POLICY_CONTRACT,
        "situation": situation,
        "requested_method": requested_method,
        "effective_weight_method": effective_method,
        "news_event_active": news_active,
        "pair": pair,
        "atr_percentile_by_timeframe": dict(
            evidence["atr_percentile_by_timeframe"]
        ),
        "strategy_edge_multiplier": float(strategy["multiplier"]),
    }
    weights = dynamic_tf_weights_from_policy_inputs(policy_inputs)
    parts = [situation]
    if pair in PAIR_OVERLAY:
        parts.append(f"pair:{pair}")
    if news_active:
        parts.append(f"news:{news['reason']}")
    hot = [
        timeframe
        for timeframe in BASELINE_WEIGHTS
        if evidence["atr_percentile_by_timeframe"].get(timeframe) is not None
        and float(evidence["atr_percentile_by_timeframe"][timeframe])
        >= ATR_PERCENTILE_HIGH
    ]
    if hot:
        parts.append(f"atr_hot:{'/'.join(hot)}")
    if strategy["label"]:
        parts.append(str(strategy["label"]))
    return policy_inputs, weights, " | ".join(parts)


def dynamic_tf_weights_from_policy_inputs(
    policy_inputs: dict[str, Any],
) -> dict[str, float]:
    """Purely recompute weights from a bounded, receipt-safe input set.

    File reads, wall-clock news lookup, and situation classification happen
    before this boundary.  Both the live builder and receipt verifier use this
    same arithmetic so changing weights and merely re-hashing the artifact
    cannot forge a valid dynamic-TF policy result.
    """

    if set(policy_inputs) != DYNAMIC_TF_POLICY_FIELDS:
        raise ValueError("dynamic TF policy schema invalid")
    if policy_inputs.get("contract") != DYNAMIC_TF_POLICY_CONTRACT:
        raise ValueError("dynamic TF policy contract invalid")
    situation = policy_inputs.get("situation")
    if situation not in SITUATION_WEIGHTS:
        raise ValueError("dynamic TF situation invalid")
    requested_method = policy_inputs.get("requested_method")
    if requested_method is not None and requested_method not in METHOD_OVERLAY:
        raise ValueError("dynamic TF requested method invalid")
    effective_method = policy_inputs.get("effective_weight_method")
    if effective_method is not None and effective_method not in METHOD_OVERLAY:
        raise ValueError("dynamic TF effective method invalid")
    news_active = policy_inputs.get("news_event_active")
    if not isinstance(news_active, bool):
        raise ValueError("dynamic TF news flag invalid")
    if news_active:
        if situation != "NY_OVERLAP" or effective_method != "EVENT_RISK":
            raise ValueError("dynamic TF news override invalid")
    elif effective_method != requested_method:
        raise ValueError("dynamic TF method binding invalid")
    pair = policy_inputs.get("pair")
    if (
        not isinstance(pair, str)
        or not pair
        or pair != pair.strip().upper()
        or len(pair) > 16
    ):
        raise ValueError("dynamic TF pair invalid")
    edge_multiplier = policy_inputs.get("strategy_edge_multiplier")
    if isinstance(edge_multiplier, bool) or not isinstance(
        edge_multiplier, (int, float)
    ):
        raise ValueError("dynamic TF edge multiplier invalid")
    edge_multiplier = float(edge_multiplier)
    if edge_multiplier not in ALLOWED_EDGE_MULTIPLIERS:
        raise ValueError("dynamic TF edge multiplier invalid")
    atr_values = policy_inputs.get("atr_percentile_by_timeframe")
    if not isinstance(atr_values, dict) or set(atr_values) != set(BASELINE_WEIGHTS):
        raise ValueError("dynamic TF ATR inputs invalid")
    parsed_atr: dict[str, float | None] = {}
    for timeframe in BASELINE_WEIGHTS:
        value = atr_values.get(timeframe)
        if value is None:
            parsed_atr[timeframe] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("dynamic TF ATR input invalid")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 100.0:
            raise ValueError("dynamic TF ATR input invalid")
        parsed_atr[timeframe] = parsed

    base = SITUATION_WEIGHTS[situation].copy()
    overlay = METHOD_OVERLAY.get(effective_method, {}) if effective_method else {}
    for timeframe in base:
        overlay_multiplier = overlay.get(timeframe, 1.0)
        adjusted = 1.0 + (overlay_multiplier - 1.0) * edge_multiplier
        base[timeframe] *= adjusted

    pair_overlay = PAIR_OVERLAY.get(pair, {})
    for timeframe in base:
        base[timeframe] *= pair_overlay.get(timeframe, 1.0)

    for timeframe, atr_percentile in parsed_atr.items():
        if atr_percentile is None:
            continue
        if atr_percentile >= ATR_PERCENTILE_HIGH:
            base[timeframe] *= ATR_HIGH_BOOST
        elif atr_percentile <= ATR_PERCENTILE_LOW:
            base[timeframe] *= ATR_LOW_DAMP

    total = sum(base.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("dynamic TF total invalid")
    return {timeframe: value / total for timeframe, value in base.items()}


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
    include_trace: bool = False,
    include_evidence: bool = False,
) -> (
    tuple[dict[str, float], str]
    | tuple[dict[str, float], str, dict[str, Any]]
    | tuple[dict[str, float], str, dict[str, Any], dict[str, Any]]
):
    """Pick a per-TF weight dict for the current situation × method.

    Returns ``(weights, situation_label)``. With ``include_trace=True``, also
    returns the derived policy inputs. With both ``include_trace`` and
    ``include_evidence``, the fourth value is the verified raw evidence.
    weights sums to 1.0. Combines
    in order:
      1. Situation profile (session × dominant_regime × ADX)
      2. Method overlay (RANGE/BREAKOUT/TREND/EVENT)
      3. Pair character overlay (USD_JPY slow vs JPY crosses fast etc)
      4. ATR percentile boost (high-vol TF → +20%, low-vol → -15%)
      5. News calendar override (force EVENT_RISK overlay near events)
    """
    evidence = build_dynamic_tf_policy_evidence(
        session=session,
        chart_story=chart_story,
        dominant_regime=dominant_regime,
        method=method,
        pair=pair,
        pair_chart=pair_chart,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=now_utc,
    )
    policy_inputs, weights, label = derive_dynamic_tf_policy_from_evidence(
        evidence
    )
    if include_trace and include_evidence:
        return weights, label, policy_inputs, evidence
    if include_trace:
        return weights, label, policy_inputs
    if include_evidence:
        return weights, label, evidence
    return weights, label
