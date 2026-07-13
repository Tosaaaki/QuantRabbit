"""Normalized point-in-time technical context for forecast evaluation.

The directional forecaster already consumes rich chart state, but historical
rows used to retain only its final scores and prose drivers.  This module
freezes a small, deterministic subset of the chart state at forecast creation
time so later replay can test conditions without reconstructing them from
future or mutable artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from quant_rabbit.instruments import instrument_pip_factor


SCHEMA_VERSION = "technical_context_v1"
MAX_CONTEXT_BYTES = 8192
TOP_LEVEL_FIELDS = {
    "schema_version",
    "identity",
    "regime",
    "volatility",
    "execution",
    "location",
    "structure",
    "families",
    "completeness",
    "context_sha256",
}
CONTEXT_TIMEFRAMES = ("M5", "M15", "H1", "H4")
REQUIRED_FIELDS = (
    "identity.pair",
    "regime.dominant",
    "volatility.primary_atr_band",
    "execution.spread_band",
    "location.range_location_24h",
    "structure.primary_direction",
)


def build_forecast_technical_context(
    pair_chart: Mapping[str, Any] | None,
    *,
    pair: str,
    current_price: float | None,
    spread_pips: float | None,
) -> dict[str, Any]:
    """Return a bounded content-addressed forecast-time context.

    Buckets are descriptive audit cohorts, not trading thresholds.  Raw values
    are retained beside every bucket so future evaluators can reproduce the
    classification exactly.
    """

    chart = pair_chart if isinstance(pair_chart, Mapping) else {}
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), Mapping) else {}
    views = _views_by_timeframe(chart)

    regime_by_timeframe: dict[str, str] = {}
    atr_percentile_by_timeframe: dict[str, float | None] = {}
    atr_band_by_timeframe: dict[str, str] = {}
    atr_pips_by_timeframe: dict[str, float | None] = {}
    structure_by_timeframe: dict[str, dict[str, Any]] = {}
    family_scores_by_timeframe: dict[str, dict[str, float | None]] = {}
    for timeframe in CONTEXT_TIMEFRAMES:
        view = views.get(timeframe) or {}
        reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), Mapping) else {}
        indicators = view.get("indicators") if isinstance(view.get("indicators"), Mapping) else {}
        regime_by_timeframe[timeframe] = _normalized_regime(
            reading.get("state") or view.get("regime")
        )
        atr_percentile = _percent_0_100(
            reading.get("atr_percentile")
            if reading.get("atr_percentile") is not None
            else indicators.get("atr_percentile_100")
        )
        atr_percentile_by_timeframe[timeframe] = _round_or_none(atr_percentile, 4)
        atr_band_by_timeframe[timeframe] = _atr_band(atr_percentile)
        atr_pips_by_timeframe[timeframe] = _round_or_none(
            _number(indicators.get("atr_pips")), 6
        )
        structure_by_timeframe[timeframe] = _latest_close_confirmed_structure(view)
        families = view.get("family_scores") if isinstance(view.get("family_scores"), Mapping) else {}
        family_scores_by_timeframe[timeframe] = {
            "trend_score": _round_or_none(_number((families or {}).get("trend_score")), 6),
            "mean_reversion_score": _round_or_none(
                _number((families or {}).get("mean_rev_score")), 6
            ),
            "breakout_score": _round_or_none(_number((families or {}).get("breakout_score")), 6),
            "disagreement": _round_or_none(_number((families or {}).get("disagreement")), 6),
        }

    dominant_regime = _normalized_regime((confluence or {}).get("dominant_regime"))
    primary_regime = _first_known(
        regime_by_timeframe.get("M15"),
        regime_by_timeframe.get("H1"),
        dominant_regime,
    )
    primary_atr_percentile = _first_number(
        atr_percentile_by_timeframe.get("M15"),
        atr_percentile_by_timeframe.get("M5"),
        _percent_0_100((confluence or {}).get("atr_percentile_24h")),
    )
    primary_atr_band = _atr_band(primary_atr_percentile)
    m5_atr_pips = atr_pips_by_timeframe.get("M5")
    spread = _number(spread_pips)
    spread_to_m5_atr = (
        spread / m5_atr_pips
        if spread is not None and m5_atr_pips is not None and m5_atr_pips > 0.0
        else None
    )
    location_24h_value = _fraction_0_1((confluence or {}).get("price_percentile_24h"))
    location_7d_value = _fraction_0_1((confluence or {}).get("price_percentile_7d"))
    primary_structure = _primary_structure(structure_by_timeframe)

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {"pair": str(pair or "").strip().upper() or "UNKNOWN"},
        "regime": {
            "dominant": dominant_regime,
            "primary": primary_regime,
            "by_timeframe": regime_by_timeframe,
        },
        "volatility": {
            "primary_atr_percentile": _round_or_none(primary_atr_percentile, 4),
            "primary_atr_band": primary_atr_band,
            "atr_percentile_by_timeframe": atr_percentile_by_timeframe,
            "atr_band_by_timeframe": atr_band_by_timeframe,
            "atr_pips_by_timeframe": atr_pips_by_timeframe,
        },
        "execution": {
            "spread_pips": _round_or_none(spread, 6),
            "m5_atr_pips": _round_or_none(m5_atr_pips, 6),
            "spread_to_m5_atr": _round_or_none(spread_to_m5_atr, 6),
            "spread_band": _spread_band(spread_to_m5_atr),
        },
        "location": {
            "current_price": _round_or_none(_number(current_price), 8),
            "price_percentile_24h": _round_or_none(location_24h_value, 6),
            "range_location_24h": _range_location(location_24h_value),
            "price_percentile_7d": _round_or_none(location_7d_value, 6),
            "range_location_7d": _range_location(location_7d_value),
        },
        "structure": {
            "primary_timeframe": primary_structure.get("timeframe"),
            "primary_kind": primary_structure.get("kind"),
            "primary_direction": primary_structure.get("direction", "UNKNOWN"),
            "by_timeframe": structure_by_timeframe,
        },
        "families": {"by_timeframe": family_scores_by_timeframe},
    }
    missing = _missing_required_fields(body)
    body["completeness"] = {
        "required_fields": list(REQUIRED_FIELDS),
        "missing_fields": missing,
        "complete": not missing,
    }
    body["context_sha256"] = technical_context_sha256(body)
    return body


def verify_forecast_technical_context(
    value: object,
    *,
    pair: str | None = None,
    current_price: float | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(value, Mapping):
        return False, "TECHNICAL_CONTEXT_MISSING"
    if value.get("schema_version") != SCHEMA_VERSION:
        return False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if set(value) != TOP_LEVEL_FIELDS:
        return False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    try:
        encoded_size = len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    if encoded_size > MAX_CONTEXT_BYTES:
        return False, "TECHNICAL_CONTEXT_TOO_LARGE"
    stored = str(value.get("context_sha256") or "").strip().lower()
    try:
        expected = technical_context_sha256(value)
    except (TypeError, ValueError):
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    if len(stored) != 64 or stored != expected:
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    identity = value.get("identity") if isinstance(value.get("identity"), Mapping) else {}
    if pair is not None:
        stored_pair = str((identity or {}).get("pair") or "").upper()
        if stored_pair != str(pair or "").strip().upper():
            return False, "TECHNICAL_CONTEXT_PAIR_MISMATCH"
    if current_price is not None:
        location = value.get("location") if isinstance(value.get("location"), Mapping) else {}
        stored_price = _number((location or {}).get("current_price"))
        expected_price = _number(current_price)
        if (
            stored_price is None
            or expected_price is None
        ):
            return False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"
        pair_name = str(pair or (identity or {}).get("pair") or "").upper()
        try:
            quote_unit = 1.0 / (float(instrument_pip_factor(pair_name)) * 10.0)
        except (TypeError, ValueError, ZeroDivisionError):
            quote_unit = 1e-8
        if abs(stored_price - expected_price) > quote_unit + 1e-12:
            return False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"
    return True, None


def technical_context_sha256(value: Mapping[str, Any]) -> str:
    body = {str(key): item for key, item in value.items() if key != "context_sha256"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _views_by_timeframe(chart: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for raw in chart.get("views") or []:
        if not isinstance(raw, Mapping):
            continue
        timeframe = str(raw.get("granularity") or "").upper()
        if timeframe in CONTEXT_TIMEFRAMES and timeframe not in out:
            out[timeframe] = raw
    return out


def _latest_close_confirmed_structure(view: Mapping[str, Any]) -> dict[str, Any]:
    structure = view.get("structure") if isinstance(view.get("structure"), Mapping) else {}
    latest: tuple[float, dict[str, Any]] | None = None
    for order, raw in enumerate((structure or {}).get("structure_events") or []):
        if not isinstance(raw, Mapping) or not bool(raw.get("close_confirmed")):
            continue
        kind = str(raw.get("kind") or "").upper().split(":", 1)[0]
        direction = "UP" if kind.endswith("_UP") else "DOWN" if kind.endswith("_DOWN") else None
        if direction is None:
            continue
        index = _number(raw.get("index"))
        rank = index if index is not None else float(order)
        item = {
            "kind": kind,
            "direction": direction,
            "index": _round_or_none(index, 6),
            "timestamp_utc": str(raw.get("timestamp") or "").strip() or None,
        }
        if latest is None or rank >= latest[0]:
            latest = (rank, item)
    return latest[1] if latest is not None else {
        "kind": None,
        "direction": "UNKNOWN",
        "index": None,
        "timestamp_utc": None,
    }


def _primary_structure(by_timeframe: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for timeframe in ("H1", "M15", "M5", "H4"):
        item = by_timeframe.get(timeframe) or {}
        if item.get("direction") in {"UP", "DOWN"}:
            return {"timeframe": timeframe, **dict(item)}
    return {"timeframe": None, "kind": None, "direction": "UNKNOWN"}


def _normalized_regime(value: object) -> str:
    text = str(value or "").upper().strip()
    if not text:
        return "UNKNOWN"
    if "BREAKOUT_PENDING" in text:
        return "BREAKOUT_PENDING"
    if "RANGE" in text:
        return "RANGE"
    if "TREND_STRONG" in text:
        return "TREND_STRONG"
    if "TREND_WEAK" in text:
        return "TREND_WEAK"
    if "TREND_UP" in text or text == "UP":
        return "TREND_UP"
    if "TREND_DOWN" in text or text == "DOWN":
        return "TREND_DOWN"
    if "TRANSITION" in text:
        return "TRANSITION"
    return text[:32]


def _atr_band(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 25.0:
        return "LOW"
    if value >= 75.0:
        return "HIGH"
    return "NORMAL"


def _spread_band(spread_to_atr: float | None) -> str:
    if spread_to_atr is None:
        return "UNKNOWN"
    if spread_to_atr <= 0.25:
        return "TIGHT"
    if spread_to_atr <= 0.75:
        return "NORMAL"
    return "WIDE"


def _range_location(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 1.0 / 3.0:
        return "LOWER"
    if value >= 2.0 / 3.0:
        return "UPPER"
    return "MIDDLE"


def _missing_required_fields(body: Mapping[str, Any]) -> list[str]:
    values = {
        "identity.pair": ((body.get("identity") or {}).get("pair")),
        "regime.dominant": ((body.get("regime") or {}).get("dominant")),
        "volatility.primary_atr_band": ((body.get("volatility") or {}).get("primary_atr_band")),
        "execution.spread_band": ((body.get("execution") or {}).get("spread_band")),
        "location.range_location_24h": ((body.get("location") or {}).get("range_location_24h")),
        "structure.primary_direction": ((body.get("structure") or {}).get("primary_direction")),
    }
    return [
        name
        for name in REQUIRED_FIELDS
        if values.get(name) in {None, "", "UNKNOWN"}
    ]


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent_0_100(value: object) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return min(100.0, max(0.0, number))


def _fraction_0_1(value: object) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _round_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _first_number(*values: object) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _first_known(*values: object) -> str:
    for value in values:
        text = str(value or "").strip().upper()
        if text and text != "UNKNOWN":
            return text
    return "UNKNOWN"
