"""Canonical, bounded M5 failed-break evidence.

The method label ``BREAKOUT_FAILURE`` is not evidence.  This module freezes
the exact last complete M5 candle plus its twenty predecessors and recomputes
the close-back-inside predicate from those immutable inputs.  Both forecast
context verification and intent generation use the same arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


CONTRACT = "QR_M5_FAILED_BREAK_EVIDENCE_V1"
TIMEFRAME = "M5"
PRIOR_LOOKBACK = 20
REQUIRED_CANDLES = PRIOR_LOOKBACK + 1
MAX_EVIDENCE_BYTES = 6144
MAX_PRICE_ABS = 1_000_000_000.0
MAX_REASON_CHARS = 96
_TOP_LEVEL_FIELDS = {
    "contract",
    "timeframe",
    "status",
    "reason",
    "prior_lookback",
    "candles",
    "prior_high",
    "prior_low",
    "prior_width",
    "inside_buffer_ratio",
    "direction",
    "acceptance_zone",
    "evidence_sha256",
}
_CANDLE_FIELDS = {"t", "o", "h", "l", "c", "complete"}
_DIRECTIONS = {"LONG", "SHORT", "NONE"}


def build_m5_failed_break_evidence(
    pair_chart: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build evidence from the exact last 21 canonical complete M5 candles."""

    chart = pair_chart if isinstance(pair_chart, Mapping) else {}
    raw_candles: object = None
    views = chart.get("views")
    for raw_view in views if isinstance(views, list) else []:
        if not isinstance(raw_view, Mapping):
            continue
        if str(raw_view.get("granularity") or "").strip().upper() == TIMEFRAME:
            raw_candles = raw_view.get("recent_candles")
            break
    return build_m5_failed_break_evidence_from_candles(raw_candles)


def build_m5_failed_break_evidence_from_candles(
    raw_candles: object,
) -> dict[str, Any]:
    """Build the bounded proof from a raw candle sequence.

    Only explicitly complete candles are eligible.  A malformed candle in the
    selected final 21, a gap, or insufficient history returns a content-
    addressed ``UNAVAILABLE`` artifact and cannot select a method.
    """

    if not isinstance(raw_candles, list):
        return _unavailable("M5_CANDLES_MISSING")
    complete = [raw for raw in raw_candles if isinstance(raw, Mapping) and raw.get("complete") is True]
    if len(complete) < REQUIRED_CANDLES:
        return _unavailable("M5_COMPLETE_CANDLES_INSUFFICIENT")
    selected = complete[-REQUIRED_CANDLES:]
    candles: list[dict[str, Any]] = []
    for raw in selected:
        candle = _canonical_candle(raw)
        if candle is None:
            return _unavailable("M5_CANDLE_INVALID")
        candles.append(candle)
    if not _strict_m5_sequence(candles):
        return _unavailable("M5_CANDLE_SEQUENCE_INVALID")
    material = _derived_material(candles)
    evidence = {
        "contract": CONTRACT,
        "timeframe": TIMEFRAME,
        "status": "VALID",
        "reason": material["reason"],
        "prior_lookback": PRIOR_LOOKBACK,
        "candles": candles,
        "prior_high": material["prior_high"],
        "prior_low": material["prior_low"],
        "prior_width": material["prior_width"],
        "inside_buffer_ratio": 0.05,
        "direction": material["direction"],
        "acceptance_zone": material["acceptance_zone"],
    }
    evidence["evidence_sha256"] = _sha256(evidence)
    return evidence


def verify_m5_failed_break_evidence(value: object) -> tuple[bool, str | None]:
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_FIELDS:
        return False, "M5_FAILED_BREAK_EVIDENCE_SCHEMA_INVALID"
    if (
        value.get("contract") != CONTRACT
        or value.get("timeframe") != TIMEFRAME
        or value.get("prior_lookback") != PRIOR_LOOKBACK
        or value.get("inside_buffer_ratio") != 0.05
    ):
        return False, "M5_FAILED_BREAK_EVIDENCE_SCHEMA_INVALID"
    try:
        encoded = _canonical_json_bytes(value)
        expected_sha = _sha256(_without_sha(value))
    except (TypeError, ValueError, OverflowError):
        return False, "M5_FAILED_BREAK_EVIDENCE_SHA_MISMATCH"
    if len(encoded) > MAX_EVIDENCE_BYTES:
        return False, "M5_FAILED_BREAK_EVIDENCE_TOO_LARGE"
    stored_sha = value.get("evidence_sha256")
    if not isinstance(stored_sha, str) or len(stored_sha) != 64 or stored_sha != expected_sha:
        return False, "M5_FAILED_BREAK_EVIDENCE_SHA_MISMATCH"

    status = value.get("status")
    reason = value.get("reason")
    candles = value.get("candles")
    if status == "UNAVAILABLE":
        if (
            not isinstance(reason, str)
            or not reason
            or reason != reason.strip().upper()
            or len(reason) > MAX_REASON_CHARS
            or candles != []
            or value.get("prior_high") is not None
            or value.get("prior_low") is not None
            or value.get("prior_width") is not None
            or value.get("direction") != "NONE"
            or value.get("acceptance_zone") is not None
        ):
            return False, "M5_FAILED_BREAK_EVIDENCE_UNAVAILABLE_INVALID"
        return True, None
    if status != "VALID" or not isinstance(candles, list) or len(candles) != REQUIRED_CANDLES:
        return False, "M5_FAILED_BREAK_EVIDENCE_STATUS_INVALID"
    direction = value.get("direction")
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        return False, "M5_FAILED_BREAK_EVIDENCE_DIRECTION_INVALID"
    canonical: list[dict[str, Any]] = []
    for raw in candles:
        if not isinstance(raw, Mapping) or set(raw) != _CANDLE_FIELDS:
            return False, "M5_FAILED_BREAK_EVIDENCE_CANDLE_INVALID"
        candle = _canonical_candle(raw)
        if candle is None or dict(raw) != candle:
            return False, "M5_FAILED_BREAK_EVIDENCE_CANDLE_INVALID"
        canonical.append(candle)
    if not _strict_m5_sequence(canonical):
        return False, "M5_FAILED_BREAK_EVIDENCE_SEQUENCE_INVALID"
    expected = _derived_material(canonical)
    for field in (
        "reason",
        "prior_high",
        "prior_low",
        "prior_width",
        "direction",
        "acceptance_zone",
    ):
        if value.get(field) != expected[field]:
            return False, "M5_FAILED_BREAK_EVIDENCE_DERIVATION_MISMATCH"
    return True, None


def failed_break_direction(value: object) -> str | None:
    valid, _error = verify_m5_failed_break_evidence(value)
    if not valid or not isinstance(value, Mapping) or value.get("status") != "VALID":
        return None
    direction = str(value.get("direction") or "")
    return direction if direction in {"LONG", "SHORT"} else None


def failed_break_for_side(
    raw_candles: object,
    *,
    side: str,
) -> tuple[bool, float | None]:
    """Compatibility helper for intent metadata using the canonical predicate.

    Current pair-chart candles always carry timestamps and therefore take the
    fully verified evidence path.  A small amount of historical/test metadata
    predates candle timestamps, so an all-timestamp-free final window is still
    evaluated with the same canonical OHLC and exactly-one-side arithmetic.
    Mixed or malformed timestamps never enter this compatibility path.
    """

    evidence = build_m5_failed_break_evidence_from_candles(raw_candles)
    if evidence.get("status") == "VALID":
        direction = failed_break_direction(evidence)
        acceptance = evidence.get("acceptance_zone")
    else:
        legacy_material = _timestamp_free_compat_material(raw_candles)
        direction = (
            str(legacy_material.get("direction"))
            if legacy_material is not None
            else None
        )
        acceptance = (
            legacy_material.get("acceptance_zone")
            if legacy_material is not None
            else None
        )
    normalized_side = str(side or "").strip().upper()
    if direction != normalized_side:
        return False, None
    return True, float(acceptance) if isinstance(acceptance, (int, float)) else None


def _derived_material(candles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prior = candles[:PRIOR_LOOKBACK]
    current = candles[-1]
    prior_high = max(float(item["h"]) for item in prior)
    prior_low = min(float(item["l"]) for item in prior)
    width = prior_high - prior_low
    buffer = width * 0.05
    current_high = float(current["h"])
    current_low = float(current["l"])
    current_close = float(current["c"])
    long_match = current_low < prior_low and current_close > prior_low + buffer
    short_match = current_high > prior_high and current_close < prior_high - buffer
    if long_match is short_match:
        direction = "NONE"
        acceptance_zone = None
        reason = "BOTH_SIDES_AMBIGUOUS" if long_match else "NO_FAILED_BREAK"
    elif long_match:
        direction = "LONG"
        acceptance_zone = prior_low
        reason = "LONG_LOW_SWEEP_REACCEPTED"
    else:
        direction = "SHORT"
        acceptance_zone = prior_high
        reason = "SHORT_HIGH_SWEEP_REACCEPTED"
    return {
        "reason": reason,
        "prior_high": _round_price(prior_high),
        "prior_low": _round_price(prior_low),
        "prior_width": _round_price(width),
        "direction": direction,
        "acceptance_zone": _round_price(acceptance_zone) if acceptance_zone is not None else None,
    }


def _canonical_candle(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if raw.get("complete") is not True:
        return None
    timestamp = _canonical_timestamp(raw.get("t"))
    values = [_price(raw.get(field)) for field in ("o", "h", "l", "c")]
    if timestamp is None or any(value is None for value in values):
        return None
    open_, high, low, close = (_round_price(float(value)) for value in values)
    if min(open_, high, low, close) <= 0.0:
        return None
    if low > min(open_, close) or high < max(open_, close) or high < low:
        return None
    return {
        "t": timestamp,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "complete": True,
    }


def _timestamp_free_compat_material(raw_candles: object) -> dict[str, Any] | None:
    """Evaluate only a wholly timestamp-free legacy final window.

    This does not produce a verifiable receipt.  It exists solely so older
    intent metadata keeps its historical boolean/acceptance-zone surface while
    sharing the receipt's exact 20+1 and ambiguity semantics.
    """

    if not isinstance(raw_candles, list):
        return None
    complete = [
        raw
        for raw in raw_candles
        if isinstance(raw, Mapping) and raw.get("complete") is True
    ]
    if len(complete) < REQUIRED_CANDLES:
        return None
    selected = complete[-REQUIRED_CANDLES:]
    if any(raw.get("t") is not None for raw in selected):
        return None
    candles: list[dict[str, Any]] = []
    for raw in selected:
        values = [_price(raw.get(field)) for field in ("o", "h", "l", "c")]
        if any(value is None for value in values):
            return None
        open_, high, low, close = (_round_price(float(value)) for value in values)
        if min(open_, high, low, close) <= 0.0:
            return None
        if low > min(open_, close) or high < max(open_, close) or high < low:
            return None
        candles.append(
            {
                "o": open_,
                "h": high,
                "l": low,
                "c": close,
                "complete": True,
            }
        )
    return _derived_material(candles)


def _strict_m5_sequence(candles: Sequence[Mapping[str, Any]]) -> bool:
    if len(candles) != REQUIRED_CANDLES:
        return False
    parsed: list[datetime] = []
    for candle in candles:
        try:
            parsed.append(datetime.fromisoformat(str(candle["t"]).replace("Z", "+00:00")))
        except (KeyError, TypeError, ValueError):
            return False
    return all((right - left).total_seconds() == 300.0 for left, right in zip(parsed, parsed[1:]))


def _canonical_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_PRICE_ABS:
        return None
    return parsed


def _round_price(value: float) -> float:
    return round(float(value), 10)


def _unavailable(reason: str) -> dict[str, Any]:
    evidence = {
        "contract": CONTRACT,
        "timeframe": TIMEFRAME,
        "status": "UNAVAILABLE",
        "reason": str(reason or "M5_FAILED_BREAK_UNAVAILABLE").strip().upper()[:MAX_REASON_CHARS],
        "prior_lookback": PRIOR_LOOKBACK,
        "candles": [],
        "prior_high": None,
        "prior_low": None,
        "prior_width": None,
        "inside_buffer_ratio": 0.05,
        "direction": "NONE",
        "acceptance_zone": None,
    }
    evidence["evidence_sha256"] = _sha256(evidence)
    return evidence


def _without_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "evidence_sha256"}


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
