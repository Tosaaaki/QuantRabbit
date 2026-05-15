"""Score momentum helpers for pair chart snapshots.

Pair chart long/short scores are useful as a current state, but current state
alone is late around turns. This module compares the latest pair chart snapshot
with the prior snapshot and publishes the slope of the score gap so the
forecaster can see whether SHORT dominance is decaying or LONG dominance is
accelerating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


MAX_SCORE_MOMENTUM_WINDOW_MIN = 360.0
MIN_SCORE_MOMENTUM_WINDOW_MIN = 1.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _score_gap(chart: dict[str, Any]) -> float | None:
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {}
    from_confluence = _to_float((confluence or {}).get("score_gap"))
    if from_confluence is not None:
        return from_confluence
    long_score = _to_float(chart.get("long_score"))
    short_score = _to_float(chart.get("short_score"))
    if long_score is None or short_score is None:
        return None
    return long_score - short_score


def attach_score_momentum(
    chart_payloads: list[dict[str, Any]],
    previous_payload: dict[str, Any] | None,
    generated_at_utc: str,
) -> None:
    """Attach per-pair score slope into ``chart["confluence"]`` in place.

    Missing or stale previous snapshots are ignored. We intentionally do not
    invent a zero slope because "unknown" and "flat" mean different things for
    entries.
    """

    if not isinstance(previous_payload, dict):
        return

    current_ts = _parse_time(generated_at_utc)
    previous_ts = _parse_time(previous_payload.get("generated_at_utc"))
    if current_ts is None or previous_ts is None:
        return

    elapsed_min = (current_ts - previous_ts).total_seconds() / 60.0
    if elapsed_min < MIN_SCORE_MOMENTUM_WINDOW_MIN:
        return
    if elapsed_min > MAX_SCORE_MOMENTUM_WINDOW_MIN:
        return

    previous_by_pair: dict[str, dict[str, Any]] = {}
    for chart in previous_payload.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        pair = str(chart.get("pair") or "").upper()
        if pair:
            previous_by_pair[pair] = chart

    for chart in chart_payloads:
        if not isinstance(chart, dict):
            continue
        pair = str(chart.get("pair") or "").upper()
        previous = previous_by_pair.get(pair)
        if previous is None:
            continue

        long_score = _to_float(chart.get("long_score"))
        short_score = _to_float(chart.get("short_score"))
        previous_long = _to_float(previous.get("long_score"))
        previous_short = _to_float(previous.get("short_score"))
        current_gap = _score_gap(chart)
        previous_gap = _score_gap(previous)
        if (
            long_score is None
            or short_score is None
            or previous_long is None
            or previous_short is None
            or current_gap is None
            or previous_gap is None
        ):
            continue

        long_delta = long_score - previous_long
        short_delta = short_score - previous_short
        gap_delta = current_gap - previous_gap
        if gap_delta > 0:
            direction = "UP"
        elif gap_delta < 0:
            direction = "DOWN"
        else:
            direction = "FLAT"

        slope_per_hour = gap_delta / elapsed_min * 60.0
        confluence = chart.setdefault("confluence", {})
        if not isinstance(confluence, dict):
            confluence = {}
            chart["confluence"] = confluence
        confluence["score_momentum"] = {
            "direction": direction,
            "elapsed_min": round(elapsed_min, 3),
            "long_score_delta": round(long_delta, 4),
            "short_score_delta": round(short_delta, 4),
            "score_gap_delta": round(gap_delta, 4),
            "score_gap_slope_per_hour": round(slope_per_hour, 4),
            "previous_score_gap": round(previous_gap, 4),
            "current_score_gap": round(current_gap, 4),
            "previous_generated_at_utc": previous_ts.isoformat(),
            "current_generated_at_utc": current_ts.isoformat(),
        }
