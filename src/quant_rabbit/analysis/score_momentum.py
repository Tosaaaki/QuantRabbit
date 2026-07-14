"""Score momentum helpers for pair chart snapshots.

Pair chart long/short scores are useful as a current state, but current state
alone is late around turns. This module compares the latest pair chart snapshot
with the prior snapshot and publishes the slope of the score gap so the
forecaster can see whether SHORT dominance is decaying or LONG dominance is
accelerating.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


MAX_SCORE_MOMENTUM_WINDOW_MIN = 360.0
MIN_SCORE_MOMENTUM_WINDOW_MIN = 1.0
# Consolidated refresh/sidecar passes can reacquire the same 28-pair packet
# several times within one live cycle.  Those reanchors must retain the prior
# cycle's score baseline instead of turning a few minutes of fetch jitter into
# the advertised momentum window.  The scheduled trader itself is hourly, so
# an immediately preceding packet inside this bound is a reanchor candidate;
# the next scheduled cycle rolls to the latest completed packet.
SAME_CYCLE_REANCHOR_MAX_MIN = 30.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed
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


def _embedded_score_baseline(
    previous: dict[str, Any],
    *,
    previous_packet_ts: datetime,
    current_ts: datetime,
) -> tuple[datetime, float, float, float] | None:
    """Recover the prior-cycle baseline carried by a reanchored snapshot."""

    confluence = previous.get("confluence") if isinstance(previous.get("confluence"), dict) else {}
    momentum = confluence.get("score_momentum") if isinstance(confluence.get("score_momentum"), dict) else None
    if momentum is None:
        return None
    if momentum.get("baseline_lineage") not in {"PREVIOUS_PACKET", "PRIOR_CYCLE_CARRIED"}:
        return None
    baseline_ts = _parse_time(momentum.get("previous_generated_at_utc"))
    embedded_current_ts = _parse_time(momentum.get("current_generated_at_utc"))
    previous_long = _to_float(previous.get("long_score"))
    previous_short = _to_float(previous.get("short_score"))
    previous_gap = _score_gap(previous)
    long_delta = _to_float(momentum.get("long_score_delta"))
    short_delta = _to_float(momentum.get("short_score_delta"))
    baseline_gap = _to_float(momentum.get("previous_score_gap"))
    embedded_current_gap = _to_float(momentum.get("current_score_gap"))
    embedded_gap_delta = _to_float(momentum.get("score_gap_delta"))
    embedded_elapsed_min = _to_float(momentum.get("elapsed_min"))
    if (
        baseline_ts is None
        or embedded_current_ts is None
        or previous_long is None
        or previous_short is None
        or previous_gap is None
        or long_delta is None
        or short_delta is None
        or baseline_gap is None
        or embedded_current_gap is None
        or embedded_gap_delta is None
        or embedded_elapsed_min is None
    ):
        return None
    if embedded_current_ts != previous_packet_ts or baseline_ts > previous_packet_ts:
        return None
    carried_elapsed_min = (previous_packet_ts - baseline_ts).total_seconds() / 60.0
    if abs(carried_elapsed_min - embedded_elapsed_min) > 0.01:
        return None
    if abs(previous_gap - embedded_current_gap) > 0.02:
        return None
    if abs((embedded_current_gap - baseline_gap) - embedded_gap_delta) > 0.02:
        return None
    elapsed_min = (current_ts - baseline_ts).total_seconds() / 60.0
    if elapsed_min < MIN_SCORE_MOMENTUM_WINDOW_MIN or elapsed_min > MAX_SCORE_MOMENTUM_WINDOW_MIN:
        return None
    baseline_long = previous_long - long_delta
    baseline_short = previous_short - short_delta
    if abs((baseline_long - baseline_short) - baseline_gap) > 0.02:
        return None
    return baseline_ts, baseline_long, baseline_short, baseline_gap


def attach_score_momentum(
    chart_payloads: list[dict[str, Any]],
    previous_payload: dict[str, Any] | None,
    generated_at_utc: str,
    *,
    cycle_id: str | None = None,
    cycle_lineage_status: str | None = None,
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

    packet_elapsed_min = (current_ts - previous_ts).total_seconds() / 60.0
    if packet_elapsed_min < 0 or packet_elapsed_min > MAX_SCORE_MOMENTUM_WINDOW_MIN:
        return
    current_cycle_id = str(cycle_id or "").strip()
    previous_cycle_id = str(previous_payload.get("cycle_id") or "").strip()
    current_lineage_status = str(cycle_lineage_status or "").strip().upper()
    if current_lineage_status == "UNBOUND_WRAPPER":
        # The wrapper could not prove that its input packet belongs to the
        # preceding refresh. A new fallback id is useful for keeping all of
        # this wrapper's writers together, but it is not evidence that the
        # prior packet is an independent market sample.
        return
    if current_cycle_id and previous_cycle_id:
        # Consolidated paths publish a stable cycle identity. It is the exact
        # re-anchor boundary: duration is irrelevant, and two separate short
        # cycles must never be merged merely because they are close in time.
        same_cycle_reanchor = current_cycle_id == previous_cycle_id
    elif current_cycle_id or previous_cycle_id:
        # A partially migrated lineage is ambiguous. Treating it as a new
        # sample would turn deployment timing (or a missing writer contract)
        # into apparent score momentum, while treating it as a re-anchor has
        # no verifiable prior-cycle baseline. Unknown is the only honest state.
        return
    else:
        # Legacy/standalone packets lack a cycle id. Keep the bounded temporal
        # fallback until every historical writer has migrated.
        same_cycle_reanchor = packet_elapsed_min <= SAME_CYCLE_REANCHOR_MAX_MIN

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

        baseline_ts = previous_ts
        baseline_long = previous_long
        baseline_short = previous_short
        baseline_gap = previous_gap
        if same_cycle_reanchor:
            embedded = _embedded_score_baseline(
                previous,
                previous_packet_ts=previous_ts,
                current_ts=current_ts,
            )
            if embedded is None:
                # A same-cycle packet without a verified prior-cycle lineage
                # must not manufacture short-window momentum from fetch jitter.
                continue
            baseline_ts, baseline_long, baseline_short, baseline_gap = embedded
        elapsed_min = (current_ts - baseline_ts).total_seconds() / 60.0
        if elapsed_min < MIN_SCORE_MOMENTUM_WINDOW_MIN or elapsed_min > MAX_SCORE_MOMENTUM_WINDOW_MIN:
            continue

        long_delta = long_score - baseline_long
        short_delta = short_score - baseline_short
        gap_delta = current_gap - baseline_gap
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
            "previous_score_gap": round(baseline_gap, 4),
            "current_score_gap": round(current_gap, 4),
            "previous_generated_at_utc": baseline_ts.isoformat(),
            "current_generated_at_utc": current_ts.isoformat(),
            "baseline_lineage": "PRIOR_CYCLE_CARRIED" if same_cycle_reanchor else "PREVIOUS_PACKET",
        }
