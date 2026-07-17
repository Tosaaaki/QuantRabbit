"""Sealed all-pair observation surfaces for the deterministic fast bot.

The position guardian keeps its bounded active packet for position management.
This module publishes a separate exact-28 current M1/M5 packet and incrementally
retains the last structurally valid slower views seen by the existing rotation.
Neither artifact grants order authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.strategy.directional_forecaster import (
    validate_mba_integrity_receipt,
)


CURRENT_M1_CONTRACT = "QR_GUARDIAN_ALL_PAIR_FAST_V2"
SLOW_RETENTION_CONTRACT = "QR_GUARDIAN_SLOW_CHART_RETENTION_V1"
CURRENT_TIMEFRAMES = ("M1", "M5")
SLOW_TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "D")
CURRENT_PACKET_MAX_AGE_SECONDS = 180
QUOTE_MAX_AGE_SECONDS = 45
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D": 24 * 60 * 60,
}


class ObservationContractError(ValueError):
    """Raised when an observation source cannot be sealed safely."""


def observation_refresh_due(
    path: Path,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Return true unless a valid CURRENT packet covers the current cadence."""

    value = _read_object(path)
    cadence_contract_valid = bool(
        _sealed_contract_valid(value, CURRENT_M1_CONTRACT)
        and value.get("schema_version") == 1
        and value.get("status") in {"CURRENT", "BLOCKED"}
        and value.get("configured_pairs") == list(DEFAULT_TRADER_PAIRS)
        and value.get("required_timeframes") == list(CURRENT_TIMEFRAMES)
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
    )
    if not cadence_contract_valid:
        return True
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    next_refresh = _parse_utc(value.get("next_refresh_after_utc"))
    return next_refresh is None or now >= next_refresh


def publish_current_m1(
    *,
    source_path: Path,
    snapshot_path: Path,
    output_path: Path,
    active_pair_count: int,
    active_chart_wall_time_seconds: float,
    all_pair_m1_wall_time_seconds: float,
    post_chart_snapshot_wall_time_seconds: float,
    candle_close_grace_seconds: int = 5,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish an exact-28 current M1/M5 packet."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    source = _read_required_object(source_path, label="all-pair M1 source")
    snapshot = _read_required_object(snapshot_path, label="post-chart broker snapshot")
    generated = _parse_utc(source.get("generated_at_utc"))
    if generated is None or not _timestamp_current(
        now,
        generated,
        max_age_seconds=CURRENT_PACKET_MAX_AGE_SECONDS,
    ):
        raise ObservationContractError("all-pair M1/M5 source generated_at_utc is stale, future, or invalid")
    charts = _validated_source_charts(
        source,
        expected_pairs=DEFAULT_TRADER_PAIRS,
        required_timeframes=CURRENT_TIMEFRAMES,
        chart_generated_at=generated,
        now_utc=now,
        require_current_close=True,
    )
    quote_metrics = _validate_snapshot(snapshot, now_utc=now)
    request_metrics = _request_metrics(
        active_pair_count=active_pair_count,
        active_chart_wall_time_seconds=active_chart_wall_time_seconds,
        all_pair_m1_wall_time_seconds=all_pair_m1_wall_time_seconds,
        post_chart_snapshot_wall_time_seconds=post_chart_snapshot_wall_time_seconds,
        quote_metrics=quote_metrics,
    )
    body: dict[str, Any] = {
        "contract": CURRENT_M1_CONTRACT,
        "schema_version": 1,
        "status": "CURRENT",
        "generated_at_utc": generated.isoformat(),
        "sealed_at_utc": now.isoformat(),
        "next_refresh_after_utc": _next_m1_refresh(
            now,
            grace_seconds=candle_close_grace_seconds,
        ).isoformat(),
        "configured_pairs": list(DEFAULT_TRADER_PAIRS),
        "required_timeframes": list(CURRENT_TIMEFRAMES),
        "timeframes": list(CURRENT_TIMEFRAMES),
        "coverage_complete": True,
        "pairs_requested": len(DEFAULT_TRADER_PAIRS),
        "pairs_succeeded": len(charts),
        "pairs_failed": 0,
        "partial": False,
        "failures": [],
        "candidate_projection": "ALL_PAIR_ROWS_PRESERVED_NO_TOP1_SELECTION",
        "shadow_observation_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "source_pair_charts_sha256": _file_sha256(source_path),
        "post_chart_snapshot_sha256": _canonical_sha(snapshot),
        "request_metrics": request_metrics,
        "charts": charts,
    }
    sealed = _seal(body)
    _write_json_atomic(output_path, sealed)
    return sealed


def publish_blocked_current_m1(
    *,
    output_path: Path,
    reason: str,
    active_pair_count: int,
    active_chart_wall_time_seconds: float,
    all_pair_m1_wall_time_seconds: float,
    post_chart_snapshot_wall_time_seconds: float,
    candle_close_grace_seconds: int = 5,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Replace any previous current packet with an explicit empty block."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    body = {
        "contract": CURRENT_M1_CONTRACT,
        "schema_version": 1,
        "status": "BLOCKED",
        "generated_at_utc": now.isoformat(),
        "sealed_at_utc": now.isoformat(),
        "next_refresh_after_utc": _next_m1_refresh(
            now,
            grace_seconds=candle_close_grace_seconds,
        ).isoformat(),
        "configured_pairs": list(DEFAULT_TRADER_PAIRS),
        "required_timeframes": list(CURRENT_TIMEFRAMES),
        "timeframes": list(CURRENT_TIMEFRAMES),
        "coverage_complete": False,
        "pairs_requested": len(DEFAULT_TRADER_PAIRS),
        "pairs_succeeded": 0,
        "pairs_failed": len(DEFAULT_TRADER_PAIRS),
        "partial": True,
        "failures": [str(reason or "ALL_PAIR_M1_REFRESH_FAILED")],
        "candidate_projection": "ALL_PAIR_ROWS_PRESERVED_NO_TOP1_SELECTION",
        "shadow_observation_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "blockers": [str(reason or "ALL_PAIR_M1_REFRESH_FAILED")],
        "request_metrics": _request_metrics(
            active_pair_count=active_pair_count,
            active_chart_wall_time_seconds=active_chart_wall_time_seconds,
            all_pair_m1_wall_time_seconds=all_pair_m1_wall_time_seconds,
            post_chart_snapshot_wall_time_seconds=post_chart_snapshot_wall_time_seconds,
            quote_metrics=None,
        ),
        "charts": [],
    }
    sealed = _seal(body)
    _write_json_atomic(output_path, sealed)
    return sealed


def update_slow_retention(
    *,
    source_path: Path,
    output_path: Path,
    source_pairs: Sequence[str],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Atomically merge one valid rotation packet into last-good slow views."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    normalized_pairs = tuple(_normalize_pair(item) for item in source_pairs)
    if not normalized_pairs or any(not item for item in normalized_pairs):
        raise ObservationContractError("slow retention source_pairs must not be empty")
    if len(set(normalized_pairs)) != len(normalized_pairs):
        raise ObservationContractError("slow retention source_pairs contains duplicates")
    if not set(normalized_pairs).issubset(DEFAULT_TRADER_PAIRS):
        raise ObservationContractError("slow retention source_pairs exceeds configured universe")
    source = _read_required_object(source_path, label="active guardian chart source")
    generated = _parse_utc(source.get("generated_at_utc"))
    if generated is None:
        raise ObservationContractError("active guardian chart generated_at_utc is invalid")
    source_charts = _validated_source_charts(
        source,
        expected_pairs=normalized_pairs,
        required_timeframes=SLOW_TIMEFRAMES,
        chart_generated_at=generated,
        now_utc=None,
        require_current_close=False,
        allow_additional_timeframes=True,
    )
    prior = _read_object(output_path)
    retained = _validated_prior_retention(prior)
    source_sha = _file_sha256(source_path)
    for chart in source_charts:
        pair = str(chart["pair"])
        pair_views = retained[pair]
        for view in chart["views"]:
            timeframe = str(view["granularity"])
            pair_views[timeframe] = {
                "view": view,
                "source_generated_at_utc": generated.isoformat(),
                "source_pair_charts_sha256": source_sha,
                "source_view_sha256": _canonical_sha(view),
                "retained_at_utc": now.isoformat(),
            }
    rows: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        pair_views = retained[pair]
        views = [pair_views[tf]["view"] for tf in SLOW_TIMEFRAMES if tf in pair_views]
        absent = [tf for tf in SLOW_TIMEFRAMES if tf not in pair_views]
        if absent:
            missing[pair] = absent
        rows.append(
            {
                "pair": pair,
                "views": views,
                "retained_view_sources": {
                    tf: {key: value for key, value in pair_views[tf].items() if key != "view"}
                    for tf in SLOW_TIMEFRAMES
                    if tf in pair_views
                },
                "coverage_complete": not absent,
                "missing_timeframes": absent,
            }
        )
    body = {
        "contract": SLOW_RETENTION_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "updated_at_utc": now.isoformat(),
        "configured_pairs": list(DEFAULT_TRADER_PAIRS),
        "required_timeframes": list(SLOW_TIMEFRAMES),
        "coverage_complete": not missing,
        "pairs_complete": len(DEFAULT_TRADER_PAIRS) - len(missing),
        "pairs_total": len(DEFAULT_TRADER_PAIRS),
        "missing_by_pair": missing,
        "last_source_pairs": list(normalized_pairs),
        "last_source_generated_at_utc": generated.isoformat(),
        "last_source_pair_charts_sha256": source_sha,
        "candidate_projection": "ALL_PAIR_ROWS_PRESERVED_NO_TOP1_SELECTION",
        "shadow_observation_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "charts": rows,
    }
    sealed = _seal(body)
    _write_json_atomic(output_path, sealed)
    return sealed


def validate_current_m1_contract(
    value: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Validate the consumer-visible shape and seal of a current packet."""

    if not _sealed_contract_valid(value, CURRENT_M1_CONTRACT):
        return False
    if value.get("schema_version") != 1 or value.get("status") != "CURRENT":
        return False
    if value.get("configured_pairs") != list(DEFAULT_TRADER_PAIRS):
        return False
    if value.get("required_timeframes") != list(CURRENT_TIMEFRAMES):
        return False
    if value.get("coverage_complete") is not True:
        return False
    if (
        value.get("pairs_requested") != len(DEFAULT_TRADER_PAIRS)
        or value.get("pairs_succeeded") != len(DEFAULT_TRADER_PAIRS)
        or value.get("pairs_failed") != 0
        or value.get("partial") is not False
        or value.get("failures") != []
        or value.get("candidate_projection") != "ALL_PAIR_ROWS_PRESERVED_NO_TOP1_SELECTION"
        or value.get("shadow_observation_only") is not True
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or not _sha256_text(value.get("source_pair_charts_sha256"))
        or not _sha256_text(value.get("post_chart_snapshot_sha256"))
        or not isinstance(value.get("request_metrics"), Mapping)
    ):
        return False
    generated = _parse_utc(value.get("generated_at_utc"))
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    if generated is None or not _timestamp_current(
        now,
        generated,
        max_age_seconds=CURRENT_PACKET_MAX_AGE_SECONDS,
    ):
        return False
    try:
        _validated_source_charts(
            value,
            expected_pairs=DEFAULT_TRADER_PAIRS,
            required_timeframes=CURRENT_TIMEFRAMES,
            chart_generated_at=generated,
            now_utc=now,
            require_current_close=True,
        )
    except ObservationContractError:
        return False
    return True


def validate_slow_retention_contract(value: Mapping[str, Any]) -> bool:
    """Validate the exact-28 retained slow-view container and all view seals."""

    if not _sealed_contract_valid(value, SLOW_RETENTION_CONTRACT):
        return False
    if value.get("schema_version") != 1:
        return False
    if value.get("configured_pairs") != list(DEFAULT_TRADER_PAIRS):
        return False
    if value.get("required_timeframes") != list(SLOW_TIMEFRAMES):
        return False
    if value.get("live_permission") is not False or value.get("broker_mutation_allowed") is not False:
        return False
    charts = value.get("charts")
    if not isinstance(charts, list) or len(charts) != len(DEFAULT_TRADER_PAIRS):
        return False
    if [item.get("pair") for item in charts if isinstance(item, Mapping)] != list(DEFAULT_TRADER_PAIRS):
        return False
    missing_by_pair: dict[str, list[str]] = {}
    pairs_complete = 0
    for chart in charts:
        if not isinstance(chart, Mapping):
            return False
        pair = str(chart.get("pair") or "")
        views = chart.get("views")
        sources = chart.get("retained_view_sources")
        if not isinstance(views, list) or not isinstance(sources, Mapping):
            return False
        observed: set[str] = set()
        for view in views:
            if not isinstance(view, Mapping):
                return False
            timeframe = str(view.get("granularity") or "").upper()
            if timeframe not in SLOW_TIMEFRAMES or timeframe in observed:
                return False
            source = sources.get(timeframe)
            if not isinstance(source, Mapping):
                return False
            generated = _parse_utc(source.get("source_generated_at_utc"))
            integrity = view.get("candle_integrity")
            if (
                generated is None
                or not isinstance(integrity, dict)
                or integrity.get("pair") != pair
                or integrity.get("granularity") != timeframe
                or source.get("source_view_sha256") != _canonical_sha(view)
                or not _sha256_text(source.get("source_pair_charts_sha256"))
                or _parse_utc(source.get("retained_at_utc")) is None
                or not validate_mba_integrity_receipt(
                    integrity,
                    chart_generated_at=generated,
                    view=dict(view),
                    now_utc=None,
                )
            ):
                return False
            observed.add(timeframe)
        absent = [tf for tf in SLOW_TIMEFRAMES if tf not in observed]
        if chart.get("missing_timeframes") != absent:
            return False
        if chart.get("coverage_complete") is not (not absent):
            return False
        if absent:
            missing_by_pair[pair] = absent
        else:
            pairs_complete += 1
    return bool(
        value.get("coverage_complete") is (not missing_by_pair)
        and value.get("pairs_complete") == pairs_complete
        and value.get("pairs_total") == len(DEFAULT_TRADER_PAIRS)
        and value.get("missing_by_pair") == missing_by_pair
        and value.get("candidate_projection") == "ALL_PAIR_ROWS_PRESERVED_NO_TOP1_SELECTION"
        and value.get("shadow_observation_only") is True
    )


def _validated_prior_retention(
    prior: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    retained: dict[str, dict[str, dict[str, Any]]] = {
        pair: {} for pair in DEFAULT_TRADER_PAIRS
    }
    if not prior:
        return retained
    if not validate_slow_retention_contract(prior):
        raise ObservationContractError("prior slow retention contract is invalid")
    for chart in prior.get("charts", []):
        pair = str(chart["pair"])
        sources = chart["retained_view_sources"]
        for view in chart["views"]:
            timeframe = str(view["granularity"])
            retained[pair][timeframe] = {"view": dict(view), **dict(sources[timeframe])}
    return retained


def _validated_source_charts(
    source: Mapping[str, Any],
    *,
    expected_pairs: Sequence[str],
    required_timeframes: Sequence[str],
    chart_generated_at: datetime,
    now_utc: datetime | None,
    require_current_close: bool,
    allow_additional_timeframes: bool = False,
) -> list[dict[str, Any]]:
    charts = source.get("charts")
    if not isinstance(charts, list):
        raise ObservationContractError("chart source charts must be a list")
    by_pair: dict[str, Mapping[str, Any]] = {}
    for chart in charts:
        if not isinstance(chart, Mapping):
            raise ObservationContractError("chart source contains a non-object row")
        pair = _normalize_pair(chart.get("pair"))
        if not pair or pair in by_pair:
            raise ObservationContractError("chart source contains an invalid or duplicate pair")
        by_pair[pair] = chart
    if set(by_pair) != set(expected_pairs) or len(by_pair) != len(expected_pairs):
        raise ObservationContractError("chart source does not have exact expected pair coverage")
    source_timeframes = source.get("timeframes")
    if not isinstance(source_timeframes, list) or len(set(source_timeframes)) != len(source_timeframes):
        raise ObservationContractError("chart source timeframes is invalid")
    if not set(required_timeframes).issubset(source_timeframes) or (
        not allow_additional_timeframes and source_timeframes != list(required_timeframes)
    ):
        raise ObservationContractError("chart source timeframes does not match required coverage")
    if source.get("pairs_requested") != len(expected_pairs):
        raise ObservationContractError("chart source pairs_requested does not match coverage")
    if source.get("pairs_succeeded") != len(expected_pairs):
        raise ObservationContractError("chart source pairs_succeeded does not match coverage")
    if (
        source.get("pairs_failed") != 0
        or source.get("partial") is not False
        or source.get("failures") != []
    ):
        raise ObservationContractError("chart source reports partial or failed coverage")
    copied: list[dict[str, Any]] = []
    required = set(required_timeframes)
    for pair in expected_pairs:
        chart = by_pair[pair]
        views = chart.get("views")
        if not isinstance(views, list):
            raise ObservationContractError(f"{pair} views must be a list")
        by_timeframe: dict[str, Mapping[str, Any]] = {}
        for view in views:
            if not isinstance(view, Mapping):
                raise ObservationContractError(f"{pair} contains a non-object view")
            timeframe = str(view.get("granularity") or "").upper()
            if not timeframe or timeframe in by_timeframe:
                raise ObservationContractError(f"{pair} contains an invalid or duplicate timeframe")
            by_timeframe[timeframe] = view
        if not required.issubset(by_timeframe) or (
            not allow_additional_timeframes and set(by_timeframe) != required
        ):
            raise ObservationContractError(f"{pair} does not contain exact required timeframe coverage")
        selected: list[dict[str, Any]] = []
        for timeframe in required_timeframes:
            view = by_timeframe[timeframe]
            integrity = view.get("candle_integrity")
            if (
                not isinstance(integrity, dict)
                or integrity.get("pair") != pair
                or integrity.get("granularity") != timeframe
                or not validate_mba_integrity_receipt(
                    integrity,
                    chart_generated_at=chart_generated_at,
                    view=dict(view),
                    now_utc=now_utc,
                )
            ):
                raise ObservationContractError(f"{pair} {timeframe} receipt is invalid")
            if require_current_close and not _view_close_current(
                view,
                timeframe=timeframe,
                now_utc=now_utc,
            ):
                raise ObservationContractError(f"{pair} {timeframe} latest complete close is stale or future")
            selected.append(dict(view))
        copied.append({"pair": pair, "views": selected})
    return copied


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    fetched = _parse_utc(snapshot.get("fetched_at_utc"))
    if fetched is None or not _timestamp_current(
        now_utc,
        fetched,
        max_age_seconds=QUOTE_MAX_AGE_SECONDS,
    ):
        raise ObservationContractError("post-chart broker snapshot is stale, future, or invalid")
    quotes = snapshot.get("quotes")
    if not isinstance(quotes, Mapping) or set(quotes) != set(DEFAULT_TRADER_PAIRS):
        raise ObservationContractError("post-chart broker snapshot does not have exact-28 quote coverage")
    ages: list[float] = []
    for pair in DEFAULT_TRADER_PAIRS:
        quote = quotes.get(pair)
        if not isinstance(quote, Mapping):
            raise ObservationContractError(f"post-chart quote missing for {pair}")
        bid = _finite_number(quote.get("bid"))
        ask = _finite_number(quote.get("ask"))
        quote_at = _parse_utc(quote.get("timestamp_utc"))
        if bid is None or ask is None or bid <= 0.0 or ask <= bid or quote_at is None:
            raise ObservationContractError(f"post-chart quote invalid for {pair}")
        age = (now_utc - quote_at).total_seconds()
        if age < 0.0 or age > QUOTE_MAX_AGE_SECONDS:
            raise ObservationContractError(f"post-chart quote stale or future for {pair}")
        ages.append(age)
    return {
        "post_chart_snapshot_fetched_at_utc": fetched.isoformat(),
        "quote_pairs_requested": len(DEFAULT_TRADER_PAIRS),
        "quote_pairs_succeeded": len(ages),
        "oldest_quote_age_seconds": round(max(ages), 6),
        "newest_quote_age_seconds": round(min(ages), 6),
        "quote_max_age_seconds": QUOTE_MAX_AGE_SECONDS,
    }


def _request_metrics(
    *,
    active_pair_count: int,
    active_chart_wall_time_seconds: float,
    all_pair_m1_wall_time_seconds: float,
    post_chart_snapshot_wall_time_seconds: float,
    quote_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(active_pair_count, bool) or not 0 <= int(active_pair_count) <= len(DEFAULT_TRADER_PAIRS):
        raise ObservationContractError("active_pair_count must be in 0..28")
    timings = {
        "active_chart_wall_time_seconds": active_chart_wall_time_seconds,
        "all_pair_m1_wall_time_seconds": all_pair_m1_wall_time_seconds,
        "post_chart_snapshot_wall_time_seconds": post_chart_snapshot_wall_time_seconds,
    }
    for name, value in timings.items():
        if _finite_number(value) is None or float(value) < 0.0:
            raise ObservationContractError(f"{name} must be finite and non-negative")
    active_requests = int(active_pair_count) * 7
    metrics: dict[str, Any] = {
        "request_plan": "EXACT_28_M1_M5_PLUS_BOUNDED_ACTIVE_ROTATION",
        "active_rotation_pairs": int(active_pair_count),
        "active_rotation_timeframes": 7,
        "active_rotation_candle_requests": active_requests,
        "all_pair_m1_pairs": len(DEFAULT_TRADER_PAIRS),
        "all_pair_m1_timeframes": 1,
        "all_pair_m1_candle_requests": len(DEFAULT_TRADER_PAIRS),
        "all_pair_fast_pairs": len(DEFAULT_TRADER_PAIRS),
        "all_pair_fast_timeframes": len(CURRENT_TIMEFRAMES),
        "all_pair_fast_candle_requests": (
            len(DEFAULT_TRADER_PAIRS) * len(CURRENT_TIMEFRAMES)
        ),
        "total_candle_requests": (
            active_requests + len(DEFAULT_TRADER_PAIRS) * len(CURRENT_TIMEFRAMES)
        ),
        **{name: round(float(value), 6) for name, value in timings.items()},
    }
    if quote_metrics:
        metrics.update(dict(quote_metrics))
    return metrics


def _view_close_current(
    view: Mapping[str, Any],
    *,
    timeframe: str,
    now_utc: datetime | None,
) -> bool:
    if now_utc is None:
        return False
    integrity = view.get("candle_integrity")
    started = _parse_utc(
        integrity.get("latest_complete_timestamp_utc")
        if isinstance(integrity, Mapping)
        else None
    )
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if started is None or seconds is None:
        return False
    age = (now_utc - (started + timedelta(seconds=seconds))).total_seconds()
    return 0.0 <= age <= seconds * 2


def _next_m1_refresh(now_utc: datetime, *, grace_seconds: int) -> datetime:
    minute = now_utc.replace(second=0, microsecond=0)
    return minute + timedelta(minutes=1, seconds=max(0, int(grace_seconds)))


def _read_required_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_object(path)
    if not value:
        raise ObservationContractError(f"{label} is missing or invalid JSON")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_contract_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(stored and stored == _canonical_sha(body))


def _canonical_sha(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raw = b"INVALID"
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_pair(value: Any) -> str:
    pair = str(value or "").strip().upper()
    return pair if pair in DEFAULT_TRADER_PAIRS else ""


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp_current(
    now_utc: datetime,
    then: datetime,
    *,
    max_age_seconds: float,
) -> bool:
    age = (now_utc - then).total_seconds()
    return 0.0 <= age <= max_age_seconds


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    due = sub.add_parser("due")
    due.add_argument("--current", type=Path, required=True)
    retain = sub.add_parser("retain")
    retain.add_argument("--source", type=Path, required=True)
    retain.add_argument("--output", type=Path, required=True)
    retain.add_argument("--source-pairs", required=True)
    current = sub.add_parser("publish-current")
    current.add_argument("--source", type=Path, required=True)
    current.add_argument("--snapshot", type=Path, required=True)
    current.add_argument("--output", type=Path, required=True)
    blocked = sub.add_parser("publish-blocked")
    blocked.add_argument("--output", type=Path, required=True)
    blocked.add_argument("--reason", required=True)
    for command in (current, blocked):
        command.add_argument("--active-pair-count", type=int, required=True)
        command.add_argument("--active-chart-wall-seconds", type=float, required=True)
        command.add_argument("--all-pair-m1-wall-seconds", type=float, required=True)
        command.add_argument("--post-chart-snapshot-wall-seconds", type=float, required=True)
        command.add_argument("--candle-close-grace-seconds", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "due":
            print("1" if observation_refresh_due(args.current) else "0")
            return 0
        if args.command == "retain":
            result = update_slow_retention(
                source_path=args.source,
                output_path=args.output,
                source_pairs=[item for item in args.source_pairs.split(",") if item],
            )
        elif args.command == "publish-current":
            result = publish_current_m1(
                source_path=args.source,
                snapshot_path=args.snapshot,
                output_path=args.output,
                active_pair_count=args.active_pair_count,
                active_chart_wall_time_seconds=args.active_chart_wall_seconds,
                all_pair_m1_wall_time_seconds=args.all_pair_m1_wall_seconds,
                post_chart_snapshot_wall_time_seconds=args.post_chart_snapshot_wall_seconds,
                candle_close_grace_seconds=args.candle_close_grace_seconds,
            )
        else:
            result = publish_blocked_current_m1(
                output_path=args.output,
                reason=args.reason,
                active_pair_count=args.active_pair_count,
                active_chart_wall_time_seconds=args.active_chart_wall_seconds,
                all_pair_m1_wall_time_seconds=args.all_pair_m1_wall_seconds,
                post_chart_snapshot_wall_time_seconds=args.post_chart_snapshot_wall_seconds,
                candle_close_grace_seconds=args.candle_close_grace_seconds,
            )
    except (ObservationContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status", "UPDATED"),
                "contract": result.get("contract"),
                "contract_sha256": result.get("contract_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
