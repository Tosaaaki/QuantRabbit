"""Immutable OANDA source acquisition for the DOJO worker forward smoke.

This module is intentionally separate from :mod:`dojo_worker_forward`: the
active smoke precommit binds that lifecycle module's bytes.  The collector is
therefore additional self-attested diagnostic evidence, not a retroactive
change to the precommit and never a source of live authority.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_worker_forward import (
    build_day_seal,
    canonical_sha256,
    validate_day_seal,
    validate_precommit,
    validate_start_receipt,
)
from quant_rabbit.dojo_market_calendar import (
    OANDA_FX_HOURS_POLICY,
    OANDA_FX_HOURS_SOURCE,
    expected_oanda_fx_slots,
)


REQUEST_CONTRACT = "QR_DOJO_WORKER_DAY_SOURCE_REQUEST_V1"
CAPTURE_CONTRACT = "QR_DOJO_WORKER_DAY_SOURCE_CAPTURE_V1"
RECEIPT_CONTRACT = "QR_DOJO_WORKER_DAY_SOURCE_RECEIPT_V1"
PAIR = "USD_JPY"
GRANULARITY = "M1"
PRICE_COMPONENT = "BA"
SOURCE_ID = f"{PAIR}:{GRANULARITY}"
OFFICIAL_OANDA_BASE_URL = "https://api-fxtrade.oanda.com"
MINIMUM_MATURITY_DELAY = timedelta(minutes=2)
FULL_DAY_MINIMUM_EXPECTED_SLOTS = 1_000
FULL_DAY_MINIMUM_COVERAGE = (98, 100)
PARTIAL_DAY_MINIMUM_COVERAGE = (80, 100)
MAX_CONTIGUOUS_GAP = timedelta(minutes=15)
BOUNDARY_TOLERANCE = timedelta(minutes=15)
COVERAGE_POLICY = "OANDA_M1_TRUTHFUL_SPARSE_FIXED_COVERAGE_V1"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]+")
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(0+))?Z")
_OHLC_KEYS = frozenset({"o", "h", "l", "c"})


class DojoWorkerSourceError(ValueError):
    """The source request, response, persisted bytes, or timing is invalid."""


def collect_and_seal_day(
    run_dir: Path,
    *,
    ordinal: int,
    client: Any,
    now_utc: datetime | None = None,
    repo_root: Path,
    collector_paths: Sequence[Path] = (),
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Fetch one exact UTC day, persist immutable bytes, and seal it once.

    The caller cannot provide a payload, window, pair, price component, or
    manifest.  All of those are derived from the validated precommit and the
    read-only OANDA response.
    """

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_clock = clock or (lambda: datetime.now(timezone.utc))
    fixed_clock = now_utc is not None
    initial_now = _utc(now_utc if now_utc is not None else resolved_clock(), "now_utc")
    lock_path = run_dir / ".source-collector.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoWorkerSourceError(
                "another source collector owns the run"
            ) from exc
        return _collect_under_lock(
            run_dir,
            ordinal=ordinal,
            client=client,
            now_utc=initial_now,
            repo_root=repo_root.resolve(),
            collector_paths=collector_paths,
            clock=resolved_clock,
            fixed_clock=fixed_clock,
        )
    finally:
        os.close(descriptor)


def verify_collected_day(
    run_dir: Path,
    *,
    ordinal: int,
) -> dict[str, Any]:
    """Re-open and re-hash one persisted source bundle without network use."""

    run_dir = run_dir.resolve()
    precommit, start = _load_parents(run_dir)
    day_start, day_end = _day_window(precommit, ordinal)
    evidence_relpath = Path("source-evidence") / f"day-{ordinal:03d}"
    _safe_artifact_directory(run_dir, evidence_relpath)
    request = _strict_json(
        _safe_artifact_path(run_dir, evidence_relpath / "request.json")
    )
    receipt = _strict_json(
        _safe_artifact_path(run_dir, evidence_relpath / "acquisition-receipt.json")
    )
    manifest = _strict_json(
        _safe_artifact_path(run_dir, evidence_relpath / "source-manifest.json")
    )
    _validate_request(
        request,
        precommit=precommit,
        start=start,
        ordinal=ordinal,
        day_start=day_start,
        day_end=day_end,
    )
    normalized = _validate_receipt_and_files(
        receipt,
        manifest=manifest,
        request=request,
        run_dir=run_dir,
        precommit=precommit,
        start=start,
        ordinal=ordinal,
        day_start=day_start,
        day_end=day_end,
    )
    day_path = run_dir / "days" / f"day-{ordinal:03d}.json"
    if day_path.is_file():
        day_path = _safe_artifact_path(
            run_dir, Path("days") / f"day-{ordinal:03d}.json"
        )
        previous = None
        if ordinal > 1:
            previous = _strict_json(
                _safe_artifact_path(
                    run_dir, Path("days") / f"day-{ordinal - 1:03d}.json"
                )
            )
        seal = validate_day_seal(
            _strict_json(day_path),
            precommit,
            start,
            expected_ordinal=ordinal,
        )
        expected_previous = (
            start["start_receipt_sha256"]
            if ordinal == 1
            else validate_day_seal(
                previous,
                precommit,
                start,
                expected_ordinal=ordinal - 1,
            )["day_seal_sha256"]
        )
        if seal["previous_receipt_sha256"] != expected_previous:
            raise DojoWorkerSourceError("day seal chain parent is invalid")
        if seal["source_manifest"] != manifest:
            raise DojoWorkerSourceError(
                "day seal does not bind persisted source manifest"
            )
        if _parse_utc(seal["sealed_at_utc"], "sealed_at_utc") < _parse_utc(
            receipt["recorded_at_utc"], "recorded_at_utc"
        ):
            raise DojoWorkerSourceError("day seal predates captured source response")
        normalized["day_seal_sha256"] = seal["day_seal_sha256"]
        normalized["state"] = seal["state"]
    else:
        normalized["day_seal_sha256"] = None
        normalized["state"] = "SOURCE_PERSISTED_UNSEALED"
    return normalized


def expected_open_slots(day_start: datetime, day_end: datetime) -> list[str]:
    """Return DST-aware retail-FX M1 opens for a half-open UTC day."""

    start = _utc(day_start, "day_start")
    end = _utc(day_end, "day_end")
    if end - start != timedelta(days=1):
        raise DojoWorkerSourceError("source day must be exactly 24 hours")
    return [
        _iso(slot)
        for slot in expected_oanda_fx_slots(
            start,
            end,
            step=timedelta(minutes=1),
        )
    ]


def normalize_oanda_payload(
    payload: Mapping[str, Any],
    *,
    day_start: datetime,
    day_end: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly validate a single OANDA BA response without dropping rows."""

    source = _mapping(payload, "OANDA response")
    if set(source) != {"instrument", "granularity", "candles"}:
        raise DojoWorkerSourceError("OANDA response schema is not exact")
    if source["instrument"] != PAIR or source["granularity"] != GRANULARITY:
        raise DojoWorkerSourceError("OANDA response identity drifted")
    candles = source["candles"]
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise DojoWorkerSourceError("OANDA candles must be a sequence")
    start = _utc(day_start, "day_start")
    end = _utc(day_end, "day_end")
    allowed_slots = expected_open_slots(start, end)
    allowed = set(allowed_slots)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous: datetime | None = None
    actual_instants: list[datetime] = []
    for index, raw in enumerate(candles):
        item = _mapping(raw, f"candle[{index}]")
        if set(item) != {"complete", "volume", "time", "bid", "ask"}:
            raise DojoWorkerSourceError(f"candle[{index}] schema is not exact BA")
        if item["complete"] is not True:
            raise DojoWorkerSourceError(f"candle[{index}] is incomplete")
        if (
            isinstance(item["volume"], bool)
            or not isinstance(item["volume"], int)
            or item["volume"] < 0
        ):
            raise DojoWorkerSourceError(f"candle[{index}] volume is invalid")
        stamp, stamp_text = _parse_m1_time(item["time"], f"candle[{index}].time")
        if stamp < start or stamp >= end or stamp_text not in allowed:
            raise DojoWorkerSourceError(f"candle[{index}] is outside an open M1 slot")
        if stamp_text in seen:
            raise DojoWorkerSourceError("duplicate candle timestamp")
        if previous is not None and stamp <= previous:
            raise DojoWorkerSourceError("candles are not strictly chronological")
        seen.add(stamp_text)
        previous = stamp
        actual_instants.append(stamp)
        bid = _normalize_ohlc(item["bid"], f"candle[{index}].bid")
        ask = _normalize_ohlc(item["ask"], f"candle[{index}].ask")
        for key in ("o", "h", "l", "c"):
            if Decimal(ask[key]) < Decimal(bid[key]):
                raise DojoWorkerSourceError(
                    f"candle[{index}] ask is below bid at {key}"
                )
        rows.append(
            {
                "ask": ask,
                "bid": bid,
                "complete": True,
                "granularity": GRANULARITY,
                "pair": PAIR,
                "price": PRICE_COMPONENT,
                "time": stamp_text,
                "volume": item["volume"],
            }
        )
    missing = [slot for slot in allowed_slots if slot not in seen]
    if allowed_slots:
        numerator, denominator = (
            FULL_DAY_MINIMUM_COVERAGE
            if len(allowed_slots) >= FULL_DAY_MINIMUM_EXPECTED_SLOTS
            else PARTIAL_DAY_MINIMUM_COVERAGE
        )
        required = (len(allowed_slots) * numerator + denominator - 1) // denominator
        if len(rows) < required:
            raise DojoWorkerSourceError(
                "OANDA response is below the fixed open-market coverage floor"
            )
        first_expected = _parse_utc(allowed_slots[0], "first expected slot")
        last_expected = _parse_utc(allowed_slots[-1], "last expected slot")
        first_lag = actual_instants[0] - first_expected
        last_lead = last_expected - actual_instants[-1]
        if first_lag > BOUNDARY_TOLERANCE or last_lead > BOUNDARY_TOLERANCE:
            raise DojoWorkerSourceError(
                "OANDA response violates the fixed source boundary tolerance"
            )
        observed_gaps = [
            right - left for left, right in zip(actual_instants, actual_instants[1:])
        ]
        max_gap = max(observed_gaps, default=timedelta(0))
        if max_gap > MAX_CONTIGUOUS_GAP:
            raise DojoWorkerSourceError(
                "OANDA response violates the fixed contiguous-gap ceiling"
            )
    else:
        numerator, denominator = (100, 100)
        required = 0
        first_lag = None
        last_lead = None
        max_gap = timedelta(0)
        if rows:
            raise DojoWorkerSourceError("full market closure cannot contain candles")
    coverage = {
        "coverage_policy": COVERAGE_POLICY,
        "expected_open_slot_count": len(allowed_slots),
        "returned_slot_count": len(rows),
        "missing_slot_count": len(missing),
        "missing_slots_sha256": canonical_sha256(missing),
        "minimum_coverage_numerator": numerator,
        "minimum_coverage_denominator": denominator,
        "required_returned_slot_count": required,
        "boundary_tolerance_seconds": int(BOUNDARY_TOLERANCE.total_seconds()),
        "first_boundary_lag_seconds": (
            int(first_lag.total_seconds()) if first_lag is not None else None
        ),
        "last_boundary_lead_seconds": (
            int(last_lead.total_seconds()) if last_lead is not None else None
        ),
        "max_contiguous_gap_seconds": int(MAX_CONTIGUOUS_GAP.total_seconds()),
        "max_observed_gap_seconds": int(max_gap.total_seconds()),
        "coverage_passed": True,
    }
    return rows, coverage


def deterministic_gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    plain = b"".join(_canonical_json_bytes(row) for row in rows)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(plain)
    return buffer.getvalue()


def _collect_under_lock(
    run_dir: Path,
    *,
    ordinal: int,
    client: Any,
    now_utc: datetime,
    repo_root: Path,
    collector_paths: Sequence[Path],
    clock: Callable[[], datetime],
    fixed_clock: bool,
) -> dict[str, Any]:
    precommit, start = _load_parents(run_dir)
    day_start, day_end = _day_window(precommit, ordinal)
    now = _utc(now_utc, "now_utc")
    grace = timedelta(hours=precommit["daily_source_policy"]["seal_grace_hours"])
    if now < day_end + MINIMUM_MATURITY_DELAY:
        raise DojoWorkerSourceError("source acquisition is before M1 maturity delay")
    if now > day_end + grace:
        raise DojoWorkerSourceError("source acquisition deadline was missed")
    existing_days = sorted((run_dir / "days").glob("day-*.json"))
    if ordinal < 1 or ordinal > precommit["window"]["calendar_days"]:
        raise DojoWorkerSourceError("day ordinal is outside the precommit window")
    if len(existing_days) >= ordinal:
        return verify_collected_day(run_dir, ordinal=ordinal)
    if ordinal != len(existing_days) + 1:
        raise DojoWorkerSourceError("source collection must follow strict day ordinal")

    evidence_relpath = Path("source-evidence") / f"day-{ordinal:03d}"
    evidence_dir = _ensure_safe_directory(run_dir, evidence_relpath)
    request_path = evidence_dir / "request.json"
    source_checks = _pinned_source_checks(precommit, repo_root)
    collector_checks = _collector_source_checks(repo_root, collector_paths)
    query = {
        "from": _iso(day_start),
        "granularity": GRANULARITY,
        "includeFirst": "true",
        "price": PRICE_COMPONENT,
        "to": _iso(day_end),
    }
    request_body = {
        "contract": REQUEST_CONTRACT,
        "schema_version": 1,
        "experiment_id": precommit["experiment_id"],
        "ordinal": ordinal,
        "requested_at_utc": _iso(now),
        "day_start_utc": _iso(day_start),
        "day_end_utc": _iso(day_end),
        "precommit_sha256": precommit["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "method": "GET",
        "base_url": OFFICIAL_OANDA_BASE_URL,
        "market_hours_policy": OANDA_FX_HOURS_POLICY,
        "market_hours_source": OANDA_FX_HOURS_SOURCE,
        "coverage_policy": _coverage_policy(),
        "path": f"/v3/instruments/{PAIR}/candles",
        "query": query,
        "source_checks": source_checks,
        "collector_checks": collector_checks,
        "authority": _authority(),
    }
    request = {**request_body, "request_sha256": canonical_sha256(request_body)}
    if request_path.exists():
        request = _strict_json(request_path)
        _validate_request(
            request,
            precommit=precommit,
            start=start,
            ordinal=ordinal,
            day_start=day_start,
            day_end=day_end,
        )
        if (
            request["source_checks"] != source_checks
            or request["collector_checks"] != collector_checks
        ):
            raise DojoWorkerSourceError(
                "persisted request source bindings differ from current collector"
            )
    else:
        _write_pretty_json_new_or_same(request_path, request)
    _validate_day_one_policy_lock(run_dir, request, precommit, start, ordinal)

    receipt_path = evidence_dir / "acquisition-receipt.json"
    if receipt_path.exists():
        verified = verify_collected_day(run_dir, ordinal=ordinal)
        if verified["day_seal_sha256"] is not None:
            return verified
        manifest = _strict_json(evidence_dir / "source-manifest.json")
    else:
        capture_path = evidence_dir / "capture.json"
        if capture_path.exists():
            capture = _validate_capture(
                _strict_json(capture_path),
                request=request,
                day_end=day_end,
                grace=grace,
            )
            payload = capture["response"]
            recorded_now = _parse_utc(capture["recorded_at_utc"], "recorded_at_utc")
        else:
            if getattr(client, "base_url", None) != OFFICIAL_OANDA_BASE_URL:
                raise DojoWorkerSourceError(
                    "worker source requires the official OANDA production HTTPS host"
                )
            payload = client.get_json(request["path"], dict(request["query"]))
            recorded_now = now if fixed_clock else _utc(clock(), "recorded_at_utc")
            if recorded_now < now:
                raise DojoWorkerSourceError("source response clock moved backwards")
            if recorded_now > day_end + grace:
                raise DojoWorkerSourceError(
                    "source response arrived after acquisition deadline"
                )
            # Validate before first-write capture so malformed transport output
            # cannot poison the day's immutable namespace.
            normalize_oanda_payload(
                payload,
                day_start=day_start,
                day_end=day_end,
            )
            response_value = _snapshot(payload)
            capture_body = {
                "contract": CAPTURE_CONTRACT,
                "schema_version": 1,
                "request_sha256": request["request_sha256"],
                "recorded_at_utc": _iso(recorded_now),
                "response": response_value,
                "response_sha256": canonical_sha256(response_value),
                "authority": _authority(),
            }
            capture = {
                **capture_body,
                "capture_sha256": canonical_sha256(capture_body),
            }
            _write_pretty_json_new_or_same(capture_path, capture)
        rows, coverage = normalize_oanda_payload(
            payload,
            day_start=day_start,
            day_end=day_end,
        )
        response_bytes = _canonical_json_bytes(payload)
        response_sha = _sha256(response_bytes)
        response_relpath = (
            Path("source-evidence")
            / f"day-{ordinal:03d}"
            / f"oanda-response-{response_sha}.json"
        )
        _write_new_or_same(run_dir / response_relpath, response_bytes)

        source_bytes = deterministic_gzip_jsonl(rows)
        source_sha = _sha256(source_bytes)
        stamp_from = day_start.strftime("%Y%m%dT%H%M%SZ")
        stamp_to = day_end.strftime("%Y%m%dT%H%M%SZ")
        source_relpath = (
            Path("corpus")
            / f"day-{ordinal:03d}"
            / PAIR
            / f"{PAIR}_{GRANULARITY}_{PRICE_COMPONENT}_{stamp_from}_{stamp_to}_{source_sha}.jsonl.gz"
        )
        _ensure_safe_directory(run_dir, source_relpath.parent)
        _write_new_or_same(run_dir / source_relpath, source_bytes)
        if rows:
            manifest = {
                "market_closed": False,
                "closure_reason": None,
                "sources": [
                    {
                        "source_id": SOURCE_ID,
                        "pair": PAIR,
                        "granularity": GRANULARITY,
                        "content_sha256": source_sha,
                        "size_bytes": len(source_bytes),
                        "row_count": len(rows),
                        "first_event_utc": rows[0]["time"],
                        "last_event_utc": rows[-1]["time"],
                    }
                ],
            }
        else:
            manifest = {
                "market_closed": True,
                "closure_reason": f"FOREX_FULL_UTC_CLOSURE;OANDA_RESPONSE_SHA256:{response_sha}",
                "sources": [],
            }
        manifest_path = evidence_dir / "source-manifest.json"
        _write_json_new_or_same(manifest_path, manifest)
        receipt_body = {
            "contract": RECEIPT_CONTRACT,
            "schema_version": 1,
            "experiment_id": precommit["experiment_id"],
            "ordinal": ordinal,
            "recorded_at_utc": _iso(recorded_now),
            "day_start_utc": _iso(day_start),
            "day_end_utc": _iso(day_end),
            "precommit_sha256": precommit["precommit_sha256"],
            "start_receipt_sha256": start["start_receipt_sha256"],
            "request_sha256": request["request_sha256"],
            "request_relpath": _relative_text(request_path, run_dir),
            "capture_sha256": capture["capture_sha256"],
            "capture_relpath": _relative_text(capture_path, run_dir),
            "response_relpath": response_relpath.as_posix(),
            "response_sha256": response_sha,
            "response_size_bytes": len(response_bytes),
            "source_relpath": source_relpath.as_posix() if rows else None,
            "source_content_sha256": source_sha if rows else None,
            "source_size_bytes": len(source_bytes) if rows else None,
            "source_row_count": len(rows),
            "first_event_utc": rows[0]["time"] if rows else None,
            "last_event_utc": rows[-1]["time"] if rows else None,
            "slot_coverage": coverage,
            "source_manifest_sha256": canonical_sha256(manifest),
            "source_manifest_relpath": _relative_text(manifest_path, run_dir),
            "evidence_tier": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
            "limitations": [
                "SOURCE_PRODUCER_NOT_PRESENT_IN_ORIGINAL_PRECOMMIT",
                "UPSTREAM_HTTP_DATE_AND_EXTERNAL_WITNESS_ABSENT",
                "DOJO_HAS_NO_LIVE_AUTHORITY",
            ],
            "authority": _authority(),
        }
        receipt = {
            **receipt_body,
            "acquisition_receipt_sha256": canonical_sha256(receipt_body),
        }
        _write_pretty_json_new_or_same(receipt_path, receipt)

    previous = None
    if ordinal > 1:
        previous = _strict_json(
            _safe_artifact_path(run_dir, Path("days") / f"day-{ordinal - 1:03d}.json")
        )
    seal_now = now if fixed_clock else _utc(clock(), "sealed_at_utc")
    recorded_for_order = _parse_utc(
        _strict_json(receipt_path)["recorded_at_utc"], "recorded_at_utc"
    )
    if seal_now < recorded_for_order:
        raise DojoWorkerSourceError("source evidence clock moved backwards")
    if seal_now > day_end + grace:
        raise DojoWorkerSourceError("source seal deadline elapsed before persistence")
    seal = build_day_seal(
        precommit,
        start,
        previous,
        manifest,
        ordinal=ordinal,
        now_utc=seal_now,
    )
    days_dir = _ensure_safe_directory(run_dir, Path("days"))
    _write_pretty_json_new_or_same(days_dir / f"day-{ordinal:03d}.json", seal)
    return verify_collected_day(run_dir, ordinal=ordinal)


def _validate_request(
    request: Mapping[str, Any],
    *,
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    ordinal: int,
    day_start: datetime,
    day_end: datetime,
) -> None:
    expected_keys = {
        "contract",
        "schema_version",
        "experiment_id",
        "ordinal",
        "requested_at_utc",
        "day_start_utc",
        "day_end_utc",
        "precommit_sha256",
        "start_receipt_sha256",
        "method",
        "base_url",
        "market_hours_policy",
        "market_hours_source",
        "coverage_policy",
        "path",
        "query",
        "source_checks",
        "collector_checks",
        "authority",
        "request_sha256",
    }
    if set(request) != expected_keys:
        raise DojoWorkerSourceError("source request schema drifted")
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    if canonical_sha256(body) != _sha(request["request_sha256"], "request_sha256"):
        raise DojoWorkerSourceError("source request digest mismatch")
    expected_query = {
        "from": _iso(day_start),
        "granularity": GRANULARITY,
        "includeFirst": "true",
        "price": PRICE_COMPONENT,
        "to": _iso(day_end),
    }
    if (
        request["contract"] != REQUEST_CONTRACT
        or request["schema_version"] != 1
        or request["experiment_id"] != precommit["experiment_id"]
        or request["ordinal"] != ordinal
        or request["day_start_utc"] != _iso(day_start)
        or request["day_end_utc"] != _iso(day_end)
        or request["precommit_sha256"] != precommit["precommit_sha256"]
        or request["start_receipt_sha256"] != start["start_receipt_sha256"]
        or request["method"] != "GET"
        or request["base_url"] != OFFICIAL_OANDA_BASE_URL
        or request["market_hours_policy"] != OANDA_FX_HOURS_POLICY
        or request["market_hours_source"] != OANDA_FX_HOURS_SOURCE
        or request["coverage_policy"] != _coverage_policy()
        or request["path"] != f"/v3/instruments/{PAIR}/candles"
        or request["query"] != expected_query
        or request["authority"] != _authority()
    ):
        raise DojoWorkerSourceError("source request identity or policy drifted")
    requested = _parse_utc(request["requested_at_utc"], "requested_at_utc")
    grace = timedelta(hours=precommit["daily_source_policy"]["seal_grace_hours"])
    if requested < day_end + MINIMUM_MATURITY_DELAY or requested > day_end + grace:
        raise DojoWorkerSourceError(
            "source request timestamp is outside acquisition window"
        )
    _validate_source_checks(request["source_checks"], precommit)
    _validate_collector_checks(request["collector_checks"])


def _validate_day_one_policy_lock(
    run_dir: Path,
    request: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    ordinal: int,
) -> None:
    if ordinal == 1:
        return
    day_one_start, day_one_end = _day_window(precommit, 1)
    day_one = _strict_json(
        _safe_artifact_path(
            run_dir, Path("source-evidence/day-001/request.json")
        )
    )
    _validate_request(
        day_one,
        precommit=precommit,
        start=start,
        ordinal=1,
        day_start=day_one_start,
        day_end=day_one_end,
    )
    fixed_fields = (
        "base_url",
        "market_hours_policy",
        "market_hours_source",
        "coverage_policy",
        "path",
        "source_checks",
        "collector_checks",
        "authority",
    )
    if any(request[field] != day_one[field] for field in fixed_fields):
        raise DojoWorkerSourceError(
            "source policy or code bindings differ from the immutable day-1 lock"
        )
    fixed_query_fields = ("granularity", "includeFirst", "price")
    if any(
        request["query"][field] != day_one["query"][field]
        for field in fixed_query_fields
    ):
        raise DojoWorkerSourceError("source query policy differs from the day-1 lock")


def _coverage_policy() -> dict[str, Any]:
    return {
        "contract": COVERAGE_POLICY,
        "full_day_minimum_expected_slots": FULL_DAY_MINIMUM_EXPECTED_SLOTS,
        "full_day_minimum_coverage_numerator": FULL_DAY_MINIMUM_COVERAGE[0],
        "full_day_minimum_coverage_denominator": FULL_DAY_MINIMUM_COVERAGE[1],
        "partial_day_minimum_coverage_numerator": PARTIAL_DAY_MINIMUM_COVERAGE[0],
        "partial_day_minimum_coverage_denominator": PARTIAL_DAY_MINIMUM_COVERAGE[1],
        "max_contiguous_gap_seconds": int(MAX_CONTIGUOUS_GAP.total_seconds()),
        "boundary_tolerance_seconds": int(BOUNDARY_TOLERANCE.total_seconds()),
        "missing_slots_retained": True,
        "cross_day_policy_equality_required": True,
    }


def _validate_capture(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    day_end: datetime,
    grace: timedelta,
) -> dict[str, Any]:
    capture = _mapping(value, "source capture")
    expected_keys = {
        "contract",
        "schema_version",
        "request_sha256",
        "recorded_at_utc",
        "response",
        "response_sha256",
        "authority",
        "capture_sha256",
    }
    if set(capture) != expected_keys:
        raise DojoWorkerSourceError("source capture schema drifted")
    body = {key: item for key, item in capture.items() if key != "capture_sha256"}
    if canonical_sha256(body) != _sha(capture["capture_sha256"], "capture_sha256"):
        raise DojoWorkerSourceError("source capture digest mismatch")
    response = _mapping(capture["response"], "captured OANDA response")
    recorded = _parse_utc(capture["recorded_at_utc"], "recorded_at_utc")
    requested = _parse_utc(request["requested_at_utc"], "requested_at_utc")
    if (
        capture["contract"] != CAPTURE_CONTRACT
        or capture["schema_version"] != 1
        or capture["request_sha256"] != request["request_sha256"]
        or capture["response_sha256"] != canonical_sha256(response)
        or capture["authority"] != _authority()
        or recorded < requested
        or recorded > day_end + grace
    ):
        raise DojoWorkerSourceError("source capture identity or timing drifted")
    return _snapshot(capture)


def _validate_receipt_and_files(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    run_dir: Path,
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    ordinal: int,
    day_start: datetime,
    day_end: datetime,
) -> dict[str, Any]:
    expected_keys = {
        "contract",
        "schema_version",
        "experiment_id",
        "ordinal",
        "recorded_at_utc",
        "day_start_utc",
        "day_end_utc",
        "precommit_sha256",
        "start_receipt_sha256",
        "request_sha256",
        "request_relpath",
        "capture_sha256",
        "capture_relpath",
        "response_relpath",
        "response_sha256",
        "response_size_bytes",
        "source_relpath",
        "source_content_sha256",
        "source_size_bytes",
        "source_row_count",
        "first_event_utc",
        "last_event_utc",
        "slot_coverage",
        "source_manifest_sha256",
        "source_manifest_relpath",
        "evidence_tier",
        "limitations",
        "authority",
        "acquisition_receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise DojoWorkerSourceError("acquisition receipt schema drifted")
    body = {
        key: value
        for key, value in receipt.items()
        if key != "acquisition_receipt_sha256"
    }
    if canonical_sha256(body) != _sha(
        receipt["acquisition_receipt_sha256"], "acquisition_receipt_sha256"
    ):
        raise DojoWorkerSourceError("acquisition receipt digest mismatch")
    recorded = _parse_utc(receipt["recorded_at_utc"], "recorded_at_utc")
    grace = timedelta(hours=precommit["daily_source_policy"]["seal_grace_hours"])
    if (
        receipt["contract"] != RECEIPT_CONTRACT
        or receipt["schema_version"] != 1
        or receipt["experiment_id"] != precommit["experiment_id"]
        or receipt["ordinal"] != ordinal
        or receipt["day_start_utc"] != _iso(day_start)
        or receipt["day_end_utc"] != _iso(day_end)
        or receipt["precommit_sha256"] != precommit["precommit_sha256"]
        or receipt["start_receipt_sha256"] != start["start_receipt_sha256"]
        or receipt["request_sha256"] != request["request_sha256"]
        or receipt["request_relpath"]
        != f"source-evidence/day-{ordinal:03d}/request.json"
        or receipt["capture_relpath"]
        != f"source-evidence/day-{ordinal:03d}/capture.json"
        or receipt["source_manifest_relpath"]
        != f"source-evidence/day-{ordinal:03d}/source-manifest.json"
        or receipt["source_manifest_sha256"] != canonical_sha256(manifest)
        or receipt["evidence_tier"] != "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
        or receipt["authority"] != _authority()
        or recorded < day_end + MINIMUM_MATURITY_DELAY
        or recorded > day_end + grace
    ):
        raise DojoWorkerSourceError("acquisition receipt identity or timing drifted")
    capture_path = _safe_artifact_path(run_dir, receipt["capture_relpath"])
    capture = _validate_capture(
        _strict_json(capture_path),
        request=request,
        day_end=day_end,
        grace=grace,
    )
    if receipt["capture_sha256"] != capture["capture_sha256"]:
        raise DojoWorkerSourceError("acquisition receipt capture binding drifted")
    response_path = _safe_artifact_path(run_dir, receipt["response_relpath"])
    response_bytes = response_path.read_bytes()
    if len(response_bytes) != _integer(
        receipt["response_size_bytes"], "response_size_bytes", 1
    ):
        raise DojoWorkerSourceError("persisted OANDA response size mismatch")
    if _sha256(response_bytes) != _sha(receipt["response_sha256"], "response_sha256"):
        raise DojoWorkerSourceError("persisted OANDA response hash mismatch")
    response = _strict_json_bytes(response_bytes, "OANDA response artifact")
    if response != capture["response"]:
        raise DojoWorkerSourceError(
            "response artifact differs from first-write capture"
        )
    rows, coverage = normalize_oanda_payload(
        response,
        day_start=day_start,
        day_end=day_end,
    )
    if receipt["slot_coverage"] != coverage or receipt["source_row_count"] != len(rows):
        raise DojoWorkerSourceError("source slot coverage drifted")
    if rows:
        source_relpath = receipt["source_relpath"]
        if not isinstance(source_relpath, str):
            raise DojoWorkerSourceError("open day source path is absent")
        source_path = _safe_artifact_path(run_dir, source_relpath)
        source_bytes = source_path.read_bytes()
        source_sha = _sha(receipt["source_content_sha256"], "source_content_sha256")
        if (
            len(source_bytes)
            != _integer(receipt["source_size_bytes"], "source_size_bytes", 1)
            or _sha256(source_bytes) != source_sha
            or source_bytes != deterministic_gzip_jsonl(rows)
        ):
            raise DojoWorkerSourceError("persisted source bytes mismatch")
        expected_manifest = {
            "market_closed": False,
            "closure_reason": None,
            "sources": [
                {
                    "source_id": SOURCE_ID,
                    "pair": PAIR,
                    "granularity": GRANULARITY,
                    "content_sha256": source_sha,
                    "size_bytes": len(source_bytes),
                    "row_count": len(rows),
                    "first_event_utc": rows[0]["time"],
                    "last_event_utc": rows[-1]["time"],
                }
            ],
        }
        if (
            receipt["first_event_utc"] != rows[0]["time"]
            or receipt["last_event_utc"] != rows[-1]["time"]
        ):
            raise DojoWorkerSourceError("source event bounds drifted")
    else:
        if any(
            receipt[key] is not None
            for key in (
                "source_relpath",
                "source_content_sha256",
                "source_size_bytes",
                "first_event_utc",
                "last_event_utc",
            )
        ):
            raise DojoWorkerSourceError("closed day cannot carry a source file")
        expected_manifest = {
            "market_closed": True,
            "closure_reason": (
                "FOREX_FULL_UTC_CLOSURE;OANDA_RESPONSE_SHA256:"
                + receipt["response_sha256"]
            ),
            "sources": [],
        }
    if manifest != expected_manifest:
        raise DojoWorkerSourceError(
            "persisted source manifest is not derived from bytes"
        )
    return {
        "contract": RECEIPT_CONTRACT,
        "experiment_id": precommit["experiment_id"],
        "ordinal": ordinal,
        "market_closed": not bool(rows),
        "source_row_count": len(rows),
        "expected_open_slot_count": coverage["expected_open_slot_count"],
        "missing_slot_count": coverage["missing_slot_count"],
        "source_manifest_sha256": canonical_sha256(manifest),
        "acquisition_receipt_sha256": receipt["acquisition_receipt_sha256"],
        "promotion_eligible": False,
        "live_permission": False,
    }


def _pinned_source_checks(
    precommit: Mapping[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    commit = precommit["source_bindings"]["git_commit"]
    fixed_paths = [
        Path("src/quant_rabbit/broker/oanda.py"),
        Path("src/quant_rabbit/dojo_worker_forward.py"),
        Path("scripts/oanda_history_fetch.py"),
    ]
    dependency_paths = [
        Path(path)
        for path in precommit["source_bindings"]["bot_dependency_sha256"]
    ]
    paths = sorted(set(fixed_paths + dependency_paths), key=lambda path: path.as_posix())
    rows: list[dict[str, Any]] = []
    for relpath in paths:
        current = (repo_root / relpath).read_bytes()
        try:
            historical = subprocess.run(
                ["git", "show", f"{commit}:{relpath.as_posix()}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DojoWorkerSourceError(
                f"cannot verify pinned source {relpath.as_posix()}"
            ) from exc
        current_sha = _sha256(current)
        commit_sha = _sha256(historical)
        if current_sha != commit_sha:
            raise DojoWorkerSourceError(
                f"pinned source bytes drifted: {relpath.as_posix()}"
            )
        rows.append(
            {
                "path": relpath.as_posix(),
                "current_sha256": current_sha,
                "precommit_commit_sha256": commit_sha,
                "matches_precommit_commit": True,
            }
        )
    builder = next(
        row for row in rows if row["path"] == "src/quant_rabbit/dojo_worker_forward.py"
    )
    if (
        builder["current_sha256"]
        != precommit["source_bindings"]["precommit_builder_sha256"]
    ):
        raise DojoWorkerSourceError("worker lifecycle builder differs from precommit")
    return rows


def _collector_source_checks(
    repo_root: Path, collector_paths: Sequence[Path]
) -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("dojo_market_calendar.py").resolve(),
        *(path.resolve() for path in collector_paths),
    ]
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise DojoWorkerSourceError("collector source escapes repository") from exc
        rows.append({"path": relative, "sha256": _sha256(path.read_bytes())})
    return sorted(rows, key=lambda row: row["path"])


def _validate_source_checks(value: Any, precommit: Mapping[str, Any]) -> None:
    if not isinstance(value, list) or not value:
        raise DojoWorkerSourceError("source checks are absent")
    for row in value:
        item = _mapping(row, "source check")
        if set(item) != {
            "path",
            "current_sha256",
            "precommit_commit_sha256",
            "matches_precommit_commit",
        }:
            raise DojoWorkerSourceError("source check schema drifted")
        if item["matches_precommit_commit"] is not True:
            raise DojoWorkerSourceError("source check does not match precommit commit")
        if _sha(item["current_sha256"], "current_sha256") != _sha(
            item["precommit_commit_sha256"], "precommit_commit_sha256"
        ):
            raise DojoWorkerSourceError("source check digest mismatch")
    if precommit["source_bindings"]["git_commit"] == "":
        raise DojoWorkerSourceError("precommit commit is absent")


def _validate_collector_checks(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise DojoWorkerSourceError("collector source checks are absent")
    for row in value:
        item = _mapping(row, "collector check")
        if set(item) != {"path", "sha256"}:
            raise DojoWorkerSourceError("collector check schema drifted")
        if not isinstance(item["path"], str) or not item["path"]:
            raise DojoWorkerSourceError("collector check path is invalid")
        _sha(item["sha256"], "collector sha256")


def _load_parents(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    precommit = validate_precommit(
        _strict_json(_safe_artifact_path(run_dir, "precommit.json"))
    )
    start = validate_start_receipt(
        _strict_json(_safe_artifact_path(run_dir, "start.json")), precommit
    )
    if precommit["mechanics"]["pairs"] != [PAIR]:
        raise DojoWorkerSourceError("collector supports only the locked USD_JPY smoke")
    if precommit["mechanics"]["granularity"] != GRANULARITY or precommit[
        "daily_source_policy"
    ]["expected_source_ids"] != [SOURCE_ID]:
        raise DojoWorkerSourceError("collector identity differs from precommit")
    return precommit, start


def _day_window(
    precommit: Mapping[str, Any], ordinal: int
) -> tuple[datetime, datetime]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise DojoWorkerSourceError("day ordinal must be an integer")
    count = precommit["window"]["calendar_days"]
    if ordinal < 1 or ordinal > count:
        raise DojoWorkerSourceError("day ordinal is outside the precommit window")
    start = _parse_utc(precommit["window"]["start_utc"], "window start")
    day_start = start + timedelta(days=ordinal - 1)
    return day_start, day_start + timedelta(days=1)


def _normalize_ohlc(value: Any, label: str) -> dict[str, str]:
    source = _mapping(value, label)
    if set(source) != _OHLC_KEYS:
        raise DojoWorkerSourceError(f"{label} OHLC schema is invalid")
    normalized: dict[str, str] = {}
    numbers: dict[str, Decimal] = {}
    for key in ("o", "h", "l", "c"):
        raw = source[key]
        if not isinstance(raw, str) or _DECIMAL.fullmatch(raw) is None:
            raise DojoWorkerSourceError(f"{label}.{key} must be a plain decimal string")
        try:
            number = Decimal(raw)
        except InvalidOperation as exc:
            raise DojoWorkerSourceError(f"{label}.{key} is invalid") from exc
        if not number.is_finite() or number <= 0:
            raise DojoWorkerSourceError(f"{label}.{key} must be finite and positive")
        normalized[key] = raw
        numbers[key] = number
    if numbers["h"] < max(numbers["o"], numbers["l"], numbers["c"]):
        raise DojoWorkerSourceError(f"{label} high geometry is invalid")
    if numbers["l"] > min(numbers["o"], numbers["h"], numbers["c"]):
        raise DojoWorkerSourceError(f"{label} low geometry is invalid")
    return normalized


def _parse_m1_time(value: Any, label: str) -> tuple[datetime, str]:
    if not isinstance(value, str):
        raise DojoWorkerSourceError(f"{label} must be text")
    match = _TIME.fullmatch(value)
    if match is None:
        raise DojoWorkerSourceError(f"{label} is not an exact UTC OANDA timestamp")
    try:
        parsed = datetime.fromisoformat(match.group(1) + "+00:00")
    except ValueError as exc:
        raise DojoWorkerSourceError(f"{label} is invalid") from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        raise DojoWorkerSourceError(f"{label} is not M1-aligned")
    return parsed, _iso(parsed)


def _strict_json(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), str(path))


def _strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoWorkerSourceError(f"{label} has duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DojoWorkerSourceError(f"{label} has non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoWorkerSourceError(f"{label} is not strict JSON") from exc
    return _mapping(value, label)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_json_new_or_same(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_or_same(path, _canonical_json_bytes(value))


def _write_pretty_json_new_or_same(path: Path, value: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _write_new_or_same(path, data)


def _write_new_or_same(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending_prefix = f".{path.name}.pending-"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise DojoWorkerSourceError(f"immutable artifact is unsafe: {path}")
        if path.read_bytes() != data:
            raise DojoWorkerSourceError(f"immutable artifact already differs: {path}")
        _clean_pending_files(path.parent, pending_prefix)
        return
    descriptor, pending_text = tempfile.mkstemp(prefix=pending_prefix, dir=path.parent)
    pending = Path(pending_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise DojoWorkerSourceError(
                    f"immutable artifact concurrently differs: {path}"
                )
        pending.unlink()
        _fsync_directory(path.parent)
        _clean_pending_files(path.parent, pending_prefix)
    except BaseException:
        try:
            pending.unlink()
        except OSError:
            pass
        raise


def _clean_pending_files(parent: Path, prefix: str) -> None:
    for candidate in parent.glob(prefix + "*"):
        if candidate.is_symlink() or not candidate.is_file():
            raise DojoWorkerSourceError("pending evidence artifact is unsafe")
        candidate.unlink()
    _fsync_directory(parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, (str, Path)) or not str(relative):
        raise DojoWorkerSourceError("artifact relative path is invalid")
    relpath = Path(relative)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise DojoWorkerSourceError("artifact path escapes run directory")
    lexical = root
    for part in relpath.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise DojoWorkerSourceError("artifact path contains a symlink")
    path = lexical.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DojoWorkerSourceError("artifact path escapes run directory") from exc
    if not path.is_file():
        raise DojoWorkerSourceError("artifact path is absent")
    return path


def _safe_artifact_directory(root: Path, relative: Any) -> Path:
    if not isinstance(relative, (str, Path)):
        raise DojoWorkerSourceError("artifact directory path is invalid")
    relpath = Path(relative)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise DojoWorkerSourceError("artifact directory escapes run directory")
    path = root
    for part in relpath.parts:
        path = path / part
        if path.is_symlink() or not path.is_dir():
            raise DojoWorkerSourceError("artifact directory is absent or a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise DojoWorkerSourceError("artifact directory escapes run directory") from exc
    return resolved


def _ensure_safe_directory(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise DojoWorkerSourceError("artifact directory escapes run directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise DojoWorkerSourceError(
                    "artifact directory is not a safe directory"
                )
        else:
            path.mkdir(mode=0o700)
    return _safe_artifact_directory(root, relative)


def _relative_text(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoWorkerSourceError(f"{label} must be a mapping")
    return dict(value)


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoWorkerSourceError("source artifact is not finite JSON") from exc


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DojoWorkerSourceError(f"{label} must be lowercase SHA-256")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoWorkerSourceError(f"{label} must be an integer >= {minimum}")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoWorkerSourceError(f"{label} must be timezone-aware")
    result = value.astimezone(timezone.utc)
    if not math.isfinite(result.timestamp()):
        raise DojoWorkerSourceError(f"{label} is invalid")
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DojoWorkerSourceError(f"{label} must be UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DojoWorkerSourceError(f"{label} is invalid") from exc
    return _utc(parsed, label)


def _iso(value: datetime) -> str:
    parsed = _utc(value, "timestamp")
    if parsed.microsecond:
        return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _authority() -> dict[str, Any]:
    return {
        "broker_write_authority": False,
        "live_permission": False,
        "model_order_authority": "NONE",
        "read_only_market_data": True,
    }


__all__ = [
    "DojoWorkerSourceError",
    "MINIMUM_MATURITY_DELAY",
    "collect_and_seal_day",
    "deterministic_gzip_jsonl",
    "expected_open_slots",
    "normalize_oanda_payload",
    "verify_collected_day",
]
