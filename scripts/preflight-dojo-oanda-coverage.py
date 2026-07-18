#!/usr/bin/env python3
"""Seal sanitized OANDA M1/M5 coverage calibration evidence for DOJO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.dojo_ai_discretion import canonical_sha256  # noqa: E402
from quant_rabbit.dojo_market_calendar import (  # noqa: E402
    OANDA_FX_HOURS_POLICY,
    OANDA_FX_HOURS_SOURCE,
    expected_oanda_fx_slots,
)
from quant_rabbit.dojo_worker_source import (  # noqa: E402
    BOUNDARY_TOLERANCE,
    COVERAGE_POLICY,
    FULL_DAY_MINIMUM_COVERAGE,
    MAX_CONTIGUOUS_GAP,
    PARTIAL_DAY_MINIMUM_COVERAGE,
    normalize_oanda_payload,
)


OFFICIAL_BASE_URL = "https://api-fxtrade.oanda.com"
M1_START = datetime(2026, 7, 4, tzinfo=timezone.utc)
M1_DAY_COUNT = 14
M5_CUTOFFS = tuple(
    datetime(2026, 7, day, 15, tzinfo=timezone.utc)
    for day in (6, 7, 8, 9, 13, 14, 15, 16)
)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _parse_oanda_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("OANDA calibration timestamp is invalid")
    core = value[:-1]
    if "." in core:
        head, fraction = core.split(".", 1)
        if not fraction or len(fraction) > 9 or set(fraction) != {"0"}:
            raise RuntimeError("OANDA calibration timestamp is not aligned")
        core = head
    return datetime.fromisoformat(core + "+00:00").astimezone(timezone.utc)


def _query(start: datetime, end: datetime, granularity: str) -> dict[str, str]:
    return {
        "from": _iso(start),
        "granularity": granularity,
        "includeFirst": "true",
        "price": "BA",
        "to": _iso(end),
    }


def _m5_metrics(response: Mapping[str, Any], start: datetime, end: datetime) -> dict:
    if set(response) != {"instrument", "granularity", "candles"}:
        raise RuntimeError("M5 calibration response schema drifted")
    if response["instrument"] != "USD_JPY" or response["granularity"] != "M5":
        raise RuntimeError("M5 calibration response identity drifted")
    candles = response["candles"]
    if not isinstance(candles, list) or any(
        not isinstance(row, Mapping) or row.get("complete") is not True
        for row in candles
    ):
        raise RuntimeError("M5 calibration response is incomplete")
    actual = [_parse_oanda_time(row.get("time")) for row in candles]
    if any(right <= left for left, right in zip(actual, actual[1:])):
        raise RuntimeError("M5 calibration response is not chronological")
    expected = expected_oanda_fx_slots(start, end, step=timedelta(minutes=5))
    expected_set = set(expected)
    actual_set = set(actual)
    missing = [_iso(slot) for slot in expected if slot not in actual_set]
    extras = [_iso(slot) for slot in actual if slot not in expected_set]
    max_gap = max(
        ((right - left).total_seconds() for left, right in zip(actual, actual[1:])),
        default=0.0,
    )
    required = (len(expected) * 98 + 99) // 100
    first_lag = (actual[0] - expected[0]).total_seconds() if actual else None
    last_lead = (expected[-1] - actual[-1]).total_seconds() if actual else None
    passed = (
        len(actual) >= required
        and not extras
        and first_lag is not None
        and last_lead is not None
        and first_lag <= 900
        and last_lead <= 900
        and max_gap <= 900
    )
    return {
        "expected_slot_count": len(expected),
        "returned_slot_count": len(actual),
        "covered_slot_count": sum(slot in expected_set for slot in actual),
        "missing_slot_count": len(missing),
        "missing_slots_sha256": canonical_sha256(missing),
        "extra_slot_count": len(extras),
        "extra_slots_sha256": canonical_sha256(extras),
        "required_returned_slot_count": required,
        "first_event_utc": _iso(actual[0]) if actual else None,
        "last_event_utc": _iso(actual[-1]) if actual else None,
        "max_observed_gap_seconds": int(max_gap),
        "first_boundary_lag_seconds": int(first_lag) if first_lag is not None else None,
        "last_boundary_lead_seconds": int(last_lead) if last_lead is not None else None,
        "calendar_matches_observed": not extras,
        "coverage_passed": passed,
    }


def _fetch_report(client: OandaReadOnlyClient) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    m1_rows = []
    for offset in range(M1_DAY_COUNT):
        start = M1_START + timedelta(days=offset)
        end = start + timedelta(days=1)
        query = _query(start, end, "M1")
        response = client.get_json("/v3/instruments/USD_JPY/candles", query)
        rows, coverage = normalize_oanda_payload(
            response,
            day_start=start,
            day_end=end,
        )
        m1_rows.append(
            {
                "query": query,
                "response_sha256": canonical_sha256(response),
                "response_candle_count": len(response.get("candles", [])),
                "normalized_row_count": len(rows),
                "coverage": coverage,
            }
        )
    m5_rows = []
    for cutoff in M5_CUTOFFS:
        start = cutoff - timedelta(days=1)
        query = _query(start, cutoff, "M5")
        response = client.get_json("/v3/instruments/USD_JPY/candles", query)
        m5_rows.append(
            {
                "query": query,
                "response_sha256": canonical_sha256(response),
                "response_candle_count": len(response.get("candles", [])),
                "coverage": _m5_metrics(response, start, cutoff),
            }
        )
    if not all(row["coverage"]["coverage_passed"] for row in m5_rows):
        raise RuntimeError("M5 calibration does not satisfy the frozen source policy")
    files = {
        path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        for path in (
            "src/quant_rabbit/dojo_market_calendar.py",
            "src/quant_rabbit/dojo_worker_source.py",
            "src/quant_rabbit/dojo_worker_execution.py",
            "src/quant_rabbit/dojo_lab_provenance.py",
            "src/quant_rabbit/dojo_ai_forward.py",
            "src/quant_rabbit/dojo_ai_discretion.py",
            "src/quant_rabbit/dojo_ai_truth.py",
            "src/quant_rabbit/broker/oanda.py",
            "bots/lab_bot.py",
            "scripts/run-virtual-market-session.py",
            "scripts/run-dojo-worker-forward.py",
            "scripts/run-dojo-ai-forward.py",
            "scripts/preflight-dojo-oanda-coverage.py",
        )
    }
    body = {
        "contract": "QR_DOJO_OANDA_COVERAGE_CALIBRATION_V1",
        "schema_version": 1,
        "state": "POST_HOC_CALIBRATION_BEFORE_FORWARD_WINDOW",
        "started_at_utc": _iso(started),
        "completed_at_utc": _iso(datetime.now(timezone.utc)),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_file_sha256": files,
        "source": {
            "method": "GET",
            "base_url": OFFICIAL_BASE_URL,
            "path": "/v3/instruments/USD_JPY/candles",
            "instrument": "USD_JPY",
            "price": "BA",
            "market_hours_policy": OANDA_FX_HOURS_POLICY,
            "market_hours_source": OANDA_FX_HOURS_SOURCE,
            "credentials_persisted": False,
            "raw_http_headers_persisted": False,
        },
        "calibrated_worker_policy": {
            "coverage_policy": COVERAGE_POLICY,
            "full_day_minimum_coverage": list(FULL_DAY_MINIMUM_COVERAGE),
            "partial_day_minimum_coverage": list(PARTIAL_DAY_MINIMUM_COVERAGE),
            "max_contiguous_gap_seconds": int(MAX_CONTIGUOUS_GAP.total_seconds()),
            "boundary_tolerance_seconds": int(BOUNDARY_TOLERANCE.total_seconds()),
        },
        "m1_windows": m1_rows,
        "m5_windows": m5_rows,
        "limitations": [
            "CALIBRATION_WAS_NOT_PREREGISTERED_BEFORE_OBSERVING_THESE_WINDOWS",
            "CANONICAL_RESPONSE_HASH_WITHOUT_RAW_HTTP_HEADER_ATTESTATION",
            "SELF_ATTESTED_NO_EXTERNAL_WITNESS",
            "NOT_FORWARD_PERFORMANCE_EVIDENCE",
        ],
        "promotion_eligible": False,
        "live_permission": False,
    }
    return {**body, "calibration_sha256": canonical_sha256(body)}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = OandaReadOnlyClient(env_file=args.env_file)
    if client.base_url != OFFICIAL_BASE_URL:
        raise RuntimeError("calibration requires official OANDA production HTTPS")
    report = _fetch_report(client)
    _write_new(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calibration_sha256": report["calibration_sha256"],
                "m1_window_count": len(report["m1_windows"]),
                "m5_window_count": len(report["m5_windows"]),
                "live_permission": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
