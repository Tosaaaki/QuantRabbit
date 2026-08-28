#!/usr/bin/env python3
"""Refresh the pinned OANDA spread calibration from GET-only M5 BID/ASK data.

The tool deliberately avoids importing ``quant_rabbit.instruments`` because an
expired calibration must fail closed at package import time.  It writes a
replayable source-evidence artifact first, then seals the calibration against
the evidence bytes.  It never calls an account or order endpoint.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterable


UTC = timezone.utc
SESSION_START_HOUR = 12
SESSION_END_HOUR = 15
BUSINESS_DAY_COUNT = 6
EXPECTED_SAMPLES = 216
CALIBRATION_POLICY = "OANDA_M5_MBA_SESSION_SPREAD_MONTHLY_V1"
EVIDENCE_SCHEMA = "QR_OANDA_SPREAD_SOURCE_EVIDENCE_V1"
CALIBRATION_SCHEMA = "QR_OANDA_SPREAD_CALIBRATION_V1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key not in {"QR_OANDA_TOKEN", "QR_OANDA_BASE_URL"}:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_oanda_time(value: object) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        whole, fractional_and_zone = text.split(".", 1)
        zone_index = max(fractional_and_zone.find("+"), fractional_and_zone.find("-"))
        if zone_index < 0:
            raise ValueError("OANDA timestamp is missing a UTC offset")
        fraction = fractional_and_zone[:zone_index][:6].ljust(6, "0")
        zone = fractional_and_zone[zone_index:]
        text = f"{whole}.{fraction}{zone}"
    return datetime.fromisoformat(text).astimezone(UTC)


def _last_complete_business_days(now_utc: datetime) -> tuple[date, ...]:
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    cursor = now_utc.astimezone(UTC).date()
    session_end = datetime.combine(cursor, time(SESSION_END_HOUR), tzinfo=UTC)
    if now_utc.astimezone(UTC) < session_end:
        cursor -= timedelta(days=1)
    days: list[date] = []
    while len(days) < BUSINESS_DAY_COUNT:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return tuple(reversed(days))


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10_000


def _request_json(
    *, token: str, base_url: str, pair: str, start: datetime, end: datetime
) -> tuple[dict[str, Any], str, dict[str, str]]:
    query = {
        "from": _utc_text(start),
        "to": _utc_text(end),
        "granularity": "M5",
        "price": "BA",
        "smooth": "false",
        "includeFirst": "true",
    }
    path = f"/v3/instruments/{pair}/candles"
    url = f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{pair}: OANDA response is not an object")
    return payload, path, query


def _spread_samples(
    *, pair: str, payload: dict[str, Any], business_days: Iterable[date]
) -> list[dict[str, object]]:
    accepted_days = {item.isoformat() for item in business_days}
    factor = _pip_factor(pair)
    samples: list[dict[str, object]] = []
    for candle in payload.get("candles", []) or []:
        if not isinstance(candle, dict) or candle.get("complete") is not True:
            continue
        timestamp = str(candle.get("time") or "")
        try:
            at = _parse_oanda_time(timestamp)
        except ValueError:
            continue
        if at.date().isoformat() not in accepted_days:
            continue
        if not SESSION_START_HOUR <= at.hour < SESSION_END_HOUR:
            continue
        bid = candle.get("bid")
        ask = candle.get("ask")
        if not isinstance(bid, dict) or not isinstance(ask, dict):
            continue
        endpoint_spread = max(
            (float(ask["o"]) - float(bid["o"])) * factor,
            (float(ask["c"]) - float(bid["c"])) * factor,
        )
        samples.append(
            {
                "time_utc": _utc_text(at),
                "endpoint_spread_pips": round(endpoint_spread, 6),
            }
        )
    samples.sort(key=lambda item: str(item["time_utc"]))
    if len(samples) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"{pair}: expected {EXPECTED_SAMPLES} complete session samples, got {len(samples)}"
        )
    if len({str(item["time_utc"]) for item in samples}) != EXPECTED_SAMPLES:
        raise RuntimeError(f"{pair}: duplicate candle timestamp in calibration window")
    return samples


def _pair_calibration(pair: str, samples: list[dict[str, object]]) -> dict[str, object]:
    values = [float(item["endpoint_spread_pips"]) for item in samples]
    p50 = round(_nearest_rank(values, 0.50), 1)
    p95 = round(_nearest_rank(values, 0.95), 1)
    p99 = round(_nearest_rank(values, 0.99), 1)
    maximum = round(max(values), 1)
    baseline = (Decimal(str(p95)) / Decimal("2.5")).quantize(
        Decimal("0.1"), rounding=ROUND_CEILING
    )
    return {
        "pair": pair,
        "sample_count": len(samples),
        "p50_pips": float(p50),
        "p95_pips": float(p95),
        "p99_pips": float(p99),
        "max_pips": float(maximum),
        "recommended_baseline_pips": float(baseline),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=_repo_root() / "config" / "oanda_spread_calibration_v1.json",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=_repo_root() / "config" / "oanda_spread_calibration_source_v1.json.gz",
    )
    parser.add_argument("--now-utc", help="Test-only aware ISO timestamp override")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    now_utc = (
        datetime.fromisoformat(args.now_utc.replace("Z", "+00:00")).astimezone(UTC)
        if args.now_utc
        else datetime.now(UTC)
    )
    env_values = _read_env_file(args.env_file)
    token = os.environ.get("QR_OANDA_TOKEN") or env_values.get("QR_OANDA_TOKEN")
    base_url = (
        os.environ.get("QR_OANDA_BASE_URL")
        or env_values.get("QR_OANDA_BASE_URL")
        or "https://api-fxtrade.oanda.com"
    )
    if not token:
        raise RuntimeError("QR_OANDA_TOKEN is required for GET-only calibration")
    if urllib.parse.urlparse(base_url).scheme != "https":
        raise RuntimeError("QR_OANDA_BASE_URL must use HTTPS")

    existing = json.loads(args.calibration_out.read_text(encoding="utf-8"))
    pairs = [str(item["pair"]) for item in existing["pairs"]]
    business_days = _last_complete_business_days(now_utc)
    start = datetime.combine(business_days[0], time(SESSION_START_HOUR), tzinfo=UTC)
    end = datetime.combine(business_days[-1], time(SESSION_END_HOUR), tzinfo=UTC)

    evidence_pairs: list[dict[str, object]] = []
    calibrations: list[dict[str, object]] = []
    for pair in pairs:
        payload, path, query = _request_json(
            token=token,
            base_url=base_url,
            pair=pair,
            start=start,
            end=end,
        )
        samples = _spread_samples(pair=pair, payload=payload, business_days=business_days)
        candle_material = payload.get("candles", []) or []
        evidence_pairs.append(
            {
                "pair": pair,
                "request_method": "GET",
                "request_path": path,
                "request_query": query,
                "response_candles_sha256": _canonical_digest(candle_material),
                "samples": samples,
            }
        )
        calibrations.append(_pair_calibration(pair, samples))

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "policy": CALIBRATION_POLICY,
        "fetched_at_utc": _utc_text(now_utc),
        "broker": "OANDA",
        "broker_base_host": urllib.parse.urlparse(base_url).netloc,
        "http_methods_used": ["GET"],
        "broker_write_performed": False,
        "window": {"from_utc": _utc_text(start), "to_utc": _utc_text(end)},
        "business_days_utc": [item.isoformat() for item in business_days],
        "pairs": evidence_pairs,
    }
    evidence_bytes = gzip.compress(_canonical_bytes(evidence), compresslevel=9, mtime=0)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()

    calibration: dict[str, object] = {
        "schema": CALIBRATION_SCHEMA,
        "calibration_sha256": "",
        "source_evidence_sha256": evidence_sha256,
        "evidence_policy_version": CALIBRATION_POLICY,
        "max_age_days_after_window": 31,
        "valid_until_utc": _utc_text(end + timedelta(days=31)),
        "method": existing["method"],
        "window": {"from_utc": _utc_text(start), "to_utc": _utc_text(end)},
        "business_days_utc": [item.isoformat() for item in business_days],
        "session": existing["session"],
        "broker_http_methods_used": ["GET"],
        "broker_write_performed": False,
        "pairs": calibrations,
    }
    digest_material = dict(calibration)
    digest_material.pop("calibration_sha256")
    calibration["calibration_sha256"] = _canonical_digest(digest_material)
    calibration_bytes = _canonical_bytes(calibration)

    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_bytes(evidence_bytes)
    args.calibration_out.write_bytes(calibration_bytes)
    readback = {
        "ok": True,
        "broker_http_methods_used": ["GET"],
        "broker_write_performed": False,
        "window_to_utc": _utc_text(end),
        "valid_until_utc": _utc_text(end + timedelta(days=31)),
        "business_days_utc": [item.isoformat() for item in business_days],
        "pair_count": len(calibrations),
        "sample_count_per_pair": EXPECTED_SAMPLES,
        "source_evidence_sha256": evidence_sha256,
        "calibration_bytes_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
    }
    print(json.dumps(readback, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
