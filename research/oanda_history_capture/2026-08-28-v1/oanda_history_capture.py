#!/usr/bin/env python3
"""Immutable, GET-only OANDA LIVE historical BID/ASK M5 capture.

This module is deliberately isolated from the forward shadow ledgers.  It
downloads a fixed 730-day, completed-candle history for three FX instruments,
validates every BID/ASK OHLC row, and atomically publishes an immutable run.
It has no broker mutation surface and never stores account or token values.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import math
import os
import shutil
import ssl
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
APPROVED_LOADER_ROOT = REPO_ROOT / "research" / "shadow_runtime" / "2026-08-27-v1"
if str(APPROVED_LOADER_ROOT) not in sys.path:
    sys.path.insert(0, str(APPROVED_LOADER_ROOT))

# Importing this function has no secret or network side effect.  Only the
# capture CLI invokes it, once, immediately before the first GET.
from oanda_live_feed import load_approved_live_credentials  # noqa: E402


SCHEMA = "QR_OANDA_LIVE_HISTORICAL_M5_BA_CAPTURE_V1"
ROW_SCHEMA = "QR_OANDA_HISTORICAL_M5_BA_ROW_V1"
WINDOW_RECEIPT_SCHEMA = "QR_OANDA_HISTORY_WINDOW_RECEIPT_V1"
RUN_RECEIPT_SCHEMA = "QR_OANDA_HISTORY_RUN_RECEIPT_V1"
GAP_SCHEMA = "QR_OANDA_HISTORY_GAP_REPORT_V1"
PROVIDER = "OANDA_V20_LIVE_CANDLES"
REST_HOST = "https://api-fxtrade.oanda.com"
REST_NETLOC = "api-fxtrade.oanda.com"
HTTP_METHOD = "GET"
SYMBOLS = ("EUR_USD", "USD_JPY", "AUD_USD")
GRANULARITY = "M5"
PRICE_COMPONENT = "BA"
LOOKBACK_DAYS = 730
BAR_SECONDS = 300
MAX_CANDLES_PER_GET = 5000
# A one-slot cushion keeps an inclusive from-boundary below the provider cap.
WINDOW_GRID_SLOTS = MAX_CANDLES_PER_GET - 1
REQUEST_SPACING_SECONDS = 0.6
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_RETRIES = 3
CONTRACT_PATH = HERE / "capture_contract.json"
DEFAULT_OUTPUT_ROOT = HERE / "runs"
FX_TIMEZONE = ZoneInfo("America/New_York")
KNOWN_HOLIDAYS = {(1, 1), (12, 25)}


class CaptureError(RuntimeError):
    """Fail-closed capture or verification error with a non-secret code."""


@dataclass(frozen=True)
class Window:
    instrument: str
    index: int
    start: datetime
    end: datetime

    @property
    def id(self) -> str:
        return f"{self.instrument}:{self.index:04d}:{utc_text(self.start)}:{utc_text(self.end)}"


@dataclass(frozen=True)
class CapturePlan:
    run_id: str
    start: datetime
    end: datetime
    windows: tuple[Window, ...]
    plan_sha256: str


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise CaptureError("NAIVE_DATETIME")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_oanda_time(value: str) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, fraction = core.split(".", 1)
            text = f"{head}.{fraction[:6].ljust(6, '0')}+00:00"
        else:
            text = f"{core}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CaptureError("INVALID_OANDA_TIMESTAMP") from exc
    if parsed.tzinfo is None:
        raise CaptureError("NAIVE_OANDA_TIMESTAMP")
    return parsed.astimezone(timezone.utc)


def floor_completed_m5(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise CaptureError("NAIVE_END_TIME")
    stamp = int(value.astimezone(timezone.utc).timestamp())
    return datetime.fromtimestamp(stamp // BAR_SECONDS * BAR_SECONDS, timezone.utc)


def _load_contract() -> dict[str, Any]:
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError("CAPTURE_CONTRACT_UNREADABLE") from exc
    expected = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "rest_host": REST_HOST,
        "symbols": list(SYMBOLS),
        "granularity": GRANULARITY,
        "price_component": PRICE_COMPONENT,
        "lookback_days": LOOKBACK_DAYS,
        "max_candles_per_get": MAX_CANDLES_PER_GET,
        "request_spacing_seconds": REQUEST_SPACING_SECONDS,
        "http_method_allowlist": [HTTP_METHOD],
        "fallback_providers": [],
        "live_market_data_read": True,
        "live_order_authority": False,
        "external_orders": 0,
        "forward_pnl_included": False,
    }
    if contract != expected:
        raise CaptureError("CAPTURE_CONTRACT_MISMATCH")
    return contract


def _plan_body(start: datetime, end: datetime, windows: Iterable[Window]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "host": REST_HOST,
        "symbols": list(SYMBOLS),
        "granularity": GRANULARITY,
        "price_component": PRICE_COMPONENT,
        "window": {"from_utc": utc_text(start), "to_utc": utc_text(end)},
        "lookback_days": LOOKBACK_DAYS,
        "bar_seconds": BAR_SECONDS,
        "max_candles_per_get": MAX_CANDLES_PER_GET,
        "window_grid_slots": WINDOW_GRID_SLOTS,
        "request_spacing_seconds": REQUEST_SPACING_SECONDS,
        "windows": [
            {
                "instrument": item.instrument,
                "index": item.index,
                "from_utc": utc_text(item.start),
                "to_utc": utc_text(item.end),
                "grid_slots": int((item.end - item.start).total_seconds()) // BAR_SECONDS,
            }
            for item in windows
        ],
        "historical_input_only": True,
        "forward_pnl_included": False,
        "live_order_authority": False,
        "external_orders": 0,
    }


def build_plan(end: datetime) -> CapturePlan:
    frozen_end = floor_completed_m5(end)
    if frozen_end != end.astimezone(timezone.utc):
        raise CaptureError("END_TIME_NOT_M5_ALIGNED")
    start = frozen_end - timedelta(days=LOOKBACK_DAYS)
    windows: list[Window] = []
    for instrument in SYMBOLS:
        cursor = start
        index = 0
        while cursor < frozen_end:
            window_end = min(
                frozen_end,
                cursor + timedelta(seconds=BAR_SECONDS * WINDOW_GRID_SLOTS),
            )
            windows.append(Window(instrument, index, cursor, window_end))
            cursor = window_end
            index += 1
    body = _plan_body(start, frozen_end, windows)
    plan_sha = sha256_bytes(canonical_bytes(body))
    run_id = f"oanda-live-m5-ba-730d-{frozen_end.strftime('%Y%m%dT%H%M%SZ')}-{plan_sha[:12]}"
    return CapturePlan(run_id, start, frozen_end, tuple(windows), plan_sha)


def plan_document(plan: CapturePlan) -> dict[str, Any]:
    body = _plan_body(plan.start, plan.end, plan.windows)
    return {**body, "run_id": plan.run_id, "plan_sha256": plan.plan_sha256}


def plan_from_document(document: dict[str, Any]) -> CapturePlan:
    try:
        start = parse_oanda_time(document["window"]["from_utc"])
        end = parse_oanda_time(document["window"]["to_utc"])
        expected = build_plan(end)
    except (KeyError, TypeError) as exc:
        raise CaptureError("PARTIAL_PLAN_SCHEMA_INVALID") from exc
    if document != plan_document(expected) or start != expected.start:
        raise CaptureError("PARTIAL_PLAN_MISMATCH")
    return expected


def resolve_plan(output_root: Path, end_text: str | None, now: datetime | None = None) -> CapturePlan:
    if end_text:
        end = parse_oanda_time(end_text)
        if end != floor_completed_m5(end):
            raise CaptureError("END_TIME_NOT_M5_ALIGNED")
        current = floor_completed_m5(now or datetime.now(timezone.utc))
        if end > current:
            raise CaptureError("END_TIME_IN_FUTURE")
        return build_plan(end)
    partials = sorted(output_root.glob("oanda-live-m5-ba-730d-*.partial")) if output_root.exists() else []
    partials = [path for path in partials if path.is_dir() and not path.is_symlink()]
    if len(partials) > 1:
        raise CaptureError("MULTIPLE_PARTIAL_RUNS_REQUIRE_EXPLICIT_END")
    if partials:
        plan_path = partials[0] / "plan.json"
        if not plan_path.is_file() or plan_path.is_symlink():
            raise CaptureError("PARTIAL_PLAN_MISSING")
        return plan_from_document(json.loads(plan_path.read_text(encoding="utf-8")))
    return build_plan(floor_completed_m5(now or datetime.now(timezone.utc)))


def _secure_output_root(path: Path) -> Path:
    if path.exists() and path.is_symlink():
        raise CaptureError("OUTPUT_ROOT_SYMLINK")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = path.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise CaptureError("OUTPUT_ROOT_UNSAFE")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value) + b"\n")


def _decimal_text(value: Any) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CaptureError("INVALID_OHLC_NUMBER") from exc
    if not number.is_finite() or number <= 0:
        raise CaptureError("NONPOSITIVE_OR_NONFINITE_OHLC")
    normalized = format(number.normalize(), "f")
    return normalized if "." in normalized else f"{normalized}.0"


def _ohlc(block: Any) -> dict[str, str]:
    if not isinstance(block, dict):
        raise CaptureError("MISSING_BID_OR_ASK_OHLC")
    try:
        parsed = {key: _decimal_text(block[key]) for key in ("o", "h", "l", "c")}
    except KeyError as exc:
        raise CaptureError("INCOMPLETE_BID_OR_ASK_OHLC") from exc
    values = {key: Decimal(value) for key, value in parsed.items()}
    if values["l"] > min(values["o"], values["c"]) or values["h"] < max(values["o"], values["c"]):
        raise CaptureError("OHLC_ENVELOPE_INVALID")
    if values["l"] > values["h"]:
        raise CaptureError("OHLC_RANGE_INVALID")
    return parsed


def validate_payload(payload: Any, window: Window) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise CaptureError("RESPONSE_NOT_OBJECT")
    if payload.get("instrument") != window.instrument or payload.get("granularity") != GRANULARITY:
        raise CaptureError("RESPONSE_IDENTITY_MISMATCH")
    candles = payload.get("candles")
    if not isinstance(candles, list) or len(candles) > MAX_CANDLES_PER_GET:
        raise CaptureError("RESPONSE_CANDLE_COUNT_INVALID")
    rows: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for candle in candles:
        if not isinstance(candle, dict) or candle.get("complete") is not True:
            raise CaptureError("INCOMPLETE_OR_MALFORMED_CANDLE")
        source = parse_oanda_time(str(candle.get("time", "")))
        if source < window.start or source >= window.end:
            raise CaptureError("CANDLE_OUTSIDE_REQUEST_WINDOW")
        if int(source.timestamp()) % BAR_SECONDS:
            raise CaptureError("CANDLE_NOT_M5_ALIGNED")
        if source in seen:
            raise CaptureError("DUPLICATE_CANDLE_IN_RESPONSE")
        seen.add(source)
        bid = _ohlc(candle.get("bid"))
        ask = _ohlc(candle.get("ask"))
        if Decimal(bid["o"]) > Decimal(ask["o"]) or Decimal(bid["c"]) > Decimal(ask["c"]):
            raise CaptureError("BID_ASK_ENDPOINT_CROSSED")
        try:
            volume = int(candle.get("volume", 0))
        except (TypeError, ValueError) as exc:
            raise CaptureError("INVALID_PRICE_COUNT_VOLUME") from exc
        if volume < 0:
            raise CaptureError("NEGATIVE_PRICE_COUNT_VOLUME")
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "instrument": window.instrument,
                "granularity": GRANULARITY,
                "price_component": PRICE_COMPONENT,
                "time_utc": utc_text(source),
                "complete": True,
                "volume": volume,
                "volume_semantics": "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME",
                "bid": bid,
                "ask": ask,
            }
        )
    return sorted(rows, key=lambda row: row["time_utc"])


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


class RequestPacer:
    def __init__(self, sleeper: Callable[[float], None]):
        self.sleeper = sleeper
        self.requests = 0

    def before_request(self) -> None:
        if self.requests:
            self.sleeper(REQUEST_SPACING_SECONDS)
        self.requests += 1


def _request_window(
    account_id: str,
    token: str,
    window: Window,
    *,
    connection_factory: Any,
    pacer: RequestPacer,
    retries: int,
    progress_callback: Callable[[], None],
) -> tuple[list[dict[str, Any]], str, int]:
    query_values = {
        "from": utc_text(window.start),
        "to": utc_text(window.end),
        "granularity": GRANULARITY,
        "price": PRICE_COMPONENT,
        "smooth": "false",
        "includeFirst": "true",
    }
    query = urllib.parse.urlencode(query_values)
    target = (
        f"/v3/accounts/{urllib.parse.quote(account_id, safe='')}"
        f"/instruments/{window.instrument}/candles?{query}"
    )
    last_error = "WINDOW_FETCH_FAILED"
    for _attempt in range(max(1, retries)):
        pacer.before_request()
        progress_callback()
        connection = connection_factory(REST_NETLOC, timeout=30.0, context=ssl.create_default_context())
        try:
            connection.request(
                HTTP_METHOD,
                target,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept-Datetime-Format": "RFC3339",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                last_error = f"WINDOW_HTTP_STATUS_{int(response.status)}"
                continue
            raw = response.read()
            if len(raw) > MAX_RESPONSE_BYTES:
                raise CaptureError("WINDOW_RESPONSE_TOO_LARGE")
            try:
                payload = json.loads(raw.decode("utf-8", "strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CaptureError("WINDOW_RESPONSE_JSON_INVALID") from exc
            rows = validate_payload(payload, window)
            return rows, sha256_bytes(raw), len(raw)
        except CaptureError:
            raise
        except Exception:
            last_error = "WINDOW_TRANSPORT_FAILURE"
        finally:
            connection.close()
    raise CaptureError(last_error)


def _window_paths(partial_dir: Path, window: Window) -> tuple[Path, Path]:
    data = partial_dir / "window_cache" / window.instrument / f"{window.index:04d}.jsonl"
    meta = partial_dir / "window_meta" / window.instrument / f"{window.index:04d}.json"
    return data, meta


def _window_meta_body(
    window: Window,
    rows: list[dict[str, Any]],
    payload: bytes,
    raw_sha256: str,
    raw_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": WINDOW_RECEIPT_SCHEMA,
        "fetched_at_utc": utc_text(datetime.now(timezone.utc)),
        "window_id": window.id,
        "instrument": window.instrument,
        "index": window.index,
        "from_utc": utc_text(window.start),
        "to_utc": utc_text(window.end),
        "request_grid_slots": int((window.end - window.start).total_seconds()) // BAR_SECONDS,
        "rows": len(rows),
        "canonical_window_sha256": sha256_bytes(payload),
        "raw_response_sha256": raw_sha256,
        "raw_response_bytes": raw_bytes,
        "method": HTTP_METHOD,
        "host": REST_HOST,
        "price_component": PRICE_COMPONENT,
        "complete_only": True,
        "credential_values_persisted": 0,
        "external_orders": 0,
    }


def _verify_window_cache(window: Window, data_path: Path, meta_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = data_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError("WINDOW_CACHE_UNREADABLE") from exc
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("WINDOW_CACHE_JSONL_INVALID") from exc
    body = {key: value for key, value in meta.items() if key != "receipt_sha256"}
    if meta.get("receipt_sha256") != sha256_bytes(canonical_bytes(body)):
        raise CaptureError("WINDOW_RECEIPT_HASH_MISMATCH")
    expected_static = {
        "window_id": window.id,
        "instrument": window.instrument,
        "index": window.index,
        "from_utc": utc_text(window.start),
        "to_utc": utc_text(window.end),
        "canonical_window_sha256": sha256_bytes(payload),
        "rows": len(rows),
    }
    if any(body.get(key) != value for key, value in expected_static.items()):
        raise CaptureError("WINDOW_CACHE_PLAN_MISMATCH")
    _verify_canonical_rows(rows, window.instrument, window.start, window.end)
    if payload != _canonical_jsonl(rows):
        raise CaptureError("WINDOW_CACHE_NONCANONICAL")
    return rows, meta


def _fetch_or_resume_window(
    account_id: str,
    token: str,
    partial_dir: Path,
    window: Window,
    *,
    connection_factory: Any,
    pacer: RequestPacer,
    retries: int,
    progress_callback: Callable[[], None],
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    data_path, meta_path = _window_paths(partial_dir, window)
    if data_path.is_file() and meta_path.is_file() and not data_path.is_symlink() and not meta_path.is_symlink():
        rows, meta = _verify_window_cache(window, data_path, meta_path)
        return rows, meta, True
    rows, raw_sha, raw_bytes = _request_window(
        account_id,
        token,
        window,
        connection_factory=connection_factory,
        pacer=pacer,
        retries=retries,
        progress_callback=progress_callback,
    )
    payload = _canonical_jsonl(rows)
    body = _window_meta_body(window, rows, payload, raw_sha, raw_bytes)
    meta = {**body, "receipt_sha256": sha256_bytes(canonical_bytes(body))}
    _atomic_write(data_path, payload)
    _atomic_json(meta_path, meta)
    _verify_window_cache(window, data_path, meta_path)
    return rows, meta, False


def _verify_canonical_rows(
    rows: list[dict[str, Any]],
    instrument: str,
    start: datetime,
    end: datetime,
) -> None:
    previous: datetime | None = None
    for row in rows:
        if not isinstance(row, dict) or row.get("schema") != ROW_SCHEMA:
            raise CaptureError("CANONICAL_ROW_SCHEMA_INVALID")
        if row.get("instrument") != instrument or row.get("granularity") != GRANULARITY:
            raise CaptureError("CANONICAL_ROW_IDENTITY_INVALID")
        if row.get("price_component") != PRICE_COMPONENT or row.get("complete") is not True:
            raise CaptureError("CANONICAL_ROW_PRICE_OR_COMPLETENESS_INVALID")
        source = parse_oanda_time(str(row.get("time_utc", "")))
        if source < start or source >= end or int(source.timestamp()) % BAR_SECONDS:
            raise CaptureError("CANONICAL_ROW_TIME_INVALID")
        if previous is not None and source <= previous:
            raise CaptureError("CANONICAL_ROWS_NOT_STRICTLY_SORTED")
        previous = source
        bid, ask = _ohlc(row.get("bid")), _ohlc(row.get("ask"))
        if bid != row.get("bid") or ask != row.get("ask"):
            raise CaptureError("CANONICAL_OHLC_REPRESENTATION_INVALID")
        if Decimal(bid["o"]) > Decimal(ask["o"]) or Decimal(bid["c"]) > Decimal(ask["c"]):
            raise CaptureError("CANONICAL_BID_ASK_ENDPOINT_CROSSED")
        if row.get("volume_semantics") != "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME":
            raise CaptureError("CANONICAL_VOLUME_SEMANTICS_INVALID")
        volume = row.get("volume")
        if not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
            raise CaptureError("CANONICAL_VOLUME_INVALID")


def _classify_missing_slot(stamp: datetime) -> str:
    local = stamp.astimezone(FX_TIMEZONE)
    if local.weekday() == 4 and local.time().hour >= 17:
        return "WEEKEND_CLOSED"
    if local.weekday() == 5:
        return "WEEKEND_CLOSED"
    if local.weekday() == 6 and local.time().hour < 17:
        return "WEEKEND_CLOSED"
    if (stamp.month, stamp.day) in KNOWN_HOLIDAYS:
        return "KNOWN_HOLIDAY"
    return "UNEXPLAINED_WEEKDAY"


def analyze_gaps(rows: list[dict[str, Any]], start: datetime, end: datetime) -> dict[str, Any]:
    actual = {parse_oanda_time(row["time_utc"]) for row in rows}
    counts = {"WEEKEND_CLOSED": 0, "KNOWN_HOLIDAY": 0, "UNEXPLAINED_WEEKDAY": 0}
    intervals: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    cursor = start
    while cursor < end:
        if cursor not in actual:
            category = _classify_missing_slot(cursor)
            counts[category] += 1
            if active and active["category"] == category and active["to_utc"] == utc_text(cursor):
                active["to_utc"] = utc_text(cursor + timedelta(seconds=BAR_SECONDS))
                active["missing_slots"] += 1
            else:
                active = {
                    "category": category,
                    "from_utc": utc_text(cursor),
                    "to_utc": utc_text(cursor + timedelta(seconds=BAR_SECONDS)),
                    "missing_slots": 1,
                }
                intervals.append(active)
        else:
            active = None
        cursor += timedelta(seconds=BAR_SECONDS)
    return {
        "expected_grid_slots": int((end - start).total_seconds()) // BAR_SECONDS,
        "observed_rows": len(rows),
        "missing_slots": sum(counts.values()),
        "missing_slot_counts": counts,
        "intervals": intervals,
        "unexplained_weekday_gap_present": counts["UNEXPLAINED_WEEKDAY"] > 0,
        "missing_prices_synthesized": 0,
    }


def _merge_instrument(
    plan: CapturePlan,
    instrument: str,
    partial_dir: Path,
    publish_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_time: dict[str, dict[str, Any]] = {}
    metas: list[dict[str, Any]] = []
    for window in (item for item in plan.windows if item.instrument == instrument):
        rows, meta = _verify_window_cache(window, *_window_paths(partial_dir, window))
        metas.append(meta)
        for row in rows:
            existing = by_time.get(row["time_utc"])
            if existing is not None and existing != row:
                raise CaptureError("CONFLICTING_CROSS_WINDOW_DUPLICATE")
            by_time[row["time_utc"]] = row
    rows = [by_time[key] for key in sorted(by_time)]
    _verify_canonical_rows(rows, instrument, plan.start, plan.end)
    payload = _canonical_jsonl(rows)
    relative = Path("data") / f"{instrument}_M5_BA.jsonl"
    data_path = publish_dir / relative
    _atomic_write(data_path, payload)
    return (
        {
            "instrument": instrument,
            "path": relative.as_posix(),
            "rows": len(rows),
            "file_sha256": sha256_bytes(payload),
            "canonical_uncompressed_sha256": sha256_bytes(payload),
            "first_time_utc": rows[0]["time_utc"] if rows else None,
            "last_time_utc": rows[-1]["time_utc"] if rows else None,
            "window_receipts": len(metas),
        },
        rows,
    )


def _window_receipt_chain(plan: CapturePlan, partial_dir: Path) -> tuple[bytes, str]:
    chained: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, window in enumerate(plan.windows, 1):
        _rows, meta = _verify_window_cache(window, *_window_paths(partial_dir, window))
        body = {
            "schema": WINDOW_RECEIPT_SCHEMA,
            "sequence": sequence,
            "previous_receipt_sha256": previous,
            "window_receipt": meta,
        }
        digest = sha256_bytes(canonical_bytes(body))
        chained.append({**body, "chain_receipt_sha256": digest})
        previous = digest
    payload = _canonical_jsonl(chained)
    return payload, previous or sha256_bytes(b"")


def _dataset_sha(files: list[dict[str, Any]], publish_dir: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda item: item["instrument"]):
        digest.update(entry["instrument"].encode("ascii"))
        digest.update(b"\0")
        digest.update((publish_dir / entry["path"]).read_bytes())
    return digest.hexdigest()


def _publish(plan: CapturePlan, partial_dir: Path, progress: dict[str, Any]) -> Path:
    final_dir = partial_dir.with_name(plan.run_id)
    if final_dir.exists():
        verify_run(final_dir)
        return final_dir
    publish_dir = partial_dir / "publish"
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True, mode=0o700)
    files: list[dict[str, Any]] = []
    gap_reports: dict[str, Any] = {}
    for instrument in SYMBOLS:
        entry, rows = _merge_instrument(plan, instrument, partial_dir, publish_dir)
        files.append(entry)
        gap_reports[instrument] = analyze_gaps(rows, plan.start, plan.end)
    gap_document = {
        "schema": GAP_SCHEMA,
        "policy": {
            "weekend": "FRIDAY_17_TO_SUNDAY_17_AMERICA_NEW_YORK",
            "known_holidays_utc_month_day": ["01-01", "12-25"],
            "all_other_missing_m5_slots": "UNEXPLAINED_WEEKDAY",
            "missing_prices_synthesized": 0,
        },
        "instruments": gap_reports,
    }
    gap_path = publish_dir / "gap_report.json"
    _atomic_json(gap_path, gap_document)
    window_payload, window_head = _window_receipt_chain(plan, partial_dir)
    window_path = publish_dir / "window_receipts.jsonl"
    _atomic_write(window_path, window_payload)
    manifest = {
        "schema": SCHEMA,
        "run_id": plan.run_id,
        "status": "HISTORICAL_INPUT_ONLY_NOT_FORWARD_PNL",
        "provider": PROVIDER,
        "rest_host": REST_HOST,
        "endpoint": "/v3/accounts/{accountID}/instruments/{instrument}/candles",
        "http_method_allowlist": [HTTP_METHOD],
        "fallback_providers": [],
        "symbols": list(SYMBOLS),
        "granularity": GRANULARITY,
        "price_component": PRICE_COMPONENT,
        "window": {"from_utc": utc_text(plan.start), "to_utc": utc_text(plan.end)},
        "lookback_days": LOOKBACK_DAYS,
        "max_candles_per_get": MAX_CANDLES_PER_GET,
        "request_spacing_seconds": REQUEST_SPACING_SECONDS,
        "plan_sha256": plan.plan_sha256,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "successful_windows": len(plan.windows),
        "network_attempts": progress["network_attempts"],
        "resumed_windows": progress["resumed_windows"],
        "files": files,
        "canonical_dataset_sha256": _dataset_sha(files, publish_dir),
        "window_receipts_path": "window_receipts.jsonl",
        "window_receipts_sha256": sha256_bytes(window_payload),
        "window_receipt_chain_head": window_head,
        "gap_report_path": "gap_report.json",
        "gap_report_sha256": sha256_file(gap_path),
        "unexplained_weekday_gap_present": any(
            report["unexplained_weekday_gap_present"] for report in gap_reports.values()
        ),
        "missing_prices_synthesized": 0,
        "credential_loader": "oanda_live_feed.load_approved_live_credentials",
        "credential_values_persisted": 0,
        "historical_input_only": True,
        "forward_pnl_included": False,
        "profit_evidence": False,
        "strategy_admission_evidence": False,
        "live_order_authority": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "atomic_publish": True,
        "immutable_after_publish": True,
    }
    _atomic_json(publish_dir / "manifest.json", manifest)
    os.rename(publish_dir, final_dir)
    parent_fd = os.open(final_dir.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    verify_run(final_dir)
    shutil.rmtree(partial_dir)
    _make_read_only(final_dir)
    return final_dir


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
        elif path.is_dir() and not path.is_symlink():
            path.chmod(0o555)
    root.chmod(0o555)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("JSONL_UNREADABLE") from exc


def _verify_window_receipt_chain(rows: list[dict[str, Any]]) -> str:
    previous: str | None = None
    for index, row in enumerate(rows, 1):
        if row.get("schema") != WINDOW_RECEIPT_SCHEMA or row.get("sequence") != index:
            raise CaptureError("WINDOW_CHAIN_SCHEMA_OR_SEQUENCE_INVALID")
        if row.get("previous_receipt_sha256") != previous:
            raise CaptureError("WINDOW_CHAIN_PREVIOUS_INVALID")
        body = {key: value for key, value in row.items() if key != "chain_receipt_sha256"}
        digest = sha256_bytes(canonical_bytes(body))
        if row.get("chain_receipt_sha256") != digest:
            raise CaptureError("WINDOW_CHAIN_HASH_INVALID")
        meta = row.get("window_receipt")
        if not isinstance(meta, dict):
            raise CaptureError("WINDOW_CHAIN_META_INVALID")
        meta_body = {key: value for key, value in meta.items() if key != "receipt_sha256"}
        if meta.get("receipt_sha256") != sha256_bytes(canonical_bytes(meta_body)):
            raise CaptureError("WINDOW_META_HASH_INVALID")
        previous = digest
    return previous or sha256_bytes(b"")


def verify_run(final_dir: Path) -> dict[str, Any]:
    if not final_dir.is_dir() or final_dir.is_symlink():
        raise CaptureError("FINAL_RUN_DIRECTORY_UNSAFE")
    manifest_path = final_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError("MANIFEST_UNREADABLE") from exc
    if manifest.get("schema") != SCHEMA or manifest.get("run_id") != final_dir.name:
        raise CaptureError("MANIFEST_IDENTITY_INVALID")
    if manifest.get("symbols") != list(SYMBOLS) or manifest.get("price_component") != PRICE_COMPONENT:
        raise CaptureError("MANIFEST_UNIVERSE_INVALID")
    if manifest.get("http_method_allowlist") != [HTTP_METHOD] or manifest.get("fallback_providers") != []:
        raise CaptureError("MANIFEST_NETWORK_BOUNDARY_INVALID")
    if manifest.get("live_order_authority") or manifest.get("external_orders") != 0:
        raise CaptureError("MANIFEST_AUTHORITY_INVALID")
    if manifest.get("forward_pnl_included") or not manifest.get("historical_input_only"):
        raise CaptureError("MANIFEST_EVIDENCE_BOUNDARY_INVALID")
    start = parse_oanda_time(manifest["window"]["from_utc"])
    end = parse_oanda_time(manifest["window"]["to_utc"])
    expected_plan = build_plan(end)
    if start != expected_plan.start or manifest.get("plan_sha256") != expected_plan.plan_sha256:
        raise CaptureError("MANIFEST_PLAN_INVALID")
    files = manifest.get("files")
    if not isinstance(files, list) or [item.get("instrument") for item in files] != list(SYMBOLS):
        raise CaptureError("MANIFEST_FILE_SET_INVALID")
    verified_rows: dict[str, list[dict[str, Any]]] = {}
    for entry in files:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CaptureError("MANIFEST_FILE_PATH_INVALID")
        path = final_dir / relative
        if path.is_symlink() or not path.is_file():
            raise CaptureError("MANIFEST_FILE_MISSING")
        payload = path.read_bytes()
        if sha256_bytes(payload) != entry.get("file_sha256"):
            raise CaptureError("DATA_FILE_HASH_INVALID")
        if entry.get("canonical_uncompressed_sha256") != entry.get("file_sha256"):
            raise CaptureError("CANONICAL_UNCOMPRESSED_HASH_INVALID")
        rows = _read_jsonl(path)
        _verify_canonical_rows(rows, entry["instrument"], start, end)
        verified_rows[entry["instrument"]] = rows
        if payload != _canonical_jsonl(rows) or len(rows) != entry.get("rows"):
            raise CaptureError("DATA_FILE_CANONICAL_OR_COUNT_INVALID")
    if manifest.get("canonical_dataset_sha256") != _dataset_sha(files, final_dir):
        raise CaptureError("DATASET_HASH_INVALID")
    gap_path = final_dir / manifest["gap_report_path"]
    if sha256_file(gap_path) != manifest.get("gap_report_sha256"):
        raise CaptureError("GAP_REPORT_HASH_INVALID")
    gap_report = json.loads(gap_path.read_text(encoding="utf-8"))
    if gap_report.get("schema") != GAP_SCHEMA or gap_report.get("policy", {}).get("missing_prices_synthesized") != 0:
        raise CaptureError("GAP_REPORT_SCHEMA_INVALID")
    expected_gaps = {
        instrument: analyze_gaps(verified_rows[instrument], start, end)
        for instrument in SYMBOLS
    }
    if gap_report.get("instruments") != expected_gaps:
        raise CaptureError("GAP_REPORT_CONTENT_INVALID")
    window_path = final_dir / manifest["window_receipts_path"]
    window_payload = window_path.read_bytes()
    if sha256_bytes(window_payload) != manifest.get("window_receipts_sha256"):
        raise CaptureError("WINDOW_RECEIPT_FILE_HASH_INVALID")
    window_rows = _read_jsonl(window_path)
    if len(window_rows) != len(expected_plan.windows):
        raise CaptureError("WINDOW_RECEIPT_COUNT_INVALID")
    for receipt, expected_window in zip(window_rows, expected_plan.windows):
        meta = receipt.get("window_receipt", {})
        expected_identity = {
            "window_id": expected_window.id,
            "instrument": expected_window.instrument,
            "index": expected_window.index,
            "from_utc": utc_text(expected_window.start),
            "to_utc": utc_text(expected_window.end),
            "method": HTTP_METHOD,
            "host": REST_HOST,
            "price_component": PRICE_COMPONENT,
            "complete_only": True,
            "credential_values_persisted": 0,
            "external_orders": 0,
        }
        if any(meta.get(key) != value for key, value in expected_identity.items()):
            raise CaptureError("WINDOW_RECEIPT_PLAN_OR_AUTHORITY_INVALID")
    if _verify_window_receipt_chain(window_rows) != manifest.get("window_receipt_chain_head"):
        raise CaptureError("WINDOW_RECEIPT_HEAD_INVALID")
    return manifest


def _validate_run_receipt_chain(payload: bytes) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError("RUN_RECEIPT_LEDGER_INVALID") from exc
    previous: str | None = None
    seen: set[str] = set()
    for sequence, row in enumerate(rows, 1):
        if row.get("schema") != RUN_RECEIPT_SCHEMA or row.get("sequence") != sequence:
            raise CaptureError("RUN_RECEIPT_SCHEMA_OR_SEQUENCE_INVALID")
        if row.get("previous_receipt_sha256") != previous:
            raise CaptureError("RUN_RECEIPT_PREVIOUS_INVALID")
        if row.get("run_id") in seen:
            raise CaptureError("RUN_RECEIPT_DUPLICATE_RUN")
        seen.add(row["run_id"])
        body = {key: value for key, value in row.items() if key != "receipt_sha256"}
        digest = sha256_bytes(canonical_bytes(body))
        if row.get("receipt_sha256") != digest:
            raise CaptureError("RUN_RECEIPT_HASH_INVALID")
        previous = digest
    return rows


def _append_run_receipt(output_root: Path, final_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ledger = output_root / "run_receipts.jsonl"
    with ledger.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = _validate_run_receipt_chain(handle.read())
        manifest_sha = sha256_file(final_dir / "manifest.json")
        for row in rows:
            if row["run_id"] == final_dir.name:
                if row["manifest_sha256"] != manifest_sha:
                    raise CaptureError("RUN_RECEIPT_EXISTING_MANIFEST_MISMATCH")
                return row
        previous = rows[-1]["receipt_sha256"] if rows else None
        body = {
            "schema": RUN_RECEIPT_SCHEMA,
            "sequence": len(rows) + 1,
            "recorded_at_utc": utc_text(datetime.now(timezone.utc)),
            "previous_receipt_sha256": previous,
            "run_id": final_dir.name,
            "manifest_sha256": manifest_sha,
            "canonical_dataset_sha256": manifest["canonical_dataset_sha256"],
            "window_receipt_chain_head": manifest["window_receipt_chain_head"],
            "symbols": list(SYMBOLS),
            "window": manifest["window"],
            "historical_input_only": True,
            "forward_pnl_included": False,
            "credential_values_persisted": 0,
            "live_order_authority": False,
            "external_orders": 0,
        }
        row = {**body, "receipt_sha256": sha256_bytes(canonical_bytes(body))}
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    ledger.chmod(0o600)
    return row


def capture(
    account_id: str,
    token: str,
    output_root: Path,
    plan: CapturePlan,
    *,
    connection_factory: Any = http.client.HTTPSConnection,
    sleeper: Callable[[float], None] = time.sleep,
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    _load_contract()
    root = _secure_output_root(output_root)
    lock_path = root / "capture.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaptureError("CAPTURE_ALREADY_RUNNING") from exc
        final_dir = root / plan.run_id
        partial_dir = root / f"{plan.run_id}.partial"
        if final_dir.exists():
            manifest = verify_run(final_dir)
            receipt = _append_run_receipt(root, final_dir, manifest)
            if partial_dir.exists():
                partial_plan = plan_from_document(json.loads((partial_dir / "plan.json").read_text()))
                if partial_plan.plan_sha256 != plan.plan_sha256:
                    raise CaptureError("FINAL_AND_PARTIAL_PLAN_CONFLICT")
                shutil.rmtree(partial_dir)
            return {"status": "VERIFIED_EXISTING", "manifest": manifest, "receipt": receipt}
        if partial_dir.exists() and (partial_dir.is_symlink() or not partial_dir.is_dir()):
            raise CaptureError("PARTIAL_DIRECTORY_UNSAFE")
        partial_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        plan_path = partial_dir / "plan.json"
        expected_plan_bytes = canonical_bytes(plan_document(plan)) + b"\n"
        if plan_path.exists():
            if plan_path.is_symlink() or plan_path.read_bytes() != expected_plan_bytes:
                raise CaptureError("PARTIAL_PLAN_BYTES_MISMATCH")
        else:
            _atomic_write(plan_path, expected_plan_bytes)
        progress_path = partial_dir / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("plan_sha256") != plan.plan_sha256:
                raise CaptureError("PROGRESS_PLAN_MISMATCH")
        else:
            progress = {
                "schema": SCHEMA,
                "plan_sha256": plan.plan_sha256,
                "network_attempts": 0,
                "completed_windows": 0,
                "resumed_windows": 0,
                "credential_values_persisted": 0,
                "external_orders": 0,
            }
            _atomic_json(progress_path, progress)
        pacer = RequestPacer(sleeper)

        def record_attempt() -> None:
            progress["network_attempts"] += 1
            _atomic_json(progress_path, progress)

        completed = 0
        resumed = 0
        for window in plan.windows:
            _rows, _meta, reused = _fetch_or_resume_window(
                account_id,
                token,
                partial_dir,
                window,
                connection_factory=connection_factory,
                pacer=pacer,
                retries=retries,
                progress_callback=record_attempt,
            )
            completed += 1
            resumed += int(reused)
            progress["completed_windows"] = completed
            progress["resumed_windows"] = resumed
            _atomic_json(progress_path, progress)
        final_dir = _publish(plan, partial_dir, progress)
        manifest = verify_run(final_dir)
        receipt = _append_run_receipt(root, final_dir, manifest)
        return {"status": "PUBLISHED", "manifest": manifest, "receipt": receipt}


def _safe_summary(result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    return {
        "status": result["status"],
        "run_id": manifest["run_id"],
        "symbols": manifest["symbols"],
        "window": manifest["window"],
        "rows": {entry["instrument"]: entry["rows"] for entry in manifest["files"]},
        "network_attempts": manifest["network_attempts"],
        "canonical_dataset_sha256": manifest["canonical_dataset_sha256"],
        "unexplained_weekday_gap_present": manifest["unexplained_weekday_gap_present"],
        "credential_values_persisted": 0,
        "external_orders": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "capture"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--end-utc")
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        _load_contract()
        if args.command == "verify":
            root = _secure_output_root(args.output_root)
            final_dir = root / args.run_id
            manifest = verify_run(final_dir)
            receipt = _append_run_receipt(root, final_dir, manifest)
            print(json.dumps(_safe_summary({"status": "VERIFIED", "manifest": manifest, "receipt": receipt}), sort_keys=True))
            return 0
        root = _secure_output_root(args.output_root)
        plan = resolve_plan(root, args.end_utc)
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "run_id": plan.run_id,
                        "window": {"from_utc": utc_text(plan.start), "to_utc": utc_text(plan.end)},
                        "symbols": list(SYMBOLS),
                        "planned_gets": len(plan.windows),
                        "plan_sha256": plan.plan_sha256,
                        "credential_reads": 0,
                        "network_attempts": 0,
                        "external_orders": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
        account_id, token = load_approved_live_credentials()
        result = capture(account_id, token, root, plan)
        print(json.dumps(_safe_summary(result), sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, CaptureError) else "UNCLASSIFIED_CAPTURE_FAILURE"
        print(json.dumps({"error": type(exc).__name__, "code": code}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
