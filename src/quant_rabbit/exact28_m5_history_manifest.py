"""Fail-closed admission for staged exact-28 OANDA M5 bid/ask history.

The builder is deliberately a metadata boundary.  It admits one compressed,
receipt-backed file for every configured G8 pair and every UTC calendar-year
shard in the requested historical interval, scans every JSONL row, and emits a
compact manifest.  Candle rows are never copied into the manifest and this
module has no network, account, order, or broker-mutation surface.

The stable-file, strict-JSON, OANDA-clock, receipt-chain, and canonical-hash
primitives are shared with :mod:`quant_rabbit.fast_bot_historical_s5`.  The M5
contract is intentionally stricter around source-root cleanliness and shard
completeness; it does not relax the existing S5 admission rules.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit import fast_bot_historical_s5 as _s5
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


MANIFEST_CONTRACT = "QR_EXACT28_OANDA_M5_HISTORY_MANIFEST_V1"
MANIFEST_SCHEMA_VERSION = 1
GRANULARITY = "M5"
PRICE_COMPONENT = "BA"
SHARD_POLICY = "UTC_CALENDAR_YEAR_EXACT_HALF_OPEN_V1"
COVERAGE_POLICY = (
    "SUMMARY_REQUEST_WINDOWS_PLUS_CHAINED_RECEIPT_PLUS_STRICT_FILE_SCAN_V1"
)
ENDPOINT_CONTRACT = "OANDA_V20_INSTRUMENT_CANDLES_GET_V1"
RECEIPT_LEDGER_NAME = _s5.TRUTH_RECEIPT_FILE
LATEST_SUMMARY_NAME = "latest_summary.json"

# Research boundary, not a market/risk constant.  It is the pre-holdout history
# proposed for the 2020-2026 M5 study; callers may pass a different explicit
# historical boundary and receive a different content-addressed manifest.
DEFAULT_PERIOD_FROM_UTC = datetime(2020, 1, 1, tzinfo=timezone.utc)
DEFAULT_PERIOD_TO_UTC = datetime(2026, 7, 10, tzinfo=timezone.utc)

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
_CACHE_NAME_RE = re.compile(
    r"^(?P<pair>[A-Z]{3}_[A-Z]{3})_M5_BA_"
    r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)\.jsonl\.gz$"
)
_ROW_KEYS = frozenset(
    {"ask", "bid", "complete", "granularity", "pair", "price", "time", "volume"}
)
_PRICE_KEYS = frozenset({"o", "h", "l", "c"})
_SUMMARY_KEYS = frozenset(
    {
        "dry_run",
        "errors",
        "generated_at_utc",
        "granularities",
        "max_candles_per_request",
        "output_dir",
        "pairs",
        "price",
        "tasks",
        "total_requests",
        "total_rows",
        "window",
    }
)
_TASK_KEYS = frozenset(
    {
        "compressed",
        "dry_run",
        "errors",
        "from",
        "granularity",
        "pair",
        "partial_path",
        "path",
        "price",
        "published",
        "requests",
        "rows",
        "to",
        "truth_acquisition_receipt_sha256",
        "windows",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KEYS = frozenset(
    {
        "acquisition_receipt_proved",
        "acquisition_receipt_sequence",
        "acquisition_receipt_sha256",
        "bid_ask_ohlc_proved",
        "complete_rows_only",
        "coverage_policy",
        "file_sha256",
        "file_size_bytes",
        "first_observed_utc",
        "from_utc",
        "grid_slot_count",
        "last_observed_utc",
        "leading_gap_seconds",
        "max_candles_per_request",
        "no_tick_slot_count",
        "observed_slot_coverage",
        "pair",
        "relative_path",
        "request_count",
        "row_count",
        "run_id",
        "shard_id",
        "source_sha256",
        "strict_m5_grid_proved",
        "strictly_increasing_unique_proved",
        "summary_relative_path",
        "summary_sha256",
        "to_utc",
        "trailing_gap_seconds",
        "window_count",
    }
)
_SUMMARY_MANIFEST_KEYS = frozenset(
    {
        "pair_count",
        "relative_path",
        "request_count",
        "row_count",
        "run_id",
        "shard_id",
        "summary_sha256",
    }
)
_RECEIPT_MANIFEST_KEYS = frozenset(
    {
        "exact_file_set_proved",
        "file_sha256",
        "hash_chain_proved",
        "relative_path",
        "row_count",
        "tip_receipt_sha256",
    }
)

# A complete annual FX M5 fetch should contain materially more than the five
# open weekdays out of seven. Fifty percent catches grossly truncated files
# while still allowing weekends, holidays, and sparse no-tick intervals. This
# is an acquisition-integrity floor, not a signal or execution threshold.
MIN_OBSERVED_SLOT_COVERAGE = 0.50

# Seven days is a conservative acquisition-boundary allowance for weekend plus
# year-end holidays. Longer missing edges indicate a truncated shard.
MAX_BOUNDARY_GAP_SECONDS = 7 * 24 * 60 * 60

# Reuse the S5 verifier's exact error type so failures from its stable-FD and
# receipt-chain primitives retain one fail-closed exception boundary.
HistoricalM5ManifestError = _s5.HistoricalS5CacheError


@dataclass(frozen=True, slots=True)
class _Shard:
    shard_id: str
    time_from: datetime
    time_to: datetime
    terminal_partial_year: bool

    @property
    def grid_slot_count(self) -> int:
        return int((self.time_to - self.time_from).total_seconds() // 300)


@dataclass(frozen=True, slots=True)
class _Inventory:
    directories: frozenset[str]
    files: frozenset[str]
    summaries: tuple[Path, ...]


def build_exact28_m5_history_manifest(
    root: Path,
    *,
    period_from_utc: datetime = DEFAULT_PERIOD_FROM_UTC,
    period_to_utc: datetime = DEFAULT_PERIOD_TO_UTC,
) -> dict[str, Any]:
    """Deep-scan and seal one exact M5/BA source for every pair/year cell.

    The period starts at a UTC year boundary and ends on an M5 boundary.  Each
    run summary must cover exactly one expected annual shard and all 28 pairs.
    Duplicate cells, incomplete runs, unreceipted files, receipt orphans,
    symlinks, temporary/partial files, and any unrecognized root artifact fail
    closed before a manifest is returned.
    """

    source_root = root.resolve(strict=True)
    if not source_root.is_dir():
        raise HistoricalM5ManifestError("M5 history root is not a directory")
    period_from, period_to = _validate_period(period_from_utc, period_to_utc)
    shards = _expected_shards(period_from, period_to)
    shard_by_window = {
        (item.time_from, item.time_to): item for item in shards
    }
    inventory = _inventory_source_root(source_root)
    if len(inventory.summaries) != len(shards):
        raise HistoricalM5ManifestError(
            "M5 annual shard summary count does not match the requested period"
        )

    receipts = _s5._load_truth_receipts(source_root)
    if not receipts:
        raise HistoricalM5ManifestError("M5 truth acquisition receipt ledger is missing")
    expected_fetch_script = (
        Path(__file__).resolve().parents[2] / "scripts" / "oanda_history_fetch.py"
    ).resolve(strict=True)
    expected_fetch_sha = _s5._sha256_bytes(
        _s5._read_stable_bytes(expected_fetch_script)
    )

    source_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    admitted_files: set[str] = set()
    admitted_receipts: set[str] = set()
    admitted_directories: set[str] = set()
    seen_cells: set[tuple[str, str]] = set()
    seen_shards: set[str] = set()
    summary_payloads: list[bytes] = []

    for summary_path in inventory.summaries:
        run_id = summary_path.parent.name
        _validate_run_id(run_id)
        summary_bytes = _s5._read_stable_bytes(summary_path)
        summary_payloads.append(summary_bytes)
        summary = _s5._strict_json_loads(summary_bytes, label=str(summary_path))
        if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
            raise HistoricalM5ManifestError("M5 run summary schema is invalid")
        shard, tasks, max_candles = _validate_summary(
            summary,
            run_id=run_id,
            shard_by_window=shard_by_window,
        )
        if shard.shard_id in seen_shards:
            raise HistoricalM5ManifestError("duplicate M5 annual shard summary")
        seen_shards.add(shard.shard_id)
        summary_relative = _s5._relative_inside(source_root, summary_path)
        summary_sha = _s5._sha256_bytes(summary_bytes)
        admitted_files.add(summary_relative)
        admitted_directories.add(run_id)

        task_rows: list[dict[str, Any]] = []
        for task in tasks:
            pair = str(task["pair"])
            cell = (pair, shard.shard_id)
            if cell in seen_cells:
                raise HistoricalM5ManifestError("duplicate M5 pair/year shard cell")
            seen_cells.add(cell)
            row, relative_path, receipt_sha = _validate_task_and_file(
                source_root=source_root,
                summary_path=summary_path,
                summary_sha256=summary_sha,
                task=task,
                shard=shard,
                max_candles_per_request=max_candles,
                receipts=receipts,
                expected_fetch_script=expected_fetch_script,
                expected_fetch_sha256=expected_fetch_sha,
            )
            source_rows.append(row)
            task_rows.append(row)
            admitted_files.add(relative_path)
            admitted_receipts.add(relative_path)
            admitted_directories.add(f"{run_id}/{pair}")

        summary_rows.append(
            {
                "run_id": run_id,
                "shard_id": shard.shard_id,
                "relative_path": summary_relative,
                "summary_sha256": summary_sha,
                "pair_count": len(task_rows),
                "row_count": sum(int(row["row_count"]) for row in task_rows),
                "request_count": sum(
                    int(row["request_count"]) for row in task_rows
                ),
            }
        )

    expected_cells = {
        (pair, shard.shard_id)
        for pair in DEFAULT_TRADER_PAIRS
        for shard in shards
    }
    if seen_cells != expected_cells:
        missing = sorted(expected_cells - seen_cells)
        extra = sorted(seen_cells - expected_cells)
        raise HistoricalM5ManifestError(
            f"M5 exact-28 annual shard coverage mismatch: missing={missing} extra={extra}"
        )
    if seen_shards != {item.shard_id for item in shards}:
        raise HistoricalM5ManifestError("M5 annual shard set is incomplete")

    receipt_paths = set(receipts)
    if receipt_paths != admitted_receipts:
        missing = sorted(admitted_receipts - receipt_paths)
        orphaned = sorted(receipt_paths - admitted_receipts)
        raise HistoricalM5ManifestError(
            f"M5 receipt/file set is not exact: missing={missing} orphaned={orphaned}"
        )
    admitted_files.add(RECEIPT_LEDGER_NAME)

    if LATEST_SUMMARY_NAME in inventory.files:
        latest_path = source_root / LATEST_SUMMARY_NAME
        latest_bytes = _s5._read_stable_bytes(latest_path)
        if latest_bytes not in summary_payloads:
            raise HistoricalM5ManifestError(
                "latest M5 summary is not byte-identical to an admitted run summary"
            )
        admitted_files.add(LATEST_SUMMARY_NAME)

    orphan_files = sorted(inventory.files - admitted_files)
    orphan_directories = sorted(inventory.directories - admitted_directories)
    if orphan_files or orphan_directories:
        raise HistoricalM5ManifestError(
            "M5 source root contains orphan artifacts: "
            f"files={orphan_files} directories={orphan_directories}"
        )

    source_rows.sort(
        key=lambda row: (
            DEFAULT_TRADER_PAIRS.index(str(row["pair"])),
            str(row["shard_id"]),
        )
    )
    summary_rows.sort(key=lambda row: str(row["shard_id"]))
    receipt_tip = max(receipts.values(), key=lambda row: int(row["sequence"]))
    receipt_ledger_path = source_root / RECEIPT_LEDGER_NAME
    receipt_ledger_sha = _s5._sha256_bytes(
        _s5._read_stable_bytes(receipt_ledger_path)
    )
    body: dict[str, Any] = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_root": str(source_root),
        "pair_universe": list(DEFAULT_TRADER_PAIRS),
        "expected_pair_count": len(DEFAULT_TRADER_PAIRS),
        "granularity": GRANULARITY,
        "price_component": PRICE_COMPONENT,
        "period_from_utc": period_from.isoformat(),
        "period_to_utc": period_to.isoformat(),
        "period_is_half_open": True,
        "shard_policy": SHARD_POLICY,
        "coverage_policy": COVERAGE_POLICY,
        "expected_shards": [
            {
                "shard_id": item.shard_id,
                "from_utc": item.time_from.isoformat(),
                "to_utc": item.time_to.isoformat(),
                "terminal_partial_year": item.terminal_partial_year,
                "grid_slot_count": item.grid_slot_count,
            }
            for item in shards
        ],
        "expected_shard_count": len(shards),
        "expected_pair_shard_count": len(expected_cells),
        "selected_pair_shard_count": len(source_rows),
        "complete_exact28_annual_shard_coverage": True,
        "summaries": summary_rows,
        "sources": source_rows,
        "receipt_ledger": {
            "relative_path": RECEIPT_LEDGER_NAME,
            "file_sha256": receipt_ledger_sha,
            "row_count": len(receipts),
            "tip_receipt_sha256": receipt_tip["receipt_sha256"],
            "exact_file_set_proved": True,
            "hash_chain_proved": True,
        },
        "endpoint_identity": {
            "contract": ENDPOINT_CONTRACT,
            "method": "GET",
            "path_template": "/v3/instruments/{pair}/candles",
            "query_granularity": GRANULARITY,
            "query_price_component": PRICE_COMPONENT,
            "include_first": True,
            "fetch_script_path": str(expected_fetch_script),
            "fetch_script_sha256": expected_fetch_sha,
            "method_path_bound_by_fetch_script_sha256": True,
            "request_pair_bound_by_task_receipt_and_rows": True,
            "response_top_level_instrument_receipted": False,
            "base_url_receipted": False,
        },
        "source_root_clean": True,
        "temporary_or_partial_artifact_count": 0,
        "orphan_file_count": 0,
        "orphan_directory_count": 0,
        "raw_candles_embedded": False,
        "no_tick_intervals_allowed": True,
        "every_calendar_slot_asserted_present": False,
        "limitations": [
            "OANDA_BASE_URL_NOT_RECEIPTED_BY_FETCH_V1",
            "OANDA_RESPONSE_TOP_LEVEL_INSTRUMENT_NOT_PERSISTED_BY_FETCH_V1",
            "LOCAL_HASH_CHAIN_HAS_NO_EXTERNAL_MONOTONIC_ANCHOR",
            "HISTORICAL_DATA_IS_NOT_FORWARD_PROOF",
        ],
        "historical_only": True,
        "diagnostic_only": True,
        "forward_proof_eligible": False,
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "order_authority": "NONE",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    manifest = {**body, "manifest_sha256": _s5._canonical_sha(body)}
    validate_exact28_m5_history_manifest(manifest)
    return manifest


def validate_exact28_m5_history_manifest(value: Mapping[str, Any]) -> None:
    """Validate the compact manifest's digest, exact scope, and safety flags."""

    if not isinstance(value, Mapping):
        raise HistoricalM5ManifestError("M5 history manifest is not an object")
    if (
        value.get("contract") != MANIFEST_CONTRACT
        or value.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise HistoricalM5ManifestError("M5 history manifest contract mismatch")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != _s5._canonical_sha(body):
        raise HistoricalM5ManifestError("M5 history manifest digest mismatch")
    if value.get("pair_universe") != list(DEFAULT_TRADER_PAIRS):
        raise HistoricalM5ManifestError("M5 history pair universe mismatch")
    if value.get("expected_pair_count") != len(DEFAULT_TRADER_PAIRS):
        raise HistoricalM5ManifestError("M5 history pair count mismatch")
    if value.get("granularity") != GRANULARITY or value.get("price_component") != PRICE_COMPONENT:
        raise HistoricalM5ManifestError("M5 history market-data scope mismatch")
    period_from, period_to = _validate_period(
        _s5._parse_aware_utc(value.get("period_from_utc")),
        _s5._parse_aware_utc(value.get("period_to_utc")),
    )
    expected_shards = _expected_shards(period_from, period_to)
    shard_rows = value.get("expected_shards")
    if not isinstance(shard_rows, list) or len(shard_rows) != len(expected_shards):
        raise HistoricalM5ManifestError("M5 history shard manifest mismatch")
    for actual, expected in zip(shard_rows, expected_shards, strict=True):
        if actual != {
            "shard_id": expected.shard_id,
            "from_utc": expected.time_from.isoformat(),
            "to_utc": expected.time_to.isoformat(),
            "terminal_partial_year": expected.terminal_partial_year,
            "grid_slot_count": expected.grid_slot_count,
        }:
            raise HistoricalM5ManifestError("M5 history shard row mismatch")
    expected_count = len(DEFAULT_TRADER_PAIRS) * len(expected_shards)
    if (
        value.get("expected_shard_count") != len(expected_shards)
        or value.get("expected_pair_shard_count") != expected_count
        or value.get("selected_pair_shard_count") != expected_count
        or value.get("complete_exact28_annual_shard_coverage") is not True
    ):
        raise HistoricalM5ManifestError("M5 history completeness claim mismatch")
    summaries = value.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != len(expected_shards):
        raise HistoricalM5ManifestError("M5 history summary manifest mismatch")
    if [row.get("shard_id") for row in summaries if isinstance(row, Mapping)] != [
        shard.shard_id for shard in expected_shards
    ]:
        raise HistoricalM5ManifestError("M5 history summary shard order mismatch")
    for summary in summaries:
        if (
            not isinstance(summary, Mapping)
            or set(summary) != _SUMMARY_MANIFEST_KEYS
        ):
            raise HistoricalM5ManifestError("M5 history summary row is invalid")
        if summary.get("pair_count") != len(DEFAULT_TRADER_PAIRS):
            raise HistoricalM5ManifestError("M5 history summary pair count mismatch")
        if _SHA256_RE.fullmatch(str(summary.get("summary_sha256") or "")) is None:
            raise HistoricalM5ManifestError("M5 history summary digest is invalid")
        if not _safe_relative_path(summary.get("relative_path"), expected_parts=2):
            raise HistoricalM5ManifestError("M5 history summary path is unsafe")
        if (
            summary.get("row_count").__class__ is not int
            or summary["row_count"] <= 0
            or summary.get("request_count").__class__ is not int
            or summary["request_count"] <= 0
        ):
            raise HistoricalM5ManifestError("M5 history summary count is invalid")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != expected_count:
        raise HistoricalM5ManifestError("M5 history source count mismatch")
    expected_cells = [
        (pair, shard.shard_id)
        for pair in DEFAULT_TRADER_PAIRS
        for shard in expected_shards
    ]
    actual_cells: list[tuple[str, str]] = []
    shard_by_id = {item.shard_id: item for item in expected_shards}
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
            raise HistoricalM5ManifestError("M5 history source row is invalid")
        source_body = {
            key: item for key, item in source.items() if key != "source_sha256"
        }
        if source.get("source_sha256") != _s5._canonical_sha(source_body):
            raise HistoricalM5ManifestError("M5 history source digest mismatch")
        pair = source.get("pair")
        shard_id = source.get("shard_id")
        if pair not in DEFAULT_TRADER_PAIRS or not isinstance(shard_id, str):
            raise HistoricalM5ManifestError("M5 history source identity mismatch")
        actual_cells.append((str(pair), shard_id))
        if source.get("acquisition_receipt_proved") is not True:
            raise HistoricalM5ManifestError("M5 history source is unreceipted")
        for digest_key in (
            "file_sha256",
            "summary_sha256",
            "acquisition_receipt_sha256",
            "source_sha256",
        ):
            if _SHA256_RE.fullmatch(str(source.get(digest_key) or "")) is None:
                raise HistoricalM5ManifestError("M5 history source digest is invalid")
        if not _safe_relative_path(source.get("relative_path"), expected_parts=3):
            raise HistoricalM5ManifestError("M5 history source path is unsafe")
        if not _safe_relative_path(
            source.get("summary_relative_path"), expected_parts=2
        ):
            raise HistoricalM5ManifestError("M5 history source summary path is unsafe")
        shard = shard_by_id.get(shard_id)
        if shard is None or (
            source.get("from_utc") != shard.time_from.isoformat()
            or source.get("to_utc") != shard.time_to.isoformat()
            or source.get("grid_slot_count") != shard.grid_slot_count
        ):
            raise HistoricalM5ManifestError("M5 history source shard bounds mismatch")
        rows = source.get("row_count")
        no_ticks = source.get("no_tick_slot_count")
        if (
            rows.__class__ is not int
            or rows <= 0
            or no_ticks.__class__ is not int
            or no_ticks < 0
            or rows + no_ticks != shard.grid_slot_count
        ):
            raise HistoricalM5ManifestError(
                "M5 history source coverage arithmetic mismatch"
            )
        if any(
            source.get(flag) is not True
            for flag in (
                "strict_m5_grid_proved",
                "strictly_increasing_unique_proved",
                "bid_ask_ohlc_proved",
                "complete_rows_only",
            )
        ):
            raise HistoricalM5ManifestError("M5 history source proof flag mismatch")
        if source.get("coverage_policy") != COVERAGE_POLICY:
            raise HistoricalM5ManifestError("M5 history source policy mismatch")
        coverage = source.get("observed_slot_coverage")
        if (
            coverage.__class__ not in {int, float}
            or not math.isfinite(float(coverage))
            or float(coverage) < MIN_OBSERVED_SLOT_COVERAGE
            or float(coverage) != rows / shard.grid_slot_count
        ):
            raise HistoricalM5ManifestError("M5 history source coverage ratio mismatch")
        for gap_key in ("leading_gap_seconds", "trailing_gap_seconds"):
            gap = source.get(gap_key)
            if (
                gap.__class__ is not int
                or gap < 0
                or gap > MAX_BOUNDARY_GAP_SECONDS
            ):
                raise HistoricalM5ManifestError("M5 history source boundary gap invalid")
    if actual_cells != expected_cells:
        raise HistoricalM5ManifestError("M5 history source ordering/scope mismatch")
    receipt_ledger = value.get("receipt_ledger")
    if (
        not isinstance(receipt_ledger, Mapping)
        or set(receipt_ledger) != _RECEIPT_MANIFEST_KEYS
        or receipt_ledger.get("relative_path") != RECEIPT_LEDGER_NAME
        or receipt_ledger.get("row_count") != expected_count
        or receipt_ledger.get("exact_file_set_proved") is not True
        or receipt_ledger.get("hash_chain_proved") is not True
        or _SHA256_RE.fullmatch(str(receipt_ledger.get("file_sha256") or ""))
        is None
        or _SHA256_RE.fullmatch(
            str(receipt_ledger.get("tip_receipt_sha256") or "")
        )
        is None
    ):
        raise HistoricalM5ManifestError("M5 history receipt ledger manifest mismatch")
    endpoint = value.get("endpoint_identity")
    if not isinstance(endpoint, Mapping) or endpoint.get("contract") != ENDPOINT_CONTRACT:
        raise HistoricalM5ManifestError("M5 history endpoint identity mismatch")
    if (
        endpoint.get("method") != "GET"
        or endpoint.get("path_template") != "/v3/instruments/{pair}/candles"
        or endpoint.get("query_granularity") != GRANULARITY
        or endpoint.get("query_price_component") != PRICE_COMPONENT
        or endpoint.get("method_path_bound_by_fetch_script_sha256") is not True
        or endpoint.get("response_top_level_instrument_receipted") is not False
        or endpoint.get("base_url_receipted") is not False
    ):
        raise HistoricalM5ManifestError("M5 history endpoint semantics mismatch")
    if value.get("raw_candles_embedded") is not False:
        raise HistoricalM5ManifestError("M5 history manifest embeds raw candles")
    if (
        value.get("shard_policy") != SHARD_POLICY
        or value.get("coverage_policy") != COVERAGE_POLICY
        or value.get("no_tick_intervals_allowed") is not True
        or value.get("every_calendar_slot_asserted_present") is not False
    ):
        raise HistoricalM5ManifestError("M5 history policy mismatch")
    for required_true in (
        "period_is_half_open",
        "source_root_clean",
        "historical_only",
        "diagnostic_only",
        "shadow_only",
    ):
        if value.get(required_true) is not True:
            raise HistoricalM5ManifestError(
                f"M5 history manifest required flag mismatch: {required_true}"
            )
    for required_false in (
        "forward_proof_eligible",
        "automatic_promotion_allowed",
        "promotion_allowed",
        "live_permission",
        "broker_mutation_allowed",
    ):
        if value.get(required_false) is not False:
            raise HistoricalM5ManifestError(
                f"M5 history manifest unsafe flag: {required_false}"
            )
    if value.get("order_authority") != "NONE":
        raise HistoricalM5ManifestError("M5 history order authority mismatch")
    if any(
        value.get(key) != 0
        for key in (
            "temporary_or_partial_artifact_count",
            "orphan_file_count",
            "orphan_directory_count",
        )
    ):
        raise HistoricalM5ManifestError("M5 history cleanliness claim mismatch")


def write_exact28_m5_history_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically persist a validated metadata-only manifest."""

    validate_exact28_m5_history_manifest(manifest)
    destination = path.resolve()
    source_root = Path(str(manifest["source_root"])).resolve()
    try:
        destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise HistoricalM5ManifestError(
            "M5 manifest output must be outside its immutable source root"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_exact28_m5_history_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a compact manifest without rescanning raw candles."""

    if path.is_symlink() or not path.is_file():
        raise HistoricalM5ManifestError("M5 history manifest path is invalid")
    value = _s5._strict_json_loads(_s5._read_stable_bytes(path), label=str(path))
    if not isinstance(value, dict):
        raise HistoricalM5ManifestError("M5 history manifest is not an object")
    validate_exact28_m5_history_manifest(value)
    return value


def _validate_period(
    period_from_utc: datetime,
    period_to_utc: datetime,
) -> tuple[datetime, datetime]:
    time_from = _s5._aware_utc(period_from_utc)
    time_to = _s5._aware_utc(period_to_utc)
    if time_from >= time_to:
        raise HistoricalM5ManifestError("M5 history period must be positive")
    if time_from != datetime(time_from.year, 1, 1, tzinfo=timezone.utc):
        raise HistoricalM5ManifestError(
            "M5 history period must start at a UTC calendar-year boundary"
        )
    if _floor_m5(time_to) != time_to:
        raise HistoricalM5ManifestError("M5 history period end is off the M5 grid")
    if time_to > datetime.now(timezone.utc):
        raise HistoricalM5ManifestError("M5 history period end is in the future")
    return time_from, time_to


def _expected_shards(time_from: datetime, time_to: datetime) -> tuple[_Shard, ...]:
    shards: list[_Shard] = []
    cursor = time_from
    while cursor < time_to:
        next_year = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        shard_to = min(next_year, time_to)
        shards.append(
            _Shard(
                shard_id=str(cursor.year),
                time_from=cursor,
                time_to=shard_to,
                terminal_partial_year=shard_to != next_year,
            )
        )
        cursor = shard_to
    return tuple(shards)


def _inventory_source_root(root: Path) -> _Inventory:
    directories: set[str] = set()
    files: set[str] = set()
    summaries: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(dirnames):
            path = directory_path / name
            if path.is_symlink():
                raise HistoricalM5ManifestError("M5 source root contains a symlink")
            directories.add(_s5._relative_inside(root, path))
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise HistoricalM5ManifestError(
                    "M5 source root contains a non-regular file"
                )
            relative = _s5._relative_inside(root, path)
            if name.endswith((".tmp", ".partial")) or ".tmp." in name or ".partial." in name:
                raise HistoricalM5ManifestError(
                    f"M5 source root contains temporary/partial debris: {relative}"
                )
            files.add(relative)
            if name == "summary.json":
                if len(Path(relative).parts) != 2:
                    raise HistoricalM5ManifestError(
                        "M5 summary path is outside one canonical run directory"
                    )
                summaries.append(path)
    return _Inventory(
        directories=frozenset(directories),
        files=frozenset(files),
        summaries=tuple(sorted(summaries)),
    )


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise HistoricalM5ManifestError("M5 run id is not canonical")
    try:
        parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise HistoricalM5ManifestError("M5 run id clock is invalid") from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != run_id:
        raise HistoricalM5ManifestError("M5 run id is not canonical")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    shard_by_window: Mapping[tuple[datetime, datetime], _Shard],
) -> tuple[_Shard, tuple[Mapping[str, Any], ...], int]:
    if summary.get("dry_run") is not False or summary.get("errors") != []:
        raise HistoricalM5ManifestError("M5 run summary is not error-free/published")
    if summary.get("granularities") != [GRANULARITY] or summary.get("price") != PRICE_COMPONENT:
        raise HistoricalM5ManifestError("M5 run summary market-data scope mismatch")
    if summary.get("pairs") != list(DEFAULT_TRADER_PAIRS):
        raise HistoricalM5ManifestError("M5 run summary is not exact configured 28")
    output_dir = summary.get("output_dir")
    if not isinstance(output_dir, str) or Path(output_dir).name != run_id:
        raise HistoricalM5ManifestError("M5 run summary output directory mismatch")
    window = summary.get("window")
    if not isinstance(window, Mapping) or set(window) != {"from", "to"}:
        raise HistoricalM5ManifestError("M5 run summary window schema is invalid")
    window_key = (
        _s5._parse_aware_utc(window["from"]),
        _s5._parse_aware_utc(window["to"]),
    )
    shard = shard_by_window.get(window_key)
    if shard is None:
        raise HistoricalM5ManifestError("M5 run summary is not an expected annual shard")
    max_candles = summary.get("max_candles_per_request")
    if max_candles.__class__ is not int or not 1 <= max_candles <= 5_000:
        raise HistoricalM5ManifestError("M5 max candles per request is invalid")
    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(DEFAULT_TRADER_PAIRS):
        raise HistoricalM5ManifestError("M5 run summary task count mismatch")
    if any(not isinstance(task, Mapping) or set(task) != _TASK_KEYS for task in tasks):
        raise HistoricalM5ManifestError("M5 run summary task schema is invalid")
    if [task.get("pair") for task in tasks] != list(DEFAULT_TRADER_PAIRS):
        raise HistoricalM5ManifestError("M5 run summary task pair order mismatch")
    if any(
        _s5._parse_aware_utc(task.get("from")) != shard.time_from
        or _s5._parse_aware_utc(task.get("to")) != shard.time_to
        for task in tasks
    ):
        raise HistoricalM5ManifestError("M5 task window differs from annual shard")
    total_rows = summary.get("total_rows")
    total_requests = summary.get("total_requests")
    if (
        total_rows.__class__ is not int
        or total_requests.__class__ is not int
        or total_rows != sum(int(task["rows"]) for task in tasks)
        or total_requests != sum(int(task["requests"]) for task in tasks)
    ):
        raise HistoricalM5ManifestError("M5 run summary aggregate arithmetic mismatch")
    return shard, tuple(tasks), max_candles


def _validate_task_and_file(
    *,
    source_root: Path,
    summary_path: Path,
    summary_sha256: str,
    task: Mapping[str, Any],
    shard: _Shard,
    max_candles_per_request: int,
    receipts: Mapping[str, Mapping[str, Any]],
    expected_fetch_script: Path,
    expected_fetch_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    pair = str(task.get("pair") or "")
    if pair not in DEFAULT_TRADER_PAIRS:
        raise HistoricalM5ManifestError("M5 task pair is outside exact configured 28")
    if (
        task.get("granularity") != GRANULARITY
        or task.get("price") != PRICE_COMPONENT
        or task.get("dry_run") is not False
        or task.get("published") is not True
        or task.get("errors") != []
        or task.get("partial_path") is not None
        or task.get("compressed") is not True
    ):
        raise HistoricalM5ManifestError("M5 task publication contract mismatch")
    task_path_text = task.get("path")
    if not isinstance(task_path_text, str) or not task_path_text:
        raise HistoricalM5ManifestError("M5 task path is invalid")
    task_path = Path(task_path_text)
    match = _CACHE_NAME_RE.fullmatch(task_path.name)
    if match is None or match.group("pair") != pair:
        raise HistoricalM5ManifestError("M5 cache filename scope mismatch")
    if (
        match.group("start") != _s5._filename_stamp(shard.time_from)
        or match.group("end") != _s5._filename_stamp(shard.time_to)
    ):
        raise HistoricalM5ManifestError("M5 cache filename window mismatch")
    run_id = summary_path.parent.name
    if tuple(task_path.parts[-3:]) != (run_id, pair, task_path.name):
        raise HistoricalM5ManifestError("M5 cache task path/run binding mismatch")
    source_path = summary_path.parent / pair / task_path.name
    if source_path.is_symlink() or not source_path.is_file():
        raise HistoricalM5ManifestError("M5 cache source is not a regular file")
    relative_path = _s5._relative_inside(source_root, source_path)
    rows = task.get("rows")
    requests = task.get("requests")
    windows = task.get("windows")
    expected_windows = _expected_request_windows(
        shard.time_from,
        shard.time_to,
        max_candles_per_request=max_candles_per_request,
    )
    if rows.__class__ is not int or rows <= 0:
        raise HistoricalM5ManifestError("M5 cache row count is invalid")
    if (
        requests.__class__ is not int
        or windows.__class__ is not int
        or requests != windows
        or windows != expected_windows
    ):
        raise HistoricalM5ManifestError("M5 request/window coverage is incomplete")
    receipt_sha = task.get("truth_acquisition_receipt_sha256")
    if _SHA256_RE.fullmatch(str(receipt_sha or "")) is None:
        raise HistoricalM5ManifestError("M5 task receipt digest is invalid")
    receipt = receipts.get(relative_path)
    if receipt is None or receipt.get("receipt_sha256") != receipt_sha:
        raise HistoricalM5ManifestError("M5 task receipt binding is missing/mismatched")
    scan = _scan_m5_cache(
        source_path,
        pair=pair,
        time_from=shard.time_from,
        time_to=shard.time_to,
    )
    if scan["row_count"] != rows:
        raise HistoricalM5ManifestError("M5 cache row count differs from summary")
    if float(scan["observed_slot_coverage"]) < MIN_OBSERVED_SLOT_COVERAGE:
        raise HistoricalM5ManifestError(
            "M5 cache observed-slot coverage is implausibly low"
        )
    if (
        int(scan["leading_gap_seconds"]) > MAX_BOUNDARY_GAP_SECONDS
        or int(scan["trailing_gap_seconds"]) > MAX_BOUNDARY_GAP_SECONDS
    ):
        raise HistoricalM5ManifestError("M5 cache boundary coverage is truncated")
    _validate_m5_receipt_for_candidate(
        receipt,
        root=source_root,
        relative_path=relative_path,
        pair=pair,
        declared_from=shard.time_from,
        declared_to=shard.time_to,
        rows=rows,
        file_sha256=str(scan["file_sha256"]),
    )
    fetch_script = Path(str(receipt.get("fetch_script_path") or ""))
    if fetch_script.resolve(strict=True) != expected_fetch_script:
        raise HistoricalM5ManifestError("M5 receipt fetch script path is not canonical")
    if receipt.get("fetch_script_sha256") != expected_fetch_sha256:
        raise HistoricalM5ManifestError("M5 receipt fetch script digest drifted")

    body: dict[str, Any] = {
        "pair": pair,
        "shard_id": shard.shard_id,
        "from_utc": shard.time_from.isoformat(),
        "to_utc": shard.time_to.isoformat(),
        "run_id": run_id,
        "relative_path": relative_path,
        "summary_relative_path": _s5._relative_inside(source_root, summary_path),
        "summary_sha256": summary_sha256,
        "file_size_bytes": scan["file_size_bytes"],
        "file_sha256": scan["file_sha256"],
        "row_count": rows,
        "first_observed_utc": scan["first_observed_utc"],
        "last_observed_utc": scan["last_observed_utc"],
        "grid_slot_count": shard.grid_slot_count,
        "no_tick_slot_count": shard.grid_slot_count - rows,
        "observed_slot_coverage": scan["observed_slot_coverage"],
        "leading_gap_seconds": scan["leading_gap_seconds"],
        "trailing_gap_seconds": scan["trailing_gap_seconds"],
        "request_count": requests,
        "window_count": windows,
        "max_candles_per_request": max_candles_per_request,
        "acquisition_receipt_sequence": receipt["sequence"],
        "acquisition_receipt_sha256": receipt_sha,
        "acquisition_receipt_proved": True,
        "strict_m5_grid_proved": True,
        "strictly_increasing_unique_proved": True,
        "bid_ask_ohlc_proved": True,
        "complete_rows_only": True,
        "coverage_policy": COVERAGE_POLICY,
    }
    return {**body, "source_sha256": _s5._canonical_sha(body)}, relative_path, str(receipt_sha)


def _validate_m5_receipt_for_candidate(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    relative_path: str,
    pair: str,
    declared_from: datetime,
    declared_to: datetime,
    rows: int,
    file_sha256: str,
) -> None:
    """Apply the S5 receipt binding semantics to the explicit M5 scope."""

    if receipt.get("output_root") != str(root):
        raise HistoricalM5ManifestError("M5 truth receipt output root mismatch")
    if receipt.get("pair") != pair:
        raise HistoricalM5ManifestError("M5 truth receipt pair mismatch")
    if (
        receipt.get("granularity") != GRANULARITY
        or receipt.get("price_component") != PRICE_COMPONENT
    ):
        raise HistoricalM5ManifestError("M5 truth receipt market-data scope mismatch")
    if receipt.get("rows") != rows or receipt.get("candle_sha256") != file_sha256:
        raise HistoricalM5ManifestError("M5 truth receipt file evidence mismatch")
    window = receipt.get("window")
    if not isinstance(window, Mapping) or set(window) != {"from_utc", "to_utc"}:
        raise HistoricalM5ManifestError("M5 truth receipt window schema invalid")
    if _s5._parse_aware_utc(window["from_utc"]) != declared_from:
        raise HistoricalM5ManifestError("M5 truth receipt start mismatch")
    if _s5._parse_aware_utc(window["to_utc"]) != declared_to:
        raise HistoricalM5ManifestError("M5 truth receipt end mismatch")
    if _s5._parse_aware_utc(receipt.get("recorded_at_utc")) < declared_to:
        raise HistoricalM5ManifestError(
            "M5 truth receipt predates requested window maturity"
        )
    candle_path = Path(str(receipt.get("candle_path") or ""))
    if _s5._relative_inside(root, candle_path) != relative_path:
        raise HistoricalM5ManifestError("M5 truth receipt path mismatch")


def _scan_m5_cache(
    path: Path,
    *,
    pair: str,
    time_from: datetime,
    time_to: datetime,
) -> dict[str, Any]:
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None
    row_count = 0
    try:
        with _s5._stable_regular_binary(path) as (raw_handle, opened_stat):
            file_sha = _s5._sha256_handle(raw_handle)
            raw_handle.seek(0)
            with gzip.GzipFile(fileobj=raw_handle, mode="rb") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    row = _s5._strict_json_loads(raw, label=f"{path}:{line_number}")
                    timestamp = _validate_m5_row(row, pair=pair)
                    if not time_from <= timestamp < time_to:
                        raise HistoricalM5ManifestError(
                            "M5 cache row lies outside its exact half-open shard"
                        )
                    if previous is not None:
                        if timestamp <= previous:
                            raise HistoricalM5ManifestError(
                                "M5 cache timestamps are not strictly increasing and unique"
                            )
                        if int((timestamp - previous).total_seconds()) % 300 != 0:
                            raise HistoricalM5ManifestError(
                                "M5 cache timestamp gap is not a whole M5 cadence"
                            )
                    previous = timestamp
                    first = timestamp if first is None else first
                    last = timestamp
                    row_count += 1
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise HistoricalM5ManifestError("M5 cache gzip stream is invalid") from exc
    if row_count <= 0 or first is None or last is None:
        raise HistoricalM5ManifestError("M5 cache file is empty")
    grid_slot_count = int((time_to - time_from).total_seconds() // 300)
    leading_gap_seconds = int((first - time_from).total_seconds())
    trailing_gap_seconds = max(
        0,
        int((time_to - (last + timedelta(seconds=300))).total_seconds()),
    )
    return {
        "row_count": row_count,
        "first_observed_utc": first.isoformat(),
        "last_observed_utc": last.isoformat(),
        "file_size_bytes": opened_stat.st_size,
        "file_sha256": file_sha,
        "observed_slot_coverage": row_count / grid_slot_count,
        "leading_gap_seconds": leading_gap_seconds,
        "trailing_gap_seconds": trailing_gap_seconds,
    }


def _validate_m5_row(value: Any, *, pair: str) -> datetime:
    if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache row schema is invalid")
    if value.get("pair") != pair or value.get("granularity") != GRANULARITY:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache provenance mismatch")
    if value.get("price") != PRICE_COMPONENT or value.get("complete") is not True:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache is not complete bid/ask")
    volume = value.get("volume")
    if volume.__class__ is not int or volume < 0:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache volume is invalid")
    timestamp = _s5._parse_oanda_utc(value.get("time"))
    if _floor_m5(timestamp) != timestamp:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache timestamp is off-grid")
    prices: dict[str, float] = {}
    for prefix in ("bid", "ask"):
        block = value.get(prefix)
        if not isinstance(block, Mapping) or set(block) != _PRICE_KEYS:
            raise HistoricalM5ManifestError(f"{pair}: M5 cache {prefix} schema is invalid")
        parsed = {
            key: _positive_finite(block.get(key), pair=pair) for key in _PRICE_KEYS
        }
        if not (
            parsed["l"]
            <= min(parsed["o"], parsed["c"])
            <= max(parsed["o"], parsed["c"])
            <= parsed["h"]
        ):
            raise HistoricalM5ManifestError(f"{pair}: M5 cache {prefix} OHLC is invalid")
        for key, number in parsed.items():
            prices[f"{prefix}_{key}"] = number
    if (
        prices["bid_o"] > prices["ask_o"]
        or prices["bid_h"] > prices["ask_h"]
        or prices["bid_l"] > prices["ask_l"]
        or prices["bid_c"] > prices["ask_c"]
    ):
        raise HistoricalM5ManifestError(f"{pair}: M5 executable spread is crossed")
    return timestamp


def _positive_finite(value: Any, *, pair: str) -> float:
    if value.__class__ not in {int, float}:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache price type is invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise HistoricalM5ManifestError(f"{pair}: M5 cache price is invalid")
    return number


def _expected_request_windows(
    time_from: datetime,
    time_to: datetime,
    *,
    max_candles_per_request: int,
) -> int:
    # Mirrors oanda_history_fetch._iter_time_chunks.  The one-candle cushion is
    # an API page-boundary engineering constant, not a trading parameter.
    chunk_seconds = 300 * max(1, max_candles_per_request - 1)
    total_seconds = int((time_to - time_from).total_seconds())
    return (total_seconds + chunk_seconds - 1) // chunk_seconds


def _floor_m5(value: datetime) -> datetime:
    utc = _s5._aware_utc(value)
    return utc.replace(
        minute=utc.minute - (utc.minute % 5),
        second=0,
        microsecond=0,
    )


def _safe_relative_path(value: Any, *, expected_parts: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == expected_parts
    )
