"""Strict local S5 bid/ask cache admission for historical diagnostics.

This module deliberately stops at the market-data boundary.  It discovers
only error-free, published ``oanda_history_fetch.py`` outputs, freezes one
source per pair with an outcome-blind policy, verifies every cached row, and
loads only an inward-aligned requested interval.  Historical rows are never
forward proof and this module has no broker client or order surface.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing
import os
import re
import stat
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


MANIFEST_CONTRACT = "QR_FAST_BOT_HISTORICAL_S5_CACHE_MANIFEST_V1"
SLICE_CONTRACT = "QR_FAST_BOT_HISTORICAL_S5_CACHE_SLICE_V1"
SELECTION_POLICY = (
    "SUMMARY_PUBLISHED_ERROR_FREE_WIDEST_DECLARED_WINDOW_"
    "THEN_EARLIEST_FROM_THEN_RELATIVE_PATH_V1"
)
EXPLICIT_RUN_SELECTION_POLICY = (
    "EXPLICIT_ALLOWED_RUN_IDS_EXACT_SET_THEN_" + SELECTION_POLICY
)
EXPLICIT_RUN_SCOPE_POLICY = "EXPLICIT_ALLOWED_RUN_IDS_EXACT_SET_V1"
COVERAGE_POLICY = "SUMMARY_DECLARED_INTERVAL_PLUS_STRICT_FILE_SCAN_V1"
TRUTH_RECEIPT_SCHEMA = "QR_OANDA_TRUTH_ACQUISITION_RECEIPT_V1"
TRUTH_RECEIPT_FILE = "truth_acquisition_receipts.jsonl"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
_OANDA_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)
_CACHE_NAME_RE = re.compile(
    r"^(?P<pair>[A-Z]{3}_[A-Z]{3})_S5_BA_"
    r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)"
    r"\.jsonl(?P<gzip>\.gz)?$"
)
_ROW_KEYS = frozenset(
    {"ask", "bid", "complete", "granularity", "pair", "price", "time", "volume"}
)
_PRICE_KEYS = frozenset({"o", "h", "l", "c"})
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "sequence",
        "recorded_at_utc",
        "output_root",
        "candle_path",
        "candle_sha256",
        "pair",
        "granularity",
        "price_component",
        "window",
        "rows",
        "fetch_script_path",
        "fetch_script_sha256",
        "previous_receipt_sha256",
        "receipt_sha256",
    }
)
MAX_MANIFEST_SCAN_WORKERS = 6
MIN_PARALLEL_MANIFEST_SCAN_TASKS = 4


class HistoricalS5CacheError(ValueError):
    """Raised when local historical truth cannot be admitted exactly."""


@dataclass(frozen=True, slots=True)
class HistoricalS5SliceRequest:
    """One historical-only interval request for the pair-batched loader."""

    pair: str
    time_from: datetime
    time_to: datetime


@dataclass(frozen=True, slots=True)
class HistoricalS5Slice:
    """One exact, historical-only interval loaded from a sealed manifest."""

    pair: str
    requested_from_utc: datetime
    requested_to_utc: datetime
    aligned_from_utc: datetime
    aligned_to_utc: datetime
    candles: tuple[S5BidAskCandle, ...]
    source_relative_path: str
    source_file_sha256: str
    source_manifest_sha256: str
    acquisition_receipt_proved: bool

    @property
    def grid_slot_count(self) -> int:
        return int((self.aligned_to_utc - self.aligned_from_utc).total_seconds() // 5)

    @property
    def no_tick_slot_count(self) -> int:
        return self.grid_slot_count - len(self.candles)

    def receipt(self) -> dict[str, Any]:
        """Return a sealed diagnostic receipt without serializing all candles."""

        body: dict[str, Any] = {
            "contract": SLICE_CONTRACT,
            "schema_version": 1,
            "pair": self.pair,
            "requested_from_utc": self.requested_from_utc.isoformat(),
            "requested_to_utc": self.requested_to_utc.isoformat(),
            "aligned_from_utc": self.aligned_from_utc.isoformat(),
            "aligned_to_utc": self.aligned_to_utc.isoformat(),
            "source_relative_path": self.source_relative_path,
            "source_file_sha256": self.source_file_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "coverage_policy": COVERAGE_POLICY,
            "grid_slot_count": self.grid_slot_count,
            "candle_count": len(self.candles),
            "no_tick_slot_count": self.no_tick_slot_count,
            "truth_path_sha256": _truth_path_sha256(self.candles),
            "exact_interval_membership_proved": True,
            "acquisition_receipt_proved": self.acquisition_receipt_proved,
            "historical_only": True,
            "diagnostic_only": True,
            "forward_proof_eligible": False,
            "promotion_allowed": False,
            "order_authority": "NONE",
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        return {**body, "slice_sha256": _canonical_sha(body)}


@dataclass(frozen=True, slots=True)
class _PreparedSliceRequest:
    index: int
    pair: str
    requested_from_utc: datetime
    requested_to_utc: datetime
    aligned_from_utc: datetime
    aligned_to_utc: datetime
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Candidate:
    pair: str
    relative_path: str
    summary_relative_path: str
    declared_from_utc: datetime
    declared_to_utc: datetime
    declared_rows: int
    usable_rows: int
    quarantined_boundary_rows: int
    file_size_bytes: int
    file_sha256: str
    observed_first_utc: datetime
    observed_last_utc: datetime
    source_summary_sha256: str
    acquisition_receipt_sha256: str | None

    @property
    def duration_seconds(self) -> float:
        return (self.declared_to_utc - self.declared_from_utc).total_seconds()

    def manifest_row(
        self,
        *,
        candidate_count_for_pair: int,
        selection_policy: str = SELECTION_POLICY,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "pair": self.pair,
            "relative_path": self.relative_path,
            "summary_relative_path": self.summary_relative_path,
            "declared_from_utc": self.declared_from_utc.isoformat(),
            "declared_to_utc": self.declared_to_utc.isoformat(),
            "declared_rows": self.declared_rows,
            "usable_rows": self.usable_rows,
            "quarantined_boundary_rows": self.quarantined_boundary_rows,
            "observed_first_utc": self.observed_first_utc.isoformat(),
            "observed_last_utc": self.observed_last_utc.isoformat(),
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
            "source_summary_sha256": self.source_summary_sha256,
            "acquisition_receipt_sha256": self.acquisition_receipt_sha256,
            "acquisition_receipt_proved": self.acquisition_receipt_sha256 is not None,
            "candidate_count_for_pair": candidate_count_for_pair,
            "selection_policy": selection_policy,
            "coverage_policy": COVERAGE_POLICY,
            "historical_only": True,
            "diagnostic_only": True,
            "forward_proof_eligible": False,
        }
        return {**body, "source_sha256": _canonical_sha(body)}


@dataclass(frozen=True, slots=True)
class _CandidateScanJob:
    """One immutable, process-safe strict candidate scan request."""

    source_root: Path
    summary_path: Path
    summary_sha256: str
    task: dict[str, Any]
    receipt_by_path: dict[str, Mapping[str, Any]]


def _validate_allowed_run_ids(
    value: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HistoricalS5CacheError(
            "allowed historical run ids must be a sequence or None"
        )
    run_ids = tuple(value)
    if not run_ids:
        raise HistoricalS5CacheError(
            "allowed historical run ids must be non-empty when specified"
        )
    if any(not isinstance(run_id, str) for run_id in run_ids):
        raise HistoricalS5CacheError("allowed historical run id type is invalid")
    if len(run_ids) != len(set(run_ids)):
        raise HistoricalS5CacheError("allowed historical run ids must be unique")
    for run_id in run_ids:
        if _RUN_ID_RE.fullmatch(run_id) is None:
            raise HistoricalS5CacheError("allowed historical run id is unsafe")
        try:
            parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
        except ValueError as exc:
            raise HistoricalS5CacheError(
                "allowed historical run id clock is invalid"
            ) from exc
        if parsed.strftime("%Y%m%dT%H%M%SZ") != run_id:
            raise HistoricalS5CacheError("allowed historical run id is not canonical")
    return tuple(sorted(run_ids))


def _summary_paths_for_run_scope(
    source_root: Path,
    allowed_run_ids: tuple[str, ...] | None,
) -> tuple[Path, ...]:
    if allowed_run_ids is None:
        return tuple(sorted(source_root.glob("*/summary.json")))
    summary_paths: list[Path] = []
    for run_id in allowed_run_ids:
        run_path = source_root / run_id
        if run_path.is_symlink() or not run_path.is_dir():
            raise HistoricalS5CacheError(
                f"allowed historical run does not exist as a regular directory: {run_id}"
            )
        try:
            resolved_run = run_path.resolve(strict=True)
        except OSError as exc:
            raise HistoricalS5CacheError(
                f"allowed historical run cannot be resolved: {run_id}"
            ) from exc
        if resolved_run.parent != source_root or resolved_run.name != run_id:
            raise HistoricalS5CacheError(
                f"allowed historical run escapes its exact root scope: {run_id}"
            )
        summary_path = run_path / "summary.json"
        if summary_path.is_symlink() or not summary_path.is_file():
            raise HistoricalS5CacheError(
                f"allowed historical run has no regular summary: {run_id}"
            )
        summary_paths.append(summary_path)
    actual_run_ids = tuple(sorted(path.parent.name for path in summary_paths))
    if actual_run_ids != allowed_run_ids:
        raise HistoricalS5CacheError(
            "allowed historical run ids and summary run ids are not an exact set"
        )
    return tuple(sorted(summary_paths))


def _resolved_manifest_scan_workers(
    requested: int | None,
    *,
    task_count: int,
) -> int:
    if requested is not None and (
        requested.__class__ is not int
        or requested < 1
        or requested > MAX_MANIFEST_SCAN_WORKERS
    ):
        raise HistoricalS5CacheError(
            f"manifest scan workers must be an integer from 1 to {MAX_MANIFEST_SCAN_WORKERS}"
        )
    if task_count < MIN_PARALLEL_MANIFEST_SCAN_TASKS:
        return 1
    available = max(1, int(os.cpu_count() or 1))
    # Preserve the historical serial behavior unless the caller explicitly opts
    # into process workers. Production CLI paths pass a bounded worker count.
    target = 1 if requested is None else requested
    return max(1, min(target, available, task_count))


def _scan_candidate_job(job: _CandidateScanJob) -> _Candidate:
    return _candidate_from_task(
        source_root=job.source_root,
        summary_path=job.summary_path,
        summary_sha256=job.summary_sha256,
        task=job.task,
        receipt_by_path=job.receipt_by_path,
    )


def _scan_candidate_jobs(
    jobs: Sequence[_CandidateScanJob],
    *,
    scan_workers: int | None,
) -> tuple[_Candidate, ...]:
    worker_count = _resolved_manifest_scan_workers(
        scan_workers,
        task_count=len(jobs),
    )
    if worker_count == 1:
        return tuple(_scan_candidate_job(job) for job in jobs)
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            return tuple(executor.map(_scan_candidate_job, jobs, chunksize=1))
    except HistoricalS5CacheError:
        raise
    except Exception as exc:
        raise HistoricalS5CacheError(
            "parallel historical manifest candidate scan failed closed"
        ) from exc


def build_historical_s5_manifest(
    root: Path,
    *,
    pairs: Sequence[str] = DEFAULT_TRADER_PAIRS,
    allowed_run_ids: Sequence[str] | None = None,
    scan_workers: int | None = None,
) -> dict[str, Any]:
    """Scan and seal one outcome-blind local source per requested pair.

    Selection uses only acquisition metadata: widest declared request window,
    then earliest declared start, then lexical relative path.  Prices and
    outcomes never influence which duplicate becomes canonical.
    """

    source_root = root.resolve(strict=True)
    if not source_root.is_dir():
        raise HistoricalS5CacheError("historical S5 root is not a directory")
    normalized_pairs = _validate_pairs(pairs)
    normalized_run_ids = _validate_allowed_run_ids(allowed_run_ids)
    summary_paths = _summary_paths_for_run_scope(source_root, normalized_run_ids)
    selection_policy = (
        SELECTION_POLICY
        if normalized_run_ids is None
        else EXPLICIT_RUN_SELECTION_POLICY
    )
    receipt_by_path = _load_truth_receipts(source_root)
    candidates: dict[str, list[_Candidate]] = {pair: [] for pair in normalized_pairs}
    admitted_paths: set[str] = set()
    scan_jobs: list[_CandidateScanJob] = []
    frozen_receipts = dict(receipt_by_path)

    for summary_path in summary_paths:
        if summary_path.is_symlink() or not summary_path.is_file():
            raise HistoricalS5CacheError("summary path must be a regular file")
        summary_bytes = _read_stable_bytes(summary_path)
        summary = _strict_json_loads(summary_bytes, label=str(summary_path))
        if not isinstance(summary, Mapping):
            raise HistoricalS5CacheError("history summary must be an object")
        if summary.get("dry_run") is not False:
            continue
        if summary.get("errors") != []:
            continue
        tasks = summary.get("tasks")
        if not isinstance(tasks, list):
            raise HistoricalS5CacheError("history summary tasks must be a list")
        summary_sha = _sha256_bytes(summary_bytes)
        for task in tasks:
            if not isinstance(task, Mapping):
                raise HistoricalS5CacheError("history summary task must be an object")
            if task.get("granularity") != "S5" or task.get("price") != "BA":
                continue
            pair = str(task.get("pair") or "")
            if pair not in candidates:
                continue
            scan_jobs.append(
                _CandidateScanJob(
                    source_root=source_root,
                    summary_path=summary_path,
                    summary_sha256=summary_sha,
                    task=dict(task),
                    receipt_by_path=frozen_receipts,
                )
            )
    for candidate in _scan_candidate_jobs(scan_jobs, scan_workers=scan_workers):
        if candidate.relative_path in admitted_paths:
            raise HistoricalS5CacheError(
                f"cache file admitted by multiple summaries: {candidate.relative_path}"
            )
        admitted_paths.add(candidate.relative_path)
        candidates[candidate.pair].append(candidate)

    selected_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    missing_pairs: list[str] = []
    for pair in normalized_pairs:
        options = candidates[pair]
        if not options:
            missing_pairs.append(pair)
            continue
        ordered = sorted(
            options,
            key=lambda item: (
                -item.duration_seconds,
                item.declared_from_utc,
                item.relative_path,
            ),
        )
        selected = ordered[0]
        selected_rows.append(
            selected.manifest_row(
                candidate_count_for_pair=len(ordered),
                selection_policy=selection_policy,
            )
        )
        if len(ordered) > 1:
            duplicate_rows.append(
                {
                    "pair": pair,
                    "selected_relative_path": selected.relative_path,
                    "candidate_relative_paths": [
                        item.relative_path for item in ordered
                    ],
                    "resolution_policy": selection_policy,
                    "outcome_data_used_for_selection": False,
                }
            )

    discovery_paths: Iterable[Path]
    if normalized_run_ids is None:
        discovery_paths = source_root.glob("*/*/*_S5_BA_*.jsonl*")
    else:
        discovery_paths = (
            path
            for run_id in normalized_run_ids
            for path in (source_root / run_id).glob("*/*_S5_BA_*.jsonl*")
        )
    discovered_files = {
        _relative_inside(source_root, path)
        for path in discovery_paths
        if path.is_file()
        and not path.name.endswith((".partial", ".tmp"))
        and _CACHE_NAME_RE.fullmatch(path.name)
    }
    unadmitted_files = sorted(discovered_files - admitted_paths)
    common_from: datetime | None = None
    common_to: datetime | None = None
    if selected_rows:
        common_from = max(
            _parse_aware_utc(row["declared_from_utc"]) for row in selected_rows
        )
        common_to = min(
            _parse_aware_utc(row["declared_to_utc"]) for row in selected_rows
        )
        if common_to <= common_from:
            common_from = None
            common_to = None

    body: dict[str, Any] = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": 1,
        "source_root": str(source_root),
        "selection_policy": selection_policy,
        "selection_is_outcome_blind": True,
        "coverage_policy": COVERAGE_POLICY,
        "expected_pairs": list(normalized_pairs),
        "selected_sources": selected_rows,
        "duplicate_candidates": duplicate_rows,
        "missing_pairs": missing_pairs,
        "unadmitted_files": unadmitted_files,
        "selected_pair_count": len(selected_rows),
        "expected_pair_count": len(normalized_pairs),
        "complete_pair_coverage": not missing_pairs,
        "common_declared_from_utc": (
            common_from.isoformat() if common_from is not None else None
        ),
        "common_declared_to_utc": (
            common_to.isoformat() if common_to is not None else None
        ),
        "all_selected_sources_acquisition_receipted": bool(selected_rows)
        and all(row["acquisition_receipt_proved"] is True for row in selected_rows),
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
    if normalized_run_ids is not None:
        body.update(
            {
                "run_scope_policy": EXPLICIT_RUN_SCOPE_POLICY,
                "allowed_run_ids": list(normalized_run_ids),
                "summary_run_ids": [path.parent.name for path in summary_paths],
                "summary_run_ids_exact_set_proved": True,
                "run_scope_is_outcome_blind": True,
            }
        )
    manifest = {**body, "manifest_sha256": _canonical_sha(body)}
    _validate_manifest(manifest)
    return manifest


def load_historical_s5_slice(
    manifest: Mapping[str, Any],
    *,
    pair: str,
    time_from: datetime,
    time_to: datetime,
) -> HistoricalS5Slice:
    """Load ``ceil(time_from) <= S5 open < floor(time_to)`` exactly.

    The inward alignment matches the forward adapter and prevents a partial
    candle before the decision or after the frozen horizon from entering a
    historical diagnostic.
    """

    return load_historical_s5_slices(
        manifest,
        requests=(
            HistoricalS5SliceRequest(
                pair=str(pair),
                time_from=time_from,
                time_to=time_to,
            ),
        ),
    )[0]


def load_historical_s5_slices(
    manifest: Mapping[str, Any],
    *,
    requests: Sequence[HistoricalS5SliceRequest],
) -> tuple[HistoricalS5Slice, ...]:
    """Load many intervals with one hash and one decompression pass per pair.

    Requests are returned in input order.  Grouping and interval boundaries
    depend only on the caller's frozen request clocks; neither source selection
    nor streaming is influenced by future prices or outcomes.
    """

    _validate_manifest(manifest)
    frozen_requests = tuple(requests)
    if not frozen_requests:
        return ()
    source_by_pair = {
        str(row["pair"]): row
        for row in manifest["selected_sources"]
        if isinstance(row, Mapping)
    }
    prepared: list[_PreparedSliceRequest] = []
    grouped: dict[str, list[_PreparedSliceRequest]] = {}
    for index, request in enumerate(frozen_requests):
        if not isinstance(request, HistoricalS5SliceRequest):
            raise HistoricalS5CacheError(
                "historical S5 batch request has an invalid type"
            )
        pair_name = str(request.pair)
        source = source_by_pair.get(pair_name)
        if source is None:
            raise HistoricalS5CacheError(
                f"no admitted historical S5 source for {pair_name}"
            )
        requested_from = _aware_utc(request.time_from)
        requested_to = _aware_utc(request.time_to)
        if requested_to <= requested_from:
            raise HistoricalS5CacheError("historical S5 interval must be positive")
        aligned_from = _ceil_s5(requested_from)
        aligned_to = _floor_s5(requested_to)
        if aligned_to <= aligned_from:
            raise HistoricalS5CacheError(
                "historical interval has no complete S5 candle"
            )
        declared_from = _parse_aware_utc(source["declared_from_utc"])
        declared_to = _parse_aware_utc(source["declared_to_utc"])
        if aligned_from < _ceil_s5(declared_from) or aligned_to > _floor_s5(
            declared_to
        ):
            raise HistoricalS5CacheError(
                "requested interval exceeds declared cache coverage"
            )
        item = _PreparedSliceRequest(
            index=index,
            pair=pair_name,
            requested_from_utc=requested_from,
            requested_to_utc=requested_to,
            aligned_from_utc=aligned_from,
            aligned_to_utc=aligned_to,
            source=source,
        )
        prepared.append(item)
        grouped.setdefault(pair_name, []).append(item)

    candles_by_index: dict[int, tuple[S5BidAskCandle, ...]] = {}
    source_root = Path(str(manifest["source_root"]))
    for pair_name, pair_requests in grouped.items():
        candles_by_index.update(
            _load_historical_s5_pair_batch(
                source_root=source_root,
                pair=pair_name,
                requests=pair_requests,
            )
        )

    slices: list[HistoricalS5Slice] = []
    for item in prepared:
        candles = candles_by_index[item.index]
        expected_slots = int(
            (item.aligned_to_utc - item.aligned_from_utc).total_seconds() // 5
        )
        if len(candles) > expected_slots:
            raise HistoricalS5CacheError(
                "historical S5 candle count exceeds exact grid"
            )
        source = item.source
        slices.append(
            HistoricalS5Slice(
                pair=item.pair,
                requested_from_utc=item.requested_from_utc,
                requested_to_utc=item.requested_to_utc,
                aligned_from_utc=item.aligned_from_utc,
                aligned_to_utc=item.aligned_to_utc,
                candles=candles,
                source_relative_path=str(source["relative_path"]),
                source_file_sha256=str(source["file_sha256"]),
                source_manifest_sha256=str(manifest["manifest_sha256"]),
                acquisition_receipt_proved=(
                    source["acquisition_receipt_proved"] is True
                ),
            )
        )
    return tuple(slices)


def _load_historical_s5_pair_batch(
    *,
    source_root: Path,
    pair: str,
    requests: Sequence[_PreparedSliceRequest],
) -> dict[int, tuple[S5BidAskCandle, ...]]:
    source = requests[0].source
    if any(item.source is not source for item in requests):
        raise HistoricalS5CacheError("historical S5 batch source identity diverged")
    source_path = _resolve_manifest_relative_path(
        source_root,
        str(source["relative_path"]),
    )
    expected_size = source.get("file_size_bytes")
    if expected_size.__class__ is not int or expected_size <= 0:
        raise HistoricalS5CacheError("historical S5 source size metadata is invalid")
    expected_sha = str(source.get("file_sha256") or "")
    pending = sorted(
        requests,
        key=lambda item: (item.aligned_from_utc, item.aligned_to_utc, item.index),
    )
    by_index: dict[int, list[S5BidAskCandle]] = {item.index: [] for item in requests}
    next_pending = 0
    active: dict[int, _PreparedSliceRequest] = {}
    maximum_to = max(item.aligned_to_utc for item in requests)

    with _stable_regular_binary(source_path) as (raw, opened_stat):
        if opened_stat.st_size != expected_size:
            raise HistoricalS5CacheError(
                "historical S5 source size changed after manifest"
            )
        if _sha256_handle(raw) != expected_sha:
            raise HistoricalS5CacheError(
                "historical S5 source hash changed after manifest"
            )
        previous: datetime | None = None
        with _cache_binary_stream(raw, source_path) as stream:
            for line_number, payload in enumerate(stream, start=1):
                row = _strict_json_loads(
                    payload,
                    label=f"{source_path}:{line_number}",
                )
                timestamp = _row_timestamp_only(row, pair=pair)
                if previous is not None and timestamp <= previous:
                    raise HistoricalS5CacheError(
                        "historical S5 timestamps are not strictly increasing"
                    )
                previous = timestamp
                if timestamp >= maximum_to:
                    break
                while (
                    next_pending < len(pending)
                    and pending[next_pending].aligned_from_utc <= timestamp
                ):
                    item = pending[next_pending]
                    if timestamp < item.aligned_to_utc:
                        active[item.index] = item
                    next_pending += 1
                expired = [
                    index
                    for index, item in active.items()
                    if item.aligned_to_utc <= timestamp
                ]
                for index in expired:
                    del active[index]
                if not active:
                    continue
                candle = _parse_cache_candle(row, pair=pair)
                for index in active:
                    by_index[index].append(candle)

    return {index: tuple(candles) for index, candles in by_index.items()}


def write_historical_s5_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically persist a verified read-only manifest."""

    _validate_manifest(manifest)
    destination = path.resolve()
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


def load_historical_s5_manifest(path: Path) -> dict[str, Any]:
    """Load a sealed manifest without rescanning the multi-million-row cache."""

    if path.is_symlink() or not path.is_file():
        raise HistoricalS5CacheError("historical S5 manifest path is invalid")
    value = _strict_json_loads(_read_stable_bytes(path), label=str(path))
    if not isinstance(value, dict):
        raise HistoricalS5CacheError("historical S5 manifest is not an object")
    _validate_manifest(value)
    return value


def _candidate_from_task(
    *,
    source_root: Path,
    summary_path: Path,
    summary_sha256: str,
    task: Mapping[str, Any],
    receipt_by_path: Mapping[str, Mapping[str, Any]],
) -> _Candidate:
    if task.get("dry_run") is not False:
        raise HistoricalS5CacheError("S5 cache task cannot be a dry run")
    if task.get("published") is not True or task.get("errors") != []:
        raise HistoricalS5CacheError("S5 cache task is not error-free and published")
    if task.get("partial_path") is not None:
        raise HistoricalS5CacheError("published S5 cache task retains a partial path")
    pair = str(task.get("pair") or "")
    path_text = task.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise HistoricalS5CacheError("S5 cache task path is invalid")
    task_path = Path(path_text)
    match = _CACHE_NAME_RE.fullmatch(task_path.name)
    if match is None or match.group("pair") != pair:
        raise HistoricalS5CacheError("S5 cache filename does not match its pair")
    if task.get("compressed") is not (match.group("gzip") is not None):
        raise HistoricalS5CacheError("S5 cache compression metadata mismatch")
    expected_suffix = (summary_path.parent.name, pair, task_path.name)
    if tuple(task_path.parts[-3:]) != expected_suffix:
        raise HistoricalS5CacheError("S5 cache path does not match its summary run")
    source_path = summary_path.parent / pair / task_path.name
    if source_path.is_symlink() or not source_path.is_file():
        raise HistoricalS5CacheError("S5 cache source must be a regular file")
    relative_path = _relative_inside(source_root, source_path)
    declared_from = _parse_aware_utc(task.get("from"))
    declared_to = _parse_aware_utc(task.get("to"))
    if declared_to <= declared_from:
        raise HistoricalS5CacheError("S5 cache declared interval is invalid")
    if match.group("start") != _filename_stamp(declared_from):
        raise HistoricalS5CacheError("S5 cache filename start does not match summary")
    if match.group("end") != _filename_stamp(declared_to):
        raise HistoricalS5CacheError("S5 cache filename end does not match summary")
    rows = task.get("rows")
    if rows.__class__ is not int or rows <= 0:
        raise HistoricalS5CacheError(
            "S5 cache declared rows must be a positive integer"
        )
    scan = _scan_cache_file(
        source_path,
        pair=pair,
        declared_from=declared_from,
        declared_to=declared_to,
    )
    if scan["rows"] != rows:
        raise HistoricalS5CacheError("S5 cache row count does not match summary")
    receipt = receipt_by_path.get(relative_path)
    receipt_sha: str | None = None
    if receipt is not None:
        _validate_receipt_for_candidate(
            receipt,
            root=source_root,
            relative_path=relative_path,
            pair=pair,
            declared_from=declared_from,
            declared_to=declared_to,
            rows=rows,
            file_sha256=str(scan["file_sha256"]),
        )
        receipt_sha = str(receipt["receipt_sha256"])
    return _Candidate(
        pair=pair,
        relative_path=relative_path,
        summary_relative_path=_relative_inside(source_root, summary_path),
        declared_from_utc=declared_from,
        declared_to_utc=declared_to,
        declared_rows=rows,
        usable_rows=int(scan["usable_rows"]),
        quarantined_boundary_rows=int(scan["quarantined_boundary_rows"]),
        file_size_bytes=int(scan["file_size_bytes"]),
        file_sha256=str(scan["file_sha256"]),
        observed_first_utc=scan["observed_first_utc"],
        observed_last_utc=scan["observed_last_utc"],
        source_summary_sha256=summary_sha256,
        acquisition_receipt_sha256=receipt_sha,
    )


def _scan_cache_file(
    path: Path,
    *,
    pair: str,
    declared_from: datetime,
    declared_to: datetime,
) -> dict[str, Any]:
    first: datetime | None = None
    last_usable: datetime | None = None
    previous: datetime | None = None
    rows = 0
    usable_rows = 0
    quarantined_boundary_rows = 0
    safe_start = _ceil_s5(declared_from)
    safe_end = _floor_s5(declared_to)
    outer_start = _floor_s5(declared_from)
    with _stable_regular_binary(path) as (raw_handle, opened_stat):
        file_sha256 = _sha256_handle(raw_handle)
        with _cache_binary_stream(raw_handle, path) as handle:
            for line_number, raw in enumerate(handle, start=1):
                row = _strict_json_loads(
                    raw,
                    label=f"{path}:{line_number}",
                )
                candle = _parse_cache_candle(row, pair=pair)
                timestamp = candle.timestamp_utc
                if previous is not None and timestamp <= previous:
                    raise HistoricalS5CacheError(
                        "S5 cache timestamps must be unique and strictly increasing"
                    )
                previous = timestamp
                rows += 1
                if safe_start <= timestamp < safe_end:
                    if first is None:
                        first = timestamp
                    last_usable = timestamp
                    usable_rows += 1
                elif outer_start <= timestamp < safe_start or timestamp == safe_end:
                    # The legacy bulk fetcher did not align its request inward.
                    # OANDA may therefore return one containing candle at either
                    # non-grid boundary.  Verify it, count it, and quarantine it;
                    # the interval loader below can never return it.
                    quarantined_boundary_rows += 1
                else:
                    raise HistoricalS5CacheError(
                        "S5 cache row lies outside even its bounded boundary quarantine: "
                        f"path={path} timestamp={timestamp.isoformat()} "
                        f"safe=[{safe_start.isoformat()},{safe_end.isoformat()})"
                    )
    if quarantined_boundary_rows > 2:
        raise HistoricalS5CacheError("S5 cache has too many boundary rows")
    if first is None or last_usable is None or previous is None or usable_rows <= 0:
        raise HistoricalS5CacheError("S5 cache file is empty")
    return {
        "rows": rows,
        "usable_rows": usable_rows,
        "quarantined_boundary_rows": quarantined_boundary_rows,
        "observed_first_utc": first,
        "observed_last_utc": last_usable,
        "file_size_bytes": opened_stat.st_size,
        "file_sha256": file_sha256,
    }


def _parse_cache_candle(value: Any, *, pair: str) -> S5BidAskCandle:
    if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
        raise HistoricalS5CacheError(f"{pair}: S5 cache row schema is invalid")
    if value.get("pair") != pair or value.get("granularity") != "S5":
        raise HistoricalS5CacheError(f"{pair}: S5 cache provenance mismatch")
    if value.get("price") != "BA" or value.get("complete") is not True:
        raise HistoricalS5CacheError(f"{pair}: S5 cache is not complete bid/ask")
    volume = value.get("volume")
    if volume.__class__ is not int or volume < 0:
        raise HistoricalS5CacheError(f"{pair}: S5 cache volume is invalid")
    timestamp = _parse_oanda_utc(value.get("time"))
    if _floor_s5(timestamp) != timestamp:
        raise HistoricalS5CacheError(f"{pair}: S5 cache timestamp is off-grid")
    prices: dict[str, float] = {}
    for prefix in ("bid", "ask"):
        block = value.get(prefix)
        if not isinstance(block, Mapping) or set(block) != _PRICE_KEYS:
            raise HistoricalS5CacheError(f"{pair}: S5 cache {prefix} schema is invalid")
        parsed = {
            key: _positive_finite(block.get(key), pair=pair) for key in _PRICE_KEYS
        }
        if not (
            parsed["l"]
            <= min(parsed["o"], parsed["c"])
            <= max(parsed["o"], parsed["c"])
            <= parsed["h"]
        ):
            raise HistoricalS5CacheError(f"{pair}: S5 cache {prefix} OHLC is invalid")
        for key, number in parsed.items():
            prices[f"{prefix}_{key}"] = number
    if prices["bid_o"] > prices["ask_o"] or prices["bid_c"] > prices["ask_c"]:
        raise HistoricalS5CacheError(f"{pair}: S5 cache executable spread is crossed")
    return S5BidAskCandle(timestamp_utc=timestamp, **prices)


def _row_timestamp_only(value: Any, *, pair: str) -> datetime:
    if not isinstance(value, Mapping):
        raise HistoricalS5CacheError(f"{pair}: S5 cache row is not an object")
    if value.get("pair") != pair or value.get("granularity") != "S5":
        raise HistoricalS5CacheError(f"{pair}: S5 cache provenance mismatch")
    timestamp = _parse_oanda_utc(value.get("time"))
    if _floor_s5(timestamp) != timestamp:
        raise HistoricalS5CacheError(f"{pair}: S5 cache timestamp is off-grid")
    return timestamp


def _load_truth_receipts(root: Path) -> dict[str, Mapping[str, Any]]:
    path = root / TRUTH_RECEIPT_FILE
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise HistoricalS5CacheError("truth acquisition receipt path is invalid")
    rows: dict[str, Mapping[str, Any]] = {}
    previous: str | None = None
    for line_number, raw in enumerate(_read_stable_bytes(path).splitlines(), start=1):
        if not raw.strip():
            continue
        item = _strict_json_loads(raw, label=f"{path}:{line_number}")
        if not isinstance(item, Mapping) or set(item) != _RECEIPT_KEYS:
            raise HistoricalS5CacheError("truth acquisition receipt schema invalid")
        if item.get("schema_version") != TRUTH_RECEIPT_SCHEMA:
            raise HistoricalS5CacheError("truth acquisition receipt version invalid")
        if item.get("output_root") != str(root):
            raise HistoricalS5CacheError(
                "truth acquisition receipt output root invalid"
            )
        fetch_script_path = Path(str(item.get("fetch_script_path") or ""))
        if (
            not fetch_script_path.is_absolute()
            or fetch_script_path.name != "oanda_history_fetch.py"
            or not _SHA256_RE.fullmatch(str(item.get("fetch_script_sha256") or ""))
        ):
            raise HistoricalS5CacheError(
                "truth acquisition receipt fetch provenance invalid"
            )
        if (
            _sha256_bytes(_read_stable_bytes(fetch_script_path))
            != item["fetch_script_sha256"]
        ):
            raise HistoricalS5CacheError(
                "truth acquisition receipt fetch provenance drifted"
            )
        recorded_at = _parse_aware_utc(item.get("recorded_at_utc"))
        if recorded_at > datetime.now(timezone.utc):
            raise HistoricalS5CacheError(
                "truth acquisition receipt recorded clock is in the future"
            )
        if item.get("sequence") != len(rows) + 1:
            raise HistoricalS5CacheError("truth acquisition receipt sequence invalid")
        if item.get("previous_receipt_sha256") != previous:
            raise HistoricalS5CacheError("truth acquisition receipt chain is broken")
        body = {key: value for key, value in item.items() if key != "receipt_sha256"}
        receipt_sha = _canonical_sha(body)
        if item.get("receipt_sha256") != receipt_sha:
            raise HistoricalS5CacheError("truth acquisition receipt digest mismatch")
        candle_path = Path(str(item.get("candle_path") or ""))
        if not candle_path.is_absolute():
            raise HistoricalS5CacheError(
                "truth acquisition receipt candle path is not absolute"
            )
        try:
            relative = _relative_inside(root, candle_path)
        except (OSError, ValueError) as exc:
            raise HistoricalS5CacheError(
                "truth acquisition receipt candle path escapes cache root"
            ) from exc
        if relative in rows:
            raise HistoricalS5CacheError("duplicate truth receipt for one cache file")
        rows[relative] = item
        previous = receipt_sha
    return rows


def _validate_receipt_for_candidate(
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
    if receipt.get("output_root") != str(root):
        raise HistoricalS5CacheError("truth receipt output root mismatch")
    if receipt.get("pair") != pair:
        raise HistoricalS5CacheError("truth receipt pair mismatch")
    if receipt.get("granularity") != "S5" or receipt.get("price_component") != "BA":
        raise HistoricalS5CacheError("truth receipt market-data scope mismatch")
    if receipt.get("rows") != rows or receipt.get("candle_sha256") != file_sha256:
        raise HistoricalS5CacheError("truth receipt file evidence mismatch")
    window = receipt.get("window")
    if not isinstance(window, Mapping) or set(window) != {"from_utc", "to_utc"}:
        raise HistoricalS5CacheError("truth receipt window schema invalid")
    if _parse_aware_utc(window["from_utc"]) != declared_from:
        raise HistoricalS5CacheError("truth receipt start mismatch")
    if _parse_aware_utc(window["to_utc"]) != declared_to:
        raise HistoricalS5CacheError("truth receipt end mismatch")
    if _parse_aware_utc(receipt.get("recorded_at_utc")) < declared_to:
        raise HistoricalS5CacheError("truth receipt predates requested window maturity")
    if _relative_inside(root, Path(str(receipt["candle_path"]))) != relative_path:
        raise HistoricalS5CacheError("truth receipt path mismatch")


def _validate_manifest(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise HistoricalS5CacheError("historical S5 manifest is not an object")
    if value.get("contract") != MANIFEST_CONTRACT or value.get("schema_version") != 1:
        raise HistoricalS5CacheError("historical S5 manifest contract mismatch")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != _canonical_sha(body):
        raise HistoricalS5CacheError("historical S5 manifest digest mismatch")
    selection_policy = value.get("selection_policy")
    run_scope_fields = (
        "run_scope_policy",
        "allowed_run_ids",
        "summary_run_ids",
        "summary_run_ids_exact_set_proved",
        "run_scope_is_outcome_blind",
    )
    explicit_run_ids: tuple[str, ...] | None = None
    if selection_policy == SELECTION_POLICY:
        if any(field in value for field in run_scope_fields):
            raise HistoricalS5CacheError(
                "legacy historical selection cannot claim an explicit run scope"
            )
    elif selection_policy == EXPLICIT_RUN_SELECTION_POLICY:
        allowed = value.get("allowed_run_ids")
        summaries = value.get("summary_run_ids")
        if not isinstance(allowed, list) or not isinstance(summaries, list):
            raise HistoricalS5CacheError(
                "explicit historical run scope lists are invalid"
            )
        explicit_run_ids = _validate_allowed_run_ids(allowed)
        if (
            explicit_run_ids is None
            or tuple(allowed) != explicit_run_ids
            or tuple(summaries) != explicit_run_ids
            or value.get("run_scope_policy") != EXPLICIT_RUN_SCOPE_POLICY
            or value.get("summary_run_ids_exact_set_proved") is not True
            or value.get("run_scope_is_outcome_blind") is not True
        ):
            raise HistoricalS5CacheError(
                "explicit historical run scope binding is invalid"
            )
    else:
        raise HistoricalS5CacheError("historical S5 selection policy mismatch")
    if value.get("selection_is_outcome_blind") is not True:
        raise HistoricalS5CacheError("historical S5 selection is not outcome blind")
    if value.get("historical_only") is not True:
        raise HistoricalS5CacheError("historical S5 manifest is not historical-only")
    for forbidden_true in (
        "forward_proof_eligible",
        "automatic_promotion_allowed",
        "promotion_allowed",
        "live_permission",
        "broker_mutation_allowed",
    ):
        if value.get(forbidden_true) is not False:
            raise HistoricalS5CacheError(
                f"historical S5 manifest unsafe flag: {forbidden_true}"
            )
    if value.get("order_authority") != "NONE" or value.get("shadow_only") is not True:
        raise HistoricalS5CacheError("historical S5 authority boundary mismatch")
    sources = value.get("selected_sources")
    if not isinstance(sources, list):
        raise HistoricalS5CacheError("historical S5 sources must be a list")
    seen: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping):
            raise HistoricalS5CacheError("historical S5 source must be an object")
        pair = str(row.get("pair") or "")
        if pair in seen or not _PAIR_RE.fullmatch(pair):
            raise HistoricalS5CacheError(
                "historical S5 source pair is invalid or duplicate"
            )
        seen.add(pair)
        if not _SHA256_RE.fullmatch(str(row.get("file_sha256") or "")):
            raise HistoricalS5CacheError("historical S5 source file digest invalid")
        source_body = {key: item for key, item in row.items() if key != "source_sha256"}
        if row.get("source_sha256") != _canonical_sha(source_body):
            raise HistoricalS5CacheError("historical S5 source digest mismatch")
        if (
            row.get("historical_only") is not True
            or row.get("forward_proof_eligible") is not False
        ):
            raise HistoricalS5CacheError("historical S5 source proof flags invalid")
        if row.get("selection_policy") != selection_policy:
            raise HistoricalS5CacheError(
                "historical S5 source selection policy mismatch"
            )
        if explicit_run_ids is not None:
            relative = Path(str(row.get("relative_path") or ""))
            summary_relative = Path(str(row.get("summary_relative_path") or ""))
            if (
                relative.is_absolute()
                or summary_relative.is_absolute()
                or ".." in relative.parts
                or ".." in summary_relative.parts
                or len(relative.parts) < 3
                or len(summary_relative.parts) != 2
                or relative.parts[0] not in explicit_run_ids
                or summary_relative.parts[0] not in explicit_run_ids
                or summary_relative.parts[1] != "summary.json"
            ):
                raise HistoricalS5CacheError(
                    "historical S5 source escapes explicit run scope"
                )
    duplicates = value.get("duplicate_candidates")
    if not isinstance(duplicates, list):
        raise HistoricalS5CacheError("historical duplicate candidates are invalid")
    for duplicate in duplicates:
        if not isinstance(duplicate, Mapping):
            raise HistoricalS5CacheError(
                "historical duplicate candidate row is invalid"
            )
        if duplicate.get("resolution_policy") != selection_policy:
            raise HistoricalS5CacheError(
                "historical duplicate resolution policy mismatch"
            )
        if explicit_run_ids is not None:
            paths = duplicate.get("candidate_relative_paths")
            selected = str(duplicate.get("selected_relative_path") or "")
            if not isinstance(paths, list) or any(
                Path(str(path)).is_absolute()
                or ".." in Path(str(path)).parts
                or not Path(str(path)).parts
                or Path(str(path)).parts[0] not in explicit_run_ids
                for path in [selected, *paths]
            ):
                raise HistoricalS5CacheError(
                    "historical duplicate candidate escapes explicit run scope"
                )
    unadmitted = value.get("unadmitted_files")
    if not isinstance(unadmitted, list):
        raise HistoricalS5CacheError("historical unadmitted source list is invalid")
    if explicit_run_ids is not None and any(
        Path(str(path)).is_absolute()
        or ".." in Path(str(path)).parts
        or not Path(str(path)).parts
        or Path(str(path)).parts[0] not in explicit_run_ids
        for path in unadmitted
    ):
        raise HistoricalS5CacheError(
            "historical unadmitted source escapes explicit run scope"
        )


def _validate_pairs(value: Sequence[str]) -> tuple[str, ...]:
    pairs = tuple(value)
    if not pairs or len(set(pairs)) != len(pairs):
        raise HistoricalS5CacheError(
            "historical S5 pair scope must be unique and non-empty"
        )
    if any(not isinstance(pair, str) or not _PAIR_RE.fullmatch(pair) for pair in pairs):
        raise HistoricalS5CacheError(
            "historical S5 pair scope contains an invalid pair"
        )
    return pairs


def _strict_json_loads(payload: bytes, *, label: str) -> Any:
    def object_hook(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise HistoricalS5CacheError(f"duplicate JSON key in {label}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise HistoricalS5CacheError(f"non-finite JSON number in {label}: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalS5CacheError(f"invalid JSON in {label}") from exc


def _positive_finite(value: Any, *, pair: str) -> float:
    if value.__class__ not in {int, float}:
        raise HistoricalS5CacheError(f"{pair}: S5 cache price type is invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise HistoricalS5CacheError(f"{pair}: S5 cache price is invalid")
    return number


def _parse_oanda_utc(value: Any) -> datetime:
    match = _OANDA_UTC_RE.fullmatch(str(value or ""))
    if match is None:
        raise HistoricalS5CacheError("S5 cache timestamp is not canonical OANDA UTC")
    raw_fraction = match.group("fraction") or ""
    if any(character != "0" for character in raw_fraction):
        # Python datetime stores microseconds.  Reject rather than truncate a
        # non-zero nanosecond tail that could otherwise masquerade as :00 S5.
        raise HistoricalS5CacheError("S5 cache timestamp is off-grid")
    fraction = raw_fraction[:6].ljust(6, "0")
    parsed = datetime.fromisoformat(
        match.group("seconds") + (f".{fraction}" if fraction else "") + "+00:00"
    )
    return parsed.astimezone(timezone.utc)


def _parse_aware_utc(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HistoricalS5CacheError("timestamp must be aware UTC") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise HistoricalS5CacheError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    utc = _aware_utc(value)
    return utc.replace(microsecond=0) - timedelta(seconds=utc.second % 5)


def _ceil_s5(value: datetime) -> datetime:
    utc = _aware_utc(value)
    floored = _floor_s5(utc)
    return floored if utc == floored else floored + timedelta(seconds=5)


def _filename_stamp(value: datetime) -> str:
    return _aware_utc(value).strftime("%Y%m%dT%H%M%SZ")


@contextmanager
def _cache_binary_stream(raw: BinaryIO, path: Path) -> Iterator[BinaryIO]:
    raw.seek(0)
    if path.name.endswith(".jsonl.gz"):
        with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
            yield handle
        return
    if path.name.endswith(".jsonl"):
        yield raw
        return
    raise HistoricalS5CacheError("unsupported historical S5 cache suffix")


@contextmanager
def _stable_regular_binary(path: Path) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Bind all reads to one no-follow FD and reject concurrent path drift."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise HistoricalS5CacheError(
            f"stable historical source open failed: {path}"
        ) from exc
    try:
        opened = os.fstat(fd)
        _require_regular_stat(opened, label=str(path))
        _require_path_bound_to_stat(path, opened)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            try:
                yield handle, opened
            finally:
                after = os.fstat(fd)
                if _stable_stat_fingerprint(after) != _stable_stat_fingerprint(opened):
                    raise HistoricalS5CacheError(
                        "historical S5 source changed during stable read"
                    )
                _require_path_bound_to_stat(path, opened)
    finally:
        os.close(fd)


def _read_stable_bytes(path: Path) -> bytes:
    with _stable_regular_binary(path) as (handle, _opened):
        return handle.read()


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _require_regular_stat(value: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise HistoricalS5CacheError(f"historical source is not regular: {label}")


def _require_path_bound_to_stat(path: Path, expected: os.stat_result) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise HistoricalS5CacheError(
            "historical S5 source path changed during stable read"
        ) from exc
    _require_regular_stat(current, label=str(path))
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise HistoricalS5CacheError(
            "historical S5 source path identity changed during stable read"
        )


def _stable_stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _resolve_manifest_relative_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise HistoricalS5CacheError("historical S5 relative path is unsafe")
    unresolved = root / rel
    if unresolved.is_symlink():
        raise HistoricalS5CacheError("historical S5 source cannot be a symlink")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise HistoricalS5CacheError(
            "historical S5 source escapes manifest root"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise HistoricalS5CacheError("historical S5 source is not a regular file")
    return path


def _relative_inside(root: Path, path: Path) -> str:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    return resolved_path.relative_to(resolved_root).as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _truth_path_sha256(candles: Sequence[S5BidAskCandle]) -> str:
    """Hash the canonical candle path without materializing a second full copy."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, candle in enumerate(candles):
        if index:
            digest.update(b",")
        row = {
            "timestamp_utc": candle.timestamp_utc.isoformat(),
            "bid": [candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c],
            "ask": [candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c],
        }
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
