"""Deep source seal for the fixed DOJO long-horizon historical TRAIN.

The long-horizon plan names three deliberately separate source contexts:

* exact-28 M5, 2020-01 through 2026-06;
* core-five M1, 2020-01 through 2026-06;
* exact-28 M1, 2025-01 through 2026-06.

This module resolves every required pair/year shard from the two acquisition
roots, opens each selected regular file through a stable no-follow descriptor,
validates every OANDA candle row, verifies the append-only acquisition receipt,
and seals both physical-source and selected-corpus digests.  Acquisition run
summaries are not trusted inputs.  A terminal 2026 file may extend beyond the
half-open study boundary, but only rows before 2026-07-01 enter the corpus
digest and the bounded superset is disclosed explicitly.

The artifact is worn historical TRAIN provenance.  It cannot fetch data,
start a replay, call a model, mutate a broker, promote a strategy or grant live
authority.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit import fast_bot_historical_s5 as _history
from quant_rabbit.dojo_long_horizon_plan import (
    CORE5_PAIRS,
    M1_CORE5_BINDING_ID,
    M1_CORE5_MONTHS,
    M1_FULL28_BINDING_ID,
    M1_FULL28_MONTHS,
    M5_BINDING_ID,
    M5_MONTHS,
    PERIOD_FROM_UTC,
    PERIOD_TO_UTC,
    SOURCE_BINDING_IDS,
    canonical_sha256,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


CONTRACT: Final = "QR_DOJO_LONG_HORIZON_SOURCE_ROOT_MANIFEST_V1"
SCHEMA_VERSION: Final = 1
CLASSIFICATION: Final = "WORN_HISTORICAL_TRAIN_ONLY"
PRICE_COMPONENT: Final = "BA"
MAX_BOUNDARY_GAP_SECONDS: Final = 7 * 24 * 60 * 60
MIN_OBSERVED_SLOT_COVERAGE: Final = 0.50
MAX_TERMINAL_SUPERSET_SECONDS: Final = 31 * 24 * 60 * 60

_ROW_KEYS: Final = frozenset(
    {"ask", "bid", "complete", "granularity", "pair", "price", "time", "volume"}
)
_PRICE_KEYS: Final = frozenset({"o", "h", "l", "c"})
_CACHE_RE: Final = re.compile(
    r"^(?P<pair>[A-Z]{3}_[A-Z]{3})_(?P<granularity>M1|M5)_BA_"
    r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)\.jsonl\.gz$"
)
_RUN_RE: Final = re.compile(r"\d{8}T\d{6}Z\Z")
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_RE: Final = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")
_MONTH_RE: Final = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")
_SUMMARY_KEYS: Final = frozenset(
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
_TASK_KEYS: Final = frozenset(
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
_RECEIPT_KEYS: Final = frozenset(
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


class DojoLongHorizonSourceManifestError(ValueError):
    """A required historical shard is missing, ambiguous or untrustworthy."""


@dataclass(frozen=True, slots=True)
class _BindingSpec:
    binding_id: str
    granularity: str
    pairs: tuple[str, ...]
    months: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedShard:
    granularity: str
    pair: str
    slice_from: datetime
    slice_to: datetime

    @property
    def cell_key(self) -> tuple[str, str, int]:
        return (self.granularity, self.pair, self.slice_from.year)

    @property
    def cadence_seconds(self) -> int:
        return 60 if self.granularity == "M1" else 300


@dataclass(frozen=True, slots=True)
class _Candidate:
    root_kind: str
    root: Path
    path: Path
    relative_path: str
    pair: str
    granularity: str
    declared_from: datetime
    declared_to: datetime

    @property
    def cell_key(self) -> tuple[str, str, int]:
        return (self.granularity, self.pair, self.declared_from.year)


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "forward_proof_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }


def _fixed_specs() -> tuple[_BindingSpec, ...]:
    return (
        _BindingSpec(
            binding_id=M5_BINDING_ID,
            granularity="M5",
            pairs=tuple(DEFAULT_TRADER_PAIRS),
            months=tuple(M5_MONTHS),
        ),
        _BindingSpec(
            binding_id=M1_CORE5_BINDING_ID,
            granularity="M1",
            pairs=tuple(CORE5_PAIRS),
            months=tuple(M1_CORE5_MONTHS),
        ),
        _BindingSpec(
            binding_id=M1_FULL28_BINDING_ID,
            granularity="M1",
            pairs=tuple(DEFAULT_TRADER_PAIRS),
            months=tuple(M1_FULL28_MONTHS),
        ),
    )


def _parse_stamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise DojoLongHorizonSourceManifestError(
            "source shard filename contains an invalid UTC clock"
        ) from exc


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _root(path: Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise DojoLongHorizonSourceManifestError(
            f"{field} must be an absolute path without traversal"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DojoLongHorizonSourceManifestError(f"{field} is unavailable") from exc
    if candidate != resolved or candidate.is_symlink() or not candidate.is_dir():
        raise DojoLongHorizonSourceManifestError(
            f"{field} must be a real non-symlink directory"
        )
    return candidate


def _spec_shards(spec: _BindingSpec) -> tuple[_ExpectedShard, ...]:
    if spec.granularity not in {"M1", "M5"}:
        raise DojoLongHorizonSourceManifestError("unsupported source granularity")
    if not spec.pairs or len(set(spec.pairs)) != len(spec.pairs):
        raise DojoLongHorizonSourceManifestError(
            "binding pairs must be nonempty and unique"
        )
    if any(_PAIR_RE.fullmatch(pair) is None for pair in spec.pairs):
        raise DojoLongHorizonSourceManifestError("binding pair syntax is invalid")
    if not spec.months or tuple(sorted(spec.months)) != spec.months:
        raise DojoLongHorizonSourceManifestError(
            "binding months must be nonempty, unique and sorted"
        )
    if any(_MONTH_RE.fullmatch(month) is None for month in spec.months):
        raise DojoLongHorizonSourceManifestError("binding month syntax is invalid")
    month_dates = [
        datetime(int(month[:4]), int(month[5:]), 1, tzinfo=timezone.utc)
        for month in spec.months
    ]
    if any(
        right.year * 12 + right.month != left.year * 12 + left.month + 1
        for left, right in zip(month_dates, month_dates[1:])
    ):
        raise DojoLongHorizonSourceManifestError("binding months are not contiguous")
    period_from = month_dates[0]
    final = month_dates[-1]
    period_to = (
        datetime(final.year + 1, 1, 1, tzinfo=timezone.utc)
        if final.month == 12
        else datetime(final.year, final.month + 1, 1, tzinfo=timezone.utc)
    )
    shards: list[_ExpectedShard] = []
    for pair in spec.pairs:
        cursor = period_from
        while cursor < period_to:
            next_year = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
            shard_to = min(next_year, period_to)
            shards.append(
                _ExpectedShard(
                    granularity=spec.granularity,
                    pair=pair,
                    slice_from=cursor,
                    slice_to=shard_to,
                )
            )
            cursor = shard_to
    return tuple(shards)


def _inventory(root: Path, *, root_kind: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        relative_base = base.relative_to(root)
        if len(relative_base.parts) > 2:
            raise DojoLongHorizonSourceManifestError(
                f"{root_kind} source root contains an orphan directory"
            )
        for name in sorted(dirnames):
            child = base / name
            if child.is_symlink():
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source root contains a symlink"
                )
            if (
                (not relative_base.parts and _RUN_RE.fullmatch(name) is None)
                or (len(relative_base.parts) == 1 and _PAIR_RE.fullmatch(name) is None)
                or len(relative_base.parts) >= 2
            ):
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source root contains an orphan directory"
                )
        for name in sorted(filenames):
            child = base / name
            if child.is_symlink() or not child.is_file():
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source root contains a non-regular file"
                )
            if (
                name.endswith((".tmp", ".partial"))
                or ".tmp." in name
                or ".partial." in name
            ):
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source root contains temporary/partial debris"
                )
            match = _CACHE_RE.fullmatch(name)
            if match is None:
                if name == _history.TRUTH_RECEIPT_FILE and not relative_base.parts:
                    continue
                if name == "latest_summary.json" and not relative_base.parts:
                    continue
                if name == "summary.json" and len(relative_base.parts) == 1:
                    continue
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source root contains an orphan file"
                )
            granularity = match.group("granularity")
            if granularity != root_kind:
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} root contains a {granularity} shard"
                )
            if len(relative_base.parts) != 2 or relative_base.parts[1] != match.group(
                "pair"
            ):
                raise DojoLongHorizonSourceManifestError(
                    f"{root_kind} source shard layout is not canonical"
                )
            declared_from = _parse_stamp(match.group("start"))
            declared_to = _parse_stamp(match.group("end"))
            if declared_from >= declared_to:
                raise DojoLongHorizonSourceManifestError(
                    "source shard filename window is not positive"
                )
            try:
                relative = child.relative_to(root).as_posix()
            except ValueError as exc:
                raise DojoLongHorizonSourceManifestError(
                    "source shard escapes its root"
                ) from exc
            candidates.append(
                _Candidate(
                    root_kind=root_kind,
                    root=root,
                    path=child,
                    relative_path=relative,
                    pair=match.group("pair"),
                    granularity=granularity,
                    declared_from=declared_from,
                    declared_to=declared_to,
                )
            )
    return sorted(
        candidates,
        key=lambda row: (
            row.granularity,
            row.pair,
            row.declared_from,
            row.declared_to,
            row.relative_path,
        ),
    )


def _select_candidates(
    candidates: Sequence[_Candidate],
    expected: Sequence[_ExpectedShard],
) -> dict[tuple[str, str, int], tuple[tuple[_Candidate, ...], _ExpectedShard]]:
    expected_by_cell: dict[tuple[str, str, int], _ExpectedShard] = {}
    for shard in expected:
        prior = expected_by_cell.setdefault(shard.cell_key, shard)
        if prior != shard:
            raise DojoLongHorizonSourceManifestError(
                "binding specs disagree about one physical shard slice"
            )
    result: dict[
        tuple[str, str, int], tuple[tuple[_Candidate, ...], _ExpectedShard]
    ] = {}
    for cell, shard in sorted(expected_by_cell.items()):
        overlapping = [
            candidate
            for candidate in candidates
            if candidate.cell_key == cell
            and candidate.declared_from < shard.slice_to
            and candidate.declared_to > shard.slice_from
        ]
        if not overlapping:
            raise DojoLongHorizonSourceManifestError(
                f"required source shard is missing: {cell}"
            )
        for candidate in overlapping:
            if candidate.declared_from != shard.slice_from:
                raise DojoLongHorizonSourceManifestError(
                    "source shard does not start at its exact annual boundary"
                )
            superset_seconds = int(
                (candidate.declared_to - shard.slice_to).total_seconds()
            )
            if superset_seconds < 0 or superset_seconds > MAX_TERMINAL_SUPERSET_SECONDS:
                raise DojoLongHorizonSourceManifestError(
                    "source shard does not cover the exact slice or overextends its bound"
                )
            if (
                shard.slice_to.month == 1
                and shard.slice_to.day == 1
                and superset_seconds
            ):
                raise DojoLongHorizonSourceManifestError(
                    "non-terminal annual shard may not contain later-year rows"
                )
        result[cell] = (tuple(overlapping), shard)
    return result


def _validate_row(
    value: Any, *, pair: str, granularity: str, cadence_seconds: int
) -> datetime:
    if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
        raise DojoLongHorizonSourceManifestError(
            f"{pair}: candle schema is invalid or synthetic metadata is present"
        )
    if (
        value.get("pair") != pair
        or value.get("granularity") != granularity
        or value.get("price") != PRICE_COMPONENT
        or value.get("complete") is not True
    ):
        raise DojoLongHorizonSourceManifestError(
            f"{pair}: candle provenance is not complete {granularity}/BA"
        )
    volume = value.get("volume")
    if volume.__class__ is not int or volume < 0:
        raise DojoLongHorizonSourceManifestError(f"{pair}: candle volume is invalid")
    try:
        timestamp = _history._parse_oanda_utc(value.get("time"))
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            f"{pair}: candle timestamp is invalid"
        ) from exc
    epoch = int(timestamp.timestamp())
    if timestamp.microsecond or epoch % cadence_seconds:
        raise DojoLongHorizonSourceManifestError(
            f"{pair}: candle timestamp is off the {granularity} grid"
        )
    prices: dict[str, float] = {}
    for prefix in ("bid", "ask"):
        block = value.get(prefix)
        if not isinstance(block, Mapping) or set(block) != _PRICE_KEYS:
            raise DojoLongHorizonSourceManifestError(
                f"{pair}: {prefix} OHLC schema is invalid"
            )
        parsed: dict[str, float] = {}
        for key in _PRICE_KEYS:
            raw = block.get(key)
            if raw.__class__ not in {int, float}:
                raise DojoLongHorizonSourceManifestError(
                    f"{pair}: candle price type is invalid"
                )
            number = float(raw)
            if not math.isfinite(number) or number <= 0.0:
                raise DojoLongHorizonSourceManifestError(
                    f"{pair}: candle price is invalid"
                )
            parsed[key] = number
            prices[f"{prefix}_{key}"] = number
        if not (
            parsed["l"]
            <= min(parsed["o"], parsed["c"])
            <= max(parsed["o"], parsed["c"])
            <= parsed["h"]
        ):
            raise DojoLongHorizonSourceManifestError(
                f"{pair}: candle OHLC ordering is invalid"
            )
    if any(prices[f"bid_{key}"] > prices[f"ask_{key}"] for key in _PRICE_KEYS):
        raise DojoLongHorizonSourceManifestError(
            f"{pair}: executable bid/ask spread is crossed"
        )
    return timestamp


def _approved_fetcher(granularity: str) -> tuple[Path, str]:
    filename = (
        "oanda_history_fetch_m1.py" if granularity == "M1" else "oanda_history_fetch.py"
    )
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    if path.is_symlink():
        raise DojoLongHorizonSourceManifestError(
            "approved acquisition fetcher may not be a symlink"
        )
    try:
        resolved = path.resolve(strict=True)
        raw = _history._read_stable_bytes(resolved)
    except (OSError, _history.HistoricalS5CacheError) as exc:
        raise DojoLongHorizonSourceManifestError(
            "approved acquisition fetcher is unavailable"
        ) from exc
    if not resolved.is_file():
        raise DojoLongHorizonSourceManifestError(
            "approved acquisition fetcher is not a regular file"
        )
    return resolved, hashlib.sha256(raw).hexdigest()


def _load_truth_receipts(
    root: Path,
    *,
    granularity: str,
    approved_fetcher_path: Path,
    approved_fetcher_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    """Verify one complete append-only ledger against an approved fetcher."""

    path = root / _history.TRUTH_RECEIPT_FILE
    if path.is_symlink() or not path.is_file():
        raise DojoLongHorizonSourceManifestError(
            "source acquisition receipt ledger is missing or invalid"
        )
    try:
        ledger = _history._read_stable_bytes(path)
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            "source acquisition receipt ledger changed while reading"
        ) from exc
    rows: dict[str, Mapping[str, Any]] = {}
    previous: str | None = None
    for line_number, raw in enumerate(ledger.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = _history._strict_json_loads(raw, label=f"{path}:{line_number}")
        except _history.HistoricalS5CacheError as exc:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt ledger contains invalid JSON"
            ) from exc
        if not isinstance(item, Mapping) or set(item) != _RECEIPT_KEYS:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt schema is not exact"
            )
        window = item.get("window")
        if (
            not isinstance(window, Mapping)
            or set(window) != {"from_utc", "to_utc"}
            or item.get("pair").__class__ is not str
            or _PAIR_RE.fullmatch(item["pair"]) is None
            or item.get("rows").__class__ is not int
            or item["rows"] <= 0
            or _SHA_RE.fullmatch(str(item.get("candle_sha256") or "")) is None
            or _SHA_RE.fullmatch(str(item.get("receipt_sha256") or "")) is None
        ):
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt payload is invalid"
            )
        fetch_script = Path(str(item.get("fetch_script_path") or ""))
        if (
            item.get("schema_version") != _history.TRUTH_RECEIPT_SCHEMA
            or item.get("sequence") != len(rows) + 1
            or item.get("previous_receipt_sha256") != previous
            or item.get("output_root") != str(root)
            or item.get("granularity") != granularity
            or item.get("price_component") != PRICE_COMPONENT
            or not fetch_script.is_absolute()
            or fetch_script.is_symlink()
            or fetch_script.name != approved_fetcher_path.name
            or item.get("fetch_script_sha256") != approved_fetcher_sha256
        ):
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt provenance or chain differs"
            )
        try:
            executed_fetcher = fetch_script.resolve(strict=True)
            executed_sha = hashlib.sha256(
                _history._read_stable_bytes(executed_fetcher)
            ).hexdigest()
            recorded_at = _history._parse_aware_utc(item.get("recorded_at_utc"))
            receipt_from = _history._parse_aware_utc(window["from_utc"])
            receipt_to = _history._parse_aware_utc(window["to_utc"])
        except (OSError, _history.HistoricalS5CacheError) as exc:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt fetcher or clock is invalid"
            ) from exc
        if (
            not executed_fetcher.is_file()
            or executed_sha != approved_fetcher_sha256
            or recorded_at > datetime.now(timezone.utc)
            or receipt_from >= receipt_to
            or recorded_at < receipt_to
        ):
            raise DojoLongHorizonSourceManifestError(
                "source acquisition implementation identity drifted"
            )
        body = {key: value for key, value in item.items() if key != "receipt_sha256"}
        receipt_sha = canonical_sha256(body)
        if item.get("receipt_sha256") != receipt_sha:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt digest differs"
            )
        candle_path = Path(str(item.get("candle_path") or ""))
        if not candle_path.is_absolute():
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt candle path is not absolute"
            )
        try:
            relative = _history._relative_inside(root, candle_path)
        except (OSError, ValueError) as exc:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt candle path escapes its root"
            ) from exc
        cache_match = _CACHE_RE.fullmatch(Path(relative).name)
        if (
            cache_match is None
            or cache_match.group("pair") != item["pair"]
            or cache_match.group("granularity") != granularity
            or _parse_stamp(cache_match.group("start")) != receipt_from
            or _parse_stamp(cache_match.group("end")) != receipt_to
        ):
            raise DojoLongHorizonSourceManifestError(
                "source acquisition receipt window/path binding differs"
            )
        if relative in rows:
            raise DojoLongHorizonSourceManifestError(
                "source acquisition ledger duplicates one candle file"
            )
        rows[relative] = item
        previous = receipt_sha
    if not rows:
        raise DojoLongHorizonSourceManifestError(
            "source acquisition receipt ledger is empty"
        )
    return rows


def _receipt(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    candidate: _Candidate,
    full_row_count: int,
    file_sha256: str,
) -> Mapping[str, Any]:
    receipt = receipts.get(candidate.relative_path)
    if receipt is None:
        raise DojoLongHorizonSourceManifestError(
            "selected source shard lacks an acquisition receipt"
        )
    window = receipt.get("window")
    try:
        receipt_from = _history._parse_aware_utc(window["from_utc"])
        receipt_to = _history._parse_aware_utc(window["to_utc"])
        candle_relative = (
            Path(str(receipt.get("candle_path"))).relative_to(candidate.root).as_posix()
        )
    except (KeyError, TypeError, ValueError, _history.HistoricalS5CacheError) as exc:
        raise DojoLongHorizonSourceManifestError(
            "selected source receipt is malformed"
        ) from exc
    try:
        recorded_at = _history._parse_aware_utc(receipt.get("recorded_at_utc"))
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            "selected source receipt clock is invalid"
        ) from exc
    if (
        receipt.get("output_root") != str(candidate.root)
        or receipt.get("pair") != candidate.pair
        or receipt.get("granularity") != candidate.granularity
        or receipt.get("price_component") != PRICE_COMPONENT
        or receipt.get("rows") != full_row_count
        or receipt.get("candle_sha256") != file_sha256
        or receipt_from != candidate.declared_from
        or receipt_to != candidate.declared_to
        or candle_relative != candidate.relative_path
        or recorded_at < candidate.declared_to
    ):
        raise DojoLongHorizonSourceManifestError(
            "selected source receipt does not bind the scanned shard"
        )
    receipt_sha = receipt.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or _SHA_RE.fullmatch(receipt_sha) is None:
        raise DojoLongHorizonSourceManifestError(
            "selected source receipt digest is invalid"
        )
    return receipt


def _request_window_count(
    *, start: datetime, end: datetime, cadence_seconds: int, max_candles: int
) -> int:
    chunk_seconds = cadence_seconds * max(1, max_candles - 1)
    total_seconds = int((end - start).total_seconds())
    return (total_seconds + chunk_seconds - 1) // chunk_seconds


def _summary_request_proof(
    candidate: _Candidate,
    *,
    full_row_count: int,
    receipt_sha256: str,
) -> dict[str, Any]:
    summary_path = candidate.path.parents[1] / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise DojoLongHorizonSourceManifestError(
            "selected shard lacks its canonical acquisition summary"
        )
    try:
        raw = _history._read_stable_bytes(summary_path)
        summary = _history._strict_json_loads(raw, label=str(summary_path))
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary is invalid"
        ) from exc
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary schema is not exact"
        )
    tasks = summary.get("tasks")
    if (
        summary.get("dry_run") is not False
        or summary.get("errors") != []
        or summary.get("granularities") != [candidate.granularity]
        or summary.get("price") != PRICE_COMPONENT
        or not isinstance(tasks, list)
        or not tasks
        or any(
            not isinstance(task, Mapping) or set(task) != _TASK_KEYS for task in tasks
        )
    ):
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary is not a complete published run"
        )
    if summary.get("pairs") != [task.get("pair") for task in tasks]:
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary pair/task order differs"
        )
    integer_fields = ("rows", "requests", "windows")
    if (
        summary.get("total_rows").__class__ is not int
        or summary.get("total_rows") < 0
        or summary.get("total_requests").__class__ is not int
        or summary.get("total_requests") < 1
        or any(
            task.get(field).__class__ is not int or task.get(field) < 0
            for task in tasks
            for field in integer_fields
        )
    ):
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary counters are invalid"
        )
    output_dir = summary.get("output_dir")
    if (
        not isinstance(output_dir, str)
        or Path(output_dir).name != summary_path.parent.name
    ):
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition output/run binding differs"
        )
    window = summary.get("window")
    try:
        summary_from = _history._parse_aware_utc(window["from"])
        summary_to = _history._parse_aware_utc(window["to"])
    except (KeyError, TypeError, _history.HistoricalS5CacheError) as exc:
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary window is invalid"
        ) from exc
    if summary_from != candidate.declared_from or summary_to != candidate.declared_to:
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary window differs from filename"
        )
    if summary.get("total_rows") != sum(task["rows"] for task in tasks) or summary.get(
        "total_requests"
    ) != sum(task["requests"] for task in tasks):
        raise DojoLongHorizonSourceManifestError(
            "selected shard acquisition summary aggregate arithmetic differs"
        )
    selected_tasks = [
        task
        for task in tasks
        if task.get("pair") == candidate.pair
        and Path(str(task.get("path") or "")).name == candidate.path.name
    ]
    if len(selected_tasks) != 1:
        raise DojoLongHorizonSourceManifestError(
            "selected shard task is missing or duplicated in acquisition summary"
        )
    task = selected_tasks[0]
    try:
        task_from = _history._parse_aware_utc(task["from"])
        task_to = _history._parse_aware_utc(task["to"])
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            "selected shard task clock is invalid"
        ) from exc
    max_candles = summary.get("max_candles_per_request")
    cadence = 60 if candidate.granularity == "M1" else 300
    if max_candles.__class__ is not int or not 1 <= max_candles <= 5_000:
        raise DojoLongHorizonSourceManifestError(
            "selected shard max-candles request bound is invalid"
        )
    expected_windows = _request_window_count(
        start=candidate.declared_from,
        end=candidate.declared_to,
        cadence_seconds=cadence,
        max_candles=max_candles,
    )
    if (
        task.get("granularity") != candidate.granularity
        or task.get("price") != PRICE_COMPONENT
        or task.get("dry_run") is not False
        or task.get("published") is not True
        or task.get("compressed") is not True
        or task.get("partial_path") is not None
        or task.get("errors") != []
        or task.get("rows") != full_row_count
        or task.get("requests") != expected_windows
        or task.get("windows") != expected_windows
        or task.get("truth_acquisition_receipt_sha256") != receipt_sha256
        or task_from != candidate.declared_from
        or task_to != candidate.declared_to
    ):
        raise DojoLongHorizonSourceManifestError(
            "selected shard task does not prove every request window"
        )
    relative_summary = summary_path.relative_to(candidate.root).as_posix()
    return {
        "summary_relative_path": relative_summary,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "max_candles_per_request": max_candles,
        "request_count": expected_windows,
        "window_count": expected_windows,
        "request_window_completion_report_proved": True,
    }


def _scan_candidate(
    candidate: _Candidate,
    shard: _ExpectedShard,
    *,
    receipt_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    file_hasher = hashlib.sha256()
    full_uncompressed_hasher = hashlib.sha256()
    slice_hasher = hashlib.sha256()
    full_first: datetime | None = None
    full_last: datetime | None = None
    slice_first: datetime | None = None
    slice_last: datetime | None = None
    previous: datetime | None = None
    full_rows = 0
    slice_rows = 0
    month_stats: dict[str, dict[str, Any]] = {}
    month_cursor = shard.slice_from
    while month_cursor < shard.slice_to:
        month = month_cursor.strftime("%Y-%m")
        next_month = (
            datetime(month_cursor.year + 1, 1, 1, tzinfo=timezone.utc)
            if month_cursor.month == 12
            else datetime(
                month_cursor.year, month_cursor.month + 1, 1, tzinfo=timezone.utc
            )
        )
        month_stats[month] = {
            "from": month_cursor,
            "to": min(next_month, shard.slice_to),
            "row_count": 0,
            "first": None,
            "last": None,
            "previous": None,
            "hasher": hashlib.sha256(),
            "gaps": [],
        }
        month_cursor = next_month
    try:
        with _history._stable_regular_binary(candidate.path) as (raw_handle, opened):
            for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
                file_hasher.update(chunk)
            raw_handle.seek(0)
            with gzip.GzipFile(fileobj=raw_handle, mode="rb") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    full_uncompressed_hasher.update(raw_line)
                    try:
                        value = _history._strict_json_loads(
                            raw_line,
                            label=f"{candidate.path}:{line_number}",
                        )
                    except _history.HistoricalS5CacheError as exc:
                        raise DojoLongHorizonSourceManifestError(
                            "source shard contains invalid strict JSON"
                        ) from exc
                    timestamp = _validate_row(
                        value,
                        pair=candidate.pair,
                        granularity=candidate.granularity,
                        cadence_seconds=shard.cadence_seconds,
                    )
                    if not candidate.declared_from <= timestamp < candidate.declared_to:
                        raise DojoLongHorizonSourceManifestError(
                            "source row lies outside its filename window"
                        )
                    if previous is not None and timestamp <= previous:
                        raise DojoLongHorizonSourceManifestError(
                            "source timestamps are duplicated or not increasing"
                        )
                    previous = timestamp
                    full_first = timestamp if full_first is None else full_first
                    full_last = timestamp
                    full_rows += 1
                    if shard.slice_from <= timestamp < shard.slice_to:
                        slice_hasher.update(raw_line)
                        slice_first = timestamp if slice_first is None else slice_first
                        slice_last = timestamp
                        slice_rows += 1
                        month = timestamp.strftime("%Y-%m")
                        stats = month_stats.get(month)
                        if stats is None:
                            raise DojoLongHorizonSourceManifestError(
                                "source row month is absent from the sealed slice"
                            )
                        month_previous = stats["previous"]
                        if month_previous is not None:
                            gap_seconds = int(
                                (timestamp - month_previous).total_seconds()
                            )
                            if gap_seconds > shard.cadence_seconds:
                                stats["gaps"].append(
                                    {
                                        "from_utc": _canonical_utc(
                                            month_previous
                                            + timedelta(seconds=shard.cadence_seconds)
                                        ),
                                        "to_utc": _canonical_utc(timestamp),
                                        "missing_slot_count": gap_seconds
                                        // shard.cadence_seconds
                                        - 1,
                                    }
                                )
                        stats["previous"] = timestamp
                        stats["first"] = (
                            timestamp if stats["first"] is None else stats["first"]
                        )
                        stats["last"] = timestamp
                        stats["row_count"] += 1
                        stats["hasher"].update(raw_line)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise DojoLongHorizonSourceManifestError(
            "source gzip changed, truncated or failed validation"
        ) from exc
    if (
        full_rows <= 0
        or slice_rows <= 0
        or full_first is None
        or full_last is None
        or slice_first is None
        or slice_last is None
    ):
        raise DojoLongHorizonSourceManifestError(
            "source shard or selected slice is empty"
        )
    file_sha = file_hasher.hexdigest()
    receipt = _receipt(
        receipt_rows,
        candidate=candidate,
        full_row_count=full_rows,
        file_sha256=file_sha,
    )
    request_proof = _summary_request_proof(
        candidate,
        full_row_count=full_rows,
        receipt_sha256=str(receipt["receipt_sha256"]),
    )
    grid_slots = int(
        (shard.slice_to - shard.slice_from).total_seconds() // shard.cadence_seconds
    )
    leading_gap = int((slice_first - shard.slice_from).total_seconds())
    trailing_gap = max(
        0,
        int(
            (
                shard.slice_to - (slice_last + timedelta(seconds=shard.cadence_seconds))
            ).total_seconds()
        ),
    )
    coverage = slice_rows / grid_slots
    if coverage < MIN_OBSERVED_SLOT_COVERAGE:
        raise DojoLongHorizonSourceManifestError(
            "source selected-slice observed coverage is implausibly low"
        )
    if (
        leading_gap > MAX_BOUNDARY_GAP_SECONDS
        or trailing_gap > MAX_BOUNDARY_GAP_SECONDS
    ):
        raise DojoLongHorizonSourceManifestError(
            "source selected-slice boundary coverage is truncated"
        )
    month_rows: list[dict[str, Any]] = []
    for month, stats in sorted(month_stats.items()):
        month_first = stats["first"]
        month_last = stats["last"]
        month_count = int(stats["row_count"])
        if month_count <= 0 or month_first is None or month_last is None:
            raise DojoLongHorizonSourceManifestError(
                "source selected slice contains an empty calendar month"
            )
        month_slots = int(
            (stats["to"] - stats["from"]).total_seconds() // shard.cadence_seconds
        )
        month_coverage = month_count / month_slots
        month_leading = int((month_first - stats["from"]).total_seconds())
        month_trailing = max(
            0,
            int(
                (
                    stats["to"]
                    - (month_last + timedelta(seconds=shard.cadence_seconds))
                ).total_seconds()
            ),
        )
        if month_coverage < MIN_OBSERVED_SLOT_COVERAGE:
            raise DojoLongHorizonSourceManifestError(
                "source pair-month observed coverage is implausibly low"
            )
        if (
            month_leading > MAX_BOUNDARY_GAP_SECONDS
            or month_trailing > MAX_BOUNDARY_GAP_SECONDS
        ):
            raise DojoLongHorizonSourceManifestError(
                "source pair-month boundary coverage is truncated"
            )
        gaps = stats["gaps"]
        row = {
            "pair": candidate.pair,
            "month": month,
            "from_utc": _canonical_utc(stats["from"]),
            "to_utc": _canonical_utc(stats["to"]),
            "row_count": month_count,
            "grid_slot_count": month_slots,
            "no_candle_slot_count": month_slots - month_count,
            "first_observed_utc": _canonical_utc(month_first),
            "last_observed_utc": _canonical_utc(month_last),
            "leading_gap_seconds": month_leading,
            "trailing_gap_seconds": month_trailing,
            "internal_gap_interval_count": len(gaps),
            "max_internal_gap_seconds": max(
                (
                    int(
                        (
                            _history._parse_aware_utc(gap["to_utc"])
                            - _history._parse_aware_utc(gap["from_utc"])
                        ).total_seconds()
                    )
                    for gap in gaps
                ),
                default=0,
            ),
            "gap_intervals_sha256": canonical_sha256(gaps),
            "uncompressed_row_bytes_sha256": stats["hasher"].hexdigest(),
            "request_window_completion_report_proved": True,
            "missing_slot_legitimacy_proved": False,
            "calendar_open_quote_coverage_proved": False,
        }
        month_rows.append(row)
    month_rows_sha = canonical_sha256(month_rows)
    source_body = {
        "root_kind": candidate.root_kind,
        "relative_path": candidate.relative_path,
        "file_size_bytes": opened.st_size,
        "file_sha256": file_sha,
        "full_uncompressed_bytes_sha256": full_uncompressed_hasher.hexdigest(),
        "declared_from_utc": _canonical_utc(candidate.declared_from),
        "declared_to_utc": _canonical_utc(candidate.declared_to),
        "full_file_row_count": full_rows,
        "full_first_observed_utc": _canonical_utc(full_first),
        "full_last_observed_utc": _canonical_utc(full_last),
        "acquisition_receipt_sha256": receipt["receipt_sha256"],
        "acquisition_recorded_at_utc": _canonical_utc(
            _history._parse_aware_utc(receipt["recorded_at_utc"])
        ),
        **request_proof,
    }
    corpus_body = {
        "granularity": candidate.granularity,
        "pair": candidate.pair,
        "slice_from_utc": _canonical_utc(shard.slice_from),
        "slice_to_utc": _canonical_utc(shard.slice_to),
        "slice_row_count": slice_rows,
        "slice_first_observed_utc": _canonical_utc(slice_first),
        "slice_last_observed_utc": _canonical_utc(slice_last),
        "slice_uncompressed_bytes_sha256": slice_hasher.hexdigest(),
        "cadence_seconds": shard.cadence_seconds,
        "grid_slot_count": grid_slots,
        "observed_slot_coverage": coverage,
        "leading_gap_seconds": leading_gap,
        "trailing_gap_seconds": trailing_gap,
        "synthetic_row_count": 0,
        "duplicate_timestamp_count": 0,
        "pair_month_coverage_data_sha256": month_rows_sha,
    }
    body_without_id = {
        **source_body,
        **corpus_body,
        "terminal_file_superset_seconds": int(
            (candidate.declared_to - shard.slice_to).total_seconds()
        ),
        "source_shard_sha256": canonical_sha256(source_body),
        "corpus_shard_sha256": canonical_sha256(corpus_body),
    }
    physical_shard_id = canonical_sha256(body_without_id)
    coverage_rows = []
    for month_row in month_rows:
        coverage_body = {
            **month_row,
            "physical_shard_id": physical_shard_id,
        }
        coverage_rows.append(
            {
                **coverage_body,
                "coverage_cell_sha256": canonical_sha256(coverage_body),
            }
        )
    return {
        "physical_shard_id": physical_shard_id,
        **body_without_id,
        "pair_month_coverage": coverage_rows,
        "pair_month_coverage_sha256": canonical_sha256(coverage_rows),
    }


def _binding_row(
    spec: _BindingSpec,
    *,
    scans_by_cell: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    scans = [scans_by_cell[shard.cell_key] for shard in _spec_shards(spec)]
    coverage_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for scan in scans:
        for coverage in scan["pair_month_coverage"]:
            key = (coverage["pair"], coverage["month"])
            if key in coverage_by_key:
                raise DojoLongHorizonSourceManifestError(
                    "pair-month coverage is duplicated across physical shards"
                )
            coverage_by_key[key] = dict(coverage)
    expected_coverage_keys = {
        (pair, month) for pair in spec.pairs for month in spec.months
    }
    if set(coverage_by_key) != expected_coverage_keys:
        raise DojoLongHorizonSourceManifestError(
            "binding pair-month coverage denominator is not exact"
        )
    coverage_rows = [
        coverage_by_key[(pair, month)] for pair in spec.pairs for month in spec.months
    ]
    physical_ids = [row["physical_shard_id"] for row in scans]
    source_body = {
        "binding_id": spec.binding_id,
        "granularity": spec.granularity,
        "price_component": PRICE_COMPONENT,
        "pairs": list(spec.pairs),
        "months": list(spec.months),
        "source_shard_sha256_values": [row["source_shard_sha256"] for row in scans],
    }
    corpus_body = {
        "binding_id": spec.binding_id,
        "granularity": spec.granularity,
        "price_component": PRICE_COMPONENT,
        "pairs": list(spec.pairs),
        "months": list(spec.months),
        "corpus_shard_sha256_values": [row["corpus_shard_sha256"] for row in scans],
    }
    return {
        "binding_id": spec.binding_id,
        "granularity": spec.granularity,
        "price_component": PRICE_COMPONENT,
        "pairs": list(spec.pairs),
        "months": list(spec.months),
        "pair_count": len(spec.pairs),
        "month_count": len(spec.months),
        "physical_shard_count": len(scans),
        "physical_shard_ids": physical_ids,
        "physical_shard_ids_sha256": canonical_sha256(physical_ids),
        "month_pair_coverage_count": len(coverage_rows),
        "month_pair_coverage": coverage_rows,
        "month_pair_coverage_sha256": canonical_sha256(coverage_rows),
        "selected_row_count": sum(int(row["slice_row_count"]) for row in scans),
        "source_digest_sha256": canonical_sha256(source_body),
        "corpus_digest_sha256": canonical_sha256(corpus_body),
    }


def _resolve_equivalent_candidates(
    scans: Sequence[Mapping[str, Any]],
    *,
    shard: _ExpectedShard,
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    if not scans:
        raise DojoLongHorizonSourceManifestError(
            "required source shard has no verified candidate"
        )
    if len(scans) == 1:
        return scans[0], None
    equality_fields = (
        "declared_from_utc",
        "declared_to_utc",
        "full_uncompressed_bytes_sha256",
        "full_file_row_count",
        "full_first_observed_utc",
        "full_last_observed_utc",
    )
    reference = tuple(scans[0][field] for field in equality_fields)
    if any(
        tuple(scan[field] for field in equality_fields) != reference for scan in scans
    ):
        raise DojoLongHorizonSourceManifestError(
            "duplicate source candidates are not byte/time/row equivalent"
        )
    recorded = [
        _parse_manifest_utc(
            scan["acquisition_recorded_at_utc"],
            field="duplicate acquisition recorded at",
        )
        for scan in scans
    ]
    latest = max(recorded)
    latest_indexes = [index for index, clock in enumerate(recorded) if clock == latest]
    if len(latest_indexes) != 1:
        raise DojoLongHorizonSourceManifestError(
            "duplicate source candidates tie at the latest receipt clock"
        )
    selected = scans[latest_indexes[0]]
    candidates = sorted(
        (
            {
                "relative_path": scan["relative_path"],
                "run_id": Path(str(scan["relative_path"])).parts[0],
                "compressed_file_sha256": scan["file_sha256"],
                "full_uncompressed_bytes_sha256": scan[
                    "full_uncompressed_bytes_sha256"
                ],
                "full_file_row_count": scan["full_file_row_count"],
                "full_first_observed_utc": scan["full_first_observed_utc"],
                "full_last_observed_utc": scan["full_last_observed_utc"],
                "declared_from_utc": scan["declared_from_utc"],
                "declared_to_utc": scan["declared_to_utc"],
                "acquisition_receipt_sha256": scan["acquisition_receipt_sha256"],
                "acquisition_recorded_at_utc": scan["acquisition_recorded_at_utc"],
                "physical_shard_id": scan["physical_shard_id"],
            }
            for scan in scans
        ),
        key=lambda row: (row["acquisition_recorded_at_utc"], row["relative_path"]),
    )
    body = {
        "equivalence_contract": "FULL_UNCOMPRESSED_CANDLE_BYTES_TIME_ROWS_V1",
        "granularity": shard.granularity,
        "pair": shard.pair,
        "slice_from_utc": _canonical_utc(shard.slice_from),
        "slice_to_utc": _canonical_utc(shard.slice_to),
        "candidate_count": len(candidates),
        "equality_fields": list(equality_fields),
        "candidates": candidates,
        "selected_relative_path": selected["relative_path"],
        "selected_physical_shard_id": selected["physical_shard_id"],
        "selection_policy": "UNIQUE_LATEST_ACQUISITION_RECORDED_AT_UTC_V1",
        "equivalence_proved": True,
    }
    return selected, {**body, "duplicate_equivalence_sha256": canonical_sha256(body)}


def _build_from_specs(
    *,
    m5_root: Path,
    m1_root: Path,
    specs: Sequence[_BindingSpec],
) -> dict[str, Any]:
    source_roots = {
        "M5": _root(m5_root, field="m5_root"),
        "M1": _root(m1_root, field="m1_root"),
    }
    expected = [shard for spec in specs for shard in _spec_shards(spec)]
    unique_expected = {shard.cell_key: shard for shard in expected}
    inventories = {
        granularity: _inventory(root, root_kind=granularity)
        for granularity, root in source_roots.items()
    }
    approved_fetchers = {
        granularity: _approved_fetcher(granularity) for granularity in source_roots
    }
    selected: dict[
        tuple[str, str, int], tuple[tuple[_Candidate, ...], _ExpectedShard]
    ] = {}
    for granularity in ("M5", "M1"):
        relevant = [
            shard
            for shard in unique_expected.values()
            if shard.granularity == granularity
        ]
        if relevant:
            selected.update(_select_candidates(inventories[granularity], relevant))
    receipt_rows = {
        granularity: _load_truth_receipts(
            root,
            granularity=granularity,
            approved_fetcher_path=approved_fetchers[granularity][0],
            approved_fetcher_sha256=approved_fetchers[granularity][1],
        )
        for granularity, root in source_roots.items()
    }
    scans_by_cell: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_equivalence_records: list[dict[str, Any]] = []
    for cell, (candidates, shard) in sorted(selected.items()):
        candidate_scans = [
            _scan_candidate(
                candidate,
                shard,
                receipt_rows=receipt_rows[candidate.granularity],
            )
            for candidate in candidates
        ]
        chosen, duplicate_record = _resolve_equivalent_candidates(
            candidate_scans,
            shard=shard,
        )
        scans_by_cell[cell] = dict(chosen)
        if duplicate_record is not None:
            duplicate_equivalence_records.append(duplicate_record)
    bindings = [_binding_row(spec, scans_by_cell=scans_by_cell) for spec in specs]
    source_digests = {
        row["binding_id"]: row["source_digest_sha256"] for row in bindings
    }
    corpus_digests = {
        row["binding_id"]: row["corpus_digest_sha256"] for row in bindings
    }
    physical = sorted(
        scans_by_cell.values(),
        key=lambda row: (
            row["granularity"],
            row["pair"],
            row["slice_from_utc"],
        ),
    )
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "source_roots": {key: str(value) for key, value in source_roots.items()},
        "study_period": {
            "from_utc": PERIOD_FROM_UTC,
            "to_utc": PERIOD_TO_UTC,
            "half_open": True,
        },
        "selection_policy": (
            "EXACT_PAIR_YEAR_CELL_SINGLE_FILE_WITH_BOUNDED_TERMINAL_SUPERSET_V1"
        ),
        "acquisition_identity": {
            "contract": "OANDA_V20_INSTRUMENT_CANDLES_GET_V1",
            "method": "GET",
            "path_template": "/v3/instruments/{pair}/candles",
            "price_component": PRICE_COMPONENT,
            "include_first": True,
            "approved_fetchers": {
                granularity: {
                    "path": str(path),
                    "sha256": digest,
                }
                for granularity, (path, digest) in approved_fetchers.items()
            },
            "method_path_bound_by_fetch_script_sha256": True,
            "base_url_receipted": False,
            "response_top_level_instrument_receipted": False,
        },
        "row_validation": {
            "strict_json": True,
            "complete_bid_ask_ohlc_only": True,
            "synthetic_rows_allowed": False,
            "strictly_increasing_unique_timestamps": True,
            "minimum_observed_slot_coverage": MIN_OBSERVED_SLOT_COVERAGE,
            "maximum_boundary_gap_seconds": MAX_BOUNDARY_GAP_SECONDS,
            "missing_slot_legitimacy_proved": False,
            "calendar_open_quote_coverage_proved": False,
        },
        "binding_count": len(bindings),
        "bindings": bindings,
        "physical_shard_count": len(physical),
        "physical_shards": physical,
        "physical_shard_ids_sha256": canonical_sha256(
            [row["physical_shard_id"] for row in physical]
        ),
        "duplicate_equivalence_record_count": len(duplicate_equivalence_records),
        "duplicate_equivalence_records": duplicate_equivalence_records,
        "duplicate_equivalence_records_sha256": canonical_sha256(
            duplicate_equivalence_records
        ),
        "plan_digest_inputs": {
            "source_digests": source_digests,
            "corpus_digests": corpus_digests,
        },
        "summary_only_admission_allowed": False,
        "raw_rows_embedded": False,
        "authority": _authority(),
    }
    return {**body, "source_manifest_sha256": canonical_sha256(body)}


def build_long_horizon_source_manifest(
    *, m5_root: Path, m1_root: Path
) -> dict[str, Any]:
    """Deep-scan and seal the exact fixed three-binding source denominator."""

    manifest = _build_from_specs(
        m5_root=m5_root,
        m1_root=m1_root,
        specs=_fixed_specs(),
    )
    verify_long_horizon_source_manifest_seal(manifest)
    return manifest


def _validate_structure(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "contract",
        "schema_version",
        "classification",
        "source_roots",
        "study_period",
        "selection_policy",
        "acquisition_identity",
        "row_validation",
        "binding_count",
        "bindings",
        "physical_shard_count",
        "physical_shards",
        "physical_shard_ids_sha256",
        "duplicate_equivalence_record_count",
        "duplicate_equivalence_records",
        "duplicate_equivalence_records_sha256",
        "plan_digest_inputs",
        "summary_only_admission_allowed",
        "raw_rows_embedded",
        "authority",
        "source_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DojoLongHorizonSourceManifestError(
            "source manifest top-level schema is not exact"
        )
    if (
        value["contract"] != CONTRACT
        or value["schema_version"] != SCHEMA_VERSION
        or value["classification"] != CLASSIFICATION
        or value["authority"] != _authority()
        or value["summary_only_admission_allowed"] is not False
        or value["raw_rows_embedded"] is not False
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest authority or contract drifted"
        )
    unsigned = {
        key: item for key, item in value.items() if key != "source_manifest_sha256"
    }
    if value["source_manifest_sha256"] != canonical_sha256(unsigned):
        raise DojoLongHorizonSourceManifestError("source manifest digest drifted")


def _parse_manifest_utc(value: Any, *, field: str) -> datetime:
    try:
        return _history._parse_aware_utc(value)
    except _history.HistoricalS5CacheError as exc:
        raise DojoLongHorizonSourceManifestError(
            f"source manifest {field} clock is invalid"
        ) from exc


def _require_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoLongHorizonSourceManifestError(
            f"source manifest {field} digest is invalid"
        )
    return value


def _verify_seal_for_specs(
    value: Mapping[str, Any],
    *,
    specs: Sequence[_BindingSpec],
) -> dict[str, Any]:
    """Purely verify every compact seal and fixed relation without raw I/O."""

    _validate_structure(value)
    roots = value["source_roots"]
    if not isinstance(roots, Mapping) or set(roots) != {"M5", "M1"}:
        raise DojoLongHorizonSourceManifestError("source root mapping is not exact")
    for root in roots.values():
        root_path = Path(str(root))
        if not root_path.is_absolute() or ".." in root_path.parts:
            raise DojoLongHorizonSourceManifestError(
                "sealed source root syntax is invalid"
            )
    if value["study_period"] != {
        "from_utc": PERIOD_FROM_UTC,
        "to_utc": PERIOD_TO_UTC,
        "half_open": True,
    }:
        raise DojoLongHorizonSourceManifestError("source manifest study period differs")
    if value["selection_policy"] != (
        "EXACT_PAIR_YEAR_CELL_SINGLE_FILE_WITH_BOUNDED_TERMINAL_SUPERSET_V1"
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest selection policy differs"
        )
    if value["row_validation"] != {
        "strict_json": True,
        "complete_bid_ask_ohlc_only": True,
        "synthetic_rows_allowed": False,
        "strictly_increasing_unique_timestamps": True,
        "minimum_observed_slot_coverage": MIN_OBSERVED_SLOT_COVERAGE,
        "maximum_boundary_gap_seconds": MAX_BOUNDARY_GAP_SECONDS,
        "missing_slot_legitimacy_proved": False,
        "calendar_open_quote_coverage_proved": False,
    }:
        raise DojoLongHorizonSourceManifestError(
            "source manifest row-validation policy differs"
        )
    acquisition = value["acquisition_identity"]
    acquisition_keys = {
        "contract",
        "method",
        "path_template",
        "price_component",
        "include_first",
        "approved_fetchers",
        "method_path_bound_by_fetch_script_sha256",
        "base_url_receipted",
        "response_top_level_instrument_receipted",
    }
    if (
        not isinstance(acquisition, Mapping)
        or set(acquisition) != acquisition_keys
        or acquisition["contract"] != "OANDA_V20_INSTRUMENT_CANDLES_GET_V1"
        or acquisition["method"] != "GET"
        or acquisition["path_template"] != "/v3/instruments/{pair}/candles"
        or acquisition["price_component"] != PRICE_COMPONENT
        or acquisition["include_first"] is not True
        or acquisition["method_path_bound_by_fetch_script_sha256"] is not True
        or acquisition["base_url_receipted"] is not False
        or acquisition["response_top_level_instrument_receipted"] is not False
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest acquisition identity differs"
        )
    fetchers = acquisition["approved_fetchers"]
    if not isinstance(fetchers, Mapping) or set(fetchers) != {"M1", "M5"}:
        raise DojoLongHorizonSourceManifestError(
            "source manifest approved-fetcher mapping differs"
        )
    expected_fetcher_names = {
        "M1": "oanda_history_fetch_m1.py",
        "M5": "oanda_history_fetch.py",
    }
    for granularity, expected_name in expected_fetcher_names.items():
        fetcher = fetchers[granularity]
        if not isinstance(fetcher, Mapping) or set(fetcher) != {"path", "sha256"}:
            raise DojoLongHorizonSourceManifestError(
                "source manifest approved-fetcher schema differs"
            )
        fetcher_path = Path(str(fetcher["path"]))
        if (
            not fetcher_path.is_absolute()
            or ".." in fetcher_path.parts
            or fetcher_path.name != expected_name
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest approved-fetcher path differs"
            )
        _require_sha(fetcher["sha256"], field="approved fetcher")

    expected_shards = {
        shard.cell_key: shard for spec in specs for shard in _spec_shards(spec)
    }
    physical = value["physical_shards"]
    if (
        not isinstance(physical, list)
        or value["physical_shard_count"] != len(expected_shards)
        or len(physical) != len(expected_shards)
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest physical-shard denominator differs"
        )
    expected_physical_keys = {
        "physical_shard_id",
        "root_kind",
        "relative_path",
        "file_size_bytes",
        "file_sha256",
        "full_uncompressed_bytes_sha256",
        "declared_from_utc",
        "declared_to_utc",
        "full_file_row_count",
        "full_first_observed_utc",
        "full_last_observed_utc",
        "acquisition_receipt_sha256",
        "acquisition_recorded_at_utc",
        "summary_relative_path",
        "summary_sha256",
        "max_candles_per_request",
        "request_count",
        "window_count",
        "request_window_completion_report_proved",
        "granularity",
        "pair",
        "slice_from_utc",
        "slice_to_utc",
        "slice_row_count",
        "slice_first_observed_utc",
        "slice_last_observed_utc",
        "slice_uncompressed_bytes_sha256",
        "cadence_seconds",
        "grid_slot_count",
        "observed_slot_coverage",
        "leading_gap_seconds",
        "trailing_gap_seconds",
        "synthetic_row_count",
        "duplicate_timestamp_count",
        "pair_month_coverage_data_sha256",
        "terminal_file_superset_seconds",
        "source_shard_sha256",
        "corpus_shard_sha256",
        "pair_month_coverage",
        "pair_month_coverage_sha256",
    }
    month_keys = {
        "pair",
        "month",
        "from_utc",
        "to_utc",
        "row_count",
        "grid_slot_count",
        "no_candle_slot_count",
        "first_observed_utc",
        "last_observed_utc",
        "leading_gap_seconds",
        "trailing_gap_seconds",
        "internal_gap_interval_count",
        "max_internal_gap_seconds",
        "gap_intervals_sha256",
        "uncompressed_row_bytes_sha256",
        "request_window_completion_report_proved",
        "missing_slot_legitimacy_proved",
        "calendar_open_quote_coverage_proved",
        "physical_shard_id",
        "coverage_cell_sha256",
    }
    scans_by_cell: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    physical_ids: list[str] = []
    sort_keys: list[tuple[str, str, str]] = []
    for row in physical:
        if not isinstance(row, Mapping) or set(row) != expected_physical_keys:
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard schema is not exact"
            )
        pair = row["pair"]
        granularity = row["granularity"]
        if (
            not isinstance(pair, str)
            or _PAIR_RE.fullmatch(pair) is None
            or granularity not in {"M1", "M5"}
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard market scope is invalid"
            )
        slice_from = _parse_manifest_utc(row["slice_from_utc"], field="slice from")
        slice_to = _parse_manifest_utc(row["slice_to_utc"], field="slice to")
        cell = (granularity, pair, slice_from.year)
        shard = expected_shards.get(cell)
        if (
            shard is None
            or row["root_kind"] != granularity
            or slice_from != shard.slice_from
            or slice_to != shard.slice_to
            or row["cadence_seconds"] != shard.cadence_seconds
            or cell in scans_by_cell
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard cell differs or duplicates"
            )
        relative_path = Path(str(row["relative_path"]))
        summary_relative = Path(str(row["summary_relative_path"]))
        cache_match = _CACHE_RE.fullmatch(relative_path.name)
        if (
            relative_path.is_absolute()
            or summary_relative.is_absolute()
            or ".." in relative_path.parts
            or ".." in summary_relative.parts
            or cache_match is None
            or cache_match.group("pair") != pair
            or cache_match.group("granularity") != granularity
            or summary_relative != relative_path.parents[1] / "summary.json"
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest relative source path is invalid"
            )
        declared_from = _parse_manifest_utc(
            row["declared_from_utc"], field="declared from"
        )
        declared_to = _parse_manifest_utc(row["declared_to_utc"], field="declared to")
        acquisition_recorded_at = _parse_manifest_utc(
            row["acquisition_recorded_at_utc"],
            field="acquisition recorded at",
        )
        full_first = _parse_manifest_utc(
            row["full_first_observed_utc"], field="full first"
        )
        full_last = _parse_manifest_utc(
            row["full_last_observed_utc"], field="full last"
        )
        slice_first = _parse_manifest_utc(
            row["slice_first_observed_utc"], field="slice first"
        )
        slice_last = _parse_manifest_utc(
            row["slice_last_observed_utc"], field="slice last"
        )
        terminal_superset = int((declared_to - slice_to).total_seconds())
        if (
            _parse_stamp(cache_match.group("start")) != declared_from
            or _parse_stamp(cache_match.group("end")) != declared_to
            or declared_from != slice_from
            or acquisition_recorded_at < declared_to
            or not declared_from <= full_first <= full_last < declared_to
            or not slice_from <= slice_first <= slice_last < slice_to
            or terminal_superset != row["terminal_file_superset_seconds"]
            or not 0 <= terminal_superset <= MAX_TERMINAL_SUPERSET_SECONDS
            or (slice_to.month == 1 and slice_to.day == 1 and terminal_superset)
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical/shard interval relation differs"
            )
        int_fields = (
            "file_size_bytes",
            "full_file_row_count",
            "max_candles_per_request",
            "request_count",
            "window_count",
            "slice_row_count",
            "grid_slot_count",
            "leading_gap_seconds",
            "trailing_gap_seconds",
            "synthetic_row_count",
            "duplicate_timestamp_count",
            "terminal_file_superset_seconds",
        )
        if any(row[field].__class__ is not int for field in int_fields):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard counter type is invalid"
            )
        expected_grid = int(
            (slice_to - slice_from).total_seconds() // shard.cadence_seconds
        )
        full_grid = int(
            (declared_to - declared_from).total_seconds() // shard.cadence_seconds
        )
        expected_requests = _request_window_count(
            start=declared_from,
            end=declared_to,
            cadence_seconds=shard.cadence_seconds,
            max_candles=row["max_candles_per_request"],
        )
        expected_leading = int((slice_first - slice_from).total_seconds())
        expected_trailing = max(
            0,
            int(
                (
                    slice_to - (slice_last + timedelta(seconds=shard.cadence_seconds))
                ).total_seconds()
            ),
        )
        if (
            row["file_size_bytes"] <= 0
            or row["full_file_row_count"] <= 0
            or row["full_file_row_count"] > full_grid
            or row["full_file_row_count"] < row["slice_row_count"]
            or row["slice_row_count"] <= 0
            or row["grid_slot_count"] != expected_grid
            or row["slice_row_count"] > expected_grid
            or row["observed_slot_coverage"] != row["slice_row_count"] / expected_grid
            or row["observed_slot_coverage"] < MIN_OBSERVED_SLOT_COVERAGE
            or row["leading_gap_seconds"] != expected_leading
            or row["trailing_gap_seconds"] != expected_trailing
            or expected_leading > MAX_BOUNDARY_GAP_SECONDS
            or expected_trailing > MAX_BOUNDARY_GAP_SECONDS
            or row["max_candles_per_request"] < 1
            or row["max_candles_per_request"] > 5_000
            or row["request_count"] < 1
            or row["request_count"] != expected_requests
            or row["window_count"] != row["request_count"]
            or row["request_window_completion_report_proved"] is not True
            or row["synthetic_row_count"] != 0
            or row["duplicate_timestamp_count"] != 0
            or row["observed_slot_coverage"].__class__ is not float
            or not math.isfinite(row["observed_slot_coverage"])
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard counters differ"
            )
        for field in (
            "file_sha256",
            "full_uncompressed_bytes_sha256",
            "acquisition_receipt_sha256",
            "summary_sha256",
            "slice_uncompressed_bytes_sha256",
            "pair_month_coverage_data_sha256",
            "source_shard_sha256",
            "corpus_shard_sha256",
            "physical_shard_id",
            "pair_month_coverage_sha256",
        ):
            _require_sha(row[field], field=field)

        coverage_rows = row["pair_month_coverage"]
        expected_months = []
        cursor = slice_from
        while cursor < slice_to:
            expected_months.append(cursor.strftime("%Y-%m"))
            cursor = (
                datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
                if cursor.month == 12
                else datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
            )
        if (
            not isinstance(coverage_rows, list)
            or [
                item.get("month") for item in coverage_rows if isinstance(item, Mapping)
            ]
            != expected_months
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest pair-month coverage denominator differs"
            )
        unparented_month_rows: list[dict[str, Any]] = []
        monthly_count = 0
        for coverage in coverage_rows:
            if not isinstance(coverage, Mapping) or set(coverage) != month_keys:
                raise DojoLongHorizonSourceManifestError(
                    "source manifest pair-month coverage schema is not exact"
                )
            month_from = _parse_manifest_utc(coverage["from_utc"], field="month from")
            month_to = _parse_manifest_utc(coverage["to_utc"], field="month to")
            month_first = _parse_manifest_utc(
                coverage["first_observed_utc"], field="month first"
            )
            month_last = _parse_manifest_utc(
                coverage["last_observed_utc"], field="month last"
            )
            month = coverage["month"]
            expected_month_from = datetime(
                int(month[:4]), int(month[5:]), 1, tzinfo=timezone.utc
            )
            expected_month_to = (
                datetime(expected_month_from.year + 1, 1, 1, tzinfo=timezone.utc)
                if expected_month_from.month == 12
                else datetime(
                    expected_month_from.year,
                    expected_month_from.month + 1,
                    1,
                    tzinfo=timezone.utc,
                )
            )
            month_int_fields = (
                "row_count",
                "grid_slot_count",
                "no_candle_slot_count",
                "leading_gap_seconds",
                "trailing_gap_seconds",
                "internal_gap_interval_count",
                "max_internal_gap_seconds",
            )
            if any(coverage[field].__class__ is not int for field in month_int_fields):
                raise DojoLongHorizonSourceManifestError(
                    "source manifest pair-month counter type is invalid"
                )
            month_grid = int(
                (expected_month_to - expected_month_from).total_seconds()
                // shard.cadence_seconds
            )
            month_leading = int((month_first - expected_month_from).total_seconds())
            month_trailing = max(
                0,
                int(
                    (
                        expected_month_to
                        - (month_last + timedelta(seconds=shard.cadence_seconds))
                    ).total_seconds()
                ),
            )
            if (
                coverage["pair"] != pair
                or month_from != expected_month_from
                or month_to != expected_month_to
                or not month_from <= month_first <= month_last < month_to
                or coverage["row_count"] <= 0
                or coverage["grid_slot_count"] != month_grid
                or coverage["row_count"] > month_grid
                or coverage["no_candle_slot_count"]
                != month_grid - coverage["row_count"]
                or coverage["leading_gap_seconds"] != month_leading
                or coverage["trailing_gap_seconds"] != month_trailing
                or month_leading > MAX_BOUNDARY_GAP_SECONDS
                or month_trailing > MAX_BOUNDARY_GAP_SECONDS
                or coverage["internal_gap_interval_count"] < 0
                or coverage["internal_gap_interval_count"]
                > coverage["no_candle_slot_count"]
                or coverage["max_internal_gap_seconds"] < 0
                or (
                    coverage["internal_gap_interval_count"] == 0
                    and coverage["max_internal_gap_seconds"] != 0
                )
                or (
                    coverage["internal_gap_interval_count"] == 0
                    and coverage["gap_intervals_sha256"] != canonical_sha256([])
                )
                or (
                    coverage["internal_gap_interval_count"] > 0
                    and coverage["max_internal_gap_seconds"] < shard.cadence_seconds
                )
                or coverage["request_window_completion_report_proved"] is not True
                or coverage["missing_slot_legitimacy_proved"] is not False
                or coverage["calendar_open_quote_coverage_proved"] is not False
                or coverage["physical_shard_id"] != row["physical_shard_id"]
            ):
                raise DojoLongHorizonSourceManifestError(
                    "source manifest pair-month coverage relation differs"
                )
            _require_sha(coverage["gap_intervals_sha256"], field="month gaps")
            _require_sha(coverage["uncompressed_row_bytes_sha256"], field="month rows")
            coverage_body = {
                key: item
                for key, item in coverage.items()
                if key != "coverage_cell_sha256"
            }
            if coverage["coverage_cell_sha256"] != canonical_sha256(coverage_body):
                raise DojoLongHorizonSourceManifestError(
                    "source manifest pair-month cell digest differs"
                )
            unparented_month_rows.append(
                {
                    key: item
                    for key, item in coverage_body.items()
                    if key != "physical_shard_id"
                }
            )
            monthly_count += coverage["row_count"]
        if (
            monthly_count != row["slice_row_count"]
            or row["pair_month_coverage_data_sha256"]
            != canonical_sha256(unparented_month_rows)
            or row["pair_month_coverage_sha256"] != canonical_sha256(coverage_rows)
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest pair-month aggregate digest differs"
            )

        source_body = {
            key: row[key]
            for key in (
                "root_kind",
                "relative_path",
                "file_size_bytes",
                "file_sha256",
                "full_uncompressed_bytes_sha256",
                "declared_from_utc",
                "declared_to_utc",
                "full_file_row_count",
                "full_first_observed_utc",
                "full_last_observed_utc",
                "acquisition_receipt_sha256",
                "acquisition_recorded_at_utc",
                "summary_relative_path",
                "summary_sha256",
                "max_candles_per_request",
                "request_count",
                "window_count",
                "request_window_completion_report_proved",
            )
        }
        corpus_body = {
            key: row[key]
            for key in (
                "granularity",
                "pair",
                "slice_from_utc",
                "slice_to_utc",
                "slice_row_count",
                "slice_first_observed_utc",
                "slice_last_observed_utc",
                "slice_uncompressed_bytes_sha256",
                "cadence_seconds",
                "grid_slot_count",
                "observed_slot_coverage",
                "leading_gap_seconds",
                "trailing_gap_seconds",
                "synthetic_row_count",
                "duplicate_timestamp_count",
                "pair_month_coverage_data_sha256",
            )
        }
        if row["source_shard_sha256"] != canonical_sha256(source_body) or row[
            "corpus_shard_sha256"
        ] != canonical_sha256(corpus_body):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical source/corpus digest differs"
            )
        body_without_id = {
            **source_body,
            **corpus_body,
            "terminal_file_superset_seconds": row["terminal_file_superset_seconds"],
            "source_shard_sha256": row["source_shard_sha256"],
            "corpus_shard_sha256": row["corpus_shard_sha256"],
        }
        if row["physical_shard_id"] != canonical_sha256(body_without_id):
            raise DojoLongHorizonSourceManifestError(
                "source manifest physical-shard identifier differs"
            )
        scans_by_cell[cell] = row
        physical_ids.append(row["physical_shard_id"])
        sort_keys.append((granularity, pair, row["slice_from_utc"]))
    if sort_keys != sorted(sort_keys) or value[
        "physical_shard_ids_sha256"
    ] != canonical_sha256(physical_ids):
        raise DojoLongHorizonSourceManifestError(
            "source manifest physical-shard order or aggregate digest differs"
        )

    duplicate_records = value["duplicate_equivalence_records"]
    if (
        not isinstance(duplicate_records, list)
        or value["duplicate_equivalence_record_count"] != len(duplicate_records)
        or value["duplicate_equivalence_records_sha256"]
        != canonical_sha256(duplicate_records)
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest duplicate-equivalence aggregate differs"
        )
    record_keys = {
        "equivalence_contract",
        "granularity",
        "pair",
        "slice_from_utc",
        "slice_to_utc",
        "candidate_count",
        "equality_fields",
        "candidates",
        "selected_relative_path",
        "selected_physical_shard_id",
        "selection_policy",
        "equivalence_proved",
        "duplicate_equivalence_sha256",
    }
    candidate_keys = {
        "relative_path",
        "run_id",
        "compressed_file_sha256",
        "full_uncompressed_bytes_sha256",
        "full_file_row_count",
        "full_first_observed_utc",
        "full_last_observed_utc",
        "declared_from_utc",
        "declared_to_utc",
        "acquisition_receipt_sha256",
        "acquisition_recorded_at_utc",
        "physical_shard_id",
    }
    equality_fields = [
        "declared_from_utc",
        "declared_to_utc",
        "full_uncompressed_bytes_sha256",
        "full_file_row_count",
        "full_first_observed_utc",
        "full_last_observed_utc",
    ]
    duplicate_cells: set[tuple[str, str, int]] = set()
    prior_record_key: tuple[str, str, str] | None = None
    for record in duplicate_records:
        if not isinstance(record, Mapping) or set(record) != record_keys:
            raise DojoLongHorizonSourceManifestError(
                "source manifest duplicate-equivalence schema is not exact"
            )
        record_body = {
            key: item
            for key, item in record.items()
            if key != "duplicate_equivalence_sha256"
        }
        if record["duplicate_equivalence_sha256"] != canonical_sha256(record_body):
            raise DojoLongHorizonSourceManifestError(
                "source manifest duplicate-equivalence digest differs"
            )
        granularity = record["granularity"]
        pair = record["pair"]
        slice_from = _parse_manifest_utc(
            record["slice_from_utc"], field="duplicate slice from"
        )
        slice_to = _parse_manifest_utc(
            record["slice_to_utc"], field="duplicate slice to"
        )
        cell = (granularity, pair, slice_from.year)
        selected_physical = scans_by_cell.get(cell)
        record_sort_key = (granularity, pair, record["slice_from_utc"])
        if (
            selected_physical is None
            or cell in duplicate_cells
            or prior_record_key is not None
            and record_sort_key <= prior_record_key
            or slice_from
            != _parse_manifest_utc(
                selected_physical["slice_from_utc"], field="selected slice from"
            )
            or slice_to
            != _parse_manifest_utc(
                selected_physical["slice_to_utc"], field="selected slice to"
            )
            or record["equivalence_contract"]
            != "FULL_UNCOMPRESSED_CANDLE_BYTES_TIME_ROWS_V1"
            or record["selection_policy"]
            != "UNIQUE_LATEST_ACQUISITION_RECORDED_AT_UTC_V1"
            or record["equivalence_proved"] is not True
            or record["equality_fields"] != equality_fields
            or record["selected_relative_path"] != selected_physical["relative_path"]
            or record["selected_physical_shard_id"]
            != selected_physical["physical_shard_id"]
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest duplicate-equivalence selection differs"
            )
        candidates = record["candidates"]
        if (
            not isinstance(candidates, list)
            or record["candidate_count"].__class__ is not int
            or record["candidate_count"] != len(candidates)
            or len(candidates) < 2
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest duplicate-equivalence denominator differs"
            )
        candidate_sort_keys: list[tuple[str, str]] = []
        candidate_paths: set[str] = set()
        reference: tuple[Any, ...] | None = None
        latest_clock: datetime | None = None
        latest_count = 0
        selected_candidate: Mapping[str, Any] | None = None
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or set(candidate) != candidate_keys:
                raise DojoLongHorizonSourceManifestError(
                    "source manifest duplicate candidate schema is not exact"
                )
            relative = Path(str(candidate["relative_path"]))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 3
                or candidate["run_id"] != relative.parts[0]
                or _RUN_RE.fullmatch(candidate["run_id"]) is None
                or candidate["relative_path"] in candidate_paths
                or candidate["full_file_row_count"].__class__ is not int
                or candidate["full_file_row_count"] <= 0
            ):
                raise DojoLongHorizonSourceManifestError(
                    "source manifest duplicate candidate path/counter differs"
                )
            for field in (
                "compressed_file_sha256",
                "full_uncompressed_bytes_sha256",
                "acquisition_receipt_sha256",
                "physical_shard_id",
            ):
                _require_sha(candidate[field], field=f"duplicate {field}")
            recorded_at = _parse_manifest_utc(
                candidate["acquisition_recorded_at_utc"],
                field="duplicate recorded at",
            )
            candidate_reference = tuple(candidate[field] for field in equality_fields)
            if reference is None:
                reference = candidate_reference
            elif candidate_reference != reference:
                raise DojoLongHorizonSourceManifestError(
                    "source manifest duplicate candidate equivalence differs"
                )
            if latest_clock is None or recorded_at > latest_clock:
                latest_clock = recorded_at
                latest_count = 1
            elif recorded_at == latest_clock:
                latest_count += 1
            if candidate["relative_path"] == record["selected_relative_path"]:
                selected_candidate = candidate
            candidate_sort_keys.append(
                (candidate["acquisition_recorded_at_utc"], candidate["relative_path"])
            )
            candidate_paths.add(candidate["relative_path"])
        if (
            candidate_sort_keys != sorted(candidate_sort_keys)
            or latest_count != 1
            or selected_candidate is None
            or _parse_manifest_utc(
                selected_candidate["acquisition_recorded_at_utc"],
                field="selected duplicate recorded at",
            )
            != latest_clock
            or selected_candidate["physical_shard_id"]
            != selected_physical["physical_shard_id"]
        ):
            raise DojoLongHorizonSourceManifestError(
                "source manifest duplicate unique-latest selection differs"
            )
        duplicate_cells.add(cell)
        prior_record_key = record_sort_key

    bindings = value["bindings"]
    expected_bindings = [
        _binding_row(spec, scans_by_cell=scans_by_cell) for spec in specs
    ]
    if (
        not isinstance(bindings, list)
        or value["binding_count"] != len(specs)
        or bindings != expected_bindings
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest binding seals or denominator differ"
        )
    expected_inputs = {
        "source_digests": {
            row["binding_id"]: row["source_digest_sha256"] for row in bindings
        },
        "corpus_digests": {
            row["binding_id"]: row["corpus_digest_sha256"] for row in bindings
        },
    }
    if value["plan_digest_inputs"] != expected_inputs:
        raise DojoLongHorizonSourceManifestError(
            "source manifest plan digest inputs differ"
        )
    return dict(value)


def verify_long_horizon_source_manifest_seal(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the fixed compact parent/slice seal without touching raw roots."""

    return _verify_seal_for_specs(value, specs=_fixed_specs())


def validate_long_horizon_source_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild from raw roots; summaries or self-rehashed metadata never suffice."""

    verify_long_horizon_source_manifest_seal(value)
    roots = value["source_roots"]
    if not isinstance(roots, Mapping) or set(roots) != {"M5", "M1"}:
        raise DojoLongHorizonSourceManifestError("source root mapping is not exact")
    rebuilt = build_long_horizon_source_manifest(
        m5_root=Path(str(roots["M5"])),
        m1_root=Path(str(roots["M1"])),
    )
    if dict(value) != rebuilt:
        raise DojoLongHorizonSourceManifestError(
            "source manifest differs from a fresh raw-shard scan"
        )
    return rebuilt


def long_horizon_plan_digest_inputs(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Return exact source/corpus mappings accepted by the long-horizon plan."""

    verified = verify_long_horizon_source_manifest_seal(manifest)
    inputs = verified["plan_digest_inputs"]
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != {"source_digests", "corpus_digests"}
        or set(inputs["source_digests"]) != set(SOURCE_BINDING_IDS)
        or set(inputs["corpus_digests"]) != set(SOURCE_BINDING_IDS)
    ):
        raise DojoLongHorizonSourceManifestError(
            "source manifest does not expose the exact plan digest inputs"
        )
    return {
        "source_digests": dict(inputs["source_digests"]),
        "corpus_digests": dict(inputs["corpus_digests"]),
    }


def write_long_horizon_source_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Publish one canonical manifest without overwriting prior evidence."""

    verify_long_horizon_source_manifest_seal(manifest)
    _write_canonical_manifest(path, manifest)


def _write_canonical_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Crash-safe append-only publication primitive for a verified seal."""

    destination = Path(path)
    if not destination.is_absolute() or ".." in destination.parts:
        raise DojoLongHorizonSourceManifestError(
            "manifest output must be an absolute path without traversal"
        )
    parent = destination.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise DojoLongHorizonSourceManifestError(
            "manifest output parent must be a real non-symlink directory"
        )
    for source_root in manifest["source_roots"].values():
        try:
            destination.relative_to(Path(str(source_root)))
        except ValueError:
            continue
        raise DojoLongHorizonSourceManifestError(
            "manifest output must be outside immutable source roots"
        )
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    temporary: Path | None = None
    fd: int | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-",
            dir=parent,
        )
        temporary = Path(temporary_name)
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise DojoLongHorizonSourceManifestError(
                    "manifest output write was short"
                )
            written += count
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise DojoLongHorizonSourceManifestError(
                "manifest output already exists; evidence is append-only"
            ) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = [
    "CONTRACT",
    "DojoLongHorizonSourceManifestError",
    "build_long_horizon_source_manifest",
    "long_horizon_plan_digest_inputs",
    "validate_long_horizon_source_manifest",
    "verify_long_horizon_source_manifest_seal",
    "write_long_horizon_source_manifest",
]
