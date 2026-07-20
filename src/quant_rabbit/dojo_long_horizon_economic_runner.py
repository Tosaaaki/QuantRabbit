"""Single-stream economic runner for sealed DOJO long-horizon jobs.

The execution state machine intentionally does not know portfolio economics.
This module is the typed bridge from one of its runner handoffs to the shared
worker protocol and :class:`PortfolioReplaySession`.  One canonical historical
bid/ask candle shard is opened once, read in causal order, and fanned out to
every runnable account coordinate behind a batch barrier.  A worker is never
called for a batch until every still-live coordinate has received its exact
post-exit snapshot.

The module has no broker or live path.  Worker output is proposal-only and is
sealed against a reducer-owned, recursively read-only snapshot.  All fills,
costs, allocation, margin, conversion and terminal economics are computed by
the shared reducer.  Partial P/L is never returned as a terminal cell.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

from quant_rabbit.dojo_long_horizon_execution import (
    RUNNER_HANDOFF_CONTRACT,
    build_long_horizon_coordinate_result,
)
from quant_rabbit.dojo_economic_transcript import (
    DojoEconomicTranscriptError,
    EconomicTranscriptRecorder,
    build_economic_transcript_header,
    build_fixed_denominator_reexecution_attestation,
)
from quant_rabbit.dojo_builtin_strategy_runtime import (
    builtin_strategy_runtime_factory,
    verify_builtin_strategy_runtime_seal,
)
from quant_rabbit.dojo_tuned_strategy_runtime import (
    SealedTunedStrategyRuntimeFactory,
    verify_tuned_strategy_runtime_seal,
)
from quant_rabbit.dojo_long_horizon_plan import (
    STARTING_EQUITY_JPY,
    canonical_sha256,
    validate_long_horizon_train_plan,
)
from quant_rabbit.dojo_long_horizon_source_manifest import (
    verify_long_horizon_source_manifest_seal,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    quote_batch_sha256,
    validate_portfolio_replay_result,
    verify_portfolio_policy,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    readonly_post_exit_snapshot,
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


SOURCE_SLICE_CONTRACT: Final = "QR_DOJO_SYNCHRONIZED_SOURCE_SLICE_V1"
ECONOMIC_CARRY_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_ECONOMIC_CARRY_V1"
ECONOMIC_JOB_RESULT_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_ECONOMIC_JOB_RESULT_V1"
SCHEMA_VERSION: Final = 1
MAX_SOURCE_LINE_BYTES: Final = 1024 * 1024
MAX_WORKER_STATE_BYTES: Final = 1024 * 1024
MAX_RESULT_BYTES: Final = 16 * 1024 * 1024
MAX_REEXECUTION_ATTESTATION_BYTES: Final = 1024 * 1024
GENESIS_BATCH_CHAIN_SHA256: Final = "0" * 64
BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256: Final = canonical_sha256(
    {
        "contract": "QR_DOJO_BUILTIN_NO_INTENT_WORKER_RUNTIME_V1",
        "proposal_policy": "EXACTLY_ONE_EMPTY_PROPOSAL_PER_ACTIVE_WORKER",
        "external_code_loading_allowed": False,
    }
)

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_RE = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")
_GRANULARITY_SECONDS = {"M1": 60, "M5": 300}
_PHASES = {"OHLC": ("O", "H", "L", "C"), "OLHC": ("O", "L", "H", "C")}
_SOURCE_ROW_KEYS = frozenset({"complete", "epoch", "granularity", "quotes"})
_SOURCE_QUOTE_KEYS = frozenset({"ask", "bid", "pair"})
_RUNTIME_KEYS = frozenset(
    {
        "coordinate_id",
        "cost_scenario",
        "trade_pairs",
        "portfolio_policy",
        "cost_policy_sha256",
        "risk_policy_sha256",
        "replay_engine_sha256",
        "portfolio_policy_binding_sha256",
    }
)
_CATALOG_KEYS = frozenset({"worker_id", "owner_id", "family_id", "config_sha256"})


class DojoLongHorizonEconomicRunnerError(ValueError):
    """The economic job is unsealed, incomplete, non-causal or inconsistent."""


class EconomicWorkerRuntime(Protocol):
    """One coordinate-isolated strategy runtime.

    Implementations receive no allocator or broker handle.  ``propose`` must
    return raw proposal mappings for every active worker; HOLD/NO_INTENT is
    represented by empty intent arrays.  ``export_state`` must be strict JSON
    and is sealed into continuous-account carry.
    """

    def propose(self, snapshot: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...

    def export_state(self) -> Any: ...


WorkerRuntimeFactory = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, str]], Any | None],
    EconomicWorkerRuntime,
]


class _BuiltinNoIntentRuntime:
    def __init__(
        self, bindings: Sequence[Mapping[str, str]], prior_state: Any | None
    ) -> None:
        self._bindings = [dict(row) for row in bindings]
        self._calls = (
            int(prior_state.get("calls", 0)) if isinstance(prior_state, Mapping) else 0
        )

    def propose(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._calls += 1
        return [
            {
                **binding,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "risk_reducing_intents": [],
                "new_risk_intents": [],
            }
            for binding in self._bindings
        ]

    def export_state(self) -> dict[str, int]:
        return {"calls": self._calls}


def builtin_no_intent_runtime_factory(
    _coordinate: Mapping[str, Any],
    bindings: Sequence[Mapping[str, str]],
    prior_state: Any | None,
) -> EconomicWorkerRuntime:
    """Capability-closed HOLD baseline; it loads and calls no external code."""

    return _BuiltinNoIntentRuntime(bindings, prior_state)


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
        "three_x_guaranteed": False,
    }


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoLongHorizonEconomicRunnerError(
            "value is not strict canonical JSON"
        ) from exc


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be an object")
    return value


def _exact(
    value: Any, keys: frozenset[str] | set[str], *, field: str
) -> Mapping[str, Any]:
    row = _mapping(value, field=field)
    if set(row) != set(keys):
        raise DojoLongHorizonEconomicRunnerError(
            f"{field} schema mismatch: missing={sorted(set(keys)-set(row))}, "
            f"extra={sorted(set(row)-set(keys))}"
        )
    return row


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be a sequence")
    return value


def _sha(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be a SHA-256")
    if not allow_zero and value == GENESIS_BATCH_CHAIN_SHA256:
        raise DojoLongHorizonEconomicRunnerError(f"{field} cannot be all-zero")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoLongHorizonEconomicRunnerError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be finite and > 0")
    return result


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DojoLongHorizonEconomicRunnerError(
            f"{field} must be a non-empty trimmed string"
        )
    return value


def _economic_evidence_directory(path: Path) -> Path:
    directory = Path(path)
    state = directory.stat(follow_symlinks=False)
    if directory.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise DojoLongHorizonEconomicRunnerError(
            "economic evidence root must be an existing non-symlink directory"
        )
    return directory.resolve(strict=True)


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Crash-safe immutable publication for a bounded attestation."""

    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > MAX_REEXECUTION_ATTESTATION_BYTES:
        raise DojoLongHorizonEconomicRunnerError(
            "reexecution attestation exceeds its byte bound"
        )
    temporary = path.with_name(
        f".{path.name}.{canonical_sha256(value)}.{os.getpid()}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(payload) != len(payload):
                raise DojoLongHorizonEconomicRunnerError(
                    "reexecution attestation write was incomplete"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reexecute_transcript_in_separate_process(path: Path) -> dict[str, Any]:
    """Run the canonical auditor outside the runner process and parse only stdout."""

    repository_root = Path(__file__).resolve().parents[2]
    source_root = repository_root / "src"
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(source_root)
    if inherited_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, inherited_pythonpath))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_rabbit.dojo_economic_transcript",
            str(path),
        ],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
        capture_output=True,
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout
        or len(completed.stdout) > MAX_REEXECUTION_ATTESTATION_BYTES
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "separate-process economic transcript reexecution failed"
        )
    row = _strict_json_line(
        completed.stdout,
        field=f"reexecution attestation for {path.name}",
    )
    return dict(row)


def _failure_stage(code: str) -> str:
    return {
        "WORKER_RUNTIME_INITIALIZATION_FAILURE": "RUNTIME_INITIALIZATION",
        "PORTFOLIO_PREPARE_FAILURE": "POST_EXIT_PREPARATION",
        "WORKER_PROTOCOL_FAILURE": "PROPOSAL_COLLECTION",
        "PORTFOLIO_CONSUME_FAILURE": "ALLOCATION_REDUCTION",
        "SOURCE_STREAM_FAILURE": "SOURCE_STREAM",
        "SOURCE_QUOTE_COVERAGE_UNPROVEN": "SOURCE_COVERAGE",
        "PORTFOLIO_FINALIZE_FAILURE": "TERMINAL_SETTLEMENT",
    }.get(code, "UNKNOWN_FAILURE_STAGE")


def _transcript_call(action: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        return action(*args, **kwargs)
    except DojoEconomicTranscriptError as exc:
        raise DojoLongHorizonEconomicRunnerError(
            "economic transcript recording failed closed"
        ) from exc


def _strict_json_line(raw: bytes, *, field: str) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > MAX_SOURCE_LINE_BYTES:
        raise DojoLongHorizonEconomicRunnerError(
            f"{field} must be a bounded newline-terminated JSON row"
        )

    def reject_constant(token: str) -> None:
        raise DojoLongHorizonEconomicRunnerError(
            f"non-finite JSON token is forbidden: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoLongHorizonEconomicRunnerError(
                    f"duplicate JSON key in {field}: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoLongHorizonEconomicRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLongHorizonEconomicRunnerError(
            f"strict JSON parse failed: {field}"
        ) from exc
    row = _mapping(value, field=field)
    if raw != _canonical_bytes(row) + b"\n":
        raise DojoLongHorizonEconomicRunnerError(f"{field} is not canonical JSONL")
    return row


def _epoch_from_utc(value: Any, *, field: str) -> int:
    text = _identifier(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoLongHorizonEconomicRunnerError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DojoLongHorizonEconomicRunnerError(f"{field} must be timezone-aware")
    return int(parsed.astimezone(timezone.utc).timestamp())


def _validate_source_row(
    value: Any,
    *,
    job: Mapping[str, Any],
    previous_epoch: int | None,
) -> dict[str, Any]:
    row = _exact(value, _SOURCE_ROW_KEYS, field="source row")
    epoch = _integer(row["epoch"], field="source row.epoch")
    if row["complete"] is not True or row["granularity"] != job["granularity"]:
        raise DojoLongHorizonEconomicRunnerError(
            "source row is not a complete candle at the sealed granularity"
        )
    lower = _epoch_from_utc(job["from_utc"], field="job.from_utc")
    upper = _epoch_from_utc(job["to_utc"], field="job.to_utc")
    cadence = _GRANULARITY_SECONDS.get(job["granularity"])
    if cadence is None or epoch < lower or epoch >= upper or epoch % cadence != 0:
        raise DojoLongHorizonEconomicRunnerError(
            "source row epoch is outside or off-grid for the sealed month"
        )
    if previous_epoch is not None and (
        epoch <= previous_epoch or (epoch - previous_epoch) % cadence != 0
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "source epochs are not strictly increasing on the sealed cadence"
        )
    quote_rows = _sequence(row["quotes"], field="source row.quotes")
    quotes: list[dict[str, Any]] = []
    for index, raw_quote in enumerate(quote_rows):
        quote = _exact(
            raw_quote, _SOURCE_QUOTE_KEYS, field=f"source row.quotes[{index}]"
        )
        pair = _identifier(quote["pair"], field=f"source row.quotes[{index}].pair")
        if _PAIR_RE.fullmatch(pair) is None:
            raise DojoLongHorizonEconomicRunnerError("source quote pair is invalid")
        sides: dict[str, list[float]] = {}
        for side in ("bid", "ask"):
            values = _sequence(quote[side], field=f"source row.quotes[{index}].{side}")
            if len(values) != 4:
                raise DojoLongHorizonEconomicRunnerError(
                    "source bid/ask must be [O,H,L,C]"
                )
            parsed = [
                _positive(item, field=f"source row.quotes[{index}].{side}[{offset}]")
                for offset, item in enumerate(values)
            ]
            if (
                not parsed[2]
                <= min(parsed[0], parsed[3])
                <= max(parsed[0], parsed[3])
                <= parsed[1]
            ):
                raise DojoLongHorizonEconomicRunnerError(
                    "source OHLC geometry is invalid"
                )
            sides[side] = parsed
        if any(sides["bid"][offset] >= sides["ask"][offset] for offset in range(4)):
            raise DojoLongHorizonEconomicRunnerError(
                "source executable spread is crossed or zero"
            )
        quotes.append({"pair": pair, "bid": sides["bid"], "ask": sides["ask"]})
    quotes.sort(key=lambda item: item["pair"])
    if [item["pair"] for item in quotes] != list(job["feed_pairs"]):
        raise DojoLongHorizonEconomicRunnerError(
            "source row does not contain the exact ordered feed-pair denominator"
        )
    return {
        "complete": True,
        "epoch": epoch,
        "granularity": job["granularity"],
        "quotes": quotes,
    }


def _safe_source_path(source_root: Path, relative_path: Any) -> Path:
    root = source_root.resolve(strict=True)
    relative = Path(_identifier(relative_path, field="source relative_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise DojoLongHorizonEconomicRunnerError("source relative_path is unsafe")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise DojoLongHorizonEconomicRunnerError("source shard cannot be a symlink")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DojoLongHorizonEconomicRunnerError(
            "source shard escapes source root"
        ) from exc
    state = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(state.st_mode) or path.suffix != ".jsonl":
        raise DojoLongHorizonEconomicRunnerError(
            "source shard must be a regular canonical .jsonl file"
        )
    return path


_SOURCE_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "job_sha256",
        "price_stream_id",
        "source_binding_id",
        "source_digest_sha256",
        "corpus_digest_sha256",
        "month",
        "granularity",
        "from_utc",
        "to_utc",
        "feed_pairs",
        "source_manifest_sha256",
        "parent_binding_physical_shard_ids_sha256",
        "parent_month_coverage_cells",
        "parent_month_coverage_cells_sha256",
        "parent_quote_coverage_complete",
        "relative_path",
        "file_size_bytes",
        "file_sha256",
        "row_count",
        "first_epoch",
        "last_epoch",
        "complete_rows_only",
        "synthetic_quote_count",
        "source_open_count_per_economic_run",
        "normalized_price_derivation_verified",
        "normalized_price_derivation_sha256",
        "authority",
        "source_slice_receipt_sha256",
    }
)


def _source_parent_binding(
    source_manifest: Mapping[str, Any], *, job: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = verify_long_horizon_source_manifest_seal(source_manifest)
    manifest_sha = _sha(
        manifest.get("source_manifest_sha256"),
        field="source_manifest.source_manifest_sha256",
    )
    if manifest_sha != canonical_sha256(
        {key: item for key, item in manifest.items() if key != "source_manifest_sha256"}
    ):
        raise DojoLongHorizonEconomicRunnerError("source manifest seal is invalid")
    bindings = _sequence(manifest.get("bindings"), field="source manifest.bindings")
    matches = [
        row for row in bindings if row.get("binding_id") == job["source_binding_id"]
    ]
    if len(matches) != 1:
        raise DojoLongHorizonEconomicRunnerError(
            "source manifest has no unique job binding"
        )
    binding = _mapping(matches[0], field="source manifest binding")
    if (
        binding.get("source_digest_sha256") != job["source_digest_sha256"]
        or binding.get("corpus_digest_sha256") != job["corpus_digest_sha256"]
        or binding.get("granularity") != job["granularity"]
        or list(binding.get("pairs", [])) != list(job["feed_pairs"])
        or job["month"] not in binding.get("months", [])
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "source manifest binding differs from the sealed job"
        )
    physical_ids_sha = _sha(
        binding.get("physical_shard_ids_sha256"),
        field="source manifest binding.physical_shard_ids_sha256",
    )
    coverage = _sequence(
        binding.get("month_pair_coverage"),
        field="source manifest binding.month_pair_coverage",
    )
    parent_rows: list[dict[str, Any]] = []
    for pair in job["feed_pairs"]:
        rows = [
            row
            for row in coverage
            if row.get("pair") == pair and row.get("month") == job["month"]
        ]
        if len(rows) != 1:
            raise DojoLongHorizonEconomicRunnerError(
                "source manifest lacks exact pair-month coverage"
            )
        row = _mapping(rows[0], field="source pair-month coverage")
        coverage_sha = _sha(
            row.get("coverage_cell_sha256"), field="coverage_cell_sha256"
        )
        if coverage_sha != canonical_sha256(
            {key: item for key, item in row.items() if key != "coverage_cell_sha256"}
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "source pair-month coverage seal is invalid"
            )
        if (
            row.get("request_window_completion_report_proved") is not True
            or row.get("physical_shard_id") is None
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "source pair-month request completion is unproved"
            )
        parent_rows.append(
            {
                "pair": pair,
                "physical_shard_id": row["physical_shard_id"],
                "coverage_cell_sha256": coverage_sha,
                "row_count": row["row_count"],
                "first_observed_utc": row["first_observed_utc"],
                "last_observed_utc": row["last_observed_utc"],
                "missing_slot_legitimacy_proved": row.get(
                    "missing_slot_legitimacy_proved"
                ),
                "calendar_open_quote_coverage_proved": row.get(
                    "calendar_open_quote_coverage_proved"
                ),
            }
        )
    quote_coverage_complete = all(
        row["missing_slot_legitimacy_proved"] is True
        and row["calendar_open_quote_coverage_proved"] is True
        for row in parent_rows
    )
    return {
        "source_manifest_sha256": manifest_sha,
        "parent_binding_physical_shard_ids_sha256": physical_ids_sha,
        "parent_month_coverage_cells": parent_rows,
        "parent_month_coverage_cells_sha256": canonical_sha256(parent_rows),
        "parent_quote_coverage_complete": quote_coverage_complete,
    }


def _parse_parent_time(value: Any) -> int:
    text = _identifier(value, field="parent source time")
    if text.endswith("Z") and "." in text:
        head, _fraction = text[:-1].split(".", 1)
        text = head + "+00:00"
    return _epoch_from_utc(text, field="parent source time")


def _parent_price_rows(
    source_manifest: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, dict[int, dict[str, Any]]]:
    manifest = _mapping(source_manifest, field="source manifest")
    roots = _mapping(manifest.get("source_roots"), field="source manifest.source_roots")
    physical_rows = _sequence(
        manifest.get("physical_shards"), field="source manifest.physical_shards"
    )
    by_id = {row.get("physical_shard_id"): row for row in physical_rows}
    lower = _epoch_from_utc(job["from_utc"], field="job.from_utc")
    upper = _epoch_from_utc(job["to_utc"], field="job.to_utc")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for coverage in parent["parent_month_coverage_cells"]:
        pair = coverage["pair"]
        raw_physical = by_id.get(coverage["physical_shard_id"])
        if raw_physical is None:
            raise DojoLongHorizonEconomicRunnerError(
                "parent coverage references an unknown physical shard"
            )
        physical = _mapping(raw_physical, field="source physical shard")
        if (
            physical.get("pair") != pair
            or physical.get("granularity") != job["granularity"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "parent physical shard pair/granularity drifted"
            )
        root_kind = physical.get("root_kind")
        if root_kind not in roots:
            raise DojoLongHorizonEconomicRunnerError(
                "parent physical shard root is absent"
            )
        path = _safe_source_path_any_suffix(
            Path(str(roots[root_kind])), physical.get("relative_path")
        )
        expected_size = _integer(
            physical.get("file_size_bytes"),
            field="physical shard.file_size_bytes",
            minimum=1,
        )
        expected_sha = _sha(
            physical.get("file_sha256"), field="physical shard.file_sha256"
        )
        digest = hashlib.sha256()
        rows: dict[int, dict[str, Any]] = {}
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as raw_handle:
            opened = os.fstat(raw_handle.fileno())
            if opened.st_size != expected_size:
                raise DojoLongHorizonEconomicRunnerError(
                    "parent physical shard size drifted"
                )
            for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
                digest.update(chunk)
            raw_handle.seek(0)
            try:
                with gzip.GzipFile(fileobj=raw_handle, mode="rb") as stream:
                    for line_number, raw_line in enumerate(stream, 1):
                        raw = _strict_json_value(
                            raw_line,
                            field=f"parent source {pair}:{line_number}",
                        )
                        if (
                            raw.get("pair") != pair
                            or raw.get("granularity") != job["granularity"]
                            or raw.get("price") != "BA"
                            or raw.get("complete") is not True
                        ):
                            raise DojoLongHorizonEconomicRunnerError(
                                "parent source row provenance is invalid"
                            )
                        epoch = _parse_parent_time(raw.get("time"))
                        if not lower <= epoch < upper:
                            continue
                        if epoch in rows:
                            raise DojoLongHorizonEconomicRunnerError(
                                "parent source contains a duplicate month timestamp"
                            )
                        sides: dict[str, list[float]] = {}
                        for side in ("bid", "ask"):
                            block = _mapping(raw.get(side), field=f"parent {side}")
                            if set(block) != {"o", "h", "l", "c"}:
                                raise DojoLongHorizonEconomicRunnerError(
                                    "parent source OHLC schema is invalid"
                                )
                            sides[side] = [
                                _positive(block[key], field=f"parent {side}.{key}")
                                for key in ("o", "h", "l", "c")
                            ]
                        rows[epoch] = {"pair": pair, **sides}
            except (gzip.BadGzipFile, EOFError, OSError) as exc:
                raise DojoLongHorizonEconomicRunnerError(
                    "parent source gzip is invalid"
                ) from exc
        if digest.hexdigest() != expected_sha:
            raise DojoLongHorizonEconomicRunnerError(
                "parent physical shard bytes drifted"
            )
        if len(rows) != coverage["row_count"]:
            raise DojoLongHorizonEconomicRunnerError(
                "parent source month row count differs from coverage"
            )
        result[pair] = rows
    epoch_sets = {tuple(sorted(rows)) for rows in result.values()}
    if len(epoch_sets) != 1:
        raise DojoLongHorizonEconomicRunnerError(
            "pair source timestamps are not exactly synchronized; imputation is forbidden"
        )
    return result


def _safe_source_path_any_suffix(source_root: Path, relative_path: Any) -> Path:
    root = source_root.resolve(strict=True)
    relative = Path(_identifier(relative_path, field="source relative_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise DojoLongHorizonEconomicRunnerError("source relative_path is unsafe")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DojoLongHorizonEconomicRunnerError("source path escapes root") from exc
    state = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(state.st_mode):
        raise DojoLongHorizonEconomicRunnerError("source path is not a regular file")
    return path


def _strict_json_value(raw: bytes, *, field: str) -> Mapping[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoLongHorizonEconomicRunnerError(
            f"non-finite JSON token is forbidden: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoLongHorizonEconomicRunnerError(
                    f"duplicate JSON key in {field}: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoLongHorizonEconomicRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLongHorizonEconomicRunnerError(f"invalid JSON: {field}") from exc
    return _mapping(value, field=field)


def _verify_normalized_prices(
    normalized_rows: Sequence[Mapping[str, Any]],
    parent_rows: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> str:
    normalized_epochs = [row["epoch"] for row in normalized_rows]
    expected_epochs = sorted(next(iter(parent_rows.values())))
    if normalized_epochs != expected_epochs:
        raise DojoLongHorizonEconomicRunnerError(
            "normalized source timestamps differ from parent raw shards"
        )
    for row in normalized_rows:
        for quote in row["quotes"]:
            expected = parent_rows[quote["pair"]].get(row["epoch"])
            if expected != quote:
                raise DojoLongHorizonEconomicRunnerError(
                    "normalized bid/ask OHLC differs from parent raw shard"
                )
    return canonical_sha256(
        {
            "contract": "QR_DOJO_EXACT_NORMALIZED_PRICE_DERIVATION_V1",
            "epoch_count": len(normalized_epochs),
            "first_epoch": normalized_epochs[0],
            "last_epoch": normalized_epochs[-1],
            "pairs": sorted(parent_rows),
            "normalized_rows_sha256": canonical_sha256(list(normalized_rows)),
        }
    )


def build_month_source_slice_receipt(
    *,
    source_root: Path,
    relative_path: str,
    job: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Scan a prepared synchronized shard and seal its immutable receipt.

    This acquisition/preparation operation is intentionally separate from an
    economic run.  The economic runner later opens the sealed shard exactly
    once, hashes it while consuming it, and rejects any receipt drift at EOF.
    """

    path = _safe_source_path(source_root, relative_path)
    digest = hashlib.sha256()
    previous: int | None = None
    first: int | None = None
    count = 0
    normalized_rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while raw := handle.readline(MAX_SOURCE_LINE_BYTES + 1):
            digest.update(raw)
            parsed = _strict_json_line(raw, field=f"source row {count + 1}")
            row = _validate_source_row(parsed, job=job, previous_epoch=previous)
            previous = row["epoch"]
            first = row["epoch"] if first is None else first
            count += 1
            normalized_rows.append(row)
    if count == 0 or first is None or previous is None:
        raise DojoLongHorizonEconomicRunnerError("source shard is empty")
    parent = _source_parent_binding(source_manifest, job=job)
    parent_rows = _parent_price_rows(source_manifest, job=job, parent=parent)
    derivation_sha = _verify_normalized_prices(normalized_rows, parent_rows)
    body = {
        "contract": SOURCE_SLICE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "price_stream_id": job["price_stream_id"],
        "source_binding_id": job["source_binding_id"],
        "source_digest_sha256": job["source_digest_sha256"],
        "corpus_digest_sha256": job["corpus_digest_sha256"],
        "month": job["month"],
        "granularity": job["granularity"],
        "from_utc": job["from_utc"],
        "to_utc": job["to_utc"],
        "feed_pairs": list(job["feed_pairs"]),
        **parent,
        "relative_path": relative_path,
        "file_size_bytes": path.stat(follow_symlinks=False).st_size,
        "file_sha256": digest.hexdigest(),
        "row_count": count,
        "first_epoch": first,
        "last_epoch": previous,
        "complete_rows_only": True,
        "synthetic_quote_count": 0,
        "source_open_count_per_economic_run": 1,
        "normalized_price_derivation_verified": True,
        "normalized_price_derivation_sha256": derivation_sha,
        "authority": _authority(),
    }
    return {**body, "source_slice_receipt_sha256": canonical_sha256(body)}


def validate_month_source_slice_receipt(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact(value, _SOURCE_RECEIPT_KEYS, field="source receipt")
    body = {key: row[key] for key in row if key != "source_slice_receipt_sha256"}
    if (
        row["contract"] != SOURCE_SLICE_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["source_slice_receipt_sha256"] != canonical_sha256(body)
    ):
        raise DojoLongHorizonEconomicRunnerError("source receipt seal is invalid")
    for key in (
        "job_sha256",
        "price_stream_id",
        "source_binding_id",
        "source_digest_sha256",
        "corpus_digest_sha256",
        "month",
        "granularity",
        "from_utc",
        "to_utc",
    ):
        if row[key] != job[key]:
            raise DojoLongHorizonEconomicRunnerError(
                f"source receipt {key} differs from the sealed job"
            )
    parent = _source_parent_binding(source_manifest, job=job)
    for key, item in parent.items():
        if row[key] != item:
            raise DojoLongHorizonEconomicRunnerError(
                "source receipt parent-manifest binding drifted"
            )
    if (
        row["feed_pairs"] != job["feed_pairs"]
        or row["complete_rows_only"] is not True
        or row["synthetic_quote_count"] != 0
        or row["source_open_count_per_economic_run"] != 1
        or row["normalized_price_derivation_verified"] is not True
        or row["authority"] != _authority()
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "source receipt coverage or authority is invalid"
        )
    _integer(row["file_size_bytes"], field="source receipt.file_size_bytes", minimum=1)
    _sha(row["file_sha256"], field="source receipt.file_sha256")
    _sha(
        row["normalized_price_derivation_sha256"],
        field="source receipt.normalized_price_derivation_sha256",
    )
    _integer(row["row_count"], field="source receipt.row_count", minimum=1)
    _integer(row["first_epoch"], field="source receipt.first_epoch")
    _integer(row["last_epoch"], field="source receipt.last_epoch")
    return dict(row)


def _verify_runner_handoff(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(value, field="runner handoff")
    digest = row.get("runner_handoff_sha256")
    _sha(digest, field="runner handoff.runner_handoff_sha256")
    body = {key: item for key, item in row.items() if key != "runner_handoff_sha256"}
    if (
        digest != canonical_sha256(body)
        or row.get("contract") != RUNNER_HANDOFF_CONTRACT
    ):
        raise DojoLongHorizonEconomicRunnerError("runner handoff seal is invalid")
    job = _mapping(row.get("job"), field="runner handoff.job")
    job_sha = _sha(job.get("job_sha256"), field="runner handoff.job.job_sha256")
    if job_sha != canonical_sha256(
        {key: item for key, item in job.items() if key != "job_sha256"}
    ):
        raise DojoLongHorizonEconomicRunnerError("runner job digest drifted")
    claim = _mapping(row.get("claim"), field="runner handoff.claim")
    claim_sha = _sha(
        claim.get("claim_sha256"), field="runner handoff.claim.claim_sha256"
    )
    if (
        claim_sha
        != canonical_sha256(
            {key: item for key, item in claim.items() if key != "claim_sha256"}
        )
        or claim.get("job_sha256") != job_sha
    ):
        raise DojoLongHorizonEconomicRunnerError("runner claim digest drifted")
    obligations = _mapping(row.get("runner_obligations"), field="runner obligations")
    if not (
        obligations.get("one_synchronized_source_stream") is True
        and obligations.get("fanout_before_any_coordinate_decision") is True
        and obligations.get("source_reopen_or_resort_allowed") is False
        and obligations.get("broker_or_live_path_allowed") is False
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "runner obligations are not fail-closed"
        )
    if row.get("terminal_status") is not None:
        raise DojoLongHorizonEconomicRunnerError(
            "terminal runner handoff cannot execute"
        )
    if row.get("recorded_coordinate_count") != 0:
        raise DojoLongHorizonEconomicRunnerError(
            "economic runner requires a fresh all-coordinate handoff; partial-cell "
            "source replay is forbidden"
        )
    if row.get("predecessor_blocked_coordinate_count") != 0:
        raise DojoLongHorizonEconomicRunnerError(
            "predecessor-blocked handoff cannot execute economics"
        )
    if row.get("runnable_coordinate_ids") != row.get("pending_coordinate_ids"):
        raise DojoLongHorizonEconomicRunnerError(
            "runnable coordinates do not equal the fresh pending denominator"
        )
    return dict(row)


def _worker_catalog(
    value: Sequence[Mapping[str, Any]], *, job: Mapping[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(value, field="worker catalog")):
        row = _exact(raw, _CATALOG_KEYS, field=f"worker catalog[{index}]")
        worker_id = _identifier(row["worker_id"], field="worker_id")
        if worker_id in seen:
            raise DojoLongHorizonEconomicRunnerError("duplicate worker_id in catalog")
        seen.add(worker_id)
        rows.append(
            {
                "worker_id": worker_id,
                "owner_id": _identifier(row["owner_id"], field="owner_id"),
                "family_id": _identifier(row["family_id"], field="family_id"),
                "config_sha256": _sha(row["config_sha256"], field="config_sha256"),
            }
        )
    stripped = [
        {
            "worker_id": row["worker_id"],
            "family_id": row["family_id"],
            "config_sha256": row["config_sha256"],
        }
        for row in rows
    ]
    if canonical_sha256({"bindings": stripped}) != job["worker_set_sha256"]:
        raise DojoLongHorizonEconomicRunnerError(
            "worker catalog does not match the sealed schedule worker set"
        )
    return rows


def _selected(values: Sequence[Any], mask: Any, *, field: str) -> list[Any]:
    if not isinstance(mask, str) or len(mask) != len(values) or set(mask) - {"0", "1"}:
        raise DojoLongHorizonEconomicRunnerError(f"{field} is not a valid exact mask")
    return [item for item, bit in zip(values, mask, strict=True) if bit == "1"]


def _coordinate_runtime_rows(
    value: Mapping[str, Mapping[str, Any]],
    *,
    handoff: Mapping[str, Any],
    catalog: Sequence[Mapping[str, str]],
    implementation_digests: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    raw_map = _mapping(value, field="coordinate runtimes")
    job = handoff["job"]
    coordinates = {row["coordinate_id"]: row for row in job["coordinates"]}
    expected_ids = list(handoff["runnable_coordinate_ids"])
    if set(raw_map) != set(expected_ids):
        raise DojoLongHorizonEconomicRunnerError(
            "coordinate runtimes do not cover the exact runnable denominator"
        )
    result: dict[str, dict[str, Any]] = {}
    for coordinate_id in expected_ids:
        raw = _exact(raw_map[coordinate_id], _RUNTIME_KEYS, field="coordinate runtime")
        coordinate = coordinates[coordinate_id]
        if (
            raw["coordinate_id"] != coordinate_id
            or raw["cost_scenario"] != coordinate["cost_scenario"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "coordinate runtime identity or cost scenario drifted"
            )
        trade_pairs = _selected(
            job["feed_pairs"], coordinate["trade_pair_mask"], field="trade_pair_mask"
        )
        raw_trade_pairs = _sequence(
            raw["trade_pairs"], field="coordinate runtime.trade_pairs"
        )
        if (
            list(raw_trade_pairs) != trade_pairs
            or len(trade_pairs) != coordinate["trade_pair_count"]
            or canonical_sha256({"pairs": trade_pairs})
            != coordinate["trade_pair_set_sha256"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "coordinate trade-pair set drifted"
            )
        active = _selected(
            catalog, coordinate["active_worker_mask"], field="active_worker_mask"
        )
        stripped = [
            {
                "worker_id": row["worker_id"],
                "family_id": row["family_id"],
                "config_sha256": row["config_sha256"],
            }
            for row in active
        ]
        if (
            len(active) != coordinate["active_worker_count"]
            or canonical_sha256({"bindings": stripped})
            != coordinate["active_worker_bindings_sha256"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "coordinate active-worker set drifted"
            )
        policy = verify_portfolio_policy(raw["portfolio_policy"])
        expected_cost_sha = implementation_digests[
            "base_cost_policy_sha256"
            if coordinate["cost_scenario"] == "BASE"
            else "stress_cost_policy_sha256"
        ]
        expected_risk_sha = implementation_digests["risk_policy_sha256"]
        expected_replay_sha = implementation_digests["replay_engine_sha256"]
        for field, expected_sha in (
            ("cost_policy_sha256", expected_cost_sha),
            ("risk_policy_sha256", expected_risk_sha),
            ("replay_engine_sha256", expected_replay_sha),
        ):
            if raw[field] != expected_sha:
                raise DojoLongHorizonEconomicRunnerError(
                    f"coordinate runtime {field} differs from the sealed plan"
                )
        if policy["expected_quote_pairs"] != list(job["feed_pairs"]):
            raise DojoLongHorizonEconomicRunnerError(
                "portfolio policy does not cover the exact feed pairs"
            )
        if policy["tradable_pairs"] != sorted(trade_pairs):
            raise DojoLongHorizonEconomicRunnerError(
                "portfolio policy tradable pairs differ from the coordinate mask"
            )
        if policy["active_worker_bindings"] != sorted(
            active, key=lambda row: row["worker_id"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "portfolio policy active worker bindings drifted"
            )
        policy_binding_body = {
            "coordinate_id": coordinate_id,
            "cost_scenario": coordinate["cost_scenario"],
            "portfolio_policy_sha256": policy["policy_sha256"],
            "cost_policy_sha256": expected_cost_sha,
            "risk_policy_sha256": expected_risk_sha,
            "replay_engine_sha256": expected_replay_sha,
            "allocator_policy": coordinate["allocator_policy"],
            "initial_balance_jpy": STARTING_EQUITY_JPY,
        }
        if raw["portfolio_policy_binding_sha256"] != canonical_sha256(
            policy_binding_body
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "portfolio policy is not bound to the plan/coordinate risk envelope"
            )
        result[coordinate_id] = {
            "coordinate": dict(coordinate),
            "cost_scenario": raw["cost_scenario"],
            "trade_pairs": trade_pairs,
            "active_worker_bindings": [dict(item) for item in active],
            "portfolio_policy": policy,
            "portfolio_policy_binding_sha256": raw["portfolio_policy_binding_sha256"],
        }
    return result


_CARRY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "state_slot_id",
        "producer_job_sha256",
        "producer_coordinate_id",
        "portfolio_policy_sha256",
        "worker_runtime_binding_sha256",
        "portfolio_carry_state",
        "worker_state",
        "worker_state_sha256",
        "source_batch_chain_sha256",
        "source_cursor",
        "authority",
        "state_sha256",
    }
)


def _validate_economic_carry(
    value: Mapping[str, Any],
    *,
    state_slot_id: str,
    policy_sha256: str,
    worker_runtime_binding_sha256: str,
) -> dict[str, Any]:
    row = _exact(value, _CARRY_KEYS, field="economic carry")
    body = {key: row[key] for key in row if key != "state_sha256"}
    if (
        row["contract"] != ECONOMIC_CARRY_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["state_sha256"] != canonical_sha256(body)
        or row["state_slot_id"] != state_slot_id
        or row["portfolio_policy_sha256"] != policy_sha256
        or row["worker_runtime_binding_sha256"] != worker_runtime_binding_sha256
        or row["worker_state_sha256"] != canonical_sha256(row["worker_state"])
        or row["authority"] != _authority()
    ):
        raise DojoLongHorizonEconomicRunnerError("economic carry binding drifted")
    _sha(row["source_batch_chain_sha256"], field="carry source batch chain")
    cursor = _exact(
        row["source_cursor"],
        {
            "last_epoch",
            "granularity",
            "intrabar_path",
            "quote_watermark",
            "source_row_count",
        },
        field="economic carry.source_cursor",
    )
    _integer(cursor["last_epoch"], field="carry source cursor.last_epoch")
    _integer(
        cursor["quote_watermark"],
        field="carry source cursor.quote_watermark",
        minimum=1,
    )
    _integer(
        cursor["source_row_count"],
        field="carry source cursor.source_row_count",
        minimum=1,
    )
    if (
        cursor["granularity"] not in _GRANULARITY_SECONDS
        or cursor["intrabar_path"] not in _PHASES
    ):
        raise DojoLongHorizonEconomicRunnerError("carry source cursor is unsupported")
    portfolio = _mapping(row["portfolio_carry_state"], field="portfolio carry")
    if portfolio.get("policy_sha256") != policy_sha256:
        raise DojoLongHorizonEconomicRunnerError("portfolio carry policy drifted")
    return dict(row)


def _economic_carry(
    *,
    state_slot_id: str,
    job_sha256: str,
    coordinate_id: str,
    policy_sha256: str,
    worker_runtime_binding_sha256: str,
    portfolio_carry_state: Mapping[str, Any],
    worker_state: Any,
    source_batch_chain_sha256: str,
    source_cursor: Mapping[str, Any],
) -> dict[str, Any]:
    encoded_state = _canonical_bytes(worker_state)
    if len(encoded_state) > MAX_WORKER_STATE_BYTES:
        raise DojoLongHorizonEconomicRunnerError(
            "worker state exceeds the sealed runner bound"
        )
    body = {
        "contract": ECONOMIC_CARRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "state_slot_id": state_slot_id,
        "producer_job_sha256": job_sha256,
        "producer_coordinate_id": coordinate_id,
        "portfolio_policy_sha256": policy_sha256,
        "worker_runtime_binding_sha256": worker_runtime_binding_sha256,
        "portfolio_carry_state": dict(portfolio_carry_state),
        "worker_state": worker_state,
        "worker_state_sha256": canonical_sha256(worker_state),
        "source_batch_chain_sha256": source_batch_chain_sha256,
        "source_cursor": dict(source_cursor),
        "authority": _authority(),
    }
    return {**body, "state_sha256": canonical_sha256(body)}


def _phase_quotes(row: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    offset = {"O": 0, "H": 1, "L": 2, "C": 3}[phase]
    timestamp = datetime.fromtimestamp(row["epoch"], timezone.utc).isoformat()
    return [
        {
            "pair": quote["pair"],
            "bid": quote["bid"][offset],
            "ask": quote["ask"][offset],
            "timestamp": f"{timestamp}#{phase}",
        }
        for quote in row["quotes"]
    ]


def _batch_chain(
    previous: str, *, quote_digest: str, epoch: int, phase: str, watermark: int
) -> str:
    return canonical_sha256(
        {
            "previous_batch_chain_sha256": previous,
            "quote_batch_sha256": quote_digest,
            "epoch": epoch,
            "phase": phase,
            "quote_watermark": watermark,
        }
    )


def _raw_proposals_for_snapshot(
    runtime: EconomicWorkerRuntime,
    *,
    snapshot: Mapping[str, Any],
    trade_pairs: Sequence[str],
) -> dict[str, Any]:
    readonly = readonly_post_exit_snapshot(snapshot)
    proposals = _sequence(runtime.propose(readonly), field="worker proposals")
    sealed = []
    allowed = set(trade_pairs)
    for raw in proposals:
        proposal = seal_worker_proposal(snapshot, raw)
        if any(
            intent["parameters"]["pair"] not in allowed
            for intent in proposal["new_risk_intents"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "worker proposed new risk outside the coordinate trade-pair mask"
            )
        sealed.append(proposal)
    return seal_worker_proposal_batch(snapshot, sealed)


def _failure_cell(
    *,
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
    coordinate_id: str,
    code: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    return build_long_horizon_coordinate_result(
        job=job,
        claim=claim,
        coordinate_id=coordinate_id,
        status="FAILED",
        failure={
            "code": code,
            "retryable": False,
            "evidence_sha256": evidence_sha256,
        },
    )


def _result_number(result: Mapping[str, Any], key: str) -> float:
    if key not in result:
        raise DojoLongHorizonEconomicRunnerError(
            f"incremental reducer result is missing {key}"
        )
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoLongHorizonEconomicRunnerError(
            f"incremental reducer result {key} is not numeric"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DojoLongHorizonEconomicRunnerError(
            f"incremental reducer result {key} is non-finite"
        )
    return number


def run_long_horizon_economic_job(
    *,
    runner_handoff: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_root: Path,
    source_manifest: Mapping[str, Any],
    source_slice_receipt: Mapping[str, Any],
    economic_evidence_root: Path,
    worker_catalog: Sequence[Mapping[str, Any]],
    coordinate_runtimes: Mapping[str, Mapping[str, Any]],
    worker_runtime_factory: WorkerRuntimeFactory,
    worker_runtime_binding_sha256: str,
    worker_runtime_seal: Mapping[str, Any] | None = None,
    worker_runtime_repo_root: Path | None = None,
    carry_states_by_slot: Mapping[str, Mapping[str, Any]] | None = None,
    initial_balance_jpy: int | float = 200_000,
) -> dict[str, Any]:
    """Execute one fresh handoff without materializing source frames.

    ``PortfolioReplaySession`` is required.  Older batch-only reducers are
    rejected rather than causing the runner to collect a month of frames or
    reopen the source once per account coordinate.
    """

    handoff = _verify_runner_handoff(runner_handoff)
    job = handoff["job"]
    claim = handoff["claim"]
    verified_plan = validate_long_horizon_train_plan(plan)
    if verified_plan["plan_sha256"] != job["plan_sha256"]:
        raise DojoLongHorizonEconomicRunnerError(
            "economic runner plan differs from the sealed job"
        )
    implementation_digests = verified_plan["implementation_binding"]["digests"]
    receipt = validate_month_source_slice_receipt(
        source_slice_receipt, job=job, source_manifest=source_manifest
    )
    evidence_root = _economic_evidence_directory(economic_evidence_root)
    path = _safe_source_path(source_root, receipt["relative_path"])
    before = path.stat(follow_symlinks=False)
    catalog = _worker_catalog(worker_catalog, job=job)
    runtimes = _coordinate_runtime_rows(
        coordinate_runtimes,
        handoff=handoff,
        catalog=catalog,
        implementation_digests=implementation_digests,
    )
    resident_limit = _integer(
        _mapping(handoff["resource_limits"], field="resource limits").get(
            "max_resident_coordinates"
        ),
        field="resource_limits.max_resident_coordinates",
        minimum=1,
    )
    if len(runtimes) > resident_limit:
        raise DojoLongHorizonEconomicRunnerError(
            "runnable coordinate denominator exceeds the sealed resident limit"
        )
    runtime_sha = _sha(
        worker_runtime_binding_sha256, field="worker_runtime_binding_sha256"
    )
    verified_runtime_seal: dict[str, Any] | None = None
    if worker_runtime_factory is builtin_no_intent_runtime_factory:
        if (
            runtime_sha != BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256
            or worker_runtime_seal is not None
            or worker_runtime_repo_root is not None
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "built-in HOLD runtime binding or seal arguments are invalid"
            )
        worker_runtime_mode = "BUILTIN_NO_INTENT_BASELINE_ONLY"
    elif worker_runtime_factory is builtin_strategy_runtime_factory:
        if worker_runtime_seal is None or worker_runtime_repo_root is None:
            raise DojoLongHorizonEconomicRunnerError(
                "built-in strategy runtime requires its presealed dependency manifest"
            )
        try:
            verified_runtime_seal = verify_builtin_strategy_runtime_seal(
                worker_runtime_seal, repo_root=worker_runtime_repo_root
            )
        except ValueError as exc:
            raise DojoLongHorizonEconomicRunnerError(
                f"built-in strategy runtime seal is invalid: {exc}"
            ) from exc
        if runtime_sha != verified_runtime_seal["runtime_binding_sha256"]:
            raise DojoLongHorizonEconomicRunnerError(
                "built-in strategy runtime binding differs from its dependency seal"
            )
        if catalog != verified_runtime_seal["worker_catalog"]:
            raise DojoLongHorizonEconomicRunnerError(
                "worker catalog differs from the sealed built-in strategy catalog"
            )
        worker_runtime_mode = "SEALED_BUILTIN_MULTI_STRATEGY"
    elif type(worker_runtime_factory) is SealedTunedStrategyRuntimeFactory:
        if worker_runtime_seal is None or worker_runtime_repo_root is None:
            raise DojoLongHorizonEconomicRunnerError(
                "tuned strategy runtime requires its generation dependency seal"
            )
        try:
            verified_runtime_seal = verify_tuned_strategy_runtime_seal(
                worker_runtime_seal, repo_root=worker_runtime_repo_root
            )
        except ValueError as exc:
            raise DojoLongHorizonEconomicRunnerError(
                f"tuned strategy runtime seal is invalid: {exc}"
            ) from exc
        if (
            runtime_sha != verified_runtime_seal["runtime_binding_sha256"]
            or worker_runtime_factory.runtime_binding_sha256 != runtime_sha
            or not worker_runtime_factory.matches_verified_seal(verified_runtime_seal)
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "tuned strategy runtime factory differs from its immutable seal"
            )
        if catalog != verified_runtime_seal["worker_catalog"]:
            raise DojoLongHorizonEconomicRunnerError(
                "worker catalog differs from the tuned generation seal"
            )
        worker_runtime_mode = "SEALED_TUNED_DECLARATIVE_MULTI_STRATEGY"
    else:
        raise DojoLongHorizonEconomicRunnerError(
            "external in-process worker code is forbidden in the economic evidence "
            "runner; only explicitly sealed capability-closed built-ins are available"
        )
    carry_inputs = dict(carry_states_by_slot or {})
    ready_receipts = {
        row["state_slot_id"]: row for row in handoff["ready_predecessor_carry_receipts"]
    }
    expected_slots = {
        row["coordinate"]["predecessor_state_slot_id"]
        for row in runtimes.values()
        if row["coordinate"]["predecessor_state_slot_id"] is not None
    }
    if set(carry_inputs) != expected_slots or set(ready_receipts) != expected_slots:
        raise DojoLongHorizonEconomicRunnerError(
            "economic carry inputs do not equal the ready predecessor denominator"
        )

    try:
        from quant_rabbit.dojo_portfolio_replay_reducer import PortfolioReplaySession
    except ImportError as exc:  # pragma: no cover - compatibility guard
        raise DojoLongHorizonEconomicRunnerError(
            "incremental PortfolioReplaySession is required; batch materialization is forbidden"
        ) from exc

    sessions: dict[str, Any] = {}
    worker_instances: dict[str, EconomicWorkerRuntime] = {}
    transcript_recorders: dict[str, EconomicTranscriptRecorder] = {}
    transcript_paths: dict[str, Path] = {}
    terminal_transcript_ids: set[str] = set()
    predecessor_sha: dict[str, str | None] = {}
    failures: dict[str, tuple[str, str]] = {}
    portfolio_results: dict[str, dict[str, Any]] = {}
    economic_carries: dict[str, dict[str, Any]] = {}
    initial = _positive(initial_balance_jpy, field="initial_balance_jpy")
    if initial != float(STARTING_EQUITY_JPY):
        raise DojoLongHorizonEconomicRunnerError(
            "initial balance differs from the sealed long-horizon plan capital"
        )
    predecessor_watermarks: set[int] = set()
    predecessor_batch_chains: set[str] = set()
    predecessor_last_epochs: set[int] = set()

    for coordinate_id in handoff["runnable_coordinate_ids"]:
        runtime = runtimes[coordinate_id]
        coordinate = runtime["coordinate"]
        slot = coordinate["predecessor_state_slot_id"]
        prior_worker_state: Any | None = None
        portfolio_carry: Mapping[str, Any] | None = None
        if slot is not None:
            verified_carry = _validate_economic_carry(
                carry_inputs[slot],
                state_slot_id=slot,
                policy_sha256=runtime["portfolio_policy"]["policy_sha256"],
                worker_runtime_binding_sha256=runtime_sha,
            )
            if ready_receipts[slot]["state_sha256"] != verified_carry["state_sha256"]:
                raise DojoLongHorizonEconomicRunnerError(
                    "economic carry disagrees with execution carry receipt"
                )
            predecessor_sha[coordinate_id] = verified_carry["state_sha256"]
            portfolio_carry = verified_carry["portfolio_carry_state"]
            prior_worker_state = verified_carry["worker_state"]
            predecessor_watermarks.add(
                _integer(
                    portfolio_carry.get("last_quote_watermark"),
                    field="portfolio carry.last_quote_watermark",
                    minimum=1,
                )
            )
            predecessor_batch_chains.add(verified_carry["source_batch_chain_sha256"])
            cursor = verified_carry["source_cursor"]
            if (
                cursor["granularity"] != job["granularity"]
                or cursor["intrabar_path"] != job["intrabar_path"]
            ):
                raise DojoLongHorizonEconomicRunnerError(
                    "continuous source cursor granularity/path drifted"
                )
            predecessor_last_epochs.add(cursor["last_epoch"])
        else:
            predecessor_sha[coordinate_id] = None
        transcript_id = canonical_sha256(
            {
                "contract": "QR_DOJO_LONG_HORIZON_TRANSCRIPT_ID_V1",
                "job_sha256": job["job_sha256"],
                "claim_sha256": claim["claim_sha256"],
                "coordinate_id": coordinate_id,
                "source_slice_receipt_sha256": receipt["source_slice_receipt_sha256"],
                "worker_runtime_binding_sha256": runtime_sha,
                "portfolio_policy_binding_sha256": runtime[
                    "portfolio_policy_binding_sha256"
                ],
                "predecessor_state_sha256": predecessor_sha[coordinate_id],
            }
        )
        transcript_path = evidence_root / f"{transcript_id}.economic.jsonl"
        input_bindings = {
            "job_sha256": job["job_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "source_slice_receipt_sha256": receipt["source_slice_receipt_sha256"],
            "worker_runtime_binding_sha256": runtime_sha,
            "cost_policy_sha256": implementation_digests[
                "base_cost_policy_sha256"
                if coordinate["cost_scenario"] == "BASE"
                else "stress_cost_policy_sha256"
            ],
            "risk_policy_sha256": implementation_digests["risk_policy_sha256"],
            "replay_engine_sha256": implementation_digests["replay_engine_sha256"],
            "portfolio_policy_binding_sha256": runtime[
                "portfolio_policy_binding_sha256"
            ],
            "predecessor_state_sha256": predecessor_sha[coordinate_id],
            "predecessor_portfolio_carry_state_sha256": (
                None
                if portfolio_carry is None
                else portfolio_carry["carry_state_sha256"]
            ),
            "predecessor_source_batch_chain_sha256": (
                GENESIS_BATCH_CHAIN_SHA256
                if slot is None
                else verified_carry["source_batch_chain_sha256"]
            ),
        }
        header = _transcript_call(
            build_economic_transcript_header,
            transcript_id=transcript_id,
            coordinate_id=coordinate_id,
            portfolio_policy=runtime["portfolio_policy"],
            input_bindings=input_bindings,
            terminal_policy=coordinate["terminal_policy"],
            expected_quote_batch_count=(
                receipt["row_count"] * len(_PHASES[job["intrabar_path"]])
            ),
            initial_balance_jpy=initial if portfolio_carry is None else None,
            predecessor_portfolio_carry_state=portfolio_carry,
        )
        try:
            transcript_recorders[coordinate_id] = _transcript_call(
                EconomicTranscriptRecorder,
                transcript_path,
                header,
            )
        except Exception:
            # Preserve every already-created incomplete file as fail-closed
            # crash evidence, but do not leak its descriptor to the caller.
            for recorder in transcript_recorders.values():
                recorder.close()
            raise
        transcript_paths[coordinate_id] = transcript_path
        try:
            sessions[coordinate_id] = PortfolioReplaySession(
                policy=runtime["portfolio_policy"],
                initial_balance_jpy=initial if portfolio_carry is None else None,
                carry_state=portfolio_carry,
            )
            worker_instances[coordinate_id] = worker_runtime_factory(
                {
                    **coordinate,
                    "trade_pairs": runtime["trade_pairs"],
                    "granularity": job["granularity"],
                    "bar_seconds": _GRANULARITY_SECONDS[job["granularity"]],
                },
                runtime["active_worker_bindings"],
                prior_worker_state,
            )
        except Exception:
            evidence = canonical_sha256(
                {
                    "job_sha256": job["job_sha256"],
                    "coordinate_id": coordinate_id,
                    "failure_code": "WORKER_RUNTIME_INITIALIZATION_FAILURE",
                    "worker_runtime_binding_sha256": runtime_sha,
                }
            )
            failures[coordinate_id] = (
                "WORKER_RUNTIME_INITIALIZATION_FAILURE",
                evidence,
            )
            _transcript_call(
                transcript_recorders[coordinate_id].seal_failure,
                failure_code="WORKER_RUNTIME_INITIALIZATION_FAILURE",
                failure_stage=_failure_stage("WORKER_RUNTIME_INITIALIZATION_FAILURE"),
                failure_evidence_sha256=evidence,
            )
            terminal_transcript_ids.add(coordinate_id)
            sessions.pop(coordinate_id, None)
            worker_instances.pop(coordinate_id, None)

    digest = hashlib.sha256()
    row_count = 0
    first_epoch: int | None = None
    last_epoch: int | None = None
    if (
        len(predecessor_watermarks) > 1
        or len(predecessor_batch_chains) > 1
        or len(predecessor_last_epochs) > 1
    ):
        raise DojoLongHorizonEconomicRunnerError(
            "same-stream continuous predecessors disagree on source cursor/chain"
        )
    watermark = next(iter(predecessor_watermarks), 0)
    batch_count = 0
    batch_chain = next(iter(predecessor_batch_chains), GENESIS_BATCH_CHAIN_SHA256)
    stream_predecessor_epoch = next(iter(predecessor_last_epochs), None)
    systemic_source_failure = False
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise DojoLongHorizonEconomicRunnerError(
                "source path identity changed before economic read"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            previous_epoch: int | None = stream_predecessor_epoch
            while raw := handle.readline(MAX_SOURCE_LINE_BYTES + 1):
                digest.update(raw)
                parsed = _strict_json_line(raw, field=f"source row {row_count + 1}")
                candle = _validate_source_row(
                    parsed, job=job, previous_epoch=previous_epoch
                )
                previous_epoch = candle["epoch"]
                first_epoch = candle["epoch"] if first_epoch is None else first_epoch
                last_epoch = candle["epoch"]
                row_count += 1
                for phase in _PHASES[job["intrabar_path"]]:
                    watermark += 1
                    batch_count += 1
                    quotes = _phase_quotes(candle, phase)
                    quote_digest = quote_batch_sha256(
                        epoch=candle["epoch"],
                        phase=phase,
                        intrabar=job["intrabar_path"],
                        quote_watermark=watermark,
                        quotes=quotes,
                    )
                    batch_chain = _batch_chain(
                        batch_chain,
                        quote_digest=quote_digest,
                        epoch=candle["epoch"],
                        phase=phase,
                        watermark=watermark,
                    )
                    # Barrier 1: reducer-owned post-exit state reaches every
                    # still-live coordinate before any worker can decide.
                    snapshots: dict[str, Mapping[str, Any]] = {}
                    for coordinate_id in list(sessions):
                        if coordinate_id in failures:
                            continue
                        _transcript_call(
                            transcript_recorders[coordinate_id].record_quote_batch,
                            coordinate_id=coordinate_id,
                            epoch=candle["epoch"],
                            phase=phase,
                            intrabar=job["intrabar_path"],
                            quote_watermark=watermark,
                            quotes=quotes,
                            quote_batch_sha256_value=quote_digest,
                            source_batch_chain_sha256=batch_chain,
                        )
                        try:
                            snapshots[coordinate_id] = sessions[
                                coordinate_id
                            ].prepare_coordinate(
                                coordinate_id=coordinate_id,
                                epoch=candle["epoch"],
                                phase=phase,
                                intrabar=job["intrabar_path"],
                                quote_watermark=watermark,
                                quotes=quotes,
                                quote_batch_sha256_value=quote_digest,
                            )
                            _transcript_call(
                                transcript_recorders[
                                    coordinate_id
                                ].record_post_exit_snapshot,
                                snapshots[coordinate_id],
                            )
                        except Exception:
                            evidence = canonical_sha256(
                                {
                                    "coordinate_id": coordinate_id,
                                    "failure_code": "PORTFOLIO_PREPARE_FAILURE",
                                    "quote_batch_sha256": quote_digest,
                                }
                            )
                            failures[coordinate_id] = (
                                "PORTFOLIO_PREPARE_FAILURE",
                                evidence,
                            )
                            _transcript_call(
                                transcript_recorders[coordinate_id].seal_failure,
                                failure_code="PORTFOLIO_PREPARE_FAILURE",
                                failure_stage=_failure_stage(
                                    "PORTFOLIO_PREPARE_FAILURE"
                                ),
                                failure_evidence_sha256=evidence,
                            )
                            terminal_transcript_ids.add(coordinate_id)
                    # Barrier 2: collect and seal all decisions.  No coordinate
                    # is economically advanced while another is still deciding.
                    proposal_batches: dict[str, Mapping[str, Any]] = {}
                    for coordinate_id, snapshot in snapshots.items():
                        if coordinate_id in failures:
                            continue
                        try:
                            proposal_batches[coordinate_id] = (
                                _raw_proposals_for_snapshot(
                                    worker_instances[coordinate_id],
                                    snapshot=snapshot,
                                    trade_pairs=runtimes[coordinate_id]["trade_pairs"],
                                )
                            )
                            _transcript_call(
                                transcript_recorders[
                                    coordinate_id
                                ].record_worker_proposal_batch,
                                proposal_batches[coordinate_id],
                            )
                        except Exception:
                            evidence = canonical_sha256(
                                {
                                    "coordinate_id": coordinate_id,
                                    "failure_code": "WORKER_PROTOCOL_FAILURE",
                                    "snapshot_sha256": snapshot["snapshot_sha256"],
                                }
                            )
                            failures[coordinate_id] = (
                                "WORKER_PROTOCOL_FAILURE",
                                evidence,
                            )
                            _transcript_call(
                                transcript_recorders[coordinate_id].seal_failure,
                                failure_code="WORKER_PROTOCOL_FAILURE",
                                failure_stage=_failure_stage("WORKER_PROTOCOL_FAILURE"),
                                failure_evidence_sha256=evidence,
                            )
                            terminal_transcript_ids.add(coordinate_id)
                    for coordinate_id, proposal_batch in proposal_batches.items():
                        if coordinate_id in failures:
                            continue
                        try:
                            allocation_receipt = sessions[
                                coordinate_id
                            ].consume_proposal_batch(proposal_batch)
                            _transcript_call(
                                transcript_recorders[
                                    coordinate_id
                                ].record_allocation_receipt,
                                allocation_receipt,
                            )
                        except Exception:
                            evidence = canonical_sha256(
                                {
                                    "coordinate_id": coordinate_id,
                                    "failure_code": "PORTFOLIO_CONSUME_FAILURE",
                                    "proposal_batch_sha256": proposal_batch[
                                        "batch_sha256"
                                    ],
                                }
                            )
                            failures[coordinate_id] = (
                                "PORTFOLIO_CONSUME_FAILURE",
                                evidence,
                            )
                            _transcript_call(
                                transcript_recorders[coordinate_id].seal_failure,
                                failure_code="PORTFOLIO_CONSUME_FAILURE",
                                failure_stage=_failure_stage(
                                    "PORTFOLIO_CONSUME_FAILURE"
                                ),
                                failure_evidence_sha256=evidence,
                            )
                            terminal_transcript_ids.add(coordinate_id)
            after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "source shard changed during economic read"
            )
        if (
            row_count != receipt["row_count"]
            or first_epoch != receipt["first_epoch"]
            or last_epoch != receipt["last_epoch"]
            or digest.hexdigest() != receipt["file_sha256"]
        ):
            raise DojoLongHorizonEconomicRunnerError(
                "source bytes or coverage differ from the sealed receipt"
            )
    except (DojoLongHorizonEconomicRunnerError, OSError):
        systemic_source_failure = True
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if row_count == 0:
        systemic_source_failure = True
    if systemic_source_failure:
        evidence = canonical_sha256(
            {
                "job_sha256": job["job_sha256"],
                "failure_code": "SOURCE_STREAM_FAILURE",
                "source_slice_receipt_sha256": receipt["source_slice_receipt_sha256"],
            }
        )
        failures = {
            coordinate_id: ("SOURCE_STREAM_FAILURE", evidence)
            for coordinate_id in handoff["runnable_coordinate_ids"]
        }
    elif receipt["parent_quote_coverage_complete"] is not True:
        evidence = canonical_sha256(
            {
                "job_sha256": job["job_sha256"],
                "failure_code": "SOURCE_QUOTE_COVERAGE_UNPROVEN",
                "source_manifest_sha256": receipt["source_manifest_sha256"],
                "parent_month_coverage_cells_sha256": receipt[
                    "parent_month_coverage_cells_sha256"
                ],
            }
        )
        failures = {
            coordinate_id: ("SOURCE_QUOTE_COVERAGE_UNPROVEN", evidence)
            for coordinate_id in handoff["runnable_coordinate_ids"]
        }

    for coordinate_id, (code, evidence) in failures.items():
        if coordinate_id in terminal_transcript_ids:
            continue
        _transcript_call(
            transcript_recorders[coordinate_id].seal_failure,
            failure_code=code,
            failure_stage=_failure_stage(code),
            failure_evidence_sha256=evidence,
        )
        terminal_transcript_ids.add(coordinate_id)

    cells: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for coordinate_id in handoff["runnable_coordinate_ids"]:
        if coordinate_id in failures:
            code, evidence = failures[coordinate_id]
            cells.append(
                _failure_cell(
                    job=job,
                    claim=claim,
                    coordinate_id=coordinate_id,
                    code=code,
                    evidence_sha256=evidence,
                )
            )
            continue
        runtime = runtimes[coordinate_id]
        coordinate = runtime["coordinate"]
        try:
            result = validate_portfolio_replay_result(
                sessions[coordinate_id].finalize(
                    terminal_policy=coordinate["terminal_policy"]
                )
            )
            worker_state = worker_instances[coordinate_id].export_state()
            if len(_canonical_bytes(worker_state)) > MAX_WORKER_STATE_BYTES:
                raise DojoLongHorizonEconomicRunnerError(
                    "worker state exceeds the sealed runner bound"
                )
            carry_out_sha: str | None = None
            carry_slot = coordinate["carry_out_state_slot_id"]
            if carry_slot is not None:
                carry = _economic_carry(
                    state_slot_id=carry_slot,
                    job_sha256=job["job_sha256"],
                    coordinate_id=coordinate_id,
                    policy_sha256=result["policy_sha256"],
                    worker_runtime_binding_sha256=runtime_sha,
                    portfolio_carry_state=result["carry_state"],
                    worker_state=worker_state,
                    source_batch_chain_sha256=batch_chain,
                    source_cursor={
                        "last_epoch": last_epoch,
                        "granularity": job["granularity"],
                        "intrabar_path": job["intrabar_path"],
                        "quote_watermark": watermark,
                        "source_row_count": row_count,
                    },
                )
                economic_carries[carry_slot] = carry
                carry_out_sha = carry["state_sha256"]
            compact_evidence = canonical_sha256(
                {
                    "coordinate_id": coordinate_id,
                    "portfolio_result_sha256": result["result_sha256"],
                    "source_slice_receipt_sha256": receipt[
                        "source_slice_receipt_sha256"
                    ],
                    "batch_chain_sha256": batch_chain,
                    "worker_runtime_binding_sha256": runtime_sha,
                    "worker_state_sha256": canonical_sha256(worker_state),
                    "predecessor_state_sha256": predecessor_sha[coordinate_id],
                    "carry_out_state_sha256": carry_out_sha,
                }
            )
            cell = build_long_horizon_coordinate_result(
                job=job,
                claim=claim,
                coordinate_id=coordinate_id,
                status="COMPLETE",
                starting_balance_jpy=result["start_balance_jpy"],
                starting_equity_jpy=result["start_equity_jpy"],
                ending_balance_jpy=result["end_balance_jpy"],
                ending_equity_jpy=result["end_equity_jpy"],
                minimum_mtm_equity_jpy=_result_number(result, "minimum_mtm_equity_jpy"),
                minimum_free_margin_jpy=_result_number(
                    result, "minimum_free_margin_jpy"
                ),
                max_mtm_drawdown_fraction=result["max_drawdown_fraction"],
                peak_margin_usage_fraction=_result_number(
                    result, "peak_margin_usage_fraction"
                ),
                margin_closeout_count=result["margin_closeouts"],
                ruin_event_count=int(_result_number(result, "ruin_event_count")),
                trade_count=result["trade_count"],
                fill_count=result["execution_fill_count"],
                margin_reject_count=int(_result_number(result, "margin_reject_count")),
                financing_jpy=result["financing_cost_jpy"],
                transaction_cost_jpy=result["transaction_cost_jpy"],
                source_slice_receipt_sha256=receipt["source_slice_receipt_sha256"],
                batch_chain_sha256=batch_chain,
                compact_evidence_sha256=compact_evidence,
                quote_coverage_complete=receipt["parent_quote_coverage_complete"],
                active_worker_ack_complete=True,
                predecessor_state_sha256=predecessor_sha[coordinate_id],
                carry_out_state_sha256=carry_out_sha,
            )
            _transcript_call(
                transcript_recorders[coordinate_id].seal_success,
                terminal_policy=coordinate["terminal_policy"],
                portfolio_result=result,
                source_batch_chain_sha256=batch_chain,
            )
            terminal_transcript_ids.add(coordinate_id)
            cells.append(cell)
            portfolio_results[coordinate_id] = result
            evidence_rows.append(
                {
                    "coordinate_id": coordinate_id,
                    "portfolio_result_sha256": result["result_sha256"],
                    "compact_evidence_sha256": compact_evidence,
                }
            )
        except Exception:
            evidence = canonical_sha256(
                {
                    "coordinate_id": coordinate_id,
                    "failure_code": "PORTFOLIO_FINALIZE_FAILURE",
                    "batch_chain_sha256": batch_chain,
                }
            )
            if coordinate_id not in terminal_transcript_ids:
                _transcript_call(
                    transcript_recorders[coordinate_id].seal_failure,
                    failure_code="PORTFOLIO_FINALIZE_FAILURE",
                    failure_stage=_failure_stage("PORTFOLIO_FINALIZE_FAILURE"),
                    failure_evidence_sha256=evidence,
                )
                terminal_transcript_ids.add(coordinate_id)
            cells.append(
                _failure_cell(
                    job=job,
                    claim=claim,
                    coordinate_id=coordinate_id,
                    code="PORTFOLIO_FINALIZE_FAILURE",
                    evidence_sha256=evidence,
                )
            )

    complete_count = sum(row["status"] == "COMPLETE" for row in cells)
    failed_count = sum(row["status"] == "FAILED" for row in cells)
    expected_coordinate_ids = sorted(handoff["runnable_coordinate_ids"])
    if terminal_transcript_ids != set(expected_coordinate_ids):
        raise DojoLongHorizonEconomicRunnerError(
            "economic transcript terminal denominator is incomplete"
        )
    attestations_by_coordinate: dict[str, dict[str, Any]] = {}
    transcript_artifacts: list[dict[str, Any]] = []
    for coordinate_id in expected_coordinate_ids:
        transcript_path = transcript_paths[coordinate_id]
        attestation = _reexecute_transcript_in_separate_process(transcript_path)
        attestation_path = transcript_path.with_suffix(".attestation.json")
        _write_exclusive_json(attestation_path, attestation)
        attestations_by_coordinate[coordinate_id] = attestation
        transcript_artifacts.append(
            {
                "coordinate_id": coordinate_id,
                "transcript_filename": transcript_path.name,
                "transcript_file_sha256": attestation["transcript_file_sha256"],
                "reexecution_attestation_filename": attestation_path.name,
                "reexecution_attestation_sha256": attestation[
                    "reexecution_attestation_sha256"
                ],
                "reexecution_status": attestation["status"],
            }
        )
    fixed_attestation = _transcript_call(
        build_fixed_denominator_reexecution_attestation,
        expected_coordinate_ids=expected_coordinate_ids,
        attestations_by_coordinate=attestations_by_coordinate,
    )
    fixed_attestation_path = evidence_root / (
        f"{job['job_sha256']}.{claim['claim_sha256']}.fixed-denominator-attestation.json"
    )
    _write_exclusive_json(fixed_attestation_path, fixed_attestation)
    source_quote_coverage_proved = receipt["parent_quote_coverage_complete"] is True
    independent_reexecution_passed = (
        fixed_attestation["status"] == "VERIFIED_COMPLETE"
        and fixed_attestation["downstream_terminal_reduction_allowed"] is True
        and failed_count == 0
    )
    official_evidence_eligible = (
        independent_reexecution_passed and source_quote_coverage_proved
    )
    body = {
        "contract": ECONOMIC_JOB_RESULT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "runner_handoff_sha256": handoff["runner_handoff_sha256"],
        "plan_sha256": verified_plan["plan_sha256"],
        "implementation_binding_sha256": verified_plan["implementation_binding"][
            "implementation_binding_sha256"
        ],
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "source_slice_receipt_sha256": receipt["source_slice_receipt_sha256"],
        "source_row_count": row_count,
        "quote_batch_count": batch_count,
        "batch_chain_sha256": batch_chain,
        "source_open_count": 1,
        "fanout_barrier_policy": "ALL_POST_EXIT_SNAPSHOTS_THEN_ALL_PROPOSALS_THEN_ALL_REDUCTIONS",
        "coordinate_results": cells,
        "coordinate_result_count": len(cells),
        "complete_coordinate_count": complete_count,
        "failed_coordinate_count": failed_count,
        "job_status": "COMPLETE" if failed_count == 0 else "INCOMPLETE_FAILED",
        # Never expose a cherry-pickable portfolio summary for a mixed fixed
        # denominator.  Per-cell receipts remain solely so the execution state
        # machine can retain every success/failure and seal a FAILED terminal.
        "portfolio_results_by_coordinate": (
            portfolio_results if failed_count == 0 else {}
        ),
        "economic_carry_states_by_slot": (
            economic_carries if failed_count == 0 else {}
        ),
        "compact_evidence_rows": evidence_rows if failed_count == 0 else [],
        "worker_runtime_binding_sha256": runtime_sha,
        "coordinate_runtime_bindings_sha256": canonical_sha256(
            [
                {
                    "coordinate_id": coordinate_id,
                    "portfolio_policy_binding_sha256": runtimes[coordinate_id][
                        "portfolio_policy_binding_sha256"
                    ],
                }
                for coordinate_id in handoff["runnable_coordinate_ids"]
            ]
        ),
        "worker_capability_isolation_enforced": True,
        "external_worker_code_loaded": False,
        "worker_runtime_mode": worker_runtime_mode,
        "worker_runtime_seal_sha256": (
            verified_runtime_seal["runtime_binding_sha256"]
            if verified_runtime_seal is not None
            else None
        ),
        "worker_dependency_manifest_sha256": (
            verified_runtime_seal["dependencies_sha256"]
            if verified_runtime_seal is not None
            else None
        ),
        "partial_economics_reported": False,
        "downstream_terminal_reduction_allowed": failed_count == 0,
        "economic_transcript_available": True,
        "economic_transcript_artifacts": transcript_artifacts,
        "economic_transcript_artifacts_sha256": canonical_sha256(transcript_artifacts),
        "fixed_denominator_reexecution_attestation": fixed_attestation,
        "fixed_denominator_reexecution_attestation_filename": (
            fixed_attestation_path.name
        ),
        "fixed_denominator_reexecution_attestation_sha256": fixed_attestation[
            "fixed_denominator_attestation_sha256"
        ],
        "independent_economic_reexecution_passed": (independent_reexecution_passed),
        "source_quote_coverage_proved": source_quote_coverage_proved,
        "official_evidence_eligible": official_evidence_eligible,
        "proof_classification": (
            "INDEPENDENTLY_REEXECUTED_WORN_TRAIN"
            if official_evidence_eligible
            else "FAIL_CLOSED_INCOMPLETE_ECONOMIC_EVIDENCE"
        ),
        "authority": _authority(),
    }
    result = {**body, "economic_job_result_sha256": canonical_sha256(body)}
    if len(_canonical_bytes(result)) > MAX_RESULT_BYTES:
        raise DojoLongHorizonEconomicRunnerError(
            "economic job result exceeds the bounded control artifact size"
        )
    return result


__all__ = [
    "BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256",
    "ECONOMIC_CARRY_CONTRACT",
    "ECONOMIC_JOB_RESULT_CONTRACT",
    "SOURCE_SLICE_CONTRACT",
    "DojoLongHorizonEconomicRunnerError",
    "EconomicWorkerRuntime",
    "WorkerRuntimeFactory",
    "builtin_no_intent_runtime_factory",
    "builtin_strategy_runtime_factory",
    "build_month_source_slice_receipt",
    "run_long_horizon_economic_job",
    "validate_month_source_slice_receipt",
]
