"""Independent proposal reexecution for ``room-meta-01``.

The economic transcript already proves that each post-exit snapshot is an
exact consequence of its quote stream and portfolio state.  This scorer binds
that successful economic attestation to the same immutable transcript, then
replays the sealed tuned generation and anomaly-admission wrapper from those
snapshots.  Every recomputed all-worker proposal batch must equal the batch in
the transcript and the final hidden-candidate evidence-chain summary must
equal the runner's summary.

This proves decision reconstruction, including HOLD/resize/next-rank choices.
It deliberately does not claim the P/L a held candidate would have earned;
that needs a separate counterfactual portfolio replay.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_anomaly_admission_runtime import (
    EVIDENCE_SUMMARY_CONTRACT,
    build_anomaly_admission_runtime_factory,
    verify_anomaly_admission_runtime_seal,
)
from quant_rabbit.dojo_economic_transcript import (
    REEXECUTION_ATTESTATION_CONTRACT,
    TRANSCRIPT_HEADER_CONTRACT,
    TRANSCRIPT_RECORD_CONTRACT,
    verify_economic_transcript_header,
)
from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    readonly_post_exit_snapshot,
    seal_worker_proposal,
    seal_worker_proposal_batch,
    verify_post_exit_snapshot,
    verify_worker_proposal_batch,
)


ATTESTATION_CONTRACT: Final = "QR_DOJO_ANOMALY_DECISION_REEXECUTION_ATTESTATION_V1"
REQUEST_CONTRACT: Final = "QR_DOJO_ANOMALY_DECISION_REEXECUTION_REQUEST_V1"
SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64
MAX_RECORD_BYTES: Final = 32 * 1024 * 1024
MAX_RECORD_COUNT: Final = 50_000_000
MAX_TRANSCRIPT_BYTES: Final = 512 * 1024 * 1024 * 1024
MAX_REQUEST_BYTES: Final = 64 * 1024 * 1024
MAX_DEPENDENCY_BYTES: Final = 16 * 1024 * 1024
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GRANULARITY_SECONDS: Final = {"M1": 60, "M5": 300}
_RECORD_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_id",
        "record_index",
        "previous_record_sha256",
        "event_type",
        "payload",
        "record_sha256",
    }
)
_EVENT_TYPES: Final = frozenset(
    {
        "HEADER",
        "QUOTE_BATCH",
        "POST_EXIT_SNAPSHOT",
        "WORKER_PROPOSAL_BATCH",
        "ALLOCATION_RECEIPT",
        "TERMINAL_SUCCESS",
        "TERMINAL_FAILURE",
    }
)
_SUCCESS_ATTESTATION_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "status",
        "transcript_id",
        "coordinate_id",
        "header_sha256",
        "terminal_record_sha256",
        "transcript_file_sha256",
        "completed_coordinate_count",
        "source_batch_chain_sha256",
        "portfolio_result_sha256",
        "portfolio_carry_state_sha256",
        "transcript_integrity_passed",
        "independent_economic_reexecution_passed",
        "partial_economics_reported",
        "official_evidence_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "reexecution_attestation_sha256",
    }
)
_REQUEST_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_path",
        "economic_attestation",
        "runtime_seal",
        "job",
        "predecessor_economic_carry",
        "expected_runtime_evidence_summary",
        "request_sha256",
    }
)
_DEPENDENCY_PATHS: Final = (
    "src/quant_rabbit/dojo_anomaly_admission_controller.py",
    "src/quant_rabbit/dojo_anomaly_admission_runtime.py",
    "src/quant_rabbit/dojo_anomaly_decision_scorer.py",
    "src/quant_rabbit/dojo_economic_transcript.py",
    "src/quant_rabbit/dojo_shared_worker_protocol.py",
    "src/quant_rabbit/dojo_tuned_strategy_runtime.py",
)
_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
}


class DojoAnomalyDecisionScorerError(ValueError):
    """The transcript, runtime lineage, or recomputed decision differed."""


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
        raise DojoAnomalyDecisionScorerError(
            "value is not strict canonical JSON"
        ) from exc


def _copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoAnomalyDecisionScorerError(f"{field} must be an object")
    return _copy(value)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoAnomalyDecisionScorerError(f"{field} must be an array")
    return _copy(list(value))


def _sha(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoAnomalyDecisionScorerError(f"{field} must be a SHA-256")
    if not allow_zero and value == GENESIS_SHA256:
        raise DojoAnomalyDecisionScorerError(f"{field} must not be zero")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoAnomalyDecisionScorerError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _strict_json(raw: bytes, *, field: str) -> Any:
    def reject_constant(token: str) -> None:
        raise DojoAnomalyDecisionScorerError(
            f"non-finite token is forbidden in {field}: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoAnomalyDecisionScorerError(
                    f"duplicate key in {field}: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoAnomalyDecisionScorerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoAnomalyDecisionScorerError(f"invalid JSON in {field}") from exc


def _read_stable_file(path: Path, *, maximum_bytes: int, field: str) -> bytes:
    before = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise DojoAnomalyDecisionScorerError(
            f"{field} must be a bounded single-link regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    size = 0
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        while block := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size)):
            chunks.append(block)
            size += len(block)
            if size > maximum_bytes:
                raise DojoAnomalyDecisionScorerError(f"{field} exceeds its bound")
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identities = {
        (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
        for state in (before, opened, current)
    }
    if len(identities) != 1 or size != before.st_size:
        raise DojoAnomalyDecisionScorerError(f"{field} changed while reading")
    return b"".join(chunks)


def _economic_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(value, field="economic_attestation")
    if set(row) != set(_SUCCESS_ATTESTATION_KEYS):
        raise DojoAnomalyDecisionScorerError(
            "economic attestation success schema mismatch"
        )
    claimed = row["reexecution_attestation_sha256"]
    body = {
        key: item for key, item in row.items() if key != "reexecution_attestation_sha256"
    }
    if (
        row["contract"] != REEXECUTION_ATTESTATION_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["status"] != "VERIFIED_COMPLETE"
        or row["transcript_integrity_passed"] is not True
        or row["independent_economic_reexecution_passed"] is not True
        or row["partial_economics_reported"] is not False
        or row["official_evidence_eligible"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
        or canonical_portfolio_sha256(body)
        != _sha(claimed, field="economic attestation digest")
    ):
        raise DojoAnomalyDecisionScorerError(
            "economic attestation is not a verified complete replay"
        )
    for field in (
        "header_sha256",
        "terminal_record_sha256",
        "transcript_file_sha256",
        "source_batch_chain_sha256",
        "portfolio_result_sha256",
        "portfolio_carry_state_sha256",
    ):
        _sha(row[field], field=f"economic_attestation.{field}")
    _integer(
        row["completed_coordinate_count"],
        field="economic_attestation.completed_coordinate_count",
        minimum=1,
    )
    return row


def _job_context(
    value: Mapping[str, Any],
    *,
    coordinate_id: str,
    runtime_seal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, str]]]:
    job = _mapping(value, field="job")
    job_sha = _sha(job.get("job_sha256"), field="job.job_sha256")
    if job_sha != canonical_portfolio_sha256(
        {key: item for key, item in job.items() if key != "job_sha256"}
    ):
        raise DojoAnomalyDecisionScorerError("job digest is invalid")
    granularity = job.get("granularity")
    if granularity not in _GRANULARITY_SECONDS:
        raise DojoAnomalyDecisionScorerError("job granularity is unsupported")
    feed_pairs = _sequence(job.get("feed_pairs"), field="job.feed_pairs")
    formal_pairs = list(runtime_seal["formal_pair_universe"])
    if (
        len(feed_pairs) != len(formal_pairs)
        or set(feed_pairs) != set(formal_pairs)
    ):
        raise DojoAnomalyDecisionScorerError(
            "job feed is not the sealed exact G8 universe"
        )
    coordinates = _sequence(job.get("coordinates"), field="job.coordinates")
    matches = [
        _mapping(row, field="job coordinate")
        for row in coordinates
        if isinstance(row, Mapping) and row.get("coordinate_id") == coordinate_id
    ]
    if len(matches) != 1:
        raise DojoAnomalyDecisionScorerError(
            "job does not contain exactly one transcript coordinate"
        )
    coordinate = matches[0]
    trade_mask = coordinate.get("trade_pair_mask")
    worker_mask = coordinate.get("active_worker_mask")
    catalog = list(runtime_seal["worker_catalog"])
    if (
        not isinstance(trade_mask, str)
        or len(trade_mask) != len(feed_pairs)
        or set(trade_mask) - {"0", "1"}
        or not isinstance(worker_mask, str)
        or len(worker_mask) != len(catalog)
        or set(worker_mask) - {"0", "1"}
    ):
        raise DojoAnomalyDecisionScorerError("coordinate masks are invalid")
    trade_pairs = [
        pair for pair, bit in zip(feed_pairs, trade_mask, strict=True) if bit == "1"
    ]
    active_bindings = [
        _copy(binding)
        for binding, bit in zip(catalog, worker_mask, strict=True)
        if bit == "1"
    ]
    context = {
        **coordinate,
        "trade_pairs": trade_pairs,
        "granularity": granularity,
        "bar_seconds": _GRANULARITY_SECONDS[granularity],
    }
    return job, context, trade_pairs, active_bindings


def _prior_worker_state(
    value: Mapping[str, Any] | None,
    *,
    header: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    runtime_binding_sha256: str,
) -> Any | None:
    predecessor_sha = header["input_bindings"]["predecessor_state_sha256"]
    slot = coordinate.get("predecessor_state_slot_id")
    if value is None:
        if predecessor_sha is not None or slot is not None:
            raise DojoAnomalyDecisionScorerError(
                "predecessor worker state is missing"
            )
        return None
    carry = _mapping(value, field="predecessor_economic_carry")
    claimed = _sha(carry.get("state_sha256"), field="economic carry state")
    body = {key: item for key, item in carry.items() if key != "state_sha256"}
    portfolio_carry = _mapping(
        carry.get("portfolio_carry_state"), field="portfolio carry"
    )
    if (
        claimed != canonical_portfolio_sha256(body)
        or predecessor_sha != claimed
        or slot is None
        or carry.get("state_slot_id") != slot
        or carry.get("worker_runtime_binding_sha256") != runtime_binding_sha256
        or carry.get("portfolio_policy_sha256") != header["policy_sha256"]
        or carry.get("source_batch_chain_sha256")
        != header["input_bindings"]["predecessor_source_batch_chain_sha256"]
        or portfolio_carry.get("carry_state_sha256")
        != header["input_bindings"][
            "predecessor_portfolio_carry_state_sha256"
        ]
        or carry.get("worker_state_sha256")
        != canonical_portfolio_sha256(carry.get("worker_state"))
    ):
        raise DojoAnomalyDecisionScorerError(
            "predecessor economic carry binding is invalid"
        )
    return _copy(carry["worker_state"])


def _read_decision_records(
    path: Path, *, expected_file_sha256: str
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]], str]:
    transcript = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(transcript, flags)
    digest = hashlib.sha256()
    header: dict[str, Any] | None = None
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    snapshot: dict[str, Any] | None = None
    stage = "EXPECT_HEADER"
    previous = GENESIS_SHA256
    index = 0
    terminal_sha: str | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size <= 0
            or opened.st_size > MAX_TRANSCRIPT_BYTES
        ):
            raise DojoAnomalyDecisionScorerError(
                "transcript must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while raw := handle.readline(MAX_RECORD_BYTES + 1):
                if len(raw) > MAX_RECORD_BYTES or index >= MAX_RECORD_COUNT:
                    raise DojoAnomalyDecisionScorerError(
                        "transcript record bound exceeded"
                    )
                if terminal_sha is not None:
                    raise DojoAnomalyDecisionScorerError(
                        "record exists after terminal"
                    )
                digest.update(raw)
                record = _mapping(
                    _strict_json(raw, field=f"transcript line {index + 1}"),
                    field="transcript record",
                )
                if set(record) != set(_RECORD_KEYS):
                    raise DojoAnomalyDecisionScorerError(
                        "transcript record schema mismatch"
                    )
                body = {
                    key: item for key, item in record.items() if key != "record_sha256"
                }
                if (
                    record["contract"] != TRANSCRIPT_RECORD_CONTRACT
                    or record["schema_version"] != SCHEMA_VERSION
                    or record["record_index"] != index
                    or record["previous_record_sha256"] != previous
                    or record["event_type"] not in _EVENT_TYPES
                    or record["record_sha256"] != canonical_portfolio_sha256(body)
                ):
                    raise DojoAnomalyDecisionScorerError(
                        "transcript record chain is invalid"
                    )
                event = record["event_type"]
                payload = _mapping(record["payload"], field="record payload")
                if event == "HEADER":
                    if stage != "EXPECT_HEADER" or index != 0:
                        raise DojoAnomalyDecisionScorerError("header is out of order")
                    header = verify_economic_transcript_header(payload)
                    if record["transcript_id"] != header["transcript_id"]:
                        raise DojoAnomalyDecisionScorerError(
                            "header transcript identity drifted"
                        )
                    stage = "EXPECT_QUOTE"
                elif header is not None and record["transcript_id"] != header[
                    "transcript_id"
                ]:
                    raise DojoAnomalyDecisionScorerError(
                        "record transcript identity drifted"
                    )
                elif event == "QUOTE_BATCH":
                    if stage != "EXPECT_QUOTE" or header is None:
                        raise DojoAnomalyDecisionScorerError("quote is out of order")
                    stage = "EXPECT_SNAPSHOT"
                elif event == "POST_EXIT_SNAPSHOT":
                    if stage != "EXPECT_SNAPSHOT" or set(payload) != {"snapshot"}:
                        raise DojoAnomalyDecisionScorerError(
                            "snapshot is out of order"
                        )
                    try:
                        snapshot = verify_post_exit_snapshot(payload["snapshot"])
                    except ProtocolViolation as exc:
                        raise DojoAnomalyDecisionScorerError(
                            "snapshot protocol is invalid"
                        ) from exc
                    stage = "EXPECT_PROPOSAL"
                elif event == "WORKER_PROPOSAL_BATCH":
                    if (
                        stage != "EXPECT_PROPOSAL"
                        or snapshot is None
                        or set(payload) != {"proposal_batch"}
                    ):
                        raise DojoAnomalyDecisionScorerError(
                            "proposal batch is out of order"
                        )
                    try:
                        proposal = verify_worker_proposal_batch(
                            snapshot, payload["proposal_batch"]
                        )
                    except ProtocolViolation as exc:
                        raise DojoAnomalyDecisionScorerError(
                            "proposal batch protocol is invalid"
                        ) from exc
                    pairs.append((snapshot, proposal))
                    stage = "EXPECT_ALLOCATION"
                elif event == "ALLOCATION_RECEIPT":
                    if stage != "EXPECT_ALLOCATION":
                        raise DojoAnomalyDecisionScorerError(
                            "allocation is out of order"
                        )
                    snapshot = None
                    stage = "EXPECT_QUOTE"
                elif event == "TERMINAL_SUCCESS":
                    if stage != "EXPECT_QUOTE" or header is None or not pairs:
                        raise DojoAnomalyDecisionScorerError(
                            "terminal success is out of order"
                        )
                    terminal_sha = record["record_sha256"]
                    stage = "TERMINAL"
                else:
                    raise DojoAnomalyDecisionScorerError(
                        "decision scorer requires a complete transcript"
                    )
                previous = record["record_sha256"]
                index += 1
            after = os.fstat(handle.fileno())
        current = transcript.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise DojoAnomalyDecisionScorerError(
                "transcript changed during decision scoring"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    file_sha = digest.hexdigest()
    if (
        header is None
        or terminal_sha is None
        or stage != "TERMINAL"
        or file_sha != expected_file_sha256
    ):
        raise DojoAnomalyDecisionScorerError(
            "transcript terminal or expected file digest is invalid"
        )
    return header, pairs, terminal_sha


def _recomputed_batch(
    runtime: Any,
    *,
    snapshot: Mapping[str, Any],
    trade_pairs: Sequence[str],
) -> dict[str, Any]:
    readonly = readonly_post_exit_snapshot(snapshot)
    raw = runtime.propose(readonly)
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise DojoAnomalyDecisionScorerError(
            "runtime proposal output is not an array"
        )
    allowed = set(trade_pairs)
    try:
        proposals = [seal_worker_proposal(snapshot, row) for row in raw]
        if any(
            intent["parameters"]["pair"] not in allowed
            for proposal in proposals
            for intent in proposal["new_risk_intents"]
        ):
            raise DojoAnomalyDecisionScorerError(
                "runtime proposed outside the sealed trade mask"
            )
        return seal_worker_proposal_batch(snapshot, proposals)
    except ProtocolViolation as exc:
        raise DojoAnomalyDecisionScorerError(
            "recomputed proposal violated the worker protocol"
        ) from exc


def _dependency_manifest(repo_root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for relative_path in _DEPENDENCY_PATHS:
        path = repo_root / relative_path
        payload = _read_stable_file(
            path,
            maximum_bytes=MAX_DEPENDENCY_BYTES,
            field=f"scorer dependency {relative_path}",
        )
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows, canonical_portfolio_sha256(rows)


def score_anomaly_decision_transcript(
    transcript_path: Path,
    *,
    economic_attestation: Mapping[str, Any],
    runtime_seal: Mapping[str, Any],
    job: Mapping[str, Any],
    predecessor_economic_carry: Mapping[str, Any] | None,
    expected_runtime_evidence_summary: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Reexecute one complete coordinate's sealed anomaly decisions."""

    root = Path(repo_root).resolve(strict=True)
    if root != Path(__file__).resolve().parents[2]:
        raise DojoAnomalyDecisionScorerError(
            "repo_root must be the source tree that loaded the scorer"
        )
    seal = verify_anomaly_admission_runtime_seal(runtime_seal, repo_root=root)
    economic = _economic_attestation(economic_attestation)
    header, recorded_pairs, terminal_sha = _read_decision_records(
        transcript_path,
        expected_file_sha256=economic["transcript_file_sha256"],
    )
    if (
        header["contract"] != TRANSCRIPT_HEADER_CONTRACT
        or header["coordinate_id"] != economic["coordinate_id"]
        or header["transcript_id"] != economic["transcript_id"]
        or header["header_sha256"] != economic["header_sha256"]
        or terminal_sha != economic["terminal_record_sha256"]
        or len(recorded_pairs) != economic["completed_coordinate_count"]
        or header["input_bindings"]["worker_runtime_binding_sha256"]
        != seal["runtime_binding_sha256"]
    ):
        raise DojoAnomalyDecisionScorerError(
            "economic attestation, transcript, and runtime seal are not co-bound"
        )
    verified_job, coordinate, trade_pairs, active_bindings = _job_context(
        job,
        coordinate_id=header["coordinate_id"],
        runtime_seal=seal,
    )
    if (
        verified_job["job_sha256"] != header["input_bindings"]["job_sha256"]
        or sorted(trade_pairs) != header["portfolio_policy"]["tradable_pairs"]
        or sorted(active_bindings, key=lambda row: row["worker_id"])
        != header["portfolio_policy"]["active_worker_bindings"]
    ):
        raise DojoAnomalyDecisionScorerError(
            "job coordinate differs from the transcript denominator"
        )
    prior_state = _prior_worker_state(
        predecessor_economic_carry,
        header=header,
        coordinate=coordinate,
        runtime_binding_sha256=seal["runtime_binding_sha256"],
    )
    runtime = build_anomaly_admission_runtime_factory(
        seal, repo_root=root
    )(coordinate, active_bindings, prior_state)
    batch_chain = GENESIS_SHA256
    for snapshot, recorded_batch in recorded_pairs:
        if snapshot["intrabar"] != verified_job["intrabar_path"]:
            raise DojoAnomalyDecisionScorerError(
                "snapshot intrabar path differs from the sealed job"
            )
        recomputed = _recomputed_batch(
            runtime, snapshot=snapshot, trade_pairs=trade_pairs
        )
        if recomputed != recorded_batch:
            raise DojoAnomalyDecisionScorerError(
                "independently recomputed proposal batch differs from transcript"
            )
        batch_chain = canonical_portfolio_sha256(
            {
                "previous_proposal_batch_chain_sha256": batch_chain,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "proposal_batch_sha256": recomputed["batch_sha256"],
            }
        )
    expected_summary = _mapping(
        expected_runtime_evidence_summary,
        field="expected_runtime_evidence_summary",
    )
    summary_body = {
        key: item
        for key, item in expected_summary.items()
        if key != "evidence_summary_sha256"
    }
    if (
        expected_summary.get("contract") != EVIDENCE_SUMMARY_CONTRACT
        or expected_summary.get("runtime_binding_sha256")
        != seal["runtime_binding_sha256"]
        or expected_summary.get("evidence_summary_sha256")
        != canonical_portfolio_sha256(summary_body)
    ):
        raise DojoAnomalyDecisionScorerError(
            "expected runtime evidence summary is invalid"
        )
    recomputed_summary = runtime.export_admission_evidence()
    if recomputed_summary != expected_summary:
        raise DojoAnomalyDecisionScorerError(
            "hidden candidate/HOLD evidence summary differs from reexecution"
        )
    dependencies, dependencies_sha = _dependency_manifest(root)
    body = {
        "contract": ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_COMPLETE",
        "coordinate_id": header["coordinate_id"],
        "job_sha256": verified_job["job_sha256"],
        "transcript_id": header["transcript_id"],
        "transcript_file_sha256": economic["transcript_file_sha256"],
        "economic_reexecution_attestation_sha256": economic[
            "reexecution_attestation_sha256"
        ],
        "runtime_binding_sha256": seal["runtime_binding_sha256"],
        "policy_sha256": seal["policy_sha256"],
        "proposal_batch_count": len(recorded_pairs),
        "proposal_batch_chain_sha256": batch_chain,
        "runtime_evidence_summary_sha256": recomputed_summary[
            "evidence_summary_sha256"
        ],
        "scorer_dependencies": dependencies,
        "scorer_dependencies_sha256": dependencies_sha,
        "strategy_decision_reexecution_passed": True,
        "hidden_upstream_candidate_reconstruction_passed": True,
        "hold_resize_next_rank_reconstruction_passed": True,
        "held_economic_counterfactual_reexecution_passed": False,
        "official_evidence_eligible": False,
        "authority": _copy(_AUTHORITY),
    }
    return {
        **body,
        "decision_reexecution_attestation_sha256": canonical_portfolio_sha256(body),
    }


def _read_request(path: Path) -> dict[str, Any]:
    request_path = Path(path)
    row = _mapping(
        _strict_json(
            _read_stable_file(
                request_path,
                maximum_bytes=MAX_REQUEST_BYTES,
                field="decision scorer request",
            ),
            field="decision scorer request",
        ),
        field="decision scorer request",
    )
    if set(row) != set(_REQUEST_KEYS):
        raise DojoAnomalyDecisionScorerError("request schema mismatch")
    claimed = row["request_sha256"]
    body = {key: item for key, item in row.items() if key != "request_sha256"}
    if (
        row["contract"] != REQUEST_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or claimed != canonical_portfolio_sha256(body)
    ):
        raise DojoAnomalyDecisionScorerError("request seal is invalid")
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args(argv)
    request = _read_request(args.request)
    result = score_anomaly_decision_transcript(
        Path(request["transcript_path"]),
        economic_attestation=request["economic_attestation"],
        runtime_seal=request["runtime_seal"],
        job=request["job"],
        predecessor_economic_carry=request["predecessor_economic_carry"],
        expected_runtime_evidence_summary=request[
            "expected_runtime_evidence_summary"
        ],
        repo_root=args.repo_root,
    )
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_CONTRACT",
    "DojoAnomalyDecisionScorerError",
    "REQUEST_CONTRACT",
    "score_anomaly_decision_transcript",
]
