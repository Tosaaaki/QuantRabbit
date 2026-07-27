"""Content-addressed Codex decision queue for paired DOJO replay.

The queue exposes exactly one causal packet at a time.  It is intentionally
separate from the broker/runtime surfaces: accepted responses can only choose
from the inventory-supervision action allowlist and carry invariant
``live_permission=false``, ``broker_mutation_allowed=false`` and
``order_authority=NONE``.

This first version is a provider-executor pipeline proof.  It consumes the
already sealed causal checkpoint packets from the immutable r13 paired
diagnostic results, but it does not rewrite those results or claim that the
accepted response has been applied to economic replay.  That later application
step must preserve reducer checkpoints and the fixed 84-cell denominator.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_paired_inventory_counterfactual import (
    ACTION_IDS,
    AUTHORITY,
    CADENCE_IDS,
)


QUEUE_PLAN_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_QUEUE_PLAN_V1"
DECISION_PACKET_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_DECISION_PACKET_V1"
MODEL_RESPONSE_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_RESPONSE_V1"
QUEUE_EVENT_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_QUEUE_EVENT_V1"
QUEUE_STATUS_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_QUEUE_STATUS_V1"
PREFLIGHT_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_PREFLIGHT_V1"
QUOTA_HALT_CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_QUOTA_HALT_V1"
SCHEMA_VERSION: Final = 1
ZERO_SHA256: Final = "0" * 64
MAX_JSON_BYTES: Final = 2 * 1024 * 1024
EXPECTED_CELL_COUNT: Final = 84
HALT_SENTINEL_FILENAME: Final = "runtime-quota-halt.json"
HALT_STATES: Final = frozenset({"HALTED_QUOTA", "PAUSE_REQUESTED"})


class DojoPairedModelQueueError(ValueError):
    """The queue, causal packet, response, or event chain is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DojoPairedModelQueueError(f"{label} must be lowercase SHA-256")
    return value


def _read_json(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0 or stat.st_size > maximum_bytes:
        raise DojoPairedModelQueueError(f"JSON file size is invalid: {path}")
    with resolved.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DojoPairedModelQueueError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_or_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise DojoPairedModelQueueError(f"immutable artifact conflicts: {path}")
        return
    _write_exclusive(path, value)


def _aware_utc(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DojoPairedModelQueueError(f"{label} must be an aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoPairedModelQueueError(
            f"{label} must be an aware UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoPairedModelQueueError(f"{label} must be an aware UTC timestamp")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _halt_path(queue_dir: Path) -> Path:
    return queue_dir / HALT_SENTINEL_FILENAME


def _verify_halt_sentinel(value: Mapping[str, Any]) -> dict[str, Any]:
    sentinel = dict(value)
    expected_keys = {
        "contract",
        "schema_version",
        "state",
        "reason",
        "observed_at_utc",
        "last_accepted_model_decision_count",
        "current_ready_packet_sha256",
        "accepted_state_mutated",
        "resume_policy",
        "authority",
        "sentinel_sha256",
    }
    unsigned = {
        key: item for key, item in sentinel.items() if key != "sentinel_sha256"
    }
    reason = sentinel.get("reason")
    ready_sha = sentinel.get("current_ready_packet_sha256")
    if (
        set(sentinel) != expected_keys
        or sentinel.get("contract") != QUOTA_HALT_CONTRACT
        or sentinel.get("schema_version") != SCHEMA_VERSION
        or sentinel.get("state") not in HALT_STATES
        or not isinstance(reason, str)
        or not reason
        or len(reason) > 512
        or not isinstance(
            sentinel.get("last_accepted_model_decision_count"), int
        )
        or sentinel.get("last_accepted_model_decision_count", -1) < 0
        or sentinel.get("accepted_state_mutated") is not False
        or sentinel.get("resume_policy")
        != "EXPLICIT_RESUME_COMMAND_ONLY_SAME_READY_PACKET"
        or sentinel.get("sentinel_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoPairedModelQueueError("quota halt sentinel is invalid")
    _aware_utc(str(sentinel.get("observed_at_utc")), "observed_at_utc")
    if ready_sha is not None:
        _sha256_text(ready_sha, "halt ready packet SHA-256")
    _validate_authority(sentinel.get("authority"))
    return sentinel


def _halt_sentinel_unlocked(queue_dir: Path) -> dict[str, Any] | None:
    path = _halt_path(queue_dir)
    if not path.exists():
        return None
    return _verify_halt_sentinel(_read_json(path))


def _lock(queue_dir: Path) -> tuple[int, Path]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / ".queue.lock"
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor, path


def _unlock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _validate_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != dict(AUTHORITY):
        raise DojoPairedModelQueueError("authority boundary is invalid")
    return dict(value)


def _validate_causal_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoPairedModelQueueError("decision state packet must be an object")
    packet = dict(value)
    required = {
        "decision_epoch",
        "input_available_through_epoch",
        "phase",
        "equity_jpy",
        "balance_jpy",
        "drawdown_fraction",
        "margin_utilization_fraction",
        "gross_exposure_jpy",
        "net_exposure_jpy",
        "long_gross_exposure_jpy",
        "short_gross_exposure_jpy",
        "hedge_buildup_fraction",
        "directional_skew_fraction",
        "unrealized_pnl_jpy",
        "realized_profit_giveback_jpy",
        "position_count",
        "pending_order_count",
        "stale_valuation_pair_count",
        "maximum_position_age_seconds",
        "consecutive_losses",
        "regime_id",
        "strategy_regime_compatible",
        "paused",
        "direction_block",
        "terminal_result_visible",
        "future_quote_visible",
        "append_wall_clock_visible",
    }
    if set(packet) != required:
        raise DojoPairedModelQueueError("decision state packet schema mismatch")
    epoch = packet["decision_epoch"]
    if (
        not isinstance(epoch, int)
        or packet["input_available_through_epoch"] != epoch
        or packet["terminal_result_visible"] is not False
        or packet["future_quote_visible"] is not False
        or packet["append_wall_clock_visible"] is not False
    ):
        raise DojoPairedModelQueueError("decision state packet is not causal")
    if packet["phase"] != "C":
        raise DojoPairedModelQueueError("model checkpoint must be a candle close")
    return packet


def verify_decision_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = dict(value)
    expected_keys = {
        "contract",
        "schema_version",
        "study_id",
        "cell_id",
        "cell_ordinal",
        "coordinate_id",
        "family_id",
        "cost_scenario",
        "cadence_id",
        "source_plan_sha256",
        "source_result_sha256",
        "source_decision_id",
        "decision_epoch",
        "input_available_through_epoch",
        "state_packet",
        "state_packet_sha256",
        "action_allowlist",
        "information_policy",
        "terminal_result_allowed",
        "future_quote_allowed",
        "append_wall_clock_allowed",
        "economic_application_status",
        "classification",
        "authority",
        "decision_packet_sha256",
    }
    unsigned = {
        key: item for key, item in packet.items() if key != "decision_packet_sha256"
    }
    state_packet = _validate_causal_packet(packet.get("state_packet"))
    if (
        set(packet) != expected_keys
        or packet.get("contract") != DECISION_PACKET_CONTRACT
        or packet.get("schema_version") != SCHEMA_VERSION
        or packet.get("action_allowlist") != list(ACTION_IDS)
        or packet.get("terminal_result_allowed") is not False
        or packet.get("future_quote_allowed") is not False
        or packet.get("append_wall_clock_allowed") is not False
        or packet.get("economic_application_status")
        != "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY"
        or packet.get("classification") != "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
        or packet.get("decision_epoch") != state_packet["decision_epoch"]
        or packet.get("input_available_through_epoch")
        != state_packet["input_available_through_epoch"]
        or packet.get("state_packet_sha256") != canonical_sha256(state_packet)
        or packet.get("decision_packet_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoPairedModelQueueError("decision packet seal is invalid")
    _sha256_text(packet["source_plan_sha256"], "source_plan_sha256")
    _sha256_text(packet["source_result_sha256"], "source_result_sha256")
    _validate_authority(packet.get("authority"))
    return packet


def _build_packet(
    *,
    study_id: str,
    source_plan_sha256: str,
    result: Mapping[str, Any],
    cadence_row: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    audit_log = cadence_row.get("intervention_audit_log")
    if not isinstance(audit_log, list) or not audit_log:
        raise DojoPairedModelQueueError("paired cell has no causal decision packet")
    source = audit_log[0]
    if not isinstance(source, Mapping):
        raise DojoPairedModelQueueError("paired decision log is invalid")
    state_packet = _validate_causal_packet(source.get("packet"))
    if (
        source.get("packet_sha256") != canonical_sha256(state_packet)
        or source.get("decision_epoch") != state_packet["decision_epoch"]
        or source.get("input_available_through_epoch")
        != state_packet["input_available_through_epoch"]
        or source.get("provider_model_called") is not False
        or source.get("future_information_used") is not False
    ):
        raise DojoPairedModelQueueError("source causal decision seal is invalid")
    coordinate_id = str(result["coordinate_id"])
    cadence_id = str(cadence_row["cadence_id"])
    body = {
        "contract": DECISION_PACKET_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "cell_id": canonical_sha256(
            {
                "study_id": study_id,
                "coordinate_id": coordinate_id,
                "cadence_id": cadence_id,
            }
        ),
        "cell_ordinal": ordinal,
        "coordinate_id": coordinate_id,
        "family_id": result["family_id"],
        "cost_scenario": result["cost_scenario"],
        "cadence_id": cadence_id,
        "source_plan_sha256": source_plan_sha256,
        "source_result_sha256": result["result_sha256"],
        "source_decision_id": source["decision_id"],
        "decision_epoch": state_packet["decision_epoch"],
        "input_available_through_epoch": state_packet["input_available_through_epoch"],
        "state_packet": state_packet,
        "state_packet_sha256": canonical_sha256(state_packet),
        "action_allowlist": list(ACTION_IDS),
        "information_policy": (
            "EXECUTOR_MAY_READ_ONLY_THIS_PACKET_AND_STATIC_ACTION_CONTRACT"
        ),
        "terminal_result_allowed": False,
        "future_quote_allowed": False,
        "append_wall_clock_allowed": False,
        "economic_application_status": "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY",
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "authority": dict(AUTHORITY),
    }
    return verify_decision_packet(
        {**body, "decision_packet_sha256": canonical_sha256(body)}
    )


def build_queue_plan(
    *,
    source_plan: Mapping[str, Any],
    result_values: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_plan_sha = _sha256_text(
        source_plan.get("plan_sha256"), "source plan SHA-256"
    )
    study_id = str(source_plan.get("study_id") or "")
    if (
        canonical_sha256(
            {key: item for key, item in source_plan.items() if key != "plan_sha256"}
        )
        != source_plan_sha
        or source_plan.get("cadence_ids") != list(CADENCE_IDS)
        or source_plan.get("actual_model_checkpoint_call_required_for_rank") is not True
        or source_plan.get("terminal_result_allowed_in_decision") is not False
        or source_plan.get("future_quote_allowed") is not False
        or source_plan.get("append_wall_clock_allowed") is not False
        or source_plan.get("authority") != dict(AUTHORITY)
    ):
        raise DojoPairedModelQueueError("source paired plan boundary is invalid")
    cell_sources: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for result in result_values:
        if (
            result.get("result_sha256")
            != canonical_sha256(
                {key: item for key, item in result.items() if key != "result_sha256"}
            )
            or result.get("classification") != "EXPERIMENTAL_UNRANKED"
            or result.get("authority") != dict(AUTHORITY)
            or result.get("plan_sha256") != source_plan_sha
        ):
            raise DojoPairedModelQueueError("source paired result is invalid")
        cadence_rows = result.get("cadence_rows")
        if (
            not isinstance(cadence_rows, list)
            or len(cadence_rows) != len(CADENCE_IDS)
            or {row.get("cadence_id") for row in cadence_rows} != set(CADENCE_IDS)
        ):
            raise DojoPairedModelQueueError("source cadence denominator is invalid")
        for row in cadence_rows:
            if not isinstance(row, Mapping):
                raise DojoPairedModelQueueError("source cadence row is invalid")
            cell_sources.append(
                (
                    str(result["coordinate_id"]),
                    str(row["cadence_id"]),
                    result,
                    row,
                )
            )
    cell_sources.sort(key=lambda row: (row[0], row[1]))
    if (
        len(cell_sources) != EXPECTED_CELL_COUNT
        or len({row[0] for row in cell_sources}) != 12
        or len({(row[0], row[1]) for row in cell_sources}) != EXPECTED_CELL_COUNT
    ):
        raise DojoPairedModelQueueError("exact 84-cell denominator is required")
    packets = [
        _build_packet(
            study_id=study_id,
            source_plan_sha256=source_plan_sha,
            result=result,
            cadence_row=row,
            ordinal=index,
        )
        for index, (_, _, result, row) in enumerate(cell_sources, start=1)
    ]
    cells = [
        {
            "cell_ordinal": packet["cell_ordinal"],
            "cell_id": packet["cell_id"],
            "coordinate_id": packet["coordinate_id"],
            "cadence_id": packet["cadence_id"],
            "decision_packet_sha256": packet["decision_packet_sha256"],
            "packet_filename": (
                f"{packet['cell_ordinal']:03d}-"
                f"{packet['decision_packet_sha256']}.json"
            ),
        }
        for packet in packets
    ]
    body = {
        "contract": QUEUE_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "source_plan_sha256": source_plan_sha,
        "cell_count": len(cells),
        "cells": cells,
        "ready_policy": "EXACTLY_ONE_UNSEEN_CONTENT_HASH_AT_A_TIME",
        "duplicate_policy": "IDEMPOTENT_BY_DECISION_PACKET_SHA256",
        "idle_model_execution_allowed": False,
        "economic_application_status": "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY",
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "authority": dict(AUTHORITY),
    }
    plan = {**body, "queue_plan_sha256": canonical_sha256(body)}
    return plan, packets


def _verify_queue_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(value)
    unsigned = {key: item for key, item in plan.items() if key != "queue_plan_sha256"}
    cells = plan.get("cells")
    if (
        plan.get("contract") != QUEUE_PLAN_CONTRACT
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("cell_count") != EXPECTED_CELL_COUNT
        or not isinstance(cells, list)
        or len(cells) != EXPECTED_CELL_COUNT
        or plan.get("idle_model_execution_allowed") is not False
        or plan.get("economic_application_status")
        != "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY"
        or plan.get("queue_plan_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoPairedModelQueueError("queue plan is invalid")
    _validate_authority(plan.get("authority"))
    for expected, cell in enumerate(cells, start=1):
        if (
            not isinstance(cell, Mapping)
            or cell.get("cell_ordinal") != expected
            or cell.get("packet_filename")
            != f"{expected:03d}-{cell.get('decision_packet_sha256')}.json"
        ):
            raise DojoPairedModelQueueError("queue cell order is invalid")
        _sha256_text(cell.get("decision_packet_sha256"), "cell packet SHA-256")
    return plan


def _event_files(queue_dir: Path) -> list[Path]:
    events_dir = queue_dir / "events"
    if not events_dir.exists():
        return []
    return sorted(events_dir.glob("*.json"))


def _verify_events(
    queue_dir: Path, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    events = []
    prior_sha = ZERO_SHA256
    for index, path in enumerate(_event_files(queue_dir), start=1):
        row = _read_json(path)
        unsigned = {key: item for key, item in row.items() if key != "event_sha256"}
        expected_name = (
            f"{index:06d}-{row.get('event_type')}-{row.get('event_sha256')}.json"
        )
        if (
            row.get("contract") != QUEUE_EVENT_CONTRACT
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("queue_plan_sha256") != plan["queue_plan_sha256"]
            or row.get("event_index") != index
            or row.get("previous_event_sha256") != prior_sha
            or row.get("event_sha256") != canonical_sha256(unsigned)
            or path.name != expected_name
        ):
            raise DojoPairedModelQueueError("queue event chain is invalid")
        events.append(row)
        prior_sha = row["event_sha256"]
    return events, prior_sha


def _append_event(
    queue_dir: Path,
    *,
    plan: Mapping[str, Any],
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    events, prior_sha = _verify_events(queue_dir, plan)
    body = {
        "contract": QUEUE_EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "queue_plan_sha256": plan["queue_plan_sha256"],
        "event_index": len(events) + 1,
        "previous_event_sha256": prior_sha,
        "event_type": event_type,
        "payload": dict(payload),
        "authority": dict(AUTHORITY),
    }
    row = {**body, "event_sha256": canonical_sha256(body)}
    path = (
        queue_dir
        / "events"
        / f"{row['event_index']:06d}-{event_type}-{row['event_sha256']}.json"
    )
    _write_exclusive(path, row)
    return row


def _derived_status_unlocked(queue_dir: Path) -> dict[str, Any]:
    plan = _verify_queue_plan(_read_json(queue_dir / "queue-plan.json"))
    events, tip = _verify_events(queue_dir, plan)
    if not events or events[0]["event_type"] != "GENESIS":
        raise DojoPairedModelQueueError("queue genesis is missing")
    ready_by_packet: dict[str, dict[str, Any]] = {}
    accepted_by_packet: dict[str, dict[str, Any]] = {}
    for row in events:
        payload = row["payload"]
        if row["event_type"] == "READY":
            packet_sha = _sha256_text(
                payload.get("decision_packet_sha256"), "ready packet SHA-256"
            )
            if packet_sha in ready_by_packet:
                raise DojoPairedModelQueueError("packet was made ready twice")
            ready_by_packet[packet_sha] = row
        elif row["event_type"] == "RESPONSE_ACCEPTED":
            packet_sha = _sha256_text(
                payload.get("decision_packet_sha256"), "accepted packet SHA-256"
            )
            if packet_sha not in ready_by_packet or packet_sha in accepted_by_packet:
                raise DojoPairedModelQueueError("response event ordering is invalid")
            accepted_by_packet[packet_sha] = row
        elif row["event_type"] != "GENESIS":
            raise DojoPairedModelQueueError("unsupported queue event type")
    outstanding = [
        packet_sha
        for packet_sha in ready_by_packet
        if packet_sha not in accepted_by_packet
    ]
    if len(outstanding) > 1:
        raise DojoPairedModelQueueError("more than one decision packet is ready")
    accepted_count = len(accepted_by_packet)
    next_ordinal = accepted_count + 1
    current = outstanding[0] if outstanding else None
    state = (
        "COMPLETE"
        if accepted_count == plan["cell_count"]
        else "WAITING_FOR_MODEL"
        if current
        else "READY_TO_EMIT"
    )
    current_cell = None
    if current:
        current_cell = next(
            dict(cell)
            for cell in plan["cells"]
            if cell["decision_packet_sha256"] == current
        )
    body = {
        "contract": QUEUE_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "queue_plan_sha256": plan["queue_plan_sha256"],
        "state": state,
        "cell_count": plan["cell_count"],
        "accepted_model_decision_count": accepted_count,
        "remaining_model_decision_count": plan["cell_count"] - accepted_count,
        "next_cell_ordinal": (
            None if accepted_count == plan["cell_count"] else next_ordinal
        ),
        "current_ready_packet_sha256": current,
        "current_ready_cell": current_cell,
        "event_count": len(events),
        "event_tip_sha256": tip,
        "idle_model_execution_allowed": False,
        "economic_application_status": "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY",
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "cryptographic_provider_signature_verified": False,
        "authority": dict(AUTHORITY),
    }
    return {**body, "status_sha256": canonical_sha256(body)}


def _status_with_halt_unlocked(queue_dir: Path) -> dict[str, Any]:
    status = _derived_status_unlocked(queue_dir)
    sentinel = _halt_sentinel_unlocked(queue_dir)
    if sentinel is None:
        return status
    body = {
        key: item for key, item in status.items() if key != "status_sha256"
    }
    body.update(
        {
            "state": sentinel["state"],
            "runtime_halt": sentinel,
            "ready_packet_withheld": True,
        }
    )
    return {**body, "status_sha256": canonical_sha256(body)}


def queue_status(queue_dir: Path) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        return _status_with_halt_unlocked(queue_dir)
    finally:
        _unlock(descriptor)


def halt_for_quota(
    queue_dir: Path,
    *,
    reason: str,
    observed_at_utc: str,
    state: str = "HALTED_QUOTA",
) -> dict[str, Any]:
    """Atomically persist the first quota pause without changing queue events."""

    descriptor, _ = _lock(queue_dir)
    try:
        existing = _halt_sentinel_unlocked(queue_dir)
        if existing is not None:
            return existing
        if state not in HALT_STATES:
            raise DojoPairedModelQueueError("quota halt state is invalid")
        if not isinstance(reason, str) or not reason or len(reason) > 512:
            raise DojoPairedModelQueueError("quota halt reason is invalid")
        status = _derived_status_unlocked(queue_dir)
        body = {
            "contract": QUOTA_HALT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "reason": reason,
            "observed_at_utc": _aware_utc(observed_at_utc, "observed_at_utc"),
            "last_accepted_model_decision_count": status[
                "accepted_model_decision_count"
            ],
            "current_ready_packet_sha256": status[
                "current_ready_packet_sha256"
            ],
            "accepted_state_mutated": False,
            "resume_policy": "EXPLICIT_RESUME_COMMAND_ONLY_SAME_READY_PACKET",
            "authority": dict(AUTHORITY),
        }
        sentinel = {**body, "sentinel_sha256": canonical_sha256(body)}
        _write_exclusive(_halt_path(queue_dir), sentinel)
        return sentinel
    finally:
        _unlock(descriptor)


def resume_quota_halt(queue_dir: Path) -> dict[str, Any]:
    """Remove only a valid halt sentinel and prove the ready packet is unchanged."""

    descriptor, _ = _lock(queue_dir)
    try:
        sentinel = _halt_sentinel_unlocked(queue_dir)
        if sentinel is None:
            return _derived_status_unlocked(queue_dir)
        status = _derived_status_unlocked(queue_dir)
        if (
            status["accepted_model_decision_count"]
            != sentinel["last_accepted_model_decision_count"]
            or status["current_ready_packet_sha256"]
            != sentinel["current_ready_packet_sha256"]
        ):
            raise DojoPairedModelQueueError(
                "queue state changed while quota halt was active"
            )
        _halt_path(queue_dir).unlink()
        return _derived_status_unlocked(queue_dir)
    finally:
        _unlock(descriptor)


def preflight_model_decision(queue_dir: Path) -> dict[str, Any]:
    """Return zero work on halt, otherwise expose at most the current packet."""

    descriptor, _ = _lock(queue_dir)
    try:
        sentinel = _halt_sentinel_unlocked(queue_dir)
        if sentinel is not None:
            body = {
                "contract": PREFLIGHT_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "state": sentinel["state"],
                "zero_work": True,
                "notification": "DONT_NOTIFY",
                "halt_sentinel_sha256": sentinel["sentinel_sha256"],
                "accepted_model_decision_count": sentinel[
                    "last_accepted_model_decision_count"
                ],
                "decision_packet": None,
                "authority": dict(AUTHORITY),
            }
            return {**body, "preflight_sha256": canonical_sha256(body)}
        status = _derived_status_unlocked(queue_dir)
        packet = None
        if status["state"] == "WAITING_FOR_MODEL":
            packet_sha = status["current_ready_packet_sha256"]
            packet = verify_decision_packet(
                _read_json(queue_dir / "ready" / f"{packet_sha}.json")
            )
        body = {
            "contract": PREFLIGHT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "state": status["state"],
            "zero_work": packet is None,
            "notification": "DONT_NOTIFY" if packet is None else None,
            "halt_sentinel_sha256": None,
            "accepted_model_decision_count": status[
                "accepted_model_decision_count"
            ],
            "decision_packet": packet,
            "authority": dict(AUTHORITY),
        }
        return {**body, "preflight_sha256": canonical_sha256(body)}
    finally:
        _unlock(descriptor)


def _publish_ready_unlocked(queue_dir: Path) -> dict[str, Any]:
    if _halt_sentinel_unlocked(queue_dir) is not None:
        return _status_with_halt_unlocked(queue_dir)
    plan = _verify_queue_plan(_read_json(queue_dir / "queue-plan.json"))
    status = _derived_status_unlocked(queue_dir)
    if status["state"] in {"WAITING_FOR_MODEL", "COMPLETE"}:
        return status
    ordinal = status["next_cell_ordinal"]
    cell = plan["cells"][ordinal - 1]
    packet_path = queue_dir / "packets" / cell["packet_filename"]
    packet = verify_decision_packet(_read_json(packet_path))
    ready_path = queue_dir / "ready" / f"{packet['decision_packet_sha256']}.json"
    _same_or_write(ready_path, packet)
    _append_event(
        queue_dir,
        plan=plan,
        event_type="READY",
        payload={
            "cell_ordinal": ordinal,
            "cell_id": packet["cell_id"],
            "decision_packet_sha256": packet["decision_packet_sha256"],
            "ready_filename": ready_path.name,
        },
    )
    return _derived_status_unlocked(queue_dir)


def emit_next_ready(queue_dir: Path) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        return _publish_ready_unlocked(queue_dir)
    finally:
        _unlock(descriptor)


def initialize_queue(
    *,
    queue_dir: Path,
    source_plan: Mapping[str, Any],
    result_values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        if (queue_dir / "queue-plan.json").exists():
            raise DojoPairedModelQueueError("queue is already initialized")
        plan, packets = build_queue_plan(
            source_plan=source_plan,
            result_values=result_values,
        )
        for cell, packet in zip(plan["cells"], packets, strict=True):
            _write_exclusive(queue_dir / "packets" / cell["packet_filename"], packet)
        _write_exclusive(queue_dir / "queue-plan.json", plan)
        _append_event(
            queue_dir,
            plan=plan,
            event_type="GENESIS",
            payload={
                "cell_count": plan["cell_count"],
                "source_plan_sha256": plan["source_plan_sha256"],
            },
        )
        return _publish_ready_unlocked(queue_dir)
    finally:
        _unlock(descriptor)


def current_ready_packet(queue_dir: Path) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        if _halt_sentinel_unlocked(queue_dir) is not None:
            raise DojoPairedModelQueueError(
                "quota halt is active; ready packet is withheld"
            )
        status = _derived_status_unlocked(queue_dir)
        packet_sha = status["current_ready_packet_sha256"]
        if not packet_sha:
            raise DojoPairedModelQueueError("no model decision packet is ready")
        return verify_decision_packet(
            _read_json(queue_dir / "ready" / f"{packet_sha}.json")
        )
    finally:
        _unlock(descriptor)


def seal_model_response(
    *,
    packet: Mapping[str, Any],
    action: str,
    reason_ids: Sequence[str],
    provider_model: str,
    provider_execution_id: str,
) -> dict[str, Any]:
    verified = verify_decision_packet(packet)
    reasons = list(reason_ids)
    if action not in ACTION_IDS:
        raise DojoPairedModelQueueError("model action is outside the allowlist")
    if (
        not reasons
        or len(reasons) > 8
        or any(
            not isinstance(reason, str) or not reason or len(reason) > 96
            for reason in reasons
        )
    ):
        raise DojoPairedModelQueueError("model response reason_ids are invalid")
    if (
        not isinstance(provider_model, str)
        or not provider_model
        or len(provider_model) > 128
        or not isinstance(provider_execution_id, str)
        or not provider_execution_id
        or len(provider_execution_id) > 256
    ):
        raise DojoPairedModelQueueError("provider execution identity is invalid")
    body = {
        "contract": MODEL_RESPONSE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "decision_packet_sha256": verified["decision_packet_sha256"],
        "cell_id": verified["cell_id"],
        "decision_epoch": verified["decision_epoch"],
        "input_available_through_epoch": verified["input_available_through_epoch"],
        "action": action,
        "reason_ids": reasons,
        "provider_model": provider_model,
        "provider_execution_id": provider_execution_id,
        "provider_execution_kind": "CODEX_MODEL_EXECUTION",
        "provider_attestation_tier": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "future_information_used": False,
        "terminal_result_used": False,
        "append_wall_clock_used": False,
        "cryptographic_signature_present": False,
        "content_seal_verified": True,
        "economic_application_status": "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY",
        "authority": dict(AUTHORITY),
    }
    return {**body, "response_sha256": canonical_sha256(body)}


def verify_model_response(
    value: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    response = dict(value)
    verified_packet = verify_decision_packet(packet)
    expected_keys = {
        "contract",
        "schema_version",
        "decision_packet_sha256",
        "cell_id",
        "decision_epoch",
        "input_available_through_epoch",
        "action",
        "reason_ids",
        "provider_model",
        "provider_execution_id",
        "provider_execution_kind",
        "provider_attestation_tier",
        "future_information_used",
        "terminal_result_used",
        "append_wall_clock_used",
        "cryptographic_signature_present",
        "content_seal_verified",
        "economic_application_status",
        "authority",
        "response_sha256",
    }
    unsigned = {key: item for key, item in response.items() if key != "response_sha256"}
    if (
        set(response) != expected_keys
        or response.get("contract") != MODEL_RESPONSE_CONTRACT
        or response.get("schema_version") != SCHEMA_VERSION
        or response.get("decision_packet_sha256")
        != verified_packet["decision_packet_sha256"]
        or response.get("cell_id") != verified_packet["cell_id"]
        or response.get("decision_epoch") != verified_packet["decision_epoch"]
        or response.get("input_available_through_epoch")
        != verified_packet["input_available_through_epoch"]
        or response.get("action") not in ACTION_IDS
        or response.get("provider_execution_kind") != "CODEX_MODEL_EXECUTION"
        or response.get("provider_attestation_tier")
        != "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
        or response.get("future_information_used") is not False
        or response.get("terminal_result_used") is not False
        or response.get("append_wall_clock_used") is not False
        or response.get("cryptographic_signature_present") is not False
        or response.get("content_seal_verified") is not True
        or response.get("economic_application_status")
        != "NOT_YET_APPLIED_PIPELINE_PROOF_ONLY"
        or response.get("response_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoPairedModelQueueError("model response seal is invalid")
    _validate_authority(response.get("authority"))
    reasons = response.get("reason_ids")
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(reasons) > 8
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise DojoPairedModelQueueError("model response reasons are invalid")
    return response


def submit_model_response(
    *, queue_dir: Path, response_value: Mapping[str, Any]
) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        status = _derived_status_unlocked(queue_dir)
        candidate_sha = response_value.get("decision_packet_sha256")
        if isinstance(candidate_sha, str):
            response_path = queue_dir / "responses" / f"{candidate_sha}.json"
            if response_path.exists():
                existing = _read_json(response_path)
                if existing != dict(response_value):
                    raise DojoPairedModelQueueError(
                        "duplicate response conflicts with accepted bytes"
                    )
                return _status_with_halt_unlocked(queue_dir)
        if _halt_sentinel_unlocked(queue_dir) is not None:
            raise DojoPairedModelQueueError(
                "quota halt is active; unaccepted response remains unsubmitted"
            )
        packet_sha = status["current_ready_packet_sha256"]
        if not packet_sha:
            raise DojoPairedModelQueueError("no matching model packet is ready")
        packet = verify_decision_packet(
            _read_json(queue_dir / "ready" / f"{packet_sha}.json")
        )
        response = verify_model_response(response_value, packet)
        response_path = queue_dir / "responses" / f"{packet_sha}.json"
        _same_or_write(response_path, response)
        plan = _verify_queue_plan(_read_json(queue_dir / "queue-plan.json"))
        _append_event(
            queue_dir,
            plan=plan,
            event_type="RESPONSE_ACCEPTED",
            payload={
                "cell_ordinal": packet["cell_ordinal"],
                "cell_id": packet["cell_id"],
                "decision_packet_sha256": packet_sha,
                "response_sha256": response["response_sha256"],
                "provider_model": response["provider_model"],
                "provider_execution_id": response["provider_execution_id"],
                "provider_execution_kind": response["provider_execution_kind"],
                "content_seal_verified": True,
                "cryptographic_signature_verified": False,
                "economic_application_status": ("NOT_YET_APPLIED_PIPELINE_PROOF_ONLY"),
            },
        )
        return _publish_ready_unlocked(queue_dir)
    finally:
        _unlock(descriptor)


def complete_current_decision(
    *,
    queue_dir: Path,
    response_path: Path,
    action: str,
    reason_ids: Sequence[str],
    provider_model: str,
    provider_execution_id: str,
) -> dict[str, Any]:
    """Seal/reuse, submit, and verify one response without changing its bytes."""

    packet = current_ready_packet(queue_dir)
    response_bytes_reused = response_path.exists()
    if response_bytes_reused:
        response = verify_model_response(_read_json(response_path), packet)
    else:
        response = seal_model_response(
            packet=packet,
            action=action,
            reason_ids=reason_ids,
            provider_model=provider_model,
            provider_execution_id=provider_execution_id,
        )
        _write_exclusive(response_path, response)
    submitted = submit_model_response(
        queue_dir=queue_dir,
        response_value=response,
    )
    verified = verify_queue(queue_dir)
    if submitted != verified:
        raise DojoPairedModelQueueError(
            "post-submit queue verification differs from submitted state"
        )
    return {
        "contract": "QR_DOJO_PAIRED_MODEL_CELL_COMPLETION_V1",
        "schema_version": SCHEMA_VERSION,
        "decision_packet_sha256": packet["decision_packet_sha256"],
        "action": response["action"],
        "response_sha256": response["response_sha256"],
        "provider_model": response["provider_model"],
        "provider_execution_id": response["provider_execution_id"],
        "resulting_status": verified,
        "response_bytes_reused": response_bytes_reused,
        "authority": dict(AUTHORITY),
    }


def accepted_response_bundle(
    queue_dir: Path, *, require_complete: bool = True
) -> list[dict[str, Any]]:
    """Return verified packet/response pairs in the immutable queue order."""

    descriptor, _ = _lock(queue_dir)
    try:
        plan = _verify_queue_plan(_read_json(queue_dir / "queue-plan.json"))
        status = _derived_status_unlocked(queue_dir)
        if require_complete and status["state"] != "COMPLETE":
            raise DojoPairedModelQueueError(
                "economic application requires the complete 84-cell queue"
            )
        rows: list[dict[str, Any]] = []
        for cell in plan["cells"][: status["accepted_model_decision_count"]]:
            packet = verify_decision_packet(
                _read_json(queue_dir / "packets" / cell["packet_filename"])
            )
            response_path = (
                queue_dir
                / "responses"
                / f"{packet['decision_packet_sha256']}.json"
            )
            response = verify_model_response(_read_json(response_path), packet)
            rows.append(
                {
                    "cell_ordinal": cell["cell_ordinal"],
                    "coordinate_id": cell["coordinate_id"],
                    "cadence_id": cell["cadence_id"],
                    "source_decision_id": packet["source_decision_id"],
                    "state_packet_sha256": packet["state_packet_sha256"],
                    "decision_packet_sha256": packet["decision_packet_sha256"],
                    "response_sha256": response["response_sha256"],
                    "action": response["action"],
                    "reason_ids": response["reason_ids"],
                    "provider_model": response["provider_model"],
                    "provider_execution_id": response["provider_execution_id"],
                }
            )
        return rows
    finally:
        _unlock(descriptor)


def verify_queue(queue_dir: Path) -> dict[str, Any]:
    descriptor, _ = _lock(queue_dir)
    try:
        plan = _verify_queue_plan(_read_json(queue_dir / "queue-plan.json"))
        for cell in plan["cells"]:
            packet = verify_decision_packet(
                _read_json(queue_dir / "packets" / cell["packet_filename"])
            )
            if packet["decision_packet_sha256"] != cell["decision_packet_sha256"]:
                raise DojoPairedModelQueueError("packet inventory changed")
        status = _status_with_halt_unlocked(queue_dir)
        if status["current_ready_packet_sha256"]:
            packet_sha = status["current_ready_packet_sha256"]
            verify_decision_packet(
                _read_json(queue_dir / "ready" / f"{packet_sha}.json")
            )
        for path in sorted((queue_dir / "responses").glob("*.json")):
            response = _read_json(path)
            packet_sha = response.get("decision_packet_sha256")
            packet_cell = next(
                cell
                for cell in plan["cells"]
                if cell["decision_packet_sha256"] == packet_sha
            )
            packet = _read_json(queue_dir / "packets" / packet_cell["packet_filename"])
            verify_model_response(response, packet)
        return status
    finally:
        _unlock(descriptor)


__all__ = [
    "PREFLIGHT_CONTRACT",
    "QUOTA_HALT_CONTRACT",
    "DECISION_PACKET_CONTRACT",
    "DojoPairedModelQueueError",
    "MODEL_RESPONSE_CONTRACT",
    "QUEUE_PLAN_CONTRACT",
    "accepted_response_bundle",
    "build_queue_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "complete_current_decision",
    "current_ready_packet",
    "emit_next_ready",
    "halt_for_quota",
    "initialize_queue",
    "preflight_model_decision",
    "queue_status",
    "resume_quota_halt",
    "seal_model_response",
    "submit_model_response",
    "verify_decision_packet",
    "verify_model_response",
    "verify_queue",
]
