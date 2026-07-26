"""Dormant end-to-end controller for a future paper-AI inventory room.

This module is intentionally not imported by any existing C/D/E/G room or
paper runner.  A future versioned ``paper-ai-inventory-*`` room may invoke one
cycle after it has independently created canonical point-in-time sources and
started the separate broker-owner service.

The controller owns no broker.  It revalidates the canonical PAPER_ELIGIBLE
proof, writes one trusted evidence packet, invokes one sealed external model
sidecar, durably records the resulting V2 decision, and asks the authenticated
runner client to apply that exact decision.  Every virtual mutation therefore
happens in the broker-owner process and only after a durable AI decision.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_evidence_packet import (
    verify_ai_inventory_evidence_packet,
    write_trusted_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    append_inventory_decision,
    validate_inventory_decision_ledger,
)
from quant_rabbit.dojo_ai_inventory_broker_service import (
    DojoAIInventoryRunnerClient,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    MAX_PRODUCER_RECEIPT_BYTES,
    PRODUCER_RECEIPT_DIRECTORY,
    load_sealed_command_model_adapter,
    produce_ai_inventory_proposal,
    verify_ai_inventory_producer_receipt,
)
from quant_rabbit.dojo_replay_lifecycle import (
    canonical_paper_ai_rooms_root,
    verify_paper_ai_inventory_launch_preflight,
)


CONTROLLER_CYCLE_CONTRACT = "QR_DOJO_AI_INVENTORY_CONTROLLER_CYCLE_V1"
CONTROLLER_CONFIG_CONTRACT = "QR_DOJO_AI_INVENTORY_CONTROLLER_CONFIG_V1"
DECISION_LEDGER_NAME = "ai_inventory_decisions.jsonl"
CYCLE_LEDGER_NAME = "controller_cycles.jsonl"
CONTROLLER_LOCK_NAME = ".controller-cycle.lock"
MAX_CYCLE_LEDGER_BYTES = 16 * 1024 * 1024
MAX_CYCLE_ROW_BYTES = 512 * 1024
MAX_DECISION_VALIDITY_SECONDS = 90
GENESIS_CYCLE_SHA256 = "0" * 64

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,254}$")
_STAGES = ("CYCLE_STARTED", "PROPOSAL_PRODUCED", "DECISION_SEALED", "CYCLE_APPLIED")


class AIInventoryControllerError(RuntimeError):
    """A controller cycle failed closed before a trustworthy completion."""


class AIInventoryControllerBusyError(AIInventoryControllerError):
    """Another process currently owns the one permitted cycle lock."""


class AIInventoryControllerIntegrityError(AIInventoryControllerError):
    """A source, lifecycle, checkpoint, or broker binding is inconsistent."""


class AIInventoryControllerMarketClosedError(AIInventoryControllerError):
    """New AI evaluation and mutation are disabled while FX is closed."""


@dataclass(frozen=True)
class AIInventoryControllerConfig:
    """Code-owned inputs for one dormant future paper-AI room."""

    repository_root: Path
    experiment_id: str
    room_id: str
    adapter_id: str
    model_id: str
    adapter_config_sha256: str
    producer_id: str


@dataclass(frozen=True)
class AIInventoryCycleResult:
    """One durably completed controller cycle."""

    cycle_record: dict[str, Any]
    decision: dict[str, Any]
    applied_receipt: dict[str, Any]
    admission_reference: dict[str, Any] | None


def run_ai_inventory_cycle(
    config: AIInventoryControllerConfig,
    trusted_evidence_request: Mapping[str, Any],
    runner: DojoAIInventoryRunnerClient,
) -> AIInventoryCycleResult:
    """Run or recover exactly one paper-AI inventory cycle.

    A cycle is idempotent by the canonical trusted-evidence request and
    controller identity.  Checkpoints are appended before the decision writer
    and after it, allowing a retry to resume without asking the model for a
    different answer after a decision has become durable.
    """

    validated = _validate_config(config)
    request = _snapshot_mapping(trusted_evidence_request, "trusted evidence request")
    cycle_identity = _cycle_identity(validated, request)
    room_root = _room_root(validated)
    cycle_ledger = room_root / CYCLE_LEDGER_NAME
    decision_ledger = room_root / DECISION_LEDGER_NAME

    _require_market_open(_utc_now(), "controller clock")
    with _exclusive_cycle_lock(room_root / CONTROLLER_LOCK_NAME):
        rows = _validate_cycle_ledger(cycle_ledger)
        completed = _cycle_rows(rows, cycle_identity, "CYCLE_APPLIED")
        if completed:
            if len(completed) != 1:
                raise AIInventoryControllerIntegrityError(
                    "cycle has duplicate completion receipts"
                )
            return _restore_completed_result(completed[0], decision_ledger)

        unfinished = _unfinished_cycle(rows)
        if unfinished is not None and unfinished != cycle_identity:
            raise AIInventoryControllerIntegrityError(
                "another cycle has an unfinished durable checkpoint"
            )

        lifecycle = _verify_active_lifecycle(validated)
        stages = _cycle_rows(rows, cycle_identity)
        if not stages:
            _append_cycle_row(
                cycle_ledger,
                cycle_identity=cycle_identity,
                stage="CYCLE_STARTED",
                payload={
                    "evidence_request_sha256": _sha256(request),
                    "launch_preflight_token_sha256": lifecycle[
                        "launch_preflight_token_sha256"
                    ],
                },
            )
            stages = _cycle_rows(_validate_cycle_ledger(cycle_ledger), cycle_identity)

        proposal_rows = [row for row in stages if row["stage"] == "PROPOSAL_PRODUCED"]
        if proposal_rows:
            if len(proposal_rows) != 1:
                raise AIInventoryControllerIntegrityError(
                    "cycle has duplicate proposal checkpoints"
                )
            decision_proposal = _snapshot_mapping(
                proposal_rows[0]["payload"].get("decision_proposal"),
                "checkpoint decision proposal",
            )
        else:
            packet_path = write_trusted_ai_inventory_evidence_packet(request)
            packet = verify_ai_inventory_evidence_packet(
                validated.repository_root, packet_path
            )
            _validate_packet_scope(packet, validated, request, lifecycle)
            initial_observation = _observe_broker(runner)
            _validate_packet_against_broker(
                packet, initial_observation, request
            )

            adapter = load_sealed_command_model_adapter(
                validated.adapter_id,
                validated.adapter_config_sha256,
                experiment_id=validated.experiment_id,
                room_id=validated.room_id,
            )
            with tempfile.TemporaryDirectory(
                prefix=".controller-producer-",
                dir=room_root,
            ) as producer_staging:
                produced = produce_ai_inventory_proposal(
                    packet,
                    adapter,
                    producer_id=validated.producer_id,
                    room_root=Path(producer_staging).resolve(strict=True),
                )
                _require_market_open(_utc_now(), "post-model controller clock")
                _require_active_lifecycle_unchanged(
                    validated,
                    lifecycle,
                    phase="during model evaluation",
                )
                _promote_producer_receipt(
                    staging_root=Path(producer_staging).resolve(strict=True),
                    room_root=room_root,
                    produced=produced,
                )
            final_observation = _observe_broker(runner)
            if final_observation != initial_observation:
                raise AIInventoryControllerIntegrityError(
                    "broker quote, ledger, or inventory advanced during "
                    "model evaluation"
                )
            decision_proposal = _build_decision_proposal(
                packet=packet,
                request=request,
                lifecycle=lifecycle,
                produced=produced,
                broker_observation=final_observation,
            )
            _append_cycle_row(
                cycle_ledger,
                cycle_identity=cycle_identity,
                stage="PROPOSAL_PRODUCED",
                payload={
                    "evidence_packet_sha256": packet["packet_sha256"],
                    "producer_receipt_sha256": produced["producer_receipt"][
                        "receipt_sha256"
                    ],
                    "decision_proposal": decision_proposal,
                },
            )

        decision_rows = [
            row
            for row in _cycle_rows(_validate_cycle_ledger(cycle_ledger), cycle_identity)
            if row["stage"] == "DECISION_SEALED"
        ]
        _require_active_lifecycle_unchanged(
            validated,
            lifecycle,
            phase="before durable decision append",
        )
        appended = append_inventory_decision(decision_ledger, decision_proposal)
        decision = dict(appended.record)
        if decision_rows:
            if (
                len(decision_rows) != 1
                or decision_rows[0]["payload"].get("decision_sha256")
                != decision["decision_sha256"]
            ):
                raise AIInventoryControllerIntegrityError(
                    "decision checkpoint conflicts with the decision ledger"
                )
        else:
            _append_cycle_row(
                cycle_ledger,
                cycle_identity=cycle_identity,
                stage="DECISION_SEALED",
                payload={
                    "decision_sha256": decision["decision_sha256"],
                    "decision_identity_sha256": decision["decision_identity_sha256"],
                },
            )

        status = runner.decision_status(decision["decision_sha256"])
        if status.get("status") == "APPLIED":
            applied = _snapshot_mapping(
                status.get("receipt"), "recovered applied receipt"
            )
        elif status.get("status") in {"NONE", "RESERVED"}:
            _require_market_open(_utc_now(), "pre-mutation controller clock")
            _require_active_lifecycle_unchanged(
                validated,
                lifecycle,
                phase="before virtual mutation",
            )
            runtime = _runtime_evidence(validated, decision)
            try:
                applied = runner.apply_ai_decision(decision, runtime)
            except Exception:
                recovered = runner.decision_status(decision["decision_sha256"])
                if recovered.get("status") != "APPLIED":
                    raise
                applied = _snapshot_mapping(
                    recovered.get("receipt"), "recovered applied receipt"
                )
        else:
            raise AIInventoryControllerIntegrityError(
                "broker returned an unknown decision status"
            )
        _validate_applied_receipt(decision, applied)
        admission_reference = _admission_reference(decision, applied)
        cycle_record = _append_cycle_row(
            cycle_ledger,
            cycle_identity=cycle_identity,
            stage="CYCLE_APPLIED",
            payload={
                "decision_sha256": decision["decision_sha256"],
                "action": decision["action"],
                "applied_receipt_sha256": applied["applied_receipt_sha256"],
                "broker_ledger_terminal_sha256": applied[
                    "broker_ledger_terminal_sha256"
                ],
                "admission_reference": admission_reference,
            },
        )
        return AIInventoryCycleResult(
            cycle_record=cycle_record,
            decision=decision,
            applied_receipt=dict(applied),
            admission_reference=admission_reference,
        )


def controller_config_from_mapping(
    repository_root: Path, value: Mapping[str, Any]
) -> AIInventoryControllerConfig:
    """Parse one strict CLI config without accepting paths or credentials."""

    row = _snapshot_mapping(value, "controller config")
    expected = {
        "contract",
        "experiment_id",
        "room_id",
        "adapter_id",
        "model_id",
        "adapter_config_sha256",
        "producer_id",
        "paper_only",
        "order_authority",
        "live_permission",
    }
    if set(row) != expected or row.get("contract") != CONTROLLER_CONFIG_CONTRACT:
        raise AIInventoryControllerIntegrityError("controller config schema is invalid")
    if (
        row.get("paper_only") is not True
        or row.get("order_authority") != "NONE"
        or row.get("live_permission") is not False
    ):
        raise AIInventoryControllerIntegrityError(
            "controller config safety guard is invalid"
        )
    return _validate_config(
        AIInventoryControllerConfig(
            repository_root=repository_root,
            experiment_id=str(row["experiment_id"]),
            room_id=str(row["room_id"]),
            adapter_id=str(row["adapter_id"]),
            model_id=str(row["model_id"]),
            adapter_config_sha256=str(row["adapter_config_sha256"]),
            producer_id=str(row["producer_id"]),
        )
    )


def _build_decision_proposal(
    *,
    packet: Mapping[str, Any],
    request: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    produced: Mapping[str, Any],
    broker_observation: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = _snapshot_mapping(packet.get("bindings"), "packet bindings")
    position = _snapshot_mapping(packet.get("position"), "packet position")
    quote = _snapshot_mapping(packet.get("quote"), "packet quote")
    ai_binding = _snapshot_mapping(
        produced.get("ai_decision_binding"), "AI decision binding"
    )
    cutoff = _parse_utc(ai_binding.get("produced_at_utc"), "AI produced_at")
    expires = _canonical_utc(cutoff + timedelta(seconds=MAX_DECISION_VALIDITY_SECONDS))
    action = produced.get("action")
    admission: dict[str, Any] | None = None
    if action == "ALLOW_NEW_VIRTUAL":
        signal = packet.get("entry_signal")
        if not isinstance(signal, Mapping):
            raise AIInventoryControllerIntegrityError(
                "ALLOW_NEW_VIRTUAL lacks an authenticated entry signal"
            )
        admission = {
            "entry_signal": _snapshot_mapping(signal, "entry signal"),
            "evidence_packet_sha256": packet["packet_sha256"],
            "permit_expires_at_utc": expires,
        }
    source_files = _snapshot_mapping(request.get("source_files"), "source files")
    position_source_sha = _filename_sha(source_files.get("position"), "position")
    dynamic_age = packet.get("dynamic_binding_max_age_seconds")
    if isinstance(dynamic_age, bool) or not isinstance(dynamic_age, int):
        raise AIInventoryControllerIntegrityError("packet dynamic age is invalid")
    max_age = min(dynamic_age, 300)
    observed = packet["cutoff_utc"]
    return {
        "contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "cutoff_at_utc": _canonical_utc(cutoff),
        "expires_at_utc": expires,
        "action": action,
        "virtual_units": produced.get("virtual_units"),
        "confidence": float(produced["confidence"]),
        "admission_binding": admission,
        "reason_code": produced.get("reason_code"),
        "reason": produced.get("reason"),
        "session_binding": {
            "experiment_id": bindings["experiment_id"],
            "room_id": bindings["room_id"],
            "session_contract_sha256": bindings["session_contract_sha256"],
            "observed_at_utc": observed,
        },
        "policy_binding": {
            "policy_id": bindings["policy_id"],
            "policy_sha256": bindings["policy_sha256"],
            "observed_at_utc": observed,
        },
        "candidate_binding": {
            "candidate_id": bindings["candidate_id"],
            "candidate_sha256": bindings["candidate_sha256"],
            "observed_at_utc": observed,
        },
        "spec_binding": {
            "spec_id": bindings["spec_id"],
            "spec_sha256": bindings["spec_sha256"],
            "observed_at_utc": observed,
        },
        "lifecycle_binding": {
            "paper_eligible_event_sha256": lifecycle["paper_eligible_event_sha256"],
            "candidate_lifecycle_ledger_tip_sha256": lifecycle[
                "candidate_lifecycle_ledger_tip_sha256"
            ],
            "observed_at_utc": observed,
        },
        "ai_decision_binding": ai_binding,
        "ledger_binding": {
            "sha256": broker_observation["health"]["ledger_sha256"],
            "observed_at_utc": bindings["ledger_observed_at_utc"],
        },
        "state_binding": {
            "sha256": bindings["state_sha256"],
            "observed_at_utc": bindings["state_observed_at_utc"],
        },
        "snapshot_binding": {
            "sha256": bindings["snapshot_sha256"],
            "observed_at_utc": bindings["snapshot_observed_at_utc"],
        },
        "position_binding": {
            "position_id": position["position_id"],
            "pair": position["pair"],
            "side": position["side"],
            "units": float(position["units"]),
            "strategy_tag": position["strategy_tag"],
            "entry_context_sha256": position["entry_context_sha256"],
            "sha256": position_source_sha,
            "observed_at_utc": position["observed_at_utc"],
        },
        "quote_binding": {
            "pair": quote["pair"],
            "bid": float(quote["bid"]),
            "ask": float(quote["ask"]),
            "sha256": quote["source_sha256"],
            "observed_at_utc": quote["timestamp_utc"],
        },
        "source_watermarks": _source_watermarks(packet),
        "max_dynamic_evidence_age_seconds": max_age,
        "max_record_lag_seconds": max_age,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }


def _source_watermarks(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    quote = _snapshot_mapping(packet.get("quote"), "packet quote")
    rows: list[dict[str, Any]] = [
        {
            "source_id": f"quote:{quote['pair']}",
            "sha256": quote["source_sha256"],
            "watermark_at_utc": quote["timestamp_utc"],
            "max_age_seconds": quote["max_age_seconds"],
        }
    ]
    for candle in packet.get("candles", []):
        item = _snapshot_mapping(candle, "packet candle")
        rows.append(
            {
                "source_id": (
                    f"candle:{item['pair']}:{item['granularity']}:"
                    f"{item['completed_at_utc']}"
                ),
                "sha256": item["source_sha256"],
                "watermark_at_utc": item["completed_at_utc"],
                "max_age_seconds": item["max_age_seconds"],
            }
        )
    for packet_key, prefix in (
        ("news_items", "news"),
        ("calendar_items", "calendar"),
        ("cross_asset_items", "cross-asset"),
    ):
        for source in packet.get(packet_key, []):
            item = _snapshot_mapping(source, f"packet {packet_key} source")
            rows.append(
                {
                    "source_id": f"{prefix}:{item['source_id']}",
                    "sha256": item["content_sha256"],
                    "watermark_at_utc": item["observed_at_utc"],
                    "max_age_seconds": item["max_age_seconds"],
                }
            )
    rows.sort(key=lambda row: row["source_id"])
    if len({row["source_id"] for row in rows}) != len(rows):
        raise AIInventoryControllerIntegrityError(
            "packet source watermarks have duplicate identities"
        )
    return rows


def _observe_broker(runner: DojoAIInventoryRunnerClient) -> dict[str, Any]:
    health = runner.health()
    positions = runner.positions
    quotes = runner.quotes
    quote_provenance = runner.quote_provenance
    if (
        health.get("status") != "READY"
        or health.get("paper_only") is not True
        or health.get("order_authority") != "NONE"
        or health.get("live_permission") is not False
        or not _is_sha(health.get("ledger_sha256"))
    ):
        raise AIInventoryControllerIntegrityError(
            "broker owner health safety binding is invalid"
        )
    return {
        "health": _snapshot_mapping(health, "broker health"),
        "positions": _snapshot_mapping(positions, "broker positions"),
        "quotes": _snapshot_mapping(quotes, "broker quotes"),
        "quote_provenance": _snapshot_mapping(
            quote_provenance, "broker quote provenance"
        ),
    }


def _validate_packet_against_broker(
    packet: Mapping[str, Any],
    observation: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    position = _snapshot_mapping(packet.get("position"), "packet position")
    quote = _snapshot_mapping(packet.get("quote"), "packet quote")
    positions = _snapshot_mapping(observation.get("positions"), "broker positions")
    if position["side"] == "FLAT":
        if positions:
            raise AIInventoryControllerIntegrityError(
                "flat evidence conflicts with open broker inventory"
            )
    else:
        if set(positions) != {position["position_id"]}:
            raise AIInventoryControllerIntegrityError(
                "evidence does not identify the one broker position"
            )
        actual = _snapshot_mapping(
            positions[position["position_id"]], "broker position"
        )
        exact = (
            ("trade_id", position["position_id"]),
            ("pair", position["pair"]),
            ("side", position["side"]),
            ("units", float(position["units"])),
            ("strategy_tag", position["strategy_tag"]),
            ("entry_context_sha256", position["entry_context_sha256"]),
        )
        for field, expected in exact:
            if actual.get(field) != expected:
                raise AIInventoryControllerIntegrityError(
                    f"evidence/broker position mismatch: {field}"
                )
    quotes = _snapshot_mapping(observation.get("quotes"), "broker quotes")
    current = quotes.get(quote["pair"])
    expected_quote = [
        float(quote["bid"]),
        float(quote["ask"]),
        quote["timestamp_utc"],
    ]
    if current != expected_quote:
        raise AIInventoryControllerIntegrityError(
            "evidence quote is not the broker-owner executable quote"
        )
    source_files = _snapshot_mapping(
        request.get("source_files"), "request source files"
    )
    source_receipts = _snapshot_mapping(
        request.get("source_receipts"), "request source receipts"
    )
    quote_source_sha256 = _filename_sha(
        source_files.get("quote"), "quote source"
    )
    quote_receipt_sha256 = source_receipts.get("quote")
    if (
        quote_source_sha256 != quote["source_sha256"]
        or not _is_sha(quote_receipt_sha256)
    ):
        raise AIInventoryControllerIntegrityError(
            "evidence quote source/receipt binding is invalid"
        )
    provenance_rows = _snapshot_mapping(
        observation.get("quote_provenance"), "broker quote provenance"
    )
    provenance = _snapshot_mapping(
        provenance_rows.get(quote["pair"]), "broker quote provenance row"
    )
    exact_provenance = {
        "pair": quote["pair"],
        "bid": float(quote["bid"]),
        "ask": float(quote["ask"]),
        "timestamp_utc": quote["timestamp_utc"],
        "capture_source_sha256": quote["source_sha256"],
        "acquisition_receipt_sha256": quote_receipt_sha256,
        "test_only_raw_quote": False,
    }
    for key, expected in exact_provenance.items():
        if provenance.get(key) != expected:
            raise AIInventoryControllerIntegrityError(
                f"evidence/broker quote provenance mismatch: {key}"
            )
    if not _is_sha(provenance.get("quote_watermark_sha256")):
        raise AIInventoryControllerIntegrityError(
            "broker quote provenance lacks a durable watermark"
        )


def _runtime_evidence(
    config: AIInventoryControllerConfig, decision: Mapping[str, Any]
) -> dict[str, Any]:
    session = decision["session_binding"]
    candidate = decision["candidate_binding"]
    policy = decision["policy_binding"]
    spec = decision["spec_binding"]
    lifecycle = decision["lifecycle_binding"]
    return {
        "room_kind": "paper-ai-inventory",
        "dedicated_root": str(
            canonical_paper_ai_rooms_root(config.repository_root).resolve(strict=True)
        ),
        "room_id": session["room_id"],
        "experiment_id": session["experiment_id"],
        "session_contract_sha256": session["session_contract_sha256"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
        "spec_id": spec["spec_id"],
        "spec_sha256": spec["spec_sha256"],
        "paper_eligible_event_sha256": lifecycle["paper_eligible_event_sha256"],
        "candidate_lifecycle_ledger_tip_sha256": lifecycle[
            "candidate_lifecycle_ledger_tip_sha256"
        ],
        "ai_decision_binding": decision["ai_decision_binding"],
        "admission_binding": decision["admission_binding"],
        "ledger_sha256": decision["ledger_binding"]["sha256"],
        "state_sha256": decision["state_binding"]["sha256"],
        "snapshot_sha256": decision["snapshot_binding"]["sha256"],
        "position": {
            key: decision["position_binding"][key]
            for key in (
                "position_id",
                "pair",
                "side",
                "units",
                "strategy_tag",
                "entry_context_sha256",
                "sha256",
            )
        },
        "quote": {
            key: decision["quote_binding"][key]
            for key in ("pair", "bid", "ask", "sha256", "observed_at_utc")
        },
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }


def _admission_reference(
    decision: Mapping[str, Any], applied: Mapping[str, Any]
) -> dict[str, Any] | None:
    if decision["action"] != "ALLOW_NEW_VIRTUAL":
        return None
    signal = decision["admission_binding"]["entry_signal"]
    return {
        "contract": "QR_DOJO_AI_ENTRY_ADMISSION_REFERENCE_V1",
        "applied_receipt_sha256": applied["applied_receipt_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "room_id": decision["session_binding"]["room_id"],
        "candidate_id": decision["candidate_binding"]["candidate_id"],
        "signal_identity_sha256": signal["signal_identity_sha256"],
    }


def _validate_applied_receipt(
    decision: Mapping[str, Any], applied: Mapping[str, Any]
) -> None:
    exact = {
        "decision_sha256": decision["decision_sha256"],
        "action": decision["action"],
        "room_id": decision["session_binding"]["room_id"],
        "candidate_id": decision["candidate_binding"]["candidate_id"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "status": "APPLIED",
    }
    for key, expected in exact.items():
        if applied.get(key) != expected:
            raise AIInventoryControllerIntegrityError(
                f"broker applied receipt mismatch: {key}"
            )
    for key in ("applied_receipt_sha256", "broker_ledger_terminal_sha256"):
        if not _is_sha(applied.get(key)):
            raise AIInventoryControllerIntegrityError(
                f"broker applied receipt has invalid {key}"
            )


def _validate_packet_scope(
    packet: Mapping[str, Any],
    config: AIInventoryControllerConfig,
    request: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> None:
    bindings = _snapshot_mapping(packet.get("bindings"), "packet bindings")
    exact = (
        ("experiment_id", config.experiment_id),
        ("room_id", config.room_id),
        ("candidate_id", lifecycle["candidate_id"]),
        ("spec_sha256", lifecycle["spec_sha256"]),
        ("policy_sha256", lifecycle["policy_sha256"]),
        (
            "paper_eligible_tip_sha256",
            lifecycle["candidate_lifecycle_ledger_tip_sha256"],
        ),
        (
            "launch_preflight_token_sha256",
            lifecycle["launch_preflight_token_sha256"],
        ),
    )
    for field, expected in exact:
        if bindings.get(field) != expected:
            raise AIInventoryControllerIntegrityError(
                f"evidence/lifecycle mismatch: {field}"
            )
    if (
        packet.get("cutoff_utc") != request.get("cutoff_utc")
        or packet.get("paper_only") is not True
        or packet.get("order_authority") != "NONE"
        or packet.get("live_permission") is not False
    ):
        raise AIInventoryControllerIntegrityError(
            "evidence packet request or safety binding is invalid"
        )


def _verify_active_lifecycle(
    config: AIInventoryControllerConfig,
) -> dict[str, Any]:
    try:
        token = verify_paper_ai_inventory_launch_preflight(
            config.repository_root,
            experiment_id=config.experiment_id,
            room_id=config.room_id,
        )
    except Exception as exc:
        raise AIInventoryControllerIntegrityError(
            "canonical PAPER_ELIGIBLE launch proof is invalid"
        ) from exc
    token = _snapshot_mapping(token, "launch preflight")
    if (
        token.get("paper_only") is not True
        or token.get("order_authority") != "NONE"
        or token.get("live_permission") is not False
        or token.get("experiment_id") != config.experiment_id
        or token.get("room_id") != config.room_id
        or not _is_sha(token.get("launch_preflight_token_sha256"))
    ):
        raise AIInventoryControllerIntegrityError(
            "launch preflight safety binding is invalid"
        )
    lifecycle_identity = {
        "adapter_id": config.adapter_id,
        "model_id": config.model_id,
        "config_sha256": config.adapter_config_sha256,
        "producer_id": config.producer_id,
    }
    for key, expected in lifecycle_identity.items():
        if token.get(key) != expected:
            raise AIInventoryControllerIntegrityError(
                f"launch preflight {key} binding mismatch"
            )
    window = _snapshot_mapping(token.get("future_window"), "future window")
    now = _utc_now()
    start = _parse_utc(window.get("start_utc"), "future window start")
    end = _parse_utc(window.get("end_utc"), "future window end")
    if not start <= now < end:
        raise AIInventoryControllerIntegrityError(
            "controller clock is outside the immutable future window"
        )
    return token


def _require_active_lifecycle_unchanged(
    config: AIInventoryControllerConfig,
    expected: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    current = _verify_active_lifecycle(config)
    if current != expected:
        raise AIInventoryControllerIntegrityError(
            f"canonical launch proof changed {phase}"
        )
    window = _snapshot_mapping(expected.get("future_window"), "future window")
    end = _parse_utc(window.get("end_utc"), "future window end")
    if _utc_now() >= end:
        raise AIInventoryControllerIntegrityError(
            f"controller clock reached the immutable future-window end {phase}"
        )


def _promote_producer_receipt(
    *,
    staging_root: Path,
    room_root: Path,
    produced: Mapping[str, Any],
) -> None:
    receipt = _snapshot_mapping(
        produced.get("producer_receipt"), "producer receipt"
    )
    receipt_sha = receipt.get("receipt_sha256")
    if not _is_sha(receipt_sha):
        raise AIInventoryControllerIntegrityError(
            "producer receipt digest is invalid"
        )
    source = (
        staging_root
        / PRODUCER_RECEIPT_DIRECTORY
        / f"{receipt_sha}.json"
    )
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "staged producer receipt is unavailable"
        ) from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise AIInventoryControllerIntegrityError(
            "staged producer receipt is not a regular file"
        )
    if (
        source_stat.st_size <= 0
        or source_stat.st_size > MAX_PRODUCER_RECEIPT_BYTES
    ):
        raise AIInventoryControllerIntegrityError(
            "staged producer receipt has invalid size"
        )
    try:
        verify_ai_inventory_producer_receipt(staging_root, source)
    except Exception as exc:
        raise AIInventoryControllerIntegrityError(
            "staged producer receipt failed verification"
        ) from exc

    destination_root = room_root / PRODUCER_RECEIPT_DIRECTORY
    try:
        destination_root.mkdir(mode=0o700, exist_ok=True)
        destination_stat = destination_root.lstat()
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "canonical producer receipt directory is unavailable"
        ) from exc
    if stat.S_ISLNK(destination_stat.st_mode) or not stat.S_ISDIR(
        destination_stat.st_mode
    ):
        raise AIInventoryControllerIntegrityError(
            "canonical producer receipt directory is unsafe"
        )
    destination = destination_root / source.name
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        if _read_regular_nofollow(source) != _read_regular_nofollow(destination):
            raise AIInventoryControllerIntegrityError(
                "canonical producer receipt has conflicting bytes"
            )
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "producer receipt promotion failed"
        ) from exc
    directory = os.open(
        destination_root,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    try:
        verify_ai_inventory_producer_receipt(room_root, destination)
    except Exception as exc:
        raise AIInventoryControllerIntegrityError(
            "promoted producer receipt failed verification"
        ) from exc


def _read_regular_nofollow(path: Path) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "producer receipt cannot be opened safely"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or info.st_size > MAX_PRODUCER_RECEIPT_BYTES
        ):
            raise AIInventoryControllerIntegrityError(
                "producer receipt is not a bounded regular file"
            )
        raw = os.read(descriptor, MAX_PRODUCER_RECEIPT_BYTES + 1)
        if len(raw) != info.st_size:
            raise AIInventoryControllerIntegrityError(
                "producer receipt changed during read"
            )
        return raw
    finally:
        os.close(descriptor)


def _validate_config(
    config: AIInventoryControllerConfig,
) -> AIInventoryControllerConfig:
    if not isinstance(config, AIInventoryControllerConfig):
        raise TypeError("config must be AIInventoryControllerConfig")
    root = _trusted_repository_root()
    try:
        supplied = config.repository_root.resolve(strict=True)
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "repository root is unavailable"
        ) from exc
    if supplied != root:
        raise AIInventoryControllerIntegrityError(
            "repository root is not the package-derived worktree"
        )
    for label, value in (
        ("experiment_id", config.experiment_id),
        ("room_id", config.room_id),
    ):
        if (
            not isinstance(value, str)
            or not value.startswith("paper-ai-inventory-")
            or Path(value).name != value
            or not _ID_RE.fullmatch(value)
        ):
            raise AIInventoryControllerIntegrityError(
                f"{label} is not a dedicated paper-AI identifier"
            )
    for label, value in (
        ("adapter_id", config.adapter_id),
        ("model_id", config.model_id),
        ("producer_id", config.producer_id),
    ):
        if not isinstance(value, str) or not _ID_RE.fullmatch(value):
            raise AIInventoryControllerIntegrityError(f"{label} is invalid")
    if not _is_sha(config.adapter_config_sha256):
        raise AIInventoryControllerIntegrityError("adapter_config_sha256 is invalid")
    _room_root(config)
    return config


def _room_root(config: AIInventoryControllerConfig) -> Path:
    root = canonical_paper_ai_rooms_root(config.repository_root)
    candidate = root / config.experiment_id / config.room_id
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise AIInventoryControllerIntegrityError(
            "canonical dedicated room root is unavailable"
        ) from exc
    if not resolved.is_dir():
        raise AIInventoryControllerIntegrityError(
            "canonical dedicated room root is not a directory"
        )
    return resolved


def _trusted_repository_root() -> Path:
    try:
        root = Path(__file__).resolve(strict=True).parents[2]
        return root.resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise AIInventoryControllerIntegrityError(
            "package-derived repository root is unavailable"
        ) from exc


def _cycle_identity(
    config: AIInventoryControllerConfig, request: Mapping[str, Any]
) -> str:
    return _sha256(
        {
            "contract": CONTROLLER_CYCLE_CONTRACT,
            "experiment_id": config.experiment_id,
            "room_id": config.room_id,
            "adapter_id": config.adapter_id,
            "model_id": config.model_id,
            "adapter_config_sha256": config.adapter_config_sha256,
            "producer_id": config.producer_id,
            "trusted_evidence_request": request,
        }
    )


@contextmanager
def _exclusive_cycle_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise AIInventoryControllerIntegrityError(
            "controller lock cannot be opened"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AIInventoryControllerBusyError(
                "another AI inventory controller cycle is active"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _append_cycle_row(
    path: Path,
    *,
    cycle_identity: str,
    stage: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in _STAGES or not _is_sha(cycle_identity):
        raise AIInventoryControllerIntegrityError("cycle append identity is invalid")
    sealed_payload = _snapshot_mapping(payload, "cycle payload")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            rows = _decode_cycle_rows(handle.read())
            existing = [
                row
                for row in rows
                if row["cycle_identity_sha256"] == cycle_identity
                and row["stage"] == stage
            ]
            if existing:
                if len(existing) != 1 or existing[0]["payload"] != sealed_payload:
                    raise AIInventoryControllerIntegrityError(
                        "cycle checkpoint conflicts with existing row"
                    )
                return dict(existing[0])
            now = _canonical_utc(_utc_now())
            body = {
                "contract": CONTROLLER_CYCLE_CONTRACT,
                "sequence": len(rows) + 1,
                "previous_cycle_sha256": (
                    rows[-1]["cycle_sha256"] if rows else GENESIS_CYCLE_SHA256
                ),
                "recorded_at_utc": now,
                "cycle_identity_sha256": cycle_identity,
                "stage": stage,
                "payload": sealed_payload,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
            row = {**body, "cycle_sha256": _sha256(body)}
            raw = _canonical_json(row) + b"\n"
            if len(raw) > MAX_CYCLE_ROW_BYTES:
                raise AIInventoryControllerIntegrityError(
                    "cycle checkpoint exceeds the byte limit"
                )
            handle.seek(0, os.SEEK_END)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            return row
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_cycle_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            return _decode_cycle_rows(handle.read())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _decode_cycle_rows(raw: bytes) -> list[dict[str, Any]]:
    if len(raw) > MAX_CYCLE_LEDGER_BYTES:
        raise AIInventoryControllerIntegrityError(
            "controller cycle ledger exceeds the byte limit"
        )
    if raw and not raw.endswith(b"\n"):
        raise AIInventoryControllerIntegrityError(
            "controller cycle ledger has a partial row"
        )
    rows: list[dict[str, Any]] = []
    previous = GENESIS_CYCLE_SHA256
    expected_keys = {
        "contract",
        "sequence",
        "previous_cycle_sha256",
        "recorded_at_utc",
        "cycle_identity_sha256",
        "stage",
        "payload",
        "paper_only",
        "order_authority",
        "live_permission",
        "cycle_sha256",
    }
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AIInventoryControllerIntegrityError(
                f"invalid controller cycle row {index}"
            ) from exc
        if (
            not isinstance(row, dict)
            or set(row) != expected_keys
            or row.get("contract") != CONTROLLER_CYCLE_CONTRACT
            or row.get("sequence") != index
            or row.get("previous_cycle_sha256") != previous
            or row.get("stage") not in _STAGES
            or not isinstance(row.get("payload"), dict)
            or row.get("paper_only") is not True
            or row.get("order_authority") != "NONE"
            or row.get("live_permission") is not False
            or not _is_sha(row.get("cycle_identity_sha256"))
        ):
            raise AIInventoryControllerIntegrityError(
                f"invalid controller cycle schema at row {index}"
            )
        body = {key: value for key, value in row.items() if key != "cycle_sha256"}
        if row.get("cycle_sha256") != _sha256(body):
            raise AIInventoryControllerIntegrityError(
                f"invalid controller cycle digest at row {index}"
            )
        _parse_utc(row.get("recorded_at_utc"), "cycle recorded_at")
        previous = row["cycle_sha256"]
        rows.append(row)
    _validate_cycle_stage_order(rows)
    return rows


def _validate_cycle_stage_order(rows: list[dict[str, Any]]) -> None:
    by_cycle: dict[str, list[str]] = {}
    for row in rows:
        by_cycle.setdefault(row["cycle_identity_sha256"], []).append(row["stage"])
    for stages in by_cycle.values():
        positions = [_STAGES.index(stage) for stage in stages]
        if (
            positions != sorted(positions)
            or len(positions) != len(set(positions))
            or not positions
            or positions[0] != 0
        ):
            raise AIInventoryControllerIntegrityError(
                "controller cycle stage order is invalid"
            )


def _cycle_rows(
    rows: list[dict[str, Any]],
    cycle_identity: str,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["cycle_identity_sha256"] == cycle_identity
        and (stage is None or row["stage"] == stage)
    ]


def _unfinished_cycle(rows: list[dict[str, Any]]) -> str | None:
    current: dict[str, str] = {}
    for row in rows:
        current[row["cycle_identity_sha256"]] = row["stage"]
    unfinished = [
        identity for identity, stage in current.items() if stage != "CYCLE_APPLIED"
    ]
    if len(unfinished) > 1:
        raise AIInventoryControllerIntegrityError(
            "multiple unfinished controller cycles exist"
        )
    return unfinished[0] if unfinished else None


def _restore_completed_result(
    cycle_record: Mapping[str, Any], decision_ledger: Path
) -> AIInventoryCycleResult:
    payload = _snapshot_mapping(cycle_record.get("payload"), "cycle receipt payload")
    decision = _decision_by_sha(decision_ledger, payload.get("decision_sha256"))
    applied = {
        "decision_sha256": decision["decision_sha256"],
        "action": decision["action"],
        "room_id": decision["session_binding"]["room_id"],
        "candidate_id": decision["candidate_binding"]["candidate_id"],
        "applied_receipt_sha256": payload["applied_receipt_sha256"],
        "broker_ledger_terminal_sha256": payload["broker_ledger_terminal_sha256"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "status": "APPLIED",
    }
    admission = payload.get("admission_reference")
    return AIInventoryCycleResult(
        cycle_record=dict(cycle_record),
        decision=decision,
        applied_receipt=applied,
        admission_reference=(
            _snapshot_mapping(admission, "admission reference")
            if isinstance(admission, Mapping)
            else None
        ),
    )


def _decision_by_sha(path: Path, decision_sha256: object) -> dict[str, Any]:
    if not _is_sha(decision_sha256):
        raise AIInventoryControllerIntegrityError("decision digest is invalid")
    validation = validate_inventory_decision_ledger(path)
    if not validation.get("valid"):
        raise AIInventoryControllerIntegrityError(
            "decision ledger failed full validation"
        )
    try:
        rows = [
            json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            for line in path.read_bytes().splitlines()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryControllerIntegrityError(
            "decision ledger rows are unavailable"
        ) from exc
    matches = [row for row in rows if row.get("decision_sha256") == decision_sha256]
    if len(matches) != 1:
        raise AIInventoryControllerIntegrityError(
            "cycle decision is absent or duplicated"
        )
    return dict(matches[0])


def _filename_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.endswith(".json"):
        raise AIInventoryControllerIntegrityError(f"{label} source filename is invalid")
    digest = Path(value).stem
    if not _is_sha(digest) or Path(value).name != value:
        raise AIInventoryControllerIntegrityError(f"{label} source filename is invalid")
    return digest


def _snapshot_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AIInventoryControllerIntegrityError(f"{label} must be a mapping")
    try:
        result = json.loads(
            _canonical_json(value),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AIInventoryControllerIntegrityError(
            f"{label} is not canonical JSON"
        ) from exc
    if not isinstance(result, dict):
        raise AIInventoryControllerIntegrityError(f"{label} must be an object")
    return result


def _require_market_open(value: datetime, label: str) -> None:
    try:
        open_now = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AIInventoryControllerIntegrityError(
            f"{label} market status is unavailable"
        ) from exc
    if not open_now:
        raise AIInventoryControllerMarketClosedError(
            f"{label}: AI evaluation and virtual mutation are disabled while FX is closed"
        )


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise AIInventoryControllerIntegrityError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIInventoryControllerIntegrityError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AIInventoryControllerIntegrityError(f"{label} is naive")
    return parsed.astimezone(timezone.utc)


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as exc:
        raise AIInventoryControllerIntegrityError(
            "value is not canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
