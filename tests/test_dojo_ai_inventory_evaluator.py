from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit import dojo_ai_inventory_evaluator as evaluator
from quant_rabbit import dojo_ai_inventory_producer as producer_module
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    append_inventory_decision,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    DOJO_AI_PRODUCER_RECEIPT_CONTRACT,
    _seal_producer_receipt,
    _write_ai_inventory_producer_receipt,
)
from quant_rabbit.dojo_ai_evidence_packet import entry_signal_identity_sha256


OPEN_SCORE_TIME = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
CLOSED_SCORE_TIME = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _assessment_json(regime: str = "TREND") -> str:
    return json.dumps(
        {
            "contract": evaluator.SIGNED_ASSESSMENT_CONTRACT,
            "declared_regime": regime,
            "assessment": "Prospective paper-only inventory assessment.",
            "primary_path": "Price follows the declared regime.",
            "alternative_path": "Price remains range-bound.",
            "falsifier": "The completed horizon rejects the declared regime.",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _outcome_packet() -> dict[str, object]:
    return {
        "contract": evaluator.DOJO_AI_INVENTORY_OUTCOME_CONTRACT,
        "decision_contract": evaluator.REQUIRED_DECISION_CONTRACT,
        "producer_receipt_contract": (evaluator.REQUIRED_PRODUCER_RECEIPT_CONTRACT),
        "applied_receipt_event": evaluator.REQUIRED_APPLIED_RECEIPT_EVENT,
        "decision_sha256": SHA_A,
        "producer_receipt_sha256": SHA_B,
        "applied_receipt_sha256": SHA_C,
        "position_identity": {
            "position_id": "T000001",
            "pair": "USD_JPY",
            "side": "LONG",
            "strategy_tag": "paper_ai_inventory_v1",
            "entry_context_sha256": SHA_D,
        },
        "signal_identity": {
            "signal_identity_sha256": SHA_E,
            "pair": "USD_JPY",
            "side": "LONG",
            "strategy_tag": "paper_ai_inventory_v1",
            "entry_context_sha256": SHA_D,
        },
        "decision_cutoff_at_utc": "2026-07-20T10:00:00Z",
        "horizon_end_at_utc": "2026-07-20T11:00:00Z",
        "outcome_observed_at_utc": "2026-07-20T11:30:00Z",
        "outcome_kind": "SETTLEMENT",
        "realized_outcome": "WIN",
        "settlement_reason": "TAKE_PROFIT",
        "realized_pl_jpy": 20.0,
        "mfe_jpy": 35.0,
        "mae_jpy": -8.0,
        "review_time_executable_exit_pl_jpy": -10.0,
        "actual_exit_pl_jpy": 20.0,
        "counterfactual_delta_jpy": -30.0,
        "declared_assessment": json.loads(_assessment_json()),
        "declared_assessment_sha256": hashlib.sha256(
            _assessment_json().encode()
        ).hexdigest(),
        "declared_regime": "TREND",
        "realized_regime": "TREND",
        "regime_correct": True,
        "regime_confidence": 0.8,
        "regime_brier_score": 0.04,
        "source_watermarks": [
            {
                "source_id": "broker:ledger",
                "sha256": SHA_A,
                "watermark_at_utc": "2026-07-20T11:30:00Z",
            },
            {
                "source_id": "market:m1",
                "sha256": SHA_B,
                "watermark_at_utc": "2026-07-20T11:00:00Z",
            },
        ],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
        "evaluation_is_not_action": True,
    }


def _patch_clock(monkeypatch: pytest.MonkeyPatch, value: datetime) -> None:
    monkeypatch.setattr(evaluator, "_utc_now", lambda: value)


def _append(path: Path, packet: dict[str, object]) -> evaluator.EvaluationAppendResult:
    """Exercise ledger mechanics with an internal test-only trusted token."""

    sealed = evaluator._seal_outcome_packet(packet)
    token = evaluator._TrustedEvidenceToken(
        packet_sha256=evaluator._sha256(
            evaluator._canonical_json(sealed).encode("utf-8")
        ),
        room_root=str(path.parent),
        decision_sha256=str(sealed["decision_sha256"]),
        applied_receipt_sha256=str(sealed["applied_receipt_sha256"]),
        producer_receipt_sha256=str(sealed["producer_receipt_sha256"]),
    )
    evaluator._ACTIVE_TRUSTED_TOKENS.add(id(token))
    return evaluator.append_ai_inventory_evaluation(
        path, packet, _trusted_evidence_token=token
    )


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _append_broker_row(
    path: Path,
    rows: list[dict[str, object]],
    *,
    ts_utc: str,
    event: str,
    payload: dict[str, object],
) -> dict[str, object]:
    body = {
        "ts_utc": ts_utc,
        "event": event,
        "payload": payload,
        "prev_sha": rows[-1]["sha"] if rows else "0" * 64,
    }
    row = {**body, "sha": _sha(body)}
    rows.append(row)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows
        )
    )
    return row


def _append_quote_row(
    path: Path,
    rows: list[dict[str, object]],
    *,
    timestamp_utc: str,
    bid: float,
    ask: float,
    source_sha256: str,
    capture_source_sha256: str = "b" * 64,
    acquisition_receipt_sha256: str = "a" * 64,
    slippage_pips_per_fill: float = 0.0,
    financing_pips_per_day: float = 0.0,
) -> dict[str, object]:
    del source_sha256
    source = {
        "contract": "QR_DOJO_AI_INVENTORY_QUOTE_SOURCE_V1",
        "timestamp_utc": timestamp_utc,
        "pair": "USD_JPY",
        "bid": bid,
        "ask": ask,
        "capture_source_sha256": capture_source_sha256,
        "acquisition_receipt_sha256": acquisition_receipt_sha256,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    source_raw = (
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    source_digest = hashlib.sha256(source_raw).hexdigest()
    source_directory = path.parent / "quote_sources"
    source_directory.mkdir(exist_ok=True)
    (source_directory / f"{source_digest}.json").write_bytes(source_raw)
    body = {
        "contract": "QR_DOJO_AI_INVENTORY_QUOTE_WATERMARK_V1",
        "sequence": len(rows) + 1,
        "recorded_at_utc": timestamp_utc,
        "timestamp_utc": timestamp_utc,
        "pair": "USD_JPY",
        "bid": bid,
        "ask": ask,
        "source_sha256": source_digest,
        "capture_source_sha256": capture_source_sha256,
        "acquisition_receipt_sha256": acquisition_receipt_sha256,
        "slippage_pips_per_fill": slippage_pips_per_fill,
        "financing_pips_per_day": financing_pips_per_day,
        "previous_quote_sha256": (rows[-1]["quote_sha256"] if rows else "0" * 64),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    row = {
        **body,
        "quote_sha256": evaluator.quote_watermark_sha256(body),
    }
    rows.append(row)
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in rows
        )
    )
    return row


def _trusted_room(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    flat_allow: bool = False,
    flat_block: bool = False,
) -> dict[str, object]:
    assert not (flat_allow and flat_block)
    is_flat = flat_allow or flat_block
    monkeypatch.setattr(evaluator, "_trusted_repository_root", lambda: tmp_path)
    room = (
        tmp_path
        / "research"
        / "data"
        / "dojo_paper_ai_inventory_v1"
        / "rooms"
        / "paper-ai-inventory-experiment-001"
        / "paper-ai-inventory-room-001"
    )
    room.mkdir(parents=True)
    quote_path = room / evaluator.QUOTE_WATERMARK_LEDGER_NAME
    quote_rows: list[dict[str, object]] = []
    _append_quote_row(
        quote_path,
        quote_rows,
        timestamp_utc="2026-07-23T11:59:00Z",
        bid=162.99,
        ask=163.0,
        source_sha256="f" * 64,
    )
    decision_quote = _append_quote_row(
        quote_path,
        quote_rows,
        timestamp_utc="2026-07-23T12:00:01Z",
        bid=163.1,
        ask=163.11,
        source_sha256=SHA_A,
    )
    _append_quote_row(
        quote_path,
        quote_rows,
        timestamp_utc="2026-07-23T12:01:00Z",
        bid=162.95,
        ask=162.96,
        source_sha256=SHA_B,
    )
    endpoint = _append_quote_row(
        quote_path,
        quote_rows,
        timestamp_utc="2026-07-23T12:02:00Z",
        bid=163.2,
        ask=163.21,
        source_sha256=SHA_C,
    )
    broker_path = room / evaluator.BROKER_LEDGER_NAME
    broker_rows: list[dict[str, object]] = []
    fill = None
    if not is_flat:
        fill = _append_broker_row(
            broker_path,
            broker_rows,
            ts_utc="2026-07-23T11:59:00Z",
            event="FILL_MARKET",
            payload={
                "trade_id": "T000001",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 100.0,
                "entry": 163.0,
                "tp": 163.2,
                "sl": 162.75,
                "strategy_tag": "QR_DOJO_AI_INVENTORY_V2",
                "entry_context": {"strategy_tag": "QR_DOJO_AI_INVENTORY_V2"},
                "entry_context_sha256": SHA_D,
                "quote": {
                    "bid": 162.99,
                    "ask": 163.0,
                    "ts": "2026-07-23T11:59:00Z",
                },
            },
        )
    signal_body = {
        "pair": "USD_JPY",
        "side": "LONG",
        "order_type": "MARKET",
        "units": 100.0,
        "price": None,
        "strategy_tag": "QR_DOJO_AI_INVENTORY_V2",
        "entry_context_sha256": SHA_D,
        "tp_pips": 6.0,
        "sl_pips": 25.0,
        "observed_at_utc": "2026-07-23T12:00:01Z",
    }
    signal = {
        **signal_body,
        "signal_identity_sha256": entry_signal_identity_sha256(signal_body),
    }

    response = {
        "action": (
            "ALLOW_NEW_VIRTUAL"
            if flat_allow
            else "BLOCK_NEW"
            if flat_block
            else "HOLD"
        ),
        "reason_code": (
            "ENTRY_ALLOWED"
            if flat_allow
            else "ENTRY_BLOCKED"
            if flat_block
            else "THESIS_ALIVE"
        ),
        "reason": _assessment_json(),
        "virtual_units": None,
        "confidence": 0.8,
    }
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    executable = Path(sys.executable).resolve(strict=True)
    executable_stat = executable.stat()
    adapter_id = "evaluator-test-adapter"
    model_id = "evaluator-test-model"
    manifest = {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "argv": [str(executable), "-c", "pass"],
        "executor_uid": executable_stat.st_uid,
        "executor_gid": executable_stat.st_gid,
        "signature_key_id": "evaluator-test-key",
        "ed25519_public_key_base64": public_key,
        "timeout_seconds": 5,
    }
    manifest["command_manifest_sha256"] = (
        producer_module.command_adapter_manifest_sha256(manifest)
    )
    monkeypatch.setattr(
        producer_module,
        "_TRUSTED_COMMAND_ADAPTERS",
        {adapter_id: manifest},
    )
    request_sha = "0" * 64
    response_sha = _sha(response)
    signed_body = {
        "contract": producer_module.DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT,
        "adapter_id": adapter_id,
        "model_id": model_id,
        "request_sha256": request_sha,
        "response": response,
        "signature_key_id": "evaluator-test-key",
    }
    signed_payload = json.dumps(
        signed_body, sort_keys=True, separators=(",", ":")
    ).encode()
    invoke_body = {
        "contract": producer_module.DOJO_AI_COMMAND_INVOKE_RECEIPT_CONTRACT,
        "adapter_id": adapter_id,
        "model_id": model_id,
        "command_manifest_sha256": manifest["command_manifest_sha256"],
        "executable_sha256": manifest["executable_sha256"],
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "executor_uid": executable_stat.st_uid,
        "executor_gid": executable_stat.st_gid,
        "argv_sha256": _sha(manifest["argv"]),
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "signed_response": response,
        "signature_key_id": "evaluator-test-key",
        "signature_base64": base64.b64encode(private_key.sign(signed_payload)).decode(
            "ascii"
        ),
        "signed_payload_sha256": hashlib.sha256(signed_payload).hexdigest(),
        "started_at_utc": "2026-07-23T12:00:01Z",
        "completed_at_utc": "2026-07-23T12:00:01Z",
        "exit_code": 0,
    }
    invoke_receipt = {
        **invoke_body,
        "invoke_receipt_sha256": _sha(invoke_body),
    }
    receipt_body = {
        "contract": DOJO_AI_PRODUCER_RECEIPT_CONTRACT,
        "producer_id": "codex-ai-evaluator-test",
        "model_id": model_id,
        "evidence_packet_sha256": SHA_A,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        **response,
        "entry_signal_identity_sha256": (
            signal["signal_identity_sha256"] if is_flat else None
        ),
        "command_invoke_receipt": invoke_receipt,
        "produced_at_utc": "2026-07-23T12:00:01Z",
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed_receipt = _seal_producer_receipt(receipt_body, require_digest=False)
    receipt_path = _write_ai_inventory_producer_receipt(room, sealed_receipt)
    receipt_sha = sealed_receipt["receipt_sha256"]

    decision_path = room / evaluator.DECISION_LEDGER_NAME
    proposal = {
        "contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "cutoff_at_utc": "2026-07-23T12:00:01Z",
        "expires_at_utc": (
            "2026-07-23T12:01:31Z" if is_flat else "2026-07-23T12:10:01Z"
        ),
        "action": response["action"],
        "virtual_units": None,
        "confidence": 0.8,
        "admission_binding": (
            {
                "entry_signal": signal,
                "evidence_packet_sha256": SHA_A,
                "permit_expires_at_utc": "2026-07-23T12:01:31Z",
            }
            if flat_allow
            else None
        ),
        "reason_code": response["reason_code"],
        "reason": _assessment_json(),
        "session_binding": {
            "experiment_id": room.parent.name,
            "room_id": room.name,
            "session_contract_sha256": "1" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "policy_binding": {
            "policy_id": "paper-ai-inventory-policy-v1",
            "policy_sha256": "2" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "candidate_binding": {
            "candidate_id": "3" * 64,
            "candidate_sha256": "3" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "spec_binding": {
            "spec_id": "paper-ai-inventory-spec-v1",
            "spec_sha256": "4" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "lifecycle_binding": {
            "paper_eligible_event_sha256": "5" * 64,
            "candidate_lifecycle_ledger_tip_sha256": "6" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "ai_decision_binding": {
            "producer_id": receipt_body["producer_id"],
            "model_id": receipt_body["model_id"],
            "request_sha256": receipt_body["request_sha256"],
            "response_sha256": receipt_body["response_sha256"],
            "evidence_packet_sha256": receipt_body["evidence_packet_sha256"],
            "producer_receipt_sha256": receipt_sha,
            "produced_at_utc": receipt_body["produced_at_utc"],
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "ledger_binding": {
            "sha256": fill["sha"] if fill is not None else "0" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "state_binding": {
            "sha256": "7" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "snapshot_binding": {
            "sha256": "8" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "position_binding": {
            "position_id": "FLAT:USD_JPY" if is_flat else "T000001",
            "pair": "USD_JPY",
            "side": "FLAT" if is_flat else "LONG",
            "units": 0.0 if is_flat else 100.0,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V2",
            "entry_context_sha256": SHA_D,
            "sha256": "9" * 64,
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "quote_binding": {
            "pair": "USD_JPY",
            "bid": 163.1,
            "ask": 163.11,
            "sha256": decision_quote["source_sha256"],
            "observed_at_utc": "2026-07-23T12:00:01Z",
        },
        "source_watermarks": [
            {
                "source_id": "quotes:USD_JPY",
                "sha256": decision_quote["source_sha256"],
                "watermark_at_utc": "2026-07-23T12:00:01Z",
                "max_age_seconds": 300,
            }
        ],
        "max_dynamic_evidence_age_seconds": 300,
        "max_record_lag_seconds": 300,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    with patch(
        "quant_rabbit.dojo_ai_inventory._utc_now",
        return_value=datetime(2026, 7, 23, 12, 0, 2, tzinfo=timezone.utc),
    ):
        decision = append_inventory_decision(decision_path, proposal).record

    common_receipt = {
        "decision_sha256": decision["decision_sha256"],
        "decision_identity_sha256": decision["decision_identity_sha256"],
        "action": response["action"],
        "virtual_units": None,
        "confidence": 0.8,
        "room_id": room.name,
        "session_id": room.parent.name,
        "candidate_id": "3" * 64,
        "policy_id": "paper-ai-inventory-policy-v1",
        "spec_id": "paper-ai-inventory-spec-v1",
        "ai_producer_id": receipt_body["producer_id"],
        "ai_model_id": receipt_body["model_id"],
        "ai_request_sha256": receipt_body["request_sha256"],
        "ai_response_sha256": receipt_body["response_sha256"],
        "ai_evidence_packet_sha256": receipt_body["evidence_packet_sha256"],
        "ai_producer_receipt_sha256": receipt_sha,
        "ai_produced_at_utc": receipt_body["produced_at_utc"],
        "position_id": "FLAT:USD_JPY" if is_flat else "T000001",
        "pair": "USD_JPY",
        "strategy_tag": "QR_DOJO_AI_INVENTORY_V2",
        "admission_binding": proposal["admission_binding"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
        "consume_at_utc": "2026-07-23T12:00:02Z",
    }
    reservation = _append_broker_row(
        broker_path,
        broker_rows,
        ts_utc="2026-07-23T12:00:02Z",
        event="AI_INVENTORY_ACTION_RESERVED",
        payload=common_receipt,
    )
    applied = _append_broker_row(
        broker_path,
        broker_rows,
        ts_utc="2026-07-23T12:00:03Z",
        event="AI_INVENTORY_ACTION_APPLIED",
        payload={
            **common_receipt,
            "reservation_sha256": reservation["sha"],
            "close_sha256": None,
            "realized_pl_jpy": None,
            "block_new": flat_block,
            "allow_new_virtual": flat_allow,
            "single_use_entry_permit": flat_allow,
            "entry_proxy_consumed": False if flat_allow else None,
            "status": "APPLIED",
        },
    )
    settlement = None
    if not is_flat:
        settlement = _append_broker_row(
            broker_path,
            broker_rows,
            ts_utc="2026-07-23T12:02:00Z",
            event="EXIT_TP",
            payload={
                "trade_id": "T000001",
                "price": 163.2,
                "pl_jpy": 20.0,
                "strategy_tag": "QR_DOJO_AI_INVENTORY_V2",
                "entry_context_sha256": SHA_D,
                "quote": {
                    "bid": 163.2,
                    "ask": 163.21,
                    "ts": "2026-07-23T12:02:00Z",
                },
            },
        )

    return {
        "room": room,
        "decision": decision,
        "receipt_path": receipt_path,
        "broker_path": broker_path,
        "quote_path": quote_path,
        "applied": applied,
        "settlement": settlement,
        "endpoint": endpoint,
        "signal": signal,
    }


def test_append_and_validate_uses_internal_score_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"

    result = _append(ledger, _outcome_packet())

    assert result.appended is True
    assert result.record["scored_at_utc"] == "2026-07-20T12:00:00Z"
    assert result.record["order_authority"] == "NONE"
    assert result.record["evaluation_is_not_action"] is True
    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)
    assert validation["valid"] is True
    assert validation["row_count"] == 1
    assert (
        validation["terminal_evaluation_sha256"] == result.record["evaluation_sha256"]
    )


def test_public_raw_append_rejects_caller_supplied_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="trusted evidence token",
    ):
        evaluator.append_ai_inventory_evaluation(
            tmp_path / "evaluations.jsonl", _outcome_packet()
        )


def test_trusted_api_recomputes_metrics_from_canonical_room_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    result = evaluator.evaluate_ai_inventory_outcome(
        fixture["room"],
        decision_sha256=fixture["decision"]["decision_sha256"],
        horizon_end_at_utc="2026-07-23T12:02:00Z",
        outcome_kind="SETTLEMENT",
    )

    assert result.appended is True
    assert (
        result.record["producer_receipt_sha256"]
        == fixture["decision"]["ai_decision_binding"]["producer_receipt_sha256"]
    )
    assert result.record["applied_receipt_sha256"] == fixture["applied"]["sha"]
    assert result.record["realized_pl_jpy"] == 20
    assert result.record["review_time_executable_exit_pl_jpy"] == 10
    assert result.record["actual_exit_pl_jpy"] == 20
    assert result.record["counterfactual_delta_jpy"] == -10
    assert result.record["mfe_jpy"] == 20
    assert result.record["mae_jpy"] == -5
    assert result.record["settlement_reason"] == "TAKE_PROFIT"


def test_trusted_api_rejects_hash_valid_but_miscalculated_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    rows = [
        json.loads(line)
        for line in fixture["broker_path"].read_text(encoding="utf-8").splitlines()
    ]
    settlement = rows[-1]
    settlement["payload"]["pl_jpy"] = 999.0
    body = {
        key: settlement[key] for key in ("ts_utc", "event", "payload", "prev_sha")
    }
    settlement["sha"] = _sha(body)
    fixture["broker_path"].write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="entry/exit/units/cost recomputation",
    ):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"],
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
        )


def test_partial_reduce_uses_the_same_close_units_for_actual_and_path_metrics() -> None:
    decision = {"action": "REDUCE_VIRTUAL", "virtual_units": 40.0}
    position = {
        "position_id": "T000001",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100.0,
    }
    fill = {
        "event": "FILL_MARKET",
        "payload": {
            "entry": 163.0,
            "quote": {"bid": 162.99, "ask": 163.0, "ts": "2026-07-23T11:59:00Z"},
        },
    }
    settlement = {
        "event": "CLOSE",
        "payload": {
            "units": 40.0,
            "price": 163.2,
            "pl_jpy": 8.0,
            "quote": {"bid": 163.2, "ask": 163.21, "ts": "2026-07-23T12:02:00Z"},
        },
    }
    endpoint = {
        "timestamp_utc": "2026-07-23T12:02:00Z",
        "bid": 163.2,
        "ask": 163.21,
    }
    quote_rows = [
        {
            "pair": "USD_JPY",
            "timestamp_utc": "2026-07-23T12:02:00Z",
            "bid": 163.2,
            "ask": 163.21,
            "slippage_pips_per_fill": 0.0,
            "financing_pips_per_day": 0.0,
        }
    ]

    units = evaluator._settlement_evaluation_units(
        decision=decision,
        position=position,
        settlement=settlement,
    )
    actual = evaluator._recompute_settlement_pl_jpy(
        fill=fill,
        settlement=settlement,
        position=position,
        units=units,
        endpoint=endpoint,
        quote_rows=quote_rows,
    )
    path = evaluator._executable_pl_jpy(
        pair="USD_JPY",
        side="LONG",
        units=units,
        entry_price=163.0,
        bid=163.2,
        ask=163.21,
        timestamp_utc="2026-07-23T12:02:00Z",
        quote_rows=quote_rows,
    )

    assert units == 40.0
    assert actual == 8.0
    assert path == pytest.approx(8.0)


def test_trusted_fixed_horizon_ignores_post_horizon_settlement_and_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    result = evaluator.evaluate_ai_inventory_outcome(
        fixture["room"],
        decision_sha256=fixture["decision"]["decision_sha256"],
        horizon_end_at_utc="2026-07-23T12:01:00Z",
        outcome_kind="FIXED_HORIZON",
    )

    assert result.record["realized_outcome"] == "LOSS"
    assert result.record["actual_exit_pl_jpy"] == -5
    assert result.record["mfe_jpy"] == 10
    assert result.record["mae_jpy"] == -5
    assert result.record["settlement_reason"] is None


def test_trusted_api_scores_flat_allow_entry_gate_prospectively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch, flat_allow=True)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    result = evaluator.evaluate_ai_inventory_outcome(
        fixture["room"],
        decision_sha256=fixture["decision"]["decision_sha256"],
        horizon_end_at_utc="2026-07-23T12:02:00Z",
        outcome_kind="FIXED_HORIZON",
    )

    assert result.appended is True
    assert result.record["position_identity"] is None
    assert result.record["signal_identity"]["side"] == "LONG"
    assert result.record["realized_outcome"] == "WIN"
    assert result.record["realized_pl_jpy"] == pytest.approx(9.0)
    assert result.record["declared_regime"] == "TREND"


def test_trusted_api_scores_flat_block_entry_gate_prospectively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch, flat_block=True)
    monkeypatch.setattr(
        evaluator,
        "verify_ai_inventory_evidence_packet",
        lambda _root, _path: {
            "bindings": {
                "experiment_id": fixture["room"].parent.name,
                "room_id": fixture["room"].name,
            },
            "entry_signal": fixture["signal"],
        },
    )
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    result = evaluator.evaluate_ai_inventory_outcome(
        fixture["room"],
        decision_sha256=fixture["decision"]["decision_sha256"],
        horizon_end_at_utc="2026-07-23T12:02:00Z",
        outcome_kind="FIXED_HORIZON",
    )

    assert result.record["realized_outcome"] == "LOSS"
    assert result.record["realized_pl_jpy"] == pytest.approx(-9.0)
    assert result.record["actual_exit_pl_jpy"] == 0


def test_trusted_api_rejects_noncanonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="canonical paper-ai-inventory room",
    ):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"].parent.parent,
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
        )


def test_trusted_api_rejects_symlinked_quote_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    quote_path = fixture["quote_path"]
    actual = quote_path.with_name("actual-quotes.jsonl")
    quote_path.rename(actual)
    quote_path.symlink_to(actual)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="canonical source",
    ):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"],
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
        )


def test_trusted_api_rejects_missing_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    broker_path = fixture["broker_path"]
    rows = [json.loads(line) for line in broker_path.read_text().splitlines()]
    broker_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:-1])
    )
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="settlement",
    ):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"],
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
        )


def test_trusted_api_rejects_tampered_broker_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    broker_path = fixture["broker_path"]
    rows = [json.loads(line) for line in broker_path.read_text().splitlines()]
    rows[-1]["payload"]["pl_jpy"] = 999
    broker_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(
        evaluator.AiInventoryEvaluationIntegrityError,
        match="digest mismatch",
    ):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"],
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
        )


def test_trusted_api_has_no_caller_supplied_performance_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    _patch_clock(
        monkeypatch,
        datetime(2026, 7, 23, 12, 3, tzinfo=timezone.utc),
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        evaluator.evaluate_ai_inventory_outcome(
            fixture["room"],
            decision_sha256=fixture["decision"]["decision_sha256"],
            horizon_end_at_utc="2026-07-23T12:02:00Z",
            outcome_kind="SETTLEMENT",
            realized_pl_jpy=999_999,
        )


def test_exact_retry_is_idempotent_even_while_market_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "evaluations.jsonl"
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    first = _append(ledger, _outcome_packet())
    _patch_clock(monkeypatch, CLOSED_SCORE_TIME)

    retried = _append(ledger, _outcome_packet())

    assert retried.appended is False
    assert retried.record == first.record
    assert ledger.read_bytes().count(b"\n") == 1


def test_new_weekend_score_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, CLOSED_SCORE_TIME)

    with pytest.raises(evaluator.AiInventoryEvaluationMarketClosedError):
        _append(tmp_path / "evaluations.jsonl", _outcome_packet())


def test_public_weekend_gate_runs_before_source_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _trusted_room(tmp_path, monkeypatch)
    _patch_clock(monkeypatch, CLOSED_SCORE_TIME)
    with patch.object(evaluator, "_build_trusted_outcome_packet") as builder:
        with pytest.raises(evaluator.AiInventoryEvaluationMarketClosedError):
            evaluator.evaluate_ai_inventory_outcome(
                fixture["room"],
                decision_sha256=fixture["decision"]["decision_sha256"],
                horizon_end_at_utc="2026-07-23T12:02:00Z",
                outcome_kind="SETTLEMENT",
            )
    builder.assert_not_called()


def test_same_decision_and_horizon_conflict_never_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"
    _append(ledger, _outcome_packet())
    before = ledger.read_bytes()
    conflicting = _outcome_packet()
    conflicting["settlement_reason"] = "CEILING"

    with pytest.raises(evaluator.AiInventoryEvaluationConflictError):
        _append(ledger, conflicting)

    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("realized_pl_jpy", float("nan")),
        ("mfe_jpy", float("inf")),
        ("regime_confidence", True),
        ("regime_brier_score", -0.01),
    ],
)
def test_nonfinite_or_invalid_numbers_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet[field] = value

    with pytest.raises(ValueError):
        _append(tmp_path / "evaluations.jsonl", packet)


def test_counterfactual_delta_and_calibration_are_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    bad_delta = _outcome_packet()
    bad_delta["counterfactual_delta_jpy"] = 999
    with pytest.raises(ValueError, match="COUNTERFACTUAL_DELTA_MISMATCH"):
        _append(tmp_path / "delta.jsonl", bad_delta)

    bad_brier = _outcome_packet()
    bad_brier["regime_brier_score"] = 0.9
    with pytest.raises(ValueError, match="REGIME_BRIER_SCORE_MISMATCH"):
        _append(tmp_path / "brier.jsonl", bad_brier)


def test_decision_cutoff_must_precede_completed_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet["decision_cutoff_at_utc"] = packet["horizon_end_at_utc"]

    with pytest.raises(ValueError, match="DECISION_CUTOFF_NOT_BEFORE_HORIZON"):
        _append(tmp_path / "evaluations.jsonl", packet)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_end_at_utc", "2026-07-20T12:00:01Z"),
        ("outcome_observed_at_utc", "2026-07-20T12:00:01Z"),
    ],
)
def test_future_outcome_time_fails_against_internal_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet[field] = value
    if field == "horizon_end_at_utc":
        packet["outcome_observed_at_utc"] = value

    with pytest.raises(
        evaluator.AiInventoryEvaluationError,
        match="internally authored score time",
    ):
        _append(tmp_path / "evaluations.jsonl", packet)


def test_future_source_watermark_fails_against_internal_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet["source_watermarks"][0]["watermark_at_utc"] = "2026-07-20T12:00:01Z"

    with pytest.raises(
        evaluator.AiInventoryEvaluationError,
        match="source watermark",
    ):
        _append(tmp_path / "evaluations.jsonl", packet)


def test_fixed_horizon_requires_null_settlement_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet["outcome_kind"] = "FIXED_HORIZON"
    packet["settlement_reason"] = None

    result = _append(tmp_path / "evaluations.jsonl", packet)

    assert result.record["outcome_kind"] == "FIXED_HORIZON"


def test_position_or_signal_identity_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet["position_identity"] = None
    packet["signal_identity"] = None

    with pytest.raises(ValueError, match="MISSING_POSITION_OR_SIGNAL_IDENTITY"):
        _append(tmp_path / "evaluations.jsonl", packet)


def test_truncated_ledger_fails_validation_and_is_not_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"
    ledger.write_bytes(b'{"partial":true}')
    before = ledger.read_bytes()

    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)
    assert validation["valid"] is False
    assert "TRUNCATED_FINAL_ROW" in validation["issues"]
    with pytest.raises(evaluator.AiInventoryEvaluationIntegrityError):
        _append(ledger, _outcome_packet())
    assert ledger.read_bytes() == before


def test_duplicate_json_keys_fail_full_ledger_validation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "evaluations.jsonl"
    ledger.write_bytes(b'{"sequence":1,"sequence":2}\n')

    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)

    assert validation["valid"] is False
    assert any("INVALID_JSON" in issue for issue in validation["issues"])


def test_tampering_breaks_digest_and_blocks_later_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"
    _append(ledger, _outcome_packet())
    row = json.loads(ledger.read_text())
    row["realized_pl_jpy"] = 999
    ledger.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    before = ledger.read_bytes()

    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)
    assert validation["valid"] is False
    assert any("EVALUATION_SHA256_MISMATCH" in issue for issue in validation["issues"])
    later = _outcome_packet()
    later["decision_sha256"] = "f" * 64
    later["horizon_end_at_utc"] = "2026-07-20T11:01:00Z"
    with pytest.raises(evaluator.AiInventoryEvaluationIntegrityError):
        _append(ledger, later)
    assert ledger.read_bytes() == before


def test_duplicate_decision_horizon_identity_is_integrity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"
    first = _append(ledger, _outcome_packet()).record
    duplicate = copy.deepcopy(first)
    duplicate["sequence"] = 2
    duplicate["previous_evaluation_sha256"] = first["evaluation_sha256"]
    duplicate["evaluation_sha256"] = evaluator._evaluation_sha256(duplicate)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(duplicate, sort_keys=True, separators=(",", ":")) + "\n"
        )

    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)

    assert validation["valid"] is False
    assert any(
        "DUPLICATE_EVALUATION_IDENTITY" in issue for issue in validation["issues"]
    )


def test_forged_weekend_score_fails_full_ledger_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    ledger = tmp_path / "evaluations.jsonl"
    row = _append(ledger, _outcome_packet()).record
    row["scored_at_utc"] = "2026-07-25T12:00:00Z"
    row["evaluation_sha256"] = evaluator._evaluation_sha256(row)
    ledger.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    validation = evaluator.validate_ai_inventory_evaluation_ledger(ledger)

    assert validation["valid"] is False
    assert any("SCORED_WHILE_FX_CLOSED" in issue for issue in validation["issues"])


def test_caller_cannot_supply_scored_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clock(monkeypatch, OPEN_SCORE_TIME)
    packet = _outcome_packet()
    packet["scored_at_utc"] = "2026-07-20T11:59:00Z"

    with pytest.raises(ValueError, match="UNKNOWN_OUTCOME_FIELD"):
        _append(tmp_path / "evaluations.jsonl", packet)


def test_module_has_no_broker_or_oanda_import() -> None:
    tree = ast.parse(Path(evaluator.__file__).read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert not any("broker" in module.lower() for module in imported_modules)
    assert not any("oanda" in module.lower() for module in imported_modules)
