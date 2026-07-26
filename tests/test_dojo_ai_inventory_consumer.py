from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    append_inventory_decision,
    inventory_decision_identity_sha256,
    inventory_decision_sha256,
    validate_inventory_decision,
)
from quant_rabbit.dojo_ai_evidence_packet import (
    DOJO_AI_EVIDENCE_PACKET_CONTRACT,
    entry_signal_identity_sha256,
    verify_ai_inventory_evidence_packet,
    write_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory_consumer import (
    InventoryConsumerIntegrityError,
    InventoryReservationOutstandingError,
    consume_inventory_decision,
    reconcile_inventory_checkpoint_suffix,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    PRODUCER_RECEIPT_DIRECTORY,
    AllowlistedCommandModelAdapter,
    command_adapter_manifest_sha256,
    produce_ai_inventory_proposal,
)
from quant_rabbit.dojo_autonomous_improvement import (
    CANDIDATE_SPEC_CONTRACT,
    append_candidate_event,
    initialize_research_root,
    validate_research_root,
)
from quant_rabbit.dojo_replay_gates import (
    PROOF_MANIFEST_CONTRACT,
    canonical_proof_manifest_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT,
    CANONICAL_RESEARCH_RELATIVE_ROOT,
    FUTURE_REGISTRY_CONTRACT,
    JOB_MANIFEST_CONTRACT,
    PROOF_ARTIFACT_CONTRACT,
    REPLAY_JOB_OWNER_CONTRACT,
    REPLAY_OUTPUT_MANIFEST_CONTRACT,
    SOURCE_MANIFEST_CONTRACT,
    canonical_proof_artifact_bytes,
    issue_paper_ai_inventory_launch_preflight,
)
from quant_rabbit.dojo_replay_worker_receipt import (
    REPLAY_WORKER_RECEIPT_CONTRACT,
    replay_worker_config_sha256,
)
from quant_rabbit.virtual_broker import VBOrder, VBPosition, VirtualBroker


UTC = timezone.utc
EXPERIMENT_ID = "paper-ai-inventory-experiment-001"
ROOM_ID = "paper-ai-inventory-room-001"
POLICY_ID = "paper-ai-inventory-policy-001"
POLICY_SHA256 = "3" * 64
QUOTE_AT = "2026-07-23T12:00:01Z"
CONSUME_AT = datetime(2026, 7, 23, 12, 0, 5, tzinfo=UTC)
LIFECYCLE_AT = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
ENTRY_CONTEXT = {
    "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
    "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
    "signal": "paper_ai_inventory_fixture",
    "pair": "USD_JPY",
    "side": "LONG",
}
ENTRY_CONTEXT_SHA256 = hashlib.sha256(
    json.dumps(
        ENTRY_CONTEXT,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _guard() -> dict[str, object]:
    return {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sealed(value: dict[str, object], field: str) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(field, None)
    digest = hashlib.sha256(_canonical_bytes(body).rstrip(b"\n")).hexdigest()
    return {**body, field: digest}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _write_source_capture_manifest(repository_root: Path) -> str:
    public_key = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    body: dict[str, object] = {
        "contract": "QR_DOJO_AI_SOURCE_CAPTURE_MANIFEST_V1",
        "manifest_id": "consumer-source-capture-v1",
        "capture_key_id": "consumer-source-capture-key-v1",
        "ed25519_public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "allowed_source_roles": ["candles", "news", "quote"],
        "allowed_provider_kinds": ["official", "read_only_broker"],
        "source_adapters": [
            {
                "source_role": "candles",
                "provider_kind": "read_only_broker",
                "adapter_id": "consumer-candles-adapter-v1",
                "adapter_module": "quant_rabbit.consumer_capture",
                "adapter_callable": "capture_candles",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "2" * 64,
            },
            {
                "source_role": "news",
                "provider_kind": "official",
                "adapter_id": "consumer-news-adapter-v1",
                "adapter_module": "quant_rabbit.consumer_capture",
                "adapter_callable": "capture_news",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "3" * 64,
            },
            {
                "source_role": "quote",
                "provider_kind": "read_only_broker",
                "adapter_id": "consumer-quote-adapter-v1",
                "adapter_module": "quant_rabbit.consumer_capture",
                "adapter_callable": "capture_quote",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "4" * 64,
            },
        ],
        **_guard(),
    }
    manifest = {**body, "manifest_sha256": _canonical_sha256(body)}
    raw = _canonical_bytes(manifest)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    path = (
        repository_root
        / "research/data/dojo_paper_ai_inventory_v1/source_capture/manifests"
        / f"{file_sha256}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return file_sha256


def _trusted_replay_worker(
    repository_root: Path,
) -> tuple[Ed25519PrivateKey, dict[str, object], list[str]]:
    private_key = Ed25519PrivateKey.generate()
    executable = Path(sys.executable).resolve()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    body: dict[str, object] = {
        "adapter_id": "consumer-replay-worker-v1",
        "model_id": "consumer-replay-model-v1",
        "producer_id": "consumer-replay-producer-v1",
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "signature_key_id": "consumer-replay-key-v1",
        "ed25519_public_key_base64": base64.b64encode(public_key).decode("ascii"),
    }
    trusted = {**body, "config_sha256": replay_worker_config_sha256(body)}
    argv = [str(executable), "-I", str(repository_root / "paper-replay.py")]
    return private_key, trusted, argv


def _write_replay_worker_receipt(
    *,
    repository_root: Path,
    replay_root: Path,
    private_key: Ed25519PrivateKey,
    trusted_worker: dict[str, object],
    argv: list[str],
    candidate_id: str,
    spec_sha256: str,
    job_manifest_sha256: str,
    output_manifest_sha256: str,
    git_head: str,
    windows: dict[str, object],
    source_file_bindings: list[dict[str, str]],
    artifact_raw: bytes,
) -> dict[str, object]:
    executable = Path(str(trusted_worker["executable_path"]))
    executable_stat = executable.stat()
    body: dict[str, object] = {
        "contract": REPLAY_WORKER_RECEIPT_CONTRACT,
        "adapter_id": trusted_worker["adapter_id"],
        "model_id": trusted_worker["model_id"],
        "config_sha256": trusted_worker["config_sha256"],
        "producer_id": trusted_worker["producer_id"],
        "executable_path": str(executable),
        "executable_sha256": trusted_worker["executable_sha256"],
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "executable_uid": executable_stat.st_uid,
        "executable_gid": executable_stat.st_gid,
        "argv": argv,
        "argv_sha256": _canonical_sha256(argv),
        "git_head": git_head,
        "git_head_sha256": hashlib.sha256(git_head.encode("ascii")).hexdigest(),
        "candidate_id": candidate_id,
        "spec_sha256": spec_sha256,
        "policy_sha256": POLICY_SHA256,
        "job_manifest_sha256": job_manifest_sha256,
        "output_manifest_sha256": output_manifest_sha256,
        "source_files": source_file_bindings,
        "windows": windows,
        "costs": ["BASE", "STRESS"],
        "intrabar_paths": ["OHLC", "OLHC"],
        "results_artifact_path": str(
            (replay_root / "proof_artifact.json").relative_to(repository_root)
        ),
        "results_artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "completed_at_utc": "2026-07-22T12:00:03Z",
        **_guard(),
        "signature_key_id": trusted_worker["signature_key_id"],
    }
    signed_payload = _canonical_bytes(body).rstrip(b"\n")
    unsigned_receipt = {
        **body,
        "signed_payload_sha256": hashlib.sha256(signed_payload).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(signed_payload)).decode(
            "ascii"
        ),
    }
    receipt = {
        **unsigned_receipt,
        "receipt_sha256": _canonical_sha256(unsigned_receipt),
    }
    (replay_root / "worker_receipt.json").write_bytes(_canonical_bytes(receipt))
    return receipt


def _passing_arms() -> list[dict[str, object]]:
    arms: list[dict[str, object]] = []
    for window in ("TRAIN", "VAL", "S5"):
        for policy in ("BASELINE", "CANDIDATE"):
            for cost in ("BASE", "STRESS"):
                for intrabar in ("OHLC", "OLHC"):
                    baseline = policy == "BASELINE"
                    net = 300.0 if baseline else 500.0
                    arms.append(
                        {
                            "window": window,
                            "policy": policy,
                            "cost": cost,
                            "intrabar": intrabar,
                            "metrics": {
                                "settlements": 40,
                                "active_days": 24,
                                "net_jpy": net,
                                "profit_factor": 1.2 if baseline else 1.4,
                                "expectancy_jpy": net / 40,
                                "worst_day_jpy": -100.0 if baseline else -90.0,
                                "realized_drawdown_jpy": (200.0 if baseline else 180.0),
                                "margin_events": 0,
                                "ruin_events": 0,
                                "unresolved_positions": 0,
                                "unresolved_orders": 0,
                                "end_of_replay_forced_close_count": 0,
                                "end_of_replay_forced_close_net_jpy": 0.0,
                            },
                        }
                    )
    return arms


def _initialize_git(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "dojo@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "DOJO Test"],
        check=True,
    )
    seed = root / "README.md"
    seed.write_text("paper AI fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _admission_binding(
    evidence_packet_sha256: str = "a" * 64,
) -> dict[str, object]:
    signal: dict[str, object] = {
        "pair": "USD_JPY",
        "side": "LONG",
        "order_type": "LIMIT",
        "units": 100.0,
        "price": 163.0,
        "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
        "entry_context_sha256": ENTRY_CONTEXT_SHA256,
        "tp_pips": 3.0,
        "sl_pips": 25.0,
        "observed_at_utc": QUOTE_AT,
    }
    signal["signal_identity_sha256"] = entry_signal_identity_sha256(signal)
    return {
        "entry_signal": signal,
        "evidence_packet_sha256": evidence_packet_sha256,
        "permit_expires_at_utc": "2026-07-23T12:00:45Z",
    }


def _write_producer_receipt(
    repository_root: Path,
    room_root: Path,
    lifecycle: dict[str, object],
    *,
    action: str,
    virtual_units: float | None,
    confidence: float,
    admission_binding: dict[str, object] | None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    flat = admission_binding is not None
    entry_signal = (
        copy.deepcopy(admission_binding["entry_signal"])
        if admission_binding is not None
        else None
    )
    packet: dict[str, object] = {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": QUOTE_AT,
        "bindings": {
            "launch_preflight_token_sha256": lifecycle.get(
                "launch_preflight_token_sha256", "0" * 64
            ),
            "git_head": lifecycle["git_head"],
            "git_branch": "codex/test-ai-inventory-consumer",
            "canonical_source_root": (
                "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
            ),
            "experiment_id": EXPERIMENT_ID,
            "room_id": ROOM_ID,
            "session_contract_sha256": "1" * 64,
            "candidate_id": lifecycle["candidate_id"],
            "candidate_sha256": lifecycle["candidate_sha256"],
            "spec_id": lifecycle["spec_id"],
            "spec_sha256": lifecycle["spec_sha256"],
            "policy_id": POLICY_ID,
            "policy_sha256": POLICY_SHA256,
            "paper_eligible_tip_sha256": lifecycle[
                "candidate_lifecycle_ledger_tip_sha256"
            ],
            "ledger_sha256": "0" * 64,
            "ledger_observed_at_utc": QUOTE_AT,
            "state_sha256": "5" * 64,
            "state_observed_at_utc": QUOTE_AT,
            "snapshot_sha256": "6" * 64,
            "snapshot_observed_at_utc": QUOTE_AT,
        },
        "position": {
            "position_id": "FLAT:USD_JPY" if flat else "T000001",
            "pair": "USD_JPY",
            "side": "FLAT" if flat else "LONG",
            "units": 0.0 if flat else 100.0,
            "entry_price": None if flat else 163.0,
            "opened_at_utc": None if flat else "2026-07-23T11:30:00Z",
            "observed_at_utc": QUOTE_AT,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": ENTRY_CONTEXT_SHA256,
            "take_profit": None if flat else 163.3,
            "stop_loss": None if flat else 162.75,
            "remaining_ceiling_seconds": 0 if flat else 1_800,
            "unrealized_pl_jpy": 0.0 if flat else 10.0,
            "gross_same_currency_units": 0.0 if flat else 100.0,
            "net_same_currency_units": 0.0 if flat else 100.0,
            "margin_used_jpy": 0.0 if flat else 1_000.0,
            "capital_locked_jpy": 0.0 if flat else 1_000.0,
            "same_direction_position_count": 0 if flat else 1,
        },
        "entry_signal": entry_signal,
        "quote": {
            "pair": "USD_JPY",
            "bid": 163.1,
            "ask": 163.11,
            "timestamp_utc": QUOTE_AT,
            "source_sha256": "a" * 64,
            "max_age_seconds": 90,
        },
        "candles": [
            {
                "pair": "USD_JPY",
                "granularity": "M1",
                "started_at_utc": "2026-07-23T11:59:00Z",
                "completed_at_utc": "2026-07-23T12:00:00Z",
                "bid_o": 163.0,
                "bid_h": 163.12,
                "bid_l": 162.99,
                "bid_c": 163.1,
                "ask_o": 163.01,
                "ask_h": 163.13,
                "ask_l": 163.0,
                "ask_c": 163.11,
                "source_sha256": "b" * 64,
                "max_age_seconds": 3_600,
            }
        ],
        "news_items": [],
        "calendar_items": [],
        "cross_asset_items": [],
        "dynamic_binding_max_age_seconds": 90,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    with patch(
        "quant_rabbit.dojo_ai_evidence_packet._utc_now",
        return_value=datetime(2026, 7, 23, 12, 0, 2, tzinfo=UTC),
    ):
        packet_path = write_ai_inventory_evidence_packet(repository_root, packet)
    verified = verify_ai_inventory_evidence_packet(repository_root, packet_path)

    response = {
        "action": action,
        "reason_code": "THESIS_INVALIDATED",
        "reason": "Prospective paper decision bound to immutable evidence.",
        "virtual_units": virtual_units,
        "confidence": confidence,
    }
    adapter_id = "trusted-consumer-test-adapter"
    model_id = "allowlisted-test-model-v1"
    executable = Path(sys.executable).resolve(strict=True)
    private_key = Ed25519PrivateKey.generate()
    private_key_base64 = base64.b64encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode("ascii")
    public_key_base64 = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    response_text = json.dumps(
        response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    script = (
        "import base64,hashlib,json,sys\n"
        "from cryptography.hazmat.primitives.asymmetric.ed25519 "
        "import Ed25519PrivateKey\n"
        "adapter_id,model_id,key_b64,response_json=sys.argv[1:]\n"
        "request=sys.stdin.buffer.read()\n"
        "body={'contract':'QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1',"
        "'adapter_id':adapter_id,'model_id':model_id,"
        "'request_sha256':hashlib.sha256(request).hexdigest(),"
        "'response':json.loads(response_json),"
        "'signature_key_id':'test-signing-key-v1'}\n"
        "payload=json.dumps(body,ensure_ascii=False,sort_keys=True,"
        "separators=(',',':'),allow_nan=False).encode()\n"
        "key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(key_b64))\n"
        "body['signature_base64']=base64.b64encode(key.sign(payload)).decode()\n"
        "sys.stdout.write(json.dumps(body,ensure_ascii=False,sort_keys=True,"
        "separators=(',',':'),allow_nan=False))\n"
    )
    item_stat = executable.stat()
    manifest: dict[str, object] = {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "argv": [
            str(executable),
            "-c",
            script,
            adapter_id,
            model_id,
            private_key_base64,
            response_text,
        ],
        "executor_uid": item_stat.st_uid,
        "executor_gid": item_stat.st_gid,
        "signature_key_id": "test-signing-key-v1",
        "ed25519_public_key_base64": public_key_base64,
        "timeout_seconds": 5,
    }
    manifest["command_manifest_sha256"] = command_adapter_manifest_sha256(manifest)
    produced_at = datetime(2026, 7, 23, 12, 0, 3, tzinfo=UTC)
    with (
        patch(
            "quant_rabbit.dojo_ai_inventory_producer._TRUSTED_COMMAND_ADAPTERS",
            {adapter_id: manifest},
        ),
        patch(
            "quant_rabbit.dojo_ai_inventory_producer._utc_now",
            return_value=produced_at,
        ),
    ):
        proposal = produce_ai_inventory_proposal(
            verified,
            AllowlistedCommandModelAdapter(adapter_id),
            producer_id="codex-dojo-single-reader-v1",
            room_root=room_root,
        )
    receipt = proposal["producer_receipt"]
    receipt_path = (
        room_root / PRODUCER_RECEIPT_DIRECTORY / f"{receipt['receipt_sha256']}.json"
    )
    return receipt_path, proposal, manifest


def _candidate_spec(
    source_sha256s: dict[str, str] | None = None,
    *,
    trusted_worker: dict[str, object],
    source_capture_manifest_sha256: str,
) -> dict[str, object]:
    source_sha256s = source_sha256s or {
        "TRAIN": "1" * 64,
        "VAL": "2" * 64,
        "S5": "3" * 64,
    }
    return {
        "contract": CANDIDATE_SPEC_CONTRACT,
        **_guard(),
        "family": "INVENTORY_RELEASE",
        "adapter_id": trusted_worker["adapter_id"],
        "model_id": trusted_worker["model_id"],
        "config_sha256": trusted_worker["config_sha256"],
        "producer_id": trusted_worker["producer_id"],
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
        "hypothesis": "release only invalidated paper inventory",
        "causal_narrative": "ceiling losses dominate bounded winners",
        "expected_mechanism": "release capital after prospective invalidation",
        "falsifier": "independent stress expectancy is non-positive",
        "affected_pair": "USD_JPY",
        "affected_strategy": "QR_DOJO_AI_INVENTORY_V1",
        "evidence_cohort": "immutable tagged settlements",
        "changed_rule": {
            "name": "ai_inventory_release",
            "baseline": False,
            "candidate": True,
        },
        "unchanged_controls": ["entry", "size", "tp", "sl", "ceiling"],
        "evidence_sha256s": ["e" * 64],
        "windows": {
            "TRAIN": {
                "from_utc": "2026-01-01T00:00:00+00:00",
                "to_utc": "2026-02-01T00:00:00+00:00",
                "source_sha256": source_sha256s["TRAIN"],
            },
            "VAL": {
                "from_utc": "2026-02-01T00:00:00+00:00",
                "to_utc": "2026-03-01T00:00:00+00:00",
                "source_sha256": source_sha256s["VAL"],
            },
            "S5": {
                "from_utc": "2026-03-01T00:00:00+00:00",
                "to_utc": "2026-04-01T00:00:00+00:00",
                "source_sha256": source_sha256s["S5"],
            },
        },
        "costs": {
            "BASE": {
                "slippage_pips_per_fill": 0.0,
                "financing_pips_per_day": 0.0,
            },
            "STRESS": {
                "slippage_pips_per_fill": 0.3,
                "financing_pips_per_day": 0.8,
            },
        },
        "intrabar_paths": ["OHLC", "OLHC"],
        "end_of_replay_forced_close_benefit": False,
        "risk_gates": {
            "min_settlements_per_independent_arm": 30,
            "min_active_days_per_independent_arm": 20,
            "min_independent_stress_pf": 1.25,
            "positive_net": True,
            "positive_expectancy": True,
            "worst_day_not_worse": True,
            "drawdown_not_worse": True,
            "margin_ruin_not_worse": True,
            "unresolved_end_exposure": False,
        },
        "death_codes": [
            "COST",
            "DIRECTION",
            "EXIT_TIMING",
            "INVENTORY",
            "MEASUREMENT",
            "OVERFIT",
            "REGIME_MISMATCH",
            "RISK",
        ],
    }


def _build_candidate_lifecycle(
    repository_root: Path,
    *,
    experiment_id: str = EXPERIMENT_ID,
    paper_eligible: bool = True,
) -> dict[str, object]:
    git_head = _initialize_git(repository_root)
    replay_private_key, trusted_replay_worker, replay_argv = _trusted_replay_worker(
        repository_root
    )
    source_capture_manifest_sha256 = _write_source_capture_manifest(repository_root)
    root = repository_root / CANONICAL_RESEARCH_RELATIVE_ROOT
    source_root = repository_root / "research/data/replay_source_fixture"
    source_root.mkdir(parents=True)
    source_manifest_bytes: dict[str, bytes] = {}
    for window in ("TRAIN", "VAL", "S5"):
        source_path = source_root / f"{window}.csv"
        raw = f"timestamp,bid,ask\n{window},163.00,163.01\n".encode()
        source_path.write_bytes(raw)
        source_manifest_bytes[window] = _canonical_bytes(
            {
                "contract": SOURCE_MANIFEST_CONTRACT,
                "granularity": "S5" if window == "S5" else "M1",
                "pairs": ["USD_JPY"],
                "files": [
                    {
                        "path": str(source_path.relative_to(repository_root)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ],
            }
        )
    source_sha256s = {
        window: hashlib.sha256(source_manifest_bytes[window]).hexdigest()
        for window in ("TRAIN", "VAL", "S5")
    }
    initialize_research_root(
        root,
        recorded_at_utc=LIFECYCLE_AT,
        implementation_sha256="f" * 64,
    )
    registration, _ = append_candidate_event(
        root,
        event_type="CANDIDATE_PREREGISTERED",
        payload={
            **_guard(),
            "spec": _candidate_spec(
                source_sha256s,
                trusted_worker=trusted_replay_worker,
                source_capture_manifest_sha256=source_capture_manifest_sha256,
            ),
        },
        recorded_at_utc=LIFECYCLE_AT + timedelta(seconds=1),
    )
    candidate_id = registration["payload"]["candidate_id"]
    sealed_spec = registration["payload"]["spec"]
    spec_sha256 = sealed_spec["spec_sha256"]
    replay_root = root / "candidates" / candidate_id / "replay"
    manifest_root = replay_root / "source_manifests"
    manifest_root.mkdir(parents=True)
    for window in ("TRAIN", "VAL", "S5"):
        (manifest_root / f"{window}.json").write_bytes(source_manifest_bytes[window])
    output = {
        "contract": REPLAY_OUTPUT_MANIFEST_CONTRACT,
        "candidate_id": candidate_id,
        "spec_sha256": spec_sha256,
        "policy_sha256": POLICY_SHA256,
        "git_head": git_head,
        "source_manifest_sha256s": source_sha256s,
        "adapter_id": trusted_replay_worker["adapter_id"],
        "model_id": trusted_replay_worker["model_id"],
        "config_sha256": trusted_replay_worker["config_sha256"],
        "producer_id": trusted_replay_worker["producer_id"],
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
        **_guard(),
    }
    output_raw = _canonical_bytes(output)
    (replay_root / "output_manifest.json").write_bytes(output_raw)
    output_sha256 = hashlib.sha256(output_raw).hexdigest()
    job = _sealed(
        {
            "contract": JOB_MANIFEST_CONTRACT,
            "candidate_id": candidate_id,
            "spec_sha256": spec_sha256,
            "policy_sha256": POLICY_SHA256,
            "git_head": git_head,
            "git_head_sha256": hashlib.sha256(git_head.encode("ascii")).hexdigest(),
            "output_manifest_sha256": output_sha256,
            "adapter_id": trusted_replay_worker["adapter_id"],
            "model_id": trusted_replay_worker["model_id"],
            "config_sha256": trusted_replay_worker["config_sha256"],
            "producer_id": trusted_replay_worker["producer_id"],
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
            "argv": replay_argv,
            "argv_sha256": _canonical_sha256(replay_argv),
            "files": [
                {
                    "path": str(
                        (manifest_root / f"{window}.json").relative_to(repository_root)
                    ),
                    "sha256": source_sha256s[window],
                }
                for window in ("TRAIN", "VAL", "S5")
            ],
            **_guard(),
        },
        "manifest_sha256",
    )
    (replay_root / "job_manifest.json").write_bytes(_canonical_bytes(job))
    process_sha256 = "6" * 64
    owner = _sealed(
        {
            "contract": REPLAY_JOB_OWNER_CONTRACT,
            "candidate_id": candidate_id,
            "job_manifest_sha256": job["manifest_sha256"],
            "pid": os.getpid(),
            "screen_name": "qr-dojo-improve-aiinventory",
            "process_command_sha256": process_sha256,
            "output_directory": str(replay_root.relative_to(repository_root)),
            "status": "COMPLETED",
            "completed_at_utc": "2026-07-22T12:00:02Z",
            **_guard(),
        },
        "owner_sha256",
    )
    (replay_root / "job_owner.json").write_bytes(_canonical_bytes(owner))
    append_candidate_event(
        root,
        event_type="REPLAY_STARTED",
        payload={
            **_guard(),
            "candidate_id": candidate_id,
            "job_lock": {
                "git_head_sha256": job["git_head_sha256"],
                "spec_sha256": spec_sha256,
                "policy_sha256": POLICY_SHA256,
                "output_manifest_sha256": output_sha256,
                "argv": replay_argv,
                "argv_sha256": job["argv_sha256"],
                "adapter_id": trusted_replay_worker["adapter_id"],
                "model_id": trusted_replay_worker["model_id"],
                "config_sha256": trusted_replay_worker["config_sha256"],
                "producer_id": trusted_replay_worker["producer_id"],
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
                "environment_allowlist": ["PATH", "PYTHONPATH"],
                "output_directory": str(replay_root.relative_to(repository_root)),
                "screen_name": "qr-dojo-improve-aiinventory",
                "pid": os.getpid(),
                "process_command_sha256": process_sha256,
                "job_manifest_sha256": job["manifest_sha256"],
                "job_owner_sha256": owner["owner_sha256"],
            },
        },
        recorded_at_utc=LIFECYCLE_AT + timedelta(seconds=2),
    )
    artifact_raw = canonical_proof_artifact_bytes(
        {
            "contract": PROOF_ARTIFACT_CONTRACT,
            "candidate_id": candidate_id,
            "spec_sha256": spec_sha256,
            "policy_sha256": POLICY_SHA256,
            "job_manifest_sha256": job["manifest_sha256"],
            "git_head": git_head,
            "git_head_sha256": job["git_head_sha256"],
            "artifact_manifest_sha256": output_sha256,
            "windows": sealed_spec["windows"],
            "completed_at_utc": "2026-07-22T12:00:02Z",
            "arms": _passing_arms(),
            **_guard(),
        }
    )
    (replay_root / "proof_artifact.json").write_bytes(artifact_raw)
    artifact = json.loads(artifact_raw)
    source_file_bindings = [
        {
            "window": window,
            "path": str((source_root / f"{window}.csv").relative_to(repository_root)),
            "sha256": hashlib.sha256(
                (source_root / f"{window}.csv").read_bytes()
            ).hexdigest(),
        }
        for window in ("TRAIN", "VAL", "S5")
    ]
    worker_receipt = _write_replay_worker_receipt(
        repository_root=repository_root,
        replay_root=replay_root,
        private_key=replay_private_key,
        trusted_worker=trusted_replay_worker,
        argv=replay_argv,
        candidate_id=candidate_id,
        spec_sha256=spec_sha256,
        job_manifest_sha256=job["manifest_sha256"],
        output_manifest_sha256=output_sha256,
        git_head=git_head,
        windows=sealed_spec["windows"],
        source_file_bindings=source_file_bindings,
        artifact_raw=artifact_raw,
    )
    proof_manifest = {
        "contract": PROOF_MANIFEST_CONTRACT,
        "candidate_id": candidate_id,
        "spec_sha256": spec_sha256,
        "policy_sha256": POLICY_SHA256,
        "artifact_manifest_sha256": output_sha256,
        "windows": sealed_spec["windows"],
        "arms": artifact["arms"],
        **_guard(),
    }
    proof_manifest_sha256 = canonical_proof_manifest_sha256(proof_manifest)
    append_candidate_event(
        root,
        event_type="REPLAY_PASSED",
        payload={
            **_guard(),
            "candidate_id": candidate_id,
            "independent_stress_metrics": {
                "pf": 1.5,
                "net": 100.0,
                "expectancy": 2.0,
                "worst_day_not_worse": True,
                "drawdown_not_worse": True,
                "margin_ruin_not_worse": True,
                "unresolved_end_exposure": False,
            },
            "proof_artifact_sha256": artifact["artifact_sha256"],
            "proof_artifact_bytes_sha256": hashlib.sha256(artifact_raw).hexdigest(),
            "proof_manifest_sha256": proof_manifest_sha256,
            "job_manifest_sha256": job["manifest_sha256"],
            "replay_worker_receipt_sha256": worker_receipt["receipt_sha256"],
        },
        recorded_at_utc=LIFECYCLE_AT + timedelta(seconds=3),
    )
    eligible_sha256 = "a" * 64
    launch_preflight_token_sha256 = "0" * 64
    registry = {
        "contract": FUTURE_REGISTRY_CONTRACT,
        "experiment_id": experiment_id,
        "proof_mode": "candidate",
        "window": {
            "start_utc": "2026-07-23T11:00:00Z",
            "end_utc": "2026-07-24T11:00:00Z",
        },
        "proof_binding": {
            "candidate_id": candidate_id,
            "spec_sha256": spec_sha256,
            "policy_sha256": POLICY_SHA256,
            "job_manifest_sha256": job["manifest_sha256"],
            "proof_artifact_sha256": artifact["artifact_sha256"],
            "git_head": git_head,
            "git_head_sha256": job["git_head_sha256"],
            "adapter_id": trusted_replay_worker["adapter_id"],
            "model_id": trusted_replay_worker["model_id"],
            "config_sha256": trusted_replay_worker["config_sha256"],
            "producer_id": trusted_replay_worker["producer_id"],
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
        },
        "rooms": [
            {
                "room_id": ROOM_ID,
                "candidate_id": candidate_id,
                "adapter_id": trusted_replay_worker["adapter_id"],
                "model_id": trusted_replay_worker["model_id"],
                "config_sha256": trusted_replay_worker["config_sha256"],
                "producer_id": trusted_replay_worker["producer_id"],
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
            }
        ],
        **_guard(),
    }
    registry_path = (
        repository_root
        / "config"
        / f"dojo_paper_rooms_ai_inventory_{candidate_id}.json"
    )
    registry_path.parent.mkdir()
    registry_path.write_bytes(_canonical_bytes(registry))
    if paper_eligible:
        with patch(
            "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
            {trusted_replay_worker["adapter_id"]: trusted_replay_worker},
        ):
            preflight = issue_paper_ai_inventory_launch_preflight(
                repository_root,
                candidate_id=candidate_id,
                future_registry_path=registry_path,
                recorded_at_utc=LIFECYCLE_AT + timedelta(seconds=4),
            )
        eligible_sha256 = preflight["paper_eligible_event_sha256"]
        launch_preflight_token_sha256 = preflight["launch_preflight_tokens"][ROOM_ID][
            "launch_preflight_token_sha256"
        ]
    research = validate_research_root(root)
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_id,
        "spec_id": f"candidate-spec:{candidate_id}",
        "spec_sha256": spec_sha256,
        "paper_eligible_event_sha256": eligible_sha256,
        "candidate_lifecycle_ledger_tip_sha256": research["candidate"]["tip_sha256"],
        "candidate_lifecycle_ledger_path": root / "candidate_ledger.jsonl",
        "repository_root": repository_root,
        "git_head": git_head,
        "launch_preflight_token_sha256": launch_preflight_token_sha256,
        "trusted_replay_workers": {
            trusted_replay_worker["adapter_id"]: trusted_replay_worker
        },
    }


def _broker(
    root: Path,
    *,
    experiment_id: str = EXPERIMENT_ID,
    room_id: str = ROOM_ID,
    units: float = 100.0,
) -> tuple[VirtualBroker, Path]:
    dedicated_root = root / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT
    room_root = dedicated_root / experiment_id / room_id
    room_root.mkdir(parents=True, exist_ok=True)
    broker = VirtualBroker(room_root / "broker.jsonl", fast_ledger=True)
    broker.positions["T000001"] = VBPosition(
        trade_id="T000001",
        pair="USD_JPY",
        side="LONG",
        units=units,
        entry_price=163.0,
        opened_ts="2026-07-23T11:30:00Z",
        strategy_tag="QR_DOJO_AI_INVENTORY_V1",
        entry_context=copy.deepcopy(ENTRY_CONTEXT),
        entry_context_sha256=ENTRY_CONTEXT_SHA256,
    )
    broker.last_quotes["USD_JPY"] = (163.1, 163.11, QUOTE_AT)
    return broker, dedicated_root


def _runtime(
    dedicated_root: Path,
    lifecycle: dict[str, object],
    *,
    ledger_sha256: str = "0" * 64,
    experiment_id: str = EXPERIMENT_ID,
    room_id: str = ROOM_ID,
    room_kind: str = "paper-ai-inventory",
    policy_sha256: str = POLICY_SHA256,
    flat_block: bool = False,
    action: str = "CLOSE_VIRTUAL",
    producer_receipt_sha256: str = "b" * 64,
    ai_decision_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "room_kind": room_kind,
        "dedicated_root": dedicated_root,
        "room_id": room_id,
        "experiment_id": experiment_id,
        "session_contract_sha256": "1" * 64,
        "candidate_id": lifecycle["candidate_id"],
        "candidate_sha256": lifecycle["candidate_sha256"],
        "policy_id": POLICY_ID,
        "policy_sha256": policy_sha256,
        "spec_id": lifecycle["spec_id"],
        "spec_sha256": lifecycle["spec_sha256"],
        "paper_eligible_event_sha256": lifecycle["paper_eligible_event_sha256"],
        "candidate_lifecycle_ledger_tip_sha256": lifecycle[
            "candidate_lifecycle_ledger_tip_sha256"
        ],
        "ai_decision_binding": ai_decision_binding
        or {
            "producer_id": "codex-dojo-single-reader-v1",
            "model_id": "gpt-5.6-sol",
            "request_sha256": "0" * 64,
            "response_sha256": "f" * 64,
            "evidence_packet_sha256": "a" * 64,
            "producer_receipt_sha256": producer_receipt_sha256,
            "produced_at_utc": QUOTE_AT,
            "observed_at_utc": QUOTE_AT,
        },
        "admission_binding": (
            _admission_binding(
                str(
                    (ai_decision_binding or {"evidence_packet_sha256": "a" * 64})[
                        "evidence_packet_sha256"
                    ]
                )
            )
            if action == "ALLOW_NEW_VIRTUAL"
            else None
        ),
        "ledger_sha256": ledger_sha256,
        "state_sha256": "5" * 64,
        "snapshot_sha256": "6" * 64,
        "position": {
            "position_id": "FLAT:USD_JPY" if flat_block else "T000001",
            "pair": "USD_JPY",
            "side": "FLAT" if flat_block else "LONG",
            "units": 0.0 if flat_block else 100.0,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": ENTRY_CONTEXT_SHA256,
            "sha256": "9" * 64,
        },
        "quote": {
            "pair": "USD_JPY",
            "bid": 163.1,
            "ask": 163.11,
            "observed_at_utc": QUOTE_AT,
            "sha256": "a" * 64,
        },
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }


def _append_decision(
    path: Path,
    lifecycle: dict[str, object],
    *,
    action: str = "CLOSE_VIRTUAL",
    virtual_units: float | None = 100.0,
    ledger_sha256: str = "0" * 64,
    experiment_id: str = EXPERIMENT_ID,
    room_id: str = ROOM_ID,
    policy_sha256: str = POLICY_SHA256,
    recorded_second: int = 4,
    flat_block: bool = False,
    quote_sha256: str = "a" * 64,
    producer_receipt_sha256: str = "b" * 64,
    ai_decision_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    bound_ai = ai_decision_binding or {
        "producer_id": "codex-dojo-single-reader-v1",
        "model_id": "gpt-5.6-sol",
        "request_sha256": "0" * 64,
        "response_sha256": "f" * 64,
        "evidence_packet_sha256": "a" * 64,
        "producer_receipt_sha256": producer_receipt_sha256,
        "produced_at_utc": QUOTE_AT,
        "observed_at_utc": QUOTE_AT,
    }
    decision_cutoff = str(bound_ai["produced_at_utc"])
    cutoff_datetime = datetime.fromisoformat(decision_cutoff.replace("Z", "+00:00"))
    decision_expiry = (
        (cutoff_datetime + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
    )
    proposal: dict[str, object] = {
        "contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "cutoff_at_utc": decision_cutoff,
        "expires_at_utc": decision_expiry,
        "action": action,
        "virtual_units": virtual_units,
        "confidence": 0.8,
        "admission_binding": (
            _admission_binding(str(bound_ai["evidence_packet_sha256"]))
            if action == "ALLOW_NEW_VIRTUAL"
            else None
        ),
        "reason_code": "THESIS_INVALIDATED",
        "reason": "Prospective paper decision bound to immutable evidence.",
        "session_binding": {
            "experiment_id": experiment_id,
            "room_id": room_id,
            "session_contract_sha256": "1" * 64,
            "observed_at_utc": "2026-07-22T12:00:00Z",
        },
        "candidate_binding": {
            "candidate_id": lifecycle["candidate_id"],
            "candidate_sha256": lifecycle["candidate_sha256"],
            "observed_at_utc": "2026-07-22T12:00:00Z",
        },
        "policy_binding": {
            "policy_id": POLICY_ID,
            "policy_sha256": policy_sha256,
            "observed_at_utc": "2026-07-22T12:00:00Z",
        },
        "spec_binding": {
            "spec_id": lifecycle["spec_id"],
            "spec_sha256": lifecycle["spec_sha256"],
            "observed_at_utc": "2026-07-22T12:00:00Z",
        },
        "lifecycle_binding": {
            "paper_eligible_event_sha256": lifecycle["paper_eligible_event_sha256"],
            "candidate_lifecycle_ledger_tip_sha256": lifecycle[
                "candidate_lifecycle_ledger_tip_sha256"
            ],
            "observed_at_utc": "2026-07-22T12:00:05Z",
        },
        "ai_decision_binding": bound_ai,
        "ledger_binding": {
            "sha256": ledger_sha256,
            "observed_at_utc": QUOTE_AT,
        },
        "state_binding": {
            "sha256": "5" * 64,
            "observed_at_utc": QUOTE_AT,
        },
        "snapshot_binding": {
            "sha256": "6" * 64,
            "observed_at_utc": QUOTE_AT,
        },
        "position_binding": {
            "position_id": "FLAT:USD_JPY" if flat_block else "T000001",
            "pair": "USD_JPY",
            "side": "FLAT" if flat_block else "LONG",
            "units": 0.0 if flat_block else 100.0,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": ENTRY_CONTEXT_SHA256,
            "sha256": "9" * 64,
            "observed_at_utc": QUOTE_AT,
        },
        "quote_binding": {
            "pair": "USD_JPY",
            "bid": 163.1,
            "ask": 163.11,
            "sha256": quote_sha256,
            "observed_at_utc": QUOTE_AT,
        },
        "source_watermarks": [
            {
                "source_id": "candles:USD_JPY:M1",
                "sha256": "b" * 64,
                "watermark_at_utc": QUOTE_AT,
                "max_age_seconds": 90,
            }
        ],
        "max_dynamic_evidence_age_seconds": 90,
        "max_record_lag_seconds": 90,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    recorded_at = datetime(2026, 7, 23, 12, 0, recorded_second, tzinfo=UTC)
    with patch(
        "quant_rabbit.dojo_ai_inventory._utc_now",
        return_value=recorded_at,
    ):
        return append_inventory_decision(path, proposal).record


def _harness(
    root: Path,
    *,
    action: str = "CLOSE_VIRTUAL",
    virtual_units: float | None = 100.0,
    experiment_id: str = EXPERIMENT_ID,
    room_id: str = ROOM_ID,
    room_kind: str = "paper-ai-inventory",
    paper_eligible: bool = True,
    broker_units: float = 100.0,
    decision_policy_sha256: str = POLICY_SHA256,
    flat_block: bool = False,
) -> dict[str, object]:
    lifecycle = _build_candidate_lifecycle(
        root,
        experiment_id=EXPERIMENT_ID,
        paper_eligible=paper_eligible,
    )
    broker, dedicated_root = _broker(
        root,
        experiment_id=experiment_id,
        room_id=room_id,
        units=broker_units,
    )
    if flat_block:
        broker.positions.clear()
    producer_admission_binding = (
        _admission_binding() if action in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"} else None
    )
    producer_receipt_path, proposal, trusted_manifest = _write_producer_receipt(
        root,
        dedicated_root / experiment_id / room_id,
        lifecycle,
        action=action,
        virtual_units=virtual_units,
        confidence=0.8,
        admission_binding=producer_admission_binding,
    )
    ai_decision_binding = proposal["ai_decision_binding"]
    producer_receipt_sha256 = ai_decision_binding["producer_receipt_sha256"]
    decision_path = dedicated_root / experiment_id / room_id / "decisions.jsonl"
    decision = _append_decision(
        decision_path,
        lifecycle,
        action=action,
        virtual_units=virtual_units,
        experiment_id=experiment_id,
        room_id=room_id,
        policy_sha256=decision_policy_sha256,
        flat_block=flat_block,
        producer_receipt_sha256=producer_receipt_sha256,
        ai_decision_binding=ai_decision_binding,
    )
    runtime = _runtime(
        dedicated_root,
        lifecycle,
        experiment_id=experiment_id,
        room_id=room_id,
        room_kind=room_kind,
        policy_sha256=decision_policy_sha256,
        flat_block=flat_block,
        action=action,
        producer_receipt_sha256=producer_receipt_sha256,
        ai_decision_binding=ai_decision_binding,
    )
    return {
        "broker": broker,
        "dedicated_root": dedicated_root,
        "decision": decision,
        "decision_path": decision_path,
        "lifecycle": lifecycle,
        "runtime": runtime,
        "candidate_path": lifecycle["candidate_lifecycle_ledger_path"],
        "producer_receipt_path": producer_receipt_path,
        "repository_root": root,
        "trusted_manifests": {
            trusted_manifest["adapter_id"]: trusted_manifest,
        },
        "trusted_replay_workers": lifecycle["trusted_replay_workers"],
    }


def _consume(
    harness: dict[str, object],
    *,
    at: datetime = CONSUME_AT,
    decision: dict[str, object] | None = None,
    decision_path: Path | None = None,
    runtime: dict[str, object] | None = None,
    producer_receipt_path: Path | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    with (
        patch(
            "quant_rabbit.dojo_ai_inventory_consumer._utc_now",
            return_value=at,
        ),
        patch(
            "quant_rabbit.dojo_ai_inventory_producer._TRUSTED_COMMAND_ADAPTERS",
            harness["trusted_manifests"],
        ),
        patch(
            "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
            harness["trusted_replay_workers"],
        ),
    ):
        return consume_inventory_decision(
            decision or harness["decision"],  # type: ignore[arg-type]
            harness["broker"],  # type: ignore[arg-type]
            runtime or harness["runtime"],  # type: ignore[arg-type]
            repository_root=repository_root or harness["repository_root"],  # type: ignore[arg-type]
            decision_ledger_path=decision_path or harness["decision_path"],  # type: ignore[arg-type]
            producer_receipt_path=(
                producer_receipt_path or harness["producer_receipt_path"]  # type: ignore[arg-type]
            ),
        )


def _close_broker(harness: dict[str, object]) -> None:
    harness["broker"]._handle.close()  # type: ignore[union-attr]


class DojoAiInventoryConsumerTest(unittest.TestCase):
    def test_close_requires_real_ledgers_and_is_adjacent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            receipt = _consume(harness)
            broker = harness["broker"]
            rows = [
                json.loads(line) for line in broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "CLOSE",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(rows[1]["prev_sha"], rows[0]["sha"])
            self.assertEqual(rows[2]["payload"]["close_sha256"], rows[1]["sha"])
            self.assertEqual(receipt["applied_receipt_sha256"], rows[2]["sha"])
            self.assertEqual(receipt["confidence"], 0.8)
            self.assertEqual(
                receipt["decision_contract"],
                DOJO_AI_INVENTORY_DECISION_CONTRACT,
            )
            self.assertIs(receipt["virtual_broker_mutation_allowed"], True)
            self.assertIs(receipt["external_broker_mutation_allowed"], False)
            self.assertEqual(
                receipt["ai_producer_receipt_sha256"],
                harness["runtime"]["ai_decision_binding"][  # type: ignore[index]
                    "producer_receipt_sha256"
                ],
            )
            self.assertNotIn("T000001", broker.positions)
            _close_broker(harness)

    def test_producer_receipt_must_exist_verify_and_match_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            missing = (
                harness["producer_receipt_path"].parent  # type: ignore[union-attr]
                / f"{'1' * 64}.json"
            )
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "producer receipt failed durable verification",
            ):
                _consume(harness, producer_receipt_path=missing)
            self.assertEqual(harness["broker"].ledger_path.read_text(), "")  # type: ignore[union-attr]
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            receipt_path = harness["producer_receipt_path"]
            raw = bytearray(receipt_path.read_bytes())  # type: ignore[union-attr]
            raw[-2] = ord(" ")
            receipt_path.write_bytes(bytes(raw))  # type: ignore[union-attr]
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "producer receipt failed durable verification",
            ):
                _consume(harness)
            self.assertEqual(harness["broker"].ledger_path.read_text(), "")  # type: ignore[union-attr]
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            mismatched_path, _, mismatched_manifest = _write_producer_receipt(
                harness["repository_root"],  # type: ignore[arg-type]
                harness["dedicated_root"] / EXPERIMENT_ID / ROOM_ID,  # type: ignore[operator]
                harness["lifecycle"],  # type: ignore[arg-type]
                action="HOLD",
                virtual_units=None,
                confidence=0.8,
                admission_binding=None,
            )
            harness["trusted_manifests"].update(  # type: ignore[union-attr]
                {mismatched_manifest["adapter_id"]: mismatched_manifest}
            )
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "producer receipt/decision mismatch",
            ):
                _consume(harness, producer_receipt_path=mismatched_path)
            self.assertEqual(harness["broker"].ledger_path.read_text(), "")  # type: ignore[union-attr]
            _close_broker(harness)

    def test_reduce_and_non_mutating_actions(self) -> None:
        for action, units, remaining in (
            ("REDUCE_VIRTUAL", 40.0, 60.0),
            ("HOLD", None, 100),
        ):
            with self.subTest(
                action=action
            ), tempfile.TemporaryDirectory() as temporary:
                harness = _harness(Path(temporary), action=action, virtual_units=units)
                receipt = _consume(harness)
                self.assertEqual(
                    harness["broker"].positions["T000001"].units,  # type: ignore[union-attr]
                    remaining,
                )
                self.assertFalse(receipt["block_new"])
                _close_broker(harness)

    def test_fake_broker_and_non_ai_rooms_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            with patch(
                "quant_rabbit.dojo_ai_inventory_consumer._utc_now",
                return_value=CONSUME_AT,
            ), self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "exact VirtualBroker"
            ):
                consume_inventory_decision(
                    harness["decision"],  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    harness["runtime"],  # type: ignore[arg-type]
                    repository_root=harness["repository_root"],  # type: ignore[arg-type]
                    decision_ledger_path=harness["decision_path"],  # type: ignore[arg-type]
                    producer_receipt_path=harness["producer_receipt_path"],  # type: ignore[arg-type]
                )
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), room_kind="paper-control")
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "paper-ai-inventory"
            ):
                _consume(harness)
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                experiment_id="episode-s5-eurusd-diagnostic",
                room_id="eurusd-01-w-fade-base",
            )
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "non-isolated"
            ):
                _consume(harness)
            _close_broker(harness)

    def test_fabricated_absent_or_tampered_decision_ledger_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _harness(root)
            absent = root / "absent-decisions.jsonl"
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "outside.*paper-AI room"
            ):
                _consume(harness, decision_path=absent)

            original = harness["decision"]
            row = json.loads(harness["decision_path"].read_text())  # type: ignore[union-attr]
            row["reason"] = "fabricated in memory and on disk"
            harness["decision_path"].write_text(json.dumps(row) + "\n")  # type: ignore[union-attr]
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "full validation"
            ):
                _consume(harness, decision=original)  # type: ignore[arg-type]
            self.assertEqual(
                harness["broker"].ledger_path.read_text(),
                "",  # type: ignore[union-attr]
            )
            _close_broker(harness)

    def test_decision_ledger_must_be_inside_exact_dedicated_room(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _harness(root)
            outside = root / "paper-ai-inventory-runtime" / "decisions.jsonl"
            outside.parent.mkdir()
            outside.write_bytes(harness["decision_path"].read_bytes())  # type: ignore[union-attr]
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "outside.*paper-AI room"
            ):
                _consume(harness, decision_path=outside)
            self.assertEqual(
                harness["broker"].ledger_path.read_text(),
                "",  # type: ignore[union-attr]
            )
            _close_broker(harness)

    def test_decision_must_be_terminal_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), action="HOLD", virtual_units=None)
            first = harness["decision"]
            _append_decision(
                harness["decision_path"],  # type: ignore[arg-type]
                harness["lifecycle"],  # type: ignore[arg-type]
                action="HOLD",
                virtual_units=None,
                recorded_second=4,
                quote_sha256="c" * 64,
            )
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "terminal ledger row"
            ):
                _consume(harness, decision=first)  # type: ignore[arg-type]
            _close_broker(harness)

    def test_no_paper_eligible_or_policy_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), paper_eligible=False)
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "launch preflight"
            ):
                _consume(harness)
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), decision_policy_sha256="d" * 64)
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "launch preflight/decision"
            ):
                _consume(harness)
            _close_broker(harness)

    def test_new_decision_cannot_bypass_outstanding_room_position_reservation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            broker = harness["broker"]
            with patch.object(
                broker,
                "close_trade",
                side_effect=RuntimeError("crash after reservation"),
            ), self.assertRaisesRegex(RuntimeError, "crash"):
                _consume(harness)

            second = _append_decision(
                harness["decision_path"],  # type: ignore[arg-type]
                harness["lifecycle"],  # type: ignore[arg-type]
                action="CLOSE_VIRTUAL",
                virtual_units=100.0,
                ledger_sha256=broker._prev_sha,
                recorded_second=4,
                producer_receipt_sha256=harness["runtime"]["ai_decision_binding"][
                    "producer_receipt_sha256"
                ],  # type: ignore[index]
                ai_decision_binding=harness["runtime"]["ai_decision_binding"],  # type: ignore[arg-type]
            )
            second_runtime = {
                **harness["runtime"],  # type: ignore[dict-item]
                "ledger_sha256": broker._prev_sha,
            }
            with self.assertRaises(InventoryReservationOutstandingError):
                _consume(
                    harness,
                    decision=second,
                    runtime=second_runtime,
                )
            self.assertEqual(len(broker.ledger_path.read_text().splitlines()), 1)
            _close_broker(harness)

    def test_weekend_cutoff_is_rejected_even_if_consume_time_is_open(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), action="HOLD", virtual_units=None)
            row = copy.deepcopy(harness["decision"])
            row["cutoff_at_utc"] = "2026-07-25T12:00:01Z"
            row["expires_at_utc"] = "2026-07-25T12:01:01Z"
            row["recorded_at_utc"] = "2026-07-25T12:00:02Z"
            row["record_lag_nanoseconds"] = 1_000_000_000
            for key in (
                "ledger_binding",
                "state_binding",
                "snapshot_binding",
                "position_binding",
                "quote_binding",
            ):
                row[key]["observed_at_utc"] = "2026-07-25T12:00:01Z"
            row["quote_binding"]["observed_at_utc"] = "2026-07-25T12:00:01Z"
            row["source_watermarks"][0]["watermark_at_utc"] = "2026-07-25T12:00:01Z"
            row["ai_decision_binding"]["observed_at_utc"] = "2026-07-25T12:00:01Z"
            row["ai_decision_binding"]["produced_at_utc"] = "2026-07-25T12:00:01Z"
            row["decision_identity_sha256"] = inventory_decision_identity_sha256(row)
            row["decision_sha256"] = inventory_decision_sha256(row)
            self.assertEqual(validate_inventory_decision(row), ())
            harness["decision_path"].write_text(  # type: ignore[union-attr]
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError, "decision cutoff"
            ):
                _consume(
                    harness,
                    decision=row,
                    at=datetime(2026, 7, 27, 12, 0, tzinfo=UTC),
                )
            _close_broker(harness)

    def test_exact_applied_decision_returns_same_receipt_without_reapplying(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="HOLD",
                virtual_units=None,
            )
            first = _consume(harness)
            broker = harness["broker"]
            before = broker.ledger_path.read_bytes()
            recovered = _consume(harness)
            self.assertEqual(broker.ledger_path.read_bytes(), before)
            self.assertEqual(
                recovered["applied_receipt_sha256"],
                first["applied_receipt_sha256"],
            )
            _close_broker(harness)

    def test_reserved_only_recovery_executes_close_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            broker = harness["broker"]
            with patch.object(
                broker,
                "close_trade",
                side_effect=RuntimeError("crash after reservation"),
            ), self.assertRaisesRegex(RuntimeError, "crash"):
                _consume(harness)
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                ["AI_INVENTORY_ACTION_RESERVED"],
            )

            recovered = _consume(harness)
            rows = [
                json.loads(line) for line in broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "CLOSE",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(broker.positions, {})
            before = broker.ledger_path.read_bytes()
            again = _consume(harness)
            self.assertEqual(broker.ledger_path.read_bytes(), before)
            self.assertEqual(
                again["applied_receipt_sha256"],
                recovered["applied_receipt_sha256"],
            )
            _close_broker(harness)

    def test_close_durable_recovery_appends_only_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            broker = harness["broker"]
            original_log = broker._log

            def crash_before_applied(event: str, payload: dict[str, object]) -> None:
                if event == "AI_INVENTORY_ACTION_APPLIED":
                    raise RuntimeError("crash after close")
                original_log(event, payload)

            with patch.object(
                broker, "_log", side_effect=crash_before_applied
            ), self.assertRaisesRegex(RuntimeError, "crash after close"):
                _consume(harness)
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                ["AI_INVENTORY_ACTION_RESERVED", "CLOSE"],
            )
            self.assertEqual(broker.positions, {})

            _consume(harness)
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "CLOSE",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(broker.positions, {})
            _close_broker(harness)

    def test_resolved_flat_block_allows_later_distinct_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="BLOCK_NEW",
                virtual_units=None,
                flat_block=True,
            )
            _consume(harness)
            broker = harness["broker"]
            second = _append_decision(
                harness["decision_path"],  # type: ignore[arg-type]
                harness["lifecycle"],  # type: ignore[arg-type]
                action="BLOCK_NEW",
                virtual_units=None,
                ledger_sha256=broker._prev_sha,
                recorded_second=4,
                flat_block=True,
                producer_receipt_sha256=harness["runtime"]["ai_decision_binding"][
                    "producer_receipt_sha256"
                ],  # type: ignore[index]
                ai_decision_binding=harness["runtime"]["ai_decision_binding"],  # type: ignore[arg-type]
            )
            second_runtime = {
                **harness["runtime"],  # type: ignore[dict-item]
                "ledger_sha256": broker._prev_sha,
            }
            receipt = _consume(
                harness,
                decision=second,
                runtime=second_runtime,
            )
            self.assertEqual(receipt["decision_sha256"], second["decision_sha256"])
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "AI_INVENTORY_ACTION_APPLIED",
                    "AI_INVENTORY_ACTION_RESERVED",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            _close_broker(harness)

    def test_fractional_virtual_broker_units_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary), broker_units=100.5)
            with self.assertRaisesRegex(InventoryConsumerIntegrityError, "fractional"):
                _consume(harness)
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="REDUCE_VIRTUAL",
                virtual_units=40.5,
            )
            with self.assertRaisesRegex(InventoryConsumerIntegrityError, "fractional"):
                _consume(harness)
            _close_broker(harness)

    def test_flat_block_new_writes_durable_receipt_without_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="BLOCK_NEW",
                virtual_units=None,
                flat_block=True,
            )
            receipt = _consume(harness)
            broker = harness["broker"]
            rows = [
                json.loads(line) for line in broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(receipt["position_id"], "FLAT:USD_JPY")
            self.assertTrue(receipt["block_new"])
            self.assertEqual(broker.positions, {})
            _close_broker(harness)

    def test_block_new_cancels_only_same_strategy_orders_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="BLOCK_NEW",
                virtual_units=None,
                flat_block=True,
            )
            broker = harness["broker"]
            broker.orders = {
                "O000003": VBOrder(
                    order_id="O000003",
                    pair="USD_JPY",
                    side="LONG",
                    units=100.0,
                    limit_price=143.1,
                    strategy_tag="QR_DOJO_AI_INVENTORY_V1",
                ),
                "O000001": VBOrder(
                    order_id="O000001",
                    pair="USD_JPY",
                    side="SHORT",
                    units=100.0,
                    limit_price=143.3,
                    strategy_tag="QR_DOJO_AI_INVENTORY_V1",
                ),
                "O000002": VBOrder(
                    order_id="O000002",
                    pair="USD_JPY",
                    side="LONG",
                    units=100.0,
                    limit_price=143.0,
                    strategy_tag="ANOTHER_STRATEGY",
                ),
            }

            receipt = _consume(harness)
            rows = [
                json.loads(line) for line in broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "ORDER_CANCEL",
                    "ORDER_CANCEL",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(
                [row["payload"]["order_id"] for row in rows[1:3]],
                ["O000001", "O000003"],
            )
            self.assertEqual(
                receipt["cancelled_order_ids"],
                ["O000001", "O000003"],
            )
            self.assertEqual(set(broker.orders), {"O000002"})
            before = broker.ledger_path.read_bytes()
            recovered = _consume(harness)
            self.assertEqual(broker.ledger_path.read_bytes(), before)
            self.assertEqual(
                recovered["applied_receipt_sha256"],
                receipt["applied_receipt_sha256"],
            )
            self.assertEqual(set(broker.orders), {"O000002"})
            _close_broker(harness)

    def test_block_new_recovers_after_first_durable_order_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="BLOCK_NEW",
                virtual_units=None,
                flat_block=True,
            )
            broker = harness["broker"]
            broker.orders = {
                order_id: VBOrder(
                    order_id=order_id,
                    pair="USD_JPY",
                    side="LONG",
                    units=100.0,
                    limit_price=143.0,
                    strategy_tag="QR_DOJO_AI_INVENTORY_V1",
                )
                for order_id in ("O000001", "O000002")
            }
            checkpoint = broker.snapshot()
            checkpoint_quotes = dict(broker.last_quotes)
            original_cancel = broker.cancel_order
            calls = 0

            def cancel_then_crash(order_id: str) -> None:
                nonlocal calls
                calls += 1
                original_cancel(order_id)
                if calls == 1:
                    raise RuntimeError("crash after first cancellation")

            with patch.object(
                broker,
                "cancel_order",
                side_effect=cancel_then_crash,
            ), self.assertRaisesRegex(RuntimeError, "first cancellation"):
                _consume(harness)
            self.assertEqual(set(broker.orders), {"O000002"})
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                ["AI_INVENTORY_ACTION_RESERVED", "ORDER_CANCEL"],
            )

            broker._handle.close()
            restarted = VirtualBroker(broker.ledger_path, fast_ledger=False)
            restarted.restore(checkpoint, require_ledger_match=False)
            restarted.last_quotes = checkpoint_quotes
            lifecycle = reconcile_inventory_checkpoint_suffix(
                restarted,
                harness["decision"],  # type: ignore[arg-type]
                [
                    json.loads(line)
                    for line in restarted.ledger_path.read_text().splitlines()
                ],
            )
            self.assertEqual(lifecycle["status"], "CANCELS_DURABLE")
            self.assertEqual(set(restarted.orders), {"O000002"})
            harness["broker"] = restarted
            _consume(harness)
            rows = [
                json.loads(line)
                for line in restarted.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "ORDER_CANCEL",
                    "ORDER_CANCEL",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            self.assertEqual(restarted.orders, {})
            _close_broker(harness)

    def test_allow_new_virtual_writes_bound_single_use_permit_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="ALLOW_NEW_VIRTUAL",
                virtual_units=None,
                flat_block=True,
            )
            receipt = _consume(harness)
            admission = harness["runtime"]["admission_binding"]
            self.assertTrue(receipt["allow_new_virtual"])
            self.assertTrue(receipt["single_use_entry_permit"])
            self.assertIs(receipt["entry_proxy_consumed"], False)
            self.assertEqual(receipt["admission_binding"], admission)
            broker = harness["broker"]
            self.assertEqual(broker.positions, {})
            self.assertEqual(
                [
                    json.loads(line)["event"]
                    for line in broker.ledger_path.read_text().splitlines()
                ],
                [
                    "AI_INVENTORY_ACTION_RESERVED",
                    "AI_INVENTORY_ACTION_APPLIED",
                ],
            )
            _close_broker(harness)

    def test_allow_requires_exact_runtime_ai_and_admission_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(
                Path(temporary),
                action="ALLOW_NEW_VIRTUAL",
                virtual_units=None,
                flat_block=True,
            )
            bad_ai = copy.deepcopy(harness["runtime"])
            bad_ai["ai_decision_binding"]["response_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "ai_decision_binding",
            ):
                _consume(harness, runtime=bad_ai)

            bad_admission = copy.deepcopy(harness["runtime"])
            bad_admission["admission_binding"]["entry_signal"][
                "signal_identity_sha256"
            ] = "e" * 64
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "admission_binding",
            ):
                _consume(harness, runtime=bad_admission)
            _close_broker(harness)

    def test_fake_research_root_and_tampered_proof_never_reach_broker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _harness(root)
            fake_root = root / "fake-repository"
            fake_root.mkdir()
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "outside the dedicated paper-AI room",
            ):
                _consume(harness, repository_root=fake_root)
            self.assertEqual(harness["broker"].ledger_path.read_text(), "")  # type: ignore[union-attr]
            _close_broker(harness)

        with tempfile.TemporaryDirectory() as temporary:
            harness = _harness(Path(temporary))
            artifact = (
                harness["repository_root"]  # type: ignore[operator]
                / CANONICAL_RESEARCH_RELATIVE_ROOT
                / "candidates"
                / harness["lifecycle"]["candidate_id"]  # type: ignore[index]
                / "replay"
                / "proof_artifact.json"
            )
            raw = bytearray(artifact.read_bytes())
            raw[-2] = ord(" ")
            artifact.write_bytes(bytes(raw))
            with self.assertRaisesRegex(
                InventoryConsumerIntegrityError,
                "canonical launch preflight",
            ):
                _consume(harness)
            self.assertEqual(harness["broker"].ledger_path.read_text(), "")  # type: ignore[union-attr]
            _close_broker(harness)

    def test_module_has_no_oanda_dependency(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "src"
            / "quant_rabbit"
            / "dojo_ai_inventory_consumer.py"
        ).read_text()
        self.assertNotIn("oanda", source.lower())


if __name__ == "__main__":
    unittest.main()
