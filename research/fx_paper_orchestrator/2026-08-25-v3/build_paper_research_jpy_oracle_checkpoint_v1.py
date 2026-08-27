#!/usr/bin/env python3
"""Build deterministic, future-only evidence for the independent JPY oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import paper_research_jpy_oracle_v1 as oracle
import paper_research_oracle_verifier_v1 as verifier


ORACLE_PATH = "paper_research_jpy_oracle_v1.py"
VERIFIER_PATH = "paper_research_oracle_verifier_v1.py"
CONTRACT_PATH = "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V1.json"
SCHEMA_PATH = "paper_research_jpy_oracle_schema_v1.json"
VERIFIER_SCHEMA_PATH = "paper_research_oracle_verifier_schema_v1.json"
ORACLE_TEST_PATH = "test_paper_research_jpy_oracle_v1.py"
VERIFIER_TEST_PATH = "test_paper_research_oracle_verifier_v1.py"
CHECKPOINT_TEST_PATH = "test_paper_research_jpy_oracle_checkpoint_v1.py"
BUILDER_PATH = "build_paper_research_jpy_oracle_checkpoint_v1.py"
EVIDENCE_ROOT = "evidence/paper_research_jpy_oracle_v1"
AUDIT_PATH = f"{EVIDENCE_ROOT}/oracle_checkpoint_v1.json"
LEDGER_PATH = f"{EVIDENCE_ROOT}/oracle_ledger_v1.jsonl"
MANIFEST_PATH = f"{EVIDENCE_ROOT}/oracle_manifest_v1.json"
RECEIPT_PATH = f"{EVIDENCE_ROOT}/oracle_verifier_receipt_v1.json"
SOURCE_PATH = f"{EVIDENCE_ROOT}/source_bbo_fixture_v1.jsonl"
SOURCE_MANIFEST_PATH = f"{EVIDENCE_ROOT}/source_bbo_manifest_fixture_v1.json"
PROPOSAL_PATH = f"{EVIDENCE_ROOT}/ex_ante_proposal_fixture_v1.json"
EXECUTION_PATH = f"{EVIDENCE_ROOT}/execution_policy_fixture_v1.json"
INVENTORY_PATH = f"{EVIDENCE_ROOT}/inventory_policy_fixture_v1.json"
ACCOUNTING_PATH = f"{EVIDENCE_ROOT}/accounting_policy_fixture_v1.json"
EVALUATION_PATH = f"{EVIDENCE_ROOT}/evaluation_policy_fixture_v1.json"
LEGACY_COVERAGE_PATH = f"{EVIDENCE_ROOT}/legacy_oracle_coverage_v1.json"
START_NS = 1_767_225_600_000_000_000
SEALED_CYCLES = (25, 27, 28, 29, 30, 31, 33, 35, 37, 38, 39, 40, 41)


class EvidenceError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def embedded(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical(unsigned))


def seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = embedded(payload, field)
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(payload) + b"\n")


def artifact(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "sha256": sha256_bytes(data), "size_bytes": len(data)}


def source_rows() -> list[dict[str, Any]]:
    offsets = (0, 1, 2, 302, 360, 361, 362, 662, 900)
    result = []
    for instrument in ("EUR_USD", "USD_JPY"):
        for sequence, seconds in enumerate(offsets, 1):
            if instrument == "EUR_USD":
                bid = 110_000 + sequence * 8
                ask = bid + 12
                scale = 100_000
            else:
                bid = 15_000 + sequence * 3
                ask = bid + 2
                scale = 100
            source = START_NS + seconds * 1_000_000_000
            result.append({
                "schema_version": 1,
                "provider_id": "ORACLE_FIXTURE",
                "instrument": instrument,
                "bid_ticks": bid,
                "ask_ticks": ask,
                "tick_scale": scale,
                "source_ts_ns": source,
                "arrival_ts_ns": source + 100_000_000,
                "provider_event_id": f"{instrument}-{sequence}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(result, key=lambda row: (
        row["source_ts_ns"], row["instrument"], row["sequence"]
    ))


def fixture(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    rows = source_rows()
    source_blob = b"".join(canonical(row) + b"\n" for row in rows)
    source = root / "source.jsonl"
    source.write_bytes(source_blob)
    source_manifest = seal({
        "schema_version": 1,
        "source_bytes_sha256": sha256_bytes(source_blob),
        "source_size_bytes": len(source_blob),
        "event_count": len(rows),
        "first_source_ts_ns": min(row["source_ts_ns"] for row in rows),
        "last_source_ts_ns": max(row["source_ts_ns"] for row in rows),
    }, "manifest_sha256")
    source_manifest_file = root / "source_manifest.json"
    write_json(source_manifest_file, source_manifest)
    proposal = seal({
        "schema_version": 1,
        "candidate_key": "ORACLE-FIXTURE-CANDIDATE",
        "rows": [
            {
                "proposal_ordinal": 1,
                "decision_source_ts_ns": START_NS,
                "decision_arrival_ts_ns": START_NS + 100_000_000,
                "available_at_ns": START_NS + 100_000_000,
                "instrument": "EUR_USD",
                "direction": 1,
                "notional_jpy_micros": 28_000_000_000,
                "max_age_ns": 300_000_000_000,
                "worker_key": "EUR_HIERARCHICAL",
                "action": "ENTER"
            },
            {
                "proposal_ordinal": 2,
                "decision_source_ts_ns": START_NS + 360_000_000_000,
                "decision_arrival_ts_ns": START_NS + 360_100_000_000,
                "available_at_ns": START_NS + 360_100_000_000,
                "instrument": "USD_JPY",
                "direction": -1,
                "notional_jpy_micros": 28_000_000_000,
                "max_age_ns": 300_000_000_000,
                "worker_key": "JPY_COST_TO_MFE",
                "action": "ENTER"
            }
        ]
    }, "proposal_sha256")
    execution = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_EXECUTION_POLICY_V1",
        "arms": {
            "RAW_SIGNAL": {
                "latency_ns": 0, "slippage_ticks_per_side": 0,
                "commission_ppm_per_side": 0, "financing_ppm_per_day": 0,
                "raw_mid": True
            },
            "EXECUTABLE_BASE": {
                "latency_ns": 500000000, "slippage_ticks_per_side": 1,
                "commission_ppm_per_side": 2, "financing_ppm_per_day": 1,
                "raw_mid": False
            },
            "ADVERSE_STRESS": {
                "latency_ns": 1500000000, "slippage_ticks_per_side": 3,
                "commission_ppm_per_side": 6, "financing_ppm_per_day": 3,
                "raw_mid": False
            }
        }
    }, "execution_policy_sha256")
    inventory = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_INVENTORY_POLICY_V1",
        "max_gross_notional_jpy_micros": 200_000_000_000,
        "max_currency_notional_jpy_micros": 200_000_000_000,
        "max_open_positions": 4,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True
    }, "inventory_policy_sha256")
    accounting = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_ACCOUNTING_POLICY_V1",
        "jpy_micros_per_yen": 1000000,
        "base_microunits_per_unit": 1000000,
        "max_conversion_staleness_ns": 400000000000,
        "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
        "asset_conversion_side": "BID",
        "liability_conversion_side": "ASK"
    }, "accounting_policy_sha256")
    evaluation = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_EVALUATION_POLICY_V1",
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 900100000000,
        "initial_equity_jpy_micros": 200_000_000_000,
        "margin_notional_cap_jpy_micros": 200_000_000_000,
        "cvar_tail_bps": 500,
        "holdout_state": "UNOPENED"
    }, "evaluation_policy_sha256")
    payloads = {
        "proposal": proposal,
        "execution_policy": execution,
        "inventory_policy": inventory,
        "accounting_policy": accounting,
        "evaluation_policy": evaluation,
    }
    files = {}
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        write_json(path, payload)
        files[name] = path
    request = {
        "schema_version": 1,
        "input_root": str(root),
        "output_root": str(root),
        "source_blob": artifact(source),
        "source_manifest": artifact(source_manifest_file),
        **{name: artifact(path) for name, path in files.items()},
        "output_directory": "oracle_output",
    }
    return request, {
        "source": str(source),
        "source_manifest": str(source_manifest_file),
        **{name: str(path) for name, path in files.items()},
    }


def legacy_coverage(root: Path) -> dict[str, Any]:
    rows = []
    missing_inputs = [
        "EXACT_EVENT_BBO_BYTES",
        "EVENT_ARRIVAL_TIMESTAMPS",
        "BASE_MICROUNITS",
        "SIGN_AWARE_CONVERSION_RECEIPTS",
        "MARGIN_GRID"
    ]
    for cycle in range(25, 42):
        seal_path = Path(f"evidence/orchestrator_state_v2/official_seal_v{cycle}.json")
        absolute = root / seal_path
        exists = absolute.is_file()
        rows.append({
            "cycle": f"V{cycle}",
            "legacy_seal_path": str(seal_path) if exists else None,
            "legacy_seal_sha256": sha256_file(absolute) if exists else None,
            "coverage_state": "RETROACTIVE" if exists else "MISSING",
            "missing_independent_oracle_inputs": missing_inputs,
            "official_oracle_pass": False,
            "retroactive_promotion_allowed": False,
        })
    payload = {
        "schema_version": 1,
        "classification": "LEGACY_ORACLE_COVERAGE_SIDECAR_ONLY",
        "cycles": rows,
        "sealed_cycle_count": sum(row["cycle"] in {f"V{x}" for x in SEALED_CYCLES} for row in rows),
        "reconstructable_count": 0,
        "official_oracle_pass_count": 0,
        "legacy_seals_changed": False,
    }
    payload["coverage_sha256"] = embedded(payload, "coverage_sha256")
    return payload


def source_hashes(root: Path) -> dict[str, str]:
    paths = (
        ORACLE_PATH, VERIFIER_PATH, CONTRACT_PATH, SCHEMA_PATH, VERIFIER_SCHEMA_PATH,
        ORACLE_TEST_PATH, VERIFIER_TEST_PATH, CHECKPOINT_TEST_PATH, BUILDER_PATH,
    )
    return {path: sha256_file(root / path) for path in paths}


def compute(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="qr-oracle-v1-") as temporary:
        scratch = Path(temporary)
        request, files = fixture(scratch)
        oracle_result = oracle.execute(request)
        oracle_manifest = oracle_result["manifest"]
        oracle_ledger = Path(oracle_result["ledger_path"]).read_text(encoding="utf-8")
        verifier_request = {
            "schema_version": 1,
            "input_root": str(scratch.resolve()),
            "output_root": str((scratch / "verified").resolve()),
            "oracle_manifest": artifact(Path(oracle_result["manifest_path"])),
            "oracle_ledger": artifact(Path(oracle_result["ledger_path"])),
            "source_blob": request["source_blob"],
            "source_manifest": request["source_manifest"],
            "proposal": request["proposal"],
            "execution_policy": request["execution_policy"],
            "inventory_policy": request["inventory_policy"],
            "accounting_policy": request["accounting_policy"],
            "evaluation_policy": request["evaluation_policy"],
            "receipt_name": "receipt.json",
        }
        Path(verifier_request["output_root"]).mkdir(parents=True)
        receipt = verifier.verify(verifier_request)
        coverage = legacy_coverage(root)
        artifact_texts = {
            SOURCE_PATH: Path(files["source"]).read_text(encoding="utf-8"),
            SOURCE_MANIFEST_PATH: Path(files["source_manifest"]).read_text(encoding="utf-8"),
            PROPOSAL_PATH: Path(files["proposal"]).read_text(encoding="utf-8"),
            EXECUTION_PATH: Path(files["execution_policy"]).read_text(encoding="utf-8"),
            INVENTORY_PATH: Path(files["inventory_policy"]).read_text(encoding="utf-8"),
            ACCOUNTING_PATH: Path(files["accounting_policy"]).read_text(encoding="utf-8"),
            EVALUATION_PATH: Path(files["evaluation_policy"]).read_text(encoding="utf-8"),
            LEDGER_PATH: oracle_ledger,
            MANIFEST_PATH: json.dumps(oracle_manifest, indent=2, sort_keys=True) + "\n",
            RECEIPT_PATH: json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            LEGACY_COVERAGE_PATH: json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        }
        audit = {
            "schema_version": 1,
            "checkpoint_id": "PAPER_RESEARCH_JPY_ORACLE_V1",
            "classification": "FUTURE_ONLY_INDEPENDENT_ECONOMIC_ORACLE",
            "source_artifact_sha256": source_hashes(root),
            "evidence_artifact_sha256": {
                path: sha256_bytes(text.encode()) for path, text in sorted(artifact_texts.items())
            },
            "oracle_root_sha256": oracle_manifest["oracle_root_sha256"],
            "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
            "producer_metrics_used": False,
            "same_signal_ids_all_arms": oracle_manifest["oracle_metrics"]["same_signal_ids_all_arms"],
            "all_proposals_have_all_arm_dispositions": oracle_manifest["oracle_metrics"][
                "all_proposals_have_all_arm_dispositions"
            ],
            "terminal_inventory_mtm_jpy_micros": 0,
            "external_orders": 0,
            "holdout_state": "UNOPENED",
            "official_strategy_run_performed": False,
            "profit_evidence_generated": False,
            "anchor_status": "LOCAL_REPRODUCIBLE",
            "remote_anchor_required_for_external_status": True,
            "legacy_coverage_sha256": coverage["coverage_sha256"],
            "legacy_official_oracle_pass_count": 0,
            "legacy_seals_changed": False,
            "authority": {
                "paper_only": True,
                "live_authority": False,
                "broker_account_access": False,
                "credential_access": False,
                "order_endpoint": False,
                "external_orders": 0,
                "deploy": False,
            },
        }
        audit["audit_sha256"] = embedded(audit, "audit_sha256")
        artifact_texts[AUDIT_PATH] = json.dumps(audit, indent=2, sort_keys=True) + "\n"
        return artifact_texts, audit


def build(root: Path) -> dict[str, Any]:
    texts, payload = compute(root)
    for relative, text in texts.items():
        atomic_text(root / relative, text)
    return payload


def validate(root: Path) -> dict[str, Any]:
    texts, payload = compute(root)
    for relative, expected in texts.items():
        path = root / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise EvidenceError(f"oracle evidence changed: {relative}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = build(args.root) if args.command == "build" else validate(args.root)
    print(json.dumps({
        "checkpoint_id": payload["checkpoint_id"],
        "audit_sha256": payload["audit_sha256"],
        "oracle_root_sha256": payload["oracle_root_sha256"],
        "verifier_receipt_sha256": payload["verifier_receipt_sha256"],
        "anchor_status": payload["anchor_status"],
        "external_orders": payload["external_orders"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
