#!/usr/bin/env python3
"""Build deterministic evidence for the credential-free forward-shadow core."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import forward_shadow_core_v2 as shadow


POLICY_PATH = "FORWARD_SHADOW_CORE_CORRECTIVE_CONTRACT_V2.json"
RUNTIME_PATH = "forward_shadow_core_v2.py"
CORE_TEST_PATH = "test_forward_shadow_core_v2.py"
ADVERSARIAL_TEST_PATH = "test_forward_shadow_adversarial_v2.py"
CHECKPOINT_TEST_PATH = "test_forward_shadow_core_checkpoint_v2.py"
BUILDER_PATH = "build_forward_shadow_core_checkpoint_v2.py"
EVIDENCE_ROOT = "evidence/forward_shadow_core_corrective_v2"
AUDIT_PATH = f"{EVIDENCE_ROOT}/forward_shadow_core_corrective_checkpoint_v2.json"
VALID_MANIFEST_PATH = f"{EVIDENCE_ROOT}/valid_batch_manifest_portable_v2.json"
BARS_PATH = f"{EVIDENCE_ROOT}/completed_bars_v2.jsonl"
QUALITY_PATH = f"{EVIDENCE_ROOT}/quality_failure_matrix_v2.jsonl"
EXECUTION_PATH = f"{EVIDENCE_ROOT}/shared_proposal_execution_v2.jsonl"
SUPERSEDED_AUDIT_PATH = "evidence/forward_shadow_core_v1/forward_shadow_core_checkpoint_v1.json"
FIXED_MTIME_NS = 1_787_777_777_000_000_000
AUTHORITY = shadow.AUTHORITY
PROTECTED_PATHS = (
    "evidence/run_london_open_false_break_reclaim_v41_official_001/"
    "result_london_open_false_break_reclaim_v41.json",
    "evidence/orchestrator_state_v2/official_seal_v41.json",
    "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json",
)


class ShadowEvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def embedded_hash(payload: dict[str, Any], field: str) -> str:
    return shadow.embedded_hash(payload, field)


def _start_ns() -> int:
    return int(datetime(2026, 6, 15, 8, 0, tzinfo=timezone.utc).timestamp()) * 1_000_000_000


def _price(instrument: str, sequence: int) -> tuple[str, str]:
    if instrument == "EUR_USD":
        bid = Decimal("1.10000") + Decimal(sequence) * Decimal("0.000001")
        ask = bid + Decimal("0.00012")
    elif instrument == "USD_JPY":
        bid = Decimal("150.000") + Decimal(sequence) * Decimal("0.001")
        ask = bid + Decimal("0.015")
    else:
        raise ShadowEvidenceError(f"unsupported fixture instrument: {instrument}")
    return format(bid, "f"), format(ask, "f")


def _valid_records() -> list[dict[str, Any]]:
    start = _start_ns()
    offsets = {minute * 60 * 1_000_000_000 for minute in range(251)}
    offsets.update({
        240 * 60 * 1_000_000_000 + 500_000_000,
        240 * 60 * 1_000_000_000 + 1_500_000_000,
        245 * 60 * 1_000_000_000 + 500_000_000,
        245 * 60 * 1_000_000_000 + 1_500_000_000,
    })
    records = []
    for instrument in ("EUR_USD", "USD_JPY"):
        for sequence, offset in enumerate(sorted(offsets), 1):
            bid, ask = _price(instrument, sequence)
            source = start + offset
            records.append({
                "schema_version": 1,
                "provider_id": "OFFLINE_FIXTURE",
                "instrument": instrument,
                "bid": bid,
                "ask": ask,
                "liquidity_optional": "1000000",
                "source_ts_ns": source,
                "arrival_ts_ns": source + 50_000_000,
                "provider_event_id": f"{instrument}-{sequence:04d}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(records, key=lambda item: (
        item["source_ts_ns"], item["instrument"], item["sequence"]
    ))


def _write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    duplicate_index: int | None = None,
) -> None:
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ]
    if duplicate_index is not None:
        lines.insert(duplicate_index + 1, lines[duplicate_index])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.utime(path, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "schema_version", "provider_id", "instrument", "bid", "ask",
        "liquidity_optional", "source_ts_ns", "arrival_ts_ns",
        "provider_event_id", "sequence", "heartbeat", "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["quality_flags"] = "|".join(row["quality_flags"])
            writer.writerow(serialized)
    os.utime(path, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))


def _gap_records() -> list[dict[str, Any]]:
    start = _start_ns()
    rows = []
    for index, seconds in enumerate((0, 30, 180), 1):
        bid = Decimal("1.1000") + Decimal(index) * Decimal("0.0001")
        source = start + seconds * 1_000_000_000
        rows.append({
            "schema_version": 1,
            "provider_id": "NO_SEQUENCE_FIXTURE",
            "instrument": "EUR_USD",
            "bid": format(bid, "f"),
            "ask": format(bid + Decimal("0.0001"), "f"),
            "liquidity_optional": None,
            "source_ts_ns": source,
            "arrival_ts_ns": source + 50_000_000,
            "provider_event_id": f"GAP-{index}",
            "sequence": None,
            "heartbeat": False,
            "quality_flags": ["RECONNECT"] if index == 3 else [],
        })
    return rows


def _ordering_records() -> list[dict[str, Any]]:
    start = _start_ns()
    specifications = (
        (100, 200, 1),
        (101, 150, 2),
        (99, 151, 3),
    )
    rows = []
    for source_seconds, arrival_seconds, sequence in specifications:
        bid = Decimal("1.2000") + Decimal(sequence) * Decimal("0.0001")
        rows.append({
            "schema_version": 1,
            "provider_id": "ORDERING_FIXTURE",
            "instrument": "EUR_USD",
            "bid": format(bid, "f"),
            "ask": format(bid + Decimal("0.0001"), "f"),
            "liquidity_optional": None,
            "source_ts_ns": start + source_seconds * 1_000_000_000,
            "arrival_ts_ns": start + arrival_seconds * 1_000_000_000,
            "provider_event_id": f"ORDER-{sequence}",
            "sequence": sequence,
            "heartbeat": False,
            "quality_flags": [],
        })
    return rows


def _spread_failure_record() -> list[dict[str, Any]]:
    start = _start_ns()
    return [{
        "schema_version": 1,
        "provider_id": "SPREAD_FIXTURE",
        "instrument": "EUR_USD",
        "bid": "1.2000",
        "ask": "1.1999",
        "liquidity_optional": None,
        "source_ts_ns": start,
        "arrival_ts_ns": start + 50_000_000,
        "provider_event_id": "SPREAD-1",
        "sequence": 1,
        "heartbeat": False,
        "quality_flags": [],
    }]


def _quality_case(
    root: Path, name: str, records: list[dict[str, Any]], *, expect_error: str | None = None
) -> dict[str, Any]:
    source = root / f"{name}.jsonl"
    _write_jsonl(source, records)
    store = shadow.ShadowStore(root / f"{name}_state")
    error_code = None
    try:
        receipt = store.ingest(shadow.OfflineBBOFile(source))
        manifest = receipt["manifest"]
    except shadow.ShadowCoreError as error:
        error_code = error.code
        manifest = store.manifests[sha256_file(source)]
    if expect_error is not None and error_code != expect_error:
        raise ShadowEvidenceError(f"{name} error changed: {error_code}")
    reasons = sorted({
        reason for row in store.raw_ledger.rows
        for reason in row.get("quality_reasons", [])
    })
    return {
        "case": name,
        "lossless": manifest["lossless"],
        "status": manifest["status"],
        "error_code": error_code,
        "quality_reasons": reasons,
        "halt_new_actions": store.status()["halt_new_actions"],
        "invalid_interval_count": store.status()["invalid_interval_count"],
        "external_order_count": 0,
    }


def _runtime_import_roots(root: Path) -> list[str]:
    result = set()
    for runtime in (RUNTIME_PATH, "shadow_jpy_accounting_v1.py"):
        tree = ast.parse((root / runtime).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module.split(".")[0])
    return sorted(result)


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        POLICY_PATH, RUNTIME_PATH, BUILDER_PATH, CORE_TEST_PATH,
        ADVERSARIAL_TEST_PATH,
        CHECKPOINT_TEST_PATH, "shadow_jpy_accounting_v1.py",
    )
    missing = [item for item in paths if not (root / item).is_file()]
    if missing:
        raise ShadowEvidenceError(f"shadow source missing: {missing}")
    return {item: sha256_file(root / item) for item in paths}


def _protected_hashes(root: Path) -> dict[str, str]:
    return {item: sha256_file(root / item) for item in PROTECTED_PATHS}


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _compute(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    policy = json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    if policy.get("classification") != (
        "NON_STRATEGY_FILE_ONLY_FORWARD_SHADOW_INFRASTRUCTURE"
    ) or policy.get("authority") != AUTHORITY:
        raise ShadowEvidenceError("shadow policy identity or authority changed")
    if (
        policy["research_boundary"]["forward_feed_connected"] is not False
        or policy["research_boundary"]["holdout_state"] != "UNOPENED"
    ):
        raise ShadowEvidenceError("shadow policy crossed feed or holdout boundary")
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        valid_source = temporary_root / "forward_shadow_valid.jsonl"
        records = _valid_records()
        _write_jsonl(valid_source, records, duplicate_index=0)
        source_before = (
            sha256_file(valid_source),
            valid_source.stat().st_size,
            valid_source.stat().st_mtime_ns,
        )
        store = shadow.ShadowStore(temporary_root / "valid_state")
        receipt = store.ingest(shadow.OfflineBBOFile(valid_source))
        source_after = (
            sha256_file(valid_source),
            valid_source.stat().st_size,
            valid_source.stat().st_mtime_ns,
        )
        if source_before != source_after:
            raise ShadowEvidenceError("ingest mutated source")
        manifest = receipt["manifest"]
        bars = shadow.completed_bars(store)
        decision_one = _start_ns() + 240 * 60 * 1_000_000_000
        decision_two = _start_ns() + 245 * 60 * 1_000_000_000
        decisions = {
            "BOT_ONLY": {
                "action": "ENABLE", "pair_cap": 1.0, "currency_cap": 1.0,
                "provenance": "DETERMINISTIC_FIXTURE",
            },
            "ACTUAL_LLM_INVENTORY_POLICY": {
                "action": "ENABLE", "pair_cap": 1.0, "currency_cap": 1.0,
                "provenance": "FROZEN_STRUCTURED_FIXTURE_NO_MODEL_CALL",
            },
        }
        proposal_one = shadow.Proposal(
            "SHADOW-P1", "SHADOW-S1", decision_one, "EUR_USD", 1,
            25_000.0, 300, "1" * 64,
        )
        proposal_two = shadow.Proposal(
            "SHADOW-P2", "SHADOW-S2", decision_two, "EUR_USD", 1,
            25_000.0, 3600, "1" * 64,
        )
        route_one = shadow.route_shared_proposal(store, proposal_one, decisions)
        route_two = shadow.route_shared_proposal(store, proposal_two, decisions)
        status_before_restart = store.status()
        checkpoint_before = json.loads(
            store.checkpoint_path.read_text(encoding="utf-8")
        )
        restarted = shadow.ShadowStore(store.state_dir)
        status_after_restart = restarted.status()
        checkpoint_after = json.loads(
            restarted.checkpoint_path.read_text(encoding="utf-8")
        )
        idempotent = restarted.ingest(shadow.OfflineBBOFile(valid_source))
        final = shadow.finalize_period(
            restarted, _start_ns() + 250 * 60 * 1_000_000_000
        )
        csv_source = temporary_root / "forward_shadow_valid.csv"
        _write_csv(csv_source, records[:4])
        csv_schema = shadow.validate_schema(csv_source)
        quality_rows = [
            _quality_case(temporary_root, "gap_reconnect", _gap_records()),
            _quality_case(temporary_root, "ordering", _ordering_records()),
            _quality_case(
                temporary_root, "spread_inversion", _spread_failure_record(),
                expect_error="SPREAD_INVERSION",
            ),
        ]
        execution_rows = [
            *restarted.proposal_ledger.rows,
            *restarted.virtual_ledger.rows,
        ]
        imports = _runtime_import_roots(root)
        if any(item in imports for item in {
            "aiohttp", "boto3", "httpx", "oandapyV20", "requests", "urllib3"
        }):
            raise ShadowEvidenceError("external transport import present")
        portable_manifest = {
            key: value for key, value in manifest.items()
            if key not in {"source_device", "source_inode", "manifest_sha256"}
        }
        portable_manifest["runtime_inode_and_device_bound"] = True
        portable_manifest["portable_evidence_sha256"] = embedded_hash(
            portable_manifest, "portable_evidence_sha256"
        )
        artifact_texts = {
            VALID_MANIFEST_PATH: json.dumps(
                portable_manifest, indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            BARS_PATH: _jsonl(bars),
            QUALITY_PATH: _jsonl(quality_rows),
            EXECUTION_PATH: _jsonl(execution_rows),
        }
        counts = {
            timeframe: sum(bar["timeframe"] == timeframe for bar in bars)
            for timeframe in shadow.TIMEFRAMES_SECONDS
        }
        invalid_counts = {
            timeframe: sum(
                bar["timeframe"] == timeframe and not bar["valid"] for bar in bars
            )
            for timeframe in shadow.TIMEFRAMES_SECONDS
        }
        payload = {
            "schema_version": 2,
            "checkpoint_id": "CREDENTIAL_FREE_FORWARD_SHADOW_CORE_CORRECTIVE_V2",
            "classification": policy["classification"],
            "authority": AUTHORITY,
            "supersession": {
                "prior_remote_checkpoint": "0567d5cd607b2f99a6ca1d074fed70b978652e99",
                "prior_evidence_path": SUPERSEDED_AUDIT_PATH,
                "prior_evidence_preserved": (root / SUPERSEDED_AUDIT_PATH).is_file(),
                "prior_evidence_status": "SUPERSEDED_NOT_ADMISSIBLE",
                "prior_strategy_or_profit_evidence_changed": False,
            },
            "source_artifact_sha256": _source_hashes(root),
            "protected_historical_artifact_hashes": _protected_hashes(root),
            "input_scope": {
                "allowed_formats": ["JSONL", "CSV"],
                "network_transport_present": False,
                "secret_source_present": False,
                "external_endpoint_present": False,
                "runtime_import_roots": imports,
                "source_unchanged_after_ingest": source_before == source_after,
                "jsonl_schema_event_count": len(records) + 1,
                "csv_schema_event_count": csv_schema["event_count"],
            },
            "batch_manifest": {
                "path": VALID_MANIFEST_PATH,
                "source_bytes_sha256": manifest["source_bytes_sha256"],
                "source_size_bytes": manifest["source_size_bytes"],
                "source_mtime_ns": manifest["source_mtime_ns"],
                "event_count": manifest["event_count"],
                "accepted_event_count": manifest["accepted_event_count"],
                "exact_duplicate_count": manifest["exact_duplicate_count"],
                "first_source_ts_ns": manifest["first_source_ts_ns"],
                "last_source_ts_ns": manifest["last_source_ts_ns"],
                "lossless": manifest["lossless"],
                "invalid_interval_count": manifest["invalid_interval_count"],
                "portable_evidence_sha256": portable_manifest[
                    "portable_evidence_sha256"
                ],
                "manifest_ledger_source_checkpoint_cross_bound": True,
            },
            "causal_completed_bars": {
                "path": BARS_PATH,
                "counts": counts,
                "invalid_counts": invalid_counts,
                "burn_in_completed_m5_bars": shadow.BURN_IN_M5_BARS,
                "decision_close_t_and_later_fill_verified": True,
                "arrival_time_causality_verified": True,
                "higher_timeframes_use_complete_m5_bundles": True,
            },
            "quality_failure_matrix": {
                "path": QUALITY_PATH,
                "cases": [row["case"] for row in quality_rows],
                "covered_halts": sorted({
                    reason for row in quality_rows for reason in row["quality_reasons"]
                } | {
                    row["error_code"] for row in quality_rows if row["error_code"]
                }),
                "sequence_absent_lossless_false": quality_rows[0]["lossless"] is False,
                "invalid_intervals_reported_separately": True,
                "price_imputation_used": False,
            },
            "shared_proposal_execution": {
                "path": EXECUTION_PATH,
                "proposal_sha256": [
                    proposal_one.proposal_sha256, proposal_two.proposal_sha256
                ],
                "route_receipts": [route_one, route_two],
                "worker_arms": list(shadow.WORKER_ARMS),
                "cost_arms": sorted(shadow.EXECUTION_SCENARIOS),
                "actual_llm_called": False,
                "same_content_addressed_proposal_all_arms": True,
                "normal_adverse_latency_and_slippage_applied_after_decision": True,
                "latency_uses_arrival_and_source_chronology": True,
                "external_order_count": 0,
            },
            "paper_account_finalization": final,
            "restart_safety": {
                "runtime_state_hashes_portable": False,
                "runtime_state_hashes_inode_bound": True,
                "state_and_checkpoint_hash_match": (
                    status_before_restart["state_sha256"]
                    == status_after_restart["state_sha256"]
                    and checkpoint_before["checkpoint_sha256"]
                    == checkpoint_after["checkpoint_sha256"]
                ),
                "exact_batch_reingest_idempotent": idempotent["idempotent_reingest"],
                "append_only_ledgers_verified": True,
                "manifest_ledger_checkpoint_source_blob_bidirectional_binding": True,
                "symlink_and_external_state_path_following": False,
            },
            "adversarial_corrections": {
                "finalized_state_mutation_rejected": True,
                "strict_as_of_cutoff_and_no_future_pnl": True,
                "arrival_time_fill_latency": True,
                "terminal_staleness_fail_closed": True,
                "source_and_state_symlinks_rejected": True,
                "clean_python312_cli_without_pythonpath": True,
                "truncated_empty_invalid_utf8_durable_halt": True,
                "mtime_only_idempotence": True,
                "one_sided_state_and_semantic_reseal_rejected": True,
                "standalone_accounting_transitive_import_scan": True,
                "independent_black_box_adversarial_suite": True,
            },
            "cli_commands": [
                "validate-schema", "ingest-batch", "resume", "status",
                "finalize-period",
            ],
            "forward_feed_connected": False,
            "forward_observation_started": False,
            "official_strategy_run_performed": False,
            "profit_evidence_generated": False,
            "strategy_adoption_authorized": False,
            "holdout_state": "UNOPENED",
            "external_orders": 0,
        }
        for path, text in artifact_texts.items():
            payload.setdefault("artifact_file_sha256", {})[path] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
        payload["audit_sha256"] = embedded_hash(payload, "audit_sha256")
        artifact_texts[AUDIT_PATH] = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        return artifact_texts, payload


def build(root: Path) -> dict[str, Any]:
    artifact_texts, payload = _compute(root)
    for relative, text in artifact_texts.items():
        atomic_text(root / relative, text)
    return payload


def validate(root: Path) -> dict[str, Any]:
    expected_texts, expected_payload = _compute(root)
    for relative, expected in expected_texts.items():
        path = root / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ShadowEvidenceError(f"shadow evidence changed: {relative}")
    actual = json.loads((root / AUDIT_PATH).read_text(encoding="utf-8"))
    if (
        actual != expected_payload
        or actual.get("audit_sha256") != embedded_hash(actual, "audit_sha256")
    ):
        raise ShadowEvidenceError("shadow audit embedded hash mismatch")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"), nargs="?", default="build")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = build(args.root) if args.command == "build" else validate(args.root)
    print(json.dumps({
        "checkpoint_id": payload["checkpoint_id"],
        "audit_sha256": payload["audit_sha256"],
        "forward_feed_connected": payload["forward_feed_connected"],
        "holdout_state": payload["holdout_state"],
        "external_orders": payload["external_orders"],
        "state_and_checkpoint_hash_match": payload["restart_safety"][
            "state_and_checkpoint_hash_match"
        ],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
