#!/usr/bin/env python3
"""Restart-safe, zero-authority coordinator for local FX paper evidence cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "commit_push_deploy": False,
}


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_result_hash(payload: dict[str, Any]) -> str:
    without_seal = dict(payload)
    without_seal.pop("result_sha256", None)
    return hashlib.sha256(canonical_bytes(without_seal)).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_within(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ContractError(f"path escapes experiment root: {path}")
    return resolved


def load_registry(root: Path, registry_path: Path) -> dict[str, Any]:
    registry_file = require_within(root, registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ContractError("unsupported registry schema")
    if registry.get("authority") != REQUIRED_AUTHORITY:
        raise ContractError("paper-only authority contract mismatch")
    cycles = registry.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ContractError("cycle registry is empty")
    identifiers = [cycle.get("cycle_id") for cycle in cycles]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ContractError("cycle ids must be present and unique")
    seen: set[str] = set()
    for cycle in cycles:
        dependencies = cycle.get("depends_on", [])
        if any(item not in seen for item in dependencies):
            raise ContractError(f"dependency is missing or not earlier: {cycle['cycle_id']}")
        seen.add(cycle["cycle_id"])
    return registry


def admission_passed(cycle: dict[str, Any], result: dict[str, Any]) -> bool:
    value = result.get(cycle["admission_field"])
    operator = cycle["admission_operator"]
    if operator == "IS_TRUE":
        return value is True
    if operator == "GT_ZERO":
        return isinstance(value, int | float) and not isinstance(value, bool) and value > 0
    raise ContractError(f"unknown admission operator: {operator}")


def check_result_safety(payload: dict[str, Any]) -> None:
    if payload.get("live_authority") is not False:
        raise ContractError("result does not prove live_authority=false")
    if payload.get("external_orders") != 0:
        raise ContractError("result contains external orders")
    if payload.get("terminal_inventory_mtm_hidden") is not False:
        raise ContractError("terminal inventory concealment is not explicitly false")
    if payload.get("terminal_open_inventory", 0) != 0:
        raise ContractError("top-level terminal inventory is nonzero")
    portfolio = payload.get("periods", {}).get("WALK_FORWARD", {})
    for arm, metrics in portfolio.items():
        if isinstance(metrics, dict) and metrics.get("terminal_open_inventory", 0) != 0:
            raise ContractError(f"terminal inventory is nonzero in {arm}")


def audit_cycle(
    root: Path, cycle: dict[str, Any], observed_seals: dict[str, str] | None = None,
) -> dict[str, Any]:
    prereg = require_within(root, root / cycle["preregistration"])
    script = require_within(root, root / cycle["script"])
    result_path = require_within(root, root / cycle["result"])
    for path, field in ((prereg, "preregistration_sha256"), (script, "script_sha256")):
        if not path.is_file():
            raise ContractError(f"missing artifact: {path}")
        actual = sha256_file(path)
        if actual != cycle[field]:
            raise ContractError(f"hash mismatch for {path.name}: {actual}")
    for dependency in cycle.get("dependency_files", []):
        dependency_path = require_within(root, root / dependency["path"])
        if not dependency_path.is_file():
            raise ContractError(f"missing code dependency: {dependency_path}")
        actual = sha256_file(dependency_path)
        if actual != dependency["sha256"]:
            raise ContractError(f"dependency hash mismatch for {dependency_path.name}: {actual}")
    if not result_path.is_file():
        return {"cycle_id": cycle["cycle_id"], "status": "PENDING", "result": str(result_path)}
    file_hash = sha256_file(result_path)
    expected_file_hash = cycle.get("result_file_sha256")
    if expected_file_hash is None and observed_seals is not None:
        expected_file_hash = observed_seals.get(cycle["cycle_id"])
    if expected_file_hash is None:
        raise ContractError(f"completed result has no pinned file seal: {cycle['cycle_id']}")
    if file_hash != expected_file_hash:
        raise ContractError(f"result file hash mismatch: {cycle['cycle_id']}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claimed = result.get("result_sha256")
    actual = embedded_result_hash(result)
    if claimed != actual:
        raise ContractError(f"embedded result seal mismatch: {cycle['cycle_id']}")
    check_result_safety(result)
    passed = admission_passed(cycle, result)
    return {
        "cycle_id": cycle["cycle_id"],
        "status": "DEVELOPMENT_PASS_NOT_FINAL" if passed else "REJECTED_DEVELOPMENT",
        "admission_passed": passed,
        "result": str(result_path),
        "result_file_sha256": file_hash,
        "embedded_result_sha256": actual,
    }


def diagnose_last(root: Path, registry: dict[str, Any], audited: list[dict[str, Any]]) -> dict[str, Any]:
    last = audited[-1]
    if last["status"] == "PENDING":
        return {"reason_code": "PENDING_REGISTERED_CYCLE", "next_cycle_id": last["cycle_id"]}
    cycle = next(item for item in registry["cycles"] if item["cycle_id"] == last["cycle_id"])
    result = json.loads((root / cycle["result"]).read_text(encoding="utf-8"))
    walk = result.get("periods", {}).get("WALK_FORWARD", {})
    walk = walk.get("arms", walk)
    equity = {arm: walk.get(arm, {}).get("equity_multiple") for arm in (
        "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")}
    mean_return = {arm: walk.get(arm, {}).get("mean_return") for arm in (
        "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")}
    proposal_raw_mean = walk.get("raw_diagnostics", {}).get("mean_gross_return")
    raw, base, adverse = equity.values()
    raw_edge = raw > 1 if raw is not None else mean_return["RAW_SIGNAL"] is not None and mean_return["RAW_SIGNAL"] > 0
    base_edge = base > 1 if base is not None else mean_return["EXECUTABLE_BASE"] is not None and mean_return["EXECUTABLE_BASE"] > 0
    adverse_edge = (
        adverse > 1 if adverse is not None
        else mean_return["ADVERSE_STRESS"] is not None and mean_return["ADVERSE_STRESS"] > 0
    )
    if proposal_raw_mean is not None and proposal_raw_mean <= 0:
        reason = "PROPOSAL_RAW_EDGE_ABSENT_DESPITE_PORTFOLIO_PATH"
        change = "do not optimize execution for this entry; test the causal feature as exit-only or replace the signal family"
    elif raw_edge and not base_edge:
        reason = "POSITIVE_RAW_EDGE_COST_DOMINANT"
        change = "lower turnover or raise edge-per-decision without changing cost assumptions"
    elif raw_edge and base_edge and not adverse_edge:
        reason = "POSITIVE_NORMAL_EDGE_ADVERSE_COST_DOMINANT"
        change = "raise edge-per-decision or reduce turnover while retaining the frozen adverse assumptions"
    elif raw_edge is False:
        reason = "RAW_EDGE_ABSENT"
        change = "replace signal family; do not tune execution or leverage"
    elif raw_edge and base_edge and adverse_edge and last["status"] == "REJECTED_DEVELOPMENT":
        reason = "AGGREGATE_EDGE_MONTHLY_INSTABILITY"
        change = "diagnose a causal regime split using tuning-only thresholds; do not increase leverage"
    else:
        reason = "INSUFFICIENT_DIAGNOSTIC_FIELDS"
        change = "register a diagnostic-only cycle"
    return {
        "reason_code": reason,
        "last_cycle_id": last["cycle_id"],
        "walk_forward_equity": equity,
        "walk_forward_mean_return": mean_return,
        "proposal_mean_gross_return": proposal_raw_mean,
        "single_next_change": change,
        "llm_policy_surface": {
            "may_propose": ["new causal feature family", "one preregistered state transition", "worker enable/freeze/unwind mode"],
            "may_not_change": ["fills", "cost arms", "leverage after results", "hard guards", "opened holdout labels"],
        },
    }


@dataclass
class Lock:
    path: Path
    descriptor: int | None = None

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ContractError(f"orchestrator lock already exists: {self.path}") from error
        os.write(self.descriptor, f"{os.getpid()}\n".encode())
        os.fsync(self.descriptor)
        return self

    def __exit__(self, *_: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def reconcile(root: Path, registry: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    state_root = require_within(root, state_dir)
    with Lock(state_root / "orchestrator.lock"):
        seals_path = state_root / "artifact_seals.json"
        observed_seals = json.loads(seals_path.read_text()) if seals_path.exists() else {}
        audited = [audit_cycle(root, cycle, observed_seals) for cycle in registry["cycles"]]
        first_pending = next((item for item in audited if item["status"] == "PENDING"), None)
        if first_pending and any(item["status"] != "PENDING" for item in audited[audited.index(first_pending) + 1:]):
            raise ContractError("completed cycle appears after a pending dependency")
        snapshot = {
            "schema_version": 1,
            "registry_id": registry["registry_id"],
            "registry_sha256": hashlib.sha256(canonical_bytes(registry)).hexdigest(),
            "authority": REQUIRED_AUTHORITY,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
            "cycles": audited,
            "system_status": "READY_FOR_NEXT_PREREGISTRATION" if first_pending is None else "READY_TO_EXECUTE_PENDING",
            "profit_proven": False,
            "final_admission_proven": False,
            "next_work_order": diagnose_last(root, registry, audited),
        }
        snapshot["snapshot_sha256"] = hashlib.sha256(canonical_bytes(snapshot)).hexdigest()
        atomic_json(state_root / "current_state.json", snapshot)
        append_journal(state_root / "journal.jsonl", {
            "event": "RECONCILED",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "registry_sha256": snapshot["registry_sha256"],
            "time": snapshot["reconciled_at"],
        })
        return snapshot


def execute_next(root: Path, registry: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    """Execute exactly one registered pending local replay without a shell.

    The first generated result is content-addressed into artifact_seals.json.
    Subsequent reconciliation therefore rejects any byte-level result mutation.
    """
    state_root = require_within(root, state_dir)
    with Lock(state_root / "orchestrator.lock"):
        seals_path = state_root / "artifact_seals.json"
        observed_seals = json.loads(seals_path.read_text()) if seals_path.exists() else {}
        audits = []
        pending = None
        for cycle in registry["cycles"]:
            audited = audit_cycle(root, cycle, observed_seals)
            audits.append(audited)
            if audited["status"] == "PENDING":
                pending = cycle
                break
        if pending is None:
            raise ContractError("no pending registered cycle")
        if any(item["status"] == "PENDING" for item in audits[:-1]):
            raise ContractError("pending dependency prevents execution")
        if pending.get("execution_class") != "LOCAL_REPLAY_ONLY":
            raise ContractError("pending cycle is not authorized for local replay")
        arguments = pending.get("arguments")
        if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
            raise ContractError("pending cycle arguments must be a frozen string list")
        script = require_within(root, root / pending["script"])
        if sha256_file(script) != pending["script_sha256"]:
            raise ContractError("pending script hash mismatch")
        prereg = require_within(root, root / pending["preregistration"])
        if sha256_file(prereg) != pending["preregistration_sha256"]:
            raise ContractError("pending preregistration hash mismatch")
        result_path = require_within(root, root / pending["result"])
        if result_path.exists():
            raise ContractError("unsealed pending result already exists")
        if "--output-root" not in arguments:
            raise ContractError("local replay must declare an output root")
        output_value = Path(arguments[arguments.index("--output-root") + 1])
        output_path = output_value if output_value.is_absolute() else root / output_value
        require_within(root, output_path)
        event_time = datetime.now(timezone.utc).isoformat()
        append_journal(state_root / "journal.jsonl", {
            "event": "CYCLE_STARTED", "cycle_id": pending["cycle_id"], "time": event_time,
            "script_sha256": pending["script_sha256"], "preregistration_sha256": pending["preregistration_sha256"],
        })
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TZ": "UTC",
            "PYTHONHASHSEED": "0",
        }
        completed = subprocess.run(
            [sys.executable, str(script), *arguments], cwd=root, env=clean_env,
            capture_output=True, text=True, timeout=1800, check=False,
        )
        if completed.returncode != 0:
            append_journal(state_root / "journal.jsonl", {
                "event": "CYCLE_FAILED", "cycle_id": pending["cycle_id"],
                "returncode": completed.returncode, "time": datetime.now(timezone.utc).isoformat(),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            })
            raise ContractError(f"local replay exited {completed.returncode}")
        if not result_path.is_file():
            raise ContractError("local replay did not create its declared result")
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        if result_payload.get("result_sha256") != embedded_result_hash(result_payload):
            raise ContractError("generated result embedded seal mismatch")
        check_result_safety(result_payload)
        result_file_hash = sha256_file(result_path)
        observed_seals[pending["cycle_id"]] = result_file_hash
        atomic_json(seals_path, observed_seals)
        append_journal(state_root / "journal.jsonl", {
            "event": "CYCLE_COMPLETED", "cycle_id": pending["cycle_id"],
            "result_file_sha256": result_file_hash,
            "embedded_result_sha256": result_payload["result_sha256"],
            "time": datetime.now(timezone.utc).isoformat(),
        })
    return reconcile(root, registry, state_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--registry", type=Path, default=Path("PAPER_RESEARCH_CYCLE_REGISTRY_V1.json"))
    parser.add_argument("--state-dir", type=Path, default=Path("evidence/orchestrator_state_v1"))
    parser.add_argument("command", choices=["audit", "reconcile", "execute-next"])
    args = parser.parse_args()
    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    state_dir = args.state_dir if args.state_dir.is_absolute() else root / args.state_dir
    try:
        registry = load_registry(root, registry_path)
        if args.command == "audit":
            payload = {"cycles": [audit_cycle(root, cycle) for cycle in registry["cycles"]]}
        elif args.command == "reconcile":
            payload = reconcile(root, registry, state_dir)
        else:
            payload = execute_next(root, registry, state_dir)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
