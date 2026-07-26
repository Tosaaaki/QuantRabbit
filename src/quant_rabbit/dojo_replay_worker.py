"""Trusted exact-bid/ask replay worker for future paper-AI inventory candidates.

The worker accepts no caller-supplied metrics.  It verifies one preregistered
candidate/job, runs the repository replay engine over the sealed source
manifests, independently reconstructs every arm from broker ledgers, seals the
proof artifact, and signs a worker receipt with an external Ed25519 key.

This module has no live broker client and cannot launch a paper room.  Its only
durable lifecycle mutation is a content-addressed ``REPLAY_PASSED`` append
after all independent proof gates pass.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import signal
import stat
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_autonomous_improvement import append_candidate_event
from quant_rabbit.dojo_replay_gates import (
    PROOF_MANIFEST_CONTRACT,
    canonical_proof_manifest_sha256,
    evaluate_inventory_release_proof_ladder,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANONICAL_REPLAY_RELATIVE_ROOT,
    CANONICAL_RESEARCH_RELATIVE_ROOT,
    PROOF_ARTIFACT_CONTRACT,
    REPLAY_OUTPUT_MANIFEST_CONTRACT,
    WINDOWS,
    _candidate_spec,
    _canonical_bytes,
    _job_manifest,
    _json_bytes,
    _raw_sha256,
    _read_canonical_file,
    _source_manifests,
    _validate_git_head,
    _validate_source_capture_manifest,
    _validate_source_file_bytes,
    canonical_proof_artifact_bytes,
    seal_proof_artifact_exclusive,
)
from quant_rabbit.dojo_replay_worker_receipt import (
    REPLAY_WORKER_RECEIPT_CONTRACT,
    validate_replay_worker_config,
)


WORKER_RESULT_CONTRACT = "QR_DOJO_TRUSTED_REPLAY_WORKER_RESULT_V1"
RUNNER_SCRIPT_RELATIVE_PATH = Path("scripts/run-dojo-inventory-release-replay.py")
SOURCE_MANIFEST_DIRECTORY = "source_manifests"
WORKER_RUN_DIRECTORY = "worker_runs"
WORKER_RECEIPT_NAME = "worker_receipt.json"
WORKER_FAILURE_DIRECTORY = "failure_receipts"
POLICY_NAME = "policy.json"
PRIVATE_KEY_MAX_BYTES = 4 * 1024
MAX_BROKER_LEDGER_BYTES = 512 * 1024 * 1024
ZERO_SHA256 = "0" * 64
REPLAY_POLL_SECONDS = 5.0
REPLAY_TERM_GRACE_SECONDS = 10.0
REPLAY_KILL_GRACE_SECONDS = 10.0
WORKER_FAILURE_CONTRACT = "QR_DOJO_REPLAY_WORKER_FAILURE_V1"
SETTLEMENT_EVENTS = frozenset(
    {"CLOSE", "EXIT_TP", "EXIT_SL", "MARGIN_CLOSE", "MARGIN_CLOSEOUT"}
)
FORCED_END_REASONS = frozenset({"END_OF_REPLAY", "REPLAY_END", "FORCED_REPLAY_END"})
POLICIES = ("BASELINE", "CANDIDATE")
COSTS = ("BASE", "STRESS")
INTRABAR_PATHS = ("OHLC", "OLHC")
_POPEN = subprocess.Popen


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_UTC_NOW = _utc_now


class TrustedReplayWorkerError(RuntimeError):
    """The trusted replay job or its source-derived result is invalid."""


class TrustedReplayWorkerRejected(TrustedReplayWorkerError):
    """TRAIN or the independent proof ladder rejected the candidate."""


class TrustedReplayWorkerMarketClosed(TrustedReplayWorkerError):
    """New replay evaluation is disabled while the FX market is closed."""


def run_trusted_replay_worker(
    repository_root: Path,
    *,
    candidate_id: str,
    worker_config_path: Path,
    private_key_path: Path,
) -> dict[str, Any]:
    """Execute and seal exactly one preregistered replay job.

    The replay command is derived entirely from canonical paths and compared
    byte-for-byte with the preregistered job before any replay subprocess
    starts.  No caller-supplied metrics or arbitrary executable argv enter the
    proof.
    """

    _require_market_open(_utc(_UTC_NOW()), "replay worker start")
    root = _repository_root(repository_root)
    candidate_id = _sha(candidate_id, "candidate_id")
    candidate_dir = (
        root / CANONICAL_RESEARCH_RELATIVE_ROOT / "candidates" / candidate_id
    )
    replay_root = candidate_dir / CANONICAL_REPLAY_RELATIVE_ROOT
    spec_path = candidate_dir / "spec.json"
    job_path = replay_root / "job_manifest.json"
    output_path = replay_root / "output_manifest.json"
    policy_path = replay_root / POLICY_NAME
    manifest_root = replay_root / SOURCE_MANIFEST_DIRECTORY

    spec_raw = _read_canonical_file(
        spec_path,
        root=candidate_dir,
        label="candidate spec",
        canonical_required=False,
    )
    candidate, _ = _candidate_spec(spec_raw)
    candidate = {
        **candidate,
        "costs": _candidate_costs(_json_bytes(spec_raw, "candidate spec")),
    }
    if candidate["candidate_id"] != candidate_id:
        raise TrustedReplayWorkerError("candidate directory/spec mismatch")
    _validate_source_capture_manifest(
        root,
        candidate["source_capture_manifest_sha256"],
    )
    job_raw = _read_canonical_file(
        job_path,
        root=replay_root,
        label="job manifest",
    )
    job = _job_manifest(job_raw, candidate=candidate)
    _validate_git_head(root, job)
    _validate_started_job(
        root,
        candidate_id=candidate_id,
        candidate=candidate,
        job=job,
    )

    worker_config = _load_worker_config(root, worker_config_path)
    for field in ("adapter_id", "model_id", "config_sha256", "producer_id"):
        if worker_config[field] != candidate[field]:
            raise TrustedReplayWorkerError(f"worker {field} differs from candidate")
    executable_sha256, executable_stat = _hash_regular_no_follow(
        Path(worker_config["executable_path"]),
        allowed_root=None,
        label="worker executable",
    )
    if executable_sha256 != worker_config["executable_sha256"]:
        raise TrustedReplayWorkerError("worker executable bytes changed")
    if executable_stat.st_mode & 0o111 == 0:
        raise TrustedReplayWorkerError("worker executable is not executable")
    private_key = _load_external_private_key(
        private_key_path,
        repository_root=root,
        expected_public_key_base64=worker_config["ed25519_public_key_base64"],
    )

    policy_raw = _read_canonical_file(
        policy_path,
        root=replay_root,
        label="replay policy",
    )
    if _raw_sha256(policy_raw) != job["policy_sha256"]:
        raise TrustedReplayWorkerError("replay policy bytes digest mismatch")
    policy = _json_bytes(policy_raw, "replay policy")
    _validate_policy(policy, candidate=candidate)

    source_raw: dict[str, bytes] = {}
    source_file_bindings: list[dict[str, str]] = []
    expected_job_files: list[dict[str, str]] = []
    for window in WINDOWS:
        source_path = manifest_root / f"{window}.json"
        raw = _read_canonical_file(
            source_path,
            root=replay_root,
            label=f"{window} source manifest",
        )
        source_raw[window] = raw
        expected_job_files.append(
            {
                "path": str(source_path.relative_to(root)),
                "sha256": _raw_sha256(raw),
            }
        )
        source_file_bindings.extend(
            _validate_source_file_bytes(
                _json_bytes(raw, f"{window} source manifest"),
                repository_root=root,
                window=window,
            )
        )
    source_bindings = _source_manifests(
        source_raw,
        windows=candidate["windows"],
    )
    job_value = _json_bytes(job_raw, "job manifest")
    if job_value.get("files") != expected_job_files:
        raise TrustedReplayWorkerError(
            "job files are not the canonical TRAIN/VAL/S5 source manifests"
        )

    output_raw = _read_canonical_file(
        output_path,
        root=replay_root,
        label="output manifest",
    )
    if _raw_sha256(output_raw) != job["output_manifest_sha256"]:
        raise TrustedReplayWorkerError("output manifest bytes digest mismatch")
    _validate_output_manifest(
        _json_bytes(output_raw, "output manifest"),
        candidate=candidate,
        job=job,
        source_bindings=source_bindings,
    )

    run_root = replay_root / WORKER_RUN_DIRECTORY
    runner_script = root / RUNNER_SCRIPT_RELATIVE_PATH
    runner_sha256, _ = _hash_regular_no_follow(
        runner_script,
        allowed_root=root,
        label="inventory-release replay runner",
    )
    command = [
        worker_config["executable_path"],
        str(runner_script),
        "--spec",
        str(spec_path),
        "--output-root",
        str(run_root),
        "--source-manifest-dir",
        str(manifest_root),
    ]
    if job["argv"] != command or job["argv_sha256"] != _canonical_sha256(command):
        raise TrustedReplayWorkerError(
            "derived replay command differs from preregistration"
        )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root / "src"),
    }
    returncode = _run_replay_subprocess(
        command,
        repository_root=root,
        replay_root=replay_root,
        research_root=root / CANONICAL_RESEARCH_RELATIVE_ROOT,
        candidate=candidate,
        job=job,
        env=env,
    )
    if returncode != 0:
        raise TrustedReplayWorkerError(f"replay engine failed with status {returncode}")

    try:
        train_arms = _read_arm_matrix(
            run_root,
            windows=("TRAIN",),
            candidate=candidate,
            source_bindings=source_bindings,
        )
        train_selected, train_reasons = _train_select(train_arms)
        if not train_selected:
            raise TrustedReplayWorkerRejected(
                "TRAIN rejected candidate: " + "; ".join(train_reasons)
            )
        arms = train_arms + _read_arm_matrix(
            run_root,
            windows=("VAL", "S5"),
            candidate=candidate,
            source_bindings=source_bindings,
        )
        _require_market_open(_utc(_UTC_NOW()), "pre-proof gate clock")
        proof_manifest = {
            "contract": PROOF_MANIFEST_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": job["policy_sha256"],
            "artifact_manifest_sha256": job["output_manifest_sha256"],
            "windows": candidate["windows"],
            "arms": arms,
        }
        proof_manifest_sha256 = canonical_proof_manifest_sha256(proof_manifest)
        proof_manifest["manifest_sha256"] = proof_manifest_sha256
        gate = evaluate_inventory_release_proof_ladder(
            proof_manifest,
            expected_manifest_sha256=proof_manifest_sha256,
        )
        if gate.get("decision") != "PROOF_ELIGIBLE":
            reasons = [
                str(item.get("message"))
                for item in gate.get("reasons", [])
                if isinstance(item, Mapping)
            ]
            raise TrustedReplayWorkerRejected(
                "independent proof rejected candidate: " + "; ".join(reasons)
            )
        completed_at = _utc(_UTC_NOW())
        _require_market_open(completed_at, "replay worker completion")
    except TrustedReplayWorkerMarketClosed as market_closed:
        detected_at = _utc(_UTC_NOW())
        _record_replay_failure(
            replay_root=replay_root,
            research_root=root / CANONICAL_RESEARCH_RELATIVE_ROOT,
            candidate=candidate,
            job=job,
            detected_at=detected_at,
            reason=str(market_closed),
            termination={
                "term_sent": False,
                "kill_sent": False,
                "returncode": returncode,
            },
        )
        raise
    artifact_raw = canonical_proof_artifact_bytes(
        {
            "contract": PROOF_ARTIFACT_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": job["policy_sha256"],
            "job_manifest_sha256": job["manifest_sha256"],
            "git_head": job["git_head"],
            "git_head_sha256": job["git_head_sha256"],
            "artifact_manifest_sha256": job["output_manifest_sha256"],
            "windows": candidate["windows"],
            "completed_at_utc": _format_utc(completed_at),
            "arms": arms,
        }
    )
    artifact_path = replay_root / "proof_artifact.json"
    artifact_seal = seal_proof_artifact_exclusive(artifact_path, artifact_raw)
    artifact = _json_bytes(artifact_raw, "proof artifact")
    receipt = _build_receipt(
        repository_root=root,
        replay_root=replay_root,
        worker_config=worker_config,
        executable_stat=executable_stat,
        invocation_argv=command,
        candidate=candidate,
        job=job,
        source_file_bindings=source_file_bindings,
        artifact_path=artifact_path,
        artifact_raw=artifact_raw,
        completed_at=completed_at,
        private_key=private_key,
    )
    receipt_path = replay_root / WORKER_RECEIPT_NAME
    _write_exclusive_fsynced(receipt_path, _canonical_bytes(receipt) + b"\n")

    event_payload = {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": candidate_id,
        "independent_stress_metrics": _independent_stress_summary(arms),
        "proof_artifact_sha256": artifact["artifact_sha256"],
        "proof_artifact_bytes_sha256": _raw_sha256(artifact_raw),
        "proof_manifest_sha256": proof_manifest_sha256,
        "job_manifest_sha256": job["manifest_sha256"],
        "replay_worker_receipt_sha256": receipt["receipt_sha256"],
    }
    event, appended = append_candidate_event(
        root / CANONICAL_RESEARCH_RELATIVE_ROOT,
        event_type="REPLAY_PASSED",
        payload=event_payload,
        recorded_at_utc=completed_at,
    )
    if not appended:
        raise TrustedReplayWorkerError("REPLAY_PASSED was not appended")

    return {
        "contract": WORKER_RESULT_CONTRACT,
        "candidate_id": candidate_id,
        "worker_receipt_sha256": receipt["receipt_sha256"],
        "proof_artifact_sha256": artifact["artifact_sha256"],
        "proof_artifact_bytes_sha256": artifact_seal["bytes_sha256"],
        "proof_manifest_sha256": proof_manifest_sha256,
        "replay_passed_event_sha256": event["event_sha256"],
        "runner_sha256": runner_sha256,
        "runner_command_sha256": _canonical_sha256(command),
        "source_manifest_sha256s": source_bindings,
        "train_selected": True,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "paper_room_launched": False,
    }


def _run_replay_subprocess(
    command: list[str],
    *,
    repository_root: Path,
    replay_root: Path,
    research_root: Path,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    env: Mapping[str, str],
) -> int:
    try:
        process = _POPEN(
            command,
            cwd=repository_root,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise TrustedReplayWorkerError("replay engine could not start") from exc
    while True:
        try:
            returncode = int(process.wait(timeout=REPLAY_POLL_SECONDS))
        except subprocess.TimeoutExpired:
            detected_at = _utc(_UTC_NOW())
            try:
                _require_market_open(detected_at, "replay worker polling clock")
            except TrustedReplayWorkerMarketClosed as market_closed:
                termination = _terminate_replay_process(process)
                _record_replay_failure(
                    replay_root=replay_root,
                    research_root=research_root,
                    candidate=candidate,
                    job=job,
                    detected_at=detected_at,
                    reason=str(market_closed),
                    termination=termination,
                )
                raise
        else:
            detected_at = _utc(_UTC_NOW())
            try:
                _require_market_open(detected_at, "post-replay worker clock")
            except TrustedReplayWorkerMarketClosed as market_closed:
                termination = _terminate_replay_process(process)
                _record_replay_failure(
                    replay_root=replay_root,
                    research_root=research_root,
                    candidate=candidate,
                    job=job,
                    detected_at=detected_at,
                    reason=str(market_closed),
                    termination=termination,
                )
                raise
            return returncode


def _terminate_replay_process(
    process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    if process.poll() is not None:
        return {
            "term_sent": False,
            "kill_sent": False,
            "returncode": int(process.returncode),
        }
    term_sent = False
    kill_sent = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        term_sent = True
    except ProcessLookupError:
        pass
    try:
        returncode = process.wait(timeout=REPLAY_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        try:
            returncode = process.wait(timeout=REPLAY_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise TrustedReplayWorkerError(
                "replay process group survived bounded TERM/KILL shutdown"
            ) from exc
    return {
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "returncode": int(returncode),
    }


def _record_replay_failure(
    *,
    replay_root: Path,
    research_root: Path,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    detected_at: datetime,
    reason: str,
    termination: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "contract": WORKER_FAILURE_CONTRACT,
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "job_manifest_sha256": job["manifest_sha256"],
        "policy_sha256": job["policy_sha256"],
        "git_head": job["git_head"],
        "git_head_sha256": job["git_head_sha256"],
        "failure_code": "MARKET_CLOSED",
        "reason": reason,
        "detected_at_utc": _format_utc(detected_at),
        "partial_output_directory": str(
            (replay_root / WORKER_RUN_DIRECTORY).relative_to(research_root)
        ),
        "partial_output_proof_eligible": False,
        "proof_artifact_written": False,
        "worker_receipt_written": False,
        "termination": dict(termination),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "failure_sha256": _canonical_sha256(body)}
    raw = _canonical_bytes(sealed) + b"\n"
    raw_sha256 = _raw_sha256(raw)
    failure_root = replay_root / WORKER_FAILURE_DIRECTORY
    failure_root.mkdir(parents=True, exist_ok=True)
    path = failure_root / f"{raw_sha256}.json"
    _write_exclusive_fsynced(path, raw)
    event, appended = append_candidate_event(
        research_root,
        event_type="REPLAY_FAILED",
        payload={
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate["candidate_id"],
            "failure_code": "MEASUREMENT",
            "reason": reason,
            "artifact_sha256": raw_sha256,
        },
        recorded_at_utc=detected_at,
    )
    if not appended:
        raise TrustedReplayWorkerError("REPLAY_FAILED was not appended")
    return {
        "path": str(path),
        "failure_sha256": sealed["failure_sha256"],
        "bytes_sha256": raw_sha256,
        "replay_failed_event_sha256": event["event_sha256"],
    }


def _validate_policy(
    policy: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> None:
    expected = {
        "contract": "QR_DOJO_INVENTORY_RELEASE_POLICY_V1",
        "candidate_id": candidate["candidate_id"],
        "family": "INVENTORY_RELEASE",
        "selection_window": "TRAIN",
        "proof_windows": ["VAL", "S5"],
        "costs": ["BASE", "STRESS"],
        "intrabar_paths": ["OHLC", "OLHC"],
        "end_of_replay_forced_close_benefit": False,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    if dict(policy) != expected:
        raise TrustedReplayWorkerError("replay policy is not the fixed proof ladder")


def _candidate_costs(spec: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    costs = spec.get("costs")
    if not isinstance(costs, Mapping) or set(costs) != set(COSTS):
        raise TrustedReplayWorkerError("candidate costs must be exactly BASE/STRESS")
    expected = {
        "BASE": {
            "slippage_pips_per_fill": 0.0,
            "financing_pips_per_day": 0.0,
        },
        "STRESS": {
            "slippage_pips_per_fill": 0.3,
            "financing_pips_per_day": 0.8,
        },
    }
    if dict(costs) != expected:
        raise TrustedReplayWorkerError("candidate replay costs changed")
    if spec.get("affected_pair") != "USD_JPY":
        raise TrustedReplayWorkerError("candidate replay pair must be USD_JPY")
    if spec.get("intrabar_paths") != list(INTRABAR_PATHS):
        raise TrustedReplayWorkerError("candidate intrabar paths changed")
    return expected


def _validate_started_job(
    root: Path,
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    ledger_path = root / CANONICAL_RESEARCH_RELATIVE_ROOT / "candidate_ledger.jsonl"
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = [
        row
        for row in rows
        if row.get("event_type") in {"REPLAY_STARTED", "REPLAY_RETRY_STARTED"}
        and row.get("payload", {}).get("candidate_id") == candidate_id
    ]
    if not starts:
        raise TrustedReplayWorkerError("candidate has no preregistered replay job")
    lock = starts[-1].get("payload", {}).get("job_lock")
    expected = {
        "job_manifest_sha256": job["manifest_sha256"],
        "git_head_sha256": job["git_head_sha256"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "output_manifest_sha256": job["output_manifest_sha256"],
        "argv": job["argv"],
        "argv_sha256": job["argv_sha256"],
        "adapter_id": candidate["adapter_id"],
        "model_id": candidate["model_id"],
        "config_sha256": candidate["config_sha256"],
        "producer_id": candidate["producer_id"],
        "source_capture_manifest_sha256": candidate["source_capture_manifest_sha256"],
    }
    if not isinstance(lock, Mapping):
        raise TrustedReplayWorkerError("candidate replay job lock is missing")
    for field, value in expected.items():
        if lock.get(field) != value:
            raise TrustedReplayWorkerError(f"candidate replay job {field} changed")


def _validate_output_manifest(
    output: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    source_bindings: Mapping[str, str],
) -> None:
    expected = {
        "contract": REPLAY_OUTPUT_MANIFEST_CONTRACT,
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "git_head": job["git_head"],
        "source_manifest_sha256s": dict(source_bindings),
        "adapter_id": candidate["adapter_id"],
        "model_id": candidate["model_id"],
        "config_sha256": candidate["config_sha256"],
        "producer_id": candidate["producer_id"],
        "source_capture_manifest_sha256": candidate["source_capture_manifest_sha256"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    if dict(output) != expected:
        raise TrustedReplayWorkerError("output manifest binding mismatch")


def _read_arm_matrix(
    run_root: Path,
    *,
    windows: Sequence[str],
    candidate: Mapping[str, Any],
    source_bindings: Mapping[str, str],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for window in windows:
        for policy in POLICIES:
            for cost in COSTS:
                for intrabar in INTRABAR_PATHS:
                    _require_market_open(
                        _utc(_UTC_NOW()),
                        "replay proof arm evaluation clock",
                    )
                    session_dir = (
                        run_root
                        / window.lower()
                        / policy.lower()
                        / cost.lower()
                        / intrabar.lower()
                    )
                    contract = _load_canonical_json(
                        session_dir / "session_contract.json",
                        "arm session contract",
                    )
                    _validate_arm_contract(
                        contract,
                        window=window,
                        policy=policy,
                        cost=cost,
                        intrabar=intrabar,
                        candidate=candidate,
                        source_manifest_sha256=source_bindings[window],
                    )
                    arms.append(
                        {
                            "window": window,
                            "policy": policy,
                            "cost": cost,
                            "intrabar": intrabar,
                            "metrics": _ledger_metrics(session_dir),
                        }
                    )
    return arms


def _validate_arm_contract(
    contract: Mapping[str, Any],
    *,
    window: str,
    policy: str,
    cost: str,
    intrabar: str,
    candidate: Mapping[str, Any],
    source_manifest_sha256: str,
) -> None:
    source = contract.get("source")
    costs = contract.get("costs")
    expected_costs = candidate.get("costs")
    if not isinstance(source, Mapping) or not isinstance(costs, Mapping):
        raise TrustedReplayWorkerError("arm session contract is incomplete")
    if (
        contract.get("candidate_id") != candidate["candidate_id"]
        or contract.get("feed") != "replay"
        or contract.get("pairs") != ["USD_JPY"]
        or source.get("source_manifest_sha256") != source_manifest_sha256
        or source.get("time_from") != candidate["windows"][window]["from_utc"]
        or source.get("time_to") != candidate["windows"][window]["to_utc"]
        or source.get("intrabar") != intrabar
        or not isinstance(expected_costs, Mapping)
        or costs.get("slippage_pips_per_fill")
        != expected_costs[cost]["slippage_pips_per_fill"]
        or costs.get("financing_pips_per_day")
        != expected_costs[cost]["financing_pips_per_day"]
    ):
        raise TrustedReplayWorkerError("arm session contract binding mismatch")
    bot = contract.get("bot")
    if not isinstance(bot, Mapping):
        raise TrustedReplayWorkerError("arm bot contract is missing")
    module = str(bot.get("module") or "")
    if policy == "BASELINE" and not module.endswith("bots/lab_bot.py"):
        raise TrustedReplayWorkerError("baseline arm used the wrong bot")
    if policy == "CANDIDATE" and not module.endswith(
        "bots/inventory_release_candidate.py"
    ):
        raise TrustedReplayWorkerError("candidate arm used the wrong bot")


def _ledger_metrics(session_dir: Path) -> dict[str, Any]:
    ledger_path = session_dir / "ledger.jsonl"
    snapshot = _load_strict_json(
        session_dir / "broker_snapshot.json",
        "broker snapshot",
    )
    rows, terminal_sha256 = _read_validated_broker_ledger(ledger_path)
    if snapshot.get("ledger_sha") != terminal_sha256:
        raise TrustedReplayWorkerError(
            "broker snapshot does not match the terminal ledger digest"
        )
    if rows[-1]["event"] != "SESSION_STOP":
        raise TrustedReplayWorkerError("replay arm has no terminal SESSION_STOP")
    if not isinstance(snapshot.get("positions"), list) or not isinstance(
        snapshot.get("orders"), list
    ):
        raise TrustedReplayWorkerError("broker snapshot inventory is invalid")
    fills: dict[str, dict[str, Any]] = {}
    settlements: list[dict[str, Any]] = []
    settled_trade_ids: set[str] = set()
    margin_events = 0
    ruin_events = 0
    forced_close_count = 0
    forced_close_net = 0.0
    for row in rows:
        event = str(row.get("event") or "")
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        trade_id = payload.get("trade_id")
        if event.startswith("FILL") and trade_id:
            if str(trade_id) in fills:
                raise TrustedReplayWorkerError("duplicate replay fill trade identity")
            fills[str(trade_id)] = row
        if event in SETTLEMENT_EVENTS:
            normalized_trade_id = str(trade_id or "")
            if (
                not normalized_trade_id
                or normalized_trade_id not in fills
                or normalized_trade_id in settled_trade_ids
            ):
                raise TrustedReplayWorkerError(
                    "replay settlement does not bind one preceding fill"
                )
            settled_trade_ids.add(normalized_trade_id)
            value = payload.get("pl_jpy")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TrustedReplayWorkerError("settlement P/L is invalid")
            settlements.append(row)
            if str(payload.get("reason") or "").upper() in FORCED_END_REASONS:
                forced_close_count += 1
                forced_close_net += float(value)
        if event.startswith("MARGIN"):
            margin_events += 1
        if event in {"RUIN", "ACCOUNT_RUIN"}:
            ruin_events += 1

    if set(fills) != settled_trade_ids:
        raise TrustedReplayWorkerError(
            "replay fill/settlement identities do not close exactly"
        )
    pnl = [float(row["payload"]["pl_jpy"]) for row in settlements]
    gross_profit = sum(value for value in pnl if value > 0.0)
    gross_loss = -sum(value for value in pnl if value < 0.0)
    if gross_loss == 0.0 and gross_profit > 0.0:
        raise TrustedReplayWorkerError(
            "infinite profit factor cannot be sealed as canonical JSON"
        )
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    cumulative = 0.0
    high_water = 0.0
    drawdown = 0.0
    daily: dict[str, float] = defaultdict(float)
    active_days: set[str] = set()
    for row, value in zip(settlements, pnl):
        cumulative += value
        high_water = max(high_water, cumulative)
        drawdown = max(drawdown, high_water - cumulative)
        exit_time = _market_event_time(row)
        day_jst = (exit_time + timedelta(hours=9)).date().isoformat()
        daily[day_jst] += value
        active_days.add(day_jst)
        trade_id = str(row["payload"].get("trade_id") or "")
        fill = fills[trade_id]
        if _market_event_time(fill) > exit_time:
            raise TrustedReplayWorkerError("settlement predates its fill")
    net = sum(pnl)
    return {
        "settlements": len(settlements),
        "active_days": len(active_days),
        "net_jpy": net,
        "profit_factor": profit_factor,
        "expectancy_jpy": net / len(settlements) if settlements else 0.0,
        "worst_day_jpy": min(daily.values()) if daily else 0.0,
        "realized_drawdown_jpy": drawdown,
        "margin_events": margin_events,
        "ruin_events": ruin_events,
        "unresolved_positions": len(snapshot.get("positions") or []),
        "unresolved_orders": len(snapshot.get("orders") or []),
        "end_of_replay_forced_close_count": forced_close_count,
        "end_of_replay_forced_close_net_jpy": forced_close_net,
    }


def _read_validated_broker_ledger(
    path: Path,
) -> tuple[list[dict[str, Any]], str]:
    raw, _ = _read_regular_no_follow(
        path,
        allowed_root=path.parent,
        max_bytes=MAX_BROKER_LEDGER_BYTES,
        label="replay broker ledger",
    )
    if not raw.endswith(b"\n"):
        raise TrustedReplayWorkerError("replay broker ledger has a torn final row")
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise TrustedReplayWorkerError(
                f"replay broker ledger has an empty row at line {line_number}"
            )
        row = _json_bytes(line, f"replay broker ledger line {line_number}")
        if set(row) != {"ts_utc", "event", "payload", "prev_sha", "sha"}:
            raise TrustedReplayWorkerError(
                f"replay broker ledger schema is invalid at line {line_number}"
            )
        if (
            not isinstance(row["ts_utc"], str)
            or not isinstance(row["event"], str)
            or not isinstance(row["payload"], Mapping)
            or row["prev_sha"] != previous
        ):
            raise TrustedReplayWorkerError(
                f"replay broker ledger fields are invalid at line {line_number}"
            )
        body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
        if row["sha"] != _canonical_sha256(body):
            raise TrustedReplayWorkerError(
                f"replay broker ledger chain mismatch at line {line_number}"
            )
        previous = _sha(row["sha"], f"ledger line {line_number} sha")
        rows.append(row)
    if not rows:
        raise TrustedReplayWorkerError("replay broker ledger is empty")
    return rows, previous


def _train_select(arms: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    by_key = {
        (row["policy"], row["cost"], row["intrabar"]): row["metrics"] for row in arms
    }
    reasons: list[str] = []
    for cost in COSTS:
        for intrabar in INTRABAR_PATHS:
            baseline = by_key[("BASELINE", cost, intrabar)]
            candidate = by_key[("CANDIDATE", cost, intrabar)]
            if candidate["settlements"] < 30 or candidate["active_days"] < 20:
                reasons.append(f"{cost}/{intrabar}: insufficient TRAIN sample")
            if candidate["net_jpy"] <= baseline["net_jpy"]:
                reasons.append(f"{cost}/{intrabar}: net did not improve")
            if candidate["expectancy_jpy"] <= baseline["expectancy_jpy"]:
                reasons.append(f"{cost}/{intrabar}: expectancy did not improve")
            if (
                candidate["worst_day_jpy"] < baseline["worst_day_jpy"]
                or candidate["realized_drawdown_jpy"]
                > baseline["realized_drawdown_jpy"]
            ):
                reasons.append(f"{cost}/{intrabar}: risk worsened")
            if (
                candidate["margin_events"] > baseline["margin_events"]
                or candidate["ruin_events"] > baseline["ruin_events"]
            ):
                reasons.append(f"{cost}/{intrabar}: margin/ruin worsened")
            if (
                candidate["unresolved_positions"] > baseline["unresolved_positions"]
                or candidate["unresolved_orders"] > baseline["unresolved_orders"]
            ):
                reasons.append(f"{cost}/{intrabar}: unresolved exposure worsened")
            if (
                candidate["end_of_replay_forced_close_count"] != 0
                or candidate["end_of_replay_forced_close_net_jpy"] != 0.0
            ):
                reasons.append(f"{cost}/{intrabar}: forced-close benefit detected")
    return not reasons, reasons


def _independent_stress_summary(
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {
        (row["window"], row["policy"], row["cost"], row["intrabar"]): row["metrics"]
        for row in arms
    }
    independent = [
        (
            window,
            intrabar,
            index[(window, "BASELINE", "STRESS", intrabar)],
            index[(window, "CANDIDATE", "STRESS", intrabar)],
        )
        for window in ("VAL", "S5")
        for intrabar in INTRABAR_PATHS
    ]
    return {
        "pf": min(
            float(candidate["profit_factor"]) for _, _, _, candidate in independent
        ),
        "net": min(float(candidate["net_jpy"]) for _, _, _, candidate in independent),
        "expectancy": min(
            float(candidate["expectancy_jpy"]) for _, _, _, candidate in independent
        ),
        "worst_day_not_worse": all(
            candidate["worst_day_jpy"] >= baseline["worst_day_jpy"]
            for _, _, baseline, candidate in independent
        ),
        "drawdown_not_worse": all(
            candidate["realized_drawdown_jpy"] <= baseline["realized_drawdown_jpy"]
            for _, _, baseline, candidate in independent
        ),
        "margin_ruin_not_worse": all(
            candidate["margin_events"] <= baseline["margin_events"]
            and candidate["ruin_events"] <= baseline["ruin_events"]
            for _, _, baseline, candidate in independent
        ),
        "unresolved_end_exposure": any(
            candidate["unresolved_positions"] != 0
            or candidate["unresolved_orders"] != 0
            for _, _, _, candidate in independent
        ),
    }


def _build_receipt(
    *,
    repository_root: Path,
    replay_root: Path,
    worker_config: Mapping[str, Any],
    executable_stat: os.stat_result,
    invocation_argv: list[str],
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    source_file_bindings: list[dict[str, str]],
    artifact_path: Path,
    artifact_raw: bytes,
    completed_at: datetime,
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    body = {
        "contract": REPLAY_WORKER_RECEIPT_CONTRACT,
        "adapter_id": candidate["adapter_id"],
        "model_id": candidate["model_id"],
        "config_sha256": candidate["config_sha256"],
        "producer_id": candidate["producer_id"],
        "executable_path": worker_config["executable_path"],
        "executable_sha256": worker_config["executable_sha256"],
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "executable_uid": executable_stat.st_uid,
        "executable_gid": executable_stat.st_gid,
        "argv": invocation_argv,
        "argv_sha256": _canonical_sha256(invocation_argv),
        "git_head": job["git_head"],
        "git_head_sha256": job["git_head_sha256"],
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "job_manifest_sha256": job["manifest_sha256"],
        "output_manifest_sha256": job["output_manifest_sha256"],
        "source_files": source_file_bindings,
        "windows": candidate["windows"],
        "costs": ["BASE", "STRESS"],
        "intrabar_paths": ["OHLC", "OLHC"],
        "results_artifact_path": str(artifact_path.relative_to(repository_root)),
        "results_artifact_sha256": _raw_sha256(artifact_raw),
        "completed_at_utc": _format_utc(completed_at),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "signature_key_id": worker_config["signature_key_id"],
    }
    signed_payload = _canonical_bytes(body)
    receipt_without_digest = {
        **body,
        "signed_payload_sha256": _raw_sha256(signed_payload),
        "signature_base64": base64.b64encode(private_key.sign(signed_payload)).decode(
            "ascii"
        ),
    }
    return {
        **receipt_without_digest,
        "receipt_sha256": _canonical_sha256(receipt_without_digest),
    }


def _load_worker_config(root: Path, path: Path) -> dict[str, Any]:
    raw = _read_canonical_file(
        path,
        root=root,
        label="trusted replay worker config",
    )
    try:
        return validate_replay_worker_config(
            _json_bytes(raw, "trusted replay worker config")
        )
    except ValueError as exc:
        raise TrustedReplayWorkerError("trusted worker config is invalid") from exc


def _load_external_private_key(
    path: Path,
    *,
    repository_root: Path,
    expected_public_key_base64: str,
) -> Ed25519PrivateKey:
    if not isinstance(path, Path) or not path.is_absolute():
        raise TrustedReplayWorkerError("private key path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repository_root)
    except ValueError:
        pass
    except OSError as exc:
        raise TrustedReplayWorkerError("external private key is unavailable") from exc
    else:
        raise TrustedReplayWorkerError("private key must remain outside repository")
    raw, item_stat = _read_regular_no_follow(
        path,
        allowed_root=None,
        max_bytes=PRIVATE_KEY_MAX_BYTES,
        label="external Ed25519 private key",
    )
    if stat.S_IMODE(item_stat.st_mode) != 0o600:
        raise TrustedReplayWorkerError("external private key mode must be 0600")
    if item_stat.st_uid != os.getuid():
        raise TrustedReplayWorkerError("external private key owner is invalid")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise TrustedReplayWorkerError("private key must be canonical base64 plus LF")
    try:
        key_bytes = base64.b64decode(raw[:-1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise TrustedReplayWorkerError("private key is not canonical base64") from exc
    if len(key_bytes) != 32 or base64.b64encode(key_bytes) + b"\n" != raw:
        raise TrustedReplayWorkerError("private key must contain raw Ed25519 bytes")
    private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if base64.b64encode(public_key).decode("ascii") != expected_public_key_base64:
        raise TrustedReplayWorkerError("private key does not match worker public key")
    return private_key


def _market_event_time(row: Mapping[str, Any]) -> datetime:
    payload = row.get("payload")
    quote = payload.get("quote") if isinstance(payload, Mapping) else None
    raw = quote.get("ts") if isinstance(quote, Mapping) else None
    if not isinstance(raw, str) or not raw:
        raise TrustedReplayWorkerError("replay event lacks immutable quote time")
    try:
        value = datetime.fromisoformat(raw.split("#", 1)[0])
    except ValueError as exc:
        raise TrustedReplayWorkerError("replay quote time is invalid") from exc
    if value.tzinfo is None:
        raise TrustedReplayWorkerError("replay quote time is not timezone-aware")
    return value


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    value, raw = _load_strict_json_bytes(path, label)
    if raw != _canonical_bytes(value) + b"\n":
        raise TrustedReplayWorkerError(f"{label} is not canonical JSON")
    return value


def _load_strict_json(path: Path, label: str) -> dict[str, Any]:
    value, _ = _load_strict_json_bytes(path, label)
    return value


def _load_strict_json_bytes(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    raw, _ = _read_regular_no_follow(
        path,
        allowed_root=path.parent,
        max_bytes=4 * 1024 * 1024,
        label=label,
    )
    value = _json_bytes(raw, label)
    return value, raw


def _hash_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path | None,
    label: str,
) -> tuple[str, os.stat_result]:
    descriptor, before = _open_regular_no_follow(
        path,
        allowed_root=allowed_root,
        label=label,
    )
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_size <= 0
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise TrustedReplayWorkerError(f"{label} changed while hashing")
    return digest.hexdigest(), after


def _read_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path | None,
    max_bytes: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_regular_no_follow(
        path,
        allowed_root=allowed_root,
        label=label,
    )
    try:
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise TrustedReplayWorkerError(f"{label} size is invalid")
        raw = bytearray()
        while len(raw) <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > max_bytes
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise TrustedReplayWorkerError(f"{label} changed while reading")
    return bytes(raw), after


def _open_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path | None,
    label: str,
) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        target = Path(os.path.abspath(path))
        if allowed_root is not None:
            root = allowed_root.resolve(strict=True)
            relative = target.relative_to(root)
            current = root
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    raise OSError("symlink component")
        lexical_stat = os.lstat(target)
        if stat.S_ISLNK(lexical_stat.st_mode):
            raise OSError("symlink target")
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        item_stat = os.fstat(descriptor)
    except (OSError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise TrustedReplayWorkerError(
            f"{label} is unavailable or outside its canonical root"
        ) from exc
    if (
        not stat.S_ISREG(item_stat.st_mode)
        or item_stat.st_dev != lexical_stat.st_dev
        or item_stat.st_ino != lexical_stat.st_ino
    ):
        os.close(descriptor)
        raise TrustedReplayWorkerError(f"{label} is not a regular no-follow file")
    return descriptor, item_stat


def _write_exclusive_fsynced(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise TrustedReplayWorkerError(f"{path.name} already exists") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _repository_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TrustedReplayWorkerError("repository_root must be a Path")
    try:
        root = value.resolve(strict=True)
    except OSError as exc:
        raise TrustedReplayWorkerError("repository root is unavailable") from exc
    if not root.is_dir():
        raise TrustedReplayWorkerError("repository root is not a directory")
    return root


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrustedReplayWorkerError(f"{label} is not a SHA-256 digest")
    return value


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TrustedReplayWorkerError("completed_at_utc is invalid") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TrustedReplayWorkerError("completed_at_utc is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise TrustedReplayWorkerError("completed_at_utc must be UTC")
    return parsed


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_market_open(value: datetime, label: str) -> None:
    try:
        open_now = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise TrustedReplayWorkerError(f"{label} market status is unavailable") from exc
    if not open_now:
        raise TrustedReplayWorkerMarketClosed(
            f"{label}: AI replay evaluation is disabled while FX is closed"
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
