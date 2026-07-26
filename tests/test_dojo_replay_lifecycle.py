from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_autonomous_improvement import (
    append_candidate_event,
    initialize_research_root,
)
from quant_rabbit.dojo_replay_gates import (
    PROOF_MANIFEST_CONTRACT,
    canonical_proof_manifest_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANDIDATE_SPEC_CONTRACT,
    CANONICAL_RESEARCH_RELATIVE_ROOT,
    FUTURE_REGISTRY_CONTRACT,
    JOB_MANIFEST_CONTRACT,
    LAUNCH_PREFLIGHT_CONTRACT,
    PROOF_ARTIFACT_CONTRACT,
    REPLAY_JOB_OWNER_CONTRACT,
    REPLAY_OUTPUT_MANIFEST_CONTRACT,
    SOURCE_MANIFEST_CONTRACT,
    DojoReplayLifecycleError,
    canonical_proof_artifact_bytes,
    evaluate_replay_proof,
    issue_paper_ai_inventory_launch_preflight,
    seal_proof_artifact_exclusive,
    verify_paper_ai_inventory_launch_preflight,
)
from quant_rabbit.dojo_replay_worker_receipt import (
    REPLAY_WORKER_RECEIPT_CONTRACT,
    replay_worker_config_sha256,
)


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sealed(value: dict[str, object], field: str) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return {**body, field: _canonical_sha256(body)}


def _metrics(
    *,
    net_jpy: float,
    profit_factor: float,
    worst_day_jpy: float,
    drawdown_jpy: float,
) -> dict[str, object]:
    settlements = 40
    return {
        "settlements": settlements,
        "active_days": 24,
        "net_jpy": net_jpy,
        "profit_factor": profit_factor,
        "expectancy_jpy": net_jpy / settlements,
        "worst_day_jpy": worst_day_jpy,
        "realized_drawdown_jpy": drawdown_jpy,
        "margin_events": 0,
        "ruin_events": 0,
        "unresolved_positions": 0,
        "unresolved_orders": 0,
        "end_of_replay_forced_close_count": 0,
        "end_of_replay_forced_close_net_jpy": 0.0,
    }


def _passing_arms() -> list[dict[str, object]]:
    arms: list[dict[str, object]] = []
    for window in ("TRAIN", "VAL", "S5"):
        for policy in ("BASELINE", "CANDIDATE"):
            for cost in ("BASE", "STRESS"):
                for intrabar in ("OHLC", "OLHC"):
                    baseline = policy == "BASELINE"
                    arms.append(
                        {
                            "window": window,
                            "policy": policy,
                            "cost": cost,
                            "intrabar": intrabar,
                            "metrics": _metrics(
                                net_jpy=300.0 if baseline else 500.0,
                                profit_factor=1.2 if baseline else 1.4,
                                worst_day_jpy=-100.0 if baseline else -90.0,
                                drawdown_jpy=200.0 if baseline else 180.0,
                            ),
                        }
                    )
    return arms


def _bundle(
    *,
    arms: list[dict[str, object]] | None = None,
    reused_window: bool = False,
) -> dict[str, object]:
    candidate_id = "a" * 64
    policy_sha256 = "b" * 64
    output_manifest_sha256 = "c" * 64
    adapter_id = "test-replay-adapter"
    model_id = "test-replay-model"
    config_sha256 = "d" * 64
    producer_id = "test-replay-producer"
    source_capture_manifest_sha256 = "e" * 64
    argv = ["/usr/bin/python3", "paper-replay.py"]
    source_bytes = {
        "TRAIN": _canonical_bytes(
            {
                "contract": SOURCE_MANIFEST_CONTRACT,
                "granularity": "M1",
                "pairs": ["USD_JPY"],
                "files": [{"path": "train", "sha256": "1" * 64}],
            }
        ),
        "VAL": _canonical_bytes(
            {
                "contract": SOURCE_MANIFEST_CONTRACT,
                "granularity": "M1",
                "pairs": ["USD_JPY"],
                "files": [{"path": "val", "sha256": "2" * 64}],
            }
        ),
        "S5": _canonical_bytes(
            {
                "contract": SOURCE_MANIFEST_CONTRACT,
                "granularity": "S5",
                "pairs": ["USD_JPY"],
                "files": [{"path": "s5", "sha256": "3" * 64}],
            }
        ),
    }
    windows = {
        "TRAIN": {
            "from_utc": "2026-01-01T00:00:00+00:00",
            "to_utc": "2026-03-01T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_bytes["TRAIN"]).hexdigest(),
        },
        "VAL": {
            "from_utc": (
                "2026-01-01T00:00:00+00:00"
                if reused_window
                else "2026-03-01T00:00:00+00:00"
            ),
            "to_utc": (
                "2026-03-01T00:00:00+00:00"
                if reused_window
                else "2026-05-01T00:00:00+00:00"
            ),
            "source_sha256": hashlib.sha256(source_bytes["VAL"]).hexdigest(),
        },
        "S5": {
            "from_utc": "2026-05-10T00:00:00Z",
            "to_utc": "2026-07-17T00:00:00Z",
            "source_sha256": hashlib.sha256(source_bytes["S5"]).hexdigest(),
        },
    }
    spec = _sealed(
        {
            "contract": CANDIDATE_SPEC_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "family": "INVENTORY_RELEASE",
            "adapter_id": adapter_id,
            "model_id": model_id,
            "config_sha256": config_sha256,
            "producer_id": producer_id,
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
            "windows": windows,
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
            "end_of_replay_forced_close_benefit": False,
        },
        "spec_sha256",
    )
    spec_bytes = _canonical_bytes(spec)
    git_head = "4" * 40
    git_head_sha256 = hashlib.sha256(git_head.encode("ascii")).hexdigest()
    job = _sealed(
        {
            "contract": JOB_MANIFEST_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": spec["spec_sha256"],
            "policy_sha256": policy_sha256,
            "git_head": git_head,
            "git_head_sha256": git_head_sha256,
            "output_manifest_sha256": output_manifest_sha256,
            "adapter_id": adapter_id,
            "model_id": model_id,
            "config_sha256": config_sha256,
            "producer_id": producer_id,
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
            "argv": argv,
            "argv_sha256": _canonical_sha256(argv),
            "files": [
                {
                    "path": f"{window.lower()}-source.json",
                    "sha256": windows[window]["source_sha256"],
                }
                for window in ("TRAIN", "VAL", "S5")
            ],
        },
        "manifest_sha256",
    )
    job_bytes = _canonical_bytes(job)
    artifact_body = {
        "contract": PROOF_ARTIFACT_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": candidate_id,
        "spec_sha256": spec["spec_sha256"],
        "policy_sha256": policy_sha256,
        "job_manifest_sha256": job["manifest_sha256"],
        "git_head": git_head,
        "git_head_sha256": git_head_sha256,
        "artifact_manifest_sha256": output_manifest_sha256,
        "windows": windows,
        "completed_at_utc": "2026-07-25T18:00:00+00:00",
        "arms": copy.deepcopy(arms if arms is not None else _passing_arms()),
    }
    proof_artifact_bytes = canonical_proof_artifact_bytes(artifact_body)
    artifact = json.loads(proof_artifact_bytes)
    registry = {
        "contract": FUTURE_REGISTRY_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "experiment_id": "paper-ai-inventory-experiment-v1",
        "proof_mode": "candidate",
        "window": {
            "start_utc": "2026-07-26T21:00:00+00:00",
            "end_utc": "2026-07-31T21:00:00+00:00",
        },
        "proof_binding": {
            "candidate_id": candidate_id,
            "spec_sha256": spec["spec_sha256"],
            "policy_sha256": policy_sha256,
            "job_manifest_sha256": job["manifest_sha256"],
            "proof_artifact_sha256": artifact["artifact_sha256"],
            "git_head": git_head,
            "git_head_sha256": git_head_sha256,
            "adapter_id": adapter_id,
            "model_id": model_id,
            "config_sha256": config_sha256,
            "producer_id": producer_id,
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
        },
        "rooms": [
            {
                "room_id": "paper-ai-inventory-candidate-base",
                "candidate_id": candidate_id,
                "adapter_id": adapter_id,
                "model_id": model_id,
                "config_sha256": config_sha256,
                "producer_id": producer_id,
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
            },
            {
                "room_id": "paper-ai-inventory-candidate-stress",
                "candidate_id": candidate_id,
                "adapter_id": adapter_id,
                "model_id": model_id,
                "config_sha256": config_sha256,
                "producer_id": producer_id,
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
            },
        ],
    }
    return {
        "candidate_spec_bytes": spec_bytes,
        "job_manifest_bytes": job_bytes,
        "source_manifest_bytes": source_bytes,
        "proof_artifact_bytes": proof_artifact_bytes,
        "future_registry_bytes": _canonical_bytes(registry),
    }


def _evaluate(bundle: dict[str, object]) -> dict[str, object]:
    return evaluate_replay_proof(
        candidate_spec_bytes=bundle["candidate_spec_bytes"],
        job_manifest_bytes=bundle["job_manifest_bytes"],
        source_manifest_bytes=bundle["source_manifest_bytes"],
        proof_artifact_bytes=bundle["proof_artifact_bytes"],
        future_registry_bytes=bundle["future_registry_bytes"],
    )


def _git_head(root: Path) -> str:
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
    seed.write_text("isolated replay fixture\n", encoding="utf-8")
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


def _trusted_worker(
    root: Path,
) -> tuple[Ed25519PrivateKey, dict[str, object], list[str]]:
    private_key = Ed25519PrivateKey.generate()
    executable = Path(sys.executable).resolve()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    config_body: dict[str, object] = {
        "adapter_id": "trusted-replay-worker-v1",
        "model_id": "bounded-replay-engine-v1",
        "producer_id": "quant-rabbit-replay-worker-v1",
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "signature_key_id": "test-replay-worker-key-v1",
        "ed25519_public_key_base64": base64.b64encode(public_key).decode("ascii"),
    }
    config = {
        **config_body,
        "config_sha256": replay_worker_config_sha256(config_body),
    }
    argv = [str(executable), "-I", str(root / "bounded-paper-replay.py")]
    return private_key, config, argv


def _write_source_capture_manifest(root: Path) -> str:
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
        "manifest_id": "source-capture-test-v1",
        "capture_key_id": "source-capture-test-key-v1",
        "ed25519_public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "allowed_source_roles": ["candles", "news", "quote"],
        "allowed_provider_kinds": ["official", "read_only_broker"],
        "source_adapters": [
            {
                "source_role": "candles",
                "provider_kind": "read_only_broker",
                "adapter_id": "test-candles-adapter-v1",
                "adapter_module": "quant_rabbit.test_capture",
                "adapter_callable": "capture_candles",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "2" * 64,
            },
            {
                "source_role": "news",
                "provider_kind": "official",
                "adapter_id": "test-news-adapter-v1",
                "adapter_module": "quant_rabbit.test_capture",
                "adapter_callable": "capture_news",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "3" * 64,
            },
            {
                "source_role": "quote",
                "provider_kind": "read_only_broker",
                "adapter_id": "test-quote-adapter-v1",
                "adapter_module": "quant_rabbit.test_capture",
                "adapter_callable": "capture_quote",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "4" * 64,
            },
        ],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    manifest = {**body, "manifest_sha256": _canonical_sha256(body)}
    raw = _canonical_bytes(manifest)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    manifest_root = (
        root / "research/data/dojo_paper_ai_inventory_v1/source_capture/manifests"
    )
    manifest_root.mkdir(parents=True)
    (manifest_root / f"{file_sha256}.json").write_bytes(raw)
    return file_sha256


def _write_worker_receipt(
    *,
    root: Path,
    replay_root: Path,
    private_key: Ed25519PrivateKey,
    trusted_worker: dict[str, object],
    argv: list[str],
    candidate_id: str,
    spec_sha256: str,
    policy_sha256: str,
    job_manifest_sha256: str,
    output_manifest_sha256: str,
    git_head: str,
    windows: dict[str, object],
    source_file_bindings: list[dict[str, str]],
    artifact_raw: bytes,
    completed_at_utc: str,
) -> dict[str, object]:
    executable = Path(str(trusted_worker["executable_path"]))
    executable_stat = executable.stat()
    artifact_path = replay_root / "proof_artifact.json"
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
        "policy_sha256": policy_sha256,
        "job_manifest_sha256": job_manifest_sha256,
        "output_manifest_sha256": output_manifest_sha256,
        "source_files": source_file_bindings,
        "windows": windows,
        "costs": ["BASE", "STRESS"],
        "intrabar_paths": ["OHLC", "OLHC"],
        "results_artifact_path": str(artifact_path.relative_to(root)),
        "results_artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "completed_at_utc": completed_at_utc,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "signature_key_id": trusted_worker["signature_key_id"],
    }
    signed_payload = _canonical_bytes(body).rstrip(b"\n")
    receipt_without_digest = {
        **body,
        "signed_payload_sha256": hashlib.sha256(signed_payload).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(signed_payload)).decode(
            "ascii"
        ),
    }
    receipt = {
        **receipt_without_digest,
        "receipt_sha256": _canonical_sha256(receipt_without_digest),
    }
    (replay_root / "worker_receipt.json").write_bytes(_canonical_bytes(receipt))
    return receipt


def _controller_fixture(root: Path) -> dict[str, object]:
    git_head = _git_head(root)
    private_key, trusted_worker, argv = _trusted_worker(root)
    source_capture_manifest_sha256 = _write_source_capture_manifest(root)
    research_root = root / CANONICAL_RESEARCH_RELATIVE_ROOT
    sources_root = root / "research/data/replay_source_fixture"
    sources_root.mkdir(parents=True)
    source_bytes: dict[str, bytes] = {}
    source_manifest_bytes: dict[str, bytes] = {}
    for window in ("TRAIN", "VAL", "S5"):
        source_path = sources_root / f"{window}.csv"
        raw = f"timestamp,bid,ask\n{window},163.00,163.01\n".encode()
        source_path.write_bytes(raw)
        source = {
            "contract": SOURCE_MANIFEST_CONTRACT,
            "granularity": "S5" if window == "S5" else "M1",
            "pairs": ["USD_JPY"],
            "files": [
                {
                    "path": str(source_path.relative_to(root)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        }
        source_bytes[window] = raw
        source_manifest_bytes[window] = _canonical_bytes(source)

    windows = {
        "TRAIN": {
            "from_utc": "2026-01-01T00:00:00+00:00",
            "to_utc": "2026-03-01T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_bytes["TRAIN"]).hexdigest(),
        },
        "VAL": {
            "from_utc": "2026-03-01T00:00:00+00:00",
            "to_utc": "2026-05-01T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_bytes["VAL"]).hexdigest(),
        },
        "S5": {
            "from_utc": "2026-05-10T00:00:00+00:00",
            "to_utc": "2026-07-17T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_bytes["S5"]).hexdigest(),
        },
    }
    spec = {
        "contract": CANDIDATE_SPEC_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "family": "INVENTORY_RELEASE",
        "adapter_id": trusted_worker["adapter_id"],
        "model_id": trusted_worker["model_id"],
        "config_sha256": trusted_worker["config_sha256"],
        "producer_id": trusted_worker["producer_id"],
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
        "hypothesis": "release only invalidated virtual inventory",
        "causal_narrative": "ceiling losses dominate bounded winners",
        "expected_mechanism": "prospective invalidation releases capital",
        "falsifier": "independent stress expectancy is non-positive",
        "affected_pair": "USD_JPY",
        "affected_strategy": "QR_DOJO_AI_INVENTORY_V1",
        "evidence_cohort": "immutable strategy-tagged settlements",
        "changed_rule": {
            "name": "ai_inventory_release",
            "baseline": False,
            "candidate": True,
        },
        "unchanged_controls": ["entry", "size", "tp", "sl", "ceiling"],
        "evidence_sha256s": ["e" * 64],
        "windows": windows,
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
    at = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
    initialize_research_root(
        research_root,
        recorded_at_utc=at,
        implementation_sha256="f" * 64,
    )
    registration, _ = append_candidate_event(
        research_root,
        event_type="CANDIDATE_PREREGISTERED",
        payload={
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "spec": spec,
        },
        recorded_at_utc=at + timedelta(seconds=1),
    )
    sealed_spec = registration["payload"]["spec"]
    candidate_id = registration["payload"]["candidate_id"]
    replay_root = research_root / "candidates" / candidate_id / "replay"
    manifest_root = replay_root / "source_manifests"
    manifest_root.mkdir(parents=True)
    for window in ("TRAIN", "VAL", "S5"):
        (manifest_root / f"{window}.json").write_bytes(source_manifest_bytes[window])

    policy_sha256 = "b" * 64
    source_bindings = {
        window: hashlib.sha256(source_manifest_bytes[window]).hexdigest()
        for window in ("TRAIN", "VAL", "S5")
    }
    output_manifest = {
        "contract": REPLAY_OUTPUT_MANIFEST_CONTRACT,
        "candidate_id": candidate_id,
        "spec_sha256": sealed_spec["spec_sha256"],
        "policy_sha256": policy_sha256,
        "git_head": git_head,
        "source_manifest_sha256s": source_bindings,
        "adapter_id": trusted_worker["adapter_id"],
        "model_id": trusted_worker["model_id"],
        "config_sha256": trusted_worker["config_sha256"],
        "producer_id": trusted_worker["producer_id"],
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    output_raw = _canonical_bytes(output_manifest)
    (replay_root / "output_manifest.json").write_bytes(output_raw)
    output_sha256 = hashlib.sha256(output_raw).hexdigest()
    job = _sealed(
        {
            "contract": JOB_MANIFEST_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": sealed_spec["spec_sha256"],
            "policy_sha256": policy_sha256,
            "git_head": git_head,
            "git_head_sha256": hashlib.sha256(git_head.encode("ascii")).hexdigest(),
            "output_manifest_sha256": output_sha256,
            "adapter_id": trusted_worker["adapter_id"],
            "model_id": trusted_worker["model_id"],
            "config_sha256": trusted_worker["config_sha256"],
            "producer_id": trusted_worker["producer_id"],
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
            "argv": argv,
            "argv_sha256": _canonical_sha256(argv),
            "files": [
                {
                    "path": str((manifest_root / f"{window}.json").relative_to(root)),
                    "sha256": source_bindings[window],
                }
                for window in ("TRAIN", "VAL", "S5")
            ],
        },
        "manifest_sha256",
    )
    job_raw = _canonical_bytes(job)
    (replay_root / "job_manifest.json").write_bytes(job_raw)
    process_sha256 = "6" * 64
    owner = _sealed(
        {
            "contract": REPLAY_JOB_OWNER_CONTRACT,
            "candidate_id": candidate_id,
            "job_manifest_sha256": job["manifest_sha256"],
            "pid": os.getpid(),
            "screen_name": "qr-dojo-improve-aiinventory",
            "process_command_sha256": process_sha256,
            "output_directory": str(replay_root.relative_to(root)),
            "status": "COMPLETED",
            "completed_at_utc": "2026-07-25T18:02:00Z",
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        },
        "owner_sha256",
    )
    (replay_root / "job_owner.json").write_bytes(_canonical_bytes(owner))
    append_candidate_event(
        research_root,
        event_type="REPLAY_STARTED",
        payload={
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "job_lock": {
                "git_head_sha256": job["git_head_sha256"],
                "spec_sha256": sealed_spec["spec_sha256"],
                "policy_sha256": policy_sha256,
                "output_manifest_sha256": output_sha256,
                "argv": argv,
                "argv_sha256": job["argv_sha256"],
                "adapter_id": trusted_worker["adapter_id"],
                "model_id": trusted_worker["model_id"],
                "config_sha256": trusted_worker["config_sha256"],
                "producer_id": trusted_worker["producer_id"],
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
                "environment_allowlist": ["PATH", "PYTHONPATH"],
                "output_directory": str(replay_root.relative_to(root)),
                "screen_name": "qr-dojo-improve-aiinventory",
                "pid": os.getpid(),
                "process_command_sha256": process_sha256,
                "job_manifest_sha256": job["manifest_sha256"],
                "job_owner_sha256": owner["owner_sha256"],
            },
        },
        recorded_at_utc=at + timedelta(seconds=2),
    )
    artifact_raw = canonical_proof_artifact_bytes(
        {
            "contract": PROOF_ARTIFACT_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": sealed_spec["spec_sha256"],
            "policy_sha256": policy_sha256,
            "job_manifest_sha256": job["manifest_sha256"],
            "git_head": git_head,
            "git_head_sha256": job["git_head_sha256"],
            "artifact_manifest_sha256": output_sha256,
            "windows": sealed_spec["windows"],
            "completed_at_utc": "2026-07-25T18:02:00Z",
            "arms": _passing_arms(),
        }
    )
    (replay_root / "proof_artifact.json").write_bytes(artifact_raw)
    artifact = json.loads(artifact_raw)
    source_file_bindings = [
        {
            "window": window,
            "path": str((sources_root / f"{window}.csv").relative_to(root)),
            "sha256": hashlib.sha256(source_bytes[window]).hexdigest(),
        }
        for window in ("TRAIN", "VAL", "S5")
    ]
    worker_receipt = _write_worker_receipt(
        root=root,
        replay_root=replay_root,
        private_key=private_key,
        trusted_worker=trusted_worker,
        argv=argv,
        candidate_id=candidate_id,
        spec_sha256=sealed_spec["spec_sha256"],
        policy_sha256=policy_sha256,
        job_manifest_sha256=job["manifest_sha256"],
        output_manifest_sha256=output_sha256,
        git_head=git_head,
        windows=sealed_spec["windows"],
        source_file_bindings=source_file_bindings,
        artifact_raw=artifact_raw,
        completed_at_utc="2026-07-25T18:02:01Z",
    )
    proof_manifest = {
        "contract": PROOF_MANIFEST_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": candidate_id,
        "spec_sha256": sealed_spec["spec_sha256"],
        "policy_sha256": policy_sha256,
        "artifact_manifest_sha256": output_sha256,
        "windows": sealed_spec["windows"],
        "arms": artifact["arms"],
    }
    proof_manifest_sha256 = canonical_proof_manifest_sha256(proof_manifest)
    append_candidate_event(
        research_root,
        event_type="REPLAY_PASSED",
        payload={
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "independent_stress_metrics": {
                "pf": 1.4,
                "net": 500.0,
                "expectancy": 12.5,
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
        recorded_at_utc=at + timedelta(seconds=3),
    )
    experiment_id = "paper-ai-inventory-experiment-v1"
    room_id = "paper-ai-inventory-room-v1"
    registry = {
        "contract": FUTURE_REGISTRY_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "experiment_id": experiment_id,
        "proof_mode": "candidate",
        "window": {
            "start_utc": "2026-07-26T21:00:00Z",
            "end_utc": "2026-07-31T21:00:00Z",
        },
        "proof_binding": {
            "candidate_id": candidate_id,
            "spec_sha256": sealed_spec["spec_sha256"],
            "policy_sha256": policy_sha256,
            "job_manifest_sha256": job["manifest_sha256"],
            "proof_artifact_sha256": artifact["artifact_sha256"],
            "git_head": git_head,
            "git_head_sha256": job["git_head_sha256"],
            "adapter_id": trusted_worker["adapter_id"],
            "model_id": trusted_worker["model_id"],
            "config_sha256": trusted_worker["config_sha256"],
            "producer_id": trusted_worker["producer_id"],
            "source_capture_manifest_sha256": source_capture_manifest_sha256,
        },
        "rooms": [
            {
                "room_id": room_id,
                "candidate_id": candidate_id,
                "adapter_id": trusted_worker["adapter_id"],
                "model_id": trusted_worker["model_id"],
                "config_sha256": trusted_worker["config_sha256"],
                "producer_id": trusted_worker["producer_id"],
                "source_capture_manifest_sha256": source_capture_manifest_sha256,
            }
        ],
    }
    registry_path = (
        root / "config" / f"dojo_paper_rooms_ai_inventory_{candidate_id}.json"
    )
    registry_path.parent.mkdir()
    registry_path.write_bytes(_canonical_bytes(registry))
    return {
        "candidate_id": candidate_id,
        "spec_sha256": sealed_spec["spec_sha256"],
        "policy_sha256": policy_sha256,
        "experiment_id": experiment_id,
        "room_id": room_id,
        "registry_path": registry_path,
        "research_root": research_root,
        "replay_root": replay_root,
        "recorded_at": at + timedelta(seconds=4),
        "source_bytes": source_bytes,
        "trusted_worker_allowlist": {str(trusted_worker["adapter_id"]): trusted_worker},
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
    }


def test_passed_synthetic_metrics_never_prepare_paper_eligibility() -> None:
    result = _evaluate(_bundle())

    assert result["decision"] == "PROOF_ELIGIBLE_UNTRUSTED"
    assert result["proof_eligible"] is True
    assert result["ledger_append_performed"] is False
    assert result["paper_room_launched"] is False
    assert result["append_controller"] == "issue_paper_ai_inventory_launch_preflight"
    assert result["paper_eligible_event_payload"] is None
    assert result["gate_decision"]["paper_eligible"] is False
    assert result["gate_decision"]["artifact_provenance_authenticated"] is False
    signature = inspect.signature(evaluate_replay_proof)
    assert "expected_manifest_sha256" not in signature.parameters


def test_exclusive_artifact_seal_fsyncs_and_never_overwrites(
    tmp_path: Path,
) -> None:
    artifact_bytes = _bundle()["proof_artifact_bytes"]
    assert isinstance(artifact_bytes, bytes)
    target = tmp_path / "proof.json"

    receipt = seal_proof_artifact_exclusive(target, artifact_bytes)

    assert target.read_bytes() == artifact_bytes
    assert receipt["exclusive"] is True
    assert receipt["fsynced"] is True
    with pytest.raises(DojoReplayLifecycleError, match="overwrite is forbidden"):
        seal_proof_artifact_exclusive(target, artifact_bytes)
    assert target.read_bytes() == artifact_bytes


def test_naked_metrics_and_partial_matrix_fail_closed() -> None:
    naked = _bundle()
    naked["proof_artifact_bytes"] = _canonical_bytes(_passing_arms())
    naked_result = _evaluate(naked)
    assert naked_result["decision"] == "MEASUREMENT_BLOCKED"
    assert naked_result["paper_eligible_event_payload"] is None

    partial = _passing_arms()
    partial.pop()
    partial_result = _evaluate(_bundle(arms=partial))
    assert partial_result["decision"] == "PROOF_REJECTED"
    assert partial_result["paper_eligible_event_payload"] is None


def test_reused_preregistered_window_is_measurement_blocked() -> None:
    result = _evaluate(_bundle(reused_window=True))

    assert result["decision"] == "MEASUREMENT_BLOCKED"
    assert "reuses or overlaps" in result["reason"]


def test_spec_artifact_and_source_digest_mismatches_fail_closed() -> None:
    for mutation in ("spec", "artifact", "source"):
        bundle = _bundle()
        if mutation == "spec":
            raw = bundle["candidate_spec_bytes"]
            assert isinstance(raw, bytes)
            spec = json.loads(raw)
            spec["risk_gates"]["positive_net"] = False
            bundle["candidate_spec_bytes"] = _canonical_bytes(spec)
        elif mutation == "artifact":
            raw = bundle["proof_artifact_bytes"]
            assert isinstance(raw, bytes)
            artifact = json.loads(raw)
            artifact["arms"][0]["metrics"]["net_jpy"] += 1.0
            bundle["proof_artifact_bytes"] = _canonical_bytes(artifact)
        else:
            sources = bundle["source_manifest_bytes"]
            assert isinstance(sources, dict)
            sources["VAL"] = sources["VAL"] + b" "

        result = _evaluate(bundle)

        assert result["decision"] == "MEASUREMENT_BLOCKED"
        assert "mismatch" in result["reason"]
        assert result["paper_eligible_event_payload"] is None


def test_git_and_future_registry_bindings_fail_closed() -> None:
    for mutation in ("git", "registry", "past_window"):
        bundle = _bundle()
        if mutation == "git":
            raw = bundle["job_manifest_bytes"]
            assert isinstance(raw, bytes)
            job = json.loads(raw)
            job["git_head"] = "5" * 40
            job = _sealed(job, "manifest_sha256")
            bundle["job_manifest_bytes"] = _canonical_bytes(job)
        else:
            raw = bundle["future_registry_bytes"]
            assert isinstance(raw, bytes)
            registry = json.loads(raw)
            if mutation == "registry":
                registry["proof_binding"]["policy_sha256"] = "9" * 64
            else:
                registry["window"]["start_utc"] = "2026-07-25T17:00:00+00:00"
            bundle["future_registry_bytes"] = _canonical_bytes(registry)

        result = _evaluate(bundle)

        assert result["decision"] == "MEASUREMENT_BLOCKED"
        assert result["paper_eligible_event_payload"] is None


def test_duplicate_keys_and_nonfinite_json_fail_closed() -> None:
    bundle = _bundle()
    spec_bytes = bundle["candidate_spec_bytes"]
    assert isinstance(spec_bytes, bytes)
    duplicate_spec = b'{"candidate_id":"' + (b"a" * 64) + b'",' + spec_bytes[1:]
    duplicate_bundle = {**bundle, "candidate_spec_bytes": duplicate_spec}
    duplicate_result = _evaluate(duplicate_bundle)
    assert duplicate_result["decision"] == "MEASUREMENT_BLOCKED"

    registry_bytes = bundle["future_registry_bytes"]
    assert isinstance(registry_bytes, bytes)
    nonfinite_registry = b'{"nonfinite":NaN,' + registry_bytes[1:]
    nonfinite_bundle = {
        **bundle,
        "future_registry_bytes": nonfinite_registry,
    }
    nonfinite_result = _evaluate(nonfinite_bundle)
    assert nonfinite_result["decision"] == "MEASUREMENT_BLOCKED"


def test_failed_independent_gate_never_builds_eligibility_payload() -> None:
    arms = _passing_arms()
    arm = next(
        item
        for item in arms
        if (
            item["window"],
            item["policy"],
            item["cost"],
            item["intrabar"],
        )
        == ("VAL", "CANDIDATE", "STRESS", "OHLC")
    )
    metrics = arm["metrics"]
    assert isinstance(metrics, dict)
    metrics["profit_factor"] = 1.0

    result = _evaluate(_bundle(arms=arms))

    assert result["decision"] == "PROOF_REJECTED"
    assert result["proof_eligible"] is False
    assert result["paper_eligible_event_payload"] is None
    assert result["ledger_append_performed"] is False
    assert result["paper_room_launched"] is False


def test_single_controller_appends_and_issues_verified_preflight(
    tmp_path: Path,
) -> None:
    fixture = _controller_fixture(tmp_path)

    with patch(
        "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
        fixture["trusted_worker_allowlist"],
    ):
        result = issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )

        assert result["decision"] == "PAPER_ELIGIBLE_APPENDED"
        assert result["ledger_append_performed"] is True
        assert result["exclusive_append_claim"] is True
        assert result["fsynced"] is True
        assert result["paper_room_launched"] is False
        token = verify_paper_ai_inventory_launch_preflight(
            tmp_path,
            experiment_id=fixture["experiment_id"],
            room_id=fixture["room_id"],
        )
        assert token["contract"] == LAUNCH_PREFLIGHT_CONTRACT
        assert token["candidate_id"] == fixture["candidate_id"]
        assert (
            token["source_capture_manifest_sha256"]
            == fixture["source_capture_manifest_sha256"]
        )
        assert token["paper_room_launched"] is False
        claim_path = (
            fixture["research_root"]
            / "candidates"
            / fixture["candidate_id"]
            / "paper_eligible_append.claim.json"
        )
        assert claim_path.is_file()
        with pytest.raises(DojoReplayLifecycleError, match="already PAPER_ELIGIBLE"):
            issue_paper_ai_inventory_launch_preflight(
                tmp_path,
                candidate_id=fixture["candidate_id"],
                future_registry_path=fixture["registry_path"],
                recorded_at_utc=fixture["recorded_at"],
            )


def test_production_worker_allowlist_is_empty_and_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _controller_fixture(tmp_path)

    with pytest.raises(
        DojoReplayLifecycleError,
        match="production allowlist",
    ):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )

    ledger = fixture["research_root"] / "candidate_ledger.jsonl"
    assert "PAPER_ELIGIBLE" not in ledger.read_text(encoding="utf-8")


def test_replay_worker_bad_signature_fails_closed(tmp_path: Path) -> None:
    fixture = _controller_fixture(tmp_path)
    receipt_path = fixture["replay_root"] / "worker_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    signature = bytearray(base64.b64decode(receipt["signature_base64"]))
    signature[0] ^= 1
    receipt["signature_base64"] = base64.b64encode(signature).decode("ascii")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt_body)
    receipt_path.write_bytes(_canonical_bytes(receipt))

    with (
        patch(
            "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
            fixture["trusted_worker_allowlist"],
        ),
        pytest.raises(DojoReplayLifecycleError, match="signature is invalid"),
    ):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )


def test_replay_worker_receipt_symlink_fails_closed(tmp_path: Path) -> None:
    fixture = _controller_fixture(tmp_path)
    receipt_path = fixture["replay_root"] / "worker_receipt.json"
    copied = fixture["replay_root"] / "worker_receipt-copy.json"
    copied.write_bytes(receipt_path.read_bytes())
    receipt_path.unlink()
    receipt_path.symlink_to(copied)

    with (
        patch(
            "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
            fixture["trusted_worker_allowlist"],
        ),
        pytest.raises(DojoReplayLifecycleError, match="canonical root"),
    ):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )


def test_unregistered_or_tampered_capture_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _controller_fixture(tmp_path)
    manifest_path = (
        tmp_path
        / "research/data/dojo_paper_ai_inventory_v1/source_capture/manifests"
        / f"{fixture['source_capture_manifest_sha256']}.json"
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with (
        patch(
            "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
            fixture["trusted_worker_allowlist"],
        ),
        pytest.raises(
            DojoReplayLifecycleError,
            match="capture manifest raw bytes digest mismatch",
        ),
    ):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )


def test_synthetic_lifecycle_metrics_without_actual_proof_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _controller_fixture(tmp_path)
    (fixture["replay_root"] / "proof_artifact.json").unlink()

    with pytest.raises(DojoReplayLifecycleError, match="proof artifact"):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )

    ledger = fixture["research_root"] / "candidate_ledger.jsonl"
    assert "PAPER_ELIGIBLE" not in ledger.read_text(encoding="utf-8")


def test_actual_source_bytes_and_git_head_are_reverified(
    tmp_path: Path,
) -> None:
    fixture = _controller_fixture(tmp_path)
    source_path = tmp_path / "research/data/replay_source_fixture/VAL.csv"
    source_path.write_bytes(b"tampered")

    with pytest.raises(DojoReplayLifecycleError, match="source file bytes"):
        issue_paper_ai_inventory_launch_preflight(
            tmp_path,
            candidate_id=fixture["candidate_id"],
            future_registry_path=fixture["registry_path"],
            recorded_at_utc=fixture["recorded_at"],
        )
