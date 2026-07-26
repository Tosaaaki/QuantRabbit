from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import os
import signal
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
from quant_rabbit.dojo_replay_lifecycle import (
    CANDIDATE_SPEC_CONTRACT,
    CANONICAL_RESEARCH_RELATIVE_ROOT,
    JOB_MANIFEST_CONTRACT,
    REPLAY_OUTPUT_MANIFEST_CONTRACT,
    SOURCE_MANIFEST_CONTRACT,
)
from quant_rabbit.dojo_replay_worker import (
    TrustedReplayWorkerError,
    TrustedReplayWorkerMarketClosed,
    _terminate_replay_process,
    run_trusted_replay_worker,
)
from quant_rabbit.dojo_replay_worker_receipt import (
    replay_worker_config_sha256,
    verify_trusted_replay_worker_receipt,
)


OPEN_AT = datetime(2026, 7, 27, 0, 2, tzinfo=timezone.utc)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sealed(value: dict[str, object], field: str) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return {**body, field: _canonical_sha256(body)}


def _git_head(root: Path) -> str:
    (root / "scripts").mkdir(parents=True)
    runner = root / "scripts/run-dojo-inventory-release-replay.py"
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "dojo@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "DOJO Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "scripts"], check=True)
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


def _write_source_capture_manifest(root: Path) -> str:
    capture_key = Ed25519PrivateKey.generate()
    public_key = capture_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    body: dict[str, object] = {
        "contract": "QR_DOJO_AI_SOURCE_CAPTURE_MANIFEST_V1",
        "manifest_id": "test-source-capture-v1",
        "capture_key_id": "test-source-capture-key-v1",
        "ed25519_public_key_base64": base64.b64encode(public_key).decode(),
        "allowed_source_roles": ["quote"],
        "allowed_provider_kinds": ["read_only_broker"],
        "source_adapters": [
            {
                "source_role": "quote",
                "provider_kind": "read_only_broker",
                "adapter_id": "test-read-only-quote-v1",
                "adapter_module": "quant_rabbit.test_quote_source",
                "adapter_callable": "capture_quote",
                "adapter_executable_sha256": "1" * 64,
                "adapter_config_sha256": "2" * 64,
            }
        ],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    manifest = {**body, "manifest_sha256": _canonical_sha256(body)}
    raw = _canonical_bytes(manifest)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    manifest_root = (
        root / "research/data/dojo_paper_ai_inventory_v1/source_capture/manifests"
    )
    manifest_root.mkdir(parents=True)
    (manifest_root / f"{raw_sha256}.json").write_bytes(raw)
    return raw_sha256


def _ledger_row(
    *,
    previous: str,
    event: str,
    payload: dict[str, object],
    at: datetime,
) -> dict[str, object]:
    body: dict[str, object] = {
        "ts_utc": at.isoformat().replace("+00:00", "Z"),
        "event": event,
        "payload": payload,
        "prev_sha": previous,
    }
    return {**body, "sha": _canonical_sha256(body)}


def _write_arm(
    session_dir: Path,
    *,
    candidate: dict[str, object],
    window: str,
    policy: str,
    cost: str,
    intrabar: str,
    source_sha256: str,
    forced_close: bool = False,
) -> None:
    session_dir.mkdir(parents=True)
    windows = candidate["windows"]
    assert isinstance(windows, dict)
    window_value = windows[window]
    assert isinstance(window_value, dict)
    costs = candidate["costs"]
    assert isinstance(costs, dict)
    contract = {
        "candidate_id": candidate["candidate_id"],
        "feed": "replay",
        "pairs": ["USD_JPY"],
        "source": {
            "source_manifest_sha256": source_sha256,
            "time_from": window_value["from_utc"],
            "time_to": window_value["to_utc"],
            "intrabar": intrabar,
        },
        "costs": costs[cost],
        "bot": {
            "module": (
                "/repo/bots/inventory_release_candidate.py"
                if policy == "CANDIDATE"
                else "/repo/bots/lab_bot.py"
            )
        },
    }
    (session_dir / "session_contract.json").write_bytes(_canonical_bytes(contract))

    start = datetime.fromisoformat(str(window_value["from_utc"]).replace("Z", "+00:00"))
    previous = "0" * 64
    rows: list[dict[str, object]] = []
    loss_every = 5 if policy == "CANDIDATE" else 3
    for index in range(40):
        trade_id = f"{window}-{policy}-{cost}-{intrabar}-{index:03d}"
        opened = start + timedelta(days=index, minutes=1)
        quote = {"ts": opened.isoformat()}
        fill = _ledger_row(
            previous=previous,
            event="FILL_MARKET",
            payload={"trade_id": trade_id, "quote": quote},
            at=opened,
        )
        rows.append(fill)
        previous = str(fill["sha"])
        pnl = -10.0 if index % loss_every == 0 else 10.0
        closed = opened + timedelta(minutes=1)
        close = _ledger_row(
            previous=previous,
            event="CLOSE",
            payload={
                "trade_id": trade_id,
                "quote": {"ts": closed.isoformat()},
                "pl_jpy": pnl,
                "reason": (
                    "END_OF_REPLAY"
                    if forced_close and index == 39
                    else ("SL" if pnl < 0 else "TP")
                ),
            },
            at=closed,
        )
        rows.append(close)
        previous = str(close["sha"])
    stopped = _ledger_row(
        previous=previous,
        event="SESSION_STOP",
        payload={},
        at=start + timedelta(days=41),
    )
    rows.append(stopped)
    previous = str(stopped["sha"])
    (session_dir / "ledger.jsonl").write_bytes(
        b"".join(_canonical_bytes(row) for row in rows)
    )
    (session_dir / "broker_snapshot.json").write_bytes(
        _canonical_bytes(
            {
                "balance_jpy": 200_000.0,
                "seq": len(rows),
                "positions": [],
                "orders": [],
                "ledger_sha": previous,
            }
        )
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "repo"
    root.mkdir()
    git_head = _git_head(root)
    executable = Path(sys.executable).resolve()
    private_key = Ed25519PrivateKey.generate()
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_key_path = tmp_path / "worker.key"
    private_key_path.write_bytes(base64.b64encode(private_key_bytes) + b"\n")
    private_key_path.chmod(0o600)
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
        "ed25519_public_key_base64": base64.b64encode(public_key).decode(),
    }
    worker_config = {
        **config_body,
        "config_sha256": replay_worker_config_sha256(config_body),
    }
    config_path = root / "config/replay_worker_test.json"
    config_path.parent.mkdir()
    config_path.write_bytes(_canonical_bytes(worker_config))
    source_capture_sha256 = _write_source_capture_manifest(root)

    sources_root = root / "research/data/replay_worker_sources"
    sources_root.mkdir(parents=True)
    source_manifest_raw: dict[str, bytes] = {}
    for window in ("TRAIN", "VAL", "S5"):
        source_file = sources_root / f"{window}.csv"
        raw = f"timestamp,bid,ask\n{window},163.00,163.01\n".encode()
        source_file.write_bytes(raw)
        source_manifest_raw[window] = _canonical_bytes(
            {
                "contract": SOURCE_MANIFEST_CONTRACT,
                "granularity": "S5" if window == "S5" else "M1",
                "pairs": ["USD_JPY"],
                "files": [
                    {
                        "path": str(source_file.relative_to(root)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ],
            }
        )
    windows = {
        "TRAIN": {
            "from_utc": "2026-01-01T00:00:00+00:00",
            "to_utc": "2026-03-01T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_raw["TRAIN"]).hexdigest(),
        },
        "VAL": {
            "from_utc": "2026-03-01T00:00:00+00:00",
            "to_utc": "2026-05-01T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_raw["VAL"]).hexdigest(),
        },
        "S5": {
            "from_utc": "2026-05-10T00:00:00+00:00",
            "to_utc": "2026-07-17T00:00:00+00:00",
            "source_sha256": hashlib.sha256(source_manifest_raw["S5"]).hexdigest(),
        },
    }
    spec = {
        "contract": CANDIDATE_SPEC_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "family": "INVENTORY_RELEASE",
        "adapter_id": worker_config["adapter_id"],
        "model_id": worker_config["model_id"],
        "config_sha256": worker_config["config_sha256"],
        "producer_id": worker_config["producer_id"],
        "source_capture_manifest_sha256": source_capture_sha256,
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
    research_root = root / CANONICAL_RESEARCH_RELATIVE_ROOT
    initialize_research_root(
        research_root,
        recorded_at_utc=OPEN_AT - timedelta(minutes=2),
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
        recorded_at_utc=OPEN_AT - timedelta(minutes=1),
    )
    candidate = registration["payload"]["spec"]
    candidate_id = registration["payload"]["candidate_id"]
    candidate_dir = research_root / "candidates" / candidate_id
    replay_root = candidate_dir / "replay"
    manifest_root = replay_root / "source_manifests"
    manifest_root.mkdir(parents=True)
    for window, raw in source_manifest_raw.items():
        (manifest_root / f"{window}.json").write_bytes(raw)

    policy = {
        "contract": "QR_DOJO_INVENTORY_RELEASE_POLICY_V1",
        "candidate_id": candidate_id,
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
    policy_raw = _canonical_bytes(policy)
    (replay_root / "policy.json").write_bytes(policy_raw)
    policy_sha256 = hashlib.sha256(policy_raw).hexdigest()
    source_bindings = {
        window: hashlib.sha256(raw).hexdigest()
        for window, raw in source_manifest_raw.items()
    }
    output = {
        "contract": REPLAY_OUTPUT_MANIFEST_CONTRACT,
        "candidate_id": candidate_id,
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": policy_sha256,
        "git_head": git_head,
        "source_manifest_sha256s": source_bindings,
        "adapter_id": worker_config["adapter_id"],
        "model_id": worker_config["model_id"],
        "config_sha256": worker_config["config_sha256"],
        "producer_id": worker_config["producer_id"],
        "source_capture_manifest_sha256": source_capture_sha256,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    output_raw = _canonical_bytes(output)
    (replay_root / "output_manifest.json").write_bytes(output_raw)
    output_sha256 = hashlib.sha256(output_raw).hexdigest()
    command = [
        str(executable),
        str(root / "scripts/run-dojo-inventory-release-replay.py"),
        "--spec",
        str(candidate_dir / "spec.json"),
        "--output-root",
        str(replay_root / "worker_runs"),
        "--source-manifest-dir",
        str(manifest_root),
    ]
    job = _sealed(
        {
            "contract": JOB_MANIFEST_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": policy_sha256,
            "git_head": git_head,
            "git_head_sha256": hashlib.sha256(git_head.encode()).hexdigest(),
            "output_manifest_sha256": output_sha256,
            "adapter_id": worker_config["adapter_id"],
            "model_id": worker_config["model_id"],
            "config_sha256": worker_config["config_sha256"],
            "producer_id": worker_config["producer_id"],
            "source_capture_manifest_sha256": source_capture_sha256,
            "argv": command,
            "argv_sha256": _canonical_sha256(command),
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
    (replay_root / "job_manifest.json").write_bytes(_canonical_bytes(job))
    append_candidate_event(
        research_root,
        event_type="REPLAY_STARTED",
        payload={
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "job_lock": {
                "job_manifest_sha256": job["manifest_sha256"],
                "git_head_sha256": job["git_head_sha256"],
                "spec_sha256": candidate["spec_sha256"],
                "policy_sha256": policy_sha256,
                "output_manifest_sha256": output_sha256,
                "argv": command,
                "argv_sha256": job["argv_sha256"],
                "adapter_id": worker_config["adapter_id"],
                "model_id": worker_config["model_id"],
                "config_sha256": worker_config["config_sha256"],
                "producer_id": worker_config["producer_id"],
                "source_capture_manifest_sha256": source_capture_sha256,
                "environment_allowlist": ["PATH", "PYTHONPATH"],
                "output_directory": str(replay_root.relative_to(root)),
                "screen_name": "qr-dojo-improve-replay-worker-test",
                "pid": os.getpid(),
                "process_command_sha256": _canonical_sha256(command),
            },
        },
        recorded_at_utc=OPEN_AT,
    )
    return {
        "root": root,
        "candidate": candidate,
        "candidate_id": candidate_id,
        "config_path": config_path,
        "private_key_path": private_key_path,
        "worker_config": worker_config,
        "replay_root": replay_root,
        "source_bindings": source_bindings,
    }


def _fake_run_for(
    fixture: dict[str, object],
    *,
    forced_close: bool = False,
):
    class CompletedProcess:
        pid = 424_241
        returncode = 0

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

    def fake_run(command: list[str], **_: object) -> CompletedProcess:
        run_root = Path(command[command.index("--output-root") + 1])
        candidate = fixture["candidate"]
        source_bindings = fixture["source_bindings"]
        assert isinstance(candidate, dict)
        assert isinstance(source_bindings, dict)
        for window in ("TRAIN", "VAL", "S5"):
            for policy in ("BASELINE", "CANDIDATE"):
                for cost in ("BASE", "STRESS"):
                    for intrabar in ("OHLC", "OLHC"):
                        _write_arm(
                            run_root
                            / window.lower()
                            / policy.lower()
                            / cost.lower()
                            / intrabar.lower(),
                            candidate=candidate,
                            window=window,
                            policy=policy,
                            cost=cost,
                            intrabar=intrabar,
                            source_sha256=source_bindings[window],
                            forced_close=forced_close and policy == "CANDIDATE",
                        )
        return CompletedProcess()

    return fake_run


def _run(fixture: dict[str, object]) -> dict[str, object]:
    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=OPEN_AT),
        patch(
            "quant_rabbit.dojo_replay_worker._POPEN",
            side_effect=_fake_run_for(fixture),
        ),
    ):
        return run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=fixture["private_key_path"],
        )


def test_mock_key_e2e_recomputes_ledgers_and_signs_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = _run(fixture)

    assert result["train_selected"] is True
    assert result["paper_only"] is True
    assert result["order_authority"] == "NONE"
    assert result["live_permission"] is False
    assert result["paper_room_launched"] is False
    receipt_path = fixture["replay_root"] / "worker_receipt.json"
    with patch(
        "quant_rabbit.dojo_replay_worker_receipt._TRUSTED_REPLAY_WORKERS",
        {fixture["worker_config"]["adapter_id"]: fixture["worker_config"]},
    ):
        receipt = verify_trusted_replay_worker_receipt(
            fixture["root"],
            receipt_path,
        )
    assert receipt["receipt_sha256"] == result["worker_receipt_sha256"]
    artifact = json.loads(
        (fixture["replay_root"] / "proof_artifact.json").read_text(encoding="utf-8")
    )
    assert len(artifact["arms"]) == 24
    assert (
        min(
            arm["metrics"]["profit_factor"]
            for arm in artifact["arms"]
            if arm["window"] in {"VAL", "S5"}
            and arm["policy"] == "CANDIDATE"
            and arm["cost"] == "STRESS"
        )
        >= 1.25
    )
    assert set(inspect.signature(run_trusted_replay_worker).parameters) == {
        "repository_root",
        "candidate_id",
        "worker_config_path",
        "private_key_path",
    }


def test_weekend_stops_before_replay_subprocess(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    saturday = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)

    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=saturday),
        patch("quant_rabbit.dojo_replay_worker._POPEN") as run_command,
        pytest.raises(
            TrustedReplayWorkerMarketClosed,
            match="disabled while FX is closed",
        ),
    ):
        run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=fixture["private_key_path"],
        )

    run_command.assert_not_called()
    assert not (fixture["replay_root"] / "worker_runs").exists()


def test_external_key_must_be_0600_and_outside_repository(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    key_path = fixture["private_key_path"]
    assert isinstance(key_path, Path)
    key_path.chmod(0o644)
    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=OPEN_AT),
        patch("quant_rabbit.dojo_replay_worker._POPEN") as run_command,
        pytest.raises(TrustedReplayWorkerError, match="mode must be 0600"),
    ):
        run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=key_path,
        )
    run_command.assert_not_called()

    key_path.chmod(0o600)
    inside = fixture["root"] / "inside.key"
    inside.write_bytes(key_path.read_bytes())
    inside.chmod(0o600)
    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=OPEN_AT),
        patch("quant_rabbit.dojo_replay_worker._POPEN") as run_command,
        pytest.raises(TrustedReplayWorkerError, match="outside repository"),
    ):
        run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=inside,
        )
    run_command.assert_not_called()


def test_forced_end_close_and_source_tamper_fail_closed(tmp_path: Path) -> None:
    forced_fixture = _fixture(tmp_path / "forced")
    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=OPEN_AT),
        patch(
            "quant_rabbit.dojo_replay_worker._POPEN",
            side_effect=_fake_run_for(forced_fixture, forced_close=True),
        ),
        pytest.raises(TrustedReplayWorkerError, match="forced-close"),
    ):
        run_trusted_replay_worker(
            forced_fixture["root"],
            candidate_id=forced_fixture["candidate_id"],
            worker_config_path=forced_fixture["config_path"],
            private_key_path=forced_fixture["private_key_path"],
        )

    tampered_fixture = _fixture(tmp_path / "tampered")
    source = tampered_fixture["root"] / "research/data/replay_worker_sources/TRAIN.csv"
    source.write_bytes(source.read_bytes() + b"tampered\n")
    with (
        patch("quant_rabbit.dojo_replay_worker._UTC_NOW", return_value=OPEN_AT),
        patch("quant_rabbit.dojo_replay_worker._POPEN") as run_command,
        pytest.raises(ValueError, match="source file bytes digest mismatch"),
    ):
        run_trusted_replay_worker(
            tampered_fixture["root"],
            candidate_id=tampered_fixture["candidate_id"],
            worker_config_path=tampered_fixture["config_path"],
            private_key_path=tampered_fixture["private_key_path"],
        )
    run_command.assert_not_called()


def test_crossing_weekend_terminates_process_group_and_records_failure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    start_at = datetime(2026, 7, 31, 20, 59, tzinfo=timezone.utc)
    closed_at = datetime(2026, 7, 31, 21, 1, tzinfo=timezone.utc)

    class RunningProcess:
        pid = 424_242
        returncode: int | None = None

        def wait(self, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("replay", timeout)
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    process = RunningProcess()

    def signal_group(pid: int, sent_signal: int) -> None:
        assert pid == process.pid
        if sent_signal == signal.SIGTERM:
            process.returncode = -signal.SIGTERM

    with (
        patch(
            "quant_rabbit.dojo_replay_worker._UTC_NOW",
            side_effect=[start_at, closed_at],
        ),
        patch("quant_rabbit.dojo_replay_worker._POPEN", return_value=process),
        patch(
            "quant_rabbit.dojo_replay_worker.os.killpg",
            side_effect=signal_group,
        ) as kill_group,
        pytest.raises(
            TrustedReplayWorkerMarketClosed,
            match="disabled while FX is closed",
        ),
    ):
        run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=fixture["private_key_path"],
        )

    kill_group.assert_called_once_with(process.pid, signal.SIGTERM)
    assert not (fixture["replay_root"] / "proof_artifact.json").exists()
    assert not (fixture["replay_root"] / "worker_receipt.json").exists()
    failures = list((fixture["replay_root"] / "failure_receipts").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["failure_code"] == "MARKET_CLOSED"
    assert failure["partial_output_proof_eligible"] is False
    assert failure["proof_artifact_written"] is False
    assert failure["worker_receipt_written"] is False
    rows = [
        json.loads(line)
        for line in (
            fixture["root"]
            / CANONICAL_RESEARCH_RELATIVE_ROOT
            / "candidate_ledger.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert rows[-1]["event_type"] == "REPLAY_FAILED"
    assert rows[-1]["payload"]["failure_code"] == "MEASUREMENT"
    assert (
        rows[-1]["payload"]["artifact_sha256"]
        == hashlib.sha256(failures[0].read_bytes()).hexdigest()
    )


def test_term_timeout_escalates_to_bounded_kill() -> None:
    class StubbornProcess:
        pid = 424_243
        returncode: int | None = None
        waits = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("replay", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    process = StubbornProcess()
    with patch("quant_rabbit.dojo_replay_worker.os.killpg") as kill_group:
        result = _terminate_replay_process(process)

    assert kill_group.call_args_list == [
        ((process.pid, signal.SIGTERM),),
        ((process.pid, signal.SIGKILL),),
    ]
    assert result == {
        "term_sent": True,
        "kill_sent": True,
        "returncode": -signal.SIGKILL,
    }


def test_market_close_during_proof_read_never_uses_partial_result(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    start_at = datetime(2026, 7, 31, 20, 59, tzinfo=timezone.utc)
    closed_at = datetime(2026, 7, 31, 21, 1, tzinfo=timezone.utc)
    with (
        patch(
            "quant_rabbit.dojo_replay_worker._UTC_NOW",
            side_effect=[start_at, start_at, closed_at, closed_at],
        ),
        patch(
            "quant_rabbit.dojo_replay_worker._POPEN",
            side_effect=_fake_run_for(fixture),
        ),
        pytest.raises(
            TrustedReplayWorkerMarketClosed,
            match="proof arm evaluation",
        ),
    ):
        run_trusted_replay_worker(
            fixture["root"],
            candidate_id=fixture["candidate_id"],
            worker_config_path=fixture["config_path"],
            private_key_path=fixture["private_key_path"],
        )

    assert not (fixture["replay_root"] / "proof_artifact.json").exists()
    assert not (fixture["replay_root"] / "worker_receipt.json").exists()
    failure_paths = list((fixture["replay_root"] / "failure_receipts").glob("*.json"))
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="utf-8"))
    assert failure["partial_output_proof_eligible"] is False
