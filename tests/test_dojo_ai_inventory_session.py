from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_ai_inventory_session import (
    DRAIN_BROKER_RESTART_DIRECTORY_NAME,
    DRAIN_BROKER_START_RECEIPT_CONTRACT,
    SESSION_CONFIG_CONTRACT,
    SESSION_CONTRACT_NAME,
    SESSION_LIFECYCLE_NAME,
    SESSION_OWNER_LOCK_NAME,
    AIInventorySessionBusyError,
    AIInventorySessionConfigError,
    AIInventorySessionDependencies,
    AIInventorySessionIntegrityError,
    AIInventorySessionUnavailableError,
    ai_inventory_session_config_from_mapping,
    load_ai_inventory_session_config,
    run_ai_inventory_session,
    run_registered_ai_inventory_session,
    session_config_sha256,
    verify_drain_broker_restart_authorization,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT,
)


SAFETY = {
    "paper_only": True,
    "order_authority": "NONE",
    "live_permission": False,
    "external_broker_mutation_allowed": False,
}
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> datetime:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class FakeRuntime:
    def __init__(
        self,
        config: Any,
        *,
        clock: SequenceClock,
        positions: int = 0,
        orders: int = 0,
        force_close: bool = False,
        due_evaluation: bool = False,
        controller_closes_inventory: bool = False,
        kill_broker_after_controller: bool = False,
        kill_bot_after_controller: bool = False,
        keep_drain_inventory: bool = False,
        kill_drain_broker_after_tick: bool = False,
        recovery_wal: bool = False,
        recovery_wal_expected_count: int = 2,
        recovery_wal_applied_count: int = 1,
    ) -> None:
        self.config = config
        self.clock = clock
        self.positions = positions
        self.orders = orders
        self.force_close = force_close
        self.due_evaluation = due_evaluation
        self.controller_closes_inventory = controller_closes_inventory
        self.kill_broker_after_controller = kill_broker_after_controller
        self.kill_bot_after_controller = kill_bot_after_controller
        self.keep_drain_inventory = keep_drain_inventory
        self.kill_drain_broker_after_tick = kill_drain_broker_after_tick
        self.recovery_wal = recovery_wal
        self.recovery_wal_expected_count = recovery_wal_expected_count
        self.recovery_wal_applied_count = recovery_wal_applied_count
        self.new_entries_allowed = False
        self.broker_alive = False
        self.broker_mode = "ACTIVE"
        self.bot_alive = False
        self.last_drain_receipt_sha256: str | None = None
        self.calls: Counter[str] = Counter()

    def dependencies(self) -> AIInventorySessionDependencies:
        return AIInventorySessionDependencies(
            dependency_id=self.config.dependency_id,
            clock=self.clock,
            sleep=self.sleep,
            verify_launch_preflight=self.verify_preflight,
            start_broker=self.start_broker,
            inspect_drain_checkpoint=self.inspect_drain_checkpoint,
            start_drain_broker=self.start_drain_broker,
            stop_broker=self.stop_broker,
            broker_health=self.broker_health,
            start_bot=self.start_bot,
            stop_bot=self.stop_bot,
            bot_health=self.bot_health,
            capture_source=self.capture,
            apply_captured_quote=self.apply_quote,
            capture_drain_quote=self.capture_drain_quote,
            apply_captured_drain_quote=self.apply_drain_quote,
            inspect_broker=self.inspect,
            build_evidence_request=self.build_request,
            run_controller=self.controller,
            evaluation_plan=self.evaluation_plan,
            evaluate=self.evaluate,
            drain_tick=self.drain,
        )

    def sleep(self, _seconds: float) -> None:
        self.calls["sleep"] += 1

    def verify_preflight(
        self, _repository_root: Path, experiment_id: str, room_id: str
    ) -> dict[str, Any]:
        self.calls["preflight"] += 1
        return {
            "experiment_id": experiment_id,
            "room_id": room_id,
            "candidate_id": self.config.candidate_id,
            "adapter_id": self.config.adapter_id,
            "model_id": self.config.model_id,
            "config_sha256": self.config.model_config_sha256,
            "producer_id": self.config.producer_id,
            "launch_preflight_token_sha256": (
                self.config.launch_preflight_token_sha256
            ),
            "future_window": {
                "start_utc": self.config.window_start_utc,
                "end_utc": self.config.window_end_utc,
            },
            "paper_eligible_event_sha256": SHA_E,
            "future_registry_sha256": SHA_F,
            "paper_room_launched": False,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }

    def start_broker(self, _context: Any) -> dict[str, Any]:
        self.calls["start_broker"] += 1
        self.broker_alive = True
        self.broker_mode = "ACTIVE"
        return {
            "owner_count": 1,
            "alive": True,
            "mode": "ACTIVE",
            "checkpoint_reconciled": True,
            "process_identity": {"pid": 1_001, "argv_sha256": SHA_A},
            "candidate_id": self.config.candidate_id,
            "room_id": self.config.room_id,
            **SAFETY,
        }

    def inspect_drain_checkpoint(self, _context: Any) -> dict[str, Any]:
        self.calls["inspect_drain_checkpoint"] += 1
        return {
            "experiment_id": self.config.experiment_id,
            "room_id": self.config.room_id,
            "candidate_id": self.config.candidate_id,
            "checkpoint_reconciled": True,
            "broker_ledger_terminal_sha256": (
                SHA_F
                if self.recovery_wal
                and self.recovery_wal_applied_count > 0
                else SHA_B
            ),
            "broker_snapshot_sha256": SHA_C,
            "broker_snapshot_ledger_terminal_sha256": SHA_B,
            "broker_recovery_wal_sha256": (
                SHA_D if self.recovery_wal else None
            ),
            "broker_recovery_wal_checkpoint_ledger_sha256": (
                SHA_B if self.recovery_wal else None
            ),
            "broker_recovery_wal_expected_event_count": (
                self.recovery_wal_expected_count if self.recovery_wal else 0
            ),
            "broker_recovery_wal_applied_event_count": (
                self.recovery_wal_applied_count if self.recovery_wal else 0
            ),
            "broker_recovery_wal_validated": self.recovery_wal,
            "positions_count": self.positions,
            "orders_count": self.orders,
            "new_entries_allowed": False,
            "ai_decision_allowed": False,
            "force_close_allowed": False,
            **SAFETY,
        }

    def start_drain_broker(
        self, context: Any, authorization_path: Path
    ) -> dict[str, Any]:
        self.calls["start_drain_broker"] += 1
        authorization = json.loads(authorization_path.read_text())
        verified = verify_drain_broker_restart_authorization(
            repository_root=self.config.repository_root,
            room_root=context.room_root,
            authorization_path=authorization_path,
            experiment_id=self.config.experiment_id,
            room_id=self.config.room_id,
            candidate_id=self.config.candidate_id,
            broker_ledger_terminal_sha256=authorization[
                "broker_ledger_terminal_sha256"
            ],
            broker_snapshot_sha256=authorization[
                "broker_snapshot_sha256"
            ],
            broker_snapshot_ledger_terminal_sha256=authorization[
                "broker_snapshot_ledger_terminal_sha256"
            ],
            positions_count=self.positions,
            orders_count=self.orders,
            broker_recovery_wal_sha256=authorization[
                "broker_recovery_wal_sha256"
            ],
            broker_recovery_wal_checkpoint_ledger_sha256=authorization[
                "broker_recovery_wal_checkpoint_ledger_sha256"
            ],
            broker_recovery_wal_expected_event_count=authorization[
                "broker_recovery_wal_expected_event_count"
            ],
            broker_recovery_wal_applied_event_count=authorization[
                "broker_recovery_wal_applied_event_count"
            ],
            broker_recovery_wal_validated=authorization[
                "broker_recovery_wal_validated"
            ],
            balance_jpy=self.config.balance_jpy,
            slippage_pips=self.config.slippage_pips,
            financing_pips_per_day=self.config.financing_pips_per_day,
            leverage=self.config.leverage,
            original_ceiling_minutes=self.config.original_ceiling_minutes,
        )
        self.calls["verify_drain_authorization"] += 1
        self.broker_alive = True
        self.broker_mode = "DRAIN_ONLY"
        assert verified == authorization
        body = {
            "contract": DRAIN_BROKER_START_RECEIPT_CONTRACT,
            "experiment_id": self.config.experiment_id,
            "room_id": self.config.room_id,
            "candidate_id": self.config.candidate_id,
            "mode": "DRAIN_ONLY",
            "owner_count": 1,
            "alive": True,
            "started_at_utc": "2026-07-27T00:10:01Z",
            "process_identity": {"pid": 1_002, "argv_sha256": SHA_A},
            "authorization_receipt_sha256": authorization["receipt_sha256"],
            "authorization_file_sha256": hashlib.sha256(
                authorization_path.read_bytes()
            ).hexdigest(),
            "authorization_path_relative": str(
                authorization_path.relative_to(context.room_root)
            ),
            "new_entries_allowed": False,
            "ai_decision_allowed": False,
            "force_close_allowed": False,
            "bot_rpc_commands": [],
            "runner_rpc_commands": [
                "ACCOUNT",
                "APPLY_DRAIN_QUOTE",
                "HEALTH",
                "ORDERS",
                "POSITIONS",
                "QUOTE_PROVENANCE",
                "QUOTES",
                "SHUTDOWN",
            ],
            **SAFETY,
        }
        return {**body, "start_receipt_sha256": _semantic_sha(body)}

    def stop_broker(self, _context: Any) -> None:
        self.calls["stop_broker"] += 1
        self.broker_alive = False

    def broker_health(self, _context: Any) -> dict[str, Any]:
        self.calls["broker_health"] += 1
        return {
            "role": "broker",
            "mode": self.broker_mode,
            "room_id": self.config.room_id,
            "candidate_id": self.config.candidate_id,
            "alive": self.broker_alive,
            "owner_count": 1 if self.broker_alive else 0,
            "process_identity": (
                {"pid": 1_002, "argv_sha256": SHA_A}
                if self.broker_alive
                else None
            ),
            **SAFETY,
        }

    def start_bot(self, _context: Any) -> dict[str, Any]:
        self.calls["start_bot"] += 1
        self.bot_alive = True
        self.new_entries_allowed = True
        return {
            "bot_process_count": 1,
            "alive": True,
            "mode": "ACTIVE",
            "process_identity": {"pid": 2_001, "argv_sha256": SHA_B},
            "room_id": self.config.room_id,
            "new_entries_allowed": True,
            **SAFETY,
        }

    def stop_bot(self, _context: Any) -> dict[str, Any]:
        self.calls["stop_bot"] += 1
        self.bot_alive = False
        self.new_entries_allowed = False
        return {
            "bot_process_count": 0,
            "room_id": self.config.room_id,
            "new_entries_allowed": False,
            **SAFETY,
        }

    def bot_health(self, _context: Any) -> dict[str, Any]:
        self.calls["bot_health"] += 1
        return {
            "role": "bot",
            "mode": "ACTIVE",
            "room_id": self.config.room_id,
            "candidate_id": self.config.candidate_id,
            "alive": self.bot_alive,
            "process_count": 1 if self.bot_alive else 0,
            "process_identity": (
                {"pid": 2_001, "argv_sha256": SHA_B}
                if self.bot_alive
                else None
            ),
            **SAFETY,
        }
    def capture(self, _context: Any, role: str, cutoff: str) -> dict[str, Any]:
        self.calls[f"capture_{role}"] += 1
        return {
            "source_role": role,
            "receipt_sha256": SHA_A if role == "quote" else SHA_B,
            "cutoff_utc": cutoff,
        }

    def apply_quote(self, _context: Any, receipt_sha256: str) -> list[dict[str, Any]]:
        self.calls["apply_quote"] += 1
        assert receipt_sha256 == SHA_A
        return []

    def capture_drain_quote(self, context: Any, cutoff: str) -> dict[str, Any]:
        self.calls["capture_drain_quote"] += 1
        state = json.loads((context.room_root / "session_state.json").read_text())
        body = {
            "contract": "QR_DOJO_AI_DRAIN_QUOTE_RECEIPT_V1",
            "sequence": 1,
            "previous_receipt_sha256": "0" * 64,
            "experiment_id": self.config.experiment_id,
            "room_id": self.config.room_id,
            "candidate_id": self.config.candidate_id,
            "source_role": "quote",
            "cutoff_utc": cutoff,
            "fixed_window_end_utc": self.config.window_end_utc,
            "session_contract_file_sha256": hashlib.sha256(
                (context.room_root / "session_contract.json").read_bytes()
            ).hexdigest(),
            "session_lifecycle_tip_sha256": state["lifecycle_tip_sha256"],
            "session_state_file_sha256": hashlib.sha256(
                (context.room_root / "session_state.json").read_bytes()
            ).hexdigest(),
            "broker_ledger_terminal_sha256": SHA_B,
            "broker_snapshot_sha256": SHA_C,
            "broker_snapshot_ledger_terminal_sha256": SHA_B,
            "positions_count": self.positions,
            "orders_count": self.orders,
            "allowed_drain_resolutions": [
                "MARGIN_CLOSEOUT",
                "ORIGINAL_CEILING",
                "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                "SL",
                "TP",
            ],
            "canonical_source_sha256": SHA_D,
            "raw_source_bytes_sha256": SHA_D,
            "source_watermark_sha256": SHA_E,
            "provider_kind": "READ_ONLY_TEST",
            "adapter_id": "drain-quote-adapter-v1",
            "adapter_module": "test.adapter",
            "adapter_callable": "capture",
            "adapter_executable_sha256": SHA_A,
            "adapter_config_sha256": SHA_B,
            "provider_timestamp_utc": cutoff,
            "fetched_at_utc": cutoff,
            "capture_manifest_file_sha256": SHA_C,
            "capture_manifest_sha256": SHA_D,
            "capture_key_id": "test-key-v1",
            "drain_only": True,
            "new_entries_allowed": False,
            "ai_evaluation_allowed": False,
            "force_close_allowed": False,
            "original_ceiling_minutes": self.config.original_ceiling_minutes,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "external_broker_mutation_allowed": False,
        }
        assert context.config == self.config
        receipt_sha256 = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.last_drain_receipt_sha256 = receipt_sha256
        return {
            **body,
            "receipt_sha256": receipt_sha256,
            "signature_base64": base64.b64encode(b"\0" * 64).decode(),
        }

    def apply_drain_quote(
        self, _context: Any, receipt_sha256: str
    ) -> list[dict[str, Any]]:
        self.calls["apply_drain_quote"] += 1
        assert receipt_sha256 == self.last_drain_receipt_sha256
        return []

    def inspect(self, _context: Any) -> dict[str, Any]:
        self.calls["inspect"] += 1
        return {
            "positions_count": self.positions,
            "orders_count": self.orders,
            "new_entries_allowed": self.new_entries_allowed,
            "balance_jpy": 200_000.0,
            "equity_jpy": 200_000.0,
            "margin_used_jpy": 0.0,
            **SAFETY,
        }

    def build_request(
        self,
        _context: Any,
        _captures: Any,
        _broker: Any,
        _cutoff: str,
    ) -> dict[str, Any]:
        self.calls["build_request"] += 1
        return {"evidence": "trusted", **SAFETY}

    def controller(self, _context: Any, _request: Any) -> dict[str, Any]:
        self.calls["controller"] += 1
        if self.controller_closes_inventory:
            self.positions = 0
            self.new_entries_allowed = False
        if self.kill_broker_after_controller:
            self.broker_alive = False
        if self.kill_bot_after_controller:
            self.bot_alive = False
        return {
            "room_id": self.config.room_id,
            "decision_sha256": SHA_C,
            "applied_receipt_sha256": SHA_D,
            **SAFETY,
        }

    def evaluation_plan(self, _context: Any, _now: str) -> dict[str, Any]:
        self.calls["evaluation_plan"] += 1
        if self.due_evaluation:
            return {
                "due": [
                    {
                        "decision_sha256": SHA_C,
                        "horizon_end_at_utc": "2026-07-27T00:00:30Z",
                        "outcome_kind": "FIXED_HORIZON",
                    }
                ],
                "pending_count": 1,
            }
        return {"due": [], "pending_count": 0}

    def evaluate(self, _context: Any, item: Any) -> dict[str, Any]:
        self.calls["evaluate"] += 1
        self.due_evaluation = False
        return {"decision_sha256": item["decision_sha256"], **SAFETY}

    def drain(
        self, _context: Any, _quote_receipt_sha: str, _now: str
    ) -> dict[str, Any]:
        self.calls["drain"] += 1
        if not self.force_close and not self.keep_drain_inventory:
            self.positions = 0
            self.orders = 0
        if self.kill_drain_broker_after_tick:
            self.broker_alive = False
        return {
            "new_entries_allowed": False,
            "force_close": self.force_close,
            "drain_quote_receipt_sha256": self.last_drain_receipt_sha256,
            "original_ceiling_minutes": (self.config.original_ceiling_minutes),
            "resolutions": ["ORIGINAL_CEILING"],
            **SAFETY,
        }


def _mapping(
    *,
    start: str = "2026-07-27T00:00:00Z",
    end: str = "2026-07-27T00:10:00Z",
) -> dict[str, Any]:
    value = {
        "contract": SESSION_CONFIG_CONTRACT,
        "experiment_id": "paper-ai-inventory-experiment-v1",
        "room_id": "paper-ai-inventory-room-v1",
        "candidate_id": SHA_A,
        "dependency_id": "paper-ai-inventory-dependency-v1",
        "pair": "USD_JPY",
        "window_start_utc": start,
        "window_end_utc": end,
        "adapter_id": "sealed-adapter-v1",
        "model_id": "sealed-model-v1",
        "model_config_sha256": SHA_B,
        "producer_id": "producer-v1",
        "bot_config_sha256": SHA_C,
        "balance_jpy": 200_000.0,
        "slippage_pips": 0.3,
        "financing_pips_per_day": 0.8,
        "leverage": 2.0,
        "original_ceiling_minutes": 60,
        "cycle_interval_seconds": 45,
        "drain_interval_seconds": 30,
        "capture_deadline_seconds": 30,
        "evaluation_horizon_seconds": 3_600,
        "launch_preflight_token_sha256": SHA_D,
        **SAFETY,
    }
    value["session_config_sha256"] = session_config_sha256(value)
    return value


def _repo(tmp_path: Path, mapping: dict[str, Any]) -> tuple[Path, Path]:
    repository_root = tmp_path / "repo"
    room_root = (
        repository_root
        / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT
        / mapping["experiment_id"]
        / mapping["room_id"]
    )
    room_root.mkdir(parents=True)
    return repository_root, room_root


def _run(
    config: Any,
    runtime: FakeRuntime,
    *,
    max_iterations: int,
) -> Any:
    return run_ai_inventory_session(
        config,
        runtime.dependencies(),
        screen_identity=f"123.qr-dojo-{config.room_id}",
        process_argv=("python3.12", "scripts/run-dojo-ai-inventory-room.py"),
        max_iterations=max_iterations,
    )


def test_config_is_strict_content_addressed_and_repository_owned(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, _ = _repo(tmp_path, mapping)
    config_dir = repository_root / "config/paper_ai_inventory"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "room.json"
    config_path.write_text(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
    )

    loaded = load_ai_inventory_session_config(repository_root, config_path)
    assert loaded.session_config_sha256 == mapping["session_config_sha256"]

    altered = dict(mapping)
    altered["leverage"] = 3.0
    with pytest.raises(AIInventorySessionConfigError, match="digest mismatch"):
        ai_inventory_session_config_from_mapping(repository_root, altered)

    outside = repository_root / "outside.json"
    outside.write_text(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(AIInventorySessionConfigError, match="direct JSON child"):
        load_ai_inventory_session_config(repository_root, outside)


def test_config_preserves_future_registry_utc_spelling(tmp_path: Path) -> None:
    mapping = _mapping(
        start="2026-07-27T00:00:00+00:00",
        end="2026-07-27T00:10:00+00:00",
    )
    repository_root, _ = _repo(tmp_path, mapping)

    config = ai_inventory_session_config_from_mapping(repository_root, mapping)

    assert config.window_start_utc == mapping["window_start_utc"]
    assert config.window_end_utc == mapping["window_end_utc"]
    assert config.window_start == datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)


def test_drain_authorization_rejects_room_symlink_escape(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root = tmp_path / "repo"
    room_parent = (
        repository_root
        / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT
        / mapping["experiment_id"]
    )
    room_parent.mkdir(parents=True)
    escaped = tmp_path / "escaped-room"
    escaped.mkdir()
    expected_room = room_parent / mapping["room_id"]
    expected_room.symlink_to(escaped, target_is_directory=True)

    with pytest.raises(
        AIInventorySessionIntegrityError,
        match="outside the canonical repository root",
    ):
        verify_drain_broker_restart_authorization(
            repository_root=repository_root,
            room_root=escaped,
            authorization_path=escaped / "missing.json",
            experiment_id=mapping["experiment_id"],
            room_id=mapping["room_id"],
            candidate_id=mapping["candidate_id"],
            broker_ledger_terminal_sha256=SHA_A,
            broker_snapshot_sha256=SHA_B,
            broker_snapshot_ledger_terminal_sha256=SHA_A,
            positions_count=0,
            orders_count=0,
            broker_recovery_wal_sha256=None,
            broker_recovery_wal_checkpoint_ledger_sha256=None,
            broker_recovery_wal_expected_event_count=0,
            broker_recovery_wal_applied_event_count=0,
            broker_recovery_wal_validated=False,
            balance_jpy=200_000.0,
            slippage_pips=0.3,
            financing_pips_per_day=0.8,
            leverage=2.0,
            original_ceiling_minutes=60,
        )


def test_unregistered_production_dependency_fails_before_room_mutation(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)

    with pytest.raises(AIInventorySessionUnavailableError):
        run_registered_ai_inventory_session(
            config,
            screen_identity=f"qr-dojo-{config.room_id}",
            process_argv=("runner",),
        )

    assert not (room_root / SESSION_CONTRACT_NAME).exists()
    assert not (room_root / SESSION_OWNER_LOCK_NAME).exists()


def test_invalid_paper_eligible_preflight_fails_before_room_mutation(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, tzinfo=timezone.utc)),
    )
    good = runtime.verify_preflight

    def bad(root: Path, experiment_id: str, room_id: str) -> dict[str, Any]:
        value = good(root, experiment_id, room_id)
        value["paper_eligible_event_sha256"] = "missing"
        return value

    dependencies = runtime.dependencies()
    dependencies = AIInventorySessionDependencies(
        **{
            **dependencies.__dict__,
            "verify_launch_preflight": bad,
        }
    )
    with pytest.raises(AIInventorySessionIntegrityError):
        run_ai_inventory_session(
            config,
            dependencies,
            screen_identity=f"qr-dojo-{config.room_id}",
            process_argv=("runner",),
            max_iterations=1,
        )

    assert not (room_root / SESSION_CONTRACT_NAME).exists()
    assert not (room_root / SESSION_OWNER_LOCK_NAME).exists()


@pytest.mark.parametrize(
    ("now", "expected_status"),
    [
        (datetime(2026, 7, 26, 23, 0, tzinfo=timezone.utc), "WAITING"),
        (datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc), "WEEKEND_PAUSED"),
    ],
)
def test_prewindow_and_weekend_call_no_external_or_ai_hooks(
    tmp_path: Path, now: datetime, expected_status: str
) -> None:
    start = (
        "2026-07-27T00:00:00Z"
        if expected_status == "WAITING"
        else "2026-07-25T00:00:00Z"
    )
    end = (
        "2026-07-27T00:10:00Z"
        if expected_status == "WAITING"
        else "2026-07-26T20:00:00Z"
    )
    mapping = _mapping(start=start, end=end)
    repository_root, _ = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(config, clock=SequenceClock(now))

    result = _run(config, runtime, max_iterations=1)

    assert result.status == expected_status
    prohibited = {
        "start_broker",
        "start_bot",
        "capture_quote",
        "capture_candles",
        "apply_quote",
        "capture_drain_quote",
        "apply_drain_quote",
        "build_request",
        "controller",
        "evaluation_plan",
        "evaluate",
        "drain",
    }
    assert not (prohibited & set(runtime.calls))


def test_active_cycle_then_fixed_window_drains_and_seals(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc),
        ),
        positions=1,
        orders=1,
    )

    result = _run(config, runtime, max_iterations=3)

    assert result.status == "SEALED"
    assert result.positions_count == 0
    assert result.orders_count == 0
    assert runtime.calls["start_broker"] == 1
    assert runtime.calls["start_bot"] == 1
    assert runtime.calls["controller"] == 1
    assert runtime.calls["capture_candles"] == 1
    assert runtime.calls["capture_drain_quote"] == 1
    assert runtime.calls["apply_drain_quote"] == 1
    assert runtime.calls["drain"] == 1
    assert runtime.calls["stop_bot"] == 1
    lifecycle = [
        json.loads(line)
        for line in (room_root / SESSION_LIFECYCLE_NAME).read_text().splitlines()
    ]
    assert [row["event"] for row in lifecycle] == [
        "SESSION_START",
        "ENTRY_STOP",
        "SESSION_STOP",
    ]
    assert lifecycle[-1]["payload"]["force_close"] is False


def test_due_evaluation_runs_only_after_active_controller_cycle(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, _ = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc)),
        due_evaluation=True,
    )

    result = _run(config, runtime, max_iterations=1)

    assert result.status == "ACTIVE"
    assert runtime.calls["controller"] == 1
    assert runtime.calls["evaluate"] == 1
    assert runtime.calls["evaluation_plan"] == 2


def test_applied_ai_virtual_close_and_block_new_are_reflected_in_state(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc)),
        positions=1,
        controller_closes_inventory=True,
    )

    result = _run(config, runtime, max_iterations=1)

    state = json.loads((room_root / "session_state.json").read_text())
    assert result.status == "ACTIVE"
    assert result.positions_count == 0
    assert state["positions_count"] == 0
    assert state["new_entries_allowed"] is False
    assert runtime.calls["inspect"] == 2


def test_crossing_weekend_stops_bot_and_runs_no_second_ai_evaluation(
    tmp_path: Path,
) -> None:
    mapping = _mapping(
        start="2026-07-24T20:50:00Z",
        end="2026-07-26T22:30:00Z",
    )
    repository_root, _ = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 24, 20, 59, tzinfo=timezone.utc),
            datetime(2026, 7, 24, 21, 1, tzinfo=timezone.utc),
        ),
    )

    result = _run(config, runtime, max_iterations=2)

    assert result.status == "WEEKEND_PAUSED"
    assert runtime.calls["controller"] == 1
    assert runtime.calls["capture_quote"] == 1
    assert runtime.calls["capture_candles"] == 1
    assert runtime.calls["evaluation_plan"] == 2
    assert runtime.calls["stop_bot"] == 1
    assert runtime.calls["capture_drain_quote"] == 0
    assert runtime.calls["drain"] == 0


def test_active_broker_death_restarts_only_from_reconciled_checkpoint(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, _ = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 27, 0, 2, tzinfo=timezone.utc),
        ),
        kill_broker_after_controller=True,
    )

    result = _run(config, runtime, max_iterations=2)

    assert result.status == "ACTIVE"
    assert runtime.calls["start_broker"] == 2
    assert runtime.calls["broker_health"] == 1
    assert runtime.calls["start_bot"] == 1
    assert runtime.calls["controller"] == 2


def test_active_bot_death_revokes_entry_and_never_restarts_bot(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 27, 0, 2, tzinfo=timezone.utc),
        ),
        kill_bot_after_controller=True,
    )

    result = _run(config, runtime, max_iterations=2)

    assert result.status == "ENTRY_STOPPED_WAITING"
    assert runtime.calls["start_bot"] == 1
    assert runtime.calls["bot_health"] == 1
    assert runtime.calls["controller"] == 1
    lifecycle = [
        json.loads(line)
        for line in (room_root / SESSION_LIFECYCLE_NAME).read_text().splitlines()
    ]
    assert lifecycle[-1]["event"] == "ENTRY_STOP"
    assert lifecycle[-1]["payload"]["reason"] == "BOT_PROCESS_DIED"


def test_drain_broker_death_requires_next_signed_authorization(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc),
            datetime(2026, 7, 27, 0, 12, tzinfo=timezone.utc),
        ),
        positions=1,
        keep_drain_inventory=True,
        kill_drain_broker_after_tick=True,
    )

    result = _run(config, runtime, max_iterations=2)

    assert result.status == "DRAINING"
    assert runtime.calls["start_drain_broker"] == 2
    assert runtime.calls["verify_drain_authorization"] == 2
    assert runtime.calls["broker_health"] == 1
    receipts = sorted(
        (
            room_root / DRAIN_BROKER_RESTART_DIRECTORY_NAME
        ).glob("*.json")
    )
    assert len(receipts) == 2
    first, second = [json.loads(path.read_text()) for path in receipts]
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_receipt_sha256"] == first["receipt_sha256"]


def test_drain_restart_authorization_binds_validated_partial_wal(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc),
        ),
        positions=1,
        keep_drain_inventory=True,
        recovery_wal=True,
    )

    result = _run(config, runtime, max_iterations=1)

    assert result.status == "DRAINING"
    receipt_path = next(
        (
            room_root / DRAIN_BROKER_RESTART_DIRECTORY_NAME
        ).glob("*.json")
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["broker_ledger_terminal_sha256"] == SHA_F
    assert receipt["broker_snapshot_ledger_terminal_sha256"] == SHA_B
    assert receipt["broker_recovery_wal_sha256"] == SHA_D
    assert receipt["broker_recovery_wal_expected_event_count"] == 2
    assert receipt["broker_recovery_wal_applied_event_count"] == 1
    assert receipt["broker_recovery_wal_validated"] is True


def test_drain_restart_authorization_accepts_validated_zero_event_wal(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(
            datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc),
        ),
        positions=1,
        keep_drain_inventory=True,
        recovery_wal=True,
        recovery_wal_expected_count=0,
        recovery_wal_applied_count=0,
    )

    result = _run(config, runtime, max_iterations=1)

    assert result.status == "DRAINING"
    receipt_path = next(
        (
            room_root / DRAIN_BROKER_RESTART_DIRECTORY_NAME
        ).glob("*.json")
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["broker_ledger_terminal_sha256"] == SHA_B
    assert receipt["broker_snapshot_ledger_terminal_sha256"] == SHA_B
    assert receipt["broker_recovery_wal_sha256"] == SHA_D
    assert receipt["broker_recovery_wal_expected_event_count"] == 0
    assert receipt["broker_recovery_wal_applied_event_count"] == 0
    assert receipt["broker_recovery_wal_validated"] is True


def test_owner_restart_from_active_revokes_entry_and_does_not_relaunch_bot(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    first = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc)),
    )
    assert _run(config, first, max_iterations=1).status == "ACTIVE"

    resumed = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 2, tzinfo=timezone.utc)),
    )
    result = _run(config, resumed, max_iterations=1)

    assert result.status == "ENTRY_STOPPED_WAITING"
    assert resumed.calls["start_bot"] == 0
    assert resumed.calls["start_broker"] == 0
    assert resumed.calls["stop_bot"] == 1
    assert resumed.calls["stop_broker"] == 1
    lifecycle = [
        json.loads(line)
        for line in (room_root / SESSION_LIFECYCLE_NAME).read_text().splitlines()
    ]
    assert lifecycle[-1]["event"] == "ENTRY_STOP"
    assert (
        lifecycle[-1]["payload"]["reason"]
        == "SESSION_OWNER_RESTART_UNPROVEN_CHILDREN"
    )


def test_force_close_drain_receipt_fails_closed_and_never_seals(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc)),
        positions=1,
        force_close=True,
    )

    with pytest.raises(AIInventorySessionIntegrityError, match="force-close"):
        _run(config, runtime, max_iterations=1)

    lifecycle = [
        json.loads(line)
        for line in (room_root / SESSION_LIFECYCLE_NAME).read_text().splitlines()
    ]
    assert lifecycle[-1]["event"] == "SESSION_ERROR"
    assert all(row["event"] != "SESSION_STOP" for row in lifecycle)


def test_sealed_restart_is_idempotent_and_starts_no_children(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    first = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 11, tzinfo=timezone.utc)),
    )
    sealed = _run(config, first, max_iterations=1)
    assert sealed.status == "SEALED"
    before = {
        path.name: path.read_bytes()
        for path in room_root.iterdir()
        if path.is_file() and path.name != SESSION_OWNER_LOCK_NAME
    }

    second = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, 0, 12, tzinfo=timezone.utc)),
    )
    restored = _run(config, second, max_iterations=1)

    assert restored == sealed
    assert second.calls == Counter({"preflight": 1})
    after = {
        path.name: path.read_bytes()
        for path in room_root.iterdir()
        if path.is_file() and path.name != SESSION_OWNER_LOCK_NAME
    }
    assert after == before


def test_owner_lock_rejects_duplicate_process(tmp_path: Path) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, tzinfo=timezone.utc)),
    )
    lock_path = room_root / SESSION_OWNER_LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(AIInventorySessionBusyError):
            _run(config, runtime, max_iterations=1)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert not (room_root / SESSION_CONTRACT_NAME).exists()


def test_screen_identity_is_exact_and_checked_before_preflight(
    tmp_path: Path,
) -> None:
    mapping = _mapping()
    repository_root, room_root = _repo(tmp_path, mapping)
    config = ai_inventory_session_config_from_mapping(repository_root, mapping)
    runtime = FakeRuntime(
        config,
        clock=SequenceClock(datetime(2026, 7, 27, tzinfo=timezone.utc)),
    )

    with pytest.raises(AIInventorySessionConfigError, match="screen identity"):
        run_ai_inventory_session(
            config,
            runtime.dependencies(),
            screen_identity="qr-dojo-wrong-room",
            process_argv=("runner",),
            max_iterations=1,
        )

    assert runtime.calls["preflight"] == 0
    assert not (room_root / SESSION_CONTRACT_NAME).exists()
