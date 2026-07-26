from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from quant_rabbit.dojo_ai_evidence_packet import (
    DOJO_AI_EVIDENCE_PACKET_CONTRACT,
    entry_signal_identity_sha256,
    write_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
)
from quant_rabbit.dojo_ai_inventory_broker_service import (
    AIInventoryBrokerServiceError,
    BrokerServiceConfig,
    DojoAIInventoryEntryClient,
    DojoAIInventoryRunnerClient,
    _BrokerOwner,
    _TEST_ONLY_RAW_QUOTES_CAPABILITY,
    _validated_drain_recovery_binding,
    _write_broker_state,
    derive_broker_socket_path,
    serve_ai_inventory_broker,
)
from quant_rabbit.dojo_ai_inventory_runtime import (
    AIInventoryEntryDeniedError,
    ENTRY_ADMISSION_REFERENCE_CONTRACT,
)
from quant_rabbit.dojo_ai_inventory_session import (
    SESSION_CONTRACT_NAME,
    AIInventorySessionContext,
    _append_drain_broker_restart_receipt,
    _append_lifecycle,
    _build_session_contract,
    _write_immutable_contract,
    _write_state,
    ai_inventory_session_config_from_mapping,
    session_config_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT,
)
from quant_rabbit.virtual_broker import VirtualBroker


UTC = timezone.utc
ROOM_ID = "paper-ai-inventory-room-rpc-001"
CANDIDATE_ID = "candidate-rpc-001"
EXPERIMENT_ID = "paper-ai-inventory-experiment-rpc-001"
PAIR = "USD_JPY"
STRATEGY_TAG = "QR_DOJO_AI_INVENTORY_RPC_V1"
SIGNAL_AT = "2026-07-23T12:00:00Z"
APPLY_AT = datetime(2026, 7, 23, 12, 0, 2, tzinfo=UTC)
ENTRY_AT = datetime(2026, 7, 23, 12, 0, 5, tzinfo=UTC)
BOT_KEY = b"bot-key-" + b"b" * 40
RUNNER_KEY = b"runner-key-" + b"r" * 40


def _serve(config: BrokerServiceConfig) -> None:
    serve_ai_inventory_broker(config)


def _context() -> dict[str, object]:
    return {
        "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
        "strategy_tag": STRATEGY_TAG,
        "trend_24h": "UP",
        "change_24h": 0.1,
        "change_6h": 0.02,
        "efficiency_6h": 0.4,
        "atr": 0.08,
    }


def _context_sha() -> str:
    raw = json.dumps(
        _context(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _signal(order_type: str = "MARKET") -> dict[str, object]:
    body: dict[str, object] = {
        "pair": PAIR,
        "side": "LONG",
        "order_type": order_type,
        "units": 100.0,
        "price": None if order_type == "MARKET" else 163.0,
        "strategy_tag": STRATEGY_TAG,
        "entry_context_sha256": _context_sha(),
        "tp_pips": 6.0,
        "sl_pips": 25.0,
        "observed_at_utc": SIGNAL_AT,
    }
    return {
        **body,
        "signal_identity_sha256": entry_signal_identity_sha256(body),
    }


def _captured_quote_fixture(
    repository: Path,
    *,
    room_suffix: str,
    bid: float,
    ask: float,
    timestamp_utc: str,
) -> tuple[BrokerServiceConfig, str, dict[str, object], dict[str, object]]:
    experiment_id = f"paper-ai-inventory-experiment-{room_suffix}"
    room_id = f"paper-ai-inventory-room-{room_suffix}"
    candidate_id = hashlib.sha256(f"candidate:{room_suffix}".encode()).hexdigest()
    token_sha256 = hashlib.sha256(f"token:{room_suffix}".encode()).hexdigest()
    room = (
        repository
        / "research/data/dojo_paper_ai_inventory_v1/rooms"
        / experiment_id
        / room_id
    )
    room.mkdir(parents=True)
    source = {
        "ask": ask,
        "bid": bid,
        "max_age_seconds": 120,
        "pair": PAIR,
        "timestamp_utc": timestamp_utc,
    }
    source_raw = json.dumps(
        source,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    source_root = (
        repository
        / "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
    )
    source_root.mkdir(parents=True)
    (source_root / f"{source_sha256}.json").write_bytes(source_raw)
    receipt_sha256 = hashlib.sha256(f"receipt:{room_suffix}".encode()).hexdigest()
    receipt: dict[str, object] = {
        "canonical_source_sha256": source_sha256,
        "cutoff_utc": timestamp_utc,
        "provider_timestamp_utc": timestamp_utc,
    }
    receipt_root = (
        repository
        / "research/data/dojo_paper_ai_inventory_v1/source_capture/receipts"
        / experiment_id
        / room_id
    )
    receipt_root.mkdir(parents=True)
    (receipt_root / f"00000001-{receipt_sha256}.json").write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    ledger_path = (room / "broker.jsonl").resolve()
    config = BrokerServiceConfig(
        socket_path=derive_broker_socket_path(ledger_path),
        ledger_path=ledger_path,
        state_path=(room / "broker_state.json").resolve(),
        repository_root=repository,
        room_id=room_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        launch_preflight_token_sha256=token_sha256,
        bot_hmac_key=BOT_KEY,
        runner_hmac_key=RUNNER_KEY,
        _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
    )
    preflight: dict[str, object] = {
        "candidate_id": candidate_id,
        "launch_preflight_token_sha256": token_sha256,
        "future_window": {
            "start_utc": "2026-07-23T11:00:00Z",
            "end_utc": "2026-07-23T13:00:00Z",
        },
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return config, receipt_sha256, receipt, preflight


def _packet(
    room_id: str, candidate_id: str, signal: dict[str, object]
) -> dict[str, object]:
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": SIGNAL_AT,
        "bindings": {
            "launch_preflight_token_sha256": "0" * 64,
            "git_head": "f" * 40,
            "git_branch": "codex/episode-s5-outcome",
            "canonical_source_root": (
                "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
            ),
            "experiment_id": EXPERIMENT_ID,
            "room_id": room_id,
            "session_contract_sha256": "1" * 64,
            "candidate_id": candidate_id,
            "candidate_sha256": "2" * 64,
            "spec_id": "spec-rpc-v1",
            "spec_sha256": "3" * 64,
            "policy_id": "policy-rpc-v1",
            "policy_sha256": "4" * 64,
            "paper_eligible_tip_sha256": "5" * 64,
            "ledger_sha256": "6" * 64,
            "ledger_observed_at_utc": SIGNAL_AT,
            "state_sha256": "7" * 64,
            "state_observed_at_utc": SIGNAL_AT,
            "snapshot_sha256": "8" * 64,
            "snapshot_observed_at_utc": SIGNAL_AT,
        },
        "position": {
            "position_id": f"FLAT:{PAIR}",
            "pair": PAIR,
            "side": "FLAT",
            "units": 0.0,
            "entry_price": None,
            "opened_at_utc": None,
            "observed_at_utc": SIGNAL_AT,
            "strategy_tag": STRATEGY_TAG,
            "entry_context_sha256": _context_sha(),
            "take_profit": None,
            "stop_loss": None,
            "remaining_ceiling_seconds": 0,
            "unrealized_pl_jpy": 0.0,
            "gross_same_currency_units": 0.0,
            "net_same_currency_units": 0.0,
            "margin_used_jpy": 0.0,
            "capital_locked_jpy": 0.0,
            "same_direction_position_count": 0,
        },
        "entry_signal": signal,
        "quote": {
            "pair": PAIR,
            "bid": 162.99,
            "ask": 163.0,
            "timestamp_utc": SIGNAL_AT,
            "source_sha256": "9" * 64,
            "max_age_seconds": 90,
        },
        "candles": [
            {
                "pair": PAIR,
                "granularity": "M1",
                "started_at_utc": "2026-07-23T11:59:00Z",
                "completed_at_utc": SIGNAL_AT,
                "bid_o": 162.98,
                "bid_h": 163.0,
                "bid_l": 162.97,
                "bid_c": 162.99,
                "ask_o": 162.99,
                "ask_h": 163.01,
                "ask_l": 162.98,
                "ask_c": 163.0,
                "source_sha256": "a" * 64,
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


def _append_gate(
    broker: VirtualBroker,
    *,
    packet_sha: str,
    signal: dict[str, object],
    action: str = "ALLOW_NEW_VIRTUAL",
    applied_at: datetime = APPLY_AT,
    decision_sha: str = "b" * 64,
) -> dict[str, object]:
    admission = (
        {
            "evidence_packet_sha256": packet_sha,
            "permit_expires_at_utc": "2026-07-23T12:01:00Z",
            "entry_signal": signal,
        }
        if action == "ALLOW_NEW_VIRTUAL"
        else None
    )
    common = {
        "decision_sha256": decision_sha,
        "decision_identity_sha256": "c" * 64,
        "action": action,
        "virtual_units": None,
        "room_id": ROOM_ID,
        "session_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "policy_id": "policy-rpc-v1",
        "spec_id": "spec-rpc-v1",
        "ai_producer_id": "codex-ai",
        "ai_model_id": "gpt-test",
        "ai_request_sha256": "d" * 64,
        "ai_response_sha256": "e" * 64,
        "ai_evidence_packet_sha256": packet_sha,
        "position_id": f"FLAT:{PAIR}",
        "pair": PAIR,
        "strategy_tag": STRATEGY_TAG,
        "admission_binding": admission,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
        "consume_at_utc": applied_at.isoformat().replace("+00:00", "Z"),
    }
    with patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
        broker_datetime.now.return_value = applied_at
        broker._log("AI_INVENTORY_ACTION_RESERVED", dict(common))
        reservation_sha = broker._prev_sha
        broker._log(
            "AI_INVENTORY_ACTION_APPLIED",
            {
                **common,
                "reservation_sha256": reservation_sha,
                "close_sha256": None,
                "cancelled_order_ids": [],
                "cancel_sha256s": [],
                "realized_pl_jpy": None,
                "block_new": action == "BLOCK_NEW",
                "allow_new_virtual": action == "ALLOW_NEW_VIRTUAL",
                "single_use_entry_permit": action == "ALLOW_NEW_VIRTUAL",
                "entry_proxy_consumed": (
                    False if action == "ALLOW_NEW_VIRTUAL" else None
                ),
                "status": "APPLIED",
            },
        )
    return {
        "contract": ENTRY_ADMISSION_REFERENCE_CONTRACT,
        "applied_receipt_sha256": broker._prev_sha,
        "decision_sha256": decision_sha,
        "room_id": ROOM_ID,
        "candidate_id": CANDIDATE_ID,
        "signal_identity_sha256": signal["signal_identity_sha256"],
    }


def _append_close_action(
    broker: VirtualBroker, trade_id: str, *, applied_at: datetime
) -> None:
    common = {
        "decision_sha256": "2" * 64,
        "decision_identity_sha256": "3" * 64,
        "action": "CLOSE_VIRTUAL",
        "virtual_units": None,
        "room_id": ROOM_ID,
        "session_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "policy_id": "policy-rpc-v1",
        "spec_id": "spec-rpc-v1",
        "ai_producer_id": "codex-ai",
        "ai_model_id": "gpt-test",
        "ai_request_sha256": "4" * 64,
        "ai_response_sha256": "5" * 64,
        "ai_evidence_packet_sha256": "6" * 64,
        "position_id": trade_id,
        "pair": PAIR,
        "strategy_tag": STRATEGY_TAG,
        "admission_binding": None,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
        "consume_at_utc": applied_at.isoformat().replace("+00:00", "Z"),
    }
    with patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
        broker_datetime.now.return_value = applied_at
        broker._log("AI_INVENTORY_ACTION_RESERVED", dict(common))
        reservation_sha = broker._prev_sha
        realized = broker.close_trade(trade_id)
        close_sha = broker._prev_sha
        broker._log(
            "AI_INVENTORY_ACTION_APPLIED",
            {
                **common,
                "reservation_sha256": reservation_sha,
                "close_sha256": close_sha,
                "realized_pl_jpy": float(realized),
                "block_new": False,
                "allow_new_virtual": False,
                "single_use_entry_permit": False,
                "entry_proxy_consumed": None,
                "status": "APPLIED",
            },
        )


class _Harness:
    def __init__(self, root: Path) -> None:
        self.repository = root
        # Keep the AF_UNIX test path below macOS's short sockaddr_un limit.
        self.room = root / ROOM_ID
        self.room.mkdir(parents=True)
        ledger_path = (self.room / "broker.jsonl").resolve()
        self.config = BrokerServiceConfig(
            socket_path=derive_broker_socket_path(ledger_path),
            ledger_path=ledger_path,
            state_path=(self.room / "broker_state.json").resolve(),
            repository_root=root.resolve(),
            room_id=ROOM_ID,
            candidate_id=CANDIDATE_ID,
            bot_hmac_key=BOT_KEY,
            runner_hmac_key=RUNNER_KEY,
            allow_test_only_raw_quotes=True,
            _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
        )
        signal = _signal()
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=datetime(2026, 7, 23, 12, 0, 1, tzinfo=UTC),
        ):
            packet_path = write_ai_inventory_evidence_packet(
                root, _packet(ROOM_ID, CANDIDATE_ID, signal)
            )
        self.packet = json.loads(packet_path.read_text())
        broker = VirtualBroker(self.config.ledger_path, fast_ledger=False)
        broker.last_quotes[PAIR] = (162.99, 163.0, SIGNAL_AT)
        self.reference = _append_gate(
            broker, packet_sha=packet_path.stem, signal=signal
        )
        _write_broker_state(broker, self.config.state_path, RUNNER_KEY)
        broker._handle.close()
        self.process: multiprocessing.Process | None = None

    def start(self) -> None:
        context = multiprocessing.get_context("fork")
        clock_patch = patch(
            "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
            return_value=ENTRY_AT,
        )
        runtime_patch = patch(
            "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
            return_value=ENTRY_AT,
        )
        broker_datetime_patch = patch(
            "quant_rabbit.virtual_broker.datetime", wraps=datetime
        )
        self._clock_patch = clock_patch
        self._runtime_patch = runtime_patch
        self._broker_datetime_patch = broker_datetime_patch
        clock_patch.start()
        runtime_patch.start()
        broker_datetime = broker_datetime_patch.start()
        broker_datetime.now.return_value = ENTRY_AT
        self.process = context.Process(target=_serve, args=(self.config,))
        self.process.start()
        deadline = time.monotonic() + 5
        client = DojoAIInventoryEntryClient(self.config.socket_path, BOT_KEY)
        while time.monotonic() < deadline:
            try:
                client.health()
                return
            except AIInventoryBrokerServiceError:
                time.sleep(0.02)
        raise AssertionError("broker service did not become ready")

    def stop(self) -> None:
        if self.process is not None and self.process.is_alive():
            DojoAIInventoryRunnerClient(self.config.socket_path, RUNNER_KEY).shutdown()
            self.process.join(5)
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self._runtime_patch.stop()
        self._clock_patch.stop()
        self._broker_datetime_patch.stop()


class TestDojoAIInventoryBrokerService(unittest.TestCase):
    def test_public_raw_quote_flag_cannot_bypass_production_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            exposed = replace(
                harness.config,
                _test_only_capability=None,
            )
            with self.assertRaisesRegex(
                AIInventoryBrokerServiceError,
                "private test capability",
            ):
                _BrokerOwner(exposed)

    def test_production_repository_must_match_loaded_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            config, _, _, _ = _captured_quote_fixture(
                repository,
                room_suffix="foreign-root-001",
                bid=163.0,
                ask=163.01,
                timestamp_utc=SIGNAL_AT,
            )
            foreign = replace(config, _test_only_capability=None)
            with self.assertRaisesRegex(
                AIInventoryBrokerServiceError,
                "differs from the loaded package",
            ):
                _BrokerOwner(foreign)

    def test_real_session_writer_authorizes_real_drain_broker_verifier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            experiment_id = "paper-ai-inventory-experiment-real-drain-e2e"
            room_id = "paper-ai-inventory-room-real-drain-e2e"
            candidate_id = hashlib.sha256(b"real-drain-e2e").hexdigest()
            mapping: dict[str, object] = {
                "contract": "QR_DOJO_AI_INVENTORY_SESSION_CONFIG_V1",
                "experiment_id": experiment_id,
                "room_id": room_id,
                "candidate_id": candidate_id,
                "dependency_id": "paper-ai-inventory-dependency-real-drain-e2e",
                "pair": PAIR,
                "window_start_utc": "2026-07-23T11:00:00Z",
                "window_end_utc": "2026-07-23T13:00:00Z",
                "adapter_id": "sealed-adapter-real-drain-e2e",
                "model_id": "sealed-model-real-drain-e2e",
                "model_config_sha256": "b" * 64,
                "producer_id": "producer-real-drain-e2e",
                "bot_config_sha256": "c" * 64,
                "balance_jpy": 200_000.0,
                "slippage_pips": 0.3,
                "financing_pips_per_day": 0.8,
                "leverage": 2.0,
                "original_ceiling_minutes": 60,
                "cycle_interval_seconds": 45,
                "drain_interval_seconds": 30,
                "capture_deadline_seconds": 30,
                "evaluation_horizon_seconds": 3_600,
                "launch_preflight_token_sha256": "d" * 64,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "external_broker_mutation_allowed": False,
            }
            mapping["session_config_sha256"] = session_config_sha256(mapping)
            room = (
                repository
                / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT
                / experiment_id
                / room_id
            )
            room.mkdir(parents=True)
            session_config = ai_inventory_session_config_from_mapping(
                repository,
                mapping,
            )
            preflight = {
                "paper_eligible_event_sha256": "e" * 64,
                "future_registry_sha256": "f" * 64,
            }
            session_contract = _build_session_contract(
                session_config,
                preflight,
                screen_name=f"qr-dojo-{room_id}",
                process_argv=("python3.12", "drain-e2e"),
            )
            context = AIInventorySessionContext(
                config=session_config,
                room_root=room,
                launch_preflight=MappingProxyType(preflight),
                session_contract=MappingProxyType(session_contract),
            )
            _write_immutable_contract(
                room / SESSION_CONTRACT_NAME,
                session_contract,
            )
            lifecycle_tip = _append_lifecycle(
                room,
                "ENTRY_STOP",
                {
                    "status": "DRAINING",
                    "new_entries_allowed": False,
                    "force_close": False,
                    "original_ceiling_minutes": 60,
                },
            )
            ledger_path = (room / "broker_ledger.jsonl").resolve()
            state_path = (room / "broker_state.json").resolve()
            broker = VirtualBroker(
                ledger_path,
                balance_jpy=200_000.0,
                slippage_pips=0.3,
                financing_pips_per_day=0.8,
                leverage=2.0,
                fast_ledger=False,
            )
            broker.last_quotes[PAIR] = (
                162.99,
                163.0,
                "2026-07-23T11:00:00Z",
            )
            broker.market_order(
                PAIR,
                "LONG",
                100.0,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            _write_broker_state(broker, state_path, RUNNER_KEY)
            checkpoint_tip = broker._prev_sha
            checkpoint_raw = state_path.read_bytes()
            broker._handle.close()
            _write_state(
                room,
                status="DRAINING",
                lifecycle_tip=lifecycle_tip,
                summary={
                    "positions_count": 1,
                    "orders_count": 0,
                    "new_entries_allowed": False,
                },
                pending_evaluations=0,
                market_open=True,
            )
            authorization_path = _append_drain_broker_restart_receipt(
                context,
                lifecycle_tip,
                {
                    "experiment_id": experiment_id,
                    "room_id": room_id,
                    "candidate_id": candidate_id,
                    "checkpoint_reconciled": True,
                    "broker_ledger_terminal_sha256": checkpoint_tip,
                    "broker_snapshot_sha256": hashlib.sha256(
                        checkpoint_raw
                    ).hexdigest(),
                    "broker_snapshot_ledger_terminal_sha256": checkpoint_tip,
                    "positions_count": 1,
                    "orders_count": 0,
                    "broker_recovery_wal_sha256": None,
                    "broker_recovery_wal_checkpoint_ledger_sha256": None,
                    "broker_recovery_wal_expected_event_count": 0,
                    "broker_recovery_wal_applied_event_count": 0,
                    "broker_recovery_wal_validated": False,
                    "new_entries_allowed": False,
                    "ai_decision_allowed": False,
                    "force_close_allowed": False,
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                    "external_broker_mutation_allowed": False,
                },
            )
            broker_config = BrokerServiceConfig(
                socket_path=derive_broker_socket_path(ledger_path),
                ledger_path=ledger_path,
                state_path=state_path,
                repository_root=repository,
                room_id=room_id,
                candidate_id=candidate_id,
                experiment_id=experiment_id,
                launch_preflight_token_sha256="d" * 64,
                bot_hmac_key=BOT_KEY,
                runner_hmac_key=RUNNER_KEY,
                balance_jpy=200_000.0,
                slippage_pips=0.3,
                financing_pips_per_day=0.8,
                leverage=2.0,
                mode="DRAIN_ONLY",
                drain_authorization_path=authorization_path,
                original_ceiling_minutes=60,
                _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
            )

            owner = _BrokerOwner(broker_config)
            self.assertEqual(owner.config.mode, "DRAIN_ONLY")
            self.assertEqual(len(owner.broker.positions), 1)
            source = {
                "ask": 163.01,
                "bid": 163.0,
                "max_age_seconds": 120,
                "pair": PAIR,
                "timestamp_utc": "2026-07-23T13:00:00Z",
            }
            source_raw = json.dumps(
                source,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            source_sha256 = hashlib.sha256(source_raw).hexdigest()
            source_root = (
                repository
                / "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
            )
            source_root.mkdir(parents=True)
            (source_root / f"{source_sha256}.json").write_bytes(source_raw)
            drain_quote_receipt_sha256 = hashlib.sha256(
                b"real-drain-e2e-quote"
            ).hexdigest()
            drain_quote_receipt = {
                "receipt_sha256": drain_quote_receipt_sha256,
                "canonical_source_sha256": source_sha256,
                "provider_timestamp_utc": "2026-07-23T13:00:00Z",
                "broker_ledger_terminal_sha256": checkpoint_tip,
                "broker_snapshot_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "broker_snapshot_ledger_terminal_sha256": checkpoint_tip,
                "positions_count": 1,
                "orders_count": 0,
                "original_ceiling_minutes": 60,
                "drain_only": True,
                "new_entries_allowed": False,
                "ai_evaluation_allowed": False,
                "force_close_allowed": False,
                "allowed_drain_resolutions": [
                    "MARGIN_CLOSEOUT",
                    "ORIGINAL_CEILING",
                    "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                    "SL",
                    "TP",
                ],
            }
            now = datetime(2026, 7, 23, 13, 0, 2, tzinfo=UTC)
            original_log = owner.broker._log

            def crash_after_ceiling(
                event: str, payload: dict[str, object]
            ) -> None:
                original_log(event, payload)
                raise RuntimeError("real authorization drain crash")

            with (
                patch(
                    "quant_rabbit.dojo_ai_drain_quote."
                    "verify_ai_drain_quote_receipt",
                    return_value=drain_quote_receipt,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=now,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark."
                    "_trusted_repository_root",
                    return_value=repository,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                    return_value=now,
                ),
                patch.object(
                    owner.broker,
                    "_log",
                    side_effect=crash_after_ceiling,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "real authorization drain crash",
                ),
            ):
                owner._dispatch(
                    "runner",
                    "APPLY_DRAIN_QUOTE",
                    {
                        "drain_quote_receipt_sha256": (
                            drain_quote_receipt_sha256
                        )
                    },
                )
            owner.broker._handle.close()
            recovery = _validated_drain_recovery_binding(
                broker_config,
                snapshot_tip=checkpoint_tip,
            )
            second_authorization_path = (
                _append_drain_broker_restart_receipt(
                    context,
                    lifecycle_tip,
                    {
                        "experiment_id": experiment_id,
                        "room_id": room_id,
                        "candidate_id": candidate_id,
                        "checkpoint_reconciled": True,
                        "broker_ledger_terminal_sha256": recovery[
                            "broker_ledger_terminal_sha256"
                        ],
                        "broker_snapshot_sha256": hashlib.sha256(
                            checkpoint_raw
                        ).hexdigest(),
                        "broker_snapshot_ledger_terminal_sha256": (
                            checkpoint_tip
                        ),
                        "positions_count": 1,
                        "orders_count": 0,
                        **{
                            key: value
                            for key, value in recovery.items()
                            if key != "broker_ledger_terminal_sha256"
                        },
                        "new_entries_allowed": False,
                        "ai_decision_allowed": False,
                        "force_close_allowed": False,
                        "paper_only": True,
                        "order_authority": "NONE",
                        "live_permission": False,
                        "external_broker_mutation_allowed": False,
                    },
                )
            )
            restarted = _BrokerOwner(
                replace(
                    broker_config,
                    drain_authorization_path=second_authorization_path,
                )
            )
            self.assertEqual(restarted.broker.positions, {})
            self.assertFalse(
                state_path.with_name("captured_quote_apply_wal.json").exists()
            )
            restarted.broker._handle.close()

    def test_drain_only_startup_requires_canonical_authorization_and_exposes_only_drain_rpc(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            active, _, _, _ = _captured_quote_fixture(
                repository,
                room_suffix="drain-startup-001",
                bid=163.0,
                ask=163.01,
                timestamp_utc="2026-07-23T13:00:00Z",
            )
            broker = VirtualBroker(active.ledger_path, fast_ledger=False)
            broker.last_quotes[PAIR] = (
                162.99,
                163.0,
                "2026-07-23T11:00:00Z",
            )
            broker.market_order(
                PAIR,
                "LONG",
                100.0,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            _write_broker_state(broker, active.state_path, RUNNER_KEY)
            broker._handle.close()
            authorization_path = (
                active.ledger_path.parent
                / "drain_broker_restarts"
                / f"00000001-{'d' * 64}.json"
            ).resolve()
            drain = replace(
                active,
                mode="DRAIN_ONLY",
                drain_authorization_path=authorization_path,
                original_ceiling_minutes=60,
            )
            with patch(
                "quant_rabbit.dojo_ai_inventory_broker_service."
                "verify_drain_broker_restart_authorization",
                return_value={
                    "receipt_sha256": "d" * 64,
                    "mode": "DRAIN_ONLY",
                },
            ) as verify:
                owner = _BrokerOwner(drain)
            call = verify.call_args.kwargs
            checkpoint_raw = drain.state_path.read_bytes()
            self.assertEqual(call["room_root"], drain.ledger_path.parent)
            self.assertEqual(call["authorization_path"], authorization_path)
            self.assertEqual(
                call["broker_snapshot_sha256"],
                hashlib.sha256(checkpoint_raw).hexdigest(),
            )
            self.assertEqual(call["positions_count"], 1)
            self.assertEqual(call["orders_count"], 0)
            self.assertEqual(
                owner._dispatch("runner", "HEALTH", {})["mode"],
                "DRAIN_ONLY",
            )
            for role, command in (
                ("bot", "HEALTH"),
                ("bot", "ENTRY_MARKET"),
                ("runner", "APPLY_CAPTURED_QUOTE"),
                ("runner", "APPLY_AI_DECISION"),
            ):
                with self.subTest(role=role, command=command), self.assertRaises(
                    AIInventoryBrokerServiceError
                ):
                    owner._dispatch(role, command, {})
            owner.broker._handle.close()

            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_drain_broker_restart_authorization",
                    side_effect=RuntimeError("tampered authorization"),
                ),
                self.assertRaisesRegex(
                    AIInventoryBrokerServiceError,
                    "canonical DRAIN_ONLY broker authorization is invalid",
                ),
            ):
                _BrokerOwner(drain)

    def test_drain_quote_cancels_orders_without_fill_and_resolves_only_original_ceiling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            active, receipt_sha256, source_receipt, _ = (
                _captured_quote_fixture(
                    repository,
                    room_suffix="drain-quote-001",
                    bid=163.0,
                    ask=163.01,
                    timestamp_utc="2026-07-23T13:00:00Z",
                )
            )
            broker = VirtualBroker(active.ledger_path, fast_ledger=False)
            broker.last_quotes[PAIR] = (
                162.99,
                163.0,
                "2026-07-23T11:00:00Z",
            )
            trade_id = broker.market_order(
                PAIR,
                "LONG",
                100.0,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            order_id = broker.limit_order(
                PAIR,
                "LONG",
                100.0,
                163.02,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            _write_broker_state(broker, active.state_path, RUNNER_KEY)
            checkpoint_tip = broker._prev_sha
            broker._handle.close()
            checkpoint_raw = active.state_path.read_bytes()
            authorization_path = (
                active.ledger_path.parent
                / "drain_broker_restarts"
                / f"00000001-{'d' * 64}.json"
            ).resolve()
            drain = replace(
                active,
                mode="DRAIN_ONLY",
                drain_authorization_path=authorization_path,
                original_ceiling_minutes=60,
            )
            drain_receipt = {
                "receipt_sha256": receipt_sha256,
                "canonical_source_sha256": source_receipt[
                    "canonical_source_sha256"
                ],
                "provider_timestamp_utc": "2026-07-23T13:00:00Z",
                "broker_ledger_terminal_sha256": checkpoint_tip,
                "broker_snapshot_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "broker_snapshot_ledger_terminal_sha256": checkpoint_tip,
                "positions_count": 1,
                "orders_count": 1,
                "original_ceiling_minutes": 60,
                "drain_only": True,
                "new_entries_allowed": False,
                "ai_evaluation_allowed": False,
                "force_close_allowed": False,
                "allowed_drain_resolutions": [
                    "MARGIN_CLOSEOUT",
                    "ORIGINAL_CEILING",
                    "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                    "SL",
                    "TP",
                ],
            }
            drain_now = datetime(2026, 7, 23, 13, 0, 2, tzinfo=UTC)
            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_drain_broker_restart_authorization",
                    return_value={"mode": "DRAIN_ONLY"},
                ),
                patch(
                    "quant_rabbit.dojo_ai_drain_quote."
                    "verify_ai_drain_quote_receipt",
                    return_value=drain_receipt,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=drain_now,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark."
                    "_trusted_repository_root",
                    return_value=repository,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                    return_value=drain_now,
                ),
            ):
                owner = _BrokerOwner(drain)
                events = owner._dispatch(
                    "runner",
                    "APPLY_DRAIN_QUOTE",
                    {"drain_quote_receipt_sha256": receipt_sha256},
                )
                self.assertEqual(
                    [event["event"] for event in events],
                    [
                        "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                        "EXIT_ORIGINAL_CEILING",
                    ],
                )
                self.assertEqual(events[0]["order_id"], order_id)
                self.assertEqual(events[1]["trade_id"], trade_id)
                self.assertEqual(owner.broker.orders, {})
                self.assertEqual(owner.broker.positions, {})
                rows = [
                    json.loads(line)
                    for line in drain.ledger_path.read_text().splitlines()
                ]
                checkpoint_index = next(
                    index
                    for index, row in enumerate(rows)
                    if row["sha"] == checkpoint_tip
                )
                self.assertFalse(
                    any(
                        row["event"].startswith("FILL_")
                        for row in rows[checkpoint_index + 1 :]
                    )
                )
                self.assertEqual(
                    owner._dispatch(
                        "runner",
                        "APPLY_DRAIN_QUOTE",
                        {"drain_quote_receipt_sha256": receipt_sha256},
                    ),
                    [],
                )
                owner.broker._handle.close()

    def test_drain_quote_partial_wal_restarts_exactly_once_and_unknown_suffix_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            active, receipt_sha256, source_receipt, _ = (
                _captured_quote_fixture(
                    repository,
                    room_suffix="drain-wal-restart-001",
                    bid=163.0,
                    ask=163.01,
                    timestamp_utc="2026-07-23T13:00:00Z",
                )
            )
            broker = VirtualBroker(active.ledger_path, fast_ledger=False)
            broker.last_quotes[PAIR] = (
                162.99,
                163.0,
                "2026-07-23T11:00:00Z",
            )
            trade_id = broker.market_order(
                PAIR,
                "LONG",
                100.0,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            broker.limit_order(
                PAIR,
                "LONG",
                100.0,
                163.02,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=STRATEGY_TAG,
            )
            _write_broker_state(broker, active.state_path, RUNNER_KEY)
            checkpoint_tip = broker._prev_sha
            checkpoint_raw = active.state_path.read_bytes()
            broker._handle.close()
            drain = replace(
                active,
                mode="DRAIN_ONLY",
                drain_authorization_path=(
                    active.ledger_path.parent
                    / "drain_broker_restarts"
                    / f"00000001-{'d' * 64}.json"
                ).resolve(),
                original_ceiling_minutes=60,
            )
            receipt = {
                "receipt_sha256": receipt_sha256,
                "canonical_source_sha256": source_receipt[
                    "canonical_source_sha256"
                ],
                "provider_timestamp_utc": "2026-07-23T13:00:00Z",
                "broker_ledger_terminal_sha256": checkpoint_tip,
                "broker_snapshot_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "broker_snapshot_ledger_terminal_sha256": checkpoint_tip,
                "positions_count": 1,
                "orders_count": 1,
                "original_ceiling_minutes": 60,
                "drain_only": True,
                "new_entries_allowed": False,
                "ai_evaluation_allowed": False,
                "force_close_allowed": False,
                "allowed_drain_resolutions": [
                    "MARGIN_CLOSEOUT",
                    "ORIGINAL_CEILING",
                    "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                    "SL",
                    "TP",
                ],
            }
            now = datetime(2026, 7, 23, 13, 0, 2, tzinfo=UTC)
            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_drain_broker_restart_authorization",
                    return_value={"mode": "DRAIN_ONLY"},
                ),
                patch(
                    "quant_rabbit.dojo_ai_drain_quote."
                    "verify_ai_drain_quote_receipt",
                    return_value=receipt,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=now,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark."
                    "_trusted_repository_root",
                    return_value=repository,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                    return_value=now,
                ),
            ):
                owner = _BrokerOwner(drain)
                original_log = owner.broker._log
                crashed = False

                def crash_after_first_durable_event(
                    event: str, payload: dict[str, object]
                ) -> None:
                    nonlocal crashed
                    original_log(event, payload)
                    if not crashed:
                        crashed = True
                        raise RuntimeError("crash in drain WAL")

                with (
                    patch.object(
                        owner.broker,
                        "_log",
                        side_effect=crash_after_first_durable_event,
                    ),
                    self.assertRaisesRegex(RuntimeError, "drain WAL"),
                ):
                    owner._dispatch(
                        "runner",
                        "APPLY_DRAIN_QUOTE",
                        {"drain_quote_receipt_sha256": receipt_sha256},
                    )
                self.assertTrue(
                    active.state_path.with_name(
                        "captured_quote_apply_wal.json"
                    ).exists()
                )
                owner.broker._handle.close()

                with patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_drain_broker_restart_authorization",
                    return_value={"mode": "DRAIN_ONLY"},
                ) as verify_restart:
                    restarted = _BrokerOwner(drain)
                restart_binding = verify_restart.call_args.kwargs
                self.assertTrue(
                    restart_binding["broker_recovery_wal_validated"]
                )
                self.assertEqual(
                    restart_binding[
                        "broker_recovery_wal_applied_event_count"
                    ],
                    1,
                )
                self.assertEqual(
                    restart_binding[
                        "broker_recovery_wal_expected_event_count"
                    ],
                    2,
                )
                self.assertIn(trade_id, {
                    row["payload"].get("trade_id")
                    for row in (
                        json.loads(line)
                        for line in active.ledger_path.read_text().splitlines()
                    )
                    if row["event"] == "EXIT_ORIGINAL_CEILING"
                })
                self.assertEqual(restarted.broker.positions, {})
                self.assertEqual(restarted.broker.orders, {})
                self.assertFalse(
                    active.state_path.with_name(
                        "captured_quote_apply_wal.json"
                    ).exists()
                )
                restarted.broker._handle.close()

        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            active, _, _, _ = _captured_quote_fixture(
                repository,
                room_suffix="drain-unknown-suffix-001",
                bid=163.0,
                ask=163.01,
                timestamp_utc="2026-07-23T13:00:00Z",
            )
            broker = VirtualBroker(active.ledger_path, fast_ledger=False)
            _write_broker_state(broker, active.state_path, RUNNER_KEY)
            broker._log("UNKNOWN_DRAIN_SUFFIX", {"bad": True})
            broker._handle.close()
            drain = replace(
                active,
                mode="DRAIN_ONLY",
                drain_authorization_path=(
                    active.ledger_path.parent
                    / "drain_broker_restarts"
                    / f"00000001-{'d' * 64}.json"
                ).resolve(),
                original_ceiling_minutes=60,
            )
            with self.assertRaisesRegex(
                AIInventoryBrokerServiceError,
                "advanced without a signed recovery WAL",
            ):
                _BrokerOwner(drain)

    def test_entry_paths_recheck_fixed_window_immediately_before_mutation(
        self,
    ) -> None:
        for command, controller_method in (
            ("ENTRY_MARKET", "market_order"),
            ("ENTRY_LIMIT", "limit_order"),
            ("ENTRY_STOP", "stop_order"),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                repository = Path(tmp).resolve()
                config, _, _, preflight = _captured_quote_fixture(
                    repository,
                    room_suffix=f"entry-boundary-{command.lower()}",
                    bid=163.0,
                    ask=163.01,
                    timestamp_utc=SIGNAL_AT,
                )
                with (
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service."
                        "verify_paper_ai_inventory_launch_preflight",
                        return_value=preflight,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                        return_value=ENTRY_AT,
                    ),
                ):
                    owner = _BrokerOwner(config)
                window_end = datetime(2026, 7, 23, 13, 0, tzinfo=UTC)
                with (
                    patch.object(owner, "_verify_current_evidence"),
                    patch.object(
                        type(owner.controller),
                        controller_method,
                        side_effect=AssertionError(
                            "entry mutation must not run after window end"
                        ),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                        side_effect=(ENTRY_AT, window_end),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service."
                        "verify_paper_ai_inventory_launch_preflight",
                        return_value=preflight,
                    ),
                    self.assertRaisesRegex(
                        AIInventoryEntryDeniedError,
                        "outside its immutable future window",
                    ),
                ):
                    owner._entry(
                        command,
                        {
                            "pair": PAIR,
                            "side": "LONG",
                            "units": 100.0,
                            "price": 163.0,
                            "tp_pips": 6.0,
                            "sl_pips": 25.0,
                            "strategy_tag": STRATEGY_TAG,
                            "entry_context": _context(),
                            "ai_admission": {},
                        },
                    )
                owner.broker._handle.close()

    def test_normal_production_owner_refuses_post_window_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            config, _, _, preflight = _captured_quote_fixture(
                repository,
                room_suffix="post-window-normal-start",
                bid=163.0,
                ask=163.01,
                timestamp_utc=SIGNAL_AT,
            )
            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_paper_ai_inventory_launch_preflight",
                    return_value=preflight,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=datetime(2026, 7, 23, 13, 0, tzinfo=UTC),
                ),
                self.assertRaisesRegex(
                    AIInventoryEntryDeniedError,
                    "outside its immutable future window",
                ),
            ):
                _BrokerOwner(config)

    def test_production_owner_accepts_only_signed_captured_quote_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            experiment_id = "paper-ai-inventory-experiment-captured-quote-001"
            room_id = "paper-ai-inventory-room-captured-quote-001"
            candidate_id = "2" * 64
            token_sha256 = "a" * 64
            room = (
                repository
                / "research/data/dojo_paper_ai_inventory_v1/rooms"
                / experiment_id
                / room_id
            )
            room.mkdir(parents=True)
            source = {
                "ask": 163.01,
                "bid": 163.0,
                "max_age_seconds": 120,
                "pair": PAIR,
                "timestamp_utc": SIGNAL_AT,
            }
            source_raw = json.dumps(
                source,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            source_sha256 = hashlib.sha256(source_raw).hexdigest()
            source_root = (
                repository
                / "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
            )
            source_root.mkdir(parents=True)
            (source_root / f"{source_sha256}.json").write_bytes(source_raw)
            receipt_sha256 = "c" * 64
            receipt = {
                "canonical_source_sha256": source_sha256,
                "cutoff_utc": SIGNAL_AT,
                "provider_timestamp_utc": SIGNAL_AT,
            }
            receipt_root = (
                repository
                / "research/data/dojo_paper_ai_inventory_v1/source_capture/receipts"
                / experiment_id
                / room_id
            )
            receipt_root.mkdir(parents=True)
            (receipt_root / f"00000001-{receipt_sha256}.json").write_text(
                json.dumps(
                    receipt,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            ledger_path = (room / "broker.jsonl").resolve()
            config = BrokerServiceConfig(
                socket_path=derive_broker_socket_path(ledger_path),
                ledger_path=ledger_path,
                state_path=(room / "broker_state.json").resolve(),
                repository_root=repository,
                room_id=room_id,
                candidate_id=candidate_id,
                experiment_id=experiment_id,
                launch_preflight_token_sha256=token_sha256,
                bot_hmac_key=BOT_KEY,
                runner_hmac_key=RUNNER_KEY,
                _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
            )
            preflight = {
                "candidate_id": candidate_id,
                "launch_preflight_token_sha256": token_sha256,
                "future_window": {
                    "start_utc": "2026-07-23T11:00:00Z",
                    "end_utc": "2026-07-23T13:00:00Z",
                },
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_paper_ai_inventory_launch_preflight",
                    return_value=preflight,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_ai_source_capture_receipt",
                    return_value=receipt,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=ENTRY_AT,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark."
                    "_trusted_repository_root",
                    return_value=repository,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                    return_value=ENTRY_AT,
                ),
            ):
                owner = _BrokerOwner(config)
                with self.assertRaisesRegex(
                    AIInventoryBrokerServiceError,
                    "raw quote RPC is test-only",
                ):
                    owner._dispatch(
                        "runner",
                        "APPLY_QUOTE",
                        {
                            "pair": PAIR,
                            "bid": 1.0,
                            "ask": 1.1,
                            "ts": SIGNAL_AT,
                        },
                    )
                self.assertEqual(
                    owner._dispatch(
                        "runner",
                        "APPLY_CAPTURED_QUOTE",
                        {"capture_receipt_sha256": receipt_sha256},
                    ),
                    [],
                )
                provenance = owner._dispatch(
                    "runner", "QUOTE_PROVENANCE", {}
                )[PAIR]
                self.assertEqual(
                    owner.broker.last_quotes[PAIR],
                    (163.0, 163.01, SIGNAL_AT),
                )
                self.assertEqual(
                    provenance["capture_source_sha256"],
                    source_sha256,
                )
                self.assertEqual(
                    provenance["acquisition_receipt_sha256"],
                    receipt_sha256,
                )
                self.assertIs(provenance["test_only_raw_quote"], False)
                self.assertRegex(
                    provenance["quote_watermark_sha256"],
                    r"^[0-9a-f]{64}$",
                )
                owner.broker._handle.close()

                restarted = _BrokerOwner(config)
                self.assertEqual(
                    restarted.broker.last_quotes[PAIR],
                    (163.0, 163.01, SIGNAL_AT),
                )
                self.assertEqual(
                    restarted._dispatch("runner", "QUOTE_PROVENANCE", {})[
                        PAIR
                    ],
                    provenance,
                )
                restarted.broker._handle.close()

    def test_captured_quote_watermark_only_crash_retries_same_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp).resolve()
            config, receipt_sha256, receipt, preflight = _captured_quote_fixture(
                repository,
                room_suffix="watermark-only-crash-001",
                bid=163.0,
                ask=163.01,
                timestamp_utc=SIGNAL_AT,
            )
            with (
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_paper_ai_inventory_launch_preflight",
                    return_value=preflight,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "verify_ai_source_capture_receipt",
                    return_value=receipt,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                    return_value=ENTRY_AT,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark."
                    "_trusted_repository_root",
                    return_value=repository,
                ),
                patch(
                    "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                    return_value=ENTRY_AT,
                ),
            ):
                owner = _BrokerOwner(config)
                with patch(
                    "quant_rabbit.dojo_ai_inventory_broker_service."
                    "_write_quote_apply_wal",
                    side_effect=RuntimeError("crash after watermark"),
                ), self.assertRaisesRegex(RuntimeError, "after watermark"):
                    owner._dispatch(
                        "runner",
                        "APPLY_CAPTURED_QUOTE",
                        {"capture_receipt_sha256": receipt_sha256},
                    )
                self.assertNotIn(PAIR, owner.broker.last_quotes)
                self.assertFalse(
                    config.state_path.with_name(
                        "captured_quote_apply_wal.json"
                    ).exists()
                )
                owner.broker._handle.close()

                restarted = _BrokerOwner(config)
                self.assertEqual(
                    restarted._dispatch(
                        "runner",
                        "APPLY_CAPTURED_QUOTE",
                        {"capture_receipt_sha256": receipt_sha256},
                    ),
                    [],
                )
                self.assertEqual(
                    restarted.broker.last_quotes[PAIR],
                    (163.0, 163.01, SIGNAL_AT),
                )
                watermark_rows = (
                    config.state_path.parent / "quote_watermarks.jsonl"
                ).read_text().splitlines()
                self.assertEqual(len(watermark_rows), 1)
                self.assertFalse(
                    config.state_path.with_name(
                        "captured_quote_apply_wal.json"
                    ).exists()
                )
                restarted.broker._handle.close()

    def test_restart_recovers_captured_quote_fill_and_settlement_before_checkpoint(
        self,
    ) -> None:
        cases = (
            {
                "name": "fill",
                "bid": 162.98,
                "ask": 162.99,
                "event": "FILL_LIMIT",
                "event_count": 1,
            },
            {
                "name": "settlement",
                "bid": 163.06,
                "ask": 163.07,
                "event": "EXIT_TP",
                "event_count": 2,
            },
        )
        for case in cases:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as tmp:
                repository = Path(tmp).resolve()
                config, receipt_sha256, receipt, preflight = (
                    _captured_quote_fixture(
                        repository,
                        room_suffix=f"{case['name']}-crash-001",
                        bid=float(case["bid"]),
                        ask=float(case["ask"]),
                        timestamp_utc=SIGNAL_AT,
                    )
                )
                with (
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service."
                        "verify_paper_ai_inventory_launch_preflight",
                        return_value=preflight,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service."
                        "verify_ai_source_capture_receipt",
                        return_value=receipt,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                        return_value=ENTRY_AT,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_quote_watermark."
                        "_trusted_repository_root",
                        return_value=repository,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_quote_watermark._utc_now",
                        return_value=ENTRY_AT,
                    ),
                    patch(
                        "quant_rabbit.virtual_broker.datetime",
                        wraps=datetime,
                    ) as broker_datetime,
                ):
                    broker_datetime.now.return_value = ENTRY_AT
                    owner = _BrokerOwner(config)
                    if case["name"] == "fill":
                        owner.broker.limit_order(
                            PAIR,
                            "LONG",
                            100.0,
                            163.0,
                            tp_pips=6.0,
                            sl_pips=25.0,
                            strategy_tag=STRATEGY_TAG,
                        )
                    else:
                        owner.broker.last_quotes[PAIR] = (
                            162.99,
                            163.0,
                            "2026-07-23T11:59:59Z",
                        )
                        owner.broker.market_order(
                            PAIR,
                            "LONG",
                            100.0,
                            tp_pips=6.0,
                            sl_pips=25.0,
                            strategy_tag=STRATEGY_TAG,
                        )
                        owner.broker.market_order(
                            PAIR,
                            "LONG",
                            100.0,
                            tp_pips=6.0,
                            sl_pips=25.0,
                            strategy_tag=STRATEGY_TAG,
                        )
                    _write_broker_state(
                        owner.broker, config.state_path, RUNNER_KEY
                    )
                    original_log = owner.broker._log
                    crashed = False

                    def crash_after_durable_event(
                        event: str, payload: dict[str, object]
                    ) -> None:
                        nonlocal crashed
                        original_log(event, payload)
                        if not crashed:
                            crashed = True
                            raise RuntimeError("crash after durable quote event")

                    with patch.object(
                        owner.broker,
                        "_log",
                        side_effect=crash_after_durable_event,
                    ), self.assertRaisesRegex(
                        RuntimeError, "durable quote event"
                    ):
                        owner._dispatch(
                            "runner",
                            "APPLY_CAPTURED_QUOTE",
                            {"capture_receipt_sha256": receipt_sha256},
                        )
                    self.assertTrue(
                        config.state_path.with_name(
                            "captured_quote_apply_wal.json"
                        ).exists()
                    )
                    owner.broker._handle.close()

                    restarted = _BrokerOwner(config)
                    rows = [
                        json.loads(line)
                        for line in config.ledger_path.read_text().splitlines()
                    ]
                    matching = [
                        row
                        for row in rows
                        if row["event"] == case["event"]
                    ]
                    self.assertEqual(len(matching), case["event_count"])
                    if case["name"] == "fill":
                        self.assertEqual(len(restarted.broker.positions), 1)
                        self.assertEqual(restarted.broker.orders, {})
                    else:
                        self.assertEqual(restarted.broker.positions, {})
                        self.assertGreater(restarted.broker.balance_jpy, 200_000.0)
                    checkpoint = json.loads(config.state_path.read_text())
                    self.assertEqual(
                        checkpoint["broker"]["ledger_sha"],
                        restarted.broker._prev_sha,
                    )
                    self.assertFalse(
                        config.state_path.with_name(
                            "captured_quote_apply_wal.json"
                        ).exists()
                    )
                    self.assertEqual(
                        restarted._dispatch(
                            "runner",
                            "APPLY_CAPTURED_QUOTE",
                            {"capture_receipt_sha256": receipt_sha256},
                        ),
                        [],
                    )
                    rows_after_retry = [
                        json.loads(line)
                        for line in config.ledger_path.read_text().splitlines()
                    ]
                    self.assertEqual(
                        len(
                            [
                                row
                                for row in rows_after_retry
                                if row["event"] == case["event"]
                            ]
                        ),
                        case["event_count"],
                    )
                    restarted.broker._handle.close()

    def test_bot_has_only_minimal_rpc_and_permit_is_single_use_after_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            harness.start()
            try:
                bot = DojoAIInventoryEntryClient(harness.config.socket_path, BOT_KEY)
                health = bot.health()
                self.assertNotEqual(health["broker_owner_pid"], os.getpid())
                for forbidden in (
                    "close_trade",
                    "cancel_order",
                    "set_exit",
                    "on_quote",
                    "apply_quote",
                    "apply_ai_decision",
                ):
                    self.assertFalse(hasattr(bot, forbidden))
                created = bot.market_order(
                    PAIR,
                    "LONG",
                    100,
                    tp_pips=6,
                    sl_pips=25,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                    ai_admission=harness.reference,
                )
                self.assertTrue(created.startswith("T"))
                # Simulate an owner crash after the durable response/checkpoint,
                # leaving the Unix socket path behind for restart recovery.
                assert harness.process is not None
                harness.process.terminate()
                harness.process.join(5)
            finally:
                harness.stop()

            harness.start()
            try:
                bot = DojoAIInventoryEntryClient(harness.config.socket_path, BOT_KEY)
                with self.assertRaisesRegex(
                    AIInventoryBrokerServiceError, "no exact current entry permit"
                ):
                    bot.market_order(
                        PAIR,
                        "LONG",
                        100,
                        tp_pips=6,
                        sl_pips=25,
                        strategy_tag=STRATEGY_TAG,
                        entry_context=_context(),
                        ai_admission=harness.reference,
                    )
                self.assertEqual(len(bot.positions), 1)
            finally:
                harness.stop()

    def test_wrong_role_key_and_weekend_mutation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            owner = _BrokerOwner(harness.config)
            with self.assertRaisesRegex(Exception, "bot role is not allowed"):
                owner._dispatch(
                    "bot",
                    "APPLY_QUOTE",
                    {
                        "pair": PAIR,
                        "bid": 162.99,
                        "ask": 163.0,
                        "ts": SIGNAL_AT,
                    },
                )
            with self.assertRaisesRegex(
                Exception, "disabled while FX is closed"
            ), patch(
                "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                return_value=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
            ):
                owner._dispatch(
                    "runner",
                    "APPLY_QUOTE",
                    {
                        "pair": PAIR,
                        "bid": 162.99,
                        "ask": 163.0,
                        "ts": "2026-07-25T12:00:00Z",
                    },
                )
            owner.broker._handle.close()

            harness.start()
            try:
                wrong = DojoAIInventoryEntryClient(
                    harness.config.socket_path, RUNNER_KEY
                )
                with self.assertRaisesRegex(
                    AIInventoryBrokerServiceError, "invalid broker response MAC"
                ):
                    wrong.health()
            finally:
                harness.stop()

    def test_entry_revalidates_exact_current_quote_inside_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            harness.start()
            try:
                runner = DojoAIInventoryRunnerClient(
                    harness.config.socket_path, RUNNER_KEY
                )
                runner.apply_quote(PAIR, 163.0, 163.01, "2026-07-23T12:00:01Z")
                bot = DojoAIInventoryEntryClient(harness.config.socket_path, BOT_KEY)
                with self.assertRaisesRegex(
                    AIInventoryBrokerServiceError,
                    "broker quote advanced or differs",
                ):
                    bot.market_order(
                        PAIR,
                        "LONG",
                        100,
                        tp_pips=6,
                        sl_pips=25,
                        strategy_tag=STRATEGY_TAG,
                        entry_context=_context(),
                        ai_admission=harness.reference,
                    )
                self.assertEqual(bot.positions, {})
            finally:
                harness.stop()

    def test_tampered_checkpoint_or_ledger_refuses_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            raw = harness.config.state_path.read_text()
            harness.config.state_path.write_text(raw.replace("200000.0", "200001.0"))
            with self.assertRaisesRegex(Exception, "checkpoint authentication failed"):
                _BrokerOwner(harness.config)

    def test_restart_reconciles_each_exact_ai_close_crash_window(self) -> None:
        from tests.test_dojo_ai_inventory_consumer import (
            _consume as consume_harness,
        )
        from tests.test_dojo_ai_inventory_consumer import (
            _harness as consumer_harness,
        )

        for crash_window in ("RESERVED", "CLOSE", "APPLIED"):
            with self.subTest(crash_window=crash_window), tempfile.TemporaryDirectory() as tmp:
                harness = consumer_harness(Path(tmp))
                broker = harness["broker"]
                state_path = (broker.ledger_path.parent / "broker_state.json").resolve()
                _write_broker_state(broker, state_path, RUNNER_KEY)

                if crash_window == "RESERVED":
                    with patch.object(
                        broker,
                        "close_trade",
                        side_effect=RuntimeError("crash after reservation"),
                    ), self.assertRaisesRegex(RuntimeError, "reservation"):
                        consume_harness(harness)
                elif crash_window == "CLOSE":
                    original_log = broker._log

                    def crash_before_applied(
                        event: str, payload: dict[str, object]
                    ) -> None:
                        if event == "AI_INVENTORY_ACTION_APPLIED":
                            raise RuntimeError("crash after close")
                        original_log(event, payload)

                    with patch.object(
                        broker, "_log", side_effect=crash_before_applied
                    ), self.assertRaisesRegex(RuntimeError, "after close"):
                        consume_harness(harness)
                else:
                    consume_harness(harness)

                expected_balance = broker.balance_jpy
                expected_positions = {
                    trade_id: dict(vars(position))
                    for trade_id, position in broker.positions.items()
                }
                broker._handle.close()
                decision = harness["decision"]
                ledger_path = broker.ledger_path.resolve()
                config = BrokerServiceConfig(
                    socket_path=derive_broker_socket_path(ledger_path),
                    ledger_path=ledger_path,
                    state_path=state_path,
                    repository_root=Path(tmp).resolve(),
                    room_id=decision["session_binding"]["room_id"],
                    candidate_id=decision["candidate_binding"]["candidate_id"],
                    bot_hmac_key=BOT_KEY,
                    runner_hmac_key=RUNNER_KEY,
                    decision_ledger_path=harness["decision_path"],
                    allow_test_only_raw_quotes=True,
                    _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
                )
                owner = _BrokerOwner(config)
                try:
                    self.assertAlmostEqual(owner.broker.balance_jpy, expected_balance)
                    self.assertEqual(
                        {
                            trade_id: dict(vars(position))
                            for trade_id, position in owner.broker.positions.items()
                        },
                        expected_positions,
                    )
                    status = owner._decision_status(
                        {"decision_sha256": decision["decision_sha256"]}
                    )
                    self.assertEqual(
                        status["status"],
                        "RESERVED" if crash_window != "APPLIED" else "APPLIED",
                    )
                    checkpoint = json.loads(state_path.read_text())
                    self.assertEqual(
                        checkpoint["broker"]["ledger_sha"],
                        owner.broker._prev_sha,
                    )
                finally:
                    owner.broker._handle.close()

    def test_restart_rejects_unknown_suffix_after_authenticated_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            broker = VirtualBroker(harness.config.ledger_path, fast_ledger=False)
            try:
                # Replace the harness checkpoint with an exact checkpoint at
                # the current terminal tip, then append an unauthorised row.
                _write_broker_state(broker, harness.config.state_path, RUNNER_KEY)
                broker._log("UNKNOWN_AFTER_CHECKPOINT", {"paper_only": True})
            finally:
                broker._handle.close()
            with self.assertRaisesRegex(Exception, "unknown broker-ledger suffix"):
                _BrokerOwner(harness.config)

    def test_restart_reconciles_entry_permit_and_fill_crash_windows(self) -> None:
        kwargs = {
            "pair": PAIR,
            "side": "LONG",
            "units": 100.0,
            "tp_pips": 6.0,
            "sl_pips": 25.0,
            "strategy_tag": STRATEGY_TAG,
            "entry_context": _context(),
        }
        for crash_window in ("RESERVED", "ENTRY", "CONSUMED"):
            with self.subTest(crash_window=crash_window), tempfile.TemporaryDirectory() as tmp:
                harness = _Harness(Path(tmp))
                owner = _BrokerOwner(harness.config)
                try:
                    if crash_window == "RESERVED":
                        with patch.object(
                            owner.broker,
                            "market_order",
                            side_effect=RuntimeError("crash after reservation"),
                        ), patch(
                            "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
                            return_value=ENTRY_AT,
                        ), patch(
                            "quant_rabbit.virtual_broker.datetime",
                            wraps=datetime,
                        ) as broker_datetime, self.assertRaisesRegex(
                            RuntimeError, "reservation"
                        ):
                            broker_datetime.now.return_value = ENTRY_AT
                            owner.controller.market_order(
                                **kwargs, ai_admission=harness.reference
                            )
                    elif crash_window == "ENTRY":
                        original_log = owner.broker._log

                        def crash_before_consumed(
                            event: str, payload: dict[str, object]
                        ) -> None:
                            if event == "AI_ENTRY_PERMIT_CONSUMED":
                                raise RuntimeError("crash after entry")
                            original_log(event, payload)

                        with patch.object(
                            owner.broker,
                            "_log",
                            side_effect=crash_before_consumed,
                        ), patch(
                            "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
                            return_value=ENTRY_AT,
                        ), patch(
                            "quant_rabbit.virtual_broker.datetime",
                            wraps=datetime,
                        ) as broker_datetime, self.assertRaisesRegex(
                            RuntimeError, "after entry"
                        ):
                            broker_datetime.now.return_value = ENTRY_AT
                            owner.controller.market_order(
                                **kwargs, ai_admission=harness.reference
                            )
                    else:
                        with patch(
                            "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
                            return_value=ENTRY_AT,
                        ), patch(
                            "quant_rabbit.virtual_broker.datetime",
                            wraps=datetime,
                        ) as broker_datetime:
                            broker_datetime.now.return_value = ENTRY_AT
                            owner.controller.market_order(
                                **kwargs, ai_admission=harness.reference
                            )
                finally:
                    owner.broker._handle.close()

                restarted = _BrokerOwner(harness.config)
                try:
                    with patch(
                        "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
                        return_value=ENTRY_AT,
                    ), patch(
                        "quant_rabbit.virtual_broker.datetime",
                        wraps=datetime,
                    ) as broker_datetime:
                        broker_datetime.now.return_value = ENTRY_AT
                        created = restarted.controller.market_order(
                            **kwargs, ai_admission=harness.reference
                        )
                    self.assertIn(created, restarted.broker.positions)
                    self.assertEqual(len(restarted.broker.positions), 1)
                    events = [
                        json.loads(line)["event"]
                        for line in harness.config.ledger_path.read_text().splitlines()
                    ]
                    self.assertEqual(events.count("AI_ENTRY_PERMIT_RESERVED"), 1)
                    self.assertEqual(events.count("FILL_MARKET"), 1)
                    self.assertEqual(events.count("AI_ENTRY_PERMIT_CONSUMED"), 1)
                finally:
                    restarted.broker._handle.close()

    def test_block_new_suppresses_existing_pending_fill_without_reexposure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            owner = _BrokerOwner(harness.config)
            pending_at = datetime(2026, 7, 23, 12, 0, 3, tzinfo=UTC)
            with patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
                broker_datetime.now.return_value = pending_at
                order_id = owner.broker.limit_order(
                    PAIR,
                    "LONG",
                    100,
                    163.0,
                    tp_pips=6,
                    sl_pips=25,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                )
            _append_gate(
                owner.broker,
                packet_sha=harness.packet["packet_sha256"],
                signal=_signal(),
                action="BLOCK_NEW",
                applied_at=datetime(2026, 7, 23, 12, 0, 4, tzinfo=UTC),
                decision_sha="1" * 64,
            )
            with patch(
                "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                return_value=ENTRY_AT,
            ), patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
                broker_datetime.now.return_value = ENTRY_AT
                events = owner._dispatch(
                    "runner",
                    "APPLY_QUOTE",
                    {
                        "pair": PAIR,
                        "bid": 162.89,
                        "ask": 162.9,
                        "ts": SIGNAL_AT,
                    },
                )
            self.assertIn(order_id, owner.broker.orders)
            self.assertEqual(owner.broker.positions, {})
            self.assertEqual(
                [event["event"] for event in events],
                ["AI_BLOCK_NEW_PENDING_FILL_REJECTED"],
            )
            owner.broker._handle.close()

    def test_close_virtual_also_blocks_pending_order_reexposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _Harness(Path(tmp))
            owner = _BrokerOwner(harness.config)
            mutation_at = datetime(2026, 7, 23, 12, 0, 3, tzinfo=UTC)
            with patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
                broker_datetime.now.return_value = mutation_at
                trade_id = owner.broker.market_order(
                    PAIR,
                    "LONG",
                    100,
                    tp_pips=6,
                    sl_pips=25,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                )
                order_id = owner.broker.limit_order(
                    PAIR,
                    "LONG",
                    100,
                    163.0,
                    tp_pips=6,
                    sl_pips=25,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                )
            _append_close_action(
                owner.broker,
                trade_id,
                applied_at=datetime(2026, 7, 23, 12, 0, 4, tzinfo=UTC),
            )
            with patch(
                "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
                return_value=ENTRY_AT,
            ), patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
                broker_datetime.now.return_value = ENTRY_AT
                events = owner._dispatch(
                    "runner",
                    "APPLY_QUOTE",
                    {
                        "pair": PAIR,
                        "bid": 162.89,
                        "ask": 162.9,
                        "ts": SIGNAL_AT,
                    },
                )
            self.assertIn(order_id, owner.broker.orders)
            self.assertEqual(owner.broker.positions, {})
            self.assertEqual(
                [event["event"] for event in events],
                ["AI_BLOCK_NEW_PENDING_FILL_REJECTED"],
            )
            owner.broker._handle.close()
