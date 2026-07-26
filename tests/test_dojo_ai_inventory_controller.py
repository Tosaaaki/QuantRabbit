from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_ai_evidence_packet import (
    DEDICATED_CANONICAL_SOURCE_ROOT,
    DOJO_AI_EVIDENCE_PACKET_CONTRACT,
    entry_signal_identity_sha256,
    write_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory_broker_service import (
    BrokerServiceConfig,
    DojoAIInventoryEntryClient,
    DojoAIInventoryRunnerClient,
    _TEST_ONLY_RAW_QUOTES_CAPABILITY,
    derive_broker_socket_path,
    serve_ai_inventory_broker,
)
from quant_rabbit.dojo_ai_inventory_controller import (
    AIInventoryControllerConfig,
    AIInventoryControllerIntegrityError,
    AIInventoryControllerMarketClosedError,
    CYCLE_LEDGER_NAME,
    DECISION_LEDGER_NAME,
    run_ai_inventory_cycle,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    AllowlistedCommandModelAdapter,
    command_adapter_manifest_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import canonical_paper_ai_rooms_root


UTC = timezone.utc
NOW = datetime(2026, 7, 23, 12, 0, 20, tzinfo=UTC)
BROKER_NOW = datetime(2026, 7, 23, 12, 0, 21, tzinfo=UTC)
WEEKEND = datetime(2026, 7, 25, 12, 0, 20, tzinfo=UTC)
EXPERIMENT_ID = "paper-ai-inventory-controller-e2e"
ROOM_ID = "paper-ai-inventory-controller-room-01"
CANDIDATE_ID = "c" * 64
SPEC_ID = "controller-spec-v1"
SPEC_SHA = "3" * 64
POLICY_ID = "controller-policy-v1"
POLICY_SHA = "4" * 64
PAPER_ELIGIBLE_SHA = "5" * 64
PREFLIGHT_SHA = "a" * 64
PAIR = "USD_JPY"
ENTRY_CONTEXT_SHA = "9" * 64
BOT_KEY = b"controller-bot-" + b"b" * 32
RUNNER_KEY = b"controller-runner-" + b"r" * 32


class _BrokerDateTime(datetime):
    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        return BROKER_NOW if tz is not None else BROKER_NOW.replace(tzinfo=None)


class _LoseFirstApplyReply:
    """Delegate to the real service, then simulate one lost RPC response."""

    def __init__(self, inner: DojoAIInventoryRunnerClient) -> None:
        self.inner = inner
        self.lost = False

    def health(self) -> dict[str, object]:
        return self.inner.health()

    @property
    def positions(self) -> dict[str, object]:
        return self.inner.positions

    @property
    def quotes(self) -> dict[str, list[object]]:
        return self.inner.quotes

    @property
    def quote_provenance(self) -> dict[str, dict[str, object]]:
        return self.inner.quote_provenance

    def decision_status(self, decision_sha256: str) -> dict[str, object]:
        return self.inner.decision_status(decision_sha256)

    def apply_ai_decision(
        self, decision: dict[str, object], runtime: dict[str, object]
    ) -> dict[str, object]:
        receipt = self.inner.apply_ai_decision(decision, runtime)
        if not self.lost:
            self.lost = True
            raise RuntimeError("simulated lost broker response")
        return receipt


def _serve(
    config: BrokerServiceConfig,
    lifecycle: dict[str, object],
    manifest: dict[str, object],
) -> None:
    with (
        patch(
            "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
            return_value=BROKER_NOW,
        ),
        patch(
            "quant_rabbit.dojo_ai_inventory_consumer._utc_now",
            return_value=BROKER_NOW,
        ),
        patch(
            "quant_rabbit.dojo_ai_inventory_consumer."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=lifecycle,
        ),
        patch(
            "quant_rabbit.dojo_ai_inventory_producer._TRUSTED_COMMAND_ADAPTERS",
            {"controller-fake-signed-model": manifest},
        ),
        patch("quant_rabbit.virtual_broker.datetime", _BrokerDateTime),
        patch(
            "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
            return_value=BROKER_NOW,
        ),
    ):
        serve_ai_inventory_broker(config)


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _entry_context(strategy_tag: str) -> dict[str, object]:
    return {
        "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
        "strategy_tag": strategy_tag,
        "trend_24h": "UP",
        "change_24h": 0.1,
        "change_6h": 0.02,
        "efficiency_6h": 0.4,
        "atr": 0.08,
    }


def _signal(strategy_tag: str) -> dict[str, object]:
    context_sha = _sha(_entry_context(strategy_tag))
    body: dict[str, object] = {
        "pair": PAIR,
        "side": "LONG",
        "order_type": "MARKET",
        "units": 100.0,
        "price": None,
        "strategy_tag": strategy_tag,
        "entry_context_sha256": context_sha,
        "tp_pips": 6.0,
        "sl_pips": 25.0,
        "observed_at_utc": "2026-07-23T12:00:00Z",
    }
    return {**body, "signal_identity_sha256": entry_signal_identity_sha256(body)}


def _position(
    *,
    side: str,
    units: float,
    strategy_tag: str,
    entry_context_sha256: str,
) -> dict[str, object]:
    flat = side == "FLAT"
    return {
        "position_id": f"FLAT:{PAIR}" if flat else "T000001",
        "pair": PAIR,
        "side": side,
        "units": units,
        "entry_price": None if flat else 163.0,
        "opened_at_utc": None if flat else "2026-07-23T12:00:00Z",
        "observed_at_utc": "2026-07-23T12:00:00Z",
        "strategy_tag": strategy_tag,
        "entry_context_sha256": entry_context_sha256,
        "take_profit": None if flat else 163.6,
        "stop_loss": None if flat else 162.75,
        "remaining_ceiling_seconds": 0 if flat else 3_500,
        "unrealized_pl_jpy": 0.0,
        "gross_same_currency_units": 0.0 if flat else units,
        "net_same_currency_units": 0.0 if flat else units,
        "margin_used_jpy": 0.0 if flat else 652.0,
        "capital_locked_jpy": 0.0 if flat else 652.0,
        "same_direction_position_count": 0 if flat else 1,
    }


def _packet(
    *,
    position: dict[str, object],
    signal: dict[str, object] | None,
    ledger_sha256: str,
) -> dict[str, object]:
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": "2026-07-23T12:00:00Z",
        "bindings": {
            "launch_preflight_token_sha256": PREFLIGHT_SHA,
            "git_head": "b" * 40,
            "git_branch": "codex/test-controller",
            "canonical_source_root": DEDICATED_CANONICAL_SOURCE_ROOT.as_posix(),
            "experiment_id": EXPERIMENT_ID,
            "room_id": ROOM_ID,
            "session_contract_sha256": "1" * 64,
            "candidate_id": CANDIDATE_ID,
            "candidate_sha256": CANDIDATE_ID,
            "spec_id": SPEC_ID,
            "spec_sha256": SPEC_SHA,
            "policy_id": POLICY_ID,
            "policy_sha256": POLICY_SHA,
            "paper_eligible_tip_sha256": PAPER_ELIGIBLE_SHA,
            "ledger_sha256": ledger_sha256,
            "ledger_observed_at_utc": "2026-07-23T12:00:00Z",
            "state_sha256": "7" * 64,
            "state_observed_at_utc": "2026-07-23T12:00:00Z",
            "snapshot_sha256": "8" * 64,
            "snapshot_observed_at_utc": "2026-07-23T12:00:00Z",
        },
        "position": position,
        "entry_signal": signal,
        "quote": {
            "pair": PAIR,
            "bid": 162.99,
            "ask": 163.0,
            "timestamp_utc": "2026-07-23T12:00:00Z",
            "source_sha256": "d" * 64,
            "max_age_seconds": 90,
        },
        "candles": [
            {
                "pair": PAIR,
                "granularity": "M1",
                "started_at_utc": "2026-07-23T11:59:00Z",
                "completed_at_utc": "2026-07-23T12:00:00Z",
                "bid_o": 162.98,
                "bid_h": 163.0,
                "bid_l": 162.97,
                "bid_c": 162.99,
                "ask_o": 162.99,
                "ask_h": 163.01,
                "ask_l": 162.98,
                "ask_c": 163.0,
                "source_sha256": "e" * 64,
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


def _request(position_source_sha: str) -> dict[str, object]:
    roles = (
        "session_contract",
        "candidate",
        "spec",
        "policy",
        "paper_eligible_event",
        "ledger",
        "state",
        "snapshot",
        "position",
        "entry_context",
        "entry_signal",
        "quote",
        "candles",
        "news_items",
        "calendar_items",
        "cross_asset_items",
    )
    source_files = {
        role: f"{hashlib.sha256(f'{position_source_sha}:{role}'.encode()).hexdigest()}.json"
        for role in roles
    }
    source_files["position"] = f"{position_source_sha}.json"
    source_files["quote"] = f"{'d' * 64}.json"
    source_receipts = {
        role: hashlib.sha256(f"capture:{role}".encode()).hexdigest()
        for role in roles
    }
    source_receipts["quote"] = "c" * 64
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": "2026-07-23T12:00:00Z",
        "experiment_id": EXPERIMENT_ID,
        "room_id": ROOM_ID,
        "candidate_id": CANDIDATE_ID,
        "spec_id": SPEC_ID,
        "policy_id": POLICY_ID,
        "source_files": source_files,
        "source_receipts": source_receipts,
        "dynamic_binding_max_age_seconds": 90,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _lifecycle() -> dict[str, object]:
    return {
        "contract": "QR_DOJO_AI_INVENTORY_LAUNCH_PREFLIGHT_V1",
        "candidate_id": CANDIDATE_ID,
        "adapter_id": "controller-fake-signed-model",
        "model_id": "controller-fake-model-v1",
        "config_sha256": "f" * 64,
        "producer_id": "controller-producer-v1",
        "spec_sha256": SPEC_SHA,
        "policy_sha256": POLICY_SHA,
        "experiment_id": EXPERIMENT_ID,
        "room_id": ROOM_ID,
        "paper_eligible_event_sha256": PAPER_ELIGIBLE_SHA,
        "candidate_lifecycle_ledger_tip_sha256": PAPER_ELIGIBLE_SHA,
        "append_claim_sha256": "6" * 64,
        "job_manifest_sha256": "7" * 64,
        "job_owner_sha256": "8" * 64,
        "proof_artifact_sha256": "9" * 64,
        "proof_artifact_bytes_sha256": "a" * 64,
        "proof_manifest_sha256": "b" * 64,
        "source_manifest_sha256s": {"TRAIN": "c" * 64},
        "future_registry_sha256": "d" * 64,
        "future_window": {
            "start_utc": "2026-07-23T11:00:00Z",
            "end_utc": "2026-07-23T13:00:00Z",
        },
        "git_head": "b" * 40,
        "git_head_sha256": "e" * 64,
        "issued_at_utc": "2026-07-23T10:00:00Z",
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "paper_room_launched": False,
        "launch_preflight_token_sha256": PREFLIGHT_SHA,
    }


def _model_manifest() -> dict[str, object]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    adapter_id = "controller-fake-signed-model"
    model_id = "controller-fake-model-v1"
    key_id = "controller-ephemeral-test-key"
    executable = Path(sys.executable).resolve(strict=True)
    script = (
        "import base64,hashlib,json,sys\n"
        "from cryptography.hazmat.primitives.asymmetric.ed25519 "
        "import Ed25519PrivateKey\n"
        "adapter,model,key_id,key_b64=sys.argv[1:]\n"
        "request=json.load(sys.stdin)\n"
        "packet=request['evidence_packet']; pos=packet['position']; "
        "signal=packet['entry_signal']\n"
        "if pos['side']=='FLAT':\n"
        " action=('BLOCK_NEW' if signal['strategy_tag'].endswith('BLOCK') "
        "else 'ALLOW_NEW_VIRTUAL'); units=None\n"
        "elif pos['units']>60.0:\n"
        " action='REDUCE_VIRTUAL'; units=40.0\n"
        "else:\n"
        " action='CLOSE_VIRTUAL'; units=pos['units']\n"
        "response={'action':action,'reason_code':'POINT_IN_TIME_TEST',"
        "'reason':'Signed point-in-time paper inventory decision.',"
        "'virtual_units':units,'confidence':0.8}\n"
        "body={'contract':'QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1',"
        "'adapter_id':adapter,'model_id':model,"
        "'request_sha256':hashlib.sha256(json.dumps(request,ensure_ascii=False,"
        "sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),"
        "'response':response,'signature_key_id':key_id}\n"
        "payload=json.dumps(body,ensure_ascii=False,sort_keys=True,"
        "separators=(',',':'),allow_nan=False).encode()\n"
        "key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(key_b64))\n"
        "body['signature_base64']=base64.b64encode(key.sign(payload)).decode()\n"
        "json.dump(body,sys.stdout,ensure_ascii=False,sort_keys=True,"
        "separators=(',',':'),allow_nan=False)\n"
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
            key_id,
            base64.b64encode(private_bytes).decode(),
        ],
        "executor_uid": item_stat.st_uid,
        "executor_gid": item_stat.st_gid,
        "signature_key_id": key_id,
        "ed25519_public_key_base64": base64.b64encode(public_bytes).decode(),
        "timeout_seconds": 10,
    }
    manifest["command_manifest_sha256"] = command_adapter_manifest_sha256(manifest)
    return manifest


class AIInventoryControllerE2ETest(unittest.TestCase):
    def setUp(self) -> None:
        self.rpc_clock = patch(
            "quant_rabbit.dojo_ai_inventory_broker_service._utc_now",
            return_value=NOW,
        )
        self.rpc_clock.start()
        self.addCleanup(self.rpc_clock.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = Path(self.temp.name)
        (self.repository / ".git").mkdir()
        self.rooms_root = canonical_paper_ai_rooms_root(self.repository)
        self.room_root = self.rooms_root / EXPERIMENT_ID / ROOM_ID
        self.room_root.mkdir(parents=True)
        self.ledger = self.room_root / "broker_ledger.jsonl"
        self.state = self.room_root / "broker_state.json"
        self.decision_ledger = self.room_root / DECISION_LEDGER_NAME
        self.socket = derive_broker_socket_path(self.ledger)
        self.lifecycle = _lifecycle()
        self.manifest = _model_manifest()
        self.service_config = BrokerServiceConfig(
            socket_path=self.socket,
            ledger_path=self.ledger,
            state_path=self.state,
            repository_root=self.repository,
            room_id=ROOM_ID,
            candidate_id=CANDIDATE_ID,
            bot_hmac_key=BOT_KEY,
            runner_hmac_key=RUNNER_KEY,
            decision_ledger_path=self.decision_ledger,
            producer_receipt_path=None,
            allow_test_only_raw_quotes=True,
            _test_only_capability=_TEST_ONLY_RAW_QUOTES_CAPABILITY,
        )
        context = multiprocessing.get_context("fork")
        self.process = context.Process(
            target=_serve,
            args=(self.service_config, self.lifecycle, self.manifest),
        )
        self.process.start()
        self.addCleanup(self._stop_service)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self.socket.exists():
            time.sleep(0.02)
        self.assertTrue(self.socket.exists())
        self.runner = DojoAIInventoryRunnerClient(self.socket, RUNNER_KEY)
        self.bot = DojoAIInventoryEntryClient(self.socket, BOT_KEY)
        self.runner.apply_quote(PAIR, 162.99, 163.0, "2026-07-23T12:00:00Z")
        self.config = AIInventoryControllerConfig(
            repository_root=self.repository,
            experiment_id=EXPERIMENT_ID,
            room_id=ROOM_ID,
            adapter_id="controller-fake-signed-model",
            model_id="controller-fake-model-v1",
            adapter_config_sha256="f" * 64,
            producer_id="controller-producer-v1",
        )

    def _stop_service(self) -> None:
        if not hasattr(self, "process") or not self.process.is_alive():
            return
        runner = getattr(self, "runner", None)
        if runner is not None:
            try:
                runner.shutdown()
            except Exception:
                self.process.terminate()
        else:
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)

    def _write_packet(self, value: dict[str, object]) -> Path:
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=datetime(2026, 7, 23, 12, 0, 1, tzinfo=UTC),
        ):
            return write_ai_inventory_evidence_packet(self.repository, value)

    def test_signed_model_controls_entry_reduce_close_and_block(self) -> None:
        allow_tag = "QR_DOJO_AI_CONTROLLER_ALLOW"
        block_tag = "QR_DOJO_AI_CONTROLLER_BLOCK"
        allow_signal = _signal(allow_tag)
        flat_allow = _packet(
            position=_position(
                side="FLAT",
                units=0.0,
                strategy_tag=allow_tag,
                entry_context_sha256=allow_signal["entry_context_sha256"],
            ),
            signal=allow_signal,
            ledger_sha256="0" * 64,
        )
        allow_path = self._write_packet(flat_allow)

        block_signal = _signal(block_tag)
        paths = [allow_path]
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._trusted_repository_root",
                return_value=self.repository.resolve(strict=True),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._utc_now",
                return_value=NOW,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.lifecycle,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "write_trusted_ai_inventory_evidence_packet",
                side_effect=lambda _request: paths.pop(0),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "load_sealed_command_model_adapter",
                return_value=AllowlistedCommandModelAdapter(
                    "controller-fake-signed-model"
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=NOW,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer._utc_now",
                return_value=NOW,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
                {"controller-fake-signed-model": self.manifest},
            ),
            patch.object(
                DojoAIInventoryRunnerClient,
                "quote_provenance",
                new_callable=PropertyMock,
                return_value={
                    PAIR: {
                        "pair": PAIR,
                        "bid": 162.99,
                        "ask": 163.0,
                        "timestamp_utc": "2026-07-23T12:00:00Z",
                        "capture_source_sha256": "d" * 64,
                        "acquisition_receipt_sha256": "c" * 64,
                        "quote_watermark_sha256": "f" * 64,
                        "test_only_raw_quote": False,
                    }
                },
            ),
        ):
            lost_reply = _LoseFirstApplyReply(self.runner)
            allow = run_ai_inventory_cycle(self.config, _request("1" * 64), lost_reply)
            self.assertTrue(lost_reply.lost)
            self.assertEqual(allow.decision["action"], "ALLOW_NEW_VIRTUAL")
            self.assertIsNotNone(allow.admission_reference)
            trade_id = self.bot.market_order(
                PAIR,
                "LONG",
                100.0,
                tp_pips=6.0,
                sl_pips=25.0,
                strategy_tag=allow_tag,
                entry_context=_entry_context(allow_tag),
                ai_admission=allow.admission_reference,
            )
            self.assertEqual(trade_id, "T000001")
            broker_context_sha = self.runner.positions[trade_id]["entry_context_sha256"]
            paths.append(
                self._write_packet(
                    _packet(
                        position=_position(
                            side="LONG",
                            units=100.0,
                            strategy_tag=allow_tag,
                            entry_context_sha256=broker_context_sha,
                        ),
                        signal=None,
                        ledger_sha256="1" * 64,
                    )
                )
            )

            reduce = run_ai_inventory_cycle(
                self.config, _request("2" * 64), self.runner
            )
            self.assertEqual(reduce.decision["action"], "REDUCE_VIRTUAL")
            self.assertEqual(self.runner.positions[trade_id]["units"], 60.0)
            paths.append(
                self._write_packet(
                    _packet(
                        position=_position(
                            side="LONG",
                            units=60.0,
                            strategy_tag=allow_tag,
                            entry_context_sha256=broker_context_sha,
                        ),
                        signal=None,
                        ledger_sha256="2" * 64,
                    )
                )
            )

            close = run_ai_inventory_cycle(self.config, _request("3" * 64), self.runner)
            self.assertEqual(close.decision["action"], "CLOSE_VIRTUAL")
            self.assertEqual(self.runner.positions, {})
            paths.append(
                self._write_packet(
                    _packet(
                        position=_position(
                            side="FLAT",
                            units=0.0,
                            strategy_tag=block_tag,
                            entry_context_sha256=block_signal["entry_context_sha256"],
                        ),
                        signal=block_signal,
                        ledger_sha256="3" * 64,
                    )
                )
            )

            block = run_ai_inventory_cycle(self.config, _request("4" * 64), self.runner)
            self.assertEqual(block.decision["action"], "BLOCK_NEW")
            self.assertIsNone(block.admission_reference)

        cycle_rows = [
            json.loads(line)
            for line in (self.room_root / CYCLE_LEDGER_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([row["stage"] for row in cycle_rows].count("CYCLE_APPLIED"), 4)
        self.assertTrue(
            all(
                row["paper_only"] is True
                and row["order_authority"] == "NONE"
                and row["live_permission"] is False
                for row in cycle_rows
            )
        )

    def test_weekend_stops_before_evidence_or_model(self) -> None:
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._trusted_repository_root",
                return_value=self.repository.resolve(strict=True),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._utc_now",
                return_value=WEEKEND,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "write_trusted_ai_inventory_evidence_packet"
            ) as writer,
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "load_sealed_command_model_adapter"
            ) as loader,
        ):
            with self.assertRaises(AIInventoryControllerMarketClosedError):
                run_ai_inventory_cycle(self.config, _request("f" * 64), self.runner)
        writer.assert_not_called()
        loader.assert_not_called()

    def test_model_crossing_fixed_window_leaves_no_receipt_decision_or_apply(
        self,
    ) -> None:
        strategy_tag = "QR_DOJO_AI_CONTROLLER_WINDOW"
        signal = _signal(strategy_tag)
        packet_path = self._write_packet(
            _packet(
                position=_position(
                    side="FLAT",
                    units=0.0,
                    strategy_tag=strategy_tag,
                    entry_context_sha256=signal["entry_context_sha256"],
                ),
                signal=signal,
                ledger_sha256="0" * 64,
            )
        )
        lifecycle = json.loads(json.dumps(self.lifecycle))
        window_end = datetime(2026, 7, 23, 12, 0, 21, tzinfo=UTC)
        lifecycle["future_window"]["end_utc"] = (
            window_end.isoformat().replace("+00:00", "Z")
        )
        clock = [NOW]

        def cross_window(
            _packet_value: object,
            _adapter: object,
            *,
            producer_id: str,
            room_root: Path,
        ) -> dict[str, object]:
            self.assertEqual(producer_id, self.config.producer_id)
            staged = room_root / "producer_receipts"
            staged.mkdir()
            (staged / f"{'1' * 64}.json").write_text(
                '{"staged":true}\n',
                encoding="utf-8",
            )
            clock[0] = window_end
            return {
                "producer_receipt": {"receipt_sha256": "1" * 64},
            }

        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._trusted_repository_root",
                return_value=self.repository.resolve(strict=True),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller._utc_now",
                side_effect=lambda: clock[0],
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=lifecycle,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "write_trusted_ai_inventory_evidence_packet",
                return_value=packet_path,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "load_sealed_command_model_adapter",
                return_value=AllowlistedCommandModelAdapter(
                    "controller-fake-signed-model"
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_controller."
                "produce_ai_inventory_proposal",
                side_effect=cross_window,
            ) as producer,
            patch.object(
                DojoAIInventoryRunnerClient,
                "quote_provenance",
                new_callable=PropertyMock,
                return_value={
                    PAIR: {
                        "pair": PAIR,
                        "bid": 162.99,
                        "ask": 163.0,
                        "timestamp_utc": "2026-07-23T12:00:00Z",
                        "capture_source_sha256": "d" * 64,
                        "acquisition_receipt_sha256": "c" * 64,
                        "quote_watermark_sha256": "f" * 64,
                        "test_only_raw_quote": False,
                    }
                },
            ),
        ):
            with self.assertRaisesRegex(
                AIInventoryControllerIntegrityError,
                "outside the immutable future window",
            ):
                run_ai_inventory_cycle(
                    self.config,
                    _request("f" * 64),
                    self.runner,
                )
        producer.assert_called_once()
        canonical_receipts = self.room_root / "producer_receipts"
        self.assertFalse(canonical_receipts.exists())
        self.assertFalse(self.decision_ledger.exists())
        cycle_rows = [
            json.loads(line)
            for line in (self.room_root / CYCLE_LEDGER_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            [row["stage"] for row in cycle_rows],
            ["CYCLE_STARTED"],
        )
        self.assertEqual(self.runner.positions, {})


if __name__ == "__main__":
    unittest.main()
