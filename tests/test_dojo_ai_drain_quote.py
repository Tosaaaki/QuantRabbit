from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_ai_drain_quote import (
    BROKER_LEDGER_NAME,
    BROKER_SNAPSHOT_NAME,
    DRAIN_QUOTE_RECEIPT_CONTRACT,
    RUNNER_HMAC_KEY_ENV,
    AiDrainQuoteError,
    AiDrainQuoteMarketClosedError,
    capture_registered_ai_drain_quote,
    verify_ai_drain_quote_receipt,
)
from quant_rabbit.dojo_ai_inventory_broker_service import BROKER_STATE_CONTRACT
from quant_rabbit.dojo_ai_inventory_session import (
    SESSION_CONFIG_CONTRACT,
    SESSION_CONTRACT_NAME,
    SESSION_LIFECYCLE_CONTRACT,
    SESSION_LIFECYCLE_NAME,
    SESSION_STATE_CONTRACT,
    SESSION_STATE_NAME,
    AIInventorySessionContext,
    _build_session_contract,
    _drain_quote_receipt_sha,
    ai_inventory_session_config_from_mapping,
    session_config_sha256,
)
from quant_rabbit.dojo_ai_source_adapters import (
    OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
    SOURCE_ADAPTER_CONFIG_CONTRACT,
    canonical_source_adapter_config_bytes,
    seal_source_adapter_config,
    source_adapter_capture_binding,
)
from quant_rabbit.dojo_ai_source_capture import (
    CAPTURE_PRIVATE_KEY_ENV,
    CAPTURE_ROOT,
    SOURCE_CAPTURE_MANIFEST_CONTRACT,
    source_capture_manifest_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import (
    CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT,
)
from quant_rabbit.models import Quote


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
RUNNER_KEY = b"runner-hmac-key-for-drain-test-000"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _utc(day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, minute, second, tzinfo=timezone.utc)


class DojoAiDrainQuoteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_context = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_context.name)
        self.repository = self.root / "repo"
        self.repository.mkdir()
        self.repository.joinpath(".git").mkdir()
        self.key = Ed25519PrivateKey.generate()
        self.key_path = self.root / "capture-key.pem"
        self.key_path.write_bytes(
            self.key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        self.key_path.chmod(0o600)
        self.mapping = self._mapping()
        self.config = ai_inventory_session_config_from_mapping(
            self.repository.resolve(),
            self.mapping,
        )
        self.room_root = (
            self.repository
            / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT
            / self.config.experiment_id
            / self.config.room_id
        )
        self.room_root.mkdir(parents=True)
        self.preflight = self._install_capture_manifest()
        self.session_contract = _build_session_contract(
            self.config,
            self.preflight,
            screen_name=f"qr-dojo-{self.config.room_id}",
            process_argv=(
                "python3.12",
                "scripts/run-dojo-ai-inventory-room.py",
            ),
        )
        self._write_json(
            self.room_root / SESSION_CONTRACT_NAME,
            self.session_contract,
        )
        self.lifecycle_tip = self._write_lifecycle()
        self._write_state(positions=1, orders=0, status="DRAINING")
        self.ledger_tip = self._write_broker_ledger()
        self._write_broker_snapshot(positions=1, orders=0)
        self.context = AIInventorySessionContext(
            config=self.config,
            room_root=self.room_root.resolve(),
            launch_preflight=self.preflight,
            session_contract=self.session_contract,
        )

    def tearDown(self) -> None:
        self.temp_context.cleanup()

    def _mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "contract": SESSION_CONFIG_CONTRACT,
            "experiment_id": "paper-ai-inventory-drain-test-v1",
            "room_id": "paper-ai-inventory-room-v1",
            "candidate_id": SHA_A,
            "dependency_id": "paper-ai-inventory-dependency-v1",
            "pair": "USD_JPY",
            "window_start_utc": "2026-07-23T11:00:00Z",
            "window_end_utc": "2026-07-23T12:00:00Z",
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
            "capture_deadline_seconds": 60,
            "evaluation_horizon_seconds": 3_600,
            "launch_preflight_token_sha256": SHA_D,
            **SAFETY,
        }
        value["session_config_sha256"] = session_config_sha256(value)
        return value

    def _install_capture_manifest(
        self,
        *,
        pair: str = "USD_JPY",
    ) -> dict[str, object]:
        adapter_config = seal_source_adapter_config(
            {
                "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
                "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
                "pair": pair,
                "max_age_seconds": 120,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        )
        config_raw = canonical_source_adapter_config_bytes(adapter_config)
        config_sha = hashlib.sha256(config_raw).hexdigest()
        config_root = self.repository / CAPTURE_ROOT / "adapter_configs"
        config_root.mkdir(parents=True, exist_ok=True)
        config_root.joinpath(f"{config_sha}.json").write_bytes(config_raw)
        public_raw = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        binding = source_adapter_capture_binding(adapter_config)
        manifest_body = {
            "contract": SOURCE_CAPTURE_MANIFEST_CONTRACT,
            "manifest_id": "paper-ai-drain-quote-capture-v1",
            "capture_key_id": "paper-ai-drain-quote-key-v1",
            "ed25519_public_key_base64": base64.b64encode(public_raw).decode("ascii"),
            "allowed_source_roles": ["quote"],
            "allowed_provider_kinds": [binding["provider_kind"]],
            "source_adapters": [binding],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": source_capture_manifest_sha256(manifest_body),
        }
        manifest_raw = _canonical(manifest) + b"\n"
        manifest_file_sha = hashlib.sha256(manifest_raw).hexdigest()
        manifest_root = self.repository / CAPTURE_ROOT / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_root.joinpath(f"{manifest_file_sha}.json").write_bytes(manifest_raw)
        return {
            "experiment_id": self.mapping["experiment_id"],
            "room_id": self.mapping["room_id"],
            "candidate_id": self.mapping["candidate_id"],
            "adapter_id": self.mapping["adapter_id"],
            "model_id": self.mapping["model_id"],
            "config_sha256": self.mapping["model_config_sha256"],
            "producer_id": self.mapping["producer_id"],
            "launch_preflight_token_sha256": SHA_D,
            "future_window": {
                "start_utc": self.mapping["window_start_utc"],
                "end_utc": self.mapping["window_end_utc"],
            },
            "paper_eligible_event_sha256": SHA_E,
            "future_registry_sha256": SHA_F,
            "source_capture_manifest_sha256": manifest_file_sha,
            "paper_room_launched": False,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }

    def _write_json(self, path: Path, value: object) -> None:
        path.write_bytes(_canonical(value) + b"\n")

    def _write_lifecycle(self) -> str:
        previous = "0" * 64
        rows: list[dict[str, object]] = []
        events = (
            (
                "SESSION_START",
                {
                    "status": "WAITING",
                    "session_config_sha256": (self.config.session_config_sha256),
                    "launch_preflight_token_sha256": SHA_D,
                },
            ),
            (
                "ENTRY_STOP",
                {
                    "status": "DRAINING",
                    "window_end_utc": self.config.window_end_utc,
                    "new_entries_allowed": False,
                    "force_close": False,
                    "original_ceiling_minutes": 60,
                },
            ),
        )
        for sequence, (event, payload) in enumerate(events, 1):
            body = {
                "contract": SESSION_LIFECYCLE_CONTRACT,
                "sequence": sequence,
                "previous_event_sha256": previous,
                "event": event,
                "recorded_at_utc": (
                    "2026-07-23T11:00:00Z" if sequence == 1 else "2026-07-23T12:00:00Z"
                ),
                "payload": payload,
                **SAFETY,
            }
            digest = hashlib.sha256(_canonical(body)).hexdigest()
            rows.append({**body, "event_sha256": digest})
            previous = digest
        path = self.room_root / SESSION_LIFECYCLE_NAME
        path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
        return previous

    def _write_state(
        self,
        *,
        positions: int,
        orders: int,
        status: str,
    ) -> None:
        body = {
            "contract": SESSION_STATE_CONTRACT,
            "status": status,
            "updated_at_utc": "2026-07-23T12:01:00Z",
            "lifecycle_tip_sha256": self.lifecycle_tip,
            "positions_count": positions,
            "orders_count": orders,
            "pending_evaluations": 0,
            "new_entries_allowed": False,
            "market_open": True,
            **SAFETY,
        }
        self._write_json(
            self.room_root / SESSION_STATE_NAME,
            {
                **body,
                "state_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
            },
        )

    def _write_broker_ledger(self) -> str:
        body = {
            "ts_utc": "2026-07-23T11:30:00Z",
            "event": "ENTRY_FILLED",
            "payload": {
                "trade_id": "T000001",
                "pair": "USD_JPY",
            },
            "prev_sha": "0" * 64,
        }
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        self.room_root.joinpath(BROKER_LEDGER_NAME).write_bytes(
            _canonical({**body, "sha": digest}) + b"\n"
        )
        return digest

    def _write_broker_snapshot(self, *, positions: int, orders: int) -> None:
        position_rows = [
            {
                "trade_id": "T000001",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 1_000.0,
                "entry_price": 163.0,
                "opened_ts": "2026-07-23T11:30:00Z",
                "tp_price": 163.1,
                "sl_price": 162.9,
                "strategy_tag": "W_FADE",
                "entry_context": None,
                "entry_context_sha256": None,
            }
            for _ in range(positions)
        ]
        order_rows = [
            {
                "order_id": "O000001",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 1_000.0,
                "limit_price": 162.9,
                "tp_pips": 6.0,
                "sl_pips": 25.0,
                "kind": "LIMIT",
                "strategy_tag": "W_FADE",
                "entry_context": None,
                "entry_context_sha256": None,
            }
            for _ in range(orders)
        ]
        value = {
            "contract": BROKER_STATE_CONTRACT,
            "broker": {
                "balance_jpy": 200_000.0,
                "seq": 1,
                "positions": position_rows,
                "orders": order_rows,
                "ledger_sha": self.ledger_tip,
            },
            "last_quotes": {},
            "quote_provenance": {},
        }
        body = {
            key: value[key]
            for key in (
                "contract",
                "broker",
                "last_quotes",
                "quote_provenance",
            )
        }
        value["mac"] = hmac.new(
            RUNNER_KEY,
            _canonical(body),
            hashlib.sha256,
        ).hexdigest()
        self._write_json(self.room_root / BROKER_SNAPSHOT_NAME, value)

    def _capture(self) -> tuple[dict[str, object], Mock]:
        client = Mock()
        client.quotes.return_value = {
            "USD_JPY": Quote(
                "USD_JPY",
                163.12,
                163.13,
                _utc(23, 12, 1, 20),
            )
        }
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                side_effect=(
                    _utc(23, 12, 1, 10),
                    _utc(23, 12, 1, 40),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                side_effect=(
                    _utc(23, 12, 1, 15),
                    _utc(23, 12, 1, 30),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                return_value=client,
            ),
            patch.dict(
                os.environ,
                {
                    CAPTURE_PRIVATE_KEY_ENV: str(self.key_path),
                    RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex(),
                },
            ),
        ):
            receipt = capture_registered_ai_drain_quote(
                self.context,
                "2026-07-23T12:02:00Z",
            )
        return receipt, client

    def test_signed_registered_quote_is_bound_to_drain_state(self) -> None:
        receipt, client = self._capture()
        self.assertEqual(receipt["contract"], DRAIN_QUOTE_RECEIPT_CONTRACT)
        self.assertEqual(receipt["source_role"], "quote")
        self.assertIs(receipt["drain_only"], True)
        self.assertIs(receipt["new_entries_allowed"], False)
        self.assertIs(receipt["ai_evaluation_allowed"], False)
        self.assertIs(receipt["force_close_allowed"], False)
        self.assertEqual(receipt["positions_count"], 1)
        self.assertEqual(receipt["orders_count"], 0)
        self.assertEqual(
            receipt["broker_ledger_terminal_sha256"],
            self.ledger_tip,
        )
        self.assertEqual(
            receipt["source_watermark_sha256"],
            receipt["canonical_source_sha256"],
        )
        client.quotes.assert_called_once_with(("USD_JPY",))
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
        ):
            verified = verify_ai_drain_quote_receipt(
                self.repository.resolve(),
                experiment_id=self.config.experiment_id,
                room_id=self.config.room_id,
                candidate_id=self.config.candidate_id,
                receipt_sha256=receipt["receipt_sha256"],
            )
        self.assertEqual(verified["receipt_sha256"], receipt["receipt_sha256"])
        accepted = _drain_quote_receipt_sha(
            receipt,
            self.context,
            "2026-07-23T12:02:00Z",
            self.lifecycle_tip,
            {
                "positions_count": 1,
                "orders_count": 0,
            },
        )
        self.assertEqual(accepted, receipt["receipt_sha256"])

        second, _ = self._capture()
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(
            second["previous_receipt_sha256"],
            receipt["receipt_sha256"],
        )

    def test_tampered_signature_and_missing_entry_stop_are_rejected(self) -> None:
        receipt, _ = self._capture()
        receipt_root = (
            self.repository
            / "research/data/dojo_paper_ai_inventory_v1/drain_quote/receipts"
            / self.config.experiment_id
            / self.config.room_id
        )
        path = next(receipt_root.glob("*.json"))
        value = json.loads(path.read_bytes())
        value["signature_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
        self._write_json(path, value)
        with patch(
            "quant_rabbit.dojo_ai_drain_quote."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=self.preflight,
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "signature"):
                verify_ai_drain_quote_receipt(
                    self.repository.resolve(),
                    experiment_id=self.config.experiment_id,
                    room_id=self.config.room_id,
                    candidate_id=self.config.candidate_id,
                    receipt_sha256=receipt["receipt_sha256"],
                )

        lifecycle_path = self.room_root / SESSION_LIFECYCLE_NAME
        start_only = lifecycle_path.read_bytes().splitlines()[0]
        lifecycle_path.write_bytes(start_only + b"\n")
        start = json.loads(start_only)
        self.lifecycle_tip = start["event_sha256"]
        self._write_state(positions=1, orders=0, status="DRAINING")
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "entry stop"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()

    def test_equivalent_utc_window_serializations_interoperate(self) -> None:
        contract_body = {
            key: value
            for key, value in self.session_contract.items()
            if key != "session_contract_sha256"
        }
        contract_body["window_end_utc"] = "2026-07-23T12:00:00+00:00"
        contract = {
            **contract_body,
            "session_contract_sha256": hashlib.sha256(
                _canonical(contract_body)
            ).hexdigest(),
        }
        self._write_json(self.room_root / SESSION_CONTRACT_NAME, contract)
        context = AIInventorySessionContext(
            config=self.config,
            room_root=self.room_root.resolve(),
            launch_preflight=self.preflight,
            session_contract=contract,
        )
        self.context = context
        receipt, _ = self._capture()
        self.assertEqual(
            receipt["fixed_window_end_utc"],
            "2026-07-23T12:00:00Z",
        )
        accepted = _drain_quote_receipt_sha(
            receipt,
            context,
            "2026-07-23T12:02:00+00:00",
            self.lifecycle_tip,
            {
                "positions_count": 1,
                "orders_count": 0,
            },
        )
        self.assertEqual(accepted, receipt["receipt_sha256"])

    def test_adapter_pair_mismatch_fails_before_network(self) -> None:
        mismatched_preflight = self._install_capture_manifest(pair="EUR_USD")
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=mismatched_preflight,
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex()},
            ),
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "pair binding"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()

    def test_window_weekend_and_zero_inventory_fail_before_network(self) -> None:
        cases = (
            ("window", _utc(23, 11, 59, 30), None),
            ("weekend", _utc(25, 12, 0), None),
            ("cutoff_after_close", _utc(24, 20, 59, 30), None),
            ("zero", _utc(23, 12, 1, 10), (0, 0, "DRAINING")),
            ("sealed", _utc(23, 12, 1, 10), (0, 0, "SEALED")),
        )
        for label, now, state in cases:
            with self.subTest(label=label):
                if state is not None:
                    self._write_state(
                        positions=state[0],
                        orders=state[1],
                        status=state[2],
                    )
                    self._write_broker_snapshot(
                        positions=state[0],
                        orders=state[1],
                    )
                with (
                    patch(
                        "quant_rabbit.dojo_ai_drain_quote." "_trusted_repository_root",
                        return_value=self.repository.resolve(),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_drain_quote._utc_now",
                        return_value=now,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_source_adapters." "OandaReadOnlyClient"
                    ) as client_factory,
                    patch.dict(
                        os.environ,
                        {RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex()},
                    ),
                ):
                    error = (
                        AiDrainQuoteMarketClosedError
                        if label == "weekend"
                        else AiDrainQuoteError
                    )
                    with self.assertRaises(error):
                        capture_registered_ai_drain_quote(
                            self.context,
                            (
                                "2026-07-25T12:01:00Z"
                                if label == "weekend"
                                else "2026-07-24T21:00:10Z"
                                if label == "cutoff_after_close"
                                else "2026-07-23T12:02:00Z"
                            ),
                        )
                client_factory.assert_not_called()
                if state is not None:
                    self._write_state(
                        positions=1,
                        orders=0,
                        status="DRAINING",
                    )
                    self._write_broker_snapshot(positions=1, orders=0)

    def test_ledger_snapshot_mismatch_and_key_mode_fail_before_network(
        self,
    ) -> None:
        snapshot_path = self.room_root / BROKER_SNAPSHOT_NAME
        snapshot = json.loads(snapshot_path.read_bytes())
        snapshot["broker"]["ledger_sha"] = "8" * 64
        body = {
            key: snapshot[key]
            for key in (
                "contract",
                "broker",
                "last_quotes",
                "quote_provenance",
            )
        }
        snapshot["mac"] = hmac.new(
            RUNNER_KEY,
            _canonical(body),
            hashlib.sha256,
        ).hexdigest()
        self._write_json(snapshot_path, snapshot)
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex()},
            ),
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "snapshot tip"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()

        self._write_broker_snapshot(positions=1, orders=0)
        self.key_path.chmod(0o644)
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {
                    CAPTURE_PRIVATE_KEY_ENV: str(self.key_path),
                    RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex(),
                },
            ),
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "0600"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()

    def test_tampered_snapshot_mac_and_ledger_chain_fail_before_network(
        self,
    ) -> None:
        snapshot_path = self.room_root / BROKER_SNAPSHOT_NAME
        snapshot = json.loads(snapshot_path.read_bytes())
        snapshot["broker"]["balance_jpy"] = 999_999.0
        self._write_json(snapshot_path, snapshot)
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex()},
            ),
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "authentication"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()

        self._write_broker_snapshot(positions=1, orders=0)
        ledger_path = self.room_root / BROKER_LEDGER_NAME
        ledger = json.loads(ledger_path.read_bytes())
        ledger["payload"]["trade_id"] = "TAMPERED"
        ledger_path.write_bytes(_canonical(ledger) + b"\n")
        with (
            patch(
                "quant_rabbit.dojo_ai_drain_quote._trusted_repository_root",
                return_value=self.repository.resolve(),
            ),
            patch(
                "quant_rabbit.dojo_ai_drain_quote._utc_now",
                return_value=_utc(23, 12, 1, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {RUNNER_HMAC_KEY_ENV: RUNNER_KEY.hex()},
            ),
        ):
            with self.assertRaisesRegex(AiDrainQuoteError, "ledger chain"):
                capture_registered_ai_drain_quote(
                    self.context,
                    "2026-07-23T12:02:00Z",
                )
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
