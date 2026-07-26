from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_ai_evidence_packet import (
    DOJO_AI_EVIDENCE_PACKET_CONTRACT,
    entry_signal_identity_sha256,
    verify_ai_inventory_evidence_packet,
    write_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    AllowlistedCommandModelAdapter,
    AiInventoryProducerEvidenceError,
    AiInventoryProducerMarketClosedError,
    AiInventoryProducerModelError,
    AiInventoryProducerReceiptIntegrityError,
    AiInventoryProducerResponseError,
    PRODUCER_RECEIPT_DIRECTORY,
    command_adapter_manifest_sha256,
    produce_ai_inventory_proposal,
    verify_ai_inventory_producer_receipt,
)


def _dt(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, minute, second, tzinfo=timezone.utc)


def _packet_input() -> dict[str, object]:
    recent = "2026-07-23T11:59:30Z"
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": "2026-07-23T12:00:00Z",
        "bindings": {
            "launch_preflight_token_sha256": "0" * 64,
            "git_head": "0" * 40,
            "git_branch": "codex/test-ai-inventory-producer",
            "canonical_source_root": (
                "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
            ),
            "experiment_id": "paper-ai-inventory-v1",
            "room_id": "paper-ai-inventory-room-01",
            "session_contract_sha256": "1" * 64,
            "candidate_id": "candidate-001",
            "candidate_sha256": "2" * 64,
            "spec_id": "candidate-spec-v1",
            "spec_sha256": "3" * 64,
            "policy_id": "inventory-policy-v1",
            "policy_sha256": "4" * 64,
            "paper_eligible_tip_sha256": "5" * 64,
            "ledger_sha256": "6" * 64,
            "ledger_observed_at_utc": recent,
            "state_sha256": "7" * 64,
            "state_observed_at_utc": recent,
            "snapshot_sha256": "8" * 64,
            "snapshot_observed_at_utc": recent,
        },
        "position": {
            "position_id": "T000001",
            "pair": "USD_JPY",
            "side": "LONG",
            "units": 2_000,
            "entry_price": 163.0,
            "opened_at_utc": "2026-07-23T11:10:00Z",
            "observed_at_utc": recent,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": "9" * 64,
            "take_profit": 163.3,
            "stop_loss": 162.75,
            "remaining_ceiling_seconds": 600,
            "unrealized_pl_jpy": 240.0,
            "gross_same_currency_units": 2_000,
            "net_same_currency_units": 2_000,
            "margin_used_jpy": 12_000.0,
            "capital_locked_jpy": 12_000.0,
            "same_direction_position_count": 1,
        },
        "entry_signal": None,
        "quote": {
            "pair": "USD_JPY",
            "bid": 163.12,
            "ask": 163.13,
            "timestamp_utc": recent,
            "source_sha256": "a" * 64,
            "max_age_seconds": 120,
        },
        "candles": [
            {
                "pair": "USD_JPY",
                "granularity": "M1",
                "started_at_utc": "2026-07-23T11:58:00Z",
                "completed_at_utc": "2026-07-23T11:59:00Z",
                "bid_o": 163.1,
                "bid_h": 163.15,
                "bid_l": 163.05,
                "bid_c": 163.12,
                "ask_o": 163.11,
                "ask_h": 163.16,
                "ask_l": 163.06,
                "ask_c": 163.13,
                "source_sha256": "b" * 64,
                "max_age_seconds": 3_600,
            }
        ],
        "news_items": [],
        "calendar_items": [],
        "cross_asset_items": [],
        "dynamic_binding_max_age_seconds": 120,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _response(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "action": "HOLD",
        "reason_code": "THESIS_ALIVE",
        "reason": "Bound evidence does not falsify the current thesis.",
        "virtual_units": None,
        "confidence": 0.7,
    }
    value.update(overrides)
    return value


def _flat_packet_input() -> dict[str, object]:
    value = _packet_input()
    position = value["position"]
    position.update(
        {
            "position_id": "FLAT:USD_JPY",
            "side": "FLAT",
            "units": 0.0,
            "entry_price": None,
            "opened_at_utc": None,
            "take_profit": None,
            "stop_loss": None,
            "remaining_ceiling_seconds": 0,
            "unrealized_pl_jpy": 0.0,
            "gross_same_currency_units": 0.0,
            "net_same_currency_units": 0.0,
            "margin_used_jpy": 0.0,
            "capital_locked_jpy": 0.0,
            "same_direction_position_count": 0,
        }
    )
    signal: dict[str, object] = {
        "pair": "USD_JPY",
        "side": "LONG",
        "order_type": "LIMIT",
        "units": 2_000.0,
        "price": 163.0,
        "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
        "entry_context_sha256": "9" * 64,
        "tp_pips": 30.0,
        "sl_pips": 25.0,
        "observed_at_utc": "2026-07-23T11:59:30Z",
    }
    signal["signal_identity_sha256"] = entry_signal_identity_sha256(signal)
    value["entry_signal"] = signal
    return value


class DojoAiInventoryProducerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._runtime = tempfile.TemporaryDirectory()
        self.addCleanup(self._runtime.cleanup)
        self._runtime_root = Path(self._runtime.name).resolve()
        self._room_root = self._runtime_root / "paper-ai-inventory-room"
        self._room_root.mkdir()
        self._adapter_count = 0
        self._trusted_manifests: dict[str, dict[str, object]] = {}

    def _verified_packet(self) -> dict[str, object]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repository = Path(temporary.name).resolve()
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            path = write_ai_inventory_evidence_packet(repository, _packet_input())
        return verify_ai_inventory_evidence_packet(repository, path)

    def _verified_flat_packet(self) -> dict[str, object]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repository = Path(temporary.name).resolve()
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            path = write_ai_inventory_evidence_packet(repository, _flat_packet_input())
        return verify_ai_inventory_evidence_packet(repository, path)

    def _adapter(
        self,
        response: object,
        *,
        exit_code: int = 0,
        corrupt_signature: bool = False,
        noncanonical_envelope: bool = False,
    ) -> tuple[AllowlistedCommandModelAdapter, Path, Path]:
        self._adapter_count += 1
        adapter_id = f"test-command-adapter-{self._adapter_count}"
        model_id = "allowlisted-test-model-v1"
        request_path = self._runtime_root / f"{adapter_id}-request.json"
        marker_path = self._runtime_root / f"{adapter_id}-called"
        executable = Path(sys.executable).resolve(strict=True)
        private_key = Ed25519PrivateKey.generate()
        private_key_base64 = (
            __import__("base64")
            .b64encode(
                private_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            .decode("ascii")
        )
        public_key_base64 = (
            __import__("base64")
            .b64encode(
                private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            )
            .decode("ascii")
        )
        response_text = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
        script = (
            "import base64,hashlib,json,sys\n"
            "from pathlib import Path\n"
            "from cryptography.hazmat.primitives.asymmetric.ed25519 "
            "import Ed25519PrivateKey\n"
            "adapter_id,model_id,key_b64,response_json,request_path,"
            "marker_path,exit_code,corrupt,noncanonical=sys.argv[1:]\n"
            "request=sys.stdin.buffer.read()\n"
            "Path(request_path).write_bytes(request)\n"
            "Path(marker_path).write_text('called')\n"
            "exit_code=int(exit_code)\n"
            "if exit_code: raise SystemExit(exit_code)\n"
            "body={'contract':'QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1',"
            "'adapter_id':adapter_id,'model_id':model_id,"
            "'request_sha256':hashlib.sha256(request).hexdigest(),"
            "'response':json.loads(response_json),"
            "'signature_key_id':'test-signing-key-v1'}\n"
            "payload=json.dumps(body,ensure_ascii=False,sort_keys=True,"
            "separators=(',',':'),allow_nan=False).encode()\n"
            "key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(key_b64))\n"
            "signature=bytearray(key.sign(payload))\n"
            "if corrupt=='1': signature[0]^=1\n"
            "body['signature_base64']=base64.b64encode(signature).decode()\n"
            "output=json.dumps(body,ensure_ascii=False,sort_keys=True,"
            "separators=(',',':'),allow_nan=False)\n"
            "if noncanonical=='1': output+='\\n'\n"
            "sys.stdout.write(output)\n"
        )
        executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
        executable_stat = executable.stat()
        manifest: dict[str, object] = {
            "adapter_id": adapter_id,
            "model_id": model_id,
            "executable_path": str(executable),
            "executable_sha256": executable_sha256,
            "argv": [
                str(executable),
                "-c",
                script,
                adapter_id,
                model_id,
                private_key_base64,
                response_text,
                str(request_path),
                str(marker_path),
                str(exit_code),
                "1" if corrupt_signature else "0",
                "1" if noncanonical_envelope else "0",
            ],
            "executor_uid": executable_stat.st_uid,
            "executor_gid": executable_stat.st_gid,
            "signature_key_id": "test-signing-key-v1",
            "ed25519_public_key_base64": public_key_base64,
            "timeout_seconds": 5,
        }
        manifest["command_manifest_sha256"] = command_adapter_manifest_sha256(manifest)
        self._trusted_manifests[adapter_id] = manifest
        return (
            AllowlistedCommandModelAdapter(adapter_id),
            request_path,
            marker_path,
        )

    def _invoke(
        self,
        packet: dict[str, object] | bytes,
        adapter: AllowlistedCommandModelAdapter,
        *,
        room_root: Path | None = None,
    ) -> dict[str, object]:
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_producer._utc_now",
                return_value=_dt(23, 12, 1, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
                self._trusted_manifests,
            ),
        ):
            return produce_ai_inventory_proposal(
                packet,
                adapter,
                producer_id="point-in-time-producer-v2",
                room_root=room_root or self._room_root,
            )

    def _produce(
        self,
        packet: dict[str, object] | bytes,
        response: object,
    ) -> dict[str, object]:
        adapter, _, _ = self._adapter(response)
        return self._invoke(packet, adapter)

    def _verify_receipt(self, path: Path) -> dict[str, object]:
        with patch(
            "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
            self._trusted_manifests,
        ):
            return verify_ai_inventory_producer_receipt(path.parent.parent, path)

    def test_weekend_stops_before_model_call(self) -> None:
        adapter, _, marker = self._adapter(_response())
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_producer._utc_now",
                return_value=_dt(25, 12, 0, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
                self._trusted_manifests,
            ),
        ):
            with self.assertRaises(AiInventoryProducerMarketClosedError):
                produce_ai_inventory_proposal(
                    b"not even parsed",
                    adapter,
                    producer_id="producer-v2",
                    room_root=self._room_root,
                )
        self.assertFalse(marker.exists())

    def test_arbitrary_callable_and_unknown_adapter_fail_closed(self) -> None:
        packet = self._verified_packet()
        with self.assertRaisesRegex(AiInventoryProducerModelError, "exact allowlisted"):
            produce_ai_inventory_proposal(  # type: ignore[arg-type]
                packet,
                lambda _: _response(),
                producer_id="producer-v2",
                room_root=self._room_root,
            )
        with patch(
            "quant_rabbit.dojo_ai_inventory_producer._utc_now",
            return_value=_dt(23, 12, 1, 0),
        ):
            with self.assertRaisesRegex(AiInventoryProducerModelError, "not present"):
                produce_ai_inventory_proposal(
                    packet,
                    AllowlistedCommandModelAdapter("not-allowlisted"),
                    producer_id="producer-v2",
                    room_root=self._room_root,
                )

    def test_loaded_adapter_requires_matching_lifecycle_binding(self) -> None:
        packet_input = _packet_input()
        packet_input["bindings"]["candidate_id"] = "c" * 64
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repository = Path(temporary.name).resolve()
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            path = write_ai_inventory_evidence_packet(repository, packet_input)
        packet = verify_ai_inventory_evidence_packet(repository, path)

        adapter, _, _ = self._adapter(_response())
        binding = {
            "adapter_id": adapter.adapter_id,
            "model_id": "allowlisted-test-model-v1",
            "config_sha256": "d" * 64,
            "producer_id": "point-in-time-producer-v2",
            "candidate_id": "c" * 64,
            "experiment_id": "paper-ai-inventory-v1",
            "room_id": "paper-ai-inventory-room-01",
            "future_window": {
                "start_utc": "2026-07-23T11:00:00Z",
                "end_utc": "2026-07-23T13:00:00Z",
            },
            "git_head": "0" * 40,
            "launch_preflight_token_sha256": "0" * 64,
        }
        with patch.dict(
            "quant_rabbit.dojo_ai_inventory_producer."
            "_LOADED_COMMAND_ADAPTER_BINDINGS",
            {adapter.adapter_id: binding},
            clear=True,
        ):
            result = self._invoke(packet, adapter)
        self.assertEqual(result["action"], "HOLD")

        for field, bad_value in (
            ("model_id", "other-model-v1"),
            ("producer_id", "other-producer-v1"),
            ("candidate_id", "e" * 64),
            ("experiment_id", "paper-ai-inventory-other"),
            ("room_id", "paper-ai-inventory-room-02"),
            ("git_head", "f" * 40),
            ("launch_preflight_token_sha256", "f" * 64),
        ):
            with self.subTest(field=field):
                candidate, _, marker = self._adapter(_response())
                mismatched = {
                    **binding,
                    "adapter_id": candidate.adapter_id,
                    field: bad_value,
                }
                with patch.dict(
                    "quant_rabbit.dojo_ai_inventory_producer."
                    "_LOADED_COMMAND_ADAPTER_BINDINGS",
                    {candidate.adapter_id: mismatched},
                    clear=True,
                ):
                    with self.assertRaises(AiInventoryProducerEvidenceError):
                        self._invoke(packet, candidate)
                self.assertFalse(marker.exists())

    def test_verified_allowlist_only_and_deterministic_digests(self) -> None:
        packet = self._verified_packet()
        adapter, request_path, _ = self._adapter(_response())
        first = self._invoke(packet, adapter)
        first_request = request_path.read_bytes()
        second = self._invoke(packet, adapter)
        second_request = request_path.read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(first_request, second_request)
        request = json.loads(first_request)
        self.assertEqual(
            request["evidence_packet"]["packet_sha256"],
            packet["packet_sha256"],
        )
        self.assertNotIn("path", json.dumps(request))
        binding = first["ai_decision_binding"]
        self.assertEqual(
            set(binding),
            {
                "producer_id",
                "model_id",
                "request_sha256",
                "response_sha256",
                "evidence_packet_sha256",
                "observed_at_utc",
                "producer_receipt_sha256",
                "produced_at_utc",
            },
        )
        self.assertEqual(binding["observed_at_utc"], packet["cutoff_utc"])
        self.assertEqual(
            binding["request_sha256"],
            hashlib.sha256(first_request).hexdigest(),
        )
        receipt = first["producer_receipt"]
        self.assertEqual(
            binding["producer_receipt_sha256"],
            receipt["receipt_sha256"],
        )
        self.assertEqual(binding["produced_at_utc"], receipt["produced_at_utc"])
        self.assertEqual(receipt["action"], first["action"])
        self.assertEqual(receipt["confidence"], first["confidence"])
        self.assertEqual(receipt["order_authority"], "NONE")
        self.assertFalse(receipt["live_permission"])
        self.assertEqual(
            receipt["command_invoke_receipt"]["adapter_id"],
            adapter.adapter_id,
        )
        receipt_path = (
            self._room_root
            / PRODUCER_RECEIPT_DIRECTORY
            / f"{receipt['receipt_sha256']}.json"
        )
        self.assertEqual(self._verify_receipt(receipt_path), receipt)

    def test_canonical_packet_bytes_are_accepted(self) -> None:
        packet = self._verified_packet()
        raw = (
            json.dumps(
                packet,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        result = self._produce(raw, _response())
        self.assertEqual(result["action"], "HOLD")

    def test_tampered_packet_is_rejected_without_model_call(self) -> None:
        packet = self._verified_packet()
        packet["quote"]["bid"] = 1.0
        adapter, _, marker = self._adapter(_response())
        with self.assertRaises(AiInventoryProducerEvidenceError):
            self._invoke(packet, adapter)
        self.assertFalse(marker.exists())

    def test_future_row_is_rejected_without_model_call(self) -> None:
        packet = self._verified_packet()
        packet["candles"][0]["completed_at_utc"] = "2026-07-23T12:00:01Z"
        body = {key: value for key, value in packet.items() if key != "packet_sha256"}
        packet["packet_sha256"] = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        adapter, _, marker = self._adapter(_response())
        with self.assertRaises(AiInventoryProducerEvidenceError):
            self._invoke(packet, adapter)
        self.assertFalse(marker.exists())

    def test_stale_packet_is_rejected_without_model_call(self) -> None:
        packet = self._verified_packet()
        adapter, _, marker = self._adapter(_response())
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_producer._utc_now",
                return_value=_dt(23, 12, 10, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
                self._trusted_manifests,
            ),
        ):
            with self.assertRaisesRegex(AiInventoryProducerEvidenceError, "stale"):
                produce_ai_inventory_proposal(
                    packet,
                    adapter,
                    producer_id="point-in-time-producer-v2",
                    room_root=self._room_root,
                )
        self.assertFalse(marker.exists())

    def test_unknown_response_field_and_arbitrary_prose_fail(self) -> None:
        packet = self._verified_packet()
        unknown = _response(debug="untrusted")
        with self.assertRaises(AiInventoryProducerResponseError):
            self._produce(packet, unknown)
        with self.assertRaises(AiInventoryProducerResponseError):
            self._produce(packet, "close everything now")

    def test_invalid_actions_units_and_confidence_fail(self) -> None:
        packet = self._verified_packet()
        invalid = (
            _response(action="SELL"),
            _response(virtual_units=1),
            _response(action="REDUCE_VIRTUAL", virtual_units=0.0),
            _response(action="REDUCE_VIRTUAL", virtual_units=2_000),
            _response(action="CLOSE_VIRTUAL", virtual_units=1_000),
            _response(confidence=-0.1),
            _response(confidence=1.1),
        )
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(AiInventoryProducerResponseError):
                    self._produce(packet, response)

    def test_reduce_and_close_are_only_bounded_proposals(self) -> None:
        packet = self._verified_packet()
        reduced = self._produce(
            packet,
            _response(
                action="REDUCE_VIRTUAL",
                reason_code="CAPITAL_LOCK",
                virtual_units=500,
            ),
        )
        closed = self._produce(
            packet,
            _response(
                action="CLOSE_VIRTUAL",
                reason_code="THESIS_INVALIDATED",
                virtual_units=2_000,
            ),
        )
        self.assertEqual(reduced["virtual_units"], 500)
        self.assertEqual(closed["virtual_units"], 2_000)
        self.assertIs(type(reduced["virtual_units"]), float)
        self.assertIs(type(closed["virtual_units"]), float)
        self.assertEqual(
            reduced["ai_decision_binding"]["evidence_packet_sha256"],
            packet["packet_sha256"],
        )

    def test_flat_v2_packet_only_allows_zero_inventory_proposals(self) -> None:
        packet = self._verified_flat_packet()
        allowed = (
            _response(action="BLOCK_NEW", reason_code="REGIME_MISMATCH"),
            _response(
                action="ALLOW_NEW_VIRTUAL",
                reason_code="REGIME_MATCH",
            ),
        )
        for response in allowed:
            with self.subTest(action=response["action"]):
                adapter, request_path, _ = self._adapter(response)
                result = self._invoke(packet, adapter)
                self.assertIsNone(result["virtual_units"])
                request = json.loads(request_path.read_bytes())
                self.assertEqual(
                    request["evidence_packet"]["entry_signal"][
                        "signal_identity_sha256"
                    ],
                    packet["entry_signal"]["signal_identity_sha256"],
                )
                self.assertEqual(
                    result["producer_receipt"]["entry_signal_identity_sha256"],
                    packet["entry_signal"]["signal_identity_sha256"],
                )

        disallowed = (
            _response(action="HOLD"),
            _response(action="REDUCE_VIRTUAL", virtual_units=1),
            _response(action="CLOSE_VIRTUAL"),
        )
        for response in disallowed:
            with self.subTest(action=response["action"]):
                with self.assertRaises(AiInventoryProducerResponseError):
                    self._produce(packet, response)

    def test_open_position_rejects_entry_admission_actions(self) -> None:
        packet = self._verified_packet()
        for action in ("BLOCK_NEW", "ALLOW_NEW_VIRTUAL"):
            with self.subTest(action=action):
                with self.assertRaises(AiInventoryProducerResponseError):
                    self._produce(
                        packet,
                        _response(action=action, reason_code="INVALID"),
                    )

    def test_command_failure_and_bad_signature_fail_closed(self) -> None:
        packet = self._verified_packet()
        failing, _, failing_marker = self._adapter(_response(), exit_code=3)
        with self.assertRaises(AiInventoryProducerModelError):
            self._invoke(packet, failing)
        self.assertTrue(failing_marker.exists())

        unsigned, _, unsigned_marker = self._adapter(
            _response(), corrupt_signature=True
        )
        with self.assertRaisesRegex(AiInventoryProducerModelError, "signature"):
            self._invoke(packet, unsigned)
        self.assertTrue(unsigned_marker.exists())

        noncanonical, _, noncanonical_marker = self._adapter(
            _response(),
            noncanonical_envelope=True,
        )
        with self.assertRaisesRegex(AiInventoryProducerModelError, "not canonical"):
            self._invoke(packet, noncanonical)
        self.assertTrue(noncanonical_marker.exists())

    def test_producer_clock_before_packet_is_rejected(self) -> None:
        packet = self._verified_packet()
        adapter, _, marker = self._adapter(_response())
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_producer._utc_now",
                return_value=_dt(23, 11, 59, 59),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_producer." "_TRUSTED_COMMAND_ADAPTERS",
                self._trusted_manifests,
            ),
        ):
            with self.assertRaises(AiInventoryProducerEvidenceError):
                produce_ai_inventory_proposal(
                    packet,
                    adapter,
                    producer_id="producer-v2",
                    room_root=self._room_root,
                )
        self.assertFalse(marker.exists())

    def test_receipt_is_written_before_return_and_retry_is_idempotent(self) -> None:
        packet = self._verified_packet()
        adapter, _, _ = self._adapter(_response())
        first = self._invoke(packet, adapter)
        second = self._invoke(packet, adapter)
        self.assertEqual(first, second)
        receipt = first["producer_receipt"]
        path = (
            self._room_root
            / PRODUCER_RECEIPT_DIRECTORY
            / f"{receipt['receipt_sha256']}.json"
        )
        self.assertTrue(path.is_file())
        self.assertEqual(self._verify_receipt(path), receipt)

    def test_receipt_tamper_blocks_verify_and_identical_production(self) -> None:
        packet = self._verified_packet()
        adapter, _, _ = self._adapter(_response())
        result = self._invoke(packet, adapter)
        receipt = result["producer_receipt"]
        path = (
            self._room_root
            / PRODUCER_RECEIPT_DIRECTORY
            / f"{receipt['receipt_sha256']}.json"
        )
        tampered = dict(receipt)
        tampered["reason"] = "Tampered after write."
        path.write_text(
            json.dumps(
                tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._verify_receipt(path)
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._invoke(packet, adapter)

    def test_receipt_symlink_paths_fail_closed(self) -> None:
        result = self._produce(self._verified_packet(), _response())
        receipt = result["producer_receipt"]
        path = (
            self._room_root
            / PRODUCER_RECEIPT_DIRECTORY
            / f"{receipt['receipt_sha256']}.json"
        )
        symlink = path.parent / f"{'0' * 64}.json"
        symlink.symlink_to(path)
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._verify_receipt(symlink)

        unsafe_room = self._runtime_root / "unsafe-room"
        unsafe_room.mkdir()
        outside = self._runtime_root / "outside"
        outside.mkdir()
        (unsafe_room / PRODUCER_RECEIPT_DIRECTORY).symlink_to(
            outside, target_is_directory=True
        )
        adapter, _, _ = self._adapter(_response())
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._invoke(self._verified_packet(), adapter, room_root=unsafe_room)

    def test_duplicate_and_noncanonical_receipts_fail_closed(self) -> None:
        result = self._produce(self._verified_packet(), _response())
        receipt = result["producer_receipt"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name, raw in (
                (
                    "duplicate",
                    (
                        '{"action":"HOLD",'
                        + json.dumps(
                            receipt,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )[1:]
                        + "\n"
                    ),
                ),
                (
                    "noncanonical",
                    json.dumps(
                        receipt,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                ),
            ):
                with self.subTest(name=name):
                    room_root = root / name
                    receipt_root = room_root / PRODUCER_RECEIPT_DIRECTORY
                    receipt_root.mkdir(parents=True)
                    path = receipt_root / f"{receipt['receipt_sha256']}.json"
                    path.write_text(raw)
                    with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
                        self._verify_receipt(path)

    def test_directly_synthesized_receipt_and_wrong_filename_fail(self) -> None:
        result = self._produce(self._verified_packet(), _response())
        receipt = result["producer_receipt"]
        forged = json.loads(json.dumps(receipt))
        forged["action"] = "CLOSE_VIRTUAL"
        forged["virtual_units"] = 2_000.0
        forged_body = {
            key: value for key, value in forged.items() if key != "receipt_sha256"
        }
        forged["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                forged_body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        room_root = self._runtime_root / "forged"
        receipt_root = room_root / PRODUCER_RECEIPT_DIRECTORY
        receipt_root.mkdir(parents=True)
        forged_path = receipt_root / f"{forged['receipt_sha256']}.json"
        forged_path.write_text(
            json.dumps(
                forged,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._verify_receipt(forged_path)

        valid_path = (
            self._room_root
            / PRODUCER_RECEIPT_DIRECTORY
            / f"{receipt['receipt_sha256']}.json"
        )
        wrong = valid_path.with_name(f"{'f' * 64}.json")
        wrong.write_bytes(valid_path.read_bytes())
        with self.assertRaises(AiInventoryProducerReceiptIntegrityError):
            self._verify_receipt(wrong)


if __name__ == "__main__":
    unittest.main()
