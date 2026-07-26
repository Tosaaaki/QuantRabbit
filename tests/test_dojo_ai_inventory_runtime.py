from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.dojo_ai_evidence_packet import entry_signal_identity_sha256
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
)
from quant_rabbit.dojo_ai_inventory_runtime import (
    AIInventoryAdmissionIntegrityError,
    AIInventoryEntryDeniedError,
    ENTRY_ADMISSION_REFERENCE_CONTRACT,
    _BrokerOwnedAdmissionController,
    build_ai_inventory_admission_state,
)
from quant_rabbit.virtual_broker import VirtualBroker


UTC = timezone.utc
ROOM_ID = "paper-ai-inventory-room-runtime-001"
CANDIDATE_ID = "a" * 64
PAIR = "USD_JPY"
STRATEGY_TAG = "QR_DOJO_AI_INVENTORY_V2"
DECISION_SHA = "b" * 64
SIGNAL_AT = "2026-07-23T12:00:00Z"
APPLY_AT = datetime(2026, 7, 23, 12, 0, 2, tzinfo=UTC)
ENTRY_AT = datetime(2026, 7, 23, 12, 0, 5, tzinfo=UTC)


@contextmanager
def _runtime_at(value: datetime):
    with patch(
        "quant_rabbit.dojo_ai_inventory_runtime._utc_now",
        return_value=value,
    ), patch(
        "quant_rabbit.virtual_broker.datetime",
        wraps=datetime,
    ) as broker_datetime:
        broker_datetime.now.return_value = value
        yield


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


def _signal(order_type: str) -> dict[str, object]:
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


class _Fixture:
    def __init__(self, root: Path) -> None:
        room = (
            root
            / "paper-ai-inventory-runtime"
            / "paper-ai-inventory-experiment-runtime-001"
            / ROOM_ID
        )
        room.mkdir(parents=True)
        self.broker = VirtualBroker(room / "broker.jsonl", fast_ledger=False)
        self.broker.last_quotes[PAIR] = (162.99, 163.0, SIGNAL_AT)

    def apply_gate(
        self,
        action: str,
        *,
        order_type: str = "MARKET",
        pair: str = PAIR,
        strategy_tag: str = STRATEGY_TAG,
        decision_sha: str = DECISION_SHA,
        candidate_id: str = CANDIDATE_ID,
        permit_expires_at: str = "2026-07-23T12:01:00Z",
        decision_contract: str = DOJO_AI_INVENTORY_DECISION_CONTRACT,
        cancelled_order_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        signal = _signal(order_type)
        admission = (
            {
                "evidence_packet_sha256": "f" * 64,
                "permit_expires_at_utc": permit_expires_at,
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
            "session_id": "paper-ai-inventory-experiment-runtime-001",
            "candidate_id": candidate_id,
            "policy_id": "policy-v2",
            "spec_id": "spec-v2",
            "ai_producer_id": "codex-ai",
            "ai_model_id": "gpt-test",
            "ai_request_sha256": "d" * 64,
            "ai_response_sha256": "e" * 64,
            "ai_evidence_packet_sha256": "f" * 64,
            "position_id": f"FLAT:{pair}",
            "pair": pair,
            "strategy_tag": strategy_tag,
            "admission_binding": admission,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "virtual_broker_mutation_allowed": True,
            "external_broker_mutation_allowed": False,
            "decision_contract": decision_contract,
            "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
            "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
            "consume_at_utc": APPLY_AT.isoformat().replace("+00:00", "Z"),
        }
        with patch("quant_rabbit.virtual_broker.datetime") as broker_datetime:
            broker_datetime.now.return_value = APPLY_AT
            reservation = dict(common)
            self.broker._log("AI_INVENTORY_ACTION_RESERVED", reservation)
            reservation_sha = self.broker._prev_sha
            cancel_sha256s = []
            for order_id in cancelled_order_ids:
                self.broker.cancel_order(order_id)
                cancel_sha256s.append(self.broker._prev_sha)
            applied = {
                **common,
                "reservation_sha256": reservation_sha,
                "close_sha256": None,
                "realized_pl_jpy": None,
                "cancelled_order_ids": list(cancelled_order_ids),
                "cancel_sha256s": cancel_sha256s,
                "block_new": action == "BLOCK_NEW",
                "allow_new_virtual": action == "ALLOW_NEW_VIRTUAL",
                "single_use_entry_permit": action == "ALLOW_NEW_VIRTUAL",
                "entry_proxy_consumed": (
                    False if action == "ALLOW_NEW_VIRTUAL" else None
                ),
                "status": "APPLIED",
            }
            self.broker._log("AI_INVENTORY_ACTION_APPLIED", applied)
        return {
            "contract": ENTRY_ADMISSION_REFERENCE_CONTRACT,
            "applied_receipt_sha256": self.broker._prev_sha,
            "decision_sha256": decision_sha,
            "room_id": ROOM_ID,
            "candidate_id": candidate_id,
            "signal_identity_sha256": signal["signal_identity_sha256"],
        }

    def proxy(self) -> _BrokerOwnedAdmissionController:
        return _BrokerOwnedAdmissionController(
            self.broker,
            room_id=ROOM_ID,
            candidate_id=CANDIDATE_ID,
        )


class TestDojoAIInventoryAdmissionRuntime(unittest.TestCase):
    def test_three_entry_paths_require_exact_applied_permit(self) -> None:
        for order_type in ("MARKET", "LIMIT", "STOP"):
            with self.subTest(
                order_type=order_type
            ), tempfile.TemporaryDirectory() as tmp:
                fixture = _Fixture(Path(tmp))
                reference = fixture.apply_gate(
                    "ALLOW_NEW_VIRTUAL", order_type=order_type
                )
                proxy = fixture.proxy()
                kwargs = {
                    "pair": PAIR,
                    "side": "LONG",
                    "units": 100,
                    "tp_pips": 6,
                    "sl_pips": 25,
                    "strategy_tag": STRATEGY_TAG,
                    "entry_context": _context(),
                    "ai_admission": reference,
                }
                with _runtime_at(ENTRY_AT):
                    if order_type == "MARKET":
                        created = proxy.market_order(**kwargs)
                    elif order_type == "LIMIT":
                        created = proxy.limit_order(price=163, **kwargs)
                    else:
                        created = proxy.stop_order(price=163, **kwargs)
                self.assertTrue(
                    created.startswith("T" if order_type == "MARKET" else "O")
                )
                rows = [
                    json.loads(line)
                    for line in fixture.broker.ledger_path.read_text().splitlines()
                ]
                self.assertEqual(rows[-1]["event"], "AI_ENTRY_PERMIT_CONSUMED")

    def test_default_deny_and_bot_has_no_close_or_cancel_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            proxy = fixture.proxy()
            self.assertFalse(hasattr(proxy, "close_trade"))
            self.assertFalse(hasattr(proxy, "cancel_order"))
            self.assertFalse(hasattr(proxy, "set_exit"))
            with _runtime_at(ENTRY_AT), self.assertRaises(AIInventoryEntryDeniedError):
                proxy.market_order(
                    PAIR,
                    "LONG",
                    100.0,
                    tp_pips=6.0,
                    sl_pips=25.0,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                )

    def test_permit_retry_returns_same_created_id_without_duplicate_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            proxy = fixture.proxy()
            with _runtime_at(ENTRY_AT):
                created = proxy.market_order(
                    PAIR,
                    "LONG",
                    100.0,
                    tp_pips=6.0,
                    sl_pips=25.0,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                    ai_admission=reference,
                )
                before = fixture.broker.ledger_path.read_bytes()
                retried = proxy.market_order(
                    PAIR,
                    "LONG",
                    100.0,
                    tp_pips=6.0,
                    sl_pips=25.0,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                    ai_admission=reference,
                )
                self.assertEqual(retried, created)
                self.assertEqual(fixture.broker.ledger_path.read_bytes(), before)
                self.assertEqual(len(fixture.broker.positions), 1)

    def test_reserved_only_retry_executes_entry_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            proxy = fixture.proxy()
            kwargs = {
                "pair": PAIR,
                "side": "LONG",
                "units": 100.0,
                "tp_pips": 6.0,
                "sl_pips": 25.0,
                "strategy_tag": STRATEGY_TAG,
                "entry_context": _context(),
                "ai_admission": reference,
            }
            with _runtime_at(ENTRY_AT), patch.object(
                fixture.broker,
                "market_order",
                side_effect=RuntimeError("crash after permit reservation"),
            ), self.assertRaisesRegex(RuntimeError, "reservation"):
                proxy.market_order(**kwargs)
            self.assertEqual(
                json.loads(
                    fixture.broker.ledger_path.read_text().splitlines()[-1]
                )["event"],
                "AI_ENTRY_PERMIT_RESERVED",
            )

            with _runtime_at(ENTRY_AT):
                created = proxy.market_order(**kwargs)
            events = [
                json.loads(line)["event"]
                for line in fixture.broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(events[-3:], [
                "AI_ENTRY_PERMIT_RESERVED",
                "FILL_MARKET",
                "AI_ENTRY_PERMIT_CONSUMED",
            ])
            self.assertIn(created, fixture.broker.positions)

    def test_entry_durable_retry_appends_only_consumed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            proxy = fixture.proxy()
            original_log = fixture.broker._log

            def crash_before_consumed(
                event: str, payload: dict[str, object]
            ) -> None:
                if event == "AI_ENTRY_PERMIT_CONSUMED":
                    raise RuntimeError("crash after entry")
                original_log(event, payload)

            kwargs = {
                "pair": PAIR,
                "side": "LONG",
                "units": 100.0,
                "tp_pips": 6.0,
                "sl_pips": 25.0,
                "strategy_tag": STRATEGY_TAG,
                "entry_context": _context(),
                "ai_admission": reference,
            }
            with _runtime_at(ENTRY_AT), patch.object(
                fixture.broker, "_log", side_effect=crash_before_consumed
            ), self.assertRaisesRegex(RuntimeError, "after entry"):
                proxy.market_order(**kwargs)
            rows = [
                json.loads(line)
                for line in fixture.broker.ledger_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows[-2:]],
                ["AI_ENTRY_PERMIT_RESERVED", "FILL_MARKET"],
            )
            created = rows[-1]["payload"]["trade_id"]
            self.assertIn(created, fixture.broker.positions)

            with _runtime_at(ENTRY_AT):
                recovered = proxy.market_order(**kwargs)
            self.assertEqual(recovered, created)
            self.assertEqual(len(fixture.broker.positions), 1)
            self.assertEqual(
                json.loads(
                    fixture.broker.ledger_path.read_text().splitlines()[-1]
                )["event"],
                "AI_ENTRY_PERMIT_CONSUMED",
            )

    def test_scope_and_signal_argument_mismatch_deny(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            proxy = fixture.proxy()
            wrong_reference = {**reference, "candidate_id": "wrong-candidate"}
            with _runtime_at(ENTRY_AT):
                with self.assertRaises(AIInventoryEntryDeniedError):
                    proxy.market_order(
                        PAIR,
                        "LONG",
                        100.0,
                        tp_pips=6.0,
                        sl_pips=25.0,
                        strategy_tag=STRATEGY_TAG,
                        entry_context=_context(),
                        ai_admission=wrong_reference,
                    )
                with self.assertRaises(AIInventoryEntryDeniedError):
                    proxy.market_order(
                        PAIR,
                        "SHORT",
                        100.0,
                        tp_pips=6.0,
                        sl_pips=25.0,
                        strategy_tag=STRATEGY_TAG,
                        entry_context=_context(),
                        ai_admission=reference,
                    )

    def test_expired_and_weekend_permits_deny(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate(
                "ALLOW_NEW_VIRTUAL",
                permit_expires_at="2026-07-23T12:00:04Z",
            )
            proxy = fixture.proxy()
            with _runtime_at(ENTRY_AT), self.assertRaises(AIInventoryEntryDeniedError):
                proxy.market_order(
                    PAIR,
                    "LONG",
                    100.0,
                    tp_pips=6.0,
                    sl_pips=25.0,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                    ai_admission=reference,
                )

        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            reference = fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            proxy = fixture.proxy()
            weekend = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
            with _runtime_at(weekend), self.assertRaises(AIInventoryEntryDeniedError):
                proxy.market_order(
                    PAIR,
                    "LONG",
                    100.0,
                    tp_pips=6.0,
                    sl_pips=25.0,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                    ai_admission=reference,
                )

    def test_block_new_restores_gate_after_exact_pending_order_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            with patch(
                "quant_rabbit.virtual_broker.datetime",
                wraps=datetime,
            ) as broker_datetime:
                broker_datetime.now.return_value = APPLY_AT - timedelta(seconds=1)
                fixture.broker.limit_order(
                    PAIR,
                    "SHORT",
                    50.0,
                    163.5,
                    strategy_tag=STRATEGY_TAG,
                    entry_context=_context(),
                )
            order_id = next(iter(fixture.broker.orders))
            reference = fixture.apply_gate(
                "BLOCK_NEW",
                cancelled_order_ids=(order_id,),
            )
            state = build_ai_inventory_admission_state(
                fixture.broker.ledger_path,
                room_id=ROOM_ID,
                candidate_id=CANDIDATE_ID,
                as_of_utc=ENTRY_AT,
            )
            self.assertIn((PAIR, STRATEGY_TAG), state.blocked_scopes)
            self.assertEqual(fixture.broker.orders, {})
            self.assertEqual(state.available_permits, ())
            proxy = fixture.proxy()
            kwargs = {
                "pair": PAIR,
                "side": "LONG",
                "units": 100.0,
                "tp_pips": 6.0,
                "sl_pips": 25.0,
                "strategy_tag": STRATEGY_TAG,
                "entry_context": _context(),
                "ai_admission": reference,
            }
            with _runtime_at(ENTRY_AT):
                for order_type in ("MARKET", "LIMIT", "STOP"):
                    with self.subTest(blocked_order_type=order_type), self.assertRaises(
                        AIInventoryEntryDeniedError
                    ):
                        if order_type == "MARKET":
                            proxy.market_order(**kwargs)
                        elif order_type == "LIMIT":
                            proxy.limit_order(price=163.0, **kwargs)
                        else:
                            proxy.stop_order(price=163.0, **kwargs)
            self.assertEqual(fixture.broker.orders, {})

    def test_block_new_rejects_detached_pending_order_cancel_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            with patch(
                "quant_rabbit.virtual_broker.datetime",
                wraps=datetime,
            ) as broker_datetime:
                broker_datetime.now.return_value = APPLY_AT - timedelta(seconds=1)
                order_id = fixture.broker.limit_order(
                    PAIR,
                    "SHORT",
                    50.0,
                    163.5,
                    strategy_tag="ANOTHER_STRATEGY",
                )
            fixture.apply_gate(
                "BLOCK_NEW",
                cancelled_order_ids=(order_id,),
            )
            with self.assertRaises(AIInventoryAdmissionIntegrityError):
                build_ai_inventory_admission_state(
                    fixture.broker.ledger_path,
                    room_id=ROOM_ID,
                    candidate_id=CANDIDATE_ID,
                    as_of_utc=ENTRY_AT,
                )

    def test_tampered_chain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            fixture.apply_gate("ALLOW_NEW_VIRTUAL")
            rows = fixture.broker.ledger_path.read_text().splitlines()
            edited = json.loads(rows[-1])
            edited["payload"]["pair"] = "EUR_USD"
            rows[-1] = json.dumps(edited, sort_keys=True)
            fixture.broker._handle.close()
            fixture.broker.ledger_path.write_text("\n".join(rows) + "\n")
            with self.assertRaises(AIInventoryAdmissionIntegrityError):
                build_ai_inventory_admission_state(
                    fixture.broker.ledger_path,
                    room_id=ROOM_ID,
                    candidate_id=CANDIDATE_ID,
                    as_of_utc=ENTRY_AT,
                )

    def test_non_v2_applied_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            fixture.apply_gate(
                "ALLOW_NEW_VIRTUAL",
                decision_contract="QR_DOJO_AI_INVENTORY_DECISION_V1",
            )
            with self.assertRaises(AIInventoryAdmissionIntegrityError):
                build_ai_inventory_admission_state(
                    fixture.broker.ledger_path,
                    room_id=ROOM_ID,
                    candidate_id=CANDIDATE_ID,
                    as_of_utc=ENTRY_AT,
                )

    def test_fake_broker_is_rejected(self) -> None:
        class FakeBroker:
            ledger_path = Path("/tmp/paper-ai-inventory-room-runtime-001/broker.jsonl")

        with self.assertRaises(AIInventoryAdmissionIntegrityError):
            _BrokerOwnedAdmissionController(
                FakeBroker(),  # type: ignore[arg-type]
                room_id=ROOM_ID,
                candidate_id=CANDIDATE_ID,
            )

    def test_non_durable_virtual_broker_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            fixture.broker.fast_ledger = True
            with self.assertRaises(AIInventoryAdmissionIntegrityError):
                fixture.proxy()


if __name__ == "__main__":
    unittest.main()
