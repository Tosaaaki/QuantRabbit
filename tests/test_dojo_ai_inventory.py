from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    InventoryDecisionConflictError,
    InventoryDecisionError,
    InventoryLedgerIntegrityError,
    InventoryMarketClosedError,
    append_inventory_decision,
    inventory_decision_identity_sha256,
    inventory_decision_sha256,
    seal_inventory_decision_proposal,
    validate_inventory_decision,
    validate_inventory_decision_ledger,
    validate_virtual_action_binding,
)
from quant_rabbit.dojo_ai_evidence_packet import (
    entry_signal_identity_sha256,
)


def _dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 7, 23, hour, minute, second, tzinfo=timezone.utc)


def _entry_signal(*, entry_context_sha256: str = "e" * 64) -> dict[str, object]:
    signal: dict[str, object] = {
        "pair": "USD_JPY",
        "side": "LONG",
        "order_type": "LIMIT",
        "units": 100.0,
        "price": 163.0,
        "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
        "entry_context_sha256": entry_context_sha256,
        "tp_pips": 3.0,
        "sl_pips": 25.0,
        "observed_at_utc": "2026-07-23T11:59:30Z",
    }
    signal["signal_identity_sha256"] = entry_signal_identity_sha256(signal)
    return signal


def _admission_binding(*, entry_context_sha256: str = "e" * 64) -> dict[str, object]:
    return {
        "entry_signal": _entry_signal(entry_context_sha256=entry_context_sha256),
        "evidence_packet_sha256": "a" * 64,
        "permit_expires_at_utc": "2026-07-23T12:00:45Z",
    }


def _proposal(**updates: object) -> dict[str, object]:
    cutoff = "2026-07-23T12:00:00Z"
    recent = "2026-07-23T11:59:30Z"
    body: dict[str, object] = {
        "contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "cutoff_at_utc": cutoff,
        "expires_at_utc": "2026-07-23T12:30:00Z",
        "action": "HOLD",
        "virtual_units": None,
        "confidence": 0.75,
        "admission_binding": None,
        "reason_code": "THESIS_ALIVE",
        "reason": "Bound evidence still supports the registered paper thesis.",
        "session_binding": {
            "experiment_id": "paper-ai-inventory-v1",
            "room_id": "paper-ai-inventory-room-01",
            "session_contract_sha256": "1" * 64,
            "observed_at_utc": "2026-07-23T00:00:00Z",
        },
        "policy_binding": {
            "policy_id": "inventory-policy-v1",
            "policy_sha256": "2" * 64,
            "observed_at_utc": "2026-07-22T00:00:00Z",
        },
        "candidate_binding": {
            "candidate_id": "candidate-001",
            "candidate_sha256": "3" * 64,
            "observed_at_utc": "2026-07-22T00:00:00Z",
        },
        "spec_binding": {
            "spec_id": "paper-ai-inventory-spec-v1",
            "spec_sha256": "b" * 64,
            "observed_at_utc": "2026-07-22T00:00:00Z",
        },
        "lifecycle_binding": {
            "paper_eligible_event_sha256": "c" * 64,
            "candidate_lifecycle_ledger_tip_sha256": "d" * 64,
            "observed_at_utc": "2026-07-22T00:00:00Z",
        },
        "ai_decision_binding": {
            "producer_id": "codex-dojo-single-reader-v1",
            "model_id": "gpt-5.6-sol",
            "request_sha256": "0" * 64,
            "response_sha256": "f" * 64,
            "evidence_packet_sha256": "a" * 64,
            "producer_receipt_sha256": "b" * 64,
            "produced_at_utc": "2026-07-23T11:59:50Z",
            "observed_at_utc": recent,
        },
        "ledger_binding": {
            "sha256": "4" * 64,
            "observed_at_utc": recent,
        },
        "state_binding": {
            "sha256": "5" * 64,
            "observed_at_utc": recent,
        },
        "snapshot_binding": {
            "sha256": "6" * 64,
            "observed_at_utc": recent,
        },
        "position_binding": {
            "position_id": "T000001",
            "pair": "USD_JPY",
            "side": "LONG",
            "units": 100.0,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": "e" * 64,
            "sha256": "7" * 64,
            "observed_at_utc": recent,
        },
        "quote_binding": {
            "pair": "USD_JPY",
            "bid": 163.1,
            "ask": 163.11,
            "sha256": "8" * 64,
            "observed_at_utc": recent,
        },
        "source_watermarks": [
            {
                "source_id": "candles:USD_JPY:M1",
                "sha256": "9" * 64,
                "watermark_at_utc": recent,
                "max_age_seconds": 120,
            },
            {
                "source_id": "news:macro",
                "sha256": "a" * 64,
                "watermark_at_utc": "2026-07-23T11:30:00Z",
                "max_age_seconds": 3600,
            },
        ],
        "max_dynamic_evidence_age_seconds": 120,
        "max_record_lag_seconds": 120,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    body.update(updates)
    return body


class DojoAiInventoryTest(unittest.TestCase):
    def test_append_is_content_addressed_chained_and_idempotent(self) -> None:
        proposal = _proposal()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 0, 30),
            ):
                first = append_inventory_decision(path, proposal)
                retry = append_inventory_decision(path, proposal)

            self.assertTrue(first.appended)
            self.assertFalse(retry.appended)
            self.assertEqual(first.record, retry.record)
            self.assertEqual(first.record["sequence"], 1)
            self.assertEqual(first.record["previous_decision_sha256"], "0" * 64)
            self.assertEqual(first.record["record_lag_nanoseconds"], 30_000_000_000)
            self.assertEqual(
                first.record["decision_sha256"],
                inventory_decision_sha256(first.record),
            )
            self.assertEqual(
                first.record["decision_identity_sha256"],
                inventory_decision_identity_sha256(first.record),
            )
            self.assertEqual(len(path.read_text().splitlines()), 1)
            validation = validate_inventory_decision_ledger(path)
            self.assertTrue(validation["valid"], validation["issues"])
            self.assertEqual(validation["row_count"], 1)
            self.assertEqual(
                validation["terminal_decision_sha256"],
                first.record["decision_sha256"],
            )
            self.assertIs(validation["paper_only"], True)
            self.assertEqual(validation["order_authority"], "NONE")
            self.assertIs(validation["live_permission"], False)
            self.assertIs(validation["virtual_broker_mutation_allowed"], True)
            self.assertIs(validation["external_broker_mutation_allowed"], False)
            self.assertEqual(
                validation["consumer_contract"],
                DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
            )

    def test_decision_before_action_binding_is_strict_and_virtual_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 0, 30),
            ):
                record = append_inventory_decision(path, _proposal()).record

            valid = validate_virtual_action_binding(
                record,
                decision_sha256=record["decision_sha256"],
                action="HOLD",
                virtual_units=None,
                action_at_utc="2026-07-23T12:00:31Z",
            )
            self.assertEqual(valid, ())

            same_clock = validate_virtual_action_binding(
                record,
                decision_sha256=record["decision_sha256"],
                action="HOLD",
                virtual_units=None,
                action_at_utc=record["recorded_at_utc"],
            )
            self.assertIn("ACTION_NOT_AFTER_DURABLE_DECISION", same_clock)

            wrong_hash = validate_virtual_action_binding(
                record,
                decision_sha256="f" * 64,
                action="CLOSE_VIRTUAL",
                virtual_units=100,
                action_at_utc="2026-07-23T12:00:31Z",
            )
            self.assertIn("BOUND_DECISION_SHA256_MISMATCH", wrong_hash)
            self.assertIn("ACTION_MISMATCH", wrong_hash)
            self.assertIn("VIRTUAL_UNITS_MISMATCH", wrong_hash)

    def test_action_quantity_contract_fails_closed(self) -> None:
        close = seal_inventory_decision_proposal(
            _proposal(
                action="CLOSE_VIRTUAL",
                virtual_units=100,
                reason_code="THESIS_INVALIDATED",
            )
        )
        self.assertEqual(close["action"], "CLOSE_VIRTUAL")
        self.assertIs(close["position_binding"]["units"].__class__, float)
        self.assertIs(close["virtual_units"].__class__, float)

        reduce = seal_inventory_decision_proposal(
            _proposal(
                action="REDUCE_VIRTUAL",
                virtual_units=40,
                reason_code="INVENTORY_CONCENTRATED",
            )
        )
        self.assertEqual(reduce["virtual_units"], 40)

        with self.assertRaisesRegex(ValueError, "INVALID_CLOSE_VIRTUAL_UNITS"):
            seal_inventory_decision_proposal(
                _proposal(action="CLOSE_VIRTUAL", virtual_units=99)
            )
        with self.assertRaisesRegex(ValueError, "INVALID_REDUCE_VIRTUAL_UNITS"):
            seal_inventory_decision_proposal(
                _proposal(action="REDUCE_VIRTUAL", virtual_units=100)
            )
        with self.assertRaisesRegex(ValueError, "VIRTUAL_UNITS_MUST_BE_NULL"):
            seal_inventory_decision_proposal(
                _proposal(action="BLOCK_NEW", virtual_units=1)
            )

    def test_fractional_virtual_broker_units_are_canonical_and_exact(self) -> None:
        fractional = _proposal(action="CLOSE_VIRTUAL", virtual_units=100.25)
        fractional["position_binding"] = {
            **fractional["position_binding"],  # type: ignore[arg-type]
            "units": 100.25,
        }
        sealed = seal_inventory_decision_proposal(fractional)
        self.assertEqual(sealed["position_binding"]["units"], 100.25)
        self.assertEqual(sealed["virtual_units"], 100.25)
        self.assertIs(sealed["position_binding"]["units"].__class__, float)
        self.assertIs(sealed["virtual_units"].__class__, float)

        exact_mismatch = dict(fractional)
        exact_mismatch["virtual_units"] = 100.25000000000001
        with self.assertRaisesRegex(ValueError, "INVALID_CLOSE_VIRTUAL_UNITS"):
            seal_inventory_decision_proposal(exact_mismatch)

        for invalid in (True, float("nan"), float("inf"), -1.0):
            bad = _proposal()
            bad["position_binding"] = {
                **bad["position_binding"],  # type: ignore[arg-type]
                "units": invalid,
            }
            with self.assertRaisesRegex(ValueError, "INVALID_POSITION_UNITS"):
                seal_inventory_decision_proposal(bad)

    def test_block_new_accepts_expiring_flat_binding_only(self) -> None:
        flat = _proposal(
            action="BLOCK_NEW",
            expires_at_utc="2026-07-23T12:01:00Z",
            reason_code="REGIME_MISMATCH",
            reason="Temporarily block a new virtual entry for this pair.",
        )
        flat["position_binding"] = {
            "position_id": "FLAT:USD_JPY",
            "pair": "USD_JPY",
            "side": "FLAT",
            "units": 0,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": "f" * 64,
            "sha256": "7" * 64,
            "observed_at_utc": "2026-07-23T11:59:30Z",
        }
        sealed = seal_inventory_decision_proposal(flat)
        self.assertEqual(sealed["position_binding"]["units"], 0.0)
        self.assertEqual(sealed["position_binding"]["position_id"], "FLAT:USD_JPY")

        hold_flat = dict(flat)
        hold_flat["action"] = "HOLD"
        with self.assertRaisesRegex(
            ValueError, "FLAT_BINDING_ONLY_ALLOWED_FOR_ENTRY_GATE"
        ):
            seal_inventory_decision_proposal(hold_flat)

        wrong_flat_id = dict(flat)
        wrong_flat_id["position_binding"] = {
            **flat["position_binding"],  # type: ignore[arg-type]
            "position_id": "FLAT:EUR_USD",
        }
        with self.assertRaisesRegex(ValueError, "INVALID_FLAT_POSITION_ID"):
            seal_inventory_decision_proposal(wrong_flat_id)

        too_long = dict(flat)
        too_long["expires_at_utc"] = "2026-07-23T12:01:31Z"
        with self.assertRaisesRegex(ValueError, "ENTRY_GATE_EXPIRY_EXCEEDS_SHORT_TTL"):
            seal_inventory_decision_proposal(too_long)

        allow = {
            **flat,
            "action": "ALLOW_NEW_VIRTUAL",
            "admission_binding": _admission_binding(entry_context_sha256="f" * 64),
            "reason_code": "ENTRY_HABITAT_MATCH",
            "reason": "Issue one short-lived permit for the isolated entry proxy.",
        }
        sealed_allow = seal_inventory_decision_proposal(allow)
        self.assertEqual(sealed_allow["action"], "ALLOW_NEW_VIRTUAL")
        self.assertIsNone(sealed_allow["virtual_units"])

        open_allow = _proposal(
            action="ALLOW_NEW_VIRTUAL",
            expires_at_utc="2026-07-23T12:01:00Z",
            admission_binding=_admission_binding(),
        )
        with self.assertRaisesRegex(ValueError, "ENTRY_GATE_REQUIRES_FLAT_BINDING"):
            seal_inventory_decision_proposal(open_allow)

        missing_admission = dict(allow)
        missing_admission["admission_binding"] = None
        with self.assertRaisesRegex(
            ValueError, "ALLOW_NEW_VIRTUAL_REQUIRES_ADMISSION_BINDING"
        ):
            seal_inventory_decision_proposal(missing_admission)

        mismatched_strategy = dict(allow)
        bad_signal = {
            **allow["admission_binding"]["entry_signal"],  # type: ignore[index]
            "strategy_tag": "OTHER_STRATEGY",
        }
        bad_signal["signal_identity_sha256"] = entry_signal_identity_sha256(bad_signal)
        mismatched_strategy["admission_binding"] = {
            **allow["admission_binding"],  # type: ignore[arg-type]
            "entry_signal": bad_signal,
        }
        with self.assertRaisesRegex(
            ValueError, "ADMISSION_POSITION_STRATEGY_TAG_MISMATCH"
        ):
            seal_inventory_decision_proposal(mismatched_strategy)

        permit_too_long = dict(allow)
        permit_too_long["admission_binding"] = {
            **allow["admission_binding"],  # type: ignore[arg-type]
            "permit_expires_at_utc": "2026-07-23T12:01:31Z",
        }
        with self.assertRaisesRegex(ValueError, "ADMISSION_PERMIT_EXCEEDS_SHORT_TTL"):
            seal_inventory_decision_proposal(permit_too_long)

        mismatched_evidence = dict(allow)
        mismatched_evidence["admission_binding"] = {
            **allow["admission_binding"],  # type: ignore[arg-type]
            "evidence_packet_sha256": "f" * 64,
        }
        with self.assertRaisesRegex(ValueError, "ADMISSION_EVIDENCE_PACKET_MISMATCH"):
            seal_inventory_decision_proposal(mismatched_evidence)

        bad_identity = dict(allow)
        bad_entry_signal = {
            **allow["admission_binding"]["entry_signal"],  # type: ignore[index]
            "signal_identity_sha256": "f" * 64,
        }
        bad_identity["admission_binding"] = {
            **allow["admission_binding"],  # type: ignore[arg-type]
            "entry_signal": bad_entry_signal,
        }
        with self.assertRaisesRegex(ValueError, "ENTRY_SIGNAL_IDENTITY_MISMATCH"):
            seal_inventory_decision_proposal(bad_identity)

        hold_with_admission = _proposal(admission_binding=allow["admission_binding"])
        with self.assertRaisesRegex(ValueError, "ADMISSION_BINDING_MUST_BE_NULL"):
            seal_inventory_decision_proposal(hold_with_admission)

    def test_confidence_is_required_as_a_finite_float(self) -> None:
        self.assertEqual(
            seal_inventory_decision_proposal(_proposal(confidence=0.0))["confidence"],
            0.0,
        )
        self.assertEqual(
            seal_inventory_decision_proposal(_proposal(confidence=1.0))["confidence"],
            1.0,
        )
        for invalid in (True, 1, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "INVALID_CONFIDENCE"
            ):
                seal_inventory_decision_proposal(_proposal(confidence=invalid))

    def test_no_lookahead_staleness_and_safety_tamper_fail_closed(self) -> None:
        future_quote = _proposal()
        future_quote["quote_binding"] = {
            **future_quote["quote_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-23T12:00:01Z",
        }
        with self.assertRaisesRegex(ValueError, "OBSERVED_AFTER_CUTOFF"):
            seal_inventory_decision_proposal(future_quote)

        stale_state = _proposal()
        stale_state["state_binding"] = {
            **stale_state["state_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-23T11:57:59Z",
        }
        with self.assertRaisesRegex(ValueError, "STALE_AT_CUTOFF"):
            seal_inventory_decision_proposal(stale_state)

        future_ai = _proposal()
        future_ai["ai_decision_binding"] = {
            **future_ai["ai_decision_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-23T12:00:01Z",
        }
        with self.assertRaisesRegex(
            ValueError, "ai_decision_binding:OBSERVED_AFTER_CUTOFF"
        ):
            seal_inventory_decision_proposal(future_ai)

        stale_ai = _proposal()
        stale_ai["ai_decision_binding"] = {
            **stale_ai["ai_decision_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-23T11:57:59Z",
        }
        with self.assertRaisesRegex(ValueError, "ai_decision_binding:STALE_AT_CUTOFF"):
            seal_inventory_decision_proposal(stale_ai)

        future_source = _proposal()
        future_source["source_watermarks"] = [
            {
                "source_id": "candles:USD_JPY:M1",
                "sha256": "9" * 64,
                "watermark_at_utc": "2026-07-23T12:00:01Z",
                "max_age_seconds": 120,
            }
        ]
        with self.assertRaisesRegex(ValueError, "WATERMARK_AFTER_CUTOFF"):
            seal_inventory_decision_proposal(future_source)

        unsafe = _proposal(order_authority="VIRTUAL")
        with self.assertRaisesRegex(ValueError, "SAFETY_INVARIANT_FAILED"):
            seal_inventory_decision_proposal(unsafe)

        unsafe_virtual = _proposal(virtual_broker_mutation_allowed=False)
        with self.assertRaisesRegex(
            ValueError,
            "SAFETY_INVARIANT_FAILED:virtual_broker_mutation_allowed",
        ):
            seal_inventory_decision_proposal(unsafe_virtual)

        legacy = _proposal()
        legacy.pop("virtual_broker_mutation_allowed")
        legacy["broker_mutation_allowed"] = False
        with self.assertRaisesRegex(
            ValueError, "MISSING_PROPOSAL_FIELD:virtual_broker_mutation_allowed"
        ):
            seal_inventory_decision_proposal(legacy)

    def test_future_room_and_candidate_lifecycle_bindings_are_mandatory(
        self,
    ) -> None:
        missing_spec = _proposal()
        missing_spec.pop("spec_binding")
        with self.assertRaisesRegex(ValueError, "MISSING_PROPOSAL_FIELD:spec_binding"):
            seal_inventory_decision_proposal(missing_spec)

        missing_ai = _proposal()
        missing_ai.pop("ai_decision_binding")
        with self.assertRaisesRegex(
            ValueError, "MISSING_PROPOSAL_FIELD:ai_decision_binding"
        ):
            seal_inventory_decision_proposal(missing_ai)

        malformed_ai = _proposal()
        malformed_ai["ai_decision_binding"] = {
            **malformed_ai["ai_decision_binding"],  # type: ignore[arg-type]
            "response_sha256": "not-a-digest",
            "unexpected": True,
        }
        with self.assertRaisesRegex(
            ValueError, "ai_decision_binding:INVALID_SHA256:response_sha256"
        ):
            seal_inventory_decision_proposal(malformed_ai)

        bad_session = _proposal()
        bad_session["session_binding"] = {
            "experiment_id": "paper-ai-inventory-v1",
            "room_id": "paper-ai-inventory-room-01",
            "session_contract_sha256": "short",
            "observed_at_utc": "2026-07-23T00:00:00Z",
        }
        with self.assertRaisesRegex(
            ValueError, "INVALID_SHA256:session_contract_sha256"
        ):
            seal_inventory_decision_proposal(bad_session)

        missing_paper_eligible = _proposal()
        missing_paper_eligible["lifecycle_binding"] = {
            "candidate_lifecycle_ledger_tip_sha256": "d" * 64,
            "observed_at_utc": "2026-07-22T00:00:00Z",
        }
        with self.assertRaisesRegex(
            ValueError, "MISSING_FIELD:paper_eligible_event_sha256"
        ):
            seal_inventory_decision_proposal(missing_paper_eligible)

        missing_entry_context = _proposal()
        position = dict(missing_entry_context["position_binding"])  # type: ignore[arg-type]
        position.pop("entry_context_sha256")
        missing_entry_context["position_binding"] = position
        with self.assertRaisesRegex(ValueError, "MISSING_FIELD:entry_context_sha256"):
            seal_inventory_decision_proposal(missing_entry_context)

    def test_tamper_or_partial_history_blocks_later_append(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 0, 30),
            ):
                first = append_inventory_decision(path, _proposal()).record

            tampered = dict(first)
            tampered["reason"] = "edited after append"
            path.write_text(
                json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n"
            )
            validation = validate_inventory_decision_ledger(path)
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any(
                    "DECISION_SHA256_MISMATCH" in issue
                    for issue in validation["issues"]
                )
            )
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 1, 0),
            ):
                with self.assertRaises(InventoryLedgerIntegrityError):
                    append_inventory_decision(path, _proposal())

            path.write_bytes(b'{"contract":"partial"}')
            validation = validate_inventory_decision_ledger(path)
            self.assertFalse(validation["valid"])
            self.assertIn("LEDGER_MISSING_TERMINAL_NEWLINE", validation["issues"])

    def test_same_identity_with_changed_action_is_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 0, 30),
            ):
                append_inventory_decision(path, _proposal())
                with self.assertRaises(InventoryDecisionConflictError):
                    append_inventory_decision(
                        path,
                        _proposal(
                            action="CLOSE_VIRTUAL",
                            virtual_units=100,
                            reason_code="THESIS_INVALIDATED",
                            reason="Same packet cannot be reinterpreted on retry.",
                        ),
                    )
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_weekend_rejects_new_decisions_but_existing_retry_is_idempotent(
        self,
    ) -> None:
        # New York is on DST in July: Friday 17:00 is 21:00 UTC.
        friday_proposal = _proposal(
            cutoff_at_utc="2026-07-24T20:59:30Z",
            expires_at_utc="2026-07-24T21:00:00Z",
        )
        for key in ("ledger_binding", "state_binding", "snapshot_binding"):
            friday_proposal[key] = {
                "sha256": friday_proposal[key]["sha256"],  # type: ignore[index]
                "observed_at_utc": "2026-07-24T20:59:00Z",
            }
        friday_proposal["position_binding"] = {
            **friday_proposal["position_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-24T20:59:00Z",
        }
        friday_proposal["quote_binding"] = {
            **friday_proposal["quote_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-24T20:59:00Z",
        }
        friday_proposal["ai_decision_binding"] = {
            **friday_proposal["ai_decision_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-24T20:59:00Z",
            "produced_at_utc": "2026-07-24T20:59:15Z",
        }
        friday_proposal["source_watermarks"] = [
            {
                "source_id": "candles:USD_JPY:M1",
                "sha256": "9" * 64,
                "watermark_at_utc": "2026-07-24T20:59:00Z",
                "max_age_seconds": 120,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=datetime(2026, 7, 24, 20, 59, 45, tzinfo=timezone.utc),
            ):
                first = append_inventory_decision(path, friday_proposal)
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
            ):
                retry = append_inventory_decision(path, friday_proposal)
                self.assertFalse(retry.appended)
                different = _proposal(
                    cutoff_at_utc="2026-07-25T12:00:00Z",
                    expires_at_utc="2026-07-25T12:30:00Z",
                )
                for key in ("ledger_binding", "state_binding", "snapshot_binding"):
                    different[key] = {
                        "sha256": different[key]["sha256"],  # type: ignore[index]
                        "observed_at_utc": "2026-07-25T11:59:30Z",
                    }
                different["position_binding"] = {
                    **different["position_binding"],  # type: ignore[arg-type]
                    "observed_at_utc": "2026-07-25T11:59:30Z",
                }
                different["quote_binding"] = {
                    **different["quote_binding"],  # type: ignore[arg-type]
                    "observed_at_utc": "2026-07-25T11:59:30Z",
                }
                different["ai_decision_binding"] = {
                    **different["ai_decision_binding"],  # type: ignore[arg-type]
                    "observed_at_utc": "2026-07-25T11:59:30Z",
                    "produced_at_utc": "2026-07-25T11:59:45Z",
                }
                different["source_watermarks"] = [
                    {
                        "source_id": "candles:USD_JPY:M1",
                        "sha256": "9" * 64,
                        "watermark_at_utc": "2026-07-25T11:59:30Z",
                        "max_age_seconds": 120,
                    }
                ]
                with self.assertRaises(InventoryMarketClosedError):
                    append_inventory_decision(path, different)
            self.assertEqual(first.record, retry.record)

        with self.assertRaises(InventoryMarketClosedError):
            seal_inventory_decision_proposal(different)
        sunday = _proposal(
            cutoff_at_utc="2026-07-26T20:59:59.999999999Z",
            expires_at_utc="2026-07-26T21:30:00Z",
        )
        for key in ("ledger_binding", "state_binding", "snapshot_binding"):
            sunday[key] = {
                "sha256": sunday[key]["sha256"],  # type: ignore[index]
                "observed_at_utc": "2026-07-26T20:59:59.999999000Z",
            }
        sunday["position_binding"] = {
            **sunday["position_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-26T20:59:59.999999000Z",
        }
        sunday["quote_binding"] = {
            **sunday["quote_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-26T20:59:59.999999000Z",
        }
        sunday["ai_decision_binding"] = {
            **sunday["ai_decision_binding"],  # type: ignore[arg-type]
            "observed_at_utc": "2026-07-26T20:59:59.999999000Z",
            "produced_at_utc": "2026-07-26T20:59:59.999999500Z",
        }
        sunday["source_watermarks"] = [
            {
                "source_id": "candles:USD_JPY:M1",
                "sha256": "9" * 64,
                "watermark_at_utc": "2026-07-26T20:59:59.999999000Z",
                "max_age_seconds": 120,
            }
        ]
        with self.assertRaises(InventoryMarketClosedError):
            seal_inventory_decision_proposal(sunday)

    def test_record_lag_is_writer_authored_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 2, 1),
            ):
                with self.assertRaisesRegex(
                    InventoryDecisionError, "record lag exceeds"
                ):
                    append_inventory_decision(path, _proposal())
            self.assertEqual(path.read_bytes(), b"")

    def test_record_validation_detects_safety_and_chain_field_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            with patch(
                "quant_rabbit.dojo_ai_inventory._utc_now",
                return_value=_dt(12, 0, 30),
            ):
                record = append_inventory_decision(path, _proposal()).record
        unsafe = dict(record)
        unsafe["live_permission"] = True
        unsafe["decision_sha256"] = inventory_decision_sha256(unsafe)
        issues = validate_inventory_decision(unsafe)
        self.assertIn("SAFETY_INVARIANT_FAILED:live_permission", issues)

        wrong_lag = dict(record)
        wrong_lag["record_lag_nanoseconds"] = 0
        wrong_lag["decision_sha256"] = inventory_decision_sha256(wrong_lag)
        issues = validate_inventory_decision(wrong_lag)
        self.assertIn("RECORD_LAG_MISMATCH", issues)

        after_expiry = validate_virtual_action_binding(
            record,
            decision_sha256=record["decision_sha256"],
            action="HOLD",
            virtual_units=None,
            action_at_utc="2026-07-23T12:30:00.000000001Z",
        )
        self.assertIn("ACTION_AFTER_DECISION_EXPIRY", after_expiry)


if __name__ == "__main__":
    unittest.main()
