from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields, replace
import hashlib
import unittest

from quant_rabbit.policy_generation_shadow import (
    GenerationScopedOutcomeObservation,
    evaluate_policy_generation_shadow,
    policy_generation_activation_sha256,
    seal_policy_generation_activation,
    validate_policy_generation_activation,
)


LANE = "range_trader:EUR_USD:SHORT:RANGE_ROTATION:LIMIT"
GENERATION = "situation-policy-v2"


def _activation() -> dict:
    return seal_policy_generation_activation(
        policy_generation_id=GENERATION,
        exact_lane_id=LANE,
        activated_at_utc="2026-07-13T16:32:11Z",
        deployment_artifact_sha256="a" * 64,
        source_revision="b" * 40,
    )


def _observation(
    trade_id: str,
    *,
    gateway_at: str,
    entry_at: str,
    pnl: float | None,
    close_at: str | None = None,
    generation: str | None = GENERATION,
    activation_sha: str | None = None,
    intent_at: str | None = None,
    intent_artifact_sha: str = "f" * 64,
    lane_id: str = LANE,
) -> GenerationScopedOutcomeObservation:
    activation_sha = activation_sha or _activation()["activation_sha256"]
    return GenerationScopedOutcomeObservation(
        trade_id=trade_id,
        exact_lane_id=lane_id,
        policy_generation_id=generation,
        activation_sha256=activation_sha,
        entry_event_uid=f"oanda:{trade_id}:ORDER_FILLED:opened:{trade_id}",
        intent_generated_at_utc=intent_at or gateway_at,
        intent_artifact_sha256=intent_artifact_sha,
        gateway_sent_at_utc=gateway_at,
        broker_entry_at_utc=entry_at,
        broker_close_at_utc=close_at,
        net_realized_jpy=pnl,
        outcome_scope="ALL_AUDITED_EXITS" if pnl is not None else None,
        broker_close_event_uid=(
            f"oanda:{trade_id}:TRADE_CLOSED:closed:{trade_id}"
            if pnl is not None
            else None
        ),
        outcome_evidence_sha256=(
            hashlib.sha256(f"outcome:{trade_id}".encode("utf-8")).hexdigest()
            if pnl is not None
            else None
        ),
    )


class _SplitActivation(Mapping[str, object]):
    """Expose the sealed activation to get/items and an older boundary to []."""

    def __init__(self, activation: Mapping[str, object]) -> None:
        self._activation = dict(activation)

    def __len__(self) -> int:
        return len(self._activation)

    def __iter__(self) -> Iterator[str]:
        return iter(self._activation)

    def __getitem__(self, key: str) -> object:
        if key == "activated_at_utc":
            return "2026-07-13T16:00:00+00:00"
        return self._activation[key]

    def get(self, key: str, default: object = None) -> object:
        return self._activation.get(key, default)

    def items(self):  # type: ignore[no-untyped-def]
        return self._activation.items()


class _AliasActivationKey:
    def __str__(self) -> str:
        return "activated_at_utc"


class _RedirectText(str):
    def __new__(cls, stored: str, converted: str):
        instance = super().__new__(cls, stored)
        instance.converted = converted
        return instance

    def __str__(self) -> str:
        return self.converted


class _RedirectPnl(float):
    def __new__(cls, stored: float, converted: float):
        instance = super().__new__(cls, stored)
        instance.converted = converted
        return instance

    def __float__(self) -> float:
        return self.converted


class _ForgedCohortSize(int):
    def __new__(cls):
        return super().__new__(cls, 0)

    def __ge__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True


class PolicyGenerationShadowTest(unittest.TestCase):
    def test_activation_is_content_addressed_and_diagnostic_only(self) -> None:
        activation = _activation()

        self.assertEqual(validate_policy_generation_activation(activation), ())
        self.assertTrue(activation["read_only"])
        self.assertFalse(activation["live_permission_allowed"])

        tampered = dict(activation)
        tampered["policy_generation_id"] = "situation-policy-v3"
        self.assertIn(
            "activation_sha256 does not match the canonical body",
            validate_policy_generation_activation(tampered),
        )

    def test_activation_snapshot_prevents_mapping_toctou_and_key_aliases(self) -> None:
        activation = _activation()
        split = _SplitActivation(activation)
        pre_activation_winner = _observation(
            "pre-activation-winner",
            intent_at="2026-07-13T16:10:00Z",
            gateway_at="2026-07-13T16:10:01Z",
            entry_at="2026-07-13T16:10:02Z",
            close_at="2026-07-13T16:20:00Z",
            pnl=1000.0,
        )

        result = evaluate_policy_generation_shadow(
            activation=split,  # type: ignore[arg-type]
            observations=[pre_activation_winner],
            cohort_size=1,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "INVALID_ACTIVATION")
        self.assertIsNone(result["metrics"])
        self.assertFalse(result["live_permission_allowed"])

        aliased: dict[object, object] = dict(activation)
        aliased[_AliasActivationKey()] = "2026-07-13T16:00:00+00:00"
        with self.assertRaises(ValueError):
            policy_generation_activation_sha256(aliased)  # type: ignore[arg-type]
        issues = validate_policy_generation_activation(aliased)  # type: ignore[arg-type]
        self.assertIn("activation keys must be exact strings", issues)

    def test_activation_and_observation_require_exact_builtin_scalars(self) -> None:
        activation = _activation()
        redirected_activation = dict(activation)
        redirected_activation["activated_at_utc"] = _RedirectText(
            str(activation["activated_at_utc"]),
            "2026-07-13T16:00:00+00:00",
        )
        redirected_activation["activation_sha256"] = activation["activation_sha256"]

        issues = validate_policy_generation_activation(redirected_activation)

        self.assertIn("activated_at_utc must be canonical UTC", issues)
        result = evaluate_policy_generation_shadow(
            activation=redirected_activation,
            observations=[],
            cohort_size=1,
            source_stream_complete=True,
        )
        self.assertEqual(result["status"], "INVALID_ACTIVATION")

        row = _observation(
            "scalar-row",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=100.0,
        )
        invalid_rows = (
            replace(row, trade_id=_RedirectText("scalar-row", "other-row")),
            replace(row, net_realized_jpy=_RedirectPnl(100.0, 1000.0)),
        )
        for invalid_row in invalid_rows:
            with self.subTest(invalid_row=invalid_row):
                invalid = evaluate_policy_generation_shadow(
                    activation=activation,
                    observations=[invalid_row],
                    cohort_size=1,
                    source_stream_complete=True,
                )
                self.assertEqual(invalid["status"], "INVALID_SOURCE")
                self.assertIsNone(invalid["metrics"])

        forged_cohort = evaluate_policy_generation_shadow(
            activation=activation,
            observations=[],
            cohort_size=_ForgedCohortSize(),
            source_stream_complete=True,
        )
        self.assertEqual(forged_cohort["status"], "INVALID_SOURCE")
        self.assertIsNone(forged_cohort["metrics"])

    def test_pre_activation_v1_losses_are_excluded_from_v2_shadow(self) -> None:
        activation = _activation()
        rows = [
            _observation(
                "v1-loss",
                gateway_at="2026-07-13T16:00:00Z",
                entry_at="2026-07-13T16:01:00Z",
                close_at="2026-07-13T16:20:00Z",
                pnl=-5000.0,
                generation="baseline-v1",
                activation_sha="c" * 64,
            ),
            _observation(
                "v2-win",
                gateway_at="2026-07-13T16:33:00Z",
                entry_at="2026-07-13T16:34:00Z",
                close_at="2026-07-13T17:00:00Z",
                pnl=600.0,
            ),
        ]

        result = evaluate_policy_generation_shadow(
            activation=activation,
            observations=rows,
            cohort_size=2,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "COLLECTING_GENERATION_EVIDENCE")
        self.assertTrue(result["shadow_activation_binding_consistent"])
        self.assertEqual(result["counts"]["pre_activation_exact_lane"], 1)
        self.assertEqual(result["metrics"]["frozen_entries"], 1)
        self.assertEqual(result["metrics"]["net_realized_jpy"], 600.0)
        self.assertTrue(result["current_all_time_gate_authoritative"])
        self.assertFalse(result["shadow_may_replace_all_time_gate"])
        self.assertFalse(result["gate_relaxation_allowed"])
        self.assertFalse(result["live_permission_allowed"])
        self.assertFalse(result["independent_deployment_proof_verified_by_evaluator"])
        self.assertFalse(result["runtime_ledger_adapter_verified_by_evaluator"])
        self.assertFalse(result["intent_artifact_digest_verified_by_evaluator"])
        self.assertTrue(result["external_intent_artifact_digest_verification_required"])
        self.assertFalse(result["outcome_evidence_digest_verified_by_evaluator"])
        self.assertTrue(
            result["external_outcome_evidence_digest_verification_required"]
        )
        self.assertTrue(result["source_stream_completeness_caller_attested"])
        self.assertFalse(
            result["independent_source_completeness_verified_by_evaluator"]
        )
        self.assertFalse(result["activation_boundary_proven_for_live"])

    def test_post_activation_entry_without_binding_fails_closed(self) -> None:
        row = _observation(
            "unbound",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=600.0,
            generation=None,
        )
        row = replace(row, activation_sha256=None)

        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[row],
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "ACTIVATION_BOUNDARY_UNPROVEN")
        self.assertFalse(result["shadow_activation_binding_consistent"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["counts"]["unbound_post_activation_exact_lane"], 1)
        self.assertFalse(result["live_permission_allowed"])

    def test_wrong_generation_or_activation_digest_fails_closed(self) -> None:
        for row in (
            _observation(
                "wrong-generation",
                gateway_at="2026-07-13T16:33:00Z",
                entry_at="2026-07-13T16:34:00Z",
                close_at="2026-07-13T17:00:00Z",
                pnl=600.0,
                generation="baseline-v1",
            ),
            _observation(
                "wrong-activation",
                gateway_at="2026-07-13T16:33:00Z",
                entry_at="2026-07-13T16:34:00Z",
                close_at="2026-07-13T17:00:00Z",
                pnl=600.0,
                activation_sha="d" * 64,
            ),
        ):
            with self.subTest(trade_id=row.trade_id):
                result = evaluate_policy_generation_shadow(
                    activation=_activation(),
                    observations=[row],
                    source_stream_complete=True,
                )
                self.assertEqual(result["status"], "ACTIVATION_BOUNDARY_UNPROVEN")
                self.assertIsNone(result["metrics"])

    def test_event_uid_accepts_live_shapes_and_rejects_unbounded_values(self) -> None:
        row = _observation(
            "473017",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=100.0,
        )
        event_uids = (
            "oanda:473017:ORDER_FILLED:opened:473017",
            "gateway:live_order:2026-07-09T17:12:06.084679+00:00:0:"
            "GATEWAY_ORDER_SENT:failure_trader:EUR_USD:LONG:"
            "BREAKOUT_FAILURE:LIMIT",
        )
        for event_uid in event_uids:
            with self.subTest(event_uid=event_uid):
                result = evaluate_policy_generation_shadow(
                    activation=_activation(),
                    observations=[replace(row, entry_event_uid=event_uid)],
                    cohort_size=1,
                    source_stream_complete=True,
                )
                self.assertEqual(result["status"], "READY_FOR_DIAGNOSTIC_REVIEW")

        invalid = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[replace(row, entry_event_uid="x" * 257)],
            source_stream_complete=True,
        )
        self.assertEqual(invalid["status"], "INVALID_SOURCE")
        self.assertIsNone(invalid["metrics"])

    def test_pre_activation_gateway_with_later_fill_cannot_enter_v2(self) -> None:
        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                _observation(
                    "old-pending-fill",
                    gateway_at="2026-07-13T16:32:00Z",
                    entry_at="2026-07-13T16:40:00Z",
                    close_at="2026-07-13T17:00:00Z",
                    pnl=500.0,
                    generation="baseline-v1",
                    activation_sha="e" * 64,
                )
            ],
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "COLLECTING_GENERATION_EVIDENCE")
        self.assertEqual(
            result["counts"]["pre_activation_gateway_post_activation_fill"],
            1,
        )
        self.assertEqual(result["metrics"]["frozen_entries"], 0)

    def test_pre_activation_intent_delayed_send_cannot_enter_v2(self) -> None:
        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                _observation(
                    "old-intent-delayed-send",
                    intent_at="2026-07-13T16:32:10.999999999Z",
                    gateway_at="2026-07-13T16:32:11.000000001Z",
                    entry_at="2026-07-13T16:32:11.000000002Z",
                    close_at="2026-07-13T16:32:11.000000003Z",
                    pnl=400.0,
                )
            ],
            cohort_size=1,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "COLLECTING_GENERATION_EVIDENCE")
        self.assertEqual(
            result["counts"]["pre_activation_intent_post_activation_gateway_fill"],
            1,
        )
        self.assertEqual(result["metrics"]["frozen_entries"], 0)

    def test_intent_digest_timestamp_and_chronology_fail_closed(self) -> None:
        row = _observation(
            "intent-contract",
            intent_at="2026-07-13T16:32:11.000000001Z",
            gateway_at="2026-07-13T16:32:11.000000002Z",
            entry_at="2026-07-13T16:32:11.000000003Z",
            close_at="2026-07-13T16:32:11.000000004Z",
            pnl=100.0,
        )
        valid = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[row],
            cohort_size=1,
            source_stream_complete=True,
        )
        self.assertEqual(valid["status"], "READY_FOR_DIAGNOSTIC_REVIEW")

        invalid_rows = (
            replace(row, intent_artifact_sha256="bad"),
            replace(row, intent_artifact_sha256=None),
            replace(row, intent_generated_at_utc=""),
            replace(
                row,
                intent_generated_at_utc="2026-07-13T16:32:11.000000003Z",
            ),
        )
        for invalid_row in invalid_rows:
            with self.subTest(invalid_row=invalid_row):
                result = evaluate_policy_generation_shadow(
                    activation=_activation(),
                    observations=[invalid_row],
                    source_stream_complete=True,
                )
                self.assertEqual(result["status"], "INVALID_SOURCE")
                self.assertIsNone(result["metrics"])

    def test_unresolved_first_n_entry_is_not_replaced_by_later_winner(self) -> None:
        rows = [
            _observation(
                "first-unresolved",
                gateway_at="2026-07-13T16:33:00Z",
                entry_at="2026-07-13T16:34:00Z",
                pnl=None,
            ),
            _observation(
                "second-win",
                gateway_at="2026-07-13T16:35:00Z",
                entry_at="2026-07-13T16:36:00Z",
                close_at="2026-07-13T17:00:00Z",
                pnl=500.0,
            ),
            _observation(
                "later-win",
                gateway_at="2026-07-13T16:37:00Z",
                entry_at="2026-07-13T16:38:00Z",
                close_at="2026-07-13T17:01:00Z",
                pnl=700.0,
            ),
        ]

        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=rows,
            cohort_size=2,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "WAITING_FOR_FIRST_N_RESOLUTION")
        self.assertEqual(result["metrics"]["frozen_entries"], 2)
        self.assertEqual(result["metrics"]["resolved_entries"], 1)
        self.assertEqual(result["metrics"]["unresolved_entries"], 1)
        self.assertEqual(result["metrics"]["net_realized_jpy"], 500.0)

    def test_nanosecond_boundary_and_first_n_order_are_preserved(self) -> None:
        activation = seal_policy_generation_activation(
            policy_generation_id=GENERATION,
            exact_lane_id=LANE,
            activated_at_utc="2026-07-13T16:32:11.123456789Z",
            deployment_artifact_sha256="a" * 64,
            source_revision="b" * 40,
        )
        equal_boundary = _observation(
            "equal-boundary",
            gateway_at="2026-07-13T16:32:11.123456789Z",
            entry_at="2026-07-13T16:32:11.123456790Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=-1000.0,
            activation_sha=activation["activation_sha256"],
        )
        later = _observation(
            "later-nanosecond",
            gateway_at="2026-07-13T16:32:11.123456790Z",
            entry_at="2026-07-13T16:32:11.123456792Z",
            close_at="2026-07-13T17:00:00.000000002Z",
            pnl=200.0,
            activation_sha=activation["activation_sha256"],
        )
        earlier = _observation(
            "earlier-nanosecond",
            gateway_at="2026-07-13T16:32:11.123456790Z",
            entry_at="2026-07-13T16:32:11.123456791Z",
            close_at="2026-07-13T17:00:00.000000001Z",
            pnl=100.0,
            activation_sha=activation["activation_sha256"],
        )

        result = evaluate_policy_generation_shadow(
            activation=activation,
            observations=[equal_boundary, later, earlier],
            cohort_size=1,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "READY_FOR_DIAGNOSTIC_REVIEW")
        self.assertEqual(result["counts"]["pre_activation_exact_lane"], 1)
        self.assertEqual(result["metrics"]["frozen_entries"], 1)
        self.assertEqual(result["metrics"]["net_realized_jpy"], 100.0)

        same_time_later_uid = replace(
            earlier,
            trade_id="same-time-later-uid",
            entry_event_uid="oanda:z:ORDER_FILLED:opened:z",
            net_realized_jpy=900.0,
            broker_close_event_uid="oanda:z:TRADE_CLOSED:closed:z",
            outcome_evidence_sha256="9" * 64,
        )
        same_time_earlier_uid = replace(
            earlier,
            trade_id="same-time-earlier-uid",
            entry_event_uid="oanda:a:ORDER_FILLED:opened:a",
            net_realized_jpy=300.0,
            broker_close_event_uid="oanda:a:TRADE_CLOSED:closed:a",
            outcome_evidence_sha256="8" * 64,
        )
        uid_order_reversed_input = evaluate_policy_generation_shadow(
            activation=activation,
            observations=[same_time_later_uid, same_time_earlier_uid],
            cohort_size=1,
            source_stream_complete=True,
        )
        uid_order_forward_input = evaluate_policy_generation_shadow(
            activation=activation,
            observations=[same_time_earlier_uid, same_time_later_uid],
            cohort_size=1,
            source_stream_complete=True,
        )
        self.assertEqual(uid_order_reversed_input["metrics"]["net_realized_jpy"], 300.0)
        self.assertEqual(
            uid_order_reversed_input["metrics"]["first_n_trade_ids_sha256"],
            uid_order_forward_input["metrics"]["first_n_trade_ids_sha256"],
        )
        self.assertNotIn(
            "entry_ledger_rowid",
            {field.name for field in fields(GenerationScopedOutcomeObservation)},
        )
        self.assertNotIn(
            "realized_pl_jpy",
            {field.name for field in fields(GenerationScopedOutcomeObservation)},
        )

    def test_complete_resolved_first_n_is_still_diagnostic_only(self) -> None:
        rows = [
            _observation(
                f"trade-{index}",
                gateway_at=f"2026-07-13T16:{33 + index:02d}:00Z",
                entry_at=f"2026-07-13T16:{33 + index:02d}:30Z",
                close_at=f"2026-07-13T17:{index:02d}:00Z",
                pnl=100.0 if index == 0 else -25.0,
            )
            for index in range(2)
        ]

        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=rows,
            cohort_size=2,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "READY_FOR_DIAGNOSTIC_REVIEW")
        self.assertEqual(result["metrics"]["net_realized_jpy"], 75.0)
        self.assertEqual(result["metrics"]["wins"], 1)
        self.assertEqual(result["metrics"]["losses"], 1)
        self.assertFalse(result["live_permission_allowed"])
        self.assertFalse(result["gate_relaxation_allowed"])
        self.assertTrue(result["current_all_time_gate_authoritative"])

    def test_resolved_outcome_requires_all_audited_net_evidence(self) -> None:
        row = _observation(
            "net-outcome",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=-125.0,
        )
        valid = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[row],
            cohort_size=1,
            source_stream_complete=True,
        )
        self.assertEqual(valid["status"], "READY_FOR_DIAGNOSTIC_REVIEW")
        self.assertEqual(valid["metrics"]["net_realized_jpy"], -125.0)

        invalid_rows = (
            replace(row, outcome_scope="TERMINAL_CLOSE_ONLY"),
            replace(row, outcome_scope=None),
            replace(row, broker_close_event_uid=None),
            replace(row, broker_close_event_uid="bad uid"),
            replace(row, outcome_evidence_sha256=None),
            replace(row, outcome_evidence_sha256="bad"),
            replace(row, net_realized_jpy=float("nan")),
        )
        for invalid_row in invalid_rows:
            with self.subTest(invalid_row=invalid_row):
                result = evaluate_policy_generation_shadow(
                    activation=_activation(),
                    observations=[invalid_row],
                    source_stream_complete=True,
                )
                self.assertEqual(result["status"], "INVALID_SOURCE")
                self.assertIsNone(result["metrics"])

    def test_generation_outcome_aggregation_overflow_fails_closed(self) -> None:
        rows = [
            _observation(
                f"overflow-{index}",
                gateway_at=f"2026-07-13T16:{33 + index:02d}:00Z",
                entry_at=f"2026-07-13T16:{33 + index:02d}:01Z",
                close_at=f"2026-07-13T17:0{index}:00Z",
                pnl=1e308,
            )
            for index in range(2)
        ]

        result = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=rows,
            cohort_size=2,
            source_stream_complete=True,
        )

        self.assertEqual(result["status"], "INVALID_SOURCE")
        self.assertIn("generation outcome economics overflow", result["issues"])
        self.assertIsNone(result["metrics"])
        self.assertFalse(result["live_permission_allowed"])

    def test_outcome_event_and_evidence_digest_reuse_fail_closed(self) -> None:
        first = _observation(
            "outcome-first",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=100.0,
        )
        second = _observation(
            "outcome-second",
            gateway_at="2026-07-13T16:35:00Z",
            entry_at="2026-07-13T16:36:00Z",
            close_at="2026-07-13T17:01:00Z",
            pnl=200.0,
        )
        duplicate_close_uid = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                first,
                replace(
                    second,
                    broker_close_event_uid=first.broker_close_event_uid,
                ),
            ],
            source_stream_complete=True,
        )
        duplicate_evidence = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                first,
                replace(
                    second,
                    outcome_evidence_sha256=first.outcome_evidence_sha256,
                ),
            ],
            source_stream_complete=True,
        )
        entry_close_uid_collision = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                first,
                replace(
                    second,
                    broker_close_event_uid=first.entry_event_uid,
                ),
            ],
            source_stream_complete=True,
        )

        self.assertEqual(duplicate_close_uid["status"], "INVALID_SOURCE")
        self.assertIsNone(duplicate_close_uid["metrics"])
        self.assertEqual(duplicate_evidence["status"], "INVALID_SOURCE")
        self.assertIsNone(duplicate_evidence["metrics"])
        self.assertEqual(entry_close_uid_collision["status"], "INVALID_SOURCE")
        self.assertIsNone(entry_close_uid_collision["metrics"])

    def test_incomplete_stream_and_duplicate_ids_fail_closed(self) -> None:
        row = _observation(
            "duplicate",
            gateway_at="2026-07-13T16:33:00Z",
            entry_at="2026-07-13T16:34:00Z",
            close_at="2026-07-13T17:00:00Z",
            pnl=100.0,
        )
        incomplete = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[row],
        )
        duplicate = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[row, row],
            source_stream_complete=True,
        )
        duplicate_event_uid = evaluate_policy_generation_shadow(
            activation=_activation(),
            observations=[
                row,
                replace(row, trade_id="different-trade-same-event-uid"),
            ],
            source_stream_complete=True,
        )

        self.assertEqual(incomplete["status"], "SOURCE_STREAM_INCOMPLETE")
        self.assertIsNone(incomplete["metrics"])
        self.assertEqual(duplicate["status"], "INVALID_SOURCE")
        self.assertIsNone(duplicate["metrics"])
        self.assertEqual(duplicate_event_uid["status"], "INVALID_SOURCE")
        self.assertIsNone(duplicate_event_uid["metrics"])


if __name__ == "__main__":
    unittest.main()
