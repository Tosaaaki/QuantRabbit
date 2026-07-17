from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import quant_rabbit.fast_bot_technical_hypothesis_vehicle_truth as truth_module
from quant_rabbit.fast_bot_learning_truth import resolve_fast_bot_learning_seat
from quant_rabbit.fast_bot_technical_hypothesis_vehicle_truth import (
    resolve_fast_bot_technical_hypothesis_vehicle_truth_v1,
    technical_hypothesis_vehicle_outcome_v1_valid,
    technical_hypothesis_vehicle_truth_v1_valid,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from tests.test_fast_bot_technical_hypothesis_vehicles import (
    CANDLE_CLOSE,
    CYCLE,
    _build,
    _sources,
)


TRUTH_RECEIPT_SHA = "b" * 64


def _candle(
    seconds: int,
    *,
    bid_o: float,
    bid_h: float,
    bid_l: float,
    bid_c: float,
    ask_o: float,
    ask_h: float,
    ask_l: float,
    ask_c: float,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=CYCLE + timedelta(seconds=seconds),
        bid_o=bid_o,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=bid_c,
        ask_o=ask_o,
        ask_h=ask_h,
        ask_l=ask_l,
        ask_c=ask_c,
    )


def _resolve(
    sources: tuple[dict[str, Any], ...],
    candles: list[S5BidAskCandle],
    *,
    base_arm_outcomes: tuple[dict[str, Any], ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature, technical, anchor, route, seat = sources
    vehicle_shadow = _build(sources)
    outcome = resolve_fast_bot_technical_hypothesis_vehicle_truth_v1(
        vehicle_shadow,
        candles,
        technical_feature_snapshot=feature,
        technical_hypothesis_shadow=technical,
        episode_anchor=anchor,
        episode_route=route,
        learning_seat=seat,
        confirmed_at_utc=CANDLE_CLOSE.isoformat(),
        resolved_at_utc=CYCLE + timedelta(hours=1),
        truth_source_receipt_sha256=TRUTH_RECEIPT_SHA,
        base_arm_outcomes=base_arm_outcomes,
    )
    return vehicle_shadow, outcome


def _outcome(value: dict[str, Any], hypothesis_id: str) -> dict[str, Any]:
    return next(
        row
        for row in value["vehicle_outcomes"]
        if row["hypothesis_id"] == hypothesis_id
    )


class FastBotTechnicalHypothesisVehicleTruthTest(unittest.TestCase):
    def test_stop_entry_and_stop_loss_use_executable_sides_and_worse_opens(
        self,
    ) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        candles = [
            _candle(
                0,
                bid_o=150.012,
                bid_h=150.018,
                bid_l=150.005,
                bid_c=150.015,
                ask_o=150.020,
                ask_h=150.026,
                ask_l=150.013,
                ask_c=150.023,
            ),
            _candle(
                5,
                bid_o=149.950,
                bid_h=149.960,
                bid_l=149.940,
                bid_c=149.955,
                ask_o=149.958,
                ask_h=149.968,
                ask_l=149.948,
                ask_c=149.963,
            ),
        ]

        _shadow, truth = _resolve(sources, candles)
        h01 = _outcome(truth, "H01")

        self.assertEqual(h01["status"], "MATURE_FILLED_STOP_LOSS")
        self.assertEqual(h01["fill_at_utc"], CYCLE.isoformat())
        self.assertEqual(h01["fill_price"], 150.020)
        self.assertTrue(h01["entry_gap_worse"])
        self.assertEqual(h01["entry_gap_pips"], 0.7)
        self.assertEqual(h01["exit_reason"], "STOP_LOSS_GAP")
        self.assertEqual(h01["exit_price"], 149.950)
        self.assertEqual(h01["post_cost_realized_pips"], -7.0)

        short_sources = _sources(
            mode="trend_down",
            branch="ACCEPTED",
            attempt_direction="DOWN",
        )
        short_candles = [
            _candle(
                0,
                bid_o=149.970,
                bid_h=149.980,
                bid_l=149.960,
                bid_c=149.975,
                ask_o=149.978,
                ask_h=149.988,
                ask_l=149.968,
                ask_c=149.983,
            ),
            _candle(
                5,
                bid_o=150.032,
                bid_h=150.042,
                bid_l=150.022,
                bid_c=150.037,
                ask_o=150.040,
                ask_h=150.050,
                ask_l=150.030,
                ask_c=150.045,
            ),
        ]
        _short_shadow, short_truth = _resolve(short_sources, short_candles)
        short_h01 = _outcome(short_truth, "H01")

        self.assertEqual(short_h01["fill_price"], 149.970)
        self.assertTrue(short_h01["entry_gap_worse"])
        self.assertEqual(short_h01["exit_reason"], "STOP_LOSS_GAP")
        self.assertEqual(short_h01["exit_price"], 150.040)
        self.assertEqual(short_h01["post_cost_realized_pips"], -7.0)

    def test_fill_candle_and_later_same_s5_touches_are_stop_first(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        fill_candle_ambiguous = [
            _candle(
                0,
                bid_o=150.005,
                bid_h=150.100,
                bid_l=149.990,
                bid_c=150.020,
                ask_o=150.013,
                ask_h=150.108,
                ask_l=149.998,
                ask_c=150.028,
            )
        ]
        _shadow, fill_truth = _resolve(sources, fill_candle_ambiguous)
        fill_h01 = _outcome(fill_truth, "H01")

        self.assertEqual(
            fill_h01["exit_reason"],
            "STOP_LOSS_AMBIGUOUS_FILL_S5",
        )
        self.assertTrue(fill_h01["ambiguous_same_s5"])
        self.assertLess(fill_h01["post_cost_realized_pips"], 0.0)

        later_ambiguous = [
            _candle(
                0,
                bid_o=150.005,
                bid_h=150.020,
                bid_l=149.990,
                bid_c=150.015,
                ask_o=150.013,
                ask_h=150.028,
                ask_l=149.998,
                ask_c=150.023,
            ),
            _candle(
                5,
                bid_o=150.015,
                bid_h=150.100,
                bid_l=149.950,
                bid_c=150.020,
                ask_o=150.023,
                ask_h=150.108,
                ask_l=149.958,
                ask_c=150.028,
            ),
        ]
        _shadow, later_truth = _resolve(sources, later_ambiguous)
        later_h01 = _outcome(later_truth, "H01")

        self.assertEqual(
            later_h01["exit_reason"],
            "STOP_LOSS_AMBIGUOUS_SAME_S5",
        )
        self.assertTrue(later_h01["ambiguous_same_s5"])
        self.assertEqual(later_h01["exit_price"], 149.963)

    def test_intrabar_fill_does_not_use_the_pre_fill_open_as_a_stop_gap(
        self,
    ) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        # The executable BID opened below the frozen SL, but the executable
        # ASK did not trigger the buy STOP until later inside the same S5.
        # That pre-fill open cannot be used as a post-fill gap price.
        candles = [
            _candle(
                0,
                bid_o=149.950,
                bid_h=150.020,
                bid_l=149.940,
                bid_c=150.010,
                ask_o=150.000,
                ask_h=150.028,
                ask_l=149.948,
                ask_c=150.018,
            )
        ]

        _shadow, truth = _resolve(sources, candles)
        h01 = _outcome(truth, "H01")

        self.assertEqual(h01["exit_reason"], "STOP_LOSS_AMBIGUOUS_FILL_S5")
        self.assertFalse(h01["stop_gap_worse"])
        self.assertEqual(h01["fill_price"], 150.013)
        self.assertEqual(h01["exit_price"], 149.963)
        self.assertEqual(h01["post_cost_realized_pips"], -5.0)

    def test_sparse_no_tick_path_is_not_synthesized(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        candles = [
            _candle(
                0,
                bid_o=150.005,
                bid_h=150.020,
                bid_l=149.990,
                bid_c=150.015,
                ask_o=150.013,
                ask_h=150.028,
                ask_l=149.998,
                ask_c=150.023,
            ),
            # The absent +5 second slot remains no-tick.  The next observed
            # candle owns the eventual executable TP touch.
            _candle(
                10,
                bid_o=150.020,
                bid_h=150.100,
                bid_l=150.010,
                bid_c=150.095,
                ask_o=150.028,
                ask_h=150.108,
                ask_l=150.018,
                ask_c=150.103,
            ),
        ]

        _shadow, truth = _resolve(sources, candles)
        h01 = _outcome(truth, "H01")

        self.assertEqual(h01["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(
            h01["exit_at_utc"], (CYCLE + timedelta(seconds=10)).isoformat()
        )
        self.assertFalse(truth["missing_no_tick_intervals_synthesized"])
        self.assertGreater(truth["truth_no_tick_slot_count"], 0)
        self.assertEqual(truth["truth_candle_count"], 2)

    def test_unfilled_stop_is_zero_and_hold_end_keeps_separate_mark(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        below_entry = [
            _candle(
                0,
                bid_o=149.980,
                bid_h=149.990,
                bid_l=149.970,
                bid_c=149.985,
                ask_o=149.988,
                ask_h=149.998,
                ask_l=149.978,
                ask_c=149.993,
            )
        ]

        _shadow, unfilled_truth = _resolve(sources, below_entry)
        unfilled = _outcome(unfilled_truth, "H01")
        self.assertEqual(unfilled["status"], "MATURE_UNFILLED")
        self.assertFalse(unfilled["filled"])
        self.assertEqual(unfilled["post_cost_realized_pips"], 0.0)

        open_path = [
            _candle(
                0,
                bid_o=150.005,
                bid_h=150.020,
                bid_l=149.990,
                bid_c=150.015,
                ask_o=150.013,
                ask_h=150.028,
                ask_l=149.998,
                ask_c=150.023,
            ),
            _candle(
                20,
                bid_o=150.030,
                bid_h=150.050,
                bid_l=150.020,
                bid_c=150.040,
                ask_o=150.038,
                ask_h=150.058,
                ask_l=150.028,
                ask_c=150.048,
            ),
        ]
        _shadow, open_truth = _resolve(sources, open_path)
        held = _outcome(open_truth, "H01")

        self.assertEqual(held["status"], "MATURE_FILLED_HOLD_END_FULL_STOP")
        self.assertEqual(held["post_cost_realized_pips"], -5.0)
        self.assertEqual(held["hold_end_mark_price"], 150.040)
        self.assertEqual(held["hold_end_mark_pips"], 2.7)

    def test_actual_fill_starts_hold_clock_and_later_tp_is_not_credited(
        self,
    ) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        candles = [
            _candle(
                0,
                bid_o=150.005,
                bid_h=150.020,
                bid_l=149.990,
                bid_c=150.015,
                ask_o=150.013,
                ask_h=150.028,
                ask_l=149.998,
                ask_c=150.023,
            ),
            # H01 filled at activation and has a 900-second hold.  This TP
            # appears before the latest-possible-fill maturity (+990s), but
            # after this actual fill's own maturity and must be ignored.
            _candle(
                905,
                bid_o=150.090,
                bid_h=150.100,
                bid_l=150.080,
                bid_c=150.095,
                ask_o=150.098,
                ask_h=150.108,
                ask_l=150.088,
                ask_c=150.103,
            ),
        ]

        _shadow, truth = _resolve(sources, candles)
        h01 = _outcome(truth, "H01")

        self.assertEqual(h01["status"], "MATURE_FILLED_HOLD_END_FULL_STOP")
        self.assertEqual(
            h01["exit_at_utc"], (CYCLE + timedelta(seconds=900)).isoformat()
        )
        self.assertEqual(h01["post_cost_realized_pips"], -5.0)
        self.assertEqual(h01["truth_candle_count"], 1)

    def test_ineligible_diagnostic_vehicle_never_becomes_scorecard_eligible(
        self,
    ) -> None:
        sources = _sources(
            mode="trend",
            branch="ACCEPTED",
            input_blocked=True,
        )
        vehicle_shadow, truth = _resolve(sources, [])

        self.assertEqual(vehicle_shadow["status"], "EMITTED_DIAGNOSTIC_ONLY")
        self.assertEqual(truth["scorecard_eligible_outcome_count"], 0)
        self.assertFalse(truth["causal_input_proof_eligible"])
        self.assertFalse(truth["paired_direction_proof_eligible"])
        self.assertFalse(truth["technical_hypothesis_proof_eligible"])
        self.assertFalse(truth["scorecard_eligible"])
        self.assertIn(
            "INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY",
            truth["scorecard_ineligibility_reasons"],
        )
        self.assertTrue(truth["vehicle_outcomes"])
        self.assertTrue(
            all(row["scorecard_eligible"] is False for row in truth["vehicle_outcomes"])
        )
        self.assertTrue(
            all(
                row["causal_input_proof_eligible"] is False
                and row["paired_direction_proof_eligible"] is False
                and row["technical_hypothesis_proof_eligible"] is False
                for row in truth["vehicle_outcomes"]
            )
        )

    def test_proxy_is_reason_coded_unresolved_and_h08_is_exact_zero(self) -> None:
        sources = _sources(mode="range", branch="REJECTED")
        _shadow, truth = _resolve(sources, [])

        for hypothesis_id in ("H03", "H05"):
            proxy = _outcome(truth, hypothesis_id)
            self.assertEqual(
                proxy["status"],
                "UNRESOLVED_BASE_ARM_OUTCOME_REQUIRED",
            )
            self.assertEqual(
                proxy["resolution_reasons"],
                ["SEALED_MATCHED_BASE_ARM_OUTCOME_MISSING"],
            )
            self.assertIsNone(proxy["post_cost_realized_pips"])
            self.assertTrue(proxy["ex_ante_scorecard_eligible"])
            self.assertFalse(proxy["scorecard_result_available"])
            self.assertFalse(proxy["scorecard_eligible"])
            self.assertIn(
                "OUTCOME_RESULT_UNAVAILABLE",
                proxy["scorecard_ineligibility_reasons"],
            )
        h08 = _outcome(truth, "H08")
        self.assertEqual(h08["status"], "ZERO_PNL_CONTROL")
        self.assertFalse(h08["filled"])
        self.assertEqual(h08["post_cost_realized_pips"], 0.0)
        self.assertEqual(truth["scorecard_eligible_outcome_count"], 1)
        self.assertEqual(truth["scorecard_result_available_outcome_count"], 1)

    def test_proxy_requires_and_accepts_the_sealed_matching_base_arm(self) -> None:
        sources = _sources(mode="range", branch="REJECTED")
        seat = sources[-1]
        learning_outcome = resolve_fast_bot_learning_seat(
            seat,
            [],
            resolved_at_utc=CYCLE + timedelta(hours=1),
            truth_chunk_sha256=["c" * 64],
        )

        _shadow, truth = _resolve(
            sources,
            [],
            base_arm_outcomes=(learning_outcome,),
        )

        for hypothesis_id in ("H03", "H05"):
            proxy = _outcome(truth, hypothesis_id)
            self.assertEqual(proxy["status"], "MATURE_PROXY_RESOLVED")
            self.assertFalse(proxy["filled"])
            self.assertEqual(proxy["post_cost_realized_pips"], 0.0)
            self.assertEqual(proxy["resolution_reasons"], [])
            self.assertEqual(
                proxy["base_proxy_source"]["learning_outcome_contract_sha256"],
                learning_outcome["contract_sha256"],
            )
            self.assertEqual(
                proxy["base_proxy_source"]["source_resolved_at_utc"],
                (CYCLE + timedelta(hours=1)).isoformat(),
            )
        h03_vehicle = next(
            row for row in _shadow["vehicles"] if row["hypothesis_id"] == "H03"
        )
        pre_activation = deepcopy(_outcome(truth, "H03"))
        pre_activation.update(
            {
                "filled": True,
                "fill_at_utc": (CYCLE - timedelta(days=1)).isoformat(),
                "fill_price": pre_activation["base_proxy_source"]["source_entry_price"],
                "exit_reason": "TAKE_PROFIT",
                "exit_at_utc": (
                    CYCLE - timedelta(days=1) + timedelta(seconds=5)
                ).isoformat(),
                "stop_gap_worse": False,
                "ambiguous_same_s5": False,
                "post_cost_realized_pips": 1.0,
                "truth_candle_count": 1,
            }
        )
        pre_activation["base_proxy_source"].update(
            {
                "source_filled": True,
                "source_fill_at_utc": pre_activation["fill_at_utc"],
                "source_exit_at_utc": pre_activation["exit_at_utc"],
                "source_exit_reason": "TAKE_PROFIT",
                "source_post_cost_realized_pips": 1.0,
                "source_ambiguous_same_s5": False,
                "source_truth_candle_count": 1,
            }
        )
        pre_activation = truth_module._seal_outcome(pre_activation)
        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                pre_activation,
                vehicle_row=h03_vehicle,
            )
        )

        wrong_sign = deepcopy(_outcome(truth, "H03"))
        wrong_sign.update(
            {
                "filled": True,
                "fill_at_utc": CYCLE.isoformat(),
                "fill_price": wrong_sign["base_proxy_source"]["source_entry_price"],
                "exit_reason": "TAKE_PROFIT",
                "exit_at_utc": (CYCLE + timedelta(seconds=5)).isoformat(),
                "stop_gap_worse": False,
                "ambiguous_same_s5": False,
                "post_cost_realized_pips": -999.0,
                "truth_candle_count": 1,
            }
        )
        wrong_sign["base_proxy_source"].update(
            {
                "source_filled": True,
                "source_fill_at_utc": wrong_sign["fill_at_utc"],
                "source_exit_at_utc": wrong_sign["exit_at_utc"],
                "source_exit_reason": "TAKE_PROFIT",
                "source_post_cost_realized_pips": -999.0,
                "source_ambiguous_same_s5": False,
                "source_truth_candle_count": 1,
            }
        )
        wrong_sign = truth_module._seal_outcome(wrong_sign)
        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                wrong_sign,
                vehicle_row=h03_vehicle,
            )
        )
        for row in truth["vehicle_outcomes"]:
            self.assertTrue(row["historical_only"])
            self.assertTrue(row["diagnostic_only"])
            self.assertEqual(row["order_authority"], "NONE")
            self.assertFalse(row["live_permission"])
            self.assertFalse(row["broker_mutation_allowed"])
            self.assertEqual(row["order_intents"], [])

    def test_rebuild_and_sealed_validators_reject_tampering(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        feature, technical, anchor, route, seat = sources
        vehicle_shadow, truth = _resolve(sources, [])
        h01_vehicle = next(
            row for row in vehicle_shadow["vehicles"] if row["hypothesis_id"] == "H01"
        )
        h01_outcome = _outcome(truth, "H01")

        self.assertTrue(
            technical_hypothesis_vehicle_outcome_v1_valid(
                h01_outcome,
                vehicle_row=h01_vehicle,
            )
        )
        self.assertTrue(
            technical_hypothesis_vehicle_truth_v1_valid(
                truth,
                vehicle_shadow,
                [],
                technical_feature_snapshot=feature,
                technical_hypothesis_shadow=technical,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
                resolved_at_utc=CYCLE + timedelta(hours=1),
                truth_source_receipt_sha256=TRUTH_RECEIPT_SHA,
            )
        )
        tampered_truth = deepcopy(truth)
        tampered_truth["truth_candle_count"] = 999
        self.assertFalse(
            technical_hypothesis_vehicle_truth_v1_valid(
                tampered_truth,
                vehicle_shadow,
                [],
                technical_feature_snapshot=feature,
                technical_hypothesis_shadow=technical,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
                resolved_at_utc=CYCLE + timedelta(hours=1),
                truth_source_receipt_sha256=TRUTH_RECEIPT_SHA,
            )
        )
        tampered_shadow = deepcopy(vehicle_shadow)
        tampered_shadow["activation_at_utc"] = datetime(
            2026,
            7,
            17,
            7,
            35,
            15,
            tzinfo=timezone.utc,
        ).isoformat()
        with self.assertRaisesRegex(ValueError, "causal rebuild"):
            resolve_fast_bot_technical_hypothesis_vehicle_truth_v1(
                tampered_shadow,
                [],
                technical_feature_snapshot=feature,
                technical_hypothesis_shadow=technical,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
                resolved_at_utc=CYCLE + timedelta(hours=1),
                truth_source_receipt_sha256=TRUTH_RECEIPT_SHA,
            )

    def test_outcome_validator_rejects_resealed_result_contradictions(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        vehicle_shadow, truth = _resolve(
            sources,
            [
                _candle(
                    0,
                    bid_o=150.005,
                    bid_h=150.020,
                    bid_l=149.990,
                    bid_c=150.015,
                    ask_o=150.013,
                    ask_h=150.028,
                    ask_l=149.998,
                    ask_c=150.023,
                ),
                _candle(
                    10,
                    bid_o=150.090,
                    bid_h=150.100,
                    bid_l=150.080,
                    bid_c=150.095,
                    ask_o=150.098,
                    ask_h=150.108,
                    ask_l=150.088,
                    ask_c=150.103,
                ),
            ],
        )
        vehicle = next(
            row for row in vehicle_shadow["vehicles"] if row["hypothesis_id"] == "H01"
        )
        tampered = deepcopy(_outcome(truth, "H01"))
        tampered.update(
            {
                "post_cost_realized_pips": -999.0,
                "exit_reason": "STOP_LOSS",
                "fill_at_utc": None,
                "exit_at_utc": None,
            }
        )
        tampered = truth_module._seal_outcome(tampered)

        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                tampered,
                vehicle_row=vehicle,
            )
        )

        unaligned_exit = deepcopy(_outcome(truth, "H01"))
        unaligned_exit["exit_at_utc"] = (
            datetime.fromisoformat(unaligned_exit["exit_at_utc"]) + timedelta(seconds=1)
        ).isoformat()
        unaligned_exit = truth_module._seal_outcome(unaligned_exit)
        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                unaligned_exit,
                vehicle_row=vehicle,
            )
        )

        zero_observed = deepcopy(_outcome(truth, "H01"))
        grid_count = (
            zero_observed["truth_candle_count"]
            + zero_observed["truth_no_tick_slot_count"]
        )
        zero_observed["truth_candle_count"] = 0
        zero_observed["truth_no_tick_slot_count"] = grid_count
        zero_observed = truth_module._seal_outcome(zero_observed)
        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                zero_observed,
                vehicle_row=vehicle,
            )
        )

    def test_unresolved_proxy_validator_rejects_unknown_reason_code(self) -> None:
        sources = _sources(mode="range", branch="REJECTED")
        vehicle_shadow, truth = _resolve(sources, [])
        vehicle = next(
            row for row in vehicle_shadow["vehicles"] if row["hypothesis_id"] == "H03"
        )
        bogus = deepcopy(_outcome(truth, "H03"))
        bogus["resolution_reasons"] = ["BOGUS"]
        bogus["scorecard_ineligibility_reasons"] = [
            "BOGUS",
            "OUTCOME_RESULT_UNAVAILABLE",
        ]
        bogus = truth_module._seal_outcome(bogus)

        self.assertFalse(
            technical_hypothesis_vehicle_outcome_v1_valid(
                bogus,
                vehicle_row=vehicle,
            )
        )


if __name__ == "__main__":
    unittest.main()
