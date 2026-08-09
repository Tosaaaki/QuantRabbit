from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from quant_rabbit.loss_close_hedge_paired_shadow import (
    FIXED_UNWIND_RULE,
    HedgeCostModel,
    HedgeExperimentSpec,
    score_loss_close_hedge_paired_shadow,
)
from quant_rabbit.loss_close_paired_shadow import (
    PAIRED_SHADOW_STATE_CONTRACT,
    S5BidAskCandle,
    S5Ohlc,
    seal_paired_shadow_state,
)


def _dt(second: int) -> datetime:
    return datetime(2026, 7, 14, 0, 0, second, tzinfo=timezone.utc)


def _ohlc(open_: float, high: float, low: float, close: float) -> S5Ohlc:
    return S5Ohlc(open=open_, high=high, low=low, close=close)


def _candle(
    second: int,
    *,
    bid: tuple[float, float, float, float],
    ask: tuple[float, float, float, float],
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=_dt(second),
        pair="USD_JPY",
        bid=_ohlc(*bid),
        ask=_ohlc(*ask),
        complete=True,
    )


def _state(**updates: object) -> dict[str, object]:
    body: dict[str, object] = {
        "contract": PAIRED_SHADOW_STATE_CONTRACT,
        "trade_id": "hedge-test-1",
        "close_decision_event_uid": "gpt-close:hedge-test-1:decision:1",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100,
        "decision_timestamp_utc": "2026-07-14T00:00:05Z",
        "quote_timestamp_utc": "2026-07-14T00:00:05Z",
        "decision_bid": 100.0,
        "decision_ask": 100.1,
        "executable_close_price": 100.0,
        "take_profit": 102.0,
        "stop_loss": 99.0,
        "quote_to_jpy": 2.0,
        "broker_snapshot_sha256": "b" * 64,
        "decision_unrealized_pnl_jpy": -20.0,
        "close_verifier_receipt_sha256": "c" * 64,
        "close_verifier_verdict": "PASS",
        "technical_context_sha256": "d" * 64,
        "cost_surface_sha256": "a" * 64,
        "take_profit_exit_non_spread_cost_jpy": 3.0,
        "stop_loss_exit_non_spread_cost_jpy": 5.0,
        "control_financing_stress_jpy": 7.0,
        "read_only": True,
        "live_permission_allowed": False,
    }
    body.update(updates)
    return seal_paired_shadow_state(body)


def _candles() -> tuple[S5BidAskCandle, ...]:
    return (
        _candle(
            0,
            bid=(100.0, 100.2, 99.9, 100.0),
            ask=(100.1, 100.3, 100.0, 100.1),
        ),
        _candle(
            5,
            bid=(100.0, 100.2, 99.5, 99.8),
            ask=(100.1, 100.3, 99.6, 99.9),
        ),
        _candle(
            10,
            bid=(99.8, 100.1, 98.9, 99.0),
            ask=(99.9, 100.2, 99.0, 99.1),
        ),
        _candle(
            15,
            bid=(99.0, 99.1, 98.0, 98.2),
            ask=(99.1, 99.2, 98.1, 98.3),
        ),
        _candle(
            20,
            bid=(98.2, 98.3, 97.0, 97.2),
            ask=(98.3, 98.4, 97.1, 97.3),
        ),
    )


def _short_state() -> dict[str, object]:
    return _state(
        side="SHORT",
        decision_bid=99.9,
        decision_ask=100.0,
        executable_close_price=100.0,
        take_profit=98.0,
        stop_loss=101.0,
    )


def _short_candles() -> tuple[S5BidAskCandle, ...]:
    return (
        _candle(0, bid=(99.9, 100.0, 99.7, 99.9), ask=(100.0, 100.1, 99.8, 100.0)),
        _candle(5, bid=(99.9, 100.4, 99.7, 100.2), ask=(100.0, 100.5, 99.8, 100.3)),
        _candle(10, bid=(100.2, 101.1, 100.0, 101.0), ask=(100.3, 101.2, 100.1, 101.1)),
        _candle(15, bid=(101.0, 101.9, 100.9, 101.8), ask=(101.1, 102.0, 101.0, 101.9)),
        _candle(20, bid=(101.8, 102.8, 101.7, 102.7), ask=(101.9, 102.9, 101.8, 102.8)),
    )


def _costs(value: float = 1.0) -> HedgeCostModel:
    return HedgeCostModel(
        original_entry_fee_jpy=value,
        original_entry_slippage_jpy=value,
        original_financing_stress_jpy=value,
        baseline_sl_fee_jpy=value,
        baseline_sl_slippage_jpy=value,
        hedge_entry_fee_jpy=value,
        hedge_entry_slippage_jpy=value,
        hedge_financing_stress_jpy=value,
        original_unwind_fee_jpy=value,
        original_unwind_slippage_jpy=value,
        hedge_unwind_fee_jpy=value,
        hedge_unwind_slippage_jpy=value,
    )


def _spec_a(scale: float = 0.25, **updates: object) -> HedgeExperimentSpec:
    values: dict[str, object] = {
        "hypothesis": "A_REVERSE_STOP",
        "hedge_timing": "SL_TRIGGER",
        "hedge_scale": scale,
        "original_entry_timestamp_utc": _dt(0),
        "original_entry_price": 100.1,
        "hedge_entry_timestamp_utc": _dt(10),
        "hedge_entry_price": 99.0,
        "unwind_timestamp_utc": _dt(20),
        "original_unwind_price": None,
        "hedge_unwind_price": 97.3,
        "unwind_rule": FIXED_UNWIND_RULE,
        "initial_equity_jpy": 10_000.0,
        "ruin_floor_jpy": 1_000.0,
        "margin_rate": 0.04,
        "margin_closeout_ratio": 0.5,
        "costs": _costs(),
        "holdout_used": False,
    }
    values.update(updates)
    return HedgeExperimentSpec(**values)  # type: ignore[arg-type]


def _spec_b(timing: str, **updates: object) -> HedgeExperimentSpec:
    entry_time = _dt(0) if timing == "INITIAL_ENTRY" else _dt(10)
    entry_price = 100.0 if timing == "INITIAL_ENTRY" else 99.0
    values: dict[str, object] = {
        "hypothesis": "B_LOSS_LOCK",
        "hedge_timing": timing,
        "hedge_scale": 1.0,
        "original_entry_timestamp_utc": _dt(0),
        "original_entry_price": 100.1,
        "hedge_entry_timestamp_utc": entry_time,
        "hedge_entry_price": entry_price,
        "unwind_timestamp_utc": _dt(20),
        "original_unwind_price": 97.2,
        "hedge_unwind_price": 97.3,
        "unwind_rule": FIXED_UNWIND_RULE,
        "initial_equity_jpy": 10_000.0,
        "ruin_floor_jpy": 1_000.0,
        "margin_rate": 0.04,
        "margin_closeout_ratio": 0.5,
        "costs": _costs(),
        "holdout_used": False,
    }
    values.update(updates)
    return HedgeExperimentSpec(**values)  # type: ignore[arg-type]


class LossCloseHedgePairedShadowTest(unittest.TestCase):
    def assert_read_only(self, result: dict[str, object]) -> None:
        self.assertIs(result["read_only"], True)
        self.assertIs(result["paper_permission_allowed"], False)
        self.assertIs(result["live_permission_allowed"], False)
        self.assertIs(result["broker_order_allowed"], False)
        self.assertIs(result["deployment_allowed"], False)
        self.assertIs(result["proof_eligible"], False)
        self.assertIs(result["always_profit_claim_allowed"], False)
        self.assertIs(result["statistical_claim_allowed"], False)
        self.assertIs(result["holdout_used"], False)

    def test_hypothesis_a_scores_only_0_25_and_0_35(self) -> None:
        quarter = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(0.25)
        )
        thirty_five = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(0.35)
        )

        self.assertEqual(quarter["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(quarter["hedge_units"], 25)
        self.assertEqual(thirty_five["hedge_units"], 35)
        self.assertGreater(thirty_five["delta_jpy"], quarter["delta_jpy"])
        self.assertEqual(
            quarter["fill_order_status"],
            "UNRESOLVED_SAME_S5_SL_CLOSE_AND_REVERSE_OPEN",
        )
        self.assertEqual(quarter["unwind_fill_order_status"], "SINGLE_HEDGE_LEG_UNWIND")
        self.assert_read_only(quarter)

        invalid = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(0.5)
        )
        self.assertEqual(invalid["status"], "BLOCKED")
        self.assertIn("HYPOTHESIS_A_SCALE_MUST_BE_0_25_OR_0_35", invalid["blockers"])

    def test_initial_equal_hedge_pays_spread_and_does_not_create_free_profit(self) -> None:
        result = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_b("INITIAL_ENTRY")
        )

        self.assertEqual(result["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(result["hedge_units"], 100)
        self.assertLess(result["alternative"]["net_jpy"], 0.0)  # type: ignore[index]
        self.assertEqual(
            result["cost_model"]["spread"],  # type: ignore[index]
            "INTRINSIC_EXECUTABLE_BID_ASK_NO_EXTRA_CHARGE",
        )
        alternative_risk = result["alternative"]["risk"]  # type: ignore[index]
        self.assertGreater(
            alternative_risk["peak_gross_notional_jpy"],  # type: ignore[index]
            alternative_risk["peak_longest_leg_margin_jpy"],  # type: ignore[index]
        )
        self.assertIs(result["strategy_hedge_authorized"], False)
        self.assertIs(result["ruin_probability_estimated"], False)
        self.assert_read_only(result)

    def test_sl_equal_hedge_can_underperform_during_trend_continuation(self) -> None:
        result = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_b("SL_TRIGGER")
        )

        self.assertEqual(result["diagnostic_outcome"], "UNDERPERFORMS_BASELINE_ON_THIS_PATH")
        trend = result["trend_continuation_after_hedge_entry"]
        self.assertGreater(trend["hedge_mfe_jpy"], 0.0)  # type: ignore[index]
        self.assertGreater(trend["hedge_mae_jpy"], 0.0)  # type: ignore[index]
        self.assertEqual(
            result["arm_definition"],
            "ORIGINAL_REMAINS_OPEN_WITH_EQUAL_OPPOSITE_LEG_UNTIL_DUAL_UNWIND",
        )
        self.assertEqual(
            result["fill_order_status"],
            "UNRESOLVED_SAME_S5_SL_TRIGGER_AND_HEDGE_OPEN_ORIGINAL_REMAINS_OPEN",
        )
        self.assertEqual(
            result["unwind_fill_order_status"],
            "UNRESOLVED_SAME_S5_DUAL_UNWIND",
        )

    def test_short_arm_uses_ask_for_sl_and_long_hedge_bid_for_unwind(self) -> None:
        spec = _spec_a(
            original_entry_price=99.9,
            hedge_entry_price=101.0,
            hedge_unwind_price=102.7,
        )
        result = score_loss_close_hedge_paired_shadow(
            _short_state(), _short_candles(), spec
        )

        self.assertEqual(result["status"], "CALCULATED_UNVERIFIED_ARTIFACT_BINDINGS")
        self.assertEqual(result["baseline_first_touch"]["quote_side"], "ASK")  # type: ignore[index]
        self.assertEqual(result["diagnostic_outcome"], "OUTPERFORMS_BASELINE_ON_THIS_PATH")

    def test_fee_slippage_and_financing_are_applied_once(self) -> None:
        zero = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(costs=_costs(0.0))
        )
        costed = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(costs=_costs(1.0))
        )

        self.assertAlmostEqual(
            zero["baseline"]["net_jpy"] - costed["baseline"]["net_jpy"],  # type: ignore[index,operator]
            5.0,
        )
        self.assertAlmostEqual(
            zero["alternative"]["net_jpy"] - costed["alternative"]["net_jpy"],  # type: ignore[index,operator]
            10.0,
        )

    def test_unwind_is_precommitted_and_after_sl(self) -> None:
        post_hoc = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(unwind_rule="BEST_PRICE_AFTER_LOOKING")
        )
        self.assertIn("UNWIND_RULE_NOT_PRECOMMITTED_FIXED_S5", post_hoc["blockers"])

        same_candle = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(unwind_timestamp_utc=_dt(10), hedge_unwind_price=99.1)
        )
        self.assertIn("UNWIND_MUST_FOLLOW_BASELINE_SL_CANDLE", same_candle["blockers"])

    def test_tp_sl_order_ambiguity_blocks_the_experiment(self) -> None:
        candles = list(_candles())
        candles[2] = _candle(
            10,
            bid=(99.8, 102.1, 98.9, 99.0),
            ask=(99.9, 102.2, 99.0, 99.1),
        )
        result = score_loss_close_hedge_paired_shadow(
            _state(), tuple(candles), _spec_a()
        )

        self.assertIn(
            "BASELINE_FIRST_TOUCH_NOT_UNAMBIGUOUS_SL:AMBIGUOUS", result["blockers"]
        )

    def test_holdout_and_unbound_prices_fail_closed(self) -> None:
        holdout = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(holdout_used=True)
        )
        self.assertIn("HOLDOUT_USE_FORBIDDEN", holdout["blockers"])

        outside = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), _spec_a(hedge_unwind_price=96.0)
        )
        self.assertIn("HEDGE_UNWIND_PRICE_OUTSIDE_S5_EXECUTABLE_RANGE", outside["blockers"])

    def test_max_drawdown_margin_and_ruin_are_deterministic_proxies(self) -> None:
        spec = _spec_a(initial_equity_jpy=250.0, ruin_floor_jpy=50.0)
        result = score_loss_close_hedge_paired_shadow(_state(), _candles(), spec)

        baseline_risk = result["baseline"]["risk"]  # type: ignore[index]
        self.assertGreater(baseline_risk["max_drawdown_jpy"], 0.0)  # type: ignore[index]
        self.assertTrue(baseline_risk["ruin_floor_breached"])  # type: ignore[index]
        self.assertFalse(baseline_risk["probability_estimated"])  # type: ignore[index]

    def test_b_requires_equal_size_and_dual_unwind(self) -> None:
        wrong_scale = score_loss_close_hedge_paired_shadow(
            _state(), _candles(), replace(_spec_b("INITIAL_ENTRY"), hedge_scale=0.35)
        )
        self.assertIn("HYPOTHESIS_B_REQUIRES_EQUAL_SCALE", wrong_scale["blockers"])

        missing_unwind = score_loss_close_hedge_paired_shadow(
            _state(),
            _candles(),
            replace(_spec_b("INITIAL_ENTRY"), original_unwind_price=None),
        )
        self.assertIn("HYPOTHESIS_B_REQUIRES_DUAL_UNWIND_PRICE", missing_unwind["blockers"])

        non_float_scale = score_loss_close_hedge_paired_shadow(
            _state(),
            _candles(),
            replace(_spec_b("INITIAL_ENTRY"), hedge_scale=1),  # type: ignore[arg-type]
        )
        self.assertIn("INVALID_POSITIVE_FLOAT:hedge_scale", non_float_scale["blockers"])


if __name__ == "__main__":
    unittest.main()
