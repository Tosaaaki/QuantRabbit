from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from quant_rabbit.loss_close_paired_shadow import (
    PAIRED_SHADOW_STATE_CONTRACT,
    S5BidAskCandle,
    S5Ohlc,
    seal_paired_shadow_state,
)
from quant_rabbit.loss_close_price_action_shadow import (
    CANDLE_ARM,
    INVENTORY_ARM,
    PRICE_ACTION_ARM,
    PairedAblationSpec,
    PriceActionFeatureSpec,
    build_price_action_context,
    evaluate_paired_price_action_ablation,
)


UTC = timezone.utc
START = datetime(2026, 7, 14, 0, 0, 0, tzinfo=UTC)


def _state(decision: datetime) -> dict[str, object]:
    stamp = decision.strftime("%Y-%m-%dT%H:%M:%SZ")
    body: dict[str, object] = {
        "contract": PAIRED_SHADOW_STATE_CONTRACT,
        "trade_id": "pa-test-1",
        "close_decision_event_uid": "pa-test-1:decision",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100,
        "decision_timestamp_utc": stamp,
        "quote_timestamp_utc": stamp,
        "decision_bid": 100.0,
        "decision_ask": 100.1,
        "executable_close_price": 100.0,
        "take_profit": 102.0,
        "stop_loss": 99.0,
        "quote_to_jpy": 1.0,
        "broker_snapshot_sha256": "b" * 64,
        "decision_unrealized_pnl_jpy": -10.0,
        "close_verifier_receipt_sha256": "c" * 64,
        "close_verifier_verdict": "PASS",
        "technical_context_sha256": "d" * 64,
        "cost_surface_sha256": "a" * 64,
        "take_profit_exit_non_spread_cost_jpy": 1.0,
        "stop_loss_exit_non_spread_cost_jpy": 1.0,
        "control_financing_stress_jpy": 0.0,
        "read_only": True,
        "live_permission_allowed": False,
    }
    return seal_paired_shadow_state(body)


def _candles(minutes: int = 40, *, future_jump: float = 0.0) -> tuple[S5BidAskCandle, ...]:
    out = []
    total = minutes * 12 + 12
    for i in range(total):
        minute = i / 12.0
        base = 100.0 + minute * 0.01
        if i >= minutes * 12:
            base += future_jump
        bid = S5Ohlc(base, base + 0.02, base - 0.01, base + 0.01)
        ask = S5Ohlc(base + 0.10, base + 0.12, base + 0.09, base + 0.11)
        out.append(
            S5BidAskCandle(
                timestamp_utc=START + timedelta(seconds=5 * i),
                pair="USD_JPY",
                bid=bid,
                ask=ask,
                complete=True,
            )
        )
    return tuple(out)


def _arm(net: float, dd: float = 10.0) -> dict[str, object]:
    return {
        "net_jpy": net,
        "max_drawdown_jpy": dd,
        "ruin_floor_breached": False,
        "margin_closeout_proxy_breached": False,
        "unwind_complete": True,
        "fill_order_resolved": True,
    }


def _row(uid: str, split: str, inv: float, candle: float, pa: float) -> dict[str, object]:
    return {
        "event_uid": uid,
        "split": split,
        "cost_model_sha256": "e" * 64,
        "arms": {
            INVENTORY_ARM: _arm(inv, 12.0),
            CANDLE_ARM: _arm(candle, 11.0),
            PRICE_ACTION_ARM: _arm(pa, 10.0),
        },
    }


class LossClosePriceActionShadowTest(unittest.TestCase):
    def assert_read_only(self, result: dict[str, object]) -> None:
        self.assertIs(result["read_only"], True)
        self.assertIs(result["paper_permission_allowed"], False)
        self.assertIs(result["live_permission_allowed"], False)
        self.assertIs(result["broker_order_allowed"], False)
        self.assertIs(result["deployment_allowed"], False)
        self.assertIs(result["always_profit_claim_allowed"], False)

    def test_context_separates_candle_shape_from_multi_bar_structure(self) -> None:
        decision = START + timedelta(minutes=60)
        result = build_price_action_context(
            _state(decision),
            _candles(minutes=60),
            spec=PriceActionFeatureSpec(
                frames_seconds=(60, 300),
                structure_bars=4,
                regime_bars=6,
                breakout_bars=2,
                acceptance_bars=2,
                attack_tolerance_ratio=0.08,
            ),
        )

        self.assertEqual(result["status"], "CONTEXT_CALCULATED_OUTCOME_NOT_EVALUATED")
        self.assertEqual(result["frames"]["M1"]["candle_1_2"]["bars_used"], 2)
        self.assertEqual(
            result["frames"]["M5"]["price_action_multi_bar"]["bars_used"], 6
        )
        self.assertTrue(result["uses_only_candles_strictly_before_decision"])
        self.assertIn(
            result["frames"]["M5"]["price_action_multi_bar"]["chart_pattern_candidate"],
            {
                "ASCENDING_TRIANGLE_CANDIDATE",
                "DESCENDING_TRIANGLE_CANDIDATE",
                "DOUBLE_BOTTOM_CANDIDATE",
                "DOUBLE_TOP_CANDIDATE",
                "NONE",
            },
        )
        self.assertIn(
            result["cross_frame"]["setup_gate"],
            {"EVALUATE_PAIRED_SHADOW_ONLY", "SKIP_NO_PRECOMMITTED_MULTI_BAR_SETUP"},
        )
        self.assertFalse(result["hypothesis_proven"])
        self.assert_read_only(result)

    def test_future_candles_cannot_change_decision_context(self) -> None:
        decision = START + timedelta(minutes=60)
        spec = PriceActionFeatureSpec(
            frames_seconds=(60, 300),
            structure_bars=4,
            regime_bars=6,
            breakout_bars=2,
            acceptance_bars=2,
            attack_tolerance_ratio=0.08,
        )
        normal = build_price_action_context(
            _state(decision), _candles(minutes=60), spec=spec
        )
        jumped = build_price_action_context(
            _state(decision),
            _candles(minutes=60, future_jump=1000.0),
            spec=spec,
        )

        self.assertEqual(normal["frames"], jumped["frames"])
        self.assertEqual(normal["cross_frame"], jumped["cross_frame"])

    def test_incomplete_history_and_holdout_fail_closed(self) -> None:
        decision = START + timedelta(minutes=2)
        short = build_price_action_context(
            _state(decision),
            _candles(minutes=2),
            spec=PriceActionFeatureSpec(
                frames_seconds=(60,),
                structure_bars=4,
                regime_bars=6,
                breakout_bars=2,
                acceptance_bars=2,
                attack_tolerance_ratio=0.08,
            ),
        )
        self.assertEqual(short["status"], "BLOCKED")
        self.assertTrue(any("INSUFFICIENT_COMPLETED_BARS" in x for x in short["blockers"]))

        held = build_price_action_context(
            _state(START + timedelta(minutes=60)),
            _candles(minutes=60),
            holdout_used=True,
        )
        self.assertIn("HOLDOUT_USE_FORBIDDEN", held["blockers"])

    def test_no_quote_s5_gap_is_warning_for_context_but_not_hidden(self) -> None:
        decision = START + timedelta(minutes=60)
        sparse = list(_candles(minutes=60))
        del sparse[100]
        result = build_price_action_context(
            _state(decision),
            tuple(sparse),
            spec=PriceActionFeatureSpec(
                frames_seconds=(60, 300),
                structure_bars=4,
                regime_bars=6,
                breakout_bars=2,
                acceptance_bars=2,
                attack_tolerance_ratio=0.08,
            ),
        )
        self.assertEqual(result["status"], "CONTEXT_CALCULATED_OUTCOME_NOT_EVALUATED")
        self.assertGreater(result["s5_gap_warning_count"], 0)
        self.assertTrue(result["s5_gaps_allowed_for_feature_context_not_fill_order"])

    def test_paired_ablation_requires_price_action_increment_in_both_splits(self) -> None:
        rows = (
            _row("tr-1", "TRAIN", -2.0, -1.0, 2.0),
            _row("tr-2", "TRAIN", -1.0, 0.0, 3.0),
            _row("va-1", "VALIDATION", -2.0, -0.5, 1.0),
            _row("va-2", "VALIDATION", -1.0, 0.0, 1.5),
        )
        result = evaluate_paired_price_action_ablation(
            rows,
            spec=PairedAblationSpec(min_events_per_split=2, min_increment_jpy=0.0),
        )

        self.assertEqual(result["status"], "PRE_HOLDOUT_ABLATION_CALCULATED")
        self.assertTrue(result["hypothesis_survives_pre_holdout"])
        self.assertFalse(result["holdout_unlock_allowed"])
        self.assertTrue(result["ai_supervisor_evaluation_allowed"])
        self.assert_read_only(result)

    def test_price_action_failure_is_reported_not_optimised_away(self) -> None:
        rows = (
            _row("tr", "TRAIN", 1.0, 2.0, 0.0),
            _row("va", "VALIDATION", 1.0, 2.0, 0.0),
        )
        result = evaluate_paired_price_action_ablation(
            rows, spec=PairedAblationSpec(min_events_per_split=1, min_increment_jpy=0.0)
        )
        self.assertFalse(result["hypothesis_survives_pre_holdout"])
        self.assertFalse(result["ai_supervisor_evaluation_allowed"])

    def test_test_split_duplicate_identity_and_unresolved_unwind_fail_closed(self) -> None:
        forbidden = evaluate_paired_price_action_ablation(
            (_row("x", "TEST", 0.0, 0.0, 1.0),),
            spec=PairedAblationSpec(min_events_per_split=1, min_increment_jpy=0.0),
        )
        self.assertIn("FORBIDDEN_OR_INVALID_SPLIT:0", forbidden["blockers"])

        duplicate = evaluate_paired_price_action_ablation(
            (_row("x", "TRAIN", 0.0, 0.0, 1.0), _row("x", "VALIDATION", 0.0, 0.0, 1.0)),
            spec=PairedAblationSpec(min_events_per_split=1, min_increment_jpy=0.0),
        )
        self.assertIn("INVALID_OR_DUPLICATE_EVENT_UID:1", duplicate["blockers"])

        bad = _row("va", "VALIDATION", 0.0, 0.0, 1.0)
        bad["arms"][PRICE_ACTION_ARM]["unwind_complete"] = False  # type: ignore[index]
        result = evaluate_paired_price_action_ablation(
            (_row("tr", "TRAIN", 0.0, 0.0, 1.0), bad),
            spec=PairedAblationSpec(min_events_per_split=1, min_increment_jpy=0.0),
        )
        self.assertFalse(result["hypothesis_survives_pre_holdout"])


if __name__ == "__main__":
    unittest.main()
