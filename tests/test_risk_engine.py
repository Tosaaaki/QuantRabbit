from __future__ import annotations

import os
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.risk import RiskEngine, RiskPolicy


def snapshot(
    *,
    positions=(),
    orders=(),
    hedging_enabled: bool = False,
    nav_jpy: float = 200_000.0,
    balance_jpy: float = 200_000.0,
    margin_used_jpy: float = 0.0,
    margin_available_jpy: float = 200_000.0,
) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=tuple(positions),
        orders=tuple(orders),
        quotes={
            "EUR_USD": Quote("EUR_USD", bid=1.17322, ask=1.17330, timestamp_utc=now),
            "USD_JPY": Quote("USD_JPY", bid=156.640, ask=156.648, timestamp_utc=now),
        },
        account=AccountSummary(
            nav_jpy=nav_jpy,
            balance_jpy=balance_jpy,
            margin_used_jpy=margin_used_jpy,
            margin_available_jpy=margin_available_jpy,
            hedging_enabled=hedging_enabled,
            fetched_at_utc=now,
        ),
    )


TEST_MAX_LOSS_JPY = 500.0


def _capped_engine(*, policy: RiskPolicy | None = None, **kwargs) -> RiskEngine:
    if policy is None:
        policy = RiskPolicy(max_loss_jpy=TEST_MAX_LOSS_JPY)
    elif policy.max_loss_jpy is None:
        policy = replace(policy, max_loss_jpy=TEST_MAX_LOSS_JPY)
    return RiskEngine(policy=policy, **kwargs)


class RiskEngineTest(unittest.TestCase):
    def test_default_policy_has_no_jpy_loss_cap_literal(self) -> None:
        self.assertIsNone(RiskPolicy().max_loss_jpy)

    def test_missing_loss_cap_blocks_validation_instead_of_using_literal_fallback(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="missing_loss_cap_must_fail_closed",
        )

        decision = RiskEngine().validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_CAP_MISSING", {issue.code for issue in decision.issues})

    def test_valid_dry_run_intent_passes(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="eurusd_direct_usd_continuation",
        )
        decision = _capped_engine().validate(intent, snapshot())
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertLessEqual(decision.metrics.risk_jpy, TEST_MAX_LOSS_JPY)
        self.assertGreaterEqual(decision.metrics.reward_risk, 1.2)
        self.assertIn("MISSING_MARKET_CONTEXT", {issue.code for issue in decision.issues})

    def test_validation_time_freezes_quote_freshness_for_batch_generation(self) -> None:
        quote_time = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
        snap = BrokerSnapshot(
            fetched_at_utc=quote_time,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17322, ask=1.17330, timestamp_utc=quote_time),
                "USD_JPY": Quote("USD_JPY", bid=156.640, ask=156.648, timestamp_utc=quote_time),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=quote_time,
            ),
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="batch_generation_must_not_self_expire_quotes",
        )

        frozen = _capped_engine(validation_time_utc=quote_time + timedelta(seconds=5)).validate(intent, snap)
        realtime = _capped_engine().validate(intent, snap)

        self.assertTrue(frozen.allowed, frozen.block_reasons)
        self.assertNotIn("STALE_QUOTE", {issue.code for issue in frozen.issues})
        self.assertFalse(realtime.allowed)
        self.assertIn("STALE_QUOTE", {issue.code for issue in realtime.issues})

    def test_live_send_requires_live_enabled(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="eurusd_direct_usd_continuation",
        )
        decision = _capped_engine(live_enabled=False).validate(intent, snapshot(), for_live_send=True)
        self.assertFalse(decision.allowed)
        self.assertIn("LIVE_DISABLED", {issue.code for issue in decision.issues})
        self.assertIn("MISSING_MARKET_CONTEXT", {issue.code for issue in decision.issues})

    def test_valid_context_removes_market_story_warning(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="eurusd_direct_usd_continuation",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="USD softness lets EUR squeeze higher",
                chart_story="green staircase into upper band with shallow pullbacks",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="1.1716 loses on M5 bodies",
                event_risk="NFP later, no hold through spread window",
                session="London-NY overlap",
            ),
        )
        decision = _capped_engine().validate(intent, snapshot())
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("MISSING_MARKET_CONTEXT", {issue.code for issue in decision.issues})

    def test_negative_capture_asymmetry_blocks_loss_larger_than_average_winner(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17600,
            sl=1.17130,
            thesis="loss_asymmetry_repair_must_size_down",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="candidate has edge but exit payoff is under repair",
                chart_story="trend staircase",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})
        self.assertNotIn("LOSS_CAP_EXCEEDED", {issue.code for issue in decision.issues})

    def test_negative_capture_asymmetry_allows_loss_inside_average_winner(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17600,
            sl=1.17130,
            thesis="loss_asymmetry_repair_sized_inside_average_winner",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="candidate risk is smaller than the observed average winner",
                chart_story="trend staircase",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_tp_proven_relaxed_loss_asymmetry_uses_normal_loss_cap(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=3000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="tp_proven_harvest_can_use_normal_cap",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="broker TP harvest shape has proved payoff while market closes leak",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 1000.0,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_USD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "capture_take_profit_trades": 93,
                "capture_take_profit_expectancy_jpy": 504.0,
                "capture_take_profit_losses": 0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_tp_proof_collection_min_lot_mode_defends_thin_evidence_receipts(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="thin_tp_collection_can_fund_min_lot_without_claiming_proven",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="thin exact broker TP harvest shape has positive stressed payoff",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 50.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "TP_PROOF_COLLECTION_MIN_LOT",
                "loss_asymmetry_guard_loss_cap_jpy": 50.0,
                "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 266.4,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_USD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "capture_take_profit_trades": 10,
                "capture_take_profit_wins": 10,
                "capture_take_profit_expectancy_jpy": 600.0,
                "capture_take_profit_avg_win_jpy": 600.0,
                "capture_take_profit_avg_loss_jpy": 0.0,
                "capture_take_profit_losses": 0,
                "capture_market_close_expectancy_jpy": -892.1,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_oanda_firepower_min_lot_mode_defends_matching_harvest_receipts(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="matching_oanda_firepower_can_fund_min_lot_without_bypassing_normal_cap",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="OANDA firepower vehicle matches this attached TP harvest shape",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 50.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "OANDA_CAMPAIGN_FIREPOWER_MIN_LOT",
                "loss_asymmetry_guard_loss_cap_jpy": 50.0,
                "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 266.4,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_oanda_campaign_min_lot_sizing": True,
                "positive_rotation_oanda_campaign_min_lot_units": 1000,
                "positive_rotation_oanda_campaign_min_lot_loss_jpy": 266.4,
                "positive_rotation_oanda_campaign_firepower_status": (
                    "VERIFIED_TARGET_10_ROUTE_ESTIMATED"
                ),
                "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "positive_rotation_oanda_campaign_estimated_return_pct_per_active_day": 20.0,
                "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day": (
                    8.0
                ),
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_oanda_firepower_min_lot_mode_requires_matching_vehicle(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="firepower_label_without_vehicle_match_cannot_bypass_avg_win_cap",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="metadata must prove the exact OANDA vehicle match",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 50.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "OANDA_CAMPAIGN_FIREPOWER_MIN_LOT",
                "loss_asymmetry_guard_loss_cap_jpy": 50.0,
                "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 266.4,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_oanda_campaign_min_lot_sizing": True,
                "positive_rotation_oanda_campaign_min_lot_units": 1000,
                "positive_rotation_oanda_campaign_min_lot_loss_jpy": 266.4,
                "positive_rotation_oanda_campaign_firepower_status": (
                    "VERIFIED_TARGET_10_ROUTE_ESTIMATED"
                ),
                "positive_rotation_oanda_campaign_firepower_vehicle_match": False,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "positive_rotation_oanda_campaign_estimated_return_pct_per_active_day": 20.0,
                "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day": (
                    8.0
                ),
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_tp_proven_relaxed_requires_scoped_capture_proof(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=3000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="global_tp_proof_cannot_exempt_unscoped_shape",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="global TP proof is not enough for this pair-side-method",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 1000.0,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "capture_take_profit_trades": 93,
                "capture_take_profit_expectancy_jpy": 504.0,
                "capture_take_profit_losses": 0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_stale_capture_economics_blocks_even_tp_proven_relaxation(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=3000,
            entry=1.17300,
            tp=1.17600,
            sl=1.17130,
            thesis="stale_capture_economics_cannot_grant_rotation",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="broker TP proof must be current versus execution ledger",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "capture_economics_stale": True,
                "capture_economics_generated_at_utc": "2026-06-17T14:05:31+00:00",
                "capture_economics_latest_realized_ts_utc": "2026-06-19T01:02:03+00:00",
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 1000.0,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "capture_take_profit_trades": 93,
                "capture_take_profit_expectancy_jpy": 504.0,
                "capture_take_profit_losses": 0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("CAPTURE_ECONOMICS_STALE", {issue.code for issue in decision.issues})
        self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_tp_proven_relaxed_loss_asymmetry_does_not_exempt_market_entries(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17600,
            sl=1.17130,
            thesis="market_entry_cannot_claim_tp_proven_relaxation",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="metadata cannot bypass the average-winner cap",
                chart_story="range lower rail",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1713 loses on M5 bodies",
            ),
            metadata={
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_avg_win_jpy": 600.0,
                "capture_avg_loss_jpy": 1100.0,
                "loss_asymmetry_guard_active": True,
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                "loss_asymmetry_guard_effective_max_loss_jpy": 1000.0,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "attach_take_profit_on_fill": True,
                "tp_target_intent": "HARVEST",
                "capture_take_profit_trades": 93,
                "capture_take_profit_expectancy_jpy": 504.0,
                "capture_take_profit_losses": 0,
            },
        )

        decision = _capped_engine(policy=RiskPolicy(max_loss_jpy=1000.0)).validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", {issue.code for issue in decision.issues})

    def test_forecast_geometry_inside_spread_noise_blocks_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="forecast_edge_must_clear_execution_noise",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="USD softness lets EUR squeeze higher",
                chart_story="green staircase into upper band with shallow pullbacks",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation must not sit inside spread noise",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_target_price": 1.17355,
                "forecast_invalidation_price": 1.17305,
            },
        )

        dry_run = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=False)
        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        dry_codes = {issue.code: issue.severity for issue in dry_run.issues}
        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertTrue(dry_run.allowed, dry_run.block_reasons)
        self.assertEqual(dry_codes["FORECAST_TARGET_TOO_THIN_FOR_SPREAD"], "WARN")
        self.assertEqual(dry_codes["FORECAST_INVALIDATION_TOO_THIN_FOR_SPREAD"], "WARN")
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_TARGET_TOO_THIN_FOR_SPREAD"], "BLOCK")
        self.assertEqual(live_codes["FORECAST_INVALIDATION_TOO_THIN_FOR_SPREAD"], "BLOCK")

    def test_forecast_geometry_clearing_spread_noise_remains_live_sendable(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="forecast_edge_clears_execution_noise",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="USD softness lets EUR squeeze higher",
                chart_story="green staircase into upper band with shallow pullbacks",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation clears spread noise",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_target_price": 1.17400,
                "forecast_invalidation_price": 1.17270,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 40,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("FORECAST_TARGET_TOO_THIN_FOR_SPREAD", codes)
        self.assertNotIn("FORECAST_INVALIDATION_TOO_THIN_FOR_SPREAD", codes)

    def test_high_headline_forecast_timeout_drag_blocks_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="headline_hit_rate_ignores_timeout_drag",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="headline forecast looks perfect but rarely arrives in time",
                chart_story="trend setup with stale projection timing",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation clears spread noise",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_target_price": 1.17400,
                "forecast_invalidation_price": 1.17270,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 80,
                "forecast_directional_economic_hit_rate": 0.50,
                "forecast_directional_economic_samples": 160,
                "forecast_directional_timeout_rate": 0.50,
                "forecast_directional_timeout_count": 80,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        codes = {issue.code: issue for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE", codes)
        self.assertIn("economic_hit_rate=0.50", codes["FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE"].message)

    def test_directional_timeout_rate_without_economic_rate_blocks_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="timeout_rate_without_economic_rate_is_unproved",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="headline bucket is high but timeout accounting is incomplete",
                chart_story="trend setup with incomplete economic calibration",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation clears spread noise",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_target_price": 1.17400,
                "forecast_invalidation_price": 1.17270,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 80,
                "forecast_directional_timeout_rate": 0.20,
                "forecast_directional_timeout_count": 20,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE", {issue.code for issue in decision.issues})

    def test_opposite_direction_forecast_blocks_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="forecast_must_match_live_entry_side",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="candidate lane is long, but pair forecast points down",
                chart_story="pullback lane remains visible",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="1.1716 loses on M5 bodies",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.83,
                "forecast_target_price": 1.17180,
                "forecast_invalidation_price": 1.17420,
                "forecast_directional_calibration_name": "directional_forecast_down",
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 40,
            },
        )

        dry_run = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=False)
        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        dry_codes = {issue.code: issue.severity for issue in dry_run.issues}
        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertTrue(dry_run.allowed, dry_run.block_reasons)
        self.assertEqual(dry_codes["FORECAST_DIRECTION_CONFLICT"], "WARN")
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTION_CONFLICT"], "BLOCK")

    def test_unclear_forecast_blocks_fresh_breakout_failure_limit_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17400,
            tp=1.17330,
            sl=1.17450,
            thesis="do_not_send_unclear_forecast_to_live",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="failed upside break rejects, but pair forecast is not executable",
                chart_story="seller response near resistance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UNCLEAR",
                "forecast_confidence": 0.22,
            },
        )

        dry_run = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=False)
        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        dry_codes = {issue.code for issue in dry_run.issues}
        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertNotIn("FORECAST_NOT_EXECUTABLE_FOR_LIVE", dry_codes)
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_NOT_EXECUTABLE_FOR_LIVE"], "BLOCK")

    def test_low_confidence_same_side_directional_forecast_blocks_live_send_without_support(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17400,
            tp=1.17330,
            sl=1.17450,
            thesis="do_not_send_low_confidence_same_side_forecast_to_live",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="selected side matches the forecast, but confidence is below live floor",
                chart_story="seller response near resistance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                },
            },
        )

        dry_run = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=False)
        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        dry_codes = {issue.code for issue in dry_run.issues}
        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertTrue(dry_run.allowed, dry_run.block_reasons)
        self.assertNotIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", dry_codes)
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"], "BLOCK")

    def test_technical_harvest_precision_supports_low_confidence_short_scalp_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17330,
            tp=1.17280,
            sl=1.17370,
            thesis="audited EUR_USD short low-ATR scalp",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="selected side matches the forecast, but confidence is below live floor",
                chart_story="M1 ATR low",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="4 pip stop",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "chart_direction_bias": "SHORT",
                "m1_atr_percentile_100": 0.10,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                },
            },
        )

        base_snapshot = snapshot()
        narrow_spread_snapshot = replace(
            base_snapshot,
            quotes={
                **base_snapshot.quotes,
                "EUR_USD": Quote(
                    "EUR_USD",
                    bid=1.17326,
                    ask=1.17330,
                    timestamp_utc=base_snapshot.fetched_at_utc,
                ),
            },
        )

        decision = _capped_engine(live_enabled=True).validate(
            intent,
            narrow_spread_snapshot,
            for_live_send=True,
        )

        codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", codes)
        self.assertTrue(intent.metadata["technical_harvest_precision_live_ready"])
        self.assertEqual(
            intent.metadata["technical_harvest_precision_support"]["name"],
            "EUR_USD_DOWN_M1_ATR_LOW_TP5_SL4",
        )

    def test_technical_harvest_negative_bucket_blocks_low_confidence_scalp_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17330,
            tp=1.17280,
            sl=1.17370,
            thesis="audited EUR_USD short low-ATR scalp with opposed M5 EMA slope",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="selected side matches the forecast, but M5 EMA slope is the audited loss bucket",
                chart_story="M1 ATR low; M5 EMA slope opposed",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="4 pip stop",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "chart_direction_bias": "SHORT",
                "m1_atr_percentile_100": 0.10,
                "m5_ema_slope_5": 0.20,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                },
            },
        )

        base_snapshot = snapshot()
        narrow_spread_snapshot = replace(
            base_snapshot,
            quotes={
                **base_snapshot.quotes,
                "EUR_USD": Quote(
                    "EUR_USD",
                    bid=1.17326,
                    ask=1.17330,
                    timestamp_utc=base_snapshot.fetched_at_utc,
                ),
            },
        )

        decision = _capped_engine(live_enabled=True).validate(
            intent,
            narrow_spread_snapshot,
            for_live_send=True,
        )

        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("TECHNICAL_HARVEST_NEGATIVE_BUCKET_FOR_LIVE", codes)
        self.assertEqual(
            intent.metadata["technical_harvest_precision_negative"]["name"],
            "EUR_USD_DOWN_M5_EMA_SLOPE5_OPPOSED_TP5_SL4",
        )

    def test_bidask_replay_rank_only_and_negative_bucket_apply_at_live_send(self) -> None:
        support_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17330,
            tp=1.17280,
            sl=1.17370,
            thesis="S5 bid/ask replay-backed EUR_USD short harvest",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="selected side matches a positive S5 bid/ask replay segment",
                chart_story="retest below resistance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="4 pip stop",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "chart_direction_bias": "SHORT",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                },
            },
        )
        base_snapshot = snapshot()
        narrow_spread_snapshot = replace(
            base_snapshot,
            quotes={
                **base_snapshot.quotes,
                "EUR_USD": Quote(
                    "EUR_USD",
                    bid=1.17326,
                    ask=1.17330,
                    timestamp_utc=base_snapshot.fetched_at_utc,
                ),
            },
        )

        support_decision = _capped_engine(live_enabled=True).validate(
            support_intent,
            narrow_spread_snapshot,
            for_live_send=True,
        )

        support_codes = {issue.code for issue in support_decision.issues}
        self.assertFalse(support_decision.allowed)
        self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", support_codes)
        self.assertNotIn("bidask_replay_precision_live_ready", support_intent.metadata)
        self.assertNotIn("bidask_replay_precision_support", support_intent.metadata)

        block_intent = OrderIntent(
            pair="AUD_JPY",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=114.289,
            tp=114.338,
            sl=114.250,
            thesis="AUD_JPY high-confidence long must not replay the losing bucket",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="forecast points up",
                chart_story="S5 replay says this pair-direction lost after spread",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.87,
                "chart_direction_bias": "LONG",
            },
        )

        aud_snapshot = snapshot()
        aud_snapshot = replace(
            aud_snapshot,
            quotes={
                **aud_snapshot.quotes,
                "AUD_JPY": Quote(
                    "AUD_JPY",
                    bid=114.286,
                    ask=114.289,
                    timestamp_utc=aud_snapshot.fetched_at_utc,
                ),
            },
        )
        block_decision = _capped_engine(live_enabled=True).validate(
            block_intent,
            aud_snapshot,
            for_live_send=True,
        )

        block_codes = {issue.code for issue in block_decision.issues}
        self.assertFalse(block_decision.allowed)
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", block_codes)
        self.assertEqual(
            block_intent.metadata["bidask_replay_precision_negative"]["name"],
            "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        )

        contrarian_intent = OrderIntent(
            pair="AUD_JPY",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=114.289,
            tp=114.189,
            sl=114.359,
            thesis="AUD_JPY UP S5 0.75-0.90 bucket is faded only when bid/ask replay supports SHORT",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="forecast UP bucket has audited contrarian S5 bid/ask edge",
                chart_story="wait for retest, then fade the weak UP forecast bucket",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="7 pip stop",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.80,
                "forecast_horizon_min": 60,
                "chart_direction_bias": "LONG",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                },
            },
        )

        contrarian_decision = _capped_engine(live_enabled=True).validate(
            contrarian_intent,
            aud_snapshot,
            for_live_send=True,
        )

        contrarian_codes = {issue.code for issue in contrarian_decision.issues}
        self.assertFalse(contrarian_decision.allowed)
        self.assertIn("FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE", contrarian_codes)
        self.assertNotIn("bidask_replay_precision_live_ready", contrarian_intent.metadata)
        self.assertNotIn("bidask_replay_precision_support", contrarian_intent.metadata)

    def test_technical_harvest_rotation_does_not_clear_low_confidence_live_send(self) -> None:
        intent = OrderIntent(
            pair="GBP_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.27330,
            tp=1.27280,
            sl=1.27370,
            thesis="audited short rotation bucket still needs forecast live support",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="M5 momentum bucket is high-rotation evidence but not a live exception",
                chart_story="M5 Bollinger momentum and hot M5 ATR",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="4 pip stop",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "chart_direction_bias": "SHORT",
                "m5_bb_pct_b": 0.40,
                "m5_atr_percentile_100": 0.80,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                },
            },
        )

        base_snapshot = snapshot()
        narrow_spread_snapshot = replace(
            base_snapshot,
            quotes={
                **base_snapshot.quotes,
                "GBP_USD": Quote(
                    "GBP_USD",
                    bid=1.27326,
                    ask=1.27330,
                    timestamp_utc=base_snapshot.fetched_at_utc,
                ),
            },
        )

        decision = _capped_engine(live_enabled=True).validate(
            intent,
            narrow_spread_snapshot,
            for_live_send=True,
        )

        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", codes)
        self.assertNotIn("TECHNICAL_HARVEST_NEGATIVE_BUCKET_FOR_LIVE", codes)
        self.assertNotIn("technical_harvest_precision_live_ready", intent.metadata)

    def test_low_confidence_opposite_forecast_does_not_become_direction_veto(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17100,
            sl=1.17500,
            thesis="short_retest_after_weak_up_forecast",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="failed upside break rejects at resistance",
                chart_story="failed break retest with seller response",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.30,
                "forecast_raw_confidence": 0.52,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.28,
                "forecast_directional_samples": 100,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                },
            },
        )

        dry_run = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=False)
        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        dry_codes = {issue.code for issue in dry_run.issues}
        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertTrue(dry_run.allowed, dry_run.block_reasons)
        self.assertNotIn("FORECAST_DIRECTION_CONFLICT", dry_codes)
        self.assertNotIn("FORECAST_DIRECTION_CONFLICT", live_codes)
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE"], "BLOCK")

    def test_unsupported_weak_directional_bucket_does_not_become_direction_veto(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17100,
            sl=1.17500,
            thesis="short_retest_after_bad_up_bucket",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="failed upside break rejects at resistance",
                chart_story="failed break retest with seller response",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_raw_confidence": 0.91,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.10,
                "forecast_directional_samples": 30,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                },
            },
        )

        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertNotIn("FORECAST_DIRECTION_CONFLICT", live_codes)
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE"], "BLOCK")

    def test_adverse_path_directional_bucket_blocks_same_side_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="long_forecast_hits_invalidation_first_too_often",
            market_context=MarketContext(
                regime="TREND_CONTINUATION pullback",
                narrative="upside continuation after pullback",
                chart_story="higher low holds with buyers returning",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="support shelf breaks on M5 close",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_raw_confidence": 0.91,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.72,
                "forecast_directional_samples": 20,
                "forecast_directional_invalidation_first_rate": 0.75,
                "forecast_directional_invalidation_first_count": 15,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                },
            },
        )

        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTIONAL_INVALIDATION_FIRST_FOR_LIVE"], "BLOCK")

    def test_adverse_path_directional_bucket_does_not_become_direction_veto(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17100,
            sl=1.17500,
            thesis="short_retest_after_adverse_path_up_bucket",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="failed upside break rejects at resistance",
                chart_story="failed break retest with seller response",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_raw_confidence": 0.91,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.72,
                "forecast_directional_samples": 20,
                "forecast_directional_invalidation_first_rate": 0.75,
                "forecast_directional_invalidation_first_count": 15,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                },
            },
        )

        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertNotIn("FORECAST_DIRECTION_CONFLICT", live_codes)
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTIONAL_INVALIDATION_FIRST_FOR_LIVE"], "BLOCK")

    def test_supported_low_confidence_opposite_forecast_still_blocks_this_side(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17100,
            sl=1.17500,
            thesis="do_not_short_supported_up_forecast",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="failed upside break is visible but projection favors continuation",
                chart_story="failed break retest has not overruled audited projection support",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance recaptures on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.58,
                "forecast_raw_confidence": 0.60,
                "chart_direction_bias": "LONG",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "best_hit_rate": 1.0,
                    "best_samples": 40,
                    "reason": "news_theme_followthrough UP supports weak calibrated forecast",
                    "signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "UP",
                            "confidence": 0.8,
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                        }
                    ],
                },
            },
        )

        live = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        live_codes = {issue.code: issue.severity for issue in live.issues}
        self.assertFalse(live.allowed)
        self.assertEqual(live_codes["FORECAST_DIRECTION_CONFLICT"], "BLOCK")

    def test_range_method_rejects_one_way_trend_story_for_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="eurusd_bad_method",
            market_context=MarketContext(
                regime="TREND-BULL impulse",
                narrative="USD softness is driving continuation",
                chart_story="one-way trend extension with no two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="1.1716 loses on bodies",
            ),
        )
        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)
        self.assertFalse(decision.allowed)
        self.assertIn("METHOD_REGIME_MISMATCH", {issue.code for issue in decision.issues})

    def test_range_forecast_blocks_non_rotation_method_for_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="range_forecast_must_not_authorize_trend_entry",
            market_context=MarketContext(
                regime="RANGE current; trend lane still visible",
                narrative="forecast says two-way box, not one-way continuation",
                chart_story="range box with no breakout confirmation",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="box low fails",
            ),
            metadata={"forecast_direction": "RANGE", "forecast_confidence": 0.72},
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("RANGE_FORECAST_REQUIRES_RANGE_ROTATION", {issue.code for issue in decision.issues})

    def test_range_forecast_blocks_opposite_unselected_projection_for_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17500,
            sl=1.17230,
            thesis="do_not_buy_range_when_audited_projection_points_down",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="box rotation but news projection points lower",
                chart_story="range rail geometry with two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "forecast_direction": "RANGE",
                "forecast_market_support": {
                    "unselected_signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                        }
                    ]
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn(
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
            {issue.code for issue in decision.issues},
        )

    def test_range_forecast_ignores_headline_only_opposite_unselected_projection_for_live_send(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17500,
            sl=1.17230,
            thesis="do_not_let_timeout_drag_signal_veto_range_side",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="headline projection looks perfect but times out too often",
                chart_story="range rail geometry with two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "forecast_direction": "RANGE",
                "forecast_market_support": {
                    "unselected_signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "economic_hit_rate": 0.40,
                            "economic_samples": 100,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "timeout_rate": 0.60,
                        }
                    ]
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertNotIn(
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
            {issue.code for issue in decision.issues},
        )

    def test_range_forecast_allows_same_side_unselected_projection_warning_free(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17400,
            tp=1.17130,
            sl=1.17470,
            thesis="sell_range_when_audited_projection_points_down",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="box rotation with downside projection support",
                chart_story="range rail geometry with two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.43,
                "forecast_horizon_min": 120,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "forecast_market_support": {
                    "unselected_signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "confidence": 0.8,
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "lead_time_min": 20.0,
                        }
                    ]
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn(
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
            {issue.code for issue in decision.issues},
        )

    def test_forecast_watch_role_blocks_live_send_without_gateway_verified_override(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17400,
            tp=1.17130,
            sl=1.17470,
            thesis="do_not_send_watch_role_without_override",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="watch-only lane has review geometry but no executable override",
                chart_story="range rail geometry with two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "campaign_role": "FORECAST_WATCH",
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.52,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_WATCH_ONLY", {issue.code for issue in decision.issues})

    def test_forecast_watch_range_rail_override_remains_live_sendable(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17400,
            tp=1.17130,
            sl=1.17470,
            thesis="range_rail_watch_override_is_audited_non_market_entry",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="measured range rail override keeps passive entry executable",
                chart_story="range rail geometry with two-way structure",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "campaign_role": "FORECAST_WATCH",
                "forecast_watch_only": True,
                "forecast_watch_only_live_override": True,
                "forecast_watch_only_live_override_reason": "range rail override",
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.52,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("FORECAST_WATCH_ONLY", {issue.code for issue in decision.issues})

    def test_forecast_watch_override_does_not_allow_market_conversion(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17130,
            sl=1.17430,
            thesis="do_not_convert_watch_override_to_market",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="watch override must remain a passive rail entry",
                chart_story="range rail geometry exists but market order chases",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "campaign_role": "FORECAST_WATCH",
                "forecast_watch_only": True,
                "forecast_watch_only_live_override": True,
                "forecast_watch_only_live_override_reason": "range rail override",
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.52,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_WATCH_ONLY", {issue.code for issue in decision.issues})

    def test_range_forecast_same_side_unselected_projection_does_not_authorize_market(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17130,
            sl=1.17430,
            thesis="do_not_market_sell_weak_range_forecast",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="same-side projection is only a passive rail fade aid",
                chart_story="range rail geometry exists but the order chases now",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.43,
                "forecast_horizon_min": 120,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "forecast_market_support": {
                    "unselected_signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "confidence": 0.8,
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "lead_time_min": 20.0,
                        }
                    ]
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", {issue.code for issue in decision.issues})

    def test_directional_forecast_blocks_opposite_unselected_projection_without_selected_support(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17130,
            sl=1.17430,
            thesis="do_not_sell_when_opposite_projection_has_better_audit",
            market_context=MarketContext(
                regime="TREND_DOWN continuation candidate",
                narrative="selected forecast is down but the audited sweep model points up",
                chart_story="breakdown attempt near lower liquidity",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="reclaims range low",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.72,
                "forecast_raw_confidence": 0.78,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                    "unselected_signals": [
                        {
                            "name": "liquidity_sweep_low",
                            "direction": "UP",
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "target_pips": 6.0,
                        }
                    ],
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn(
            "FORECAST_UNSELECTED_OPPOSITE_PROJECTION",
            {issue.code for issue in decision.issues},
        )

    def test_directional_forecast_ignores_headline_only_opposite_unselected_projection(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17130,
            sl=1.17430,
            thesis="do_not_let_timeout_drag_projection_veto_selected_side",
            market_context=MarketContext(
                regime="TREND_DOWN continuation candidate",
                narrative="opposite sweep bucket times out too often to veto the entry",
                chart_story="breakdown attempt near lower liquidity",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="reclaims range low",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.72,
                "forecast_raw_confidence": 0.78,
                "forecast_directional_calibration_name": "directional_forecast_down",
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 40,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "DOWN",
                    "aligned_projection_count": 0,
                    "unselected_signals": [
                        {
                            "name": "liquidity_sweep_low",
                            "direction": "UP",
                            "economic_hit_rate": 0.40,
                            "economic_samples": 100,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "target_pips": 6.0,
                            "timeout_rate": 0.60,
                        }
                    ],
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertNotIn(
            "FORECAST_UNSELECTED_OPPOSITE_PROJECTION",
            {issue.code for issue in decision.issues},
        )

    def test_directional_forecast_allows_opposite_unselected_projection_when_selected_support_clears(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17130,
            sl=1.17430,
            thesis="sell_when_selected_forecast_has_current_audited_support",
            market_context=MarketContext(
                regime="TREND_DOWN continuation candidate",
                narrative="selected down forecast has current projection support",
                chart_story="breakdown retest with aligned sweep continuation",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="reclaims range low",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.72,
                "forecast_raw_confidence": 0.78,
                "forecast_market_support": {
                    "ok": True,
                    "direction": "DOWN",
                    "aligned_projection_count": 1,
                    "best_hit_rate": 1.0,
                    "best_samples": 40,
                    "signals": [
                        {
                            "name": "macro_event_nowcast_central_bank",
                            "direction": "DOWN",
                            "confidence": 0.9,
                            "hit_rate": 1.0,
                            "samples": 40,
                        }
                    ],
                    "unselected_signals": [
                        {
                            "name": "liquidity_sweep_low",
                            "direction": "UP",
                            "hit_rate": 0.69,
                            "samples": 100,
                        }
                    ],
                },
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn(
            "FORECAST_UNSELECTED_OPPOSITE_PROJECTION",
            {issue.code for issue in decision.issues},
        )

    def test_risk_cap_blocks_oversized_trade(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=8000,
            tp=1.17554,
            sl=1.17234,
            thesis="oversized_rebuild_regression",
        )
        decision = _capped_engine().validate(intent, snapshot())
        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_CAP_EXCEEDED", {issue.code for issue in decision.issues})

    def test_risk_cap_still_blocks_oversized_trade_under_sl_free(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=8000,
            tp=1.17554,
            sl=1.17234,
            thesis="sl_free_loss_cap_regression",
        )

        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            decision = _capped_engine().validate(intent, snapshot())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        self.assertFalse(decision.allowed)
        self.assertIn("LOSS_CAP_EXCEEDED", {issue.code for issue in decision.issues})

    def test_margin_cap_blocks_trade_before_broker_rejects_it(self) -> None:
        now = datetime.now(timezone.utc)
        snap = BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17338, ask=1.17346, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=156.410, ask=156.418, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=220_145.7765,
                balance_jpy=208_945.7765,
                margin_used_jpy=156_414.0,
                margin_available_jpy=63_831.7765,
                fetched_at_utc=now,
            ),
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=13_000,
            tp=1.17430,
            sl=1.17274,
            thesis="eurusd_must_fit_margin_before_send",
        )

        decision = _capped_engine().validate(intent, snap)

        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("MARGIN_UTILIZATION_CAP_EXCEEDED", codes)
        self.assertIn("MARGIN_AVAILABLE_EXCEEDED", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertGreater(decision.metrics.margin_utilization_after_pct or 0.0, 92.0)

    def test_margin_cap_allows_trade_inside_92_percent_budget(self) -> None:
        now = datetime.now(timezone.utc)
        snap = BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17338, ask=1.17346, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=156.410, ask=156.418, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=220_145.7765,
                balance_jpy=208_945.7765,
                margin_used_jpy=156_414.0,
                margin_available_jpy=63_831.7765,
                fetched_at_utc=now,
            ),
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=6_000,
            tp=1.17430,
            sl=1.17274,
            thesis="eurusd_can_use_margin_up_to_92_percent",
            metadata={"max_loss_jpy": 1_000.0},
        )

        decision = _capped_engine().validate(intent, snap)

        codes = {issue.code for issue in decision.issues}
        self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", codes)
        self.assertNotIn("MARGIN_AVAILABLE_EXCEEDED", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertLessEqual(decision.metrics.margin_utilization_after_pct or 100.0, 92.0)

    def test_external_position_blocks_fresh_entries(self) -> None:
        external = BrokerPosition(
            trade_id="470012",
            pair="USD_JPY",
            side=Side.LONG,
            units=20000,
            entry_price=156.836,
            owner=Owner.EXTERNAL,
            take_profit=None,
            stop_loss=None,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="must_not_trade_while_external_risk_open",
        )
        decision = _capped_engine().validate(intent, snapshot(positions=(external,)))
        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("EXTERNAL_RISK_OPEN", codes)
        self.assertIn("UNPROTECTED_POSITION", codes)

    def test_operator_manual_position_does_not_block_fresh_entries(self) -> None:
        manual = BrokerPosition(
            trade_id="470201",
            pair="USD_JPY",
            side=Side.LONG,
            units=25000,
            entry_price=155.962,
            owner=Owner.UNKNOWN,
            take_profit=None,
            stop_loss=None,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="manual_usdjpy_is_operator_managed_parallel_exposure",
        )
        decision = _capped_engine().validate(intent, snapshot(positions=(manual,)))
        codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("EXTERNAL_RISK_OPEN", codes)
        self.assertNotIn("UNPROTECTED_POSITION", codes)
        self.assertNotIn("OPEN_POSITION_EXISTS", codes)

    def test_trader_position_without_tp_or_sl_blocks_fresh_entries(self) -> None:
        unprotected = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="USD_JPY",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=2000,
            entry=156.645,
            tp=156.445,
            sl=156.789,
            thesis="fresh_entry_must_wait_for_protection",
        )
        decision = _capped_engine().validate(intent, snapshot(positions=(unprotected,)))
        self.assertFalse(decision.allowed)
        self.assertIn("UNPROTECTED_POSITION", {issue.code for issue in decision.issues})

    def test_sl_free_no_broker_tp_runner_does_not_freeze_portfolio_adds(self) -> None:
        runner = BrokerPosition(
            trade_id="471232",
            pair="EUR_USD",
            side=Side.LONG,
            units=7000,
            entry_price=1.16768,
            owner=Owner.TRADER,
            take_profit=None,
            stop_loss=None,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=3000,
            tp=1.17554,
            sl=1.17234,
            thesis="sl_free_runner_must_not_starve_next_entry",
        )

        from quant_rabbit.risk import RiskPolicy

        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
        try:
            decision = _capped_engine(
                policy=RiskPolicy(
                    allow_protected_trader_position_adds=True,
                    max_portfolio_loss_jpy=10_000.0,
                )
            ).validate(intent, snapshot(positions=(runner,)))
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_tp is None:
                os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            else:
                os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

        codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPEN_POSITION_EXISTS", codes)
        self.assertNotIn("UNPROTECTED_POSITION", codes)
        self.assertIn("TP_LESS_RUNNER_OPEN", codes)

    def test_protected_trader_position_still_blocks_fresh_entries(self) -> None:
        protected = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1760,
            stop_loss=1.1680,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="USD_JPY",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=2000,
            entry=156.645,
            tp=156.445,
            sl=156.789,
            thesis="fresh_entry_must_not_stack_on_protected_position",
        )
        decision = _capped_engine().validate(intent, snapshot(positions=(protected,)))
        self.assertFalse(decision.allowed)
        self.assertIn("OPEN_POSITION_EXISTS", {issue.code for issue in decision.issues})

    def test_portfolio_policy_allows_protected_trader_add_within_budget(self) -> None:
        protected_break_even = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1760,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="portfolio_add_must_stay_inside_budget",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="protected runner plus fresh continuation trigger",
                chart_story="trend continuation after break-even protection",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_break_even,)))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPEN_POSITION_EXISTS", {issue.code for issue in decision.issues})

    def test_same_pair_position_cap_blocks_fresh_stack(self) -> None:
        first = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1760,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        second = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1710,
            take_profit=1.1760,
            stop_loss=1.1710,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="same_pair_stack_must_not_monopolize_basket",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="third EUR_USD add would consume opportunity slots",
                chart_story="trend continuation trigger",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
            metadata={"position_intent": "PYRAMID", "position_fill": "OPEN_ONLY"},
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(first, second)))

        self.assertFalse(decision.allowed)
        self.assertIn("PAIR_CONCENTRATION_LIMIT", {issue.code for issue in decision.issues})

    def test_bounded_adverse_add_can_use_manual_replay_third_slot(self) -> None:
        first = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        second = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="manual_replay_bounded_nanpin_third_slot",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate averages a small adverse retest inside current ATR",
                chart_story="lower-third rejection after pullback",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
                "tp_atr_pips": 4.0,
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(first, second)))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("PAIR_CONCENTRATION_LIMIT", {issue.code for issue in decision.issues})
        self.assertNotIn("ADVERSE_ADD_DISTANCE_TOO_WIDE", {issue.code for issue in decision.issues})

    def test_bounded_adverse_add_blocks_past_manual_replay_entry_cap(self) -> None:
        positions = tuple(
            BrokerPosition(
                trade_id=str(idx),
                pair="EUR_USD",
                side=Side.LONG,
                units=1000,
                entry_price=1.1740,
                take_profit=1.1760,
                stop_loss=1.1720,
                owner=Owner.TRADER,
            )
            for idx in range(1, 5)
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="manual_replay_bounded_nanpin_fifth_slot_must_wait",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate would exceed the replayable manual averaging entry count",
                chart_story="lower-third rejection after pullback",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
                "tp_atr_pips": 4.0,
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_positions=10,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=positions))

        self.assertFalse(decision.allowed)
        self.assertIn("PAIR_CONCENTRATION_LIMIT", {issue.code for issue in decision.issues})
        self.assertTrue(
            any("bounded adverse-add cap 4" in issue.message for issue in decision.issues),
            decision.block_reasons,
        )

    def test_adverse_same_pair_add_requires_position_building_classification(self) -> None:
        existing_long = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="adverse_same_pair_add_must_be_explicitly_classified",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate is a same-side retest add",
                chart_story="lower-half rejection with current spread contained",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={"position_intent": "PYRAMID", "position_fill": "OPEN_ONLY"},
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_same_pair_trader_positions=None,
                max_same_pair_margin_utilization_pct=None,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(existing_long,)))

        self.assertFalse(decision.allowed)
        self.assertIn("ADVERSE_ADD_CLASSIFICATION_MISSING", {issue.code for issue in decision.issues})

    def test_adverse_same_pair_add_requires_current_atr_metadata(self) -> None:
        existing_long = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="adverse_same_pair_add_needs_market_volatility_bound",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate is a same-side retest add",
                chart_story="lower-half rejection with current spread contained",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_same_pair_trader_positions=None,
                max_same_pair_margin_utilization_pct=None,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(existing_long,)))

        self.assertFalse(decision.allowed)
        self.assertIn("ADVERSE_ADD_ATR_MISSING", {issue.code for issue in decision.issues})

    def test_adverse_same_pair_add_blocks_when_distance_exceeds_current_atr_cap(self) -> None:
        existing_long = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="adverse_same_pair_add_must_stay_inside_current_atr_cap",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate is a same-side retest add",
                chart_story="lower-half rejection with current spread contained",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
                "tp_atr_pips": 3.0,
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_same_pair_trader_positions=None,
                max_same_pair_margin_utilization_pct=None,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(existing_long,)))

        self.assertFalse(decision.allowed)
        self.assertIn("ADVERSE_ADD_DISTANCE_TOO_WIDE", {issue.code for issue in decision.issues})

    def test_adverse_same_pair_add_allows_market_derived_bounded_retest(self) -> None:
        existing_long = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1740,
            take_profit=1.1760,
            stop_loss=1.1720,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="adverse_same_pair_add_is_current_atr_bounded",
            market_context=MarketContext(
                regime="RANGE_ROTATION bounded retest",
                narrative="candidate is a same-side retest add",
                chart_story="lower-half rejection with current spread contained",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
                "tp_atr_pips": 4.0,
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_same_pair_trader_positions=None,
                max_same_pair_margin_utilization_pct=None,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(existing_long,)))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("ADVERSE_ADD_DISTANCE_TOO_WIDE", {issue.code for issue in decision.issues})

    def test_same_pair_position_cap_does_not_block_explicit_hedge(self) -> None:
        first = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1640,
            stop_loss=1.1710,
            owner=Owner.TRADER,
        )
        second = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1690,
            take_profit=1.1640,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.1710,
            tp=1.1730,
            sl=1.1702,
            thesis="explicit_hedge_can_reduce_trapped_side_without_new_pair_stack",
            market_context=MarketContext(
                regime="RANGE_ROTATION campaign lane",
                narrative="hedge against trapped short exposure",
                chart_story="box rail reclaim into midpoint",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_timing_class": "OPPOSITE_EXPOSURE",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "next_m15_close_or_structure_change",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_same_pair_trader_positions=1,
                max_same_pair_margin_utilization_pct=1.0,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(intent, snapshot(positions=(first, second), hedging_enabled=True))

        issue_codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("PAIR_CONCENTRATION_LIMIT", issue_codes)
        self.assertNotIn("PAIR_MARGIN_CONCENTRATION_LIMIT", issue_codes)

    def test_same_pair_margin_cap_blocks_fresh_stack_before_portfolio_margin_is_full(self) -> None:
        large_same_pair = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=12_000,
            entry_price=1.1700,
            take_profit=1.1760,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="same_pair_margin_must_leave_room_for_other_pairs",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="fresh add would push EUR_USD over pair margin reserve",
                chart_story="trend continuation trigger",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
            metadata={"position_intent": "PYRAMID", "position_fill": "OPEN_ONLY"},
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=50_000.0,
            )
        ).validate(
            intent,
            snapshot(
                positions=(large_same_pair,),
                nav_jpy=200_000.0,
                margin_used_jpy=70_000.0,
                margin_available_jpy=130_000.0,
            ),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("PAIR_MARGIN_CONCENTRATION_LIMIT", {issue.code for issue in decision.issues})

    def test_portfolio_policy_blocks_opposing_same_pair_entry_without_hedging_proof(self) -> None:
        protected_short = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1640,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="opposing_entry_must_route_to_position_management",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE campaign lane",
                narrative="fresh long would oppose the protected short",
                chart_story="failed break reclaim",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,)))

        self.assertFalse(decision.allowed)
        self.assertIn("OPPOSING_POSITION_NEEDS_HEDGING", {issue.code for issue in decision.issues})

    def test_portfolio_policy_blocks_opposing_same_pair_entry_without_explicit_hedge_intent(self) -> None:
        protected_short = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1640,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.1710,
            tp=1.1730,
            sl=1.1702,
            thesis="range_reclaim_without_hedge_declaration",
            market_context=MarketContext(
                regime="RANGE_ROTATION campaign lane",
                narrative="lower-rail rotation while existing short remains protected",
                chart_story="box rail reclaim into midpoint",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,), hedging_enabled=True))

        self.assertFalse(decision.allowed)
        self.assertIn("OPPOSING_POSITION_NEEDS_HEDGING", {issue.code for issue in decision.issues})

    def test_portfolio_policy_allows_opposing_same_pair_hedge_when_account_hedging_enabled(self) -> None:
        protected_short = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1640,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.1710,
            tp=1.1730,
            sl=1.1702,
            thesis="intraday_range_hedge_against_swing_short",
            market_context=MarketContext(
                regime="RANGE_ROTATION campaign lane",
                narrative="M5 lower-rail rotation while existing short remains protected on slower thesis",
                chart_story="box rail reclaim into midpoint",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_timing_class": "OPPOSITE_EXPOSURE",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "next_m15_close_or_structure_change",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,), hedging_enabled=True))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPPOSING_POSITION_NEEDS_HEDGING", {issue.code for issue in decision.issues})

    def test_same_pair_hedge_requires_timing_metadata(self) -> None:
        protected_short = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.SHORT,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1640,
            stop_loss=1.1700,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.1710,
            tp=1.1730,
            sl=1.1702,
            thesis="intraday_range_hedge_against_swing_short",
            market_context=MarketContext(
                regime="RANGE_ROTATION campaign lane",
                narrative="M5 lower-rail rotation while existing short remains protected on slower thesis",
                chart_story="box rail reclaim into midpoint",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata={"position_intent": "HEDGE", "position_fill": "OPEN_ONLY"},
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,), hedging_enabled=True))

        issue_codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("HEDGE_TIMING_METADATA_MISSING", issue_codes)
        self.assertIn("HEDGE_UNWIND_PLAN_MISSING", issue_codes)
        self.assertIn("HEDGE_REVIEW_TRIGGER_MISSING", issue_codes)

    def test_same_pair_hedge_uses_longest_leg_margin_when_account_is_over_policy_cap(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            take_profit=1.17100,
            stop_loss=1.16600,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.16561,
            tp=1.16383,
            sl=1.16641,
            thesis="same_pair_short_hedge_inside_existing_long_leg",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE reject/retest current",
                narrative="short hedge against trapped long exposure",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_timing_class": "OPPOSITE_EXPOSURE",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "next_m15_close_or_structure_change",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(
            intent,
            snapshot(
                positions=(protected_long,),
                hedging_enabled=True,
                nav_jpy=175_988.7367,
                balance_jpy=192_275.8359,
                margin_used_jpy=162_740.16,
                margin_available_jpy=13_436.9823,
            ),
        )

        issue_codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertEqual(decision.metrics.estimated_margin_jpy, 0.0)
        self.assertNotIn("MARGIN_UTILIZATION_CAP_REACHED", issue_codes)
        self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", issue_codes)

    def test_recovery_hedge_uses_range_reward_floor_instead_of_fresh_entry_floor(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=1.17100,
            stop_loss=1.16600,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.17220,
            tp=1.17140,
            sl=1.17320,
            thesis="recovery hedge monetizes move against trapped long",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE recovery hedge",
                narrative="short hedge against trapped long exposure",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(intent, snapshot(positions=(protected_long,), hedging_enabled=True))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertGreaterEqual(decision.metrics.reward_risk, 0.6)
        self.assertLess(decision.metrics.reward_risk, 1.2)
        self.assertNotIn("REWARD_RISK_TOO_LOW", {issue.code for issue in decision.issues})

    def test_continuation_recovery_hedge_requires_capped_scale(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=1.17100,
            stop_loss=1.16600,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.17220,
            tp=1.17140,
            sl=1.17320,
            thesis="oversized continuation hedge",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE recovery hedge",
                narrative="short hedge against trapped long exposure",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_recovery": True,
                "hedge_timing_class": "CONTINUATION",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "next_m15_close_or_failed_break_trigger",
                "hedge_recovery_size_scale": 0.50,
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(intent, snapshot(positions=(protected_long,), hedging_enabled=True))

        self.assertFalse(decision.allowed)
        self.assertIn("HEDGE_CONTINUATION_SIZE_TOO_LARGE", {issue.code for issue in decision.issues})

    def test_same_pair_hedge_blocks_when_opposite_leg_already_covered(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=1.17100,
            stop_loss=1.16600,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="3",
            pair="EUR_USD",
            side=Side.SHORT,
            units=22_000,
            entry_price=1.17000,
            unrealized_pl_jpy=900.0,
            take_profit=1.16400,
            stop_loss=1.17300,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.17220,
            tp=1.17140,
            sl=1.17320,
            thesis="covered hedge must not become hidden net short add",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE recovery hedge",
                narrative="short hedge against trapped long exposure",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(intent, snapshot(positions=(protected_long, existing_short), hedging_enabled=True))

        self.assertFalse(decision.allowed)
        self.assertIn("HEDGE_REFERENCE_ALREADY_COVERED", {issue.code for issue in decision.issues})

    def test_same_pair_hedge_blocks_units_beyond_uncovered_opposite_leg(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=1.17100,
            stop_loss=1.16600,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="3",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_000,
            entry_price=1.17000,
            unrealized_pl_jpy=300.0,
            take_profit=1.16400,
            stop_loss=1.17300,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=18_000,
            entry=1.17220,
            tp=1.17140,
            sl=1.17320,
            thesis="hedge size must stop at uncovered opposite exposure",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE recovery hedge",
                narrative="short hedge against trapped long exposure",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(intent, snapshot(positions=(protected_long, existing_short), hedging_enabled=True))

        self.assertFalse(decision.allowed)
        self.assertIn("HEDGE_UNITS_EXCEED_OPPOSITE_EXPOSURE", {issue.code for issue in decision.issues})

    def test_same_pair_hedge_ignores_manual_long_when_capping_bot_short(self) -> None:
        protected_long = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=None,
            stop_loss=None,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1000.0,
            take_profit=None,
            stop_loss=None,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="3",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_700,
            entry_price=1.16013,
            unrealized_pl_jpy=-200.0,
            take_profit=1.15830,
            stop_loss=None,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=2_700,
            entry=1.16013,
            tp=1.15830,
            sl=1.16317,
            thesis="manual exposure must not authorize over-hedged bot short",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE recovery hedge",
                narrative="short hedge against trapped trader long only",
                chart_story="failed reclaim and rejection below trigger",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "HEDGE",
                "position_fill": "OPEN_ONLY",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "hedge_unwind_plan_required": True,
                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
            },
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_loss_jpy=5_000.0,
                max_portfolio_loss_jpy=50_000.0,
                max_margin_utilization_pct=92.0,
            )
        ).validate(
            intent,
            snapshot(
                positions=(protected_long, manual_long, existing_short),
                hedging_enabled=True,
                nav_jpy=181_000.0,
                margin_used_jpy=162_500.0,
                margin_available_jpy=18_000.0,
            ),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("HEDGE_UNITS_EXCEED_OPPOSITE_EXPOSURE", {issue.code for issue in decision.issues})

    def test_net_short_pyramid_uses_manual_long_for_broker_margin_offset(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        trader_long = BrokerPosition(
            trade_id="long",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            take_profit=None,
            stop_loss=None,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1000.0,
            take_profit=None,
            stop_loss=None,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="short",
            pair="EUR_USD",
            side=Side.SHORT,
            units=8_400,
            entry_price=1.17413,
            unrealized_pl_jpy=200.0,
            take_profit=1.15830,
            stop_loss=None,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=3_000,
            entry=1.17322,
            tp=1.17170,
            sl=1.17410,
            thesis="net short pyramid uses broker margin offset from account-wide long leg",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE reject/retest current",
                narrative="short entry aligned with current failure-risk setup",
                chart_story="upper retest rejection",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="SL trades",
            ),
            metadata={
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "PYRAMID_WITH_MOVE",
            },
        )

        try:
            from quant_rabbit.risk import RiskPolicy

            decision = _capped_engine(
                policy=RiskPolicy(
                    allow_protected_trader_position_adds=True,
                    max_loss_jpy=5_000.0,
                    max_portfolio_loss_jpy=50_000.0,
                    max_margin_utilization_pct=92.0,
                    max_same_pair_trader_positions=None,
                    max_same_pair_margin_utilization_pct=None,
                )
            ).validate(
                intent,
                snapshot(
                    positions=(trader_long, manual_long, existing_short),
                    hedging_enabled=True,
                    nav_jpy=181_000.0,
                    margin_used_jpy=166_000.0,
                    margin_available_jpy=18_000.0,
                ),
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

        issue_codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPPOSING_POSITION_NEEDS_HEDGING", issue_codes)
        self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", issue_codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertEqual(decision.metrics.estimated_margin_jpy, 0.0)

    def test_portfolio_policy_blocks_add_when_total_loss_budget_exceeded(self) -> None:
        protected_at_risk = BrokerPosition(
            trade_id="2",
            pair="EUR_USD",
            side=Side.LONG,
            units=3000,
            entry_price=1.1700,
            take_profit=1.1760,
            stop_loss=1.1690,
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.1735,
            tp=1.1750,
            sl=1.1725,
            thesis="portfolio_add_must_not_exceed_budget",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="protected but still risked position",
                chart_story="trend continuation trigger",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
        )

        from quant_rabbit.risk import RiskPolicy

        decision = _capped_engine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_at_risk,)))

        self.assertFalse(decision.allowed)
        self.assertIn("PORTFOLIO_LOSS_CAP_EXCEEDED", {issue.code for issue in decision.issues})

    def test_pending_entry_order_blocks_duplicate_fresh_entries(self) -> None:
        pending = BrokerOrder(
            order_id="123",
            pair="AUD_JPY",
            order_type="STOP",
            trade_id=None,
            price=112.576,
            state="PENDING",
            owner=Owner.TRADER,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.17330,
            tp=1.17450,
            sl=1.17250,
            thesis="must_not_stack_entry_orders",
        )
        decision = _capped_engine().validate(intent, snapshot(orders=(pending,)))
        self.assertFalse(decision.allowed)
        self.assertIn("PENDING_ENTRY_ORDER_OPEN", {issue.code for issue in decision.issues})

    def test_operator_manual_pending_entry_does_not_block_trader_entry(self) -> None:
        pending = BrokerOrder(
            order_id="manual-pending",
            pair="AUD_JPY",
            order_type="STOP",
            trade_id=None,
            price=112.576,
            state="PENDING",
            owner=Owner.UNKNOWN,
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17450,
            sl=1.17250,
            thesis="operator pending order is parallel manual exposure",
        )

        decision = _capped_engine().validate(intent, snapshot(orders=(pending,)))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("PENDING_ENTRY_ORDER_OPEN", {issue.code for issue in decision.issues})

    def test_pending_entry_price_must_be_on_executable_side(self) -> None:
        long_stop_below_market = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=1000,
            entry=1.17300,
            tp=1.17450,
            sl=1.17250,
            thesis="buy_stop_must_not_be_parked_below_current_ask",
        )
        short_limit_below_market = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17300,
            tp=1.17150,
            sl=1.17350,
            thesis="sell_limit_must_not_be_parked_below_current_bid",
        )

        long_decision = _capped_engine().validate(long_stop_below_market, snapshot())
        short_decision = _capped_engine().validate(short_limit_below_market, snapshot())

        self.assertFalse(long_decision.allowed)
        self.assertIn("STOP_ENTRY_NOT_ABOVE_MARKET", {issue.code for issue in long_decision.issues})
        self.assertFalse(short_decision.allowed)
        self.assertIn("LIMIT_ENTRY_NOT_ABOVE_MARKET", {issue.code for issue in short_decision.issues})

    def test_live_market_intent_blocks_stale_expected_entry(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.17000,
            tp=1.17554,
            sl=1.17234,
            thesis="market_order_expected_entry_must_match_fresh_broker_truth",
            market_context=MarketContext(
                regime="TREND-BULL continuation",
                narrative="USD softness lets EUR squeeze higher",
                chart_story="green staircase into upper band with shallow pullbacks",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="1.1716 loses on M5 bodies",
            ),
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        self.assertFalse(decision.allowed)
        self.assertIn("MARKET_ENTRY_DRIFT", {issue.code for issue in decision.issues})

    def test_bad_reward_risk_blocks(self) -> None:
        intent = OrderIntent(
            pair="USD_JPY",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=2000,
            entry=156.645,
            tp=156.545,
            sl=156.789,
            thesis="usd_jpy_low_rr_regression",
        )
        decision = _capped_engine().validate(intent, snapshot())
        self.assertFalse(decision.allowed)
        self.assertIn("REWARD_RISK_TOO_LOW", {issue.code for issue in decision.issues})

    def test_zero_units_keep_geometry_reward_risk_diagnostic(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=0,
            tp=1.17530,
            sl=1.17230,
            thesis="zero_units_should_not_make_rr_zero",
            market_context=MarketContext(
                regime="M5 range rotation",
                narrative="test geometry remains readable when margin sizing returns zero",
                chart_story="M5 range rail reclaim toward box midpoint",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="lower rail loses",
            ),
        )

        decision = _capped_engine().validate(intent, snapshot())

        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("BAD_UNITS", codes)
        self.assertNotIn("REWARD_RISK_TOO_LOW", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertAlmostEqual(decision.metrics.reward_risk, 2.0)

    def test_range_rotation_uses_range_rr_floor_inside_higher_tf_trend(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17410,
            sl=1.17230,
            thesis="tf_local_range_rotation_inside_higher_tf_trend",
            market_context=MarketContext(
                regime="H1 TREND_DOWN; M5 range rotation",
                narrative="local lower-rail reclaim scalp inside higher-TF trend context",
                chart_story="M5 range rail/box rotation, not a one-way impulse chase",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="M5 lower rail loses",
            ),
            metadata={"regime_state": "TREND_DOWN", "geometry_model": "RANGE_RAIL_MARKET"},
        )

        decision = _capped_engine().validate(intent, snapshot())

        codes = {issue.code for issue in decision.issues}
        self.assertNotIn("REWARD_RISK_TOO_LOW", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertGreaterEqual(decision.metrics.reward_risk, 0.6)
        self.assertLess(decision.metrics.reward_risk, 1.2)

    def test_countertrend_range_rotation_below_one_r_blocks_when_matrix_leans_opposite(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17410,
            sl=1.17220,
            thesis="countertrend_range_rotation_needs_one_r",
            market_context=MarketContext(
                regime="TREND_DOWN current; RANGE_ROTATION campaign lane",
                narrative="local lower-rail reclaim but higher-timeframe sellers still dominate",
                chart_story=(
                    "M5 RANGE lower rail reclaim; M15 TREND_DOWN; H1 TREND_DOWN; "
                    "H4 TREND_DOWN; matrix confluence score_balance=SHORT_LEAN"
                ),
                method=TradeMethod.RANGE_ROTATION,
                invalidation="M5 lower rail loses",
            ),
            metadata={
                "regime_state": "TREND_DOWN",
                "geometry_model": "RANGE_RAIL_LIMIT",
                "strongest_matrix_reject": "EUR_USD confluence score_balance=SHORT_LEAN",
                "strongest_matrix_warning": "EUR_USD dominant_regime=TREND_DOWN",
            },
        )

        decision = _capped_engine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("RANGE_COUNTERTREND_RR_TOO_LOW", codes)
        self.assertNotIn("REWARD_RISK_TOO_LOW", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertGreaterEqual(decision.metrics.reward_risk, 0.6)
        self.assertLess(decision.metrics.reward_risk, 1.0)

    def test_failed_break_technical_harvest_uses_one_r_floor(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17431,
            sl=1.17230,
            thesis="failed_break_harvest_1r_should_not_use_runner_floor",
            market_context=MarketContext(
                regime="M5 failed break retest",
                narrative="short-cycle failed-break harvest",
                chart_story="retest rejection with nearby structural harvest TP",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="retest low loses",
            ),
            metadata={
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_target_source": "OPERATING_HARVEST_FLOOR",
            },
        )

        decision = _capped_engine().validate(intent, snapshot())

        codes = {issue.code for issue in decision.issues}
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("REWARD_RISK_TOO_LOW", codes)
        self.assertIn("TECHNICAL_HARVEST_REWARD_RISK_FLOOR", codes)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertGreaterEqual(decision.metrics.reward_risk, 1.0)
        self.assertLess(decision.metrics.reward_risk, 1.2)

    def test_failed_break_technical_harvest_below_one_r_still_blocks(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17410,
            sl=1.17230,
            thesis="failed_break_harvest_below_1r_still_bad_geometry",
            market_context=MarketContext(
                regime="M5 failed break retest",
                narrative="short-cycle failed-break harvest",
                chart_story="retest rejection but TP too close for stop distance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="retest low loses",
            ),
            metadata={
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_target_source": "OPERATING_HARVEST_FLOOR",
            },
        )

        decision = _capped_engine().validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        self.assertIn("REWARD_RISK_TOO_LOW", {issue.code for issue in decision.issues})

    def test_trend_continuation_does_not_get_technical_harvest_rr_floor(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17431,
            sl=1.17230,
            thesis="trend_continuation_must_not_use_harvest_floor",
            market_context=MarketContext(
                regime="H1 trend continuation",
                narrative="runner thesis must clear runner floor",
                chart_story="trend continuation, not a failed-break harvest",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="trend structure fails",
            ),
            metadata={
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_target_source": "OPERATING_HARVEST_FLOOR",
            },
        )

        decision = _capped_engine().validate(intent, snapshot())

        self.assertFalse(decision.allowed)
        codes = {issue.code for issue in decision.issues}
        self.assertIn("REWARD_RISK_TOO_LOW", codes)
        self.assertNotIn("TECHNICAL_HARVEST_REWARD_RISK_FLOOR", codes)

    def test_usd_quote_risk_uses_snapshot_usd_jpy_conversion(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=10000,
            tp=1.17554,
            sl=1.17234,
            thesis="risk_must_use_current_usdjpy_conversion",
        )
        low_conversion = snapshot()
        high_conversion = BrokerSnapshot(
            fetched_at_utc=low_conversion.fetched_at_utc,
            positions=low_conversion.positions,
            orders=low_conversion.orders,
            quotes={
                **low_conversion.quotes,
                "USD_JPY": Quote("USD_JPY", bid=200.000, ask=200.010, timestamp_utc=low_conversion.fetched_at_utc),
            },
            account=low_conversion.account,
        )

        low = _capped_engine().validate(intent, low_conversion)
        high = _capped_engine().validate(intent, high_conversion)

        self.assertIsNotNone(low.metrics)
        self.assertIsNotNone(high.metrics)
        assert low.metrics is not None
        assert high.metrics is not None
        self.assertGreater(high.metrics.risk_jpy, low.metrics.risk_jpy * 1.2)

    def test_usd_quote_risk_blocks_when_conversion_quote_missing(self) -> None:
        base = snapshot()
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="missing_conversion_must_not_use_static_157",
        )
        missing_conversion = BrokerSnapshot(
            fetched_at_utc=base.fetched_at_utc,
            positions=base.positions,
            orders=base.orders,
            quotes={"EUR_USD": base.quotes["EUR_USD"]},
            account=base.account,
        )

        decision = _capped_engine().validate(intent, missing_conversion)

        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.metrics)
        self.assertIn("MISSING_CONVERSION_QUOTE", {issue.code for issue in decision.issues})

    def test_home_conversion_prevents_false_stale_conversion_block(self) -> None:
        base = snapshot()
        old = datetime.now(timezone.utc) - timedelta(seconds=120)
        snap = BrokerSnapshot(
            fetched_at_utc=base.fetched_at_utc,
            positions=base.positions,
            orders=base.orders,
            quotes={
                **base.quotes,
                "USD_JPY": Quote("USD_JPY", bid=156.640, ask=156.648, timestamp_utc=old),
            },
            account=base.account,
            home_conversions={"USD": 157.0},
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=1.17554,
            sl=1.17234,
            thesis="home_conversion_is_broker_truth",
        )

        decision = _capped_engine().validate(intent, snap)

        self.assertIsNotNone(decision.metrics)
        self.assertNotIn("STALE_CONVERSION_QUOTE", {issue.code for issue in decision.issues})


class MinLotFloorTest(unittest.TestCase):
    """Coverage for 2026-05-12 emergency fix C — `RiskEngine.validate` must
    refuse sub-`MIN_PRODUCTION_LOT_UNITS` orders even if the caller skipped
    intent_generator's fix B path. The 470901/470904/470907 sequence on
    2026-05-12T07:46 UTC fired three sub-1000u entries (201u / 322u / 2u)
    whose round-trip spread cost dominated any pip target; this gate is
    the second-line defense so a manual `stage-live-order` or replayed
    legacy receipt cannot reach the broker at micro size.
    """

    def setUp(self) -> None:
        import os
        self._prior = os.environ.pop("QR_ALLOW_TEST_MICRO_LOT", None)

    def tearDown(self) -> None:
        import os
        if self._prior is None:
            os.environ.pop("QR_ALLOW_TEST_MICRO_LOT", None)
        else:
            os.environ["QR_ALLOW_TEST_MICRO_LOT"] = self._prior

    def _intent(self, units: int):
        return OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=units,
            tp=1.17000,
            sl=1.17500,
            thesis="min_lot_floor_test",
        )

    def test_999_units_blocked_with_min_lot_violation(self) -> None:
        decision = _capped_engine().validate(self._intent(999), snapshot())
        codes = {issue.code for issue in decision.issues}
        self.assertIn("MIN_LOT_VIOLATION", codes)
        self.assertFalse(decision.allowed)

    def test_322_units_blocked_with_min_lot_violation(self) -> None:
        # The exact 470904 AUD/JPY SHORT 322u scenario.
        decision = _capped_engine().validate(self._intent(322), snapshot())
        self.assertIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})

    def test_2_units_blocked_with_min_lot_violation(self) -> None:
        # The exact 470907 GBP/USD SHORT 2u scenario.
        decision = _capped_engine().validate(self._intent(2), snapshot())
        self.assertIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})

    def test_1000_units_passes_min_lot_floor(self) -> None:
        decision = _capped_engine().validate(self._intent(1000), snapshot())
        self.assertNotIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})

    def test_5000_units_passes_min_lot_floor(self) -> None:
        decision = _capped_engine().validate(self._intent(5000), snapshot())
        self.assertNotIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})

    def test_zero_units_does_not_fire_min_lot_violation(self) -> None:
        # `units == 0` is BAD_UNITS territory (intent didn't reach broker
        # path at all), not MIN_LOT_VIOLATION. The gate covers sub-floor
        # *fillable* sizes.
        decision = _capped_engine().validate(self._intent(0), snapshot())
        codes = {issue.code for issue in decision.issues}
        self.assertNotIn("MIN_LOT_VIOLATION", codes)
        self.assertIn("BAD_UNITS", codes)

    def test_qr_allow_test_micro_lot_disables_gate(self) -> None:
        import os
        os.environ["QR_ALLOW_TEST_MICRO_LOT"] = "1"
        decision = _capped_engine().validate(self._intent(500), snapshot())
        self.assertNotIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})

    def test_short_side_negative_units_also_gated(self) -> None:
        # OrderIntent units carries the sign for SHORT in some code paths;
        # the gate checks `abs()` so SHORT micro orders are caught too.
        from quant_rabbit.risk import MIN_PRODUCTION_LOT_UNITS
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=500,
            tp=1.17000,
            sl=1.17500,
            thesis="short_micro_lot_negative_path",
        )
        decision = _capped_engine().validate(intent, snapshot())
        self.assertIn("MIN_LOT_VIOLATION", {issue.code for issue in decision.issues})
        self.assertEqual(MIN_PRODUCTION_LOT_UNITS, 1000)


if __name__ == "__main__":
    unittest.main()
