from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.risk import RiskEngine


def snapshot(*, positions=(), orders=(), hedging_enabled: bool = False) -> BrokerSnapshot:
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
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_used_jpy=0.0,
            margin_available_jpy=200_000.0,
            hedging_enabled=hedging_enabled,
            fetched_at_utc=now,
        ),
    )


class RiskEngineTest(unittest.TestCase):
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
        decision = RiskEngine().validate(intent, snapshot())
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertIsNotNone(decision.metrics)
        assert decision.metrics is not None
        self.assertLessEqual(decision.metrics.risk_jpy, 500)
        self.assertGreaterEqual(decision.metrics.reward_risk, 1.2)
        self.assertIn("MISSING_MARKET_CONTEXT", {issue.code for issue in decision.issues})

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
        decision = RiskEngine(live_enabled=False).validate(intent, snapshot(), for_live_send=True)
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
        decision = RiskEngine().validate(intent, snapshot())
        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("MISSING_MARKET_CONTEXT", {issue.code for issue in decision.issues})

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
        decision = RiskEngine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)
        self.assertFalse(decision.allowed)
        self.assertIn("METHOD_REGIME_MISMATCH", {issue.code for issue in decision.issues})

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
        decision = RiskEngine().validate(intent, snapshot())
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

        decision = RiskEngine().validate(intent, snap)

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

        decision = RiskEngine().validate(intent, snap)

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
        decision = RiskEngine().validate(intent, snapshot(positions=(external,)))
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
        decision = RiskEngine().validate(intent, snapshot(positions=(manual,)))
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
        decision = RiskEngine().validate(intent, snapshot(positions=(unprotected,)))
        self.assertFalse(decision.allowed)
        self.assertIn("UNPROTECTED_POSITION", {issue.code for issue in decision.issues})

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
        decision = RiskEngine().validate(intent, snapshot(positions=(protected,)))
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

        decision = RiskEngine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_break_even,)))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPEN_POSITION_EXISTS", {issue.code for issue in decision.issues})

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

        decision = RiskEngine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,)))

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
            metadata={"position_intent": "HEDGE", "position_fill": "OPEN_ONLY"},
        )

        from quant_rabbit.risk import RiskPolicy

        decision = RiskEngine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=500.0,
            )
        ).validate(intent, snapshot(positions=(protected_short,), hedging_enabled=True))

        self.assertTrue(decision.allowed, decision.block_reasons)
        self.assertNotIn("OPPOSING_POSITION_NEEDS_HEDGING", {issue.code for issue in decision.issues})

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

        decision = RiskEngine(
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
        decision = RiskEngine().validate(intent, snapshot(orders=(pending,)))
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

        decision = RiskEngine().validate(intent, snapshot(orders=(pending,)))

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

        long_decision = RiskEngine().validate(long_stop_below_market, snapshot())
        short_decision = RiskEngine().validate(short_limit_below_market, snapshot())

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

        decision = RiskEngine(live_enabled=True).validate(intent, snapshot(), for_live_send=True)

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
        decision = RiskEngine().validate(intent, snapshot())
        self.assertFalse(decision.allowed)
        self.assertIn("REWARD_RISK_TOO_LOW", {issue.code for issue in decision.issues})

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

        low = RiskEngine().validate(intent, low_conversion)
        high = RiskEngine().validate(intent, high_conversion)

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

        decision = RiskEngine().validate(intent, missing_conversion)

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

        decision = RiskEngine().validate(intent, snap)

        self.assertIsNotNone(decision.metrics)
        self.assertNotIn("STALE_CONVERSION_QUOTE", {issue.code for issue in decision.issues})


if __name__ == "__main__":
    unittest.main()
