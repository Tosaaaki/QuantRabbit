from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.risk import RiskEngine


def snapshot(*, positions=(), orders=()) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=tuple(positions),
        orders=tuple(orders),
        quotes={
            "EUR_USD": Quote("EUR_USD", bid=1.17322, ask=1.17330, timestamp_utc=now),
            "USD_JPY": Quote("USD_JPY", bid=156.640, ask=156.648, timestamp_utc=now),
        },
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

    def test_external_position_blocks_fresh_entries(self) -> None:
        external = BrokerPosition(
            trade_id="470012",
            pair="USD_JPY",
            side=Side.LONG,
            units=20000,
            entry_price=156.836,
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
            thesis="must_not_trade_while_external_risk_open",
        )
        decision = RiskEngine().validate(intent, snapshot(positions=(external,)))
        codes = {issue.code for issue in decision.issues}
        self.assertFalse(decision.allowed)
        self.assertIn("EXTERNAL_RISK_OPEN", codes)
        self.assertIn("UNPROTECTED_POSITION", codes)

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

    def test_pending_entry_order_blocks_duplicate_fresh_entries(self) -> None:
        pending = BrokerOrder(
            order_id="123",
            pair="AUD_JPY",
            order_type="STOP",
            trade_id=None,
            price=112.576,
            state="PENDING",
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


if __name__ == "__main__":
    unittest.main()
