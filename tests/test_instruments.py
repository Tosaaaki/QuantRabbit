from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, NORMAL_SPREAD_PIPS
from quant_rabbit.models import AccountSummary, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
from quant_rabbit.models import BrokerSnapshot
from quant_rabbit.risk import RiskEngine


class InstrumentUniverseTest(unittest.TestCase):
    def test_default_trader_universe_covers_g8_crosses_with_spread_specs(self) -> None:
        self.assertEqual(len(DEFAULT_TRADER_PAIRS), 28)
        for pair in ("NZD_USD", "USD_CAD", "USD_CHF", "CAD_JPY", "CHF_JPY", "EUR_NZD"):
            self.assertIn(pair, DEFAULT_TRADER_PAIRS)
        self.assertEqual(set(DEFAULT_TRADER_PAIRS), set(NORMAL_SPREAD_PIPS))

    def test_risk_engine_supports_expanded_g8_pair(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "NZD_USD": Quote("NZD_USD", 0.60000, 0.60008, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", 157.00, 157.01, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
        )
        intent = OrderIntent(
            pair="NZD_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            tp=0.60200,
            sl=0.59900,
            thesis="expanded G8 universe smoke test",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="NZD strength versus USD weakness",
                chart_story="trend continuation after shallow pullback",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="SL trades",
            ),
        )

        decision = RiskEngine().validate(intent, snapshot)

        self.assertTrue(decision.allowed, decision.block_reasons)


if __name__ == "__main__":
    unittest.main()
