from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
from quant_rabbit.strategy.profile import StrategyProfile


class StrategyProfileTest(unittest.TestCase):
    def test_blocks_history_rejected_profile_for_live_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "bad history",
                            },
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "cap risk",
                            },
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)
            blocked = profile.validate(_intent("USD_JPY"), for_live_send=True)
            repair_dry = profile.validate(_intent("EUR_USD"), for_live_send=False)
            repair_live = profile.validate(_intent("EUR_USD"), for_live_send=True)

            self.assertEqual(blocked[0].code, "STRATEGY_NOT_ELIGIBLE")
            self.assertEqual(blocked[0].severity, "BLOCK")
            self.assertEqual(repair_dry[0].code, "STRATEGY_RISK_REPAIR_REQUIRED")
            self.assertEqual(repair_dry[0].severity, "WARN")
            self.assertEqual(repair_live[0].severity, "BLOCK")

    def test_method_specific_profile_cannot_authorize_another_strategy_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "failed-breakout edge only",
                            }
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)

            breakout = profile.validate(
                _intent("EUR_USD", method=TradeMethod.BREAKOUT_FAILURE),
                for_live_send=True,
            )
            trend = profile.validate(
                _intent("EUR_USD", method=TradeMethod.TREND_CONTINUATION),
                for_live_send=True,
            )

            self.assertEqual(breakout, ())
            self.assertEqual(trend[0].code, "STRATEGY_METHOD_PROFILE_MISSING")
            self.assertEqual(trend[0].severity, "BLOCK")


def _intent(pair: str, *, method: TradeMethod = TradeMethod.TREND_CONTINUATION) -> OrderIntent:
    return OrderIntent(
        pair=pair,
        side=Side.LONG,
        order_type=OrderType.MARKET,
        units=1000,
        entry=1.0,
        tp=1.01,
        sl=0.99,
        thesis="test",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime=f"{method.value} test",
            narrative="test",
            chart_story="test",
            method=method,
            invalidation="test",
        ),
    )


if __name__ == "__main__":
    unittest.main()
