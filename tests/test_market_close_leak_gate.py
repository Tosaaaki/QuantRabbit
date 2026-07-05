from __future__ import annotations

import unittest

from quant_rabbit.market_close_leak_gate import (
    MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE,
    MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS,
    market_close_leak_family_block_issue,
    market_close_leak_family_payload_issue,
)
from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod


def _family_intent(metadata: dict | None = None) -> OrderIntent:
    return OrderIntent(
        pair="EUR_USD",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        units=1000,
        entry=1.17300,
        tp=1.17500,
        sl=1.17200,
        thesis="failed-break repair entry",
        market_context=MarketContext(
            regime="BREAKOUT_FAILURE failed break",
            narrative="system gateway repair candidate",
            chart_story="failed breakdown reclaimed support",
            method=TradeMethod.BREAKOUT_FAILURE,
            invalidation="support fails",
        ),
        owner=Owner.TRADER,
        metadata=metadata or {},
    )


class MarketCloseLeakGateTest(unittest.TestCase):
    def test_blocks_exact_eurusd_long_breakout_failure_family_without_exception_proof(self) -> None:
        issue = market_close_leak_family_block_issue(_family_intent())

        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue["code"], MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE)
        self.assertEqual(issue["evidence"]["system_gateway_loss_trade_ids"], list(MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS))
        self.assertEqual(
            issue["evidence"]["missing_proofs"],
            [
                "close_gate_proof",
                "contained_risk_timing_evidence",
                "tp_proven_exception_evidence",
            ],
        )

    def test_allows_only_when_close_timing_and_tp_exception_proofs_are_present(self) -> None:
        issue = market_close_leak_family_block_issue(
            _family_intent(
                {
                    "market_close_leak_family_close_gate_proof": True,
                    "market_close_leak_family_contained_risk_timing_evidence": True,
                    "market_close_leak_family_tp_proven_exception": True,
                }
            )
        )

        self.assertIsNone(issue)

    def test_excludes_manual_eurusd_trade_472987_from_system_edge_penalty(self) -> None:
        issue = market_close_leak_family_block_issue(
            _family_intent(
                {
                    "trade_id": "472987",
                    "operator_manual_position": {
                        "classification": "OPERATOR_MANUAL",
                        "management_intent": "KEEP",
                    },
                }
            )
        )

        self.assertIsNone(issue)

    def test_campaign_exposure_recovery_metadata_does_not_bypass_family_gate(self) -> None:
        issue = market_close_leak_family_payload_issue(
            {
                "lane_id": "campaign_exposure_recovery:EUR_USD:LONG:BREAKOUT_FAILURE",
                "method": "BREAKOUT_FAILURE",
                "intent": {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "owner": "trader",
                    "market_context": {"method": "BREAKOUT_FAILURE"},
                    "metadata": {
                        "desk": "campaign_exposure_recovery",
                        "campaign_role": "NOW",
                    },
                },
            }
        )

        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue["code"], MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE)


if __name__ == "__main__":
    unittest.main()
