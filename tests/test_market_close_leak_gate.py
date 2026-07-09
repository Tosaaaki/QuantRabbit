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
        issue = market_close_leak_family_block_issue(
            _family_intent({"planned_exit_reason": "MARKET_ORDER_TRADE_CLOSE"})
        )

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
                    "planned_exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                    "market_close_leak_family_close_gate_proof": True,
                    "market_close_leak_family_contained_risk_timing_evidence": True,
                    "market_close_leak_family_tp_proven_exception": True,
                }
            )
        )

        self.assertIsNone(issue)

    def test_allows_tp_proven_harvest_exception_when_market_close_exit_is_not_requested(self) -> None:
        issue = market_close_leak_family_block_issue(
            _family_intent(
                {
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_pessimistic_expectancy_jpy": 180.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                }
            )
        )

        self.assertIsNone(issue)

    def test_payload_allows_tp_proven_harvest_exception_from_lane_packet(self) -> None:
        issue = market_close_leak_family_payload_issue(
            {
                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                "method": "BREAKOUT_FAILURE",
                "intent": {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "owner": "trader",
                    "market_context": {"method": "BREAKOUT_FAILURE"},
                    "metadata": {},
                },
                "opportunity": {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                },
                "self_improvement": {
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_pessimistic_expectancy_jpy": 180.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                },
            }
        )

        self.assertIsNone(issue)

    def test_tp_proven_harvest_still_requires_close_and_timing_for_market_close_exit(self) -> None:
        issue = market_close_leak_family_block_issue(
            _family_intent(
                {
                    "planned_exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_pessimistic_expectancy_jpy": 180.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                }
            )
        )

        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(
            issue["evidence"]["missing_proofs"],
            ["close_gate_proof", "contained_risk_timing_evidence"],
        )

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
