from __future__ import annotations

import unittest

from tools import build_as_4x_proof_path as proof_path
from tools import build_as_live_ready_evidence_loop as evidence_loop


class As4xProofPathTests(unittest.TestCase):
    def test_historical_or_negative_bidask_replay_is_not_live_grade_proof(self) -> None:
        missing = evidence_loop._missing_proof_map(
            {
                "source_evidence": {
                    "historical_only": True,
                    "bidask_rule_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
                },
                "exact_proof_gaps": [],
                "current_blockers": [],
                "status": "DRY_RUN_BLOCKED",
                "risk_allowed": False,
            }
        )

        self.assertFalse(missing["fresh_744h_replay"])
        self.assertFalse(missing["s5_bidask_spread_included_replay"])

    def test_fresh_direction_evidence_marks_under_sampled_gap(self) -> None:
        replay = {
            "price_truth_coverage": {
                "status": "PRICE_TRUTH_OK",
                "adoption_level": "PAIR_LOCAL_RANK_ONLY",
            },
            "forecast_sample_coverage": {
                "pairs": [{"pair": "GBP_USD", "evaluated_samples": 13}],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "UP",
                        "evaluated_samples": 2,
                        "evaluated_active_days": 2,
                    }
                ],
            },
            "precision_rules": {
                "edge_rules": [],
                "daily_stable_edge_rules": [],
                "negative_rules": [],
                "contrarian_edge_rules": [],
                "rejected_daily_stability_segments": [],
            },
            "segments": {"by_pair_direction": []},
        }

        evidence = proof_path.fresh_direction_evidence(replay, "GBP_USD", "UP")

        self.assertEqual(evidence["status"], "EVIDENCE_GAP_UNDER_SAMPLED")
        self.assertFalse(evidence["live_grade_support"])
        self.assertFalse(evidence["can_create_live_permission"])

    def test_gbpusd_breakout_failure_market_close_is_blocked_family_repair(self) -> None:
        rows = [
            evidence_loop.Outcome(
                ts_utc="2026-07-01T00:00:00Z",
                trade_id="TEST_GBP_CLOSE_1",
                pair="GBP_USD",
                side="LONG",
                lane_id="failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                exit_reason="MARKET_ORDER_TRADE_CLOSE",
                realized_pl_jpy=-1000.0,
            ),
            evidence_loop.Outcome(
                ts_utc="2026-07-01T00:05:00Z",
                trade_id="TEST_GBP_CLOSE_2",
                pair="GBP_USD",
                side="LONG",
                lane_id="failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                exit_reason="MARKET_ORDER_TRADE_CLOSE",
                realized_pl_jpy=-500.0,
            ),
        ]

        table = evidence_loop._build_post_gate_gap_family_repair_table(
            generated_at="2026-07-06T00:00:00Z",
            attributed=rows,
            blocked={
                "market_close_trade_ids": set(),
                "residual_trade_ids": set(),
                "residual_keys": set(),
                "market_close_family_key": ("EUR_USD", "LONG", "BREAKOUT_FAILURE"),
            },
        )

        self.assertEqual(table["summary"]["largest_remaining_adverse_family"], "GBP_USD|LONG|BREAKOUT_FAILURE|MARKET_ORDER_TRADE_CLOSE")
        self.assertEqual(table["rows"][0]["action"], "MARKET_CLOSE_PATH_BLOCK")
        self.assertFalse(table["rows"][0]["market_close_path_allowed"])
        self.assertFalse(table["rows"][0]["can_create_live_permission"])
        self.assertEqual(table["containment_filter_trade_ids"], ["TEST_GBP_CLOSE_1", "TEST_GBP_CLOSE_2"])

    def test_manual_position_safety_reports_replaced_expected_tp(self) -> None:
        broker = {
            "positions": [
                {
                    "trade_id": "472987",
                    "owner": "operator_manual",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "take_profit": 1.136,
                    "raw": {
                        "takeProfitOrder": {
                            "id": "472994",
                            "price": "1.13600",
                            "replacesOrderID": "472988",
                            "createTime": "2026-07-06T01:27:42Z",
                        }
                    },
                    "operator_manual_position": {
                        "management_intent": "KEEP",
                        "system_pl_counted": False,
                    },
                }
            ],
            "orders": [
                {
                    "order_id": "472994",
                    "trade_id": "472987",
                    "price": 1.136,
                    "state": "PENDING",
                }
            ],
        }

        safety = proof_path.manual_position_safety(broker)

        self.assertEqual(safety["expected_manual_tp_order_id"], "472988")
        self.assertEqual(safety["current_manual_tp_order_id"], "472994")
        self.assertFalse(safety["expected_tp_order_present"])
        self.assertTrue(safety["expected_tp_replaced_in_broker_truth"])
        self.assertTrue(safety["untouched_by_this_run"])


if __name__ == "__main__":
    unittest.main()
