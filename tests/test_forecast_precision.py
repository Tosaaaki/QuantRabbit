from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.forecast_precision import (
    bidask_replay_negative_precision_issue,
    bidask_replay_precision_assessment,
    bidask_replay_precision_support,
    technical_harvest_precision_assessment,
    technical_harvest_precision_support,
)


class ForecastPrecisionConfluenceTest(unittest.TestCase):
    def test_bidask_replay_rules_are_data_driven_across_pairs(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at_utc": "2026-06-20T00:00:00Z",
                        "generated_from": "unit-test",
                        "edge_rules": [
                            {
                                "name": "GBP_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                                "pair": "GBP_USD",
                                "side": "SHORT",
                                "direction": "DOWN",
                                "granularity": "S5",
                                "samples": 48,
                                "directional_hit_rate": 0.72,
                                "avg_final_pips": 1.8,
                                "avg_mfe_pips": 5.4,
                                "avg_mae_pips": 3.7,
                                "optimized_take_profit_pips": 5.0,
                                "optimized_stop_loss_pips": 7.0,
                                "optimized_avg_realized_pips": 2.1,
                                "optimized_win_rate": 0.69,
                                "optimized_profit_factor": 2.4,
                                "min_target_pips": 4.8,
                                "max_target_pips": 5.5,
                                "max_stop_pips": 7.2,
                                "audit_report": "unit-test.json",
                            }
                        ],
                        "negative_rules": [],
                    }
                ),
                encoding="utf-8",
            )

            assessment = bidask_replay_precision_assessment(
                metadata,
                pair="GBP_USD",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=1.3000,
                take_profit=1.2995,
                stop_loss=1.3007,
                rules_path=rules_path,
            )

        self.assertEqual(
            assessment["primary_support"]["name"],
            "GBP_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
        )
        self.assertEqual(assessment["primary_support"]["pair"], "GBP_USD")
        self.assertEqual(assessment["score_delta"], 18.0)
        self.assertEqual(assessment["rule_source"]["generated_from"], "unit-test")

    def test_bidask_replay_supports_eurusd_down_harvest_shape(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = bidask_replay_precision_assessment(
            metadata,
            pair="EUR_USD",
            side="SHORT",
            order_type="LIMIT",
            method="BREAKOUT_FAILURE",
            entry=1.17330,
            take_profit=1.17280,
            stop_loss=1.17400,
        )

        self.assertEqual(
            assessment["primary_support"]["name"],
            "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
        )
        self.assertEqual(assessment["score_delta"], 18.0)
        self.assertEqual(assessment["primary_support"]["current_target_pips"], 5.0)
        self.assertEqual(assessment["primary_support"]["current_stop_pips"], 7.0)
        self.assertEqual(
            bidask_replay_precision_support(
                metadata,
                pair="EUR_USD",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=1.17330,
                take_profit=1.17280,
                stop_loss=1.17400,
            )["optimized_profit_factor"],
            3.7168,
        )

    def test_bidask_replay_blocks_audjpy_up_negative_pair_direction(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.87,
            "chart_direction_bias": "LONG",
        }

        issue = bidask_replay_negative_precision_issue(
            metadata,
            pair="AUD_JPY",
            side="LONG",
            order_type="MARKET",
            method="TREND_CONTINUATION",
            entry=114.289,
            take_profit=114.338,
            stop_loss=114.250,
        )

        self.assertEqual(issue["name"], "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY")
        self.assertTrue(issue["blocks_live_support"])
        self.assertEqual(issue["samples"], 124)

    def test_holdout_confluence_adds_rotation_support_without_live_override(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "m5_family_disagreement": 0.80,
            "m15_bb_pct_b": 0.35,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = technical_harvest_precision_assessment(
            metadata,
            pair="EUR_USD",
            side="LONG",
            order_type="LIMIT",
            method="BREAKOUT_FAILURE",
            entry=1.10000,
            take_profit=1.10050,
            stop_loss=1.09960,
        )

        self.assertIsNone(assessment["primary_support"])
        self.assertEqual(
            assessment["primary_rotation_support"]["name"],
            "UP_M5_DISAGREE_M15_BB_REVERSION_HOLDOUT_ROTATION_TP5_SL4",
        )
        self.assertEqual(assessment["primary_rotation_support"]["optimized_take_profit_pips"], 10.0)
        self.assertEqual(assessment["primary_rotation_support"]["optimized_stop_loss_pips"], 2.0)
        self.assertEqual(
            assessment["primary_rotation_support"]["optimized_validation_avg_realized_pips"],
            3.13,
        )
        self.assertEqual(assessment["score_delta"], 12.0)
        self.assertIsNone(
            technical_harvest_precision_support(
                metadata,
                pair="EUR_USD",
                side="LONG",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=1.10000,
                take_profit=1.10050,
                stop_loss=1.09960,
            )
        )

    def test_holdout_confluence_respects_bb_reversion_boundary(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "m5_family_disagreement": 0.80,
            "m15_bb_pct_b": 0.20,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = technical_harvest_precision_assessment(
            metadata,
            pair="EUR_USD",
            side="LONG",
            order_type="LIMIT",
            method="BREAKOUT_FAILURE",
            entry=1.10000,
            take_profit=1.10050,
            stop_loss=1.09960,
        )

        self.assertIsNone(assessment["primary_rotation_support"])
        self.assertEqual(assessment["rotation_supports"], [])


if __name__ == "__main__":
    unittest.main()
