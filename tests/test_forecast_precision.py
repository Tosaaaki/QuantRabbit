from __future__ import annotations

import unittest

from quant_rabbit.forecast_precision import (
    technical_harvest_precision_assessment,
    technical_harvest_precision_support,
)


class ForecastPrecisionConfluenceTest(unittest.TestCase):
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
