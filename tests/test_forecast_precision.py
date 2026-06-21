from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.forecast_precision import (
    bidask_replay_negative_precision_issue,
    bidask_replay_precision_assessment,
    bidask_replay_precision_support,
    oanda_universal_rotation_precision_assessment,
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
                                "daily_stability_status": "DAILY_STABLE",
                                "active_days": 4,
                                "max_daily_sample_share": 0.40,
                                "positive_day_rate": 0.75,
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
        self.assertEqual(assessment["primary_support"]["daily_stability_status"], "DAILY_STABLE")
        self.assertEqual(assessment["primary_support"]["active_days"], 4)
        self.assertEqual(assessment["score_delta"], 18.0)
        self.assertEqual(assessment["rule_source"]["generated_from"], "unit-test")

    def test_bidask_replay_keeps_non_stable_eurusd_edge_rank_only(self) -> None:
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

        self.assertIsNone(assessment["primary_support"])
        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
        )
        self.assertEqual(assessment["score_delta"], 6.0)
        self.assertEqual(assessment["primary_rank_support"]["current_target_pips"], 5.0)
        self.assertEqual(assessment["primary_rank_support"]["current_stop_pips"], 7.0)
        self.assertEqual(assessment["primary_rank_support"]["optimized_profit_factor"], 3.3399)
        self.assertIsNone(
            bidask_replay_precision_support(
                metadata,
                pair="EUR_USD",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=1.17330,
                take_profit=1.17280,
                stop_loss=1.17400,
            )
        )

    def test_bidask_replay_contrarian_support_fades_losing_forecast_bucket(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.87,
            "chart_direction_bias": "LONG",
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
                        "edge_rules": [],
                        "contrarian_edge_rules": [
                            {
                                "name": "AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP5_SL7",
                                "pair": "AUD_JPY",
                                "side": "SHORT",
                                "direction": "DOWN",
                                "forecast_direction": "UP",
                                "faded_direction": "UP",
                                "contrarian_edge": True,
                                "confidence_bucket": "0.75-0.90",
                                "granularity": "S5",
                                "samples": 124,
                                "source_directional_hit_rate": 0.2016,
                                "source_avg_final_pips": -6.7589,
                                "directional_hit_rate": 0.76,
                                "avg_final_pips": 5.8,
                                "avg_mfe_pips": 12.0,
                                "avg_mae_pips": 4.5,
                                "optimized_take_profit_pips": 5.0,
                                "optimized_stop_loss_pips": 7.0,
                                "optimized_avg_realized_pips": 2.4,
                                "optimized_win_rate": 0.70,
                                "optimized_profit_factor": 2.5,
                                "daily_stability_status": "DAILY_STABLE",
                                "active_days": 4,
                                "max_daily_sample_share": 0.40,
                                "positive_day_rate": 0.75,
                                "min_target_pips": 4.8,
                                "max_target_pips": 5.5,
                                "max_stop_pips": 7.2,
                                "audit_report": "unit-test.json",
                            }
                        ],
                        "negative_rules": [
                            {
                                "name": "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
                                "pair": "AUD_JPY",
                                "side": "LONG",
                                "direction": "UP",
                                "granularity": "S5",
                                "samples": 124,
                                "directional_hit_rate": 0.2016,
                                "avg_final_pips": -6.7589,
                                "avg_mfe_pips": 3.7556,
                                "avg_mae_pips": 14.371,
                                "optimized_take_profit_pips": 2.0,
                                "optimized_stop_loss_pips": 2.0,
                                "optimized_avg_realized_pips": -2.0,
                                "optimized_win_rate": 0.0,
                                "optimized_profit_factor": 0.0,
                                "blocks_live_support": True,
                                "audit_report": "unit-test.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = bidask_replay_precision_assessment(
                metadata,
                pair="AUD_JPY",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=114.289,
                take_profit=114.239,
                stop_loss=114.359,
                rules_path=rules_path,
            )

            long_issue = bidask_replay_negative_precision_issue(
                metadata,
                pair="AUD_JPY",
                side="LONG",
                order_type="MARKET",
                method="TREND_CONTINUATION",
                entry=114.289,
                take_profit=114.338,
                stop_loss=114.250,
                rules_path=rules_path,
            )

        self.assertEqual(
            assessment["primary_support"]["name"],
            "AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP5_SL7",
        )
        self.assertTrue(assessment["primary_support"]["contrarian_edge"])
        self.assertEqual(assessment["primary_support"]["faded_direction"], "UP")
        self.assertEqual(assessment["primary_support"]["direction"], "DOWN")
        self.assertEqual(assessment["score_delta"], 18.0)
        self.assertEqual(long_issue["name"], "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY")

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
        self.assertEqual(issue["samples"], 133)

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

    def test_oanda_universal_rotation_adds_rank_only_gbpusd_short_edge(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "m5_atr_percentile_100": 0.82,
            "session_bucket": "ASIA",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = oanda_universal_rotation_precision_assessment(
            metadata,
            pair="GBP_USD",
            side="SHORT",
            order_type="LIMIT",
            method="RANGE_ROTATION",
            entry=1.30000,
            take_profit=1.29950,
            stop_loss=1.30070,
        )

        self.assertIsNone(assessment["primary_support"])
        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "GBP_USD_SHORT_M5_RANGE_REVERSION_ATR_REGIME_HIGH_SESSION_ASIA_TP1_SL1",
        )
        self.assertEqual(assessment["primary_rank_support"]["validation_samples"], 30)
        self.assertEqual(assessment["primary_rank_support"]["validation_win_rate"], 0.70)
        self.assertEqual(assessment["primary_rank_support"]["validation_profit_factor"], 2.45916)
        self.assertTrue(assessment["primary_rank_support"]["rank_only"])
        self.assertEqual(assessment["score_delta"], 10.0)

    def test_oanda_universal_rotation_requires_current_session_and_atr_bucket(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "m5_atr_percentile_100": 0.40,
            "session_bucket": "NY",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = oanda_universal_rotation_precision_assessment(
            metadata,
            pair="GBP_USD",
            side="SHORT",
            order_type="LIMIT",
            method="RANGE_ROTATION",
            entry=1.30000,
            take_profit=1.29950,
            stop_loss=1.30070,
        )

        self.assertIsNone(assessment["primary_rank_support"])
        self.assertEqual(assessment["rank_only_supports"], [])
        self.assertEqual(assessment["score_delta"], 0.0)

    def test_oanda_universal_rotation_scores_eurusd_spread_efficiency_bucket(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "session_bucket": "LONDON_NY_OVERLAP",
            "oanda_m5_spread_regime": "mid",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = oanda_universal_rotation_precision_assessment(
            metadata,
            pair="EUR_USD",
            side="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            take_profit=1.09950,
            stop_loss=1.10070,
        )

        self.assertIsNone(assessment["primary_support"])
        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "EUR_USD_SHORT_M5_PULLBACK_CONTINUATION_SESSION_LONDON_NY_OVERLAP_SPREAD_REGIME_MID_TP1P25_SL1",
        )
        self.assertEqual(assessment["primary_rank_support"]["rank_score_bonus"], 8.0)
        self.assertEqual(assessment["score_delta"], 8.0)

    def test_oanda_universal_rotation_requires_spread_bucket_for_spread_rules(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "session_bucket": "LONDON_NY_OVERLAP",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        assessment = oanda_universal_rotation_precision_assessment(
            metadata,
            pair="EUR_USD",
            side="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            take_profit=1.09950,
            stop_loss=1.10070,
        )

        self.assertIsNone(assessment["primary_rank_support"])
        self.assertEqual(assessment["rank_only_supports"], [])
        self.assertEqual(assessment["score_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
