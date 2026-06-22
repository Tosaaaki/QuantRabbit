from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import quant_rabbit.forecast_precision as forecast_precision
from quant_rabbit.forecast_precision import (
    bidask_replay_negative_precision_issue,
    bidask_replay_precision_assessment,
    bidask_replay_precision_support,
    oanda_universal_rotation_precision_assessment,
    projection_precision_edge_summary,
    projection_precision_gap_summary,
    support_signal_clears_live_precision,
    technical_harvest_precision_geometry_candidate,
    technical_harvest_precision_assessment,
    technical_harvest_precision_support,
)


class ForecastPrecisionConfluenceTest(unittest.TestCase):
    def test_support_signal_rejects_high_headline_low_economic_precision(self) -> None:
        self.assertFalse(
            support_signal_clears_live_precision(
                {
                    "name": "liquidity_sweep_low",
                    "hit_rate": 1.0,
                    "samples": 100,
                    "economic_hit_rate": 0.50,
                    "economic_samples": 200,
                    "timeout_rate": 0.50,
                    "target_pips": 6.0,
                },
                min_wilson_lower=0.90,
                min_samples=30,
                min_target_pips=2.0,
            )
        )

    def test_support_signal_rejects_timeout_rate_without_economic_precision(self) -> None:
        self.assertFalse(
            support_signal_clears_live_precision(
                {
                    "name": "bb_squeeze_expansion_imminent",
                    "hit_rate": 0.98,
                    "samples": 200,
                    "timeout_rate": 0.20,
                },
                min_wilson_lower=0.90,
                min_samples=30,
                min_target_pips=2.0,
            )
        )

    def test_support_signal_allows_high_economic_precision(self) -> None:
        self.assertTrue(
            support_signal_clears_live_precision(
                {
                    "name": "liquidity_sweep_low",
                    "hit_rate": 0.98,
                    "samples": 200,
                    "economic_hit_rate": 0.96,
                    "economic_samples": 200,
                    "timeout_rate": 0.02,
                    "target_pips": 6.0,
                },
                min_wilson_lower=0.90,
                min_samples=30,
                min_target_pips=2.0,
            )
        )

    def test_projection_precision_gap_summary_flags_headline_only_precision(self) -> None:
        gaps = projection_precision_gap_summary(
            {
                "bb_squeeze_expansion_imminent": {
                    "EUR_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.64,
                        "economic_samples": 100,
                        "timeout_rate": 0.34,
                        "timeout_count": 34,
                    },
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                        "timeout_rate": 0.02,
                    },
                },
                "directional_forecast_up": {
                    "EUR_USD:TREND": {
                        "hit_rate": 1.0,
                        "samples": 100,
                        "economic_hit_rate": 0.10,
                        "economic_samples": 100,
                        "timeout_rate": 0.90,
                    }
                },
            },
            min_wilson_lower=0.90,
            min_samples=30,
            exclude_signals=("directional_forecast_up",),
        )

        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap["signal_name"], "bb_squeeze_expansion_imminent")
        self.assertEqual(gap["pair"], "EUR_USD")
        self.assertEqual(gap["regime"], "TREND")
        self.assertAlmostEqual(gap["hit_rate_wilson_lower"], 0.93)
        self.assertLess(gap["economic_hit_rate_wilson_lower"], 0.90)
        self.assertAlmostEqual(gap["timeout_rate"], 0.34)

    def test_projection_precision_edge_summary_keeps_economic_precision_passes(self) -> None:
        edges = projection_precision_edge_summary(
            {
                "session_expansion_london": {
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                        "timeout_rate": 0.02,
                    },
                    "EUR_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.88,
                        "economic_samples": 100,
                        "timeout_rate": 0.10,
                    },
                },
                "directional_forecast_up": {
                    "GBP_USD:TREND": {
                        "hit_rate": 1.0,
                        "samples": 100,
                        "economic_hit_rate": 1.0,
                        "economic_samples": 100,
                    }
                },
            },
            min_wilson_lower=0.90,
            min_samples=30,
            exclude_signals=("directional_forecast_up",),
        )

        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge["signal_name"], "session_expansion_london")
        self.assertEqual(edge["pair"], "GBP_USD")
        self.assertEqual(edge["regime"], "TREND")
        self.assertGreaterEqual(edge["economic_hit_rate_wilson_lower"], 0.90)
        self.assertTrue(edge["passes_economic_precision"])

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
        self.assertEqual(
            assessment["primary_rank_support"]["adoption_status"],
            "RANK_ONLY_NOT_DAILY_STABLE",
        )
        self.assertIn(
            "DAILY_SAMPLE_CONCENTRATED",
            assessment["primary_rank_support"]["adoption_blockers"],
        )
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
        self.assertEqual(issue["samples"], 135)

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

    def test_technical_harvest_geometry_candidate_selects_audited_scalp_shape(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "m1_atr_percentile_100": 0.10,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        candidate = technical_harvest_precision_geometry_candidate(
            metadata,
            pair="EUR_USD",
            side="SHORT",
            order_type="LIMIT",
            method="BREAKOUT_FAILURE",
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["name"], "EUR_USD_DOWN_M1_ATR_LOW_TP5_SL4")
        self.assertEqual(candidate["scalp_tp_pips"], 5.0)
        self.assertEqual(candidate["scalp_stop_pips"], 4.0)
        self.assertGreaterEqual(candidate["scalp_tp_first_wilson95_lower"], 0.90)
        self.assertIsNone(
            technical_harvest_precision_geometry_candidate(
                metadata,
                pair="EUR_USD",
                side="SHORT",
                order_type="MARKET",
                method="BREAKOUT_FAILURE",
            )
        )
        self.assertIsNone(
            technical_harvest_precision_geometry_candidate(
                {**metadata, "m1_atr_percentile_100": 0.50},
                pair="EUR_USD",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
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
        self.assertFalse(assessment["primary_rank_support"]["live_grade_ready"])
        self.assertIn(
            "VALIDATION_WIN_RATE_BELOW_90_PERCENT",
            assessment["primary_rank_support"]["live_gap_reasons"],
        )
        self.assertIn(
            "VALIDATION_WILSON95_LOWER_BELOW_90_PERCENT",
            assessment["primary_rank_support"]["live_gap_reasons"],
        )
        self.assertEqual(
            assessment["live_gap"]["live_grade_metrics"]["validation_win_rate"],
            0.7,
        )
        self.assertEqual(assessment["score_delta"], 10.0)

    def test_oanda_universal_rotation_loads_latest_mining_report_pair_confluence(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "oanda_m5_bar_range": "normal",
            "oanda_m5_spread_regime": "mid",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "oanda_universal_rotation_mining_latest.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:00Z",
                        "high_precision_pair_confluences": [
                            {
                                "pair": "GBP_JPY",
                                "shape": "pullback_continuation",
                                "side": "LONG",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "bar_range:normal",
                                "feature_b": "spread_regime:mid",
                                "qualification": "PASS",
                                "train_n": 43,
                                "train_win_rate": 0.186047,
                                "validation_n": 19,
                                "validation_win_rate": 0.736842,
                                "validation_win_wilson95_lower": 0.51208,
                                "validation_avg_realized_pips": 6.371241,
                                "validation_avg_realized_atr": 0.417306,
                                "validation_profit_factor": 2.969838,
                                "active_days": 9,
                                "positive_day_rate": 0.777777,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="GBP_JPY",
                side="LONG",
                order_type="LIMIT",
                method="PULLBACK_CONTINUATION",
                entry=200.00,
                take_profit=200.10,
                stop_loss=199.85,
                rules_path=rules_path,
            )

        self.assertIsNone(assessment["primary_support"])
        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "GBP_JPY_LONG_M5_PULLBACK_CONTINUATION_BAR_RANGE_NORMAL_SPREAD_REGIME_MID_TP1_SL1",
        )
        self.assertEqual(assessment["primary_rank_support"]["validation_samples"], 19)
        self.assertEqual(assessment["primary_rank_support"]["rank_score_bonus"], 6.0)
        self.assertEqual(
            assessment["primary_rank_support"]["rule_source_section"],
            "high_precision_pair_confluences",
        )
        self.assertFalse(assessment["primary_rank_support"]["live_grade_ready"])
        self.assertIn(
            "VALIDATION_WIN_RATE_BELOW_90_PERCENT",
            assessment["primary_rank_support"]["live_gap_reasons"],
        )
        self.assertEqual(assessment["rule_source"]["dynamic_rule_count"], 1)
        self.assertEqual(assessment["score_delta"], 6.0)

    def test_oanda_universal_rotation_loads_inversion_selector_report(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "m5_atr_percentile_100": 0.82,
            "session_bucket": "ASIA",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "oanda_universal_rotation_mining_latest.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:00Z",
                        "high_precision_inversion_selectors": [
                            {
                                "pair": "AUD_JPY",
                                "shape": "range_reversion",
                                "source_shape": "range_reversion",
                                "source_side": "LONG",
                                "selected_side": "SHORT",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "atr_regime:high",
                                "feature_b": "session:asia",
                                "qualification": "PASS",
                                "train_n": 40,
                                "train_win_rate": 0.75,
                                "validation_n": 12,
                                "validation_win_rate": 0.916667,
                                "validation_win_wilson95_lower": 0.64612,
                                "validation_avg_realized_pips": 4.2,
                                "validation_avg_realized_atr": 0.41,
                                "validation_profit_factor": 5.2,
                                "active_days": 8,
                                "positive_day_rate": 0.875,
                                "source_validation_win_rate": 0.083333,
                                "source_validation_avg_realized_atr": -0.28,
                                "validation_inversion_edge_atr": 0.69,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="AUD_JPY",
                side="SHORT",
                order_type="LIMIT",
                method="RANGE_ROTATION",
                entry=95.000,
                take_profit=94.920,
                stop_loss=95.120,
                rules_path=rules_path,
            )

        support = assessment["primary_rank_support"]
        self.assertEqual(
            support["name"],
            "AUD_JPY_SHORT_M5_RANGE_REVERSION_ATR_REGIME_HIGH_SESSION_ASIA_TP1_SL1",
        )
        self.assertEqual(support["rule_source_section"], "high_precision_inversion_selectors")
        self.assertEqual(support["source_side"], "LONG")
        self.assertEqual(support["source_validation_avg_realized_atr"], -0.28)
        self.assertEqual(support["validation_inversion_edge_atr"], 0.69)
        self.assertEqual(support["rank_score_bonus"], 9.0)
        self.assertTrue(support["rank_only"])
        self.assertFalse(support["live_grade_ready"])
        self.assertEqual(assessment["score_delta"], 9.0)

    def test_oanda_universal_rotation_matches_side_relative_report_features(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "m5_body_atr": -0.20,
            "m5_failed_break": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "oanda_universal_rotation_mining_latest.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:00Z",
                        "qualified_pair_confluences": [
                            {
                                "pair": "CAD_CHF",
                                "shape": "failed_break_fade",
                                "side": "SHORT",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "body:aligned",
                                "feature_b": "failed_break:1",
                                "qualification": "PASS",
                                "train_n": 25,
                                "train_win_rate": 0.52,
                                "validation_n": 12,
                                "validation_win_rate": 0.75,
                                "validation_win_wilson95_lower": 0.46769,
                                "validation_avg_realized_pips": 3.2,
                                "validation_avg_realized_atr": 0.44,
                                "validation_profit_factor": 2.2,
                                "active_days": 6,
                                "positive_day_rate": 0.833333,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="CAD_CHF",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=0.66000,
                take_profit=0.65950,
                stop_loss=0.66070,
                rules_path=rules_path,
            )

        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "CAD_CHF_SHORT_M5_FAILED_BREAK_FADE_BODY_ALIGNED_FAILED_BREAK_1_TP1_SL1",
        )
        self.assertEqual(assessment["primary_rank_support"]["current_oanda_body"], "ALIGNED")
        self.assertEqual(assessment["primary_rank_support"]["current_oanda_failed_break"], "1")
        self.assertEqual(assessment["primary_rank_support"]["rank_score_bonus"], 4.0)
        self.assertEqual(assessment["score_delta"], 4.0)

    def test_oanda_universal_rotation_reads_side_specific_failed_break(self) -> None:
        metadata = {
            "forecast_direction": "DOWN",
            "chart_direction_bias": "SHORT",
            "oanda_m5_body_atr": -0.20,
            "oanda_m5_failed_break_long": False,
            "oanda_m5_failed_break_short": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "oanda_universal_rotation_mining_latest.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:00Z",
                        "qualified_pair_confluences": [
                            {
                                "pair": "CAD_CHF",
                                "shape": "failed_break_fade",
                                "side": "SHORT",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "body:aligned",
                                "feature_b": "failed_break:1",
                                "qualification": "PASS",
                                "train_n": 25,
                                "train_win_rate": 0.52,
                                "validation_n": 12,
                                "validation_win_rate": 0.75,
                                "validation_win_wilson95_lower": 0.46769,
                                "validation_avg_realized_pips": 3.2,
                                "validation_avg_realized_atr": 0.44,
                                "validation_profit_factor": 2.2,
                                "active_days": 6,
                                "positive_day_rate": 0.833333,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="CAD_CHF",
                side="SHORT",
                order_type="LIMIT",
                method="BREAKOUT_FAILURE",
                entry=0.66000,
                take_profit=0.65950,
                stop_loss=0.66070,
                rules_path=rules_path,
            )
            trend_assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="CAD_CHF",
                side="SHORT",
                order_type="LIMIT",
                method="TREND_CONTINUATION",
                entry=0.66000,
                take_profit=0.65950,
                stop_loss=0.66070,
                rules_path=rules_path,
            )

        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "CAD_CHF_SHORT_M5_FAILED_BREAK_FADE_BODY_ALIGNED_FAILED_BREAK_1_TP1_SL1",
        )
        self.assertEqual(assessment["primary_rank_support"]["current_oanda_failed_break"], "1")
        self.assertEqual(assessment["score_delta"], 4.0)
        self.assertIsNone(trend_assessment["primary_rank_support"])
        self.assertEqual(trend_assessment["rank_only_supports"], [])

    def test_oanda_universal_rotation_loads_multi_feature_report_confluence(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "oanda_m5_bar_range": "wide",
            "oanda_m5_wick_reject_long": False,
            "oanda_m5_fast_mom_atr": 0.35,
            "session_bucket": "ASIA",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "oanda_universal_rotation_mining_latest.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00Z",
                        "high_precision_multi_confluences": [
                            {
                                "pair": "CAD_CHF",
                                "shape": "trend_continuation",
                                "side": "LONG",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "bar_range:wide",
                                "feature_b": "wick_reject:0",
                                "feature_c": "fast_mom:aligned",
                                "feature_d": "session:asia",
                                "qualification": "PASS",
                                "train_n": 40,
                                "train_win_rate": 0.72,
                                "validation_n": 12,
                                "validation_win_rate": 0.916667,
                                "validation_win_wilson95_lower": 0.64612,
                                "validation_avg_realized_pips": 2.8,
                                "validation_avg_realized_atr": 0.55,
                                "validation_profit_factor": 5.1,
                                "active_days": 5,
                                "positive_day_rate": 0.8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="CAD_CHF",
                side="LONG",
                order_type="LIMIT",
                method="TREND_CONTINUATION",
                entry=0.66000,
                take_profit=0.66050,
                stop_loss=0.65940,
                rules_path=rules_path,
            )

        support = assessment["primary_rank_support"]
        self.assertEqual(
            support["name"],
            "CAD_CHF_LONG_M5_TREND_CONTINUATION_BAR_RANGE_WIDE_WICK_REJECT_0_FAST_MOM_ALIGNED_SESSION_ASIA_TP1_SL1",
        )
        self.assertEqual(support["current_oanda_bar_range"], "WIDE")
        self.assertEqual(support["current_oanda_wick_reject"], "0")
        self.assertEqual(support["current_oanda_fast_mom"], "ALIGNED")
        self.assertEqual(support["current_oanda_session"], "ASIA")
        self.assertEqual(support["rule_source_section"], "high_precision_multi_confluences")
        self.assertEqual(support["rank_score_bonus"], 7.0)
        self.assertEqual(assessment["score_delta"], 7.0)

    def test_oanda_universal_rotation_uses_packaged_report_when_latest_missing(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "oanda_m5_bar_range": "wide",
            "oanda_m5_fast_mom_atr": 0.35,
            "session_bucket": "TOKYO",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }
        with tempfile.TemporaryDirectory() as tmp:
            missing_latest_path = Path(tmp) / "logs_missing.json"
            packaged_path = Path(tmp) / "packaged_oanda_rules.json"
            packaged_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at_utc": "2026-06-21T02:00:00Z",
                        "high_precision_multi_confluences": [
                            {
                                "pair": "CAD_CHF",
                                "shape": "trend_continuation",
                                "side": "LONG",
                                "exit_shape": "tp1_sl1",
                                "feature_a": "bar_range:wide",
                                "feature_b": "fast_mom:aligned",
                                "feature_c": "session:asia",
                                "qualification": "PASS",
                                "train_n": 44,
                                "train_win_rate": 0.72,
                                "validation_n": 13,
                                "validation_win_rate": 0.923077,
                                "validation_win_wilson95_lower": 0.66612,
                                "validation_avg_realized_pips": 3.1,
                                "validation_avg_realized_atr": 0.57,
                                "validation_profit_factor": 5.4,
                                "active_days": 6,
                                "positive_day_rate": 0.833333,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.dict(
                    os.environ,
                    {forecast_precision.OANDA_UNIVERSAL_ROTATION_RULES_ENV: ""},
                    clear=False,
                ),
                mock.patch.object(
                    forecast_precision,
                    "OANDA_UNIVERSAL_ROTATION_AUDIT_REPORT",
                    str(missing_latest_path),
                ),
                mock.patch.object(
                    forecast_precision,
                    "PACKAGED_OANDA_UNIVERSAL_ROTATION_RULES_PATH",
                    packaged_path,
                ),
            ):
                assessment = oanda_universal_rotation_precision_assessment(
                    metadata,
                    pair="CAD_CHF",
                    side="LONG",
                    order_type="LIMIT",
                    method="TREND_CONTINUATION",
                    entry=0.66000,
                    take_profit=0.66050,
                    stop_loss=0.65940,
                )

        support = assessment["primary_rank_support"]
        self.assertEqual(
            support["name"],
            "CAD_CHF_LONG_M5_TREND_CONTINUATION_BAR_RANGE_WIDE_FAST_MOM_ALIGNED_SESSION_ASIA_TP1_SL1",
        )
        self.assertEqual(assessment["rule_source"]["loaded_report_path"], str(packaged_path))
        self.assertEqual(assessment["rule_source"]["latest_report_path"], str(missing_latest_path))
        self.assertEqual(assessment["rule_source"]["source"], "dynamic_report_with_static_fallback")
        self.assertEqual(support["rule_source_section"], "high_precision_multi_confluences")
        self.assertEqual(assessment["score_delta"], 7.0)

    def test_packaged_oanda_universal_rotation_contains_inversion_selectors(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "chart_direction_bias": "LONG",
            "m5_atr_percentile_100": 0.80,
            "oanda_m5_bar_range": "normal",
            "oanda_m5_spread_regime": "mid",
            "session_bucket": "NY",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
        }

        forecast_precision._load_oanda_universal_rotation_rule_set.cache_clear()
        with (
            mock.patch.dict(
                os.environ,
                {forecast_precision.OANDA_UNIVERSAL_ROTATION_RULES_ENV: ""},
                clear=False,
            ),
            mock.patch.object(
                forecast_precision,
                "OANDA_UNIVERSAL_ROTATION_AUDIT_REPORT",
                "/tmp/qr_missing_oanda_universal_rotation_latest.json",
            ),
        ):
            assessment = oanda_universal_rotation_precision_assessment(
                metadata,
                pair="EUR_USD",
                side="LONG",
                order_type="LIMIT",
                method="TREND_CONTINUATION",
                entry=1.10000,
                take_profit=1.10125,
                stop_loss=1.09900,
            )
        forecast_precision._load_oanda_universal_rotation_rule_set.cache_clear()

        support = assessment["primary_rank_support"]
        self.assertIsNotNone(support)
        self.assertEqual(support["rule_source_section"], "qualified_inversion_selectors")
        self.assertEqual(support["source_side"], "SHORT")
        self.assertLess(support["source_validation_avg_realized_atr"], 0.0)
        self.assertGreater(support["validation_inversion_edge_atr"], 0.0)
        self.assertEqual(support["rank_score_bonus"], 7.0)
        self.assertTrue(support["rank_only"])
        self.assertFalse(support["live_grade_ready"])
        self.assertEqual(assessment["rule_source"]["source"], "dynamic_report_with_static_fallback")

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
