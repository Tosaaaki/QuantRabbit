from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from quant_rabbit.attack_advisor import AttackAdvisor
from quant_rabbit.cli import main


class AttackAdvisorTest(unittest.TestCase):
    def test_recommends_positive_edge_live_ready_lane_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                pair="EUR_USD",
                                side="LONG",
                                reward_jpy=900.0,
                                risk_jpy=300.0,
                                rr=3.0,
                            ),
                            _result(
                                lane_id="failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
                                pair="AUD_JPY",
                                side="SHORT",
                                reward_jpy=1200.0,
                                risk_jpy=350.0,
                                rr=3.42,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1500.0, "remaining_risk_budget_jpy": 800.0}))
            backtest.write_text(
                json.dumps(
                    {
                        "status": "TARGET_COVERAGE_CERTIFIED",
                        "live_permission": False,
                        "blockers": [],
                        "bucket_contributions": [
                            {"bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED", "managed_net_jpy": 3000.0, "trades": 12},
                            {"bucket": "trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED", "managed_net_jpy": -500.0, "trades": 8},
                        ]
                    }
                )
            )
            coverage.write_text(json.dumps({"status": "COVERAGE_GAP"}))

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "ATTACK_COVERAGE_READY")
            self.assertFalse(payload["live_permission"])
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["recommended_now_lane_ids"][0], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            lane = next(item for item in payload["lanes"] if item["lane_id"] == "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(lane["learning_score_delta"], 25.0)
            self.assertIn("ai_backtest_certified_positive_edge", lane["learning_influences"])
            self.assertIn("do_not_raise_loss_cap", payload["settings_advice"])
            self.assertIn("AI Attack Advice Report", (root / "advice.md").read_text())

    def test_surfaces_self_improvement_p0_shadow_candidates_from_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            intents.write_text(json.dumps({"results": []}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 800.0,
                    }
                )
            )
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_GAP",
                        "artifact_diagnostics": {
                            "self_improvement_p0_shadow_live_ready": {
                                "count": 2,
                                "lane_ids": [
                                    "range_trader:AUD_CHF:SHORT:RANGE_ROTATION",
                                    "failure_trader:CAD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
                                ],
                                "reward_jpy": 2200.0,
                                "risk_jpy": 1100.0,
                                "send_blocked": True,
                                "blocker_code": "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                            }
                        },
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["self_improvement_p0_shadow_live_ready"]["count"], 2)
            self.assertTrue(
                any("otherwise-live-ready P0-gated" in item for item in payload["action_items"])
            )

    def test_ignores_blocked_ai_backtest_edges_for_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="small_edge:EUR_USD:LONG:TREND_CONTINUATION",
                                pair="EUR_USD",
                                side="LONG",
                                reward_jpy=400.0,
                                risk_jpy=200.0,
                                rr=2.0,
                            ),
                            _result(
                                lane_id="plain:AUD_JPY:SHORT:BREAKOUT_FAILURE",
                                pair="AUD_JPY",
                                side="SHORT",
                                reward_jpy=900.0,
                                risk_jpy=300.0,
                                rr=3.0,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 800.0}))
            backtest.write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "live_permission": False,
                        "blockers": ["out-of-sample managed net is not positive"],
                        "bucket_contributions": [
                            {"bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED", "managed_net_jpy": 5000.0, "trades": 3}
                        ],
                    }
                )
            )
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["recommended_now_lane_ids"][0], "plain:AUD_JPY:SHORT:BREAKOUT_FAILURE")
            edge_lane = next(item for item in payload["lanes"] if item["lane_id"] == "small_edge:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertIsNone(edge_lane["historical_edge_jpy"])

    def test_negative_ai_backtest_edge_is_advisory_not_rank_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            intents.write_text(json.dumps({"results": [_result()]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )
            backtest.write_text(
                json.dumps(
                    {
                        "status": "TARGET_COVERAGE_CERTIFIED",
                        "live_permission": False,
                        "blockers": [],
                        "bucket_contributions": [
                            {
                                "bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED",
                                "managed_net_jpy": -1500.0,
                                "trades": 12,
                            }
                        ],
                    }
                )
            )
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(payload["recommended_now_lane_ids"], ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"])
            self.assertEqual(lane["score"], 63.0)
            self.assertEqual(lane["historical_edge_jpy"], -1500.0)
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertEqual(lane["learning_influences"], [])
            self.assertTrue(any("advisory only" in item for item in lane["rationale"]))

    def test_allows_audited_profitable_research_backtest_edges_for_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="small_edge:EUR_USD:LONG:TREND_CONTINUATION",
                                pair="EUR_USD",
                                side="LONG",
                                reward_jpy=400.0,
                                risk_jpy=200.0,
                                rr=2.0,
                            ),
                            _result(
                                lane_id="plain:AUD_JPY:SHORT:BREAKOUT_FAILURE",
                                pair="AUD_JPY",
                                side="SHORT",
                                reward_jpy=900.0,
                                risk_jpy=300.0,
                                rr=3.0,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 800.0}))
            backtest.write_text(
                json.dumps(
                    {
                        "status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                        "live_permission": False,
                        "blockers": ["10% target was missed on 37/37 validation days"],
                        "summary": {
                            "selected_trades": 40,
                            "total_managed_net_jpy": 2500.0,
                            "profit_factor": 1.4,
                        },
                        "bucket_contributions": [
                            {"bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED", "managed_net_jpy": 5000.0, "trades": 40}
                        ],
                    }
                )
            )
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["recommended_now_lane_ids"][0], "small_edge:EUR_USD:LONG:TREND_CONTINUATION")
            edge_lane = next(item for item in payload["lanes"] if item["lane_id"] == "small_edge:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(edge_lane["historical_edge_jpy"], 5000.0)
            self.assertEqual(edge_lane["learning_score_delta"], 8.0)
            self.assertIn("ai_backtest_research_positive_edge", edge_lane["learning_influences"])
            self.assertIn("RESEARCH_PROFITABLE_NOT_CERTIFIED", " ".join(edge_lane["rationale"]))

    def test_oanda_rotation_rank_edge_reaches_attack_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            oanda_lane_id = "range_trader:GBP_USD:SHORT:RANGE_ROTATION"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=oanda_lane_id,
                                pair="GBP_USD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                reward_jpy=400.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                entry=1.30000,
                                tp=1.29950,
                                sl=1.30070,
                                metadata={
                                    "forecast_direction": "DOWN",
                                    "chart_direction_bias": "SHORT",
                                    "m5_atr_percentile_100": 0.82,
                                    "session_bucket": "ASIA",
                                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                                    "tp_target_intent": "HARVEST",
                                    "opportunity_mode": "HARVEST",
                                },
                            ),
                            _result(
                                lane_id="plain:AUD_JPY:LONG:TREND_CONTINUATION",
                                pair="AUD_JPY",
                                side="LONG",
                                method="TREND_CONTINUATION",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=500.0,
                                rr=1.0,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["recommended_now_lane_ids"][0], oanda_lane_id)
            lane = next(item for item in payload["lanes"] if item["lane_id"] == oanda_lane_id)
            self.assertEqual(lane["learning_score_delta"], 10.0)
            self.assertIn("oanda_universal_rotation_rank_edge", lane["learning_influences"])
            self.assertTrue(
                any(detail.get("source") == "oanda_universal_rotation" for detail in lane["learning_influence_details"])
            )
            self.assertTrue(any("OANDA universal rotation rank edge" in item for item in lane["rationale"]))

    def test_oanda_rotation_rank_edge_is_neutral_when_capture_economics_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            lane_id = "range_trader:GBP_USD:SHORT:RANGE_ROTATION"
            metadata = {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_percentile_100": 0.82,
                "session_bucket": "ASIA",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                pair="GBP_USD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                reward_jpy=400.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                entry=1.30000,
                                tp=1.29950,
                                sl=1.30070,
                                metadata=metadata,
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = next(item for item in payload["lanes"] if item["lane_id"] == lane_id)
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertEqual(lane["learning_influences"], [])
            self.assertTrue(any("OANDA rank-only rotation edge is size-neutral" in item for item in lane["rationale"]))

    def test_capture_tp_proven_segment_boosts_rank_without_learning_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            capture = root / "capture_economics.json"
            capture_lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            plain_lane_id = "failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE"
            harvest_metadata = {
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=plain_lane_id,
                                pair="AUD_JPY",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                metadata=harvest_metadata,
                            ),
                            _result(
                                lane_id=capture_lane_id,
                                pair="EUR_USD",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                metadata=harvest_metadata,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            capture.write_text(
                json.dumps(
                    {
                        "segment_repair_priorities": {
                            "items": [
                                {
                                    "evidence_ref": "capture:segment:EUR_USD:LONG:BREAKOUT_FAILURE",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "priority_class": "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
                                    "take_profit_proven": True,
                                    "take_profit_trades": 24,
                                    "take_profit_expectancy_jpy": 420.0,
                                    "market_close_net_jpy": -2100.0,
                                }
                            ]
                        }
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["capture_segment_priority_edges"], 1)
            self.assertEqual(payload["recommended_now_lane_ids"][0], capture_lane_id)
            lane = next(item for item in payload["lanes"] if item["lane_id"] == capture_lane_id)
            plain = next(item for item in payload["lanes"] if item["lane_id"] == plain_lane_id)
            self.assertGreater(lane["score"], plain["score"])
            self.assertEqual(lane["learning_influences"], [])
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertTrue(any("capture segment rank edge" in item for item in lane["rationale"]))

    def test_capture_negative_segment_priority_is_advisory_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            capture = root / "capture_economics.json"
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            metadata = {
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                pair="EUR_USD",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                metadata=metadata,
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            capture.write_text(
                json.dumps(
                    {
                        "segment_repair_priorities": {
                            "items": [
                                {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "priority_class": "AVOID_OR_REPRICE_SEGMENT",
                                    "take_profit_proven": True,
                                    "take_profit_trades": 24,
                                }
                            ]
                        }
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(payload["capture_segment_priority_edges"], 0)
            self.assertEqual(payload["recommended_now_lane_ids"], [lane_id])
            self.assertEqual(lane["score"], 52.0)
            self.assertEqual(lane["learning_influences"], [])
            self.assertFalse(any("capture segment rank edge" in item for item in lane["rationale"]))

    def test_capture_segment_priority_does_not_boost_non_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            capture = root / "capture_economics.json"
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            metadata = {
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                                lane_id=lane_id,
                                pair="EUR_USD",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                metadata=metadata,
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            capture.write_text(
                json.dumps(
                    {
                        "segment_repair_priorities": {
                            "items": [
                                {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "priority_class": "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
                                    "take_profit_proven": True,
                                    "take_profit_trades": 24,
                                    "take_profit_expectancy_jpy": 420.0,
                                }
                            ]
                        }
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["blocked_lanes"][0]
            self.assertEqual(payload["capture_segment_priority_edges"], 1)
            self.assertEqual(payload["lanes"], [])
            self.assertEqual(payload["recommended_now_lane_ids"], [])
            self.assertEqual(lane["score"], 52.0)
            self.assertEqual(lane["blockers"], ["status is DRY_RUN_BLOCKED"])
            self.assertFalse(any("capture segment rank edge" in item for item in lane["rationale"]))

    def test_projection_economic_precision_edge_boosts_matching_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            edge_lane_id = "range_trader:GBP_AUD:SHORT:RANGE_ROTATION"
            _write_projection_edge_rows(
                projection,
                signal_name="bb_squeeze_expansion_imminent",
                pair="GBP_AUD",
                direction="EITHER",
                regime="TREND",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=edge_lane_id,
                                pair="GBP_AUD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                reward_jpy=400.0,
                                risk_jpy=200.0,
                                rr=2.0,
                                metadata={
                                    "regime_state": "TREND",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "DOWN",
                                        "signals": [
                                            {
                                                "name": "bb_squeeze_expansion_imminent",
                                                "calibration_name": "bb_squeeze_expansion_imminent",
                                                "direction": "EITHER",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            ),
                            _result(
                                lane_id="plain:AUD_JPY:LONG:TREND_CONTINUATION",
                                pair="AUD_JPY",
                                side="LONG",
                                method="TREND_CONTINUATION",
                                order_type="LIMIT",
                                reward_jpy=500.0,
                                risk_jpy=500.0,
                                rr=1.0,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["recommended_now_lane_ids"][0], edge_lane_id)
            self.assertGreater(payload["projection_economic_precision_edges"], 0)
            lane = next(item for item in payload["lanes"] if item["lane_id"] == edge_lane_id)
            self.assertEqual(lane["learning_score_delta"], 14.0)
            self.assertIn("projection_economic_precision_rank_edge", lane["learning_influences"])
            detail = lane["learning_influence_details"][0]
            self.assertEqual(detail["source"], "projection_ledger")
            self.assertEqual(detail["bucket"], "GBP_AUD:TREND")
            self.assertGreaterEqual(detail["economic_hit_rate_wilson_lower"], 0.90)
            self.assertFalse(
                any(
                    item["signal_name"] == "bb_squeeze_expansion_imminent"
                    and item["bucket"] == "GBP_AUD:TREND"
                    for item in payload["projection_edge_activation_queue"]
                )
            )

    def test_projection_economic_precision_edge_does_not_override_current_precision_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="bb_squeeze_expansion_imminent",
                pair="GBP_AUD",
                direction="EITHER",
                regime="TREND",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="range_trader:GBP_AUD:SHORT:RANGE_ROTATION",
                                pair="GBP_AUD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                metadata={
                                    "regime_state": "TREND",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "DOWN",
                                        "signals": [
                                            {
                                                "name": "bb_squeeze_expansion_imminent",
                                                "calibration_name": "bb_squeeze_expansion_imminent",
                                                "direction": "EITHER",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": False,
                                            }
                                        ],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertNotIn("projection_economic_precision_rank_edge", lane["learning_influences"])

    def test_projection_economic_precision_edge_requires_current_market_support_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="bb_squeeze_expansion_imminent",
                pair="GBP_AUD",
                direction="EITHER",
                regime="TREND",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="range_trader:GBP_AUD:SHORT:RANGE_ROTATION",
                                pair="GBP_AUD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                metadata={
                                    "regime_state": "TREND",
                                    "forecast_market_support": {
                                        "ok": False,
                                        "reason": "no current projection clears audited support floors",
                                        "direction": "DOWN",
                                        "signals": [
                                            {
                                                "name": "bb_squeeze_expansion_imminent",
                                                "calibration_name": "bb_squeeze_expansion_imminent",
                                                "direction": "EITHER",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertNotIn("projection_economic_precision_rank_edge", lane["learning_influences"])

    def test_projection_edge_activation_queue_surfaces_blocked_current_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="session_expansion_london",
                pair="EUR_USD",
                direction="EITHER",
                regime="UNCLEAR",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                pair="EUR_USD",
                                side="LONG",
                                method="TREND_CONTINUATION",
                                metadata={
                                    "regime_state": "UNCLEAR",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "UP",
                                        "signals": [
                                            {
                                                "name": "session_expansion_london",
                                                "calibration_name": "session_expansion_london",
                                                "direction": "EITHER",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            queue = payload["projection_edge_activation_queue"]
            self.assertEqual(queue[0]["activation_status"], "SURFACED_BUT_BLOCKED")
            self.assertEqual(queue[0]["signal_name"], "session_expansion_london")
            self.assertEqual(queue[0]["bucket"], "EUR_USD:UNCLEAR")
            self.assertEqual(queue[0]["matched_lane_count"], 1)
            self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", queue[0]["top_blockers"])
            self.assertNotIn("status is DRY_RUN_BLOCKED", queue[0]["top_blockers"])
            self.assertEqual(queue[0]["primary_repair_category"], "FORECAST_SUPPORT_REPAIR")
            self.assertEqual(queue[0]["repair_categories"][0]["category"], "FORECAST_SUPPORT_REPAIR")
            self.assertIn("forecast/support alignment", queue[0]["primary_repair_action"])
            self.assertTrue(
                any(
                    "activate projection economic precision edges" in item
                    and "FORECAST_SUPPORT_REPAIR" in item
                    for item in payload["action_items"]
                )
            )
            self.assertIn("Projection Edge Activation Queue", (root / "advice.md").read_text())

    def test_projection_edge_activation_queue_surfaces_unselected_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="news_theme_followthrough",
                pair="EUR_USD",
                direction="UP",
                regime="RANGE",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=["FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT"],
                                lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                                pair="EUR_USD",
                                side="SHORT",
                                method="BREAKOUT_FAILURE",
                                metadata={
                                    "regime_state": "RANGE",
                                    "forecast_market_support": {
                                        "ok": False,
                                        "direction": "RANGE",
                                        "signals": [],
                                        "unselected_signals": [
                                            {
                                                "name": "news_theme_followthrough",
                                                "calibration_name": "news_theme_followthrough",
                                                "direction": "UP",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            queue = payload["projection_edge_activation_queue"]
            self.assertEqual(queue[0]["activation_status"], "SURFACED_UNSELECTED")
            self.assertEqual(queue[0]["signal_sources"], ["unselected"])
            self.assertEqual(queue[0]["signal_name"], "news_theme_followthrough_up")
            self.assertEqual(queue[0]["edge_direction"], "UP")
            self.assertEqual(queue[0]["primary_repair_category"], "FORECAST_SUPPORT_REPAIR")
            self.assertEqual(queue[0]["activation_action"], "repair forecast/support alignment; do not trade the edge until current direction support clears")

    def test_projection_edge_activation_queue_keeps_directional_edge_on_matching_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="liquidity_sweep_high",
                pair="AUD_JPY",
                direction="UP",
                regime="UNCLEAR",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                                lane_id="failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
                                pair="AUD_JPY",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                metadata={
                                    "regime_state": "UNCLEAR",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "UP",
                                        "signals": [
                                            {
                                                "name": "liquidity_sweep_high",
                                                "calibration_name": "liquidity_sweep_high",
                                                "direction": "UP",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            ),
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                                lane_id="failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
                                pair="AUD_JPY",
                                side="SHORT",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                metadata={
                                    "regime_state": "UNCLEAR",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "DOWN",
                                        "signals": [
                                            {
                                                "name": "liquidity_sweep_high",
                                                "calibration_name": "liquidity_sweep_high",
                                                "direction": "DOWN",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            queue = payload["projection_edge_activation_queue"]
            self.assertEqual(len(queue), 1)
            surfaced = [
                item
                for item in queue
                if item["activation_status"] == "SURFACED_BUT_BLOCKED"
            ]
            self.assertEqual(len(surfaced), 1)
            self.assertEqual(surfaced[0]["signal_name"], "liquidity_sweep_high_up")
            self.assertEqual(surfaced[0]["edge_direction"], "UP")
            self.assertEqual(surfaced[0]["matched_lane_ids"], ["failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"])
            self.assertEqual(surfaced[0]["blocked_lane_ids"], ["failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"])
            self.assertEqual(surfaced[0]["matched_lane_count"], 1)

    def test_projection_edge_activation_queue_prioritizes_risk_resize_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            coverage = root / "coverage.json"
            projection = root / "projection_ledger.jsonl"
            _write_projection_edge_rows(
                projection,
                signal_name="liquidity_sweep_high",
                pair="AUD_JPY",
                direction="UP",
                regime="UNCLEAR",
            )
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                status="DRY_RUN_BLOCKED",
                                live_blockers=[
                                    "AUD_JPY LONG is BLOCK_UNTIL_NEW_EVIDENCE: historical live loss exceeded the 1742 JPY cap; only risk-resized dry-run receipts can reopen it",
                                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                ],
                                lane_id="failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
                                pair="AUD_JPY",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                metadata={
                                    "regime_state": "UNCLEAR",
                                    "forecast_market_support": {
                                        "ok": True,
                                        "direction": "UP",
                                        "signals": [
                                            {
                                                "name": "liquidity_sweep_high",
                                                "calibration_name": "liquidity_sweep_high",
                                                "direction": "UP",
                                                "hit_rate": 0.96,
                                                "samples": 100,
                                                "economic_hit_rate": 0.96,
                                                "economic_samples": 100,
                                                "live_precision_ok": True,
                                            }
                                        ],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 1000.0}))
            backtest.write_text(json.dumps({"status": "TARGET_COVERAGE_GAP", "bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                projection_ledger_path=projection,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            queue = payload["projection_edge_activation_queue"]
            self.assertEqual(queue[0]["signal_name"], "liquidity_sweep_high_up")
            self.assertEqual(queue[0]["edge_direction"], "UP")
            self.assertEqual(queue[0]["primary_repair_category"], "RISK_RESIZE_DRY_RUN")
            self.assertEqual(queue[0]["repair_categories"][0]["category"], "RISK_RESIZE_DRY_RUN")
            self.assertIn("risk-resized dry-run receipt", queue[0]["primary_repair_action"])

    def test_blocks_when_live_ready_metrics_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(json.dumps({"results": [_result(risk_metrics={})]}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "NO_ATTACK_ADVICE")
            self.assertTrue(any("no LIVE_READY lanes" in item for item in payload["blockers"]))
            self.assertEqual(payload["recommended_now_lane_ids"], [])

    def test_surfaces_matrix_supported_repair_queue_from_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            coverage = root / "coverage.json"
            intents.write_text(json.dumps({"results": []}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 800.0,
                    }
                )
            )
            coverage.write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_GAP",
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "matrix_supported_repair_queue": [
                                    {
                                        "pair": "AUD_JPY",
                                        "direction": "SHORT",
                                        "coverage_state": "SURFACED_BUT_BLOCKED",
                                        "strategy_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                        "strategy_profile_required_fix": "risk-resized dry-run receipt required",
                                        "managed_net_jpy": 7152.6508,
                                        "raw_net_jpy": 6393.126,
                                        "trades": 52,
                                        "days": 6,
                                        "matrix_support_count": 9,
                                        "matrix_reject_count": 0,
                                        "matrix_warning_count": 5,
                                        "matrix_support_context": [
                                            "RISK_ASSET_JPY_CROSS_DIRECTION: SPX down maps to SHORT"
                                        ],
                                        "matrix_cross_asset_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU pressure maps to SHORT"
                                        ],
                                        "top_blockers": ["EXHAUSTION_RANGE_CHASE"],
                                    }
                                ]
                            }
                        },
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            queue = payload["matrix_supported_repair_queue"]
            self.assertEqual(queue[0]["pair"], "AUD_JPY")
            self.assertEqual(queue[0]["direction"], "SHORT")
            self.assertEqual(queue[0]["matrix_support_count"], 9)
            self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", queue[0]["matrix_cross_asset_context"][0])
            self.assertTrue(any("repair matrix-supported profitable edges" in item for item in payload["action_items"]))
            report = (root / "advice.md").read_text()
            self.assertIn("Matrix-Supported Repair Queue", report)
            self.assertIn("AUD_JPY SHORT", report)

    def test_precision_filtered_live_ready_lanes_report_precision_blocker_not_risk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:MARKET",
                                pair="AUD_JPY",
                                side="LONG",
                                reward_jpy=500.0,
                                risk_jpy=100.0,
                                rr=5.0,
                                metadata={"tf_agreement_score": 0.3333},
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 1000.0,
                    }
                )
            )

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "NO_ATTACK_ADVICE")
            self.assertEqual(
                payload["precision_filtered_lane_ids"],
                ["failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:MARKET"],
            )
            self.assertTrue(any("precision filters excluded" in item for item in payload["blockers"]))
            self.assertFalse(any("risk budget" in item for item in payload["blockers"]))
            self.assertIn("tf agreement", (root / "advice.md").read_text())

    def test_range_rail_limit_survives_tf_precision_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                pair="EUR_USD",
                                side="SHORT",
                                method="RANGE_ROTATION",
                                order_type="LIMIT",
                                reward_jpy=1800.0,
                                risk_jpy=750.0,
                                rr=2.4,
                                metadata={
                                    "tf_agreement_score": 0.3333,
                                    "geometry_model": "RANGE_RAIL_LIMIT",
                                    "range_tp_is_inside_box": True,
                                    "range_sl_outside_box": True,
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 5000.0,
                        "remaining_risk_budget_jpy": 2000.0,
                    }
                )
            )

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "ATTACK_PARTIAL")
            self.assertEqual(payload["precision_filtered_lane_ids"], [])
            self.assertEqual(payload["recommended_now_lane_ids"], [lane_id])

    def test_limit_precision_filter_uses_planned_entry_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                pair="EUR_USD",
                                side="SHORT",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=1200.0,
                                risk_jpy=300.0,
                                rr=4.0,
                                metadata={
                                    "price_percentile_24h": 0.02,
                                    "price_percentile_7d": 0.0,
                                    "entry_price_percentile_24h": 0.30,
                                    "entry_price_percentile_7d": 0.24,
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 1000.0,
                    }
                )
            )

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "ATTACK_PARTIAL")
            self.assertEqual(payload["precision_filtered_lane_ids"], [])
            self.assertEqual(payload["recommended_now_lane_ids"], [lane_id])

    def test_limit_precision_filter_blocks_entry_still_at_broader_extreme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                pair="EUR_USD",
                                side="SHORT",
                                method="BREAKOUT_FAILURE",
                                order_type="LIMIT",
                                reward_jpy=1200.0,
                                risk_jpy=300.0,
                                rr=4.0,
                                metadata={
                                    "price_percentile_24h": 0.02,
                                    "price_percentile_7d": 0.0,
                                    "entry_price_percentile_24h": 0.0531,
                                    "entry_price_percentile_7d": 0.0288,
                                },
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 1000.0,
                    }
                )
            )

            summary = AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(summary.status, "NO_ATTACK_ADVICE")
            self.assertEqual(payload["precision_filtered_lane_ids"], [lane_id])
            self.assertIn("7d entry price percentile", payload["precision_filtered_reasons"][lane_id])

    def test_live_ready_lanes_that_pass_precision_but_exceed_budget_keep_budget_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                reward_jpy=900.0,
                                risk_jpy=600.0,
                                rr=1.5,
                                metadata={"tf_agreement_score": 1.0},
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1500.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=root / "missing_backtest.json",
                outcome_mart_path=root / "missing_outcome_mart.json",
                coverage_path=root / "missing_coverage.json",
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["precision_filtered_lane_ids"], [])
            self.assertTrue(any("risk budget" in item for item in payload["blockers"]))

    def test_cli_writes_read_only_advice_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            output = root / "advice.json"
            report = root / "advice.md"
            intents.write_text(json.dumps({"results": [_result()]}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "ai-attack-advice",
                        "--intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--ai-backtest",
                        str(root / "missing_backtest.json"),
                        "--outcome-mart",
                        str(root / "missing_outcome_mart.json"),
                        "--coverage",
                        str(root / "missing_coverage.json"),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["recommended_now_lanes"], 1)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["read_only"])
            self.assertFalse(payload["live_permission"])
            self.assertTrue(report.exists())

    def test_cli_no_attack_advice_is_diagnostic_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            output = root / "advice.json"
            report = root / "advice.md"
            intents.write_text(json.dumps({"results": []}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "ai-attack-advice",
                        "--intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--ai-backtest",
                        str(root / "missing_backtest.json"),
                        "--outcome-mart",
                        str(root / "missing_outcome_mart.json"),
                        "--coverage",
                        str(root / "missing_coverage.json"),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "NO_ATTACK_ADVICE")
            self.assertEqual(summary["recommended_now_lanes"], 0)
            payload = json.loads(output.read_text())
            self.assertTrue(any("no LIVE_READY lanes" in item for item in payload["blockers"]))
            self.assertTrue(report.exists())

    def test_archive_outcome_mart_boosts_condition_specific_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            outcome_mart = root / "outcome_mart.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T08:30:00+00:00",
                        "results": [
                            _result(
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                pair="EUR_USD",
                                side="LONG",
                                method="TREND_CONTINUATION",
                                context={
                                    "method": "TREND_CONTINUATION",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "TREND_CONTINUATION campaign lane",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "TREND_UP"},
                                reward_jpy=900.0,
                                risk_jpy=300.0,
                                rr=3.0,
                            ),
                            _result(
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                pair="EUR_USD",
                                side="LONG",
                                method="RANGE_ROTATION",
                                context={
                                    "method": "RANGE_ROTATION",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "RANGE_ROTATION campaign lane",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "TREND_UP"},
                                reward_jpy=900.0,
                                risk_jpy=300.0,
                                rr=3.0,
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 600.0}))
            backtest.write_text(json.dumps({"bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            outcome_mart.write_text(
                json.dumps(
                    {
                        "condition_edges": [
                            {
                                "key": "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING",
                                "method": "RANGE_ROTATION",
                                "order_type": "MARKET",
                                "session_bucket": "LONDON",
                                "regime": "TRENDING",
                                "net_jpy": 500.0,
                                "avg_jpy": 50.0,
                                "outcome_n": 10,
                            },
                            {
                                "key": "ALL:ALL:TREND_CONTINUATION:MARKET:LONDON:TRENDING",
                                "method": "TREND_CONTINUATION",
                                "order_type": "MARKET",
                                "session_bucket": "LONDON",
                                "regime": "TRENDING",
                                "net_jpy": -100.0,
                                "avg_jpy": -10.0,
                                "outcome_n": 10,
                            },
                        ],
                        "condition_validation": {
                            "matched_edges": [
                                {
                                    "key": "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING",
                                    "predicted_edge": "POSITIVE",
                                    "outcomes": 10,
                                    "actual_net_jpy": 400.0,
                                    "directional_hit_rate_pct": 70.0,
                                }
                            ]
                        },
                        "method_edges": [
                            {
                                "key": "EUR_USD:LONG:RANGE_ROTATION:ALL:ALL:ALL",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "RANGE_ROTATION",
                                "net_jpy": 500.0,
                                "avg_jpy": 50.0,
                                "outcome_n": 10,
                            },
                            {
                                "key": "EUR_USD:LONG:TREND_CONTINUATION:ALL:ALL:ALL",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "TREND_CONTINUATION",
                                "net_jpy": -100.0,
                                "avg_jpy": -10.0,
                                "outcome_n": 10,
                            },
                        ]
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=outcome_mart,
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            self.assertEqual(payload["recommended_now_lane_ids"][0], "range_trader:EUR_USD:LONG:RANGE_ROTATION")
            lane = payload["lanes"][0]
            self.assertEqual(lane["archive_condition_edge_jpy"], 500.0)
            self.assertEqual(lane["archive_condition_trials"], 10)
            self.assertEqual(lane["archive_condition_key"], "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING")
            self.assertEqual(lane["archive_method_edge_jpy"], 500.0)
            self.assertIn("condition=`ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING`", (root / "advice.md").read_text())

    def test_archive_outcome_mart_uses_condition_rollup_when_exact_condition_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            outcome_mart = root / "outcome_mart.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T08:30:00+00:00",
                        "results": [
                            _result(
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                pair="EUR_USD",
                                side="LONG",
                                method="RANGE_ROTATION",
                                context={
                                    "method": "RANGE_ROTATION",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "RANGE_ROTATION campaign lane",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "TREND_UP"},
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            backtest.write_text(json.dumps({"bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            outcome_mart.write_text(
                json.dumps(
                    {
                        "condition_edges": [],
                        "condition_rollups": [
                            {
                                "key": "ALL:ALL:RANGE_ROTATION:MARKET:ALL:TRENDING",
                                "method": "RANGE_ROTATION",
                                "order_type": "MARKET",
                                "session_bucket": "ALL",
                                "regime": "TRENDING",
                                "net_jpy": 750.0,
                                "avg_jpy": 75.0,
                                "outcome_n": 10,
                            }
                        ],
                        "method_edges": [],
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=outcome_mart,
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(lane["archive_condition_edge_jpy"], 750.0)
            self.assertEqual(lane["archive_condition_key"], "ALL:ALL:RANGE_ROTATION:MARKET:ALL:TRENDING")

    def test_archive_condition_boost_requires_passing_walk_forward_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            outcome_mart = root / "outcome_mart.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T08:30:00+00:00",
                        "results": [
                            _result(
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                pair="EUR_USD",
                                side="LONG",
                                method="RANGE_ROTATION",
                                context={
                                    "method": "RANGE_ROTATION",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "RANGE_ROTATION campaign lane",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "TREND_UP"},
                            )
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0, "remaining_risk_budget_jpy": 500.0}))
            backtest.write_text(json.dumps({"bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            outcome_mart.write_text(
                json.dumps(
                    {
                        "condition_edges": [
                            {
                                "key": "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING",
                                "method": "RANGE_ROTATION",
                                "order_type": "MARKET",
                                "session_bucket": "LONDON",
                                "regime": "TRENDING",
                                "net_jpy": 500.0,
                                "avg_jpy": 50.0,
                                "outcome_n": 10,
                            }
                        ],
                        "condition_validation": {
                            "matched_edges": [
                                {
                                    "key": "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING",
                                    "predicted_edge": "POSITIVE",
                                    "outcomes": 10,
                                    "actual_net_jpy": -250.0,
                                    "directional_hit_rate_pct": 30.0,
                                }
                            ]
                        },
                        "method_edges": [],
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=outcome_mart,
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(lane["score"], 63.0)
            self.assertEqual(lane["archive_condition_validation_actual_net_jpy"], -250.0)
            self.assertIn("positive archive condition edge failed walk-forward validation", lane["rationale"])

    def test_archive_condition_boost_rejects_negative_partial_walk_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            outcome_mart = root / "outcome_mart.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T08:30:00+00:00",
                        "results": [
                            _result(
                                lane_id="failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET",
                                pair="USD_CAD",
                                side="LONG",
                                method="BREAKOUT_FAILURE",
                                context={
                                    "method": "BREAKOUT_FAILURE",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "BREAKOUT_FAILURE campaign lane",
                                    "session_bucket": "NY",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "RANGE"},
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )
            backtest.write_text(json.dumps({"bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            outcome_mart.write_text(
                json.dumps(
                    {
                        "condition_edges": [
                            {
                                "key": "ALL:ALL:BREAKOUT_FAILURE:MARKET:NY:RANGE",
                                "method": "BREAKOUT_FAILURE",
                                "order_type": "MARKET",
                                "session_bucket": "NY",
                                "regime": "RANGE",
                                "net_jpy": 500.0,
                                "avg_jpy": 50.0,
                                "outcome_n": 10,
                            }
                        ],
                        "condition_validation": {
                            "matched_edges": [
                                {
                                    "key": "ALL:ALL:BREAKOUT_FAILURE:MARKET:NY:RANGE",
                                    "predicted_edge": "POSITIVE",
                                    "outcomes": 2,
                                    "actual_net_jpy": -1947.8026,
                                    "directional_hit_rate_pct": 0.0,
                                }
                            ]
                        },
                        "method_edges": [],
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=outcome_mart,
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(lane["archive_condition_validation_outcomes"], 2)
            self.assertEqual(lane["archive_condition_validation_actual_net_jpy"], -1947.8026)
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertEqual(lane["learning_influences"], [])
            self.assertIn("negative partial walk-forward validation", " ".join(lane["rationale"]))

    def test_negative_outcome_mart_edge_is_advisory_not_rank_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            backtest = root / "ai_backtest.json"
            outcome_mart = root / "outcome_mart.json"
            coverage = root / "coverage.json"
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T08:30:00+00:00",
                        "results": [
                            _result(
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                method="RANGE_ROTATION",
                                context={
                                    "method": "RANGE_ROTATION",
                                    "narrative": "test",
                                    "chart_story": "test",
                                    "invalidation": "test",
                                    "regime": "RANGE_ROTATION campaign lane",
                                    "session": "generated dry-run",
                                },
                                metadata={"regime_state": "TREND_UP"},
                            )
                        ],
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )
            backtest.write_text(json.dumps({"bucket_contributions": []}))
            coverage.write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
            outcome_mart.write_text(
                json.dumps(
                    {
                        "condition_edges": [
                            {
                                "key": "ALL:ALL:RANGE_ROTATION:MARKET:LONDON:TRENDING",
                                "method": "RANGE_ROTATION",
                                "order_type": "MARKET",
                                "session_bucket": "LONDON",
                                "regime": "TRENDING",
                                "net_jpy": -900.0,
                                "avg_jpy": -90.0,
                                "outcome_n": 10,
                            }
                        ],
                        "method_edges": [],
                    }
                )
            )

            AttackAdvisor(
                intents_path=intents,
                target_state_path=target,
                ai_backtest_path=backtest,
                outcome_mart_path=outcome_mart,
                coverage_path=coverage,
                output_path=root / "advice.json",
                report_path=root / "advice.md",
            ).run()

            payload = json.loads((root / "advice.json").read_text())
            lane = payload["lanes"][0]
            self.assertEqual(payload["recommended_now_lane_ids"], ["range_trader:EUR_USD:LONG:RANGE_ROTATION"])
            self.assertEqual(lane["score"], 63.0)
            self.assertEqual(lane["archive_condition_edge_jpy"], -900.0)
            self.assertEqual(lane["learning_score_delta"], 0.0)
            self.assertEqual(lane["learning_influences"], [])
            self.assertIn("negative archive condition edge; advisory only", lane["rationale"])


def _result(
    *,
    lane_id: str = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
    status: str = "LIVE_READY",
    pair: str = "EUR_USD",
    side: str = "LONG",
    method: str = "TREND_CONTINUATION",
    order_type: str = "MARKET",
    reward_jpy: float = 900.0,
    risk_jpy: float = 300.0,
    rr: float = 3.0,
    risk_metrics: dict | None = None,
    context: dict | None = None,
    metadata: dict | None = None,
    live_blockers: list[str] | None = None,
    risk_issues: list[str] | None = None,
    strategy_issues: list[str] | None = None,
    entry: float = 1.1,
    tp: float = 1.2,
    sl: float = 1.0,
) -> dict:
    metrics = risk_metrics if risk_metrics is not None else {"reward_jpy": reward_jpy, "risk_jpy": risk_jpy, "reward_risk": rr, "spread_pips": 0.8}
    return {
        "lane_id": lane_id,
        "status": status,
        "risk_metrics": metrics,
        "risk_issues": risk_issues or [],
        "strategy_issues": strategy_issues or [],
        "live_blockers": live_blockers or [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "units": 1000,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "market_context": context
            if context is not None
            else {"method": method, "narrative": "test", "chart_story": "test", "invalidation": "test"},
            "metadata": metadata or {},
        },
    }


def _write_projection_edge_rows(
    path: Path,
    *,
    signal_name: str,
    pair: str,
    direction: str,
    regime: str,
) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(100):
        status = "HIT" if idx < 96 else "MISS"
        rows.append(
            {
                "timestamp_emitted_utc": f"2026-06-{1 + idx // 60:02d}T00:{idx % 60:02d}:00Z",
                "resolved_at_utc": f"2026-06-{1 + idx // 60:02d}T00:{idx % 60:02d}:30Z",
                "cycle_id": f"projection-edge-{status.lower()}-{idx}",
                "pair": pair,
                "signal_name": signal_name,
                "direction": direction,
                "lead_time_min": 5.0,
                "confidence": 0.95,
                "entry_price": 1.0,
                "predicted_target_price": 1.001,
                "resolution_window_min": 5.0,
                "resolution_status": status,
                "resolution_evidence": "unit-test projection edge",
                "regime_at_emission": regime,
            }
        )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


if __name__ == "__main__":
    unittest.main()
