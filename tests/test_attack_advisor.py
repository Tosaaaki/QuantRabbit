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
            self.assertIn("do_not_raise_loss_cap", payload["settings_advice"])
            self.assertIn("AI Attack Advice Report", (root / "advice.md").read_text())

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


def _result(
    *,
    lane_id: str = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
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
) -> dict:
    metrics = risk_metrics if risk_metrics is not None else {"reward_jpy": reward_jpy, "risk_jpy": risk_jpy, "reward_risk": rr, "spread_pips": 0.8}
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_metrics": metrics,
        "risk_issues": [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "units": 1000,
            "entry": 1.1,
            "tp": 1.2,
            "sl": 1.0,
            "market_context": context
            if context is not None
            else {"method": method, "narrative": "test", "chart_story": "test", "invalidation": "test"},
            "metadata": metadata or {},
        },
    }


if __name__ == "__main__":
    unittest.main()
