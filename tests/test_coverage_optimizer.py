from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.coverage import CoverageOptimizer


class CoverageOptimizerTest(unittest.TestCase):
    def test_certifies_live_ready_reward_against_remaining_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            replay = root / "replay.json"
            intents.write_text(json.dumps({"results": [_result("LIVE_READY")]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 150.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )
            replay.write_text(json.dumps({"summary": {"days": 1, "evidence_target_covered": 1}}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=replay,
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "LIVE_READY_COVERAGE_READY")
            self.assertEqual(summary.live_ready_lanes, 1)
            self.assertGreaterEqual(summary.live_ready_reward_jpy, 150)
            self.assertIn("Coverage Contract", (root / "coverage.md").read_text())

    def test_names_target_gap_and_promotion_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(json.dumps({"results": [_result("DRY_RUN_PASSED", live_blocker="needs profile promotion")]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 500.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            self.assertEqual(summary.promotion_candidate_lanes, 1)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertTrue(any("live-ready reward misses" in item for item in payload["blockers"]))
            self.assertTrue(any("promote 1 dry-run receipts" in item for item in payload["action_items"]))

    def test_forecast_watch_only_dry_run_is_not_promotion_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_PASSED",
                                live_blocker="forecast watch-only candidate below calibrated live floor",
                                risk_issues=[
                                    {
                                        "code": "FORECAST_WATCH_ONLY",
                                        "message": "dry-run geometry only",
                                        "severity": "WARN",
                                    }
                                ],
                            )
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 500.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            self.assertEqual(summary.promotion_candidate_lanes, 0)
            self.assertEqual(summary.potential_reward_jpy, 0.0)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertFalse(any("promote 1 dry-run receipts" in item for item in payload["action_items"]))
            self.assertFalse(payload["lanes"][0]["counts_after_promotion"])

    def test_reports_harvest_and_runner_opportunity_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                                risk_issues=[
                                    {
                                        "code": "REWARD_RISK_TOO_LOW",
                                        "message": "planned reward/risk is too low",
                                        "severity": "BLOCK",
                                    }
                                ],
                                risk_metrics={
                                    "risk_jpy": 100.0,
                                    "reward_jpy": 120.0,
                                    "reward_risk": 1.2,
                                    "spread_pips": 0.8,
                                },
                            ),
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="trend_trader:GBP_USD:LONG:TREND_CONTINUATION",
                                pair="GBP_USD",
                                risk_issues=[
                                    {
                                        "code": "FORECAST_REQUIRED",
                                        "message": "fresh runner forecast is missing",
                                        "severity": "BLOCK",
                                    }
                                ],
                                risk_metrics={
                                    "risk_jpy": 100.0,
                                    "reward_jpy": 300.0,
                                    "reward_risk": 3.0,
                                    "spread_pips": 0.8,
                                },
                            ),
                        ]
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 500.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            payload = json.loads((root / "coverage.json").read_text())
            modes = payload["opportunity_modes"]
            self.assertEqual(modes["HARVEST"]["lanes"], 1)
            self.assertEqual(modes["RUNNER"]["lanes"], 1)
            self.assertEqual(payload["lanes"][0]["opportunity_mode"], "HARVEST")
            self.assertEqual(payload["lanes"][1]["opportunity_mode"], "RUNNER")
            self.assertTrue(any("harvest and runner opportunity paths" in item for item in payload["action_items"]))
            self.assertIn("Opportunity Modes", (root / "coverage.md").read_text())

    def test_cli_coverage_gap_is_diagnostic_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            output = root / "coverage.json"
            report = root / "coverage.md"
            intents.write_text(json.dumps({"results": []}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "optimize-coverage",
                        "--intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--replay",
                        str(root / "missing_replay.json"),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--market-context-matrix",
                        str(_matrix(root)),
                    ]
                )

            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "COVERAGE_GAP")
            self.assertEqual(summary["live_ready_lanes"], 0)
            payload = json.loads(output.read_text())
            self.assertTrue(any("no LIVE_READY lanes exist" in item for item in payload["blockers"]))
            self.assertTrue(report.exists())

    def test_uses_receipted_risk_metrics_instead_of_static_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(json.dumps({"results": [_result("LIVE_READY", risk_metrics={"risk_jpy": 480.0, "reward_jpy": 720.0, "reward_risk": 1.5, "spread_pips": 0.8})]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 700.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "LIVE_READY_COVERAGE_READY")
            self.assertEqual(summary.live_ready_reward_jpy, 720.0)

    def test_blocks_live_ready_lane_when_broker_truth_risk_metrics_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            intents.write_text(json.dumps({"results": [_result("LIVE_READY", include_risk_metrics=False)]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            self.assertEqual(summary.live_ready_lanes, 0)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertIn("broker-truth risk metrics are missing", payload["lanes"][0]["blockers"][0])

    def test_missing_target_state_is_not_treated_as_target_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": [_result("LIVE_READY")]}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=root / "missing_target.json",
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            payload = json.loads((root / "coverage.json").read_text())
            self.assertTrue(any("daily target state is missing" in item for item in payload["blockers"]))
            self.assertNotEqual(payload["status"], "TARGET_REACHED_PROTECT")

    def test_sequential_ladder_can_cover_target_without_simultaneous_risk_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            replay = root / "replay.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "LIVE_READY",
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                risk_metrics={"risk_jpy": 480.0, "reward_jpy": 4000.0, "reward_risk": 8.33, "spread_pips": 0.8},
                            ),
                            _result(
                                "LIVE_READY",
                                lane_id="failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE",
                                pair="AUD_JPY",
                                entry=100.0,
                                tp=101.0,
                                sl=99.5,
                                risk_metrics={"risk_jpy": 480.0, "reward_jpy": 4000.0, "reward_risk": 8.33, "spread_pips": 0.8},
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 7000.0, "remaining_risk_budget_jpy": 500.0}))
            replay.write_text(json.dumps({"summary": {"days": 1, "evidence_target_covered": 1}}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=replay,
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "LIVE_READY_COVERAGE_READY")
            payload = json.loads((root / "coverage.json").read_text())
            self.assertEqual(payload["sequential_ladder_steps"], 2)
            self.assertFalse(any("simultaneous" in item for item in payload["blockers"]))

    def test_dedupes_exact_geometry_before_counting_live_ready_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            replay = root / "replay.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "LIVE_READY",
                                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                risk_metrics={"risk_jpy": 100.0, "reward_jpy": 200.0, "reward_risk": 2.0, "spread_pips": 0.8},
                            ),
                            _result(
                                "LIVE_READY",
                                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                                risk_metrics={"risk_jpy": 100.0, "reward_jpy": 200.0, "reward_risk": 2.0, "spread_pips": 0.8},
                            ),
                        ]
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 300.0, "remaining_risk_budget_jpy": 500.0}))
            replay.write_text(json.dumps({"summary": {"days": 1, "evidence_target_covered": 1}}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=replay,
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            self.assertEqual(summary.live_ready_lanes, 1)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertEqual(payload["raw_live_ready_reward_jpy"], 400.0)
            self.assertEqual(payload["live_ready_reward_jpy"], 200.0)
            self.assertEqual(payload["duplicate_live_ready_lanes"], 1)
            self.assertTrue(any("dedupe same entry/tp/sl" in item for item in payload["action_items"]))

    def test_names_replay_gap_instead_of_profile_promotion_when_live_ready_coverage_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            replay = root / "replay.json"
            intents.write_text(json.dumps({"results": [_result("LIVE_READY", risk_metrics={"risk_jpy": 480.0, "reward_jpy": 9000.0, "reward_risk": 18.75, "spread_pips": 0.8})]}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 7000.0, "remaining_risk_budget_jpy": 500.0}))
            replay.write_text(json.dumps({"summary": {"days": 3, "evidence_target_covered": 1}}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=replay,
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_REQUIRES_REPLAY_EVIDENCE")
            payload = json.loads((root / "coverage.json").read_text())
            self.assertTrue(any("replay evidence" in item for item in payload["blockers"]))

    def test_stale_spread_blocked_intents_require_evidence_refresh_before_strategy_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            risk_issue = {
                "severity": "BLOCK",
                "code": "SPREAD_TOO_WIDE",
                "message": "current spread is too wide for live entry",
            }
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2000-01-01T00:00:00Z",
                        "results": [
                            _result("DRY_RUN_BLOCKED", risk_issues=[risk_issue]),
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
                                pair="GBP_USD",
                                side="SHORT",
                                risk_issues=[risk_issue],
                            ),
                        ],
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))

            summary = CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=root / "missing_market_context_matrix.json",
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            self.assertEqual(summary.status, "COVERAGE_GAP")
            payload = json.loads((root / "coverage.json").read_text())
            diagnostics = payload["artifact_diagnostics"]
            self.assertTrue(diagnostics["intents_artifact_stale"])
            self.assertTrue(diagnostics["all_lanes_spread_blocked"])
            self.assertTrue(diagnostics["market_context_matrix_missing"])
            self.assertEqual(diagnostics["risk_block_issue_counts"]["SPREAD_TOO_WIDE"], 2)
            self.assertEqual(diagnostics["spread_normalized_candidate_count"], 2)
            self.assertEqual(diagnostics["spread_normalized_candidate_reward_jpy"], 314.0)
            self.assertEqual(diagnostics["spread_normalized_no_live_blocker_count"], 2)
            self.assertTrue(any("order_intents artifact is stale" in item for item in payload["blockers"]))
            self.assertTrue(any("refresh broker-snapshot and generate-intents" in item for item in payload["action_items"]))
            self.assertTrue(any("spread-normalized candidates" in item for item in payload["action_items"]))
            self.assertFalse(any("build at least" in item for item in payload["action_items"]))
            self.assertIn("Artifact Diagnostics", (root / "coverage.md").read_text())
            self.assertIn("Spread-normalized candidates", (root / "coverage.md").read_text())

    def test_spread_normalized_diagnostics_name_remaining_live_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            risk_issue = {
                "severity": "BLOCK",
                "code": "SPREAD_TOO_WIDE",
                "message": "current spread is too wide for live entry",
            }
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2000-01-01T00:00:00Z",
                        "results": [
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                risk_issues=[risk_issue],
                                live_blocker="fresh executable pair forecast missing",
                            ),
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                                pair="EUR_JPY",
                                risk_issues=[risk_issue],
                                live_blocker="forecast confidence below live floor",
                            ),
                        ],
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))

            CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            payload = json.loads((root / "coverage.json").read_text())
            diagnostics = payload["artifact_diagnostics"]
            self.assertEqual(diagnostics["spread_normalized_candidate_count"], 2)
            self.assertEqual(diagnostics["spread_normalized_no_live_blocker_count"], 0)
            self.assertEqual(diagnostics["spread_normalized_live_blocker_counts"]["fresh executable pair forecast missing"], 1)
            self.assertEqual(diagnostics["spread_normalized_live_blocker_counts"]["forecast confidence below live floor"], 1)
            self.assertTrue(any("repair spread-normalized live blockers" in item for item in payload["action_items"]))
            self.assertIn("Spread-normalized live-blocker counts", (root / "coverage.md").read_text())

    def test_profitable_bucket_coverage_maps_backtest_edges_to_current_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            ai_backtest = root / "ai_test_bot_backtest.json"
            strategy_profile = root / "strategy_profile.json"
            matrix = root / "market_context_matrix.json"
            risk_issue = {
                "severity": "BLOCK",
                "code": "SPREAD_TOO_WIDE",
                "message": "current spread is too wide for live entry",
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                risk_issues=[risk_issue],
                                live_blocker="EUR_USD LONG current pair forecast is UNCLEAR",
                            )
                        ]
                    }
                )
            )
            matrix.write_text(
                json.dumps(
                    {
                        "pairs": {
                            "EUR_USD": {
                                "LONG": {
                                    "evidence_ref": "matrix:EUR_USD:LONG",
                                    "support_count": 1,
                                    "reject_count": 1,
                                    "warning_count": 0,
                                    "strongest_support": "legacy edge still positive",
                                    "strongest_reject": "XAU_USD maps to SHORT",
                                    "supports": [
                                        {
                                            "code": "CHART_MEAN_REVERSION",
                                            "layer": "chart",
                                            "message": "EUR_USD bottom rail",
                                            "evidence_refs": ["chart:EUR_USD:structure"],
                                        }
                                    ],
                                    "rejects": [
                                        {
                                            "code": "GOLD_CONTEXT_TECHNICAL_DIRECTION",
                                            "layer": "context_asset_chart",
                                            "message": "EUR_USD XAU_USD technical direction=DOWN maps to SHORT",
                                            "evidence_refs": ["context_asset:XAU_USD"],
                                        }
                                    ],
                                }
                            }
                        }
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            strategy_profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "live execution and pretrade feedback are negative",
                                "live_net_jpy": -1200.0,
                                "pretrade_net_jpy": -350.0,
                                "seat_net_jpy": -7000.0,
                                "seat_pl_n": 19,
                                "seat_win_rate_pct": 21.0,
                            }
                        ]
                    }
                )
            )
            ai_backtest.write_text(
                json.dumps(
                    {
                        "status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                        "live_permission": False,
                        "bucket_contributions": [
                            {
                                "bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED",
                                "managed_net_jpy": 900.0,
                                "raw_net_jpy": 800.0,
                                "trades": 10,
                                "days": 3,
                                "best_trade_jpy": 300.0,
                                "worst_trade_jpy": -120.0,
                            },
                            {
                                "bucket": "trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED",
                                "managed_net_jpy": 600.0,
                                "raw_net_jpy": 500.0,
                                "trades": 8,
                                "days": 2,
                                "best_trade_jpy": 250.0,
                                "worst_trade_jpy": -90.0,
                            },
                        ],
                        "evidence_bucket_contributions": [
                            {
                                "bucket": "pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED",
                                "managed_net_jpy": 850.0,
                                "raw_net_jpy": 850.0,
                                "trades": 5,
                                "days": 2,
                                "win_rate_pct": 80.0,
                            },
                            {
                                "bucket": "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED",
                                "managed_net_jpy": 900.0,
                                "raw_net_jpy": 800.0,
                                "trades": 10,
                                "days": 3,
                                "win_rate_pct": 70.0,
                            },
                        ],
                    }
                )
            )

            CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                ai_backtest_path=ai_backtest,
                strategy_profile_path=strategy_profile,
                market_context_matrix_path=matrix,
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            payload = json.loads((root / "coverage.json").read_text())
            diagnostics = payload["artifact_diagnostics"]["profitable_bucket_coverage"]
            self.assertEqual(diagnostics["positive_pair_directions"], 2)
            self.assertEqual(diagnostics["positive_managed_net_jpy"], 1500.0)
            self.assertTrue(diagnostics["discovery_evidence_not_summed"])
            self.assertEqual(diagnostics["discovery_evidence_count"], 1)
            self.assertEqual(
                diagnostics["discovery_evidence_edges"][0]["bucket"],
                "pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED",
            )
            self.assertEqual(diagnostics["state_counts"]["SPREAD_NORMALIZED_LIVE_BLOCKED"], 1)
            self.assertEqual(diagnostics["state_counts"]["NO_CURRENT_LANE"], 1)
            eur_edge = diagnostics["top_edges"][0]
            self.assertEqual(eur_edge["strategy_profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertTrue(eur_edge["strategy_profile_blocks_live"])
            self.assertEqual(eur_edge["strategy_profile_live_net_jpy"], -1200.0)
            self.assertEqual(eur_edge["strategy_profile_seat_net_jpy"], -7000.0)
            self.assertEqual(eur_edge["matrix_support_count"], 1)
            self.assertEqual(eur_edge["matrix_reject_count"], 1)
            self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", eur_edge["matrix_cross_asset_context"][0])
            self.assertTrue(
                any("repair historical-profitable bucket coverage" in item for item in payload["action_items"])
            )
            self.assertTrue(
                any("promote advisory discovery evidence into primary selection tests" in item for item in payload["action_items"])
            )
            self.assertIn("Profitable Bucket Coverage", (root / "coverage.md").read_text())
            self.assertIn("Discovery Evidence Edges", (root / "coverage.md").read_text())
            self.assertIn("strategy profile", (root / "coverage.md").read_text())

    def test_matrix_supported_profitable_edges_become_repair_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            ai_backtest = root / "ai_test_bot_backtest.json"
            strategy_profile = root / "strategy_profile.json"
            matrix = root / "market_context_matrix.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
                                pair="AUD_JPY",
                                side="SHORT",
                                strategy_issues=[
                                    {
                                        "severity": "BLOCK",
                                        "code": "STRATEGY_NOT_ELIGIBLE",
                                        "message": "AUD_JPY SHORT is BLOCK_UNTIL_NEW_EVIDENCE",
                                    }
                                ],
                            )
                        ]
                    }
                )
            )
            matrix.write_text(
                json.dumps(
                    {
                        "pairs": {
                            "AUD_JPY": {
                                "SHORT": {
                                    "evidence_ref": "matrix:AUD_JPY:SHORT",
                                    "support_count": 4,
                                    "reject_count": 1,
                                    "warning_count": 0,
                                    "strongest_support": "risk-off context maps to SHORT",
                                    "strongest_reject": "spread stressed",
                                    "supports": [
                                        {
                                            "code": "RISK_ASSET_JPY_CROSS_DIRECTION",
                                            "layer": "cross_asset",
                                            "message": "SPX down maps AUD_JPY to SHORT",
                                            "evidence_refs": ["cross:spx"],
                                        },
                                        {
                                            "code": "GOLD_CONTEXT_TECHNICAL_DIRECTION",
                                            "layer": "context_asset_chart",
                                            "message": "gold technical pressure maps to SHORT",
                                            "evidence_refs": ["context_asset:XAU_USD"],
                                        },
                                    ],
                                    "rejects": [
                                        {
                                            "code": "SPREAD_STRESSED",
                                            "layer": "flow",
                                            "message": "spread stressed",
                                            "evidence_refs": ["flow:AUD_JPY"],
                                        }
                                    ],
                                }
                            }
                        }
                    }
                )
            )
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))
            strategy_profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "AUD_JPY",
                                "direction": "SHORT",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "require a new vehicle or market-structure proof",
                            }
                        ]
                    }
                )
            )
            ai_backtest.write_text(
                json.dumps(
                    {
                        "status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                        "bucket_contributions": [
                            {
                                "bucket": "trades:AUD_JPY:SHORT:UNSPECIFIED:UNSPECIFIED",
                                "managed_net_jpy": 1200.0,
                                "raw_net_jpy": 1200.0,
                                "trades": 12,
                                "days": 4,
                            }
                        ],
                    }
                )
            )

            CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                ai_backtest_path=ai_backtest,
                strategy_profile_path=strategy_profile,
                market_context_matrix_path=matrix,
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            payload = json.loads((root / "coverage.json").read_text())
            bucket = payload["artifact_diagnostics"]["profitable_bucket_coverage"]
            queue = bucket["matrix_supported_repair_queue"]
            self.assertEqual(queue[0]["pair"], "AUD_JPY")
            self.assertEqual(queue[0]["direction"], "SHORT")
            self.assertEqual(queue[0]["matrix_support_context"][0], "RISK_ASSET_JPY_CROSS_DIRECTION: SPX down maps AUD_JPY to SHORT")
            self.assertTrue(
                any("prioritize matrix-supported profitable repairs" in item for item in payload["action_items"])
            )
            report = (root / "coverage.md").read_text()
            self.assertIn("Matrix-Supported Repair Queue", report)

    def test_risk_blocker_messages_are_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            risk_issue = {
                "severity": "BLOCK",
                "code": "STALE_QUOTE",
                "message": "quote snapshot is stale",
            }
            intents.write_text(json.dumps({"results": [_result("DRY_RUN_BLOCKED", risk_issues=[risk_issue])]}))
            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 500.0, "remaining_risk_budget_jpy": 500.0}))

            CoverageOptimizer(
                intents_path=intents,
                target_state_path=target,
                replay_path=root / "missing_replay.json",
                market_context_matrix_path=_matrix(root),
                output_path=root / "coverage.json",
                report_path=root / "coverage.md",
            ).run()

            payload = json.loads((root / "coverage.json").read_text())
            self.assertEqual(payload["lanes"][0]["blockers"].count("quote snapshot is stale"), 1)


def _result(
    status: str,
    *,
    lane_id: str = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
    pair: str = "EUR_USD",
    side: str = "LONG",
    order_type: str = "STOP-ENTRY",
    entry: float = 1.1000,
    tp: float = 1.1010,
    sl: float = 1.0995,
    live_blocker: str | None = None,
    risk_issues: list[dict] | None = None,
    strategy_issues: list[dict] | None = None,
    risk_metrics: dict | None = None,
    include_risk_metrics: bool = True,
) -> dict:
    payload = {
        "lane_id": lane_id,
        "status": status,
        "risk_allowed": True,
        "risk_issues": risk_issues or [],
        "strategy_issues": strategy_issues or [],
        "live_blockers": [live_blocker] if live_blocker else [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "units": 1000,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "thesis": "test continuation",
            "market_context": {
                "regime": "TREND_CONTINUATION campaign lane",
                "narrative": "trend continuation pressure",
                "chart_story": "trend staircase",
                "method": "TREND_CONTINUATION",
                "invalidation": "SL trades",
            },
        },
    }
    if include_risk_metrics:
        payload["risk_metrics"] = risk_metrics or {
            "risk_jpy": 78.5,
            "reward_jpy": 157.0,
            "reward_risk": 2.0,
            "spread_pips": 0.8,
        }
    return payload


def _matrix(root: Path) -> Path:
    path = root / "market_context_matrix.json"
    path.write_text(json.dumps({"generated_at_utc": "2026-06-01T00:00:00+00:00", "rows": []}))
    return path


if __name__ == "__main__":
    unittest.main()
