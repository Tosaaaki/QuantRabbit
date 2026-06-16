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
            repair_issue = {
                "code": "STRATEGY_RISK_REPAIR_REQUIRED",
                "message": "risk-repair profile can be promoted by this receipt",
                "severity": "BLOCK",
                "strategy_profile_evidence": {
                    "profile_match": "pair_side",
                    "profile_pair": "EUR_USD",
                    "profile_direction": "LONG",
                    "profile_status": "RISK_REPAIR_CANDIDATE",
                    "requested_method": "TREND_CONTINUATION",
                    "required_fix": "old sizing broke the loss cap",
                },
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_PASSED",
                                live_blocker="needs profile promotion",
                                live_strategy_issues=[repair_issue],
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
            self.assertEqual(summary.promotion_candidate_lanes, 1)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertTrue(any("live-ready reward misses" in item for item in payload["blockers"]))
            self.assertTrue(any("promote 1 dry-run receipts" in item for item in payload["action_items"]))

    def test_missing_strategy_profile_is_not_receipt_promotion_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            missing_issue = {
                "code": "STRATEGY_PROFILE_MISSING",
                "message": "CAD_JPY SHORT is absent from the mined strategy profile",
                "severity": "BLOCK",
                "strategy_profile_evidence": {
                    "profile_match": "missing",
                    "profile_pair": "CAD_JPY",
                    "profile_direction": "SHORT",
                    "requested_method": "TREND_CONTINUATION",
                },
            }
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_PASSED",
                                lane_id="trend_trader:CAD_JPY:SHORT:TREND_CONTINUATION",
                                pair="CAD_JPY",
                                side="SHORT",
                                live_blocker="CAD_JPY SHORT is absent from the mined strategy profile",
                                live_strategy_issues=[missing_issue],
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

            self.assertEqual(summary.promotion_candidate_lanes, 0)
            payload = json.loads((root / "coverage.json").read_text())
            self.assertFalse(payload["lanes"][0]["counts_after_promotion"])
            self.assertFalse(any("promote 1 dry-run receipts" in item for item in payload["action_items"]))

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
                                method="BREAKOUT_FAILURE",
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
            self.assertEqual(modes["HARVEST"]["top_issue_codes"][0]["code"], "REWARD_RISK_TOO_LOW")
            self.assertEqual(modes["RUNNER"]["top_issue_codes"][0]["code"], "FORECAST_REQUIRED")
            self.assertEqual(modes["HARVEST"]["top_live_blocker_codes"][0]["code"], "REWARD_RISK_TOO_LOW")
            self.assertEqual(modes["RUNNER"]["top_live_blocker_codes"][0]["code"], "FORECAST_REQUIRED")
            self.assertEqual(payload["lanes"][0]["opportunity_mode"], "HARVEST")
            self.assertEqual(payload["lanes"][0]["issue_codes"], ["REWARD_RISK_TOO_LOW"])
            self.assertEqual(payload["lanes"][0]["live_blocker_codes"], ["REWARD_RISK_TOO_LOW"])
            self.assertEqual(payload["lanes"][1]["opportunity_mode"], "RUNNER")
            self.assertTrue(any("harvest and runner opportunity paths" in item for item in payload["action_items"]))
            self.assertTrue(any("top codes: REWARD_RISK_TOO_LOW" in item for item in payload["action_items"]))
            self.assertIn("Opportunity Modes", (root / "coverage.md").read_text())

    def test_opportunity_modes_separate_live_blocker_codes_from_advisory_issue_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            forecast_message = "EUR_USD LONG forecast UP confidence 0.48 < 0.65"
            profile_message = "EUR_USD LONG is absent from the mined strategy profile"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                "DRY_RUN_PASSED",
                                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                                method="BREAKOUT_FAILURE",
                                live_blocker=forecast_message,
                                risk_issues=[
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "message": forecast_message,
                                        "severity": "WARN",
                                    }
                                ],
                                strategy_issues=[
                                    {
                                        "code": "STRATEGY_NOT_ELIGIBLE",
                                        "message": "pair-side is watch only in dry-run profile evidence",
                                        "severity": "WARN",
                                    }
                                ],
                                live_strategy_issues=[
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": profile_message,
                                        "severity": "BLOCK",
                                    }
                                ],
                                risk_metrics={
                                    "risk_jpy": 100.0,
                                    "reward_jpy": 120.0,
                                    "reward_risk": 1.2,
                                    "spread_pips": 0.8,
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
            lane = payload["lanes"][0]
            mode = payload["opportunity_modes"]["HARVEST"]
            self.assertEqual(
                lane["issue_codes"],
                ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "STRATEGY_NOT_ELIGIBLE"],
            )
            self.assertEqual(
                lane["live_blocker_codes"],
                ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "STRATEGY_PROFILE_MISSING"],
            )
            self.assertEqual(
                [item["code"] for item in mode["top_live_blocker_codes"]],
                ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "STRATEGY_PROFILE_MISSING"],
            )
            self.assertNotIn(
                "STRATEGY_NOT_ELIGIBLE",
                [item["code"] for item in mode["top_live_blocker_codes"]],
            )
            self.assertIn("live_blocker_codes=`FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", (root / "coverage.md").read_text())

    def test_opportunity_mode_prefers_intent_metadata_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            result = _result(
                "DRY_RUN_BLOCKED",
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 350.0,
                    "reward_risk": 3.5,
                    "spread_pips": 0.8,
                },
            )
            result["intent"]["metadata"] = {
                "opportunity_mode": "HARVEST",
                "opportunity_mode_reason": "tp_target_intent=HARVEST",
            }
            intents.write_text(json.dumps({"results": [result]}))
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
            self.assertEqual(payload["lanes"][0]["opportunity_mode"], "HARVEST")
            self.assertEqual(payload["opportunity_modes"]["HARVEST"]["lanes"], 1)
            self.assertEqual(payload["opportunity_modes"]["RUNNER"]["lanes"], 0)

    def test_opportunity_mode_fallback_harvest_intent_precedes_high_reward_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            result = _result(
                "DRY_RUN_BLOCKED",
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 350.0,
                    "reward_risk": 3.5,
                    "spread_pips": 0.8,
                },
            )
            result["intent"]["metadata"] = {"tp_target_intent": "HARVEST"}
            intents.write_text(json.dumps({"results": [result]}))
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
            self.assertEqual(payload["lanes"][0]["opportunity_mode"], "HARVEST")
            self.assertEqual(payload["opportunity_modes"]["HARVEST"]["lanes"], 1)
            self.assertEqual(payload["opportunity_modes"]["RUNNER"]["lanes"], 0)
            self.assertTrue(any("add missing RUNNER lane generation" in item for item in payload["action_items"]))

    def test_runner_candidate_diagnostics_explain_trend_lanes_demoted_to_harvest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            result = _result(
                "DRY_RUN_BLOCKED",
                lane_id="trend_trader:GBP_USD:LONG:TREND_CONTINUATION",
                pair="GBP_USD",
                risk_issues=[
                    {
                        "code": "TREND_MARKET_NOT_OPERATING_TREND",
                        "message": "RANGE regime is not a clean runner trend",
                        "severity": "BLOCK",
                    }
                ],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 280.0,
                    "reward_risk": 2.8,
                    "spread_pips": 0.8,
                },
            )
            result["intent"]["metadata"] = {
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_attach_reason": "RANGE regime is not a clean runner trend",
            }
            intents.write_text(json.dumps({"results": [result]}))
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
            diagnostics = payload["runner_candidate_diagnostics"]
            self.assertEqual(diagnostics["status"], "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST")
            self.assertEqual(diagnostics["trend_candidate_lanes"], 1)
            self.assertEqual(diagnostics["runner_qualified_lanes"], 0)
            self.assertEqual(diagnostics["attached_harvest_lanes"], 1)
            self.assertEqual(diagnostics["top_demotion_reasons"][0]["reason"], "RANGE regime is not a clean runner trend")
            runner_mode = payload["opportunity_modes"]["RUNNER"]
            self.assertEqual(runner_mode["lanes"], 0)
            self.assertEqual(runner_mode["diagnostic_candidate_lanes"], 1)
            self.assertEqual(runner_mode["demoted_to_harvest_lanes"], 1)
            self.assertEqual(runner_mode["diagnostic_status"], "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST")
            self.assertEqual(runner_mode["top_demotion_reasons"][0]["reason"], "RANGE regime is not a clean runner trend")
            self.assertEqual(runner_mode["top_issue_codes"][0]["code"], "TREND_MARKET_NOT_OPERATING_TREND")
            self.assertEqual(runner_mode["top_live_blocker_codes"][0]["code"], "TREND_MARKET_NOT_OPERATING_TREND")
            self.assertTrue(
                any(
                    "repair both harvest and runner opportunity paths" in item
                    and "repair runner qualification" in item
                    for item in payload["action_items"]
                )
            )
            self.assertIn("Runner Candidate Diagnostics", (root / "coverage.md").read_text())

    def test_live_ready_harvest_does_not_hide_demoted_runner_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            harvest = _result(
                "LIVE_READY",
                lane_id="range_trader:USD_CAD:SHORT:RANGE_ROTATION",
                pair="USD_CAD",
                side="SHORT",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 120.0,
                    "reward_risk": 1.2,
                    "spread_pips": 0.8,
                },
            )
            harvest["intent"]["metadata"] = {
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            demoted_runner = _result(
                "DRY_RUN_BLOCKED",
                lane_id="trend_trader:GBP_USD:LONG:TREND_CONTINUATION",
                pair="GBP_USD",
                risk_issues=[
                    {
                        "code": "TREND_MARKET_NOT_OPERATING_TREND",
                        "message": "UNCLEAR regime is not a clean runner trend",
                        "severity": "BLOCK",
                    }
                ],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 280.0,
                    "reward_risk": 2.8,
                    "spread_pips": 0.8,
                },
            )
            demoted_runner["intent"]["metadata"] = {
                "opportunity_mode": "HARVEST",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_attach_reason": "UNCLEAR regime is not a clean runner trend",
            }
            intents.write_text(json.dumps({"results": [harvest, demoted_runner]}))
            target.write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
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
            self.assertEqual(payload["opportunity_modes"]["HARVEST"]["live_ready_lanes"], 1)
            self.assertEqual(payload["opportunity_modes"]["RUNNER"]["lanes"], 0)
            self.assertEqual(payload["opportunity_modes"]["RUNNER"]["diagnostic_candidate_lanes"], 1)
            self.assertEqual(payload["opportunity_modes"]["RUNNER"]["demoted_to_harvest_lanes"], 1)
            self.assertEqual(
                payload["opportunity_modes"]["RUNNER"]["top_live_blocker_codes"][0]["code"],
                "TREND_MARKET_NOT_OPERATING_TREND",
            )
            self.assertEqual(payload["runner_candidate_diagnostics"]["trend_candidate_lanes"], 1)
            self.assertIn("diagnostic_candidates=`1`", (root / "coverage.md").read_text())
            self.assertTrue(
                any(
                    "repair runner qualification before widening discovery" in item
                    and "UNCLEAR regime is not a clean runner trend" in item
                    for item in payload["action_items"]
                )
            )

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

    def test_fresh_all_spread_blocked_intents_defer_strategy_expansion(self) -> None:
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
                        "generated_at_utc": "2999-01-01T00:00:00Z",
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
            self.assertFalse(diagnostics["intents_artifact_stale"])
            self.assertFalse(diagnostics["market_context_matrix_missing"])
            self.assertTrue(diagnostics["all_lanes_spread_blocked"])
            self.assertTrue(diagnostics["requires_market_evidence_refresh"])
            self.assertTrue(any("spread-blocked" in item for item in payload["blockers"]))
            self.assertTrue(any("refresh broker-snapshot and generate-intents" in item for item in payload["action_items"]))
            self.assertFalse(any("build at least" in item for item in payload["action_items"]))
            self.assertFalse(any("harvest and runner opportunity paths" in item for item in payload["action_items"]))

    def test_all_quote_stale_intents_require_quote_refresh_before_strategy_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            stale_quote_issue = {
                "severity": "BLOCK",
                "code": "STALE_QUOTE",
                "message": "quote snapshot is stale",
            }
            telemetry_issue = {
                "severity": "BLOCK",
                "code": "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
                "message": "forecast telemetry waits for a fresh quote",
            }
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2999-01-01T00:00:00Z",
                        "results": [
                            _result("DRY_RUN_BLOCKED", risk_issues=[stale_quote_issue]),
                            _result(
                                "DRY_RUN_BLOCKED",
                                lane_id="failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
                                pair="GBP_USD",
                                side="SHORT",
                                risk_issues=[stale_quote_issue],
                                strategy_issues=[telemetry_issue],
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
            self.assertTrue(diagnostics["all_lanes_quote_stale"])
            self.assertTrue(diagnostics["requires_market_evidence_refresh"])
            self.assertEqual(diagnostics["quote_stale_result_count"], 2)
            self.assertEqual(diagnostics["quote_normalized_candidate_count"], 2)
            self.assertEqual(diagnostics["quote_normalized_candidate_reward_jpy"], 314.0)
            self.assertTrue(any("quote-stale" in item for item in payload["blockers"]))
            self.assertTrue(any("quote-normalized candidates" in item for item in payload["action_items"]))
            self.assertFalse(any("build at least" in item for item in payload["action_items"]))
            report = (root / "coverage.md").read_text()
            self.assertIn("All lanes quote-stale", report)
            self.assertIn("Quote-normalized candidates", report)

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

    def test_range_forecast_method_mismatch_points_to_range_rotation_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            mismatch_issue = {
                "severity": "BLOCK",
                "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                "message": "RANGE forecast only authorizes RANGE_ROTATION rail geometry",
            }
            range_block = {
                "severity": "BLOCK",
                "code": "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                "message": "RANGE_ROTATION is on the wrong side of broader location",
            }
            failure_lane = _result(
                "DRY_RUN_PASSED",
                lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                pair="EUR_USD",
                side="SHORT",
                method="BREAKOUT_FAILURE",
                order_type="LIMIT",
                risk_issues=[mismatch_issue],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 260.0,
                    "reward_risk": 2.6,
                    "spread_pips": 0.8,
                },
            )
            failure_lane["intent"]["metadata"] = {
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.49,
                "chart_direction_bias": "SHORT",
                "range_phase": "RANGE_FORMING",
            }
            range_lane = _result(
                "DRY_RUN_BLOCKED",
                lane_id="range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                pair="EUR_USD",
                side="SHORT",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                risk_issues=[range_block],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 90.0,
                    "reward_risk": 0.9,
                    "spread_pips": 0.8,
                },
            )
            range_lane["intent"]["metadata"] = {
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.49,
                "chart_direction_bias": "SHORT",
                "range_phase": "RANGE_FORMING",
                "range_entry_side": "resistance",
                "price_percentile_24h": 0.07,
                "price_percentile_7d": 0.65,
            }
            intents.write_text(json.dumps({"results": [failure_lane, range_lane]}))
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
            diagnostics = payload["perspective_alignment_diagnostics"]
            self.assertEqual(diagnostics["status"], "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED")
            self.assertEqual(diagnostics["range_forecast_method_mismatch_groups"], 1)
            self.assertEqual(diagnostics["range_forecast_method_mismatch_lanes"], 1)
            top = diagnostics["range_forecast_method_mismatch_top"][0]
            self.assertEqual(top["pair"], "EUR_USD")
            self.assertEqual(top["direction"], "SHORT")
            self.assertEqual(top["method_mismatch_lanes"], 1)
            self.assertEqual(top["range_rotation_lanes"], 1)
            self.assertEqual(
                top["range_rotation_top_live_blocker_codes"][0]["code"],
                "RANGE_ROTATION_BROADER_LOCATION_CHASE",
            )
            self.assertTrue(
                any("repair RANGE-forecast method mismatches" in item for item in payload["action_items"])
            )
            self.assertIn("Perspective Alignment", (root / "coverage.md").read_text())

    def test_range_mismatch_surfaces_opposite_rail_rotation_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            target = root / "target.json"
            mismatch_issue = {
                "severity": "BLOCK",
                "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                "message": "RANGE forecast only authorizes RANGE_ROTATION rail geometry",
            }
            range_block = {
                "severity": "BLOCK",
                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                "message": "RANGE forecast is still below the rail-rotation live floor",
            }
            failure_lane = _result(
                "DRY_RUN_BLOCKED",
                lane_id="failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
                pair="AUD_JPY",
                side="LONG",
                method="BREAKOUT_FAILURE",
                order_type="LIMIT",
                risk_issues=[mismatch_issue],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 300.0,
                    "reward_risk": 3.0,
                    "spread_pips": 1.0,
                },
            )
            failure_lane["intent"]["metadata"] = {
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.72,
                "chart_direction_bias": "SHORT",
                "range_phase": "RANGE_FORMING",
            }
            range_lane = _result(
                "DRY_RUN_BLOCKED",
                lane_id="range_trader:AUD_JPY:SHORT:RANGE_ROTATION",
                pair="AUD_JPY",
                side="SHORT",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                risk_issues=[range_block],
                risk_metrics={
                    "risk_jpy": 100.0,
                    "reward_jpy": 140.0,
                    "reward_risk": 1.4,
                    "spread_pips": 1.0,
                },
            )
            range_lane["intent"]["metadata"] = {
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.72,
                "chart_direction_bias": "SHORT",
                "range_phase": "RANGE_FORMING",
                "range_entry_side": "resistance",
            }
            intents.write_text(json.dumps({"results": [failure_lane, range_lane]}))
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
            top = payload["perspective_alignment_diagnostics"]["range_forecast_method_mismatch_top"][0]
            self.assertEqual(top["pair"], "AUD_JPY")
            self.assertEqual(top["direction"], "LONG")
            self.assertEqual(top["range_rotation_lanes"], 0)
            self.assertEqual(top["range_rotation_absence_reason"], "OPPOSITE_RAIL_SIDE_SURFACED")
            self.assertEqual(top["range_rotation_other_side_lanes"], 1)
            self.assertEqual(top["range_rotation_other_side_directions"][0]["code"], "SHORT")
            self.assertEqual(
                top["range_rotation_other_side_top_live_blocker_codes"][0]["code"],
                "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
            )
            self.assertTrue(
                any("opposite rail side SHORT surfaced" in item for item in payload["action_items"])
            )
            self.assertIn("other_rail_sides=`SHORT`", (root / "coverage.md").read_text())

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
    live_strategy_issues: list[dict] | None = None,
    risk_metrics: dict | None = None,
    include_risk_metrics: bool = True,
    method: str = "TREND_CONTINUATION",
) -> dict:
    payload = {
        "lane_id": lane_id,
        "status": status,
        "risk_allowed": True,
        "risk_issues": risk_issues or [],
        "strategy_issues": strategy_issues or [],
        "live_strategy_issues": live_strategy_issues or [],
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
                "method": method,
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
