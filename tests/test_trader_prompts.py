from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.cli import main
from quant_rabbit.trader_prompts import (
    BRANCH_ENTRY,
    BRANCH_LEARNING,
    BRANCH_POSITION,
    BRANCH_REFRESH,
    BRANCH_VERIFY,
    _fresh_close_recommendations,
    route_trader_prompts,
)


class TraderPromptRouteTest(unittest.TestCase):
    def test_routes_missing_artifacts_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            route = route_trader_prompts(
                snapshot_path=root / "missing_snapshot.json",
                target_state_path=root / "missing_target.json",
                intents_path=root / "missing_intents.json",
                decision_response_path=None,
            )

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertIn("10_precheck_refresh.md", _read_paths(route)[-1])

    def test_routes_flat_open_target_with_live_ready_lanes_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        read_paths = _read_paths(route)
        self.assertTrue(any(path.endswith("30_entry_decision.md") for path in read_paths))
        self.assertTrue(any(path.endswith("90_decision_receipt_schema.md") for path in read_paths))

    def test_persistent_profitability_p0_routes_to_learning_repair_before_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 27 consecutive audit run(s)",
                                "evidence": {
                                    "current_streak": 27,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.258,
                                        "expectancy_jpy": -534.167,
                                        "avg_win_jpy": 442.207,
                                        "avg_loss_jpy_abs": 1244.257,
                                        "worst_segments": [
                                            {
                                                "pair": "NZD_CAD",
                                                "side": "SHORT",
                                                "method": "RANGE_ROTATION",
                                                "trades": 2,
                                                "net_jpy": -2044.4543,
                                                "trade_ids": ["472312", "472380"],
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_LEARNING)
        self.assertTrue(any("self-improvement profitability P0 blocks entry routing" in reason for reason in route.reasons))
        self.assertTrue(any("streak=27" in reason for reason in route.reasons))
        self.assertTrue(any("data/execution_ledger.db" in reason for reason in route.reasons))
        self.assertTrue(any("pair=NZD_CAD" in reason for reason in route.reasons))
        self.assertTrue(any("trade_ids=472312,472380" in reason for reason in route.reasons))

    def test_projection_p0_routes_to_learning_repair_before_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "forecast",
                                "code": "PROJECTION_LEDGER_EXPIRED_PENDING",
                                "message": "projection ledger has 33 expired PENDING projection(s)",
                                "evidence": {"expired_pending": 33},
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_LEARNING)
        self.assertTrue(any("PROJECTION_LEDGER_EXPIRED_PENDING blocks entry routing" in reason for reason in route.reasons))
        self.assertTrue(any("count=33" in reason for reason in route.reasons))

    def test_coverage_market_evidence_refresh_routes_to_refresh_before_learning_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["intents"].write_text(
                json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "results": []})
            )
            files["coverage_optimization"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "COVERAGE_GAP",
                        "artifact_diagnostics": {
                            "requires_market_evidence_refresh": True,
                            "all_lanes_spread_blocked": True,
                            "quote_stale_result_count": 4,
                            "spread_normalized_candidate_count": 2,
                        },
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("coverage optimization requires fresh market evidence" in reason for reason in route.reasons))
        self.assertTrue(any("all_lanes_spread_blocked=true" in reason for reason in route.reasons))

    def test_persistent_stale_gpt_decision_p0_routes_to_fresh_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {
                                    "current_streak": 13,
                                    "snapshot_fetched_at_utc": "2026-06-11T09:45:47+00:00",
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        read_paths = _read_paths(route)
        self.assertTrue(any(path.endswith("30_entry_decision.md") for path in read_paths))
        self.assertTrue(any(path.endswith("90_decision_receipt_schema.md") for path in read_paths))
        self.assertTrue(any("decision-history P0 persists" in reason for reason in route.reasons))
        self.assertTrue(any("streak=13" in reason for reason in route.reasons))

    def test_persistent_stale_gpt_decision_p0_with_pending_entry_routes_to_entry_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472124",
                    "pair": "EUR_GBP",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": -1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {
                                    "current_streak": 13,
                                    "snapshot_fetched_at_utc": "2026-06-11T09:45:47+00:00",
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("pending entry order" in reason for reason in route.reasons))
        self.assertTrue(any("rewrite" in reason for reason in route.reasons))

    def test_single_stale_gpt_decision_p0_still_allows_fresh_entry_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 1},
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_profitability_p0_with_pending_entry_routes_to_position_cancel_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472155",
                    "pair": "EUR_CHF",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 5700,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 27 consecutive audit run(s)",
                                "evidence": {
                                    "current_streak": 27,
                                    "system_defect_evidence": {"profit_factor": 0.258},
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("trader pending entry order(s) occupy the gateway entry slot" in reason for reason in route.reasons))
        self.assertTrue(any("write CANCEL_PENDING" in reason for reason in route.reasons))

    def test_pending_cancel_review_p0_with_live_ready_routes_to_entry_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472533",
                    "pair": "CAD_CHF",
                    "order_type": "LIMIT",
                    "state": "PENDING",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry needs cancel review",
                                "evidence": {
                                    "cancel_review_order_ids": ["472533"],
                                    "orders": [
                                        {
                                            "order_id": "472533",
                                            "pair": "CAD_CHF",
                                            "side": "LONG",
                                            "method": "RANGE_ROTATION",
                                            "review_reasons": [
                                                {"code": "PENDING_CURRENT_CANDIDATE_MISSING"}
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("pending cancel review" in reason for reason in route.reasons))
        self.assertTrue(any("current LIVE_READY replacement" in reason for reason in route.reasons))
        self.assertTrue(any("TRADE with cancel_order_ids" in reason for reason in route.reasons))
        self.assertTrue(any("472533" in reason for reason in route.reasons))
        self.assertTrue(any("PENDING_CURRENT_CANDIDATE_MISSING" in reason for reason in route.reasons))

    def test_pending_cancel_review_p0_without_live_ready_routes_to_position_cancel_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472533",
                    "pair": "CAD_CHF",
                    "order_type": "LIMIT",
                    "state": "PENDING",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            intents = json.loads(files["intents"].read_text())
            for result in intents["results"]:
                result["status"] = "DRY_RUN_BLOCKED"
                result["live_blockers"] = ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]
            files["intents"].write_text(json.dumps(intents))
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry needs cancel review",
                                "evidence": {
                                    "cancel_review_order_ids": ["472533"],
                                    "orders": [
                                        {
                                            "order_id": "472533",
                                            "pair": "CAD_CHF",
                                            "side": "LONG",
                                            "method": "RANGE_ROTATION",
                                            "review_reasons": [
                                                {"code": "PENDING_CURRENT_CANDIDATE_MISSING"}
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("pending cancel review" in reason for reason in route.reasons))
        self.assertTrue(any("472533" in reason for reason in route.reasons))
        self.assertTrue(any("PENDING_CURRENT_CANDIDATE_MISSING" in reason for reason in route.reasons))

    def test_entry_branch_names_pending_entry_gateway_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472124",
                    "pair": "EUR_GBP",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": -1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(
            any(
                "pending entry order" in reason
                and "472124" in reason
                and "cancel_order_ids" in reason
                for reason in route.reasons
            )
        )

    def test_no_live_ready_with_pending_entry_routes_to_cancel_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["orders"] = [
                {
                    "order_id": "472124",
                    "pair": "EUR_GBP",
                    "order_type": "LIMIT",
                    "state": "PENDING",
                    "units": -1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            intents = json.loads(files["intents"].read_text())
            for result in intents["results"]:
                result["status"] = "DRY_RUN_PASSED"
                result["live_blockers"] = ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]
            files["intents"].write_text(json.dumps(intents))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        read_paths = _read_paths(route)
        self.assertTrue(any(path.endswith("30_entry_decision.md") for path in read_paths))
        self.assertTrue(any(path.endswith("90_decision_receipt_schema.md") for path in read_paths))
        self.assertTrue(any("trader pending entry order(s) occupy the gateway entry slot" in reason for reason in route.reasons))
        self.assertTrue(any("no current LIVE_READY lane" in reason for reason in route.reasons))
        self.assertTrue(any("CANCEL_PENDING" in reason for reason in route.reasons))

    def test_stale_trader_overrides_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            fetched_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            files["trader_overrides"].write_text(
                json.dumps(
                    {
                        "expires_at_utc": (fetched_at - timedelta(seconds=1)).isoformat(),
                        "bias_overrides": {"EUR_USD": {"LONG": -20.0}},
                        "blocked_lanes": [],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("daily-review feedback stale" in reason for reason in route.reasons))

    def test_missing_trader_overrides_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["trader_overrides"].unlink()

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("daily-review feedback missing" in reason for reason in route.reasons))

    def test_empty_strategy_profile_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["strategy_profile"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "history_db": str(files["history_db"]),
                        "profiles": [],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("strategy profile has zero mined profiles" in reason for reason in route.reasons))

    def test_stale_strategy_profile_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _set_mtime(files["strategy_profile"], 100.0)
            _set_mtime(files["history_db"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("strategy profile is older than history DB" in reason for reason in route.reasons))

    def test_stale_campaign_plan_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = (base + timedelta(minutes=5)).isoformat()
            files["target"].write_text(json.dumps(target))

            campaign = json.loads(files["campaign_plan"].read_text())
            campaign["generated_at_utc"] = base.isoformat()
            files["campaign_plan"].write_text(json.dumps(campaign))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("campaign plan stale" in reason for reason in route.reasons))

    def test_campaign_plan_target_mismatch_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            campaign = json.loads(files["campaign_plan"].read_text())
            campaign["start_balance_jpy"] = 200000.0
            files["campaign_plan"].write_text(json.dumps(campaign))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("campaign plan target mismatch" in reason for reason in route.reasons))

    def test_stale_order_intents_against_market_context_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = base.isoformat()
            intents["results"] = []
            files["intents"].write_text(json.dumps(intents))
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (base + timedelta(minutes=30)).isoformat(),
                        "pairs": {
                            "EUR_USD": {
                                "LONG": {"context_refs": ["fixture:matrix"]},
                                "SHORT": {"context_refs": ["fixture:matrix"]},
                            }
                        },
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("order intents stale against market context" in reason for reason in route.reasons))
        self.assertFalse(any("no current LIVE_READY lane" in reason for reason in route.reasons))

    def test_stale_order_intents_against_forecast_history_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = base.isoformat()
            files["intents"].write_text(json.dumps(intents))
            files["forecast_history"].write_text(
                json.dumps(
                    {
                        "timestamp_utc": (base + timedelta(minutes=30)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.79,
                    }
                )
                + "\n"
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("order intents stale against forecast history" in reason for reason in route.reasons))
        self.assertFalse(any("no current LIVE_READY lane" in reason for reason in route.reasons))

    def test_stale_attack_advice_against_order_intents_routes_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = (base + timedelta(minutes=10)).isoformat()
            files["intents"].write_text(json.dumps(intents))
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": base.isoformat(),
                        "status": "NO_ATTACK_ADVICE",
                        "live_ready_lanes": 0,
                        "recommended_now_lane_ids": [],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("ai_attack_advice stale against order_intents" in reason for reason in route.reasons))

    def test_missing_memory_health_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["memory_health"].unlink()

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("memory_health" in reason for reason in route.reasons))

    def test_blocked_memory_health_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["memory_health"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "MEMORY_HEALTH_BLOCKED",
                        "blockers": ["strategy_profile has zero mined profiles"],
                        "issues": [
                            {
                                "severity": "BLOCK",
                                "layer": "long_term",
                                "code": "LONG_STRATEGY_PROFILE_EMPTY",
                                "message": "strategy_profile has zero mined profiles",
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("memory health audit is blocked" in reason for reason in route.reasons))
        self.assertTrue(any("LONG_STRATEGY_PROFILE_EMPTY" in reason for reason in route.reasons))

    def test_learning_quarantine_p1_does_not_block_clean_live_ready_entry_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            audit = json.loads(files["self_improvement_audit"].read_text())
            audit.update(
                {
                    "status": "SELF_IMPROVEMENT_ACTION_REQUIRED",
                    "p0_findings": 0,
                    "p1_findings": 1,
                    "p2_findings": 0,
                    "findings": [
                        {
                            "priority": "P1",
                            "layer": "learning",
                            "code": "LEARNING_AUDIT_INFLUENCED_LANES_QUARANTINED",
                            "message": (
                                "learning_audit blocks only risk-increasing learning influence; "
                                "non-learning live-ready lanes can still be routed"
                            ),
                        }
                    ],
                }
            )
            files["self_improvement_audit"].write_text(json.dumps(audit))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("current LIVE_READY lane" in reason for reason in route.reasons))
        self.assertFalse(any("self-improvement P0 blocks" in reason for reason in route.reasons))

    def test_stale_memory_health_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            snapshot_ts = base.isoformat()
            memory_ts = (base + timedelta(minutes=1)).isoformat()
            intents_ts = (base + timedelta(minutes=2)).isoformat()
            target_ts = (base + timedelta(minutes=3)).isoformat()

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts
            files["snapshot"].write_text(json.dumps(snapshot))

            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = target_ts
            files["target"].write_text(json.dumps(target))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory_health"].read_text())
            memory["generated_at_utc"] = memory_ts
            files["memory_health"].write_text(json.dumps(memory))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("memory health audit stale" in reason for reason in route.reasons))
        self.assertTrue(any("order intents" in reason for reason in route.reasons))
        self.assertFalse(any("daily target state" in reason for reason in route.reasons))

    def test_memory_health_audited_snapshot_time_keeps_entry_route_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            snapshot_ts = (base + timedelta(minutes=2)).isoformat()
            intents_ts = (base + timedelta(minutes=1)).isoformat()

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts
            files["snapshot"].write_text(json.dumps(snapshot))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory_health"].read_text())
            memory["generated_at_utc"] = base.isoformat()
            memory["metrics"] = {
                "runtime": {
                    "snapshot_fetched_at_utc": snapshot_ts,
                    "order_intents_generated_at_utc": intents_ts,
                }
            }
            files["memory_health"].write_text(json.dumps(memory))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertFalse(any("memory health audit stale" in reason for reason in route.reasons))

    def test_stale_self_improvement_audit_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            snapshot_ts = (base + timedelta(minutes=3)).isoformat()
            intents_ts = (base + timedelta(minutes=2)).isoformat()
            audit_ts = (base + timedelta(minutes=1)).isoformat()
            memory_ts = (base + timedelta(minutes=4)).isoformat()

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts
            files["snapshot"].write_text(json.dumps(snapshot))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory_health"].read_text())
            memory["generated_at_utc"] = memory_ts
            files["memory_health"].write_text(json.dumps(memory))

            audit = json.loads(files["self_improvement_audit"].read_text())
            audit["generated_at_utc"] = audit_ts
            files["self_improvement_audit"].write_text(json.dumps(audit))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("self-improvement audit stale" in reason for reason in route.reasons))
        self.assertTrue(any("broker snapshot" in reason for reason in route.reasons))

    def test_current_accepted_gateway_decision_is_not_preempted_by_stale_memory_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            snapshot_ts = (base + timedelta(minutes=2)).isoformat()
            memory_ts = base.isoformat()

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts
            files["snapshot"].write_text(json.dumps(snapshot))

            memory = json.loads(files["memory_health"].read_text())
            memory["generated_at_utc"] = memory_ts
            files["memory_health"].write_text(json.dumps(memory))

            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CANCEL_PENDING", "cancel_order_ids": ["472124"]}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CANCEL_PENDING",
                            "cancel_order_ids": ["472124"],
                        },
                    }
                )
            )
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)
            _set_mtime(files["memory_health"], 100.0)
            _set_mtime(decision_response, 101.0)
            _set_mtime(files["gpt_decision"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("accepted CANCEL_PENDING decision has no newer gateway receipt", route.reasons[0])

    def test_daily_target_refresh_after_memory_health_does_not_force_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            snapshot_ts = base.isoformat()
            intents_ts = (base + timedelta(minutes=1)).isoformat()
            memory_ts = (base + timedelta(minutes=2)).isoformat()
            target_ts = (base + timedelta(minutes=5)).isoformat()

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts
            snapshot["positions"] = []
            snapshot["orders"] = []
            files["snapshot"].write_text(json.dumps(snapshot))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory_health"].read_text())
            memory["generated_at_utc"] = memory_ts
            files["memory_health"].write_text(json.dumps(memory))

            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = target_ts
            target["status"] = "PURSUE_TARGET"
            target["remaining_target_jpy"] = 1000.0
            files["target"].write_text(json.dumps(target))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertNotEqual(route.branch, BRANCH_REFRESH)
        self.assertFalse(any("memory health audit stale" in reason for reason in route.reasons))

    def test_stale_trader_overrides_does_not_preempt_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            fetched_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            files["trader_overrides"].write_text(
                json.dumps({"expires_at_utc": (fetched_at - timedelta(seconds=1)).isoformat()})
            )

            prior = os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is not None:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("needs protection repair", route.reasons[0])

    def test_routes_unprotected_trader_position_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is not None:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("needs protection repair", route.reasons[0])

    def test_sl_free_trader_tp_only_position_does_not_force_position_branch(self) -> None:
        # Under QR_TRADER_DISABLE_SL_REPAIR=1 the operator directive 「SLいらない」
        # makes trader-owned SL=None intentional. Routing must let the operator
        # reach BRANCH_ENTRY for fresh entries instead of being trapped in
        # BRANCH_POSITION repair mode every cycle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "470395",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.17026,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_sl_free_profitable_position_with_tp_rebalance_routes_to_position_management(self) -> None:
        """A WAIT-eligible SL-free position still routes to protection when
        the deterministic TP rebalancer has a live adjustment."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471292",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": -22000,
                        "entry_price": 1.16077,
                        "take_profit": 1.15640,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": 3000.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["EUR_USD"] = {
                "bid": 1.15937,
                "ask": 1.15947,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            files["snapshot"].write_text(json.dumps(snapshot))
            files["pair_charts"].write_text(
                json.dumps(
                    {
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "confluence": {
                                    "h4_atr_pips": 19.2,
                                    "price_percentile_7d": 0.0,
                                    "tf_agreement_score": 0.33,
                                    "range_24h_sigma_multiple": 6.7,
                                },
                                "views": [
                                    {
                                        "granularity": "M15",
                                        "indicators": {
                                            "atr_pips": 19.2,
                                            "stoch_rsi": 0.13,
                                            "williams_r_14": -85.0,
                                            "close": 1.15970,
                                            "bb_lower": 1.15972,
                                        },
                                        "structure": {
                                            "liquidity": [
                                                {
                                                    "side": "EQ_LOW",
                                                    "price": 1.15875,
                                                    "indices": [1, 2, 3, 4],
                                                }
                                            ]
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            (root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "pair": "EUR_USD",
                        "direction": "UNCLEAR",
                        "confidence": 0.24,
                        "horizon_min": 0,
                    }
                )
                + "\n"
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("TP rebalance required" in reason for reason in route.reasons))
        self.assertTrue(any("forecast_harvest" in reason for reason in route.reasons))

    def test_sl_free_trader_no_broker_tp_runner_does_not_force_position_branch(self) -> None:
        # Missing broker TP is preserved as a no-broker-TP runner under the
        # SL-free runtime unless explicit TP repair is enabled. It must not
        # trap the trader in BRANCH_POSITION and starve fresh entries.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_soft_forecast_persistence_close_with_live_ready_routes_to_entry(self) -> None:
        """Soft-only close evidence must not starve current all-horizon entries.

        Regression: a protected TP-managed position with only forecast
        persistence RECOMMEND_CLOSE used to trap routing in position_management
        and then fail Gate B, even while fresh LIVE_READY lanes existed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "last 3 forecasts flipped to DOWN",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("soft close review advisory" in reason for reason in route.reasons))
        self.assertTrue(any("forecast_persistence RECOMMEND_CLOSE" in reason for reason in route.reasons))

    def test_thesis_expired_without_hold_support_routes_to_position_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "BROKEN",
                                "verdict": "RECOMMEND_CLOSE",
                                "rationale": "THESIS_EXPIRED: age 7.0h exceeds declared horizon 6.0h",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("loss-cut review required" in reason for reason in route.reasons))
        self.assertTrue(any("thesis_evolution RECOMMEND_CLOSE" in reason for reason in route.reasons))

    def test_thesis_expired_with_hold_support_routes_to_entry_as_soft_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "BROKEN",
                                "verdict": "RECOMMEND_CLOSE",
                                "rationale": "THESIS_EXPIRED: age 3.1h exceeds declared horizon 3.0h",
                            }
                        ],
                    }
                )
            )
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "aggregate_score": 62.43,
                                "rationale_lines": ["position thesis still supports LONG carry"],
                                "context_notes": [],
                            }
                        ],
                    }
                )
            )
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open LONG",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("soft close review advisory" in reason for reason in route.reasons))
        self.assertTrue(any("thesis_evolution RECOMMEND_CLOSE" in reason for reason in route.reasons))

    def test_position_thesis_invalidation_with_hold_support_routes_to_entry_as_soft_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472445",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 12.82,
                                "rationale_lines": ["chart-tech -17.2"],
                                "context_notes": [
                                    "invalidation hit: current bid 1.16900 <= buffered invalidation 1.16930",
                                    "technical invalidation confirmed against LONG: H1 MACD-; M15 ST-; M30 MACD-; M5 ST-",
                                ],
                            }
                        ],
                    }
                )
            )
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "STILL_VALID",
                                "verdict": "HOLD",
                                "rationale": (
                                    "current forecast UP conf=0.44 supports LONG, so the invalidation "
                                    "hit is HOLD/reprice/TP rebalance evidence"
                                ),
                            }
                        ],
                    }
                )
            )
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "last 10 forecasts aligned UP (position LONG) — extend TP, hold",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("soft close review advisory" in reason for reason in route.reasons))
        self.assertTrue(any("position_thesis REVIEW_CLOSE" in reason for reason in route.reasons))
        self.assertFalse(any("loss-cut review required" in reason for reason in route.reasons))

    def test_soft_position_thesis_review_routes_to_position_when_no_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472445",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            intents = json.loads(files["intents"].read_text())
            intents["results"] = [
                {
                    "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "status": "DRY_RUN_PASSED",
                    "live_blockers": ["EUR_USD LONG forecast UP confidence 0.44 < 0.65"],
                }
            ]
            files["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 12.82,
                                "rationale_lines": ["chart-tech -17.2"],
                                "context_notes": [
                                    "invalidation hit: current bid 1.16900 <= buffered invalidation 1.16930",
                                    "technical invalidation confirmed against LONG: H1 MACD-; M15 ST-",
                                ],
                            }
                        ],
                    }
                )
            )
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "STILL_VALID",
                                "verdict": "HOLD",
                                "rationale": "current forecast UP still supports LONG",
                            }
                        ],
                    }
                )
            )
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open LONG",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("soft close review advisory" in reason for reason in route.reasons))
        self.assertTrue(any("position_thesis REVIEW_CLOSE" in reason for reason in route.reasons))
        self.assertTrue(any("no current LIVE_READY lane" in reason for reason in route.reasons))
        self.assertTrue(any("active close/hold ambiguity" in reason for reason in route.reasons))

    def test_position_thesis_h4_structural_break_still_blocks_with_hold_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472445",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": -20.0,
                                "rationale_lines": ["H4 close-confirmed structural break: BOS_DOWN"],
                                "context_notes": [
                                    "technical invalidation confirmed against LONG: H4 BOS_DOWN close-confirmed; H1 MACD-",
                                ],
                            }
                        ],
                    }
                )
            )
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "STILL_VALID",
                                "verdict": "HOLD",
                                "rationale": "current forecast UP still supports LONG",
                            }
                        ],
                    }
                )
            )
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open LONG",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("loss-cut review required" in reason for reason in route.reasons))

    def test_soft_close_review_suppresses_tp_rebalance_probe_blocker(self) -> None:
        """Router TP probe must match tp-rebalance CLI close-review guard.

        A fresh soft close-review is advisory for new entries, but the TP
        probe used to ignore the same close-review trade ids that the CLI
        honors, creating a phantom TP_REBALANCE_REQUIRED route blocker.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 5000,
                        "entry_price": 1.1700,
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_time = datetime.now(timezone.utc) - timedelta(seconds=30)
            snapshot["fetched_at_utc"] = snapshot_time.isoformat()
            files["snapshot"].write_text(json.dumps(snapshot))
            files["pair_charts"].write_text(json.dumps({"charts": [{"pair": "EUR_USD"}]}))
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": (snapshot_time + timedelta(seconds=1)).isoformat(),
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "soft forecast persistence review",
                            }
                        ],
                    }
                )
            )

            calls: list[dict] = []

            def fake_compute_all_tp_adjustments(**kwargs):
                calls.append(kwargs)
                if "555" in (kwargs.get("close_review_trade_ids") or set()):
                    return []
                return [
                    SimpleNamespace(
                        pair="EUR_USD",
                        side="LONG",
                        trade_id="555",
                        current_tp=1.18,
                        new_tp=1.19,
                        rationale="synthetic TP expansion",
                    )
                ]

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                with patch(
                    "quant_rabbit.strategy.tp_rebalancer.compute_all_tp_adjustments",
                    side_effect=fake_compute_all_tp_adjustments,
                ):
                    route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(calls)
        self.assertIn("555", calls[-1]["close_review_trade_ids"])
        self.assertTrue(any("soft close review advisory" in reason for reason in route.reasons))
        self.assertFalse(any("TP rebalance required" in reason for reason in route.reasons))

    def test_soft_position_thesis_review_does_not_preempt_entry_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471739",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -57.5,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471739",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 21.32,
                                "rationale_lines": ["synthetic detector support"],
                                "context_notes": [
                                    "adverse technical loss: entry thesis lacks invalidation_price",
                                    "technical invalidation confirmed against SHORT",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("position_thesis REVIEW_CLOSE" in reason for reason in route.reasons))
        self.assertTrue(any("adverse technical loss" in reason for reason in route.reasons))

    def test_position_thesis_adverse_technical_loss_only_is_not_standing_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471414",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -2109.8,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471414",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 47.0,
                                "rationale_lines": [
                                    "patterns +30.0",
                                    "forward-proj +25.0",
                                    "chart-tech -8.0",
                                ],
                                "context_notes": [
                                    "adverse technical loss: no entry thesis; current ask 1.16310 >= entry-buffer 1.15891",
                                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; M15 BOS_UP; M30 MACD+; M5 ST+",
                                ],
                            }
                        ],
                    }
                )
            )

            recs = _fresh_close_recommendations(snapshot, data_root=root)

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_thesis")
        self.assertFalse(recs[0]["gate_b_standing_authorized"])
        self.assertIn("adverse technical loss", recs[0]["reason"])
        self.assertIn("technical invalidation confirmed", recs[0]["reason"])

    def test_position_thesis_invalidation_hit_is_standing_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471414",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -2109.8,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471414",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 47.0,
                                "rationale_lines": [
                                    "patterns +30.0",
                                    "forward-proj +25.0",
                                    "chart-tech -8.0",
                                ],
                                "context_notes": [
                                    "invalidation hit: current ask 1.16310 >= invalidation price 1.16290 plus anti-wick buffer",
                                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; M15 trend up; M30 MACD+; M5 ST+",
                                ],
                            }
                        ],
                    }
                )
            )

            recs = _fresh_close_recommendations(snapshot, data_root=root)

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_thesis")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])
        self.assertIn("invalidation hit", recs[0]["reason"])
        self.assertIn("technical invalidation confirmed", recs[0]["reason"])

    def test_position_thesis_score_only_review_is_not_standing_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471414",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471414",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 12.0,
                                "rationale_lines": ["score weakened but no invalidation hit"],
                                "context_notes": [],
                            }
                        ],
                    }
                )
            )

            recs = _fresh_close_recommendations(snapshot, data_root=root)

        self.assertEqual(len(recs), 1)
        self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_position_management_review_exit_carryforward_routes_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471817",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1900.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=20)
            ).isoformat()
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "action": "REVIEW_EXIT",
                        "positions": [
                            {
                                "trade_id": "471817",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "REVIEW_EXIT",
                                "reasons": [
                                    "score context before structural review",
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-1900 JPY)",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_management REVIEW_EXIT" in reason for reason in route.reasons))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_management")
        self.assertEqual(recs[0]["evidence_ref"], "position:management:471817")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])
        self.assertIn("loss-cut: structural OB broken", recs[0]["reason"])

    def test_demoted_position_management_review_exit_carryforward_routes_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472445",
                        "pair": "EUR_CHF",
                        "side": "LONG",
                        "take_profit": 0.92317,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1019.7829,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=20)
            ).isoformat()
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_CHF",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                                "close_review_action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@0.92000, H1@0.92100) (-1019 JPY)",
                                    "QR_DISABLE_AUTO_CLOSE=1 -> REVIEW_EXIT demoted to HOLD_PROTECTED",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_management REVIEW_EXIT" in reason for reason in route.reasons))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_management")
        self.assertEqual(recs[0]["evidence_ref"], "position:management:472445")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])
        self.assertIn("loss-cut: structural OB broken", recs[0]["reason"])

    def test_position_guardian_review_exit_carryforward_routes_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472445",
                        "pair": "EUR_CHF",
                        "side": "LONG",
                        "take_profit": 0.92317,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1019.7829,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=1)
            ).isoformat()
            (root / "position_guardian_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "472445",
                                "pair": "EUR_CHF",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                                "close_review_action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@0.92000, H1@0.92100) (-1019 JPY)",
                                    "QR_DISABLE_AUTO_CLOSE=1 -> REVIEW_EXIT demoted to HOLD_PROTECTED",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_guardian_management REVIEW_EXIT" in reason for reason in route.reasons))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_guardian_management")
        self.assertEqual(recs[0]["evidence_ref"], "position:guardian_management:472445")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])

    def test_stale_position_management_review_exit_routes_to_sidecar_refresh_not_close_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471817",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1900.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=10)
            ).isoformat()
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "positions": [
                            {
                                "trade_id": "471817",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-1900 JPY)",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                with patch("quant_rabbit.trader_prompts.POSITION_MANAGEMENT_REVIEW_EXIT_TTL_SECONDS", 60.0):
                    route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                    recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertEqual(recs, ())
        self.assertTrue(any("position_management sidecar stale" in reason for reason in route.reasons))
        self.assertFalse(any("position_management REVIEW_EXIT" in reason for reason in route.reasons))

    def test_position_management_source_snapshot_staleness_takes_precedence_over_generated_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471817",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1900.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            current_snapshot_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": (current_snapshot_at + timedelta(seconds=5)).isoformat(),
                        "snapshot_fetched_at_utc": (current_snapshot_at - timedelta(minutes=5)).isoformat(),
                        "positions": [
                            {
                                "trade_id": "471817",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                                "reasons": ["old broker truth"],
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("broker snapshot at" in reason for reason in route.reasons))
        self.assertTrue(any("position_management sidecar stale" in reason for reason in route.reasons))

    def test_entry_thesis_blocker_routes_without_close_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471910",
                        "pair": "AUD_USD",
                        "side": "SHORT",
                        "take_profit": 0.7164,
                        "stop_loss": 0.7195,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "evolutions": [
                            {
                                "trade_id": "471910",
                                "pair": "AUD_USD",
                                "side": "SHORT",
                                "status": "UNVERIFIABLE",
                                "verdict": "REQUIRE_THESIS_REPAIR",
                                "rationale": "missing entry_thesis_ledger row",
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("entry-thesis repair required" in reason for reason in route.reasons))
        self.assertFalse(any("loss-cut review required" in reason for reason in route.reasons))

    def test_stale_forecast_persistence_recommend_close_does_not_route_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(seconds=1)
            ).isoformat()
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "old forecast flip from a prior snapshot",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertFalse(any("loss-cut review required" in reason for reason in route.reasons))

    def test_fresh_close_recommendation_for_closed_trade_does_not_route_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "closed-444",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "report belongs to a trade no longer open",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertFalse(any("closed-444" in reason for reason in route.reasons))

    def test_sl_free_missing_tp_repair_opt_in_routes_to_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

        self.assertEqual(route.branch, BRANCH_POSITION)

    def test_manual_position_missing_tp_routes_to_tp_only_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                        "unrealized_pl_jpy": 120.0,
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("TP-only profit management", route.reasons[0])

    def test_underwater_manual_position_missing_tp_does_not_force_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-998",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                        "unrealized_pl_jpy": -2625.0,
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_manual_position_with_tp_missing_sl_does_not_force_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-1000",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "manual",
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_routes_existing_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "WAIT"}))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertTrue(any(path.endswith("40_verify_execute.md") for path in _read_paths(route)))

    def test_current_close_decision_response_preempts_position_sidecar_refresh(self) -> None:
        # Once a current CLOSE receipt exists, the next step is verifier ->
        # gateway. Re-running position sidecars in between can make the receipt
        # stale and leave broken positions blocking entries.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "9001",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.17,
                        "take_profit": 1.172,
                        "stop_loss": 1.168,
                        "unrealized_pl_jpy": -120.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = (base + timedelta(minutes=1)).isoformat()
            files["snapshot"].write_text(json.dumps(snapshot))
            files["position_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": base.isoformat(),
                        "snapshot_fetched_at_utc": base.isoformat(),
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "9001",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                            }
                        ],
                    }
                )
            )
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CLOSE", "close_trade_ids": ["9001"]}))
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)
            _set_mtime(files["position_management"], 100.0)
            _set_mtime(decision_response, 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("unconsumed decision response exists", route.reasons[0])

    def test_stale_decision_response_reason_is_preserved_in_position_branch(self) -> None:
        # Regression for the live -74,162 JPY close loop: a stale CLOSE receipt
        # must remain visible even when active exposure correctly routes to
        # position_management first.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "9001",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.17,
                        "take_profit": 1.172,
                        "stop_loss": 1.168,
                        "unrealized_pl_jpy": -120.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = (base + timedelta(minutes=1)).isoformat()
            files["snapshot"].write_text(json.dumps(snapshot))
            files["position_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": base.isoformat(),
                        "snapshot_fetched_at_utc": base.isoformat(),
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "9001",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                            }
                        ],
                    }
                )
            )
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CLOSE", "close_trade_ids": ["9001"]}))
            _set_mtime(decision_response, 99.0)
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)
            _set_mtime(files["position_management"], 100.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("predates refreshed broker snapshot" in reason for reason in route.reasons))
        self.assertTrue(any("position_management sidecar stale" in reason for reason in route.reasons))

    def test_position_branch_preserves_pending_entry_risk_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "9001",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.17,
                        "take_profit": 1.172,
                        "stop_loss": 1.168,
                        "unrealized_pl_jpy": 30.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = (base + timedelta(minutes=1)).isoformat()
            snapshot["orders"] = [
                {
                    "order_id": "472533",
                    "pair": "CAD_CHF",
                    "order_type": "LIMIT",
                    "state": "PENDING",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": None,
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["position_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": base.isoformat(),
                        "snapshot_fetched_at_utc": base.isoformat(),
                        "action": "HOLD_PROTECTED",
                        "positions": [{"trade_id": "9001", "action": "HOLD_PROTECTED"}],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_management sidecar stale" in reason for reason in route.reasons))
        self.assertTrue(any("trader pending entry order(s) occupy the gateway entry slot" in reason for reason in route.reasons))
        self.assertTrue(any("472533" in reason for reason in route.reasons))

    def test_routes_rejected_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [{"code": "UNKNOWN_LANE", "severity": "BLOCK"}],
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already verified as REJECTED", route.reasons[0])

    def test_routes_consumed_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["live_order"].write_text(json.dumps({"status": "SENT", "sent": True}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["live_order"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already consumed by live order gateway receipt", route.reasons[0])

    def test_routes_accepted_trade_without_newer_gateway_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["autotrade_report"].write_text("# stale no-send cycle\n")
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["autotrade_report"], 101.0)
            _set_mtime(files["gpt_decision"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("accepted TRADE decision has no newer gateway receipt", route.reasons[0])

    def test_routes_accepted_cancel_pending_without_newer_gateway_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CANCEL_PENDING", "cancel_order_ids": ["472124"]}))
            files["autotrade_report"].write_text("# stale no-send cycle\n")
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CANCEL_PENDING",
                            "cancel_order_ids": ["472124"],
                        },
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["autotrade_report"], 101.0)
            _set_mtime(files["gpt_decision"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("accepted CANCEL_PENDING decision has no newer gateway receipt", route.reasons[0])

    def test_routes_accepted_trade_with_newer_gateway_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                    }
                )
            )
            files["live_order"].write_text(json.dumps({"status": "SENT", "sent": True}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)
            _set_mtime(files["live_order"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("accepted TRADE decision already consumed by live order gateway receipt", route.reasons[0])

    def test_routes_accepted_cancel_pending_with_newer_gateway_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CANCEL_PENDING", "cancel_order_ids": ["472124"]}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CANCEL_PENDING",
                            "cancel_order_ids": ["472124"],
                        },
                    }
                )
            )
            files["live_order"].write_text(json.dumps({"status": "GPT_CANCEL_PENDING", "sent": False}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)
            _set_mtime(files["live_order"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn(
            "accepted CANCEL_PENDING decision already consumed by live order gateway receipt",
            route.reasons[0],
        )

    def test_routes_accepted_position_action_without_newer_gateway_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "9001",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.17,
                        "take_profit": 1.172,
                        "stop_loss": 1.168,
                        "unrealized_pl_jpy": 120.0,
                    }
                ],
            )
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CLOSE", "close_trade_ids": ["9001"]}))
            files["autotrade_report"].write_text("# stale no-send cycle\n")
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CLOSE",
                            "close_trade_ids": ["9001"],
                        },
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["autotrade_report"], 101.0)
            _set_mtime(files["gpt_decision"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("accepted CLOSE decision has no newer gateway receipt", route.reasons[0])

    def test_consumed_position_action_names_gateway_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "9001",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.3979,
                        "take_profit": 1.4001,
                        "stop_loss": 1.3920,
                        "unrealized_pl_jpy": 100.0,
                    }
                ],
            )
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CLOSE", "close_trade_ids": ["9001"]}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["9001"]},
                    }
                )
            )
            files["position_execution"].write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "sent": False,
                        "actions": [
                            {
                                "trade_id": "9001",
                                "pair": "USD_CAD",
                                "issues": [
                                    {
                                        "severity": "BLOCK",
                                        "code": "POSITION_CLOSE_SPREAD_TOO_WIDE",
                                        "message": "USD_CAD market CLOSE spread 1.9pip exceeds cap",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)
            _set_mtime(files["position_execution"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertTrue(any("accepted CLOSE decision already consumed" in reason for reason in route.reasons))
        self.assertTrue(any("POSITION_CLOSE_SPREAD_TOO_WIDE" in reason for reason in route.reasons))

    def test_routes_decision_older_than_refreshed_broker_truth_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 101.0)
            _set_mtime(files["intents"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("predates refreshed broker snapshot", route.reasons[0])

    def test_routes_decision_with_stale_payload_timestamp_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(
                json.dumps(
                    {
                        "action": "WAIT",
                        "generated_at_utc": (snapshot_ts - timedelta(minutes=1)).isoformat(),
                    }
                )
            )
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)
            _set_mtime(decision_response, 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertTrue(any("generated_at_utc" in reason for reason in route.reasons))
        self.assertTrue(any("predates broker snapshot" in reason for reason in route.reasons))

    def test_stale_position_action_names_previous_gateway_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "CLOSE", "close_trade_ids": ["9001"]}))
            files["position_execution"].write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "sent": False,
                        "actions": [
                            {
                                "trade_id": "9001",
                                "pair": "USD_CAD",
                                "issues": [
                                    {
                                        "severity": "BLOCK",
                                        "code": "POSITION_CLOSE_SPREAD_TOO_WIDE",
                                        "message": "USD_CAD market CLOSE spread 1.9pip exceeds cap",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["position_execution"], 102.0)
            _set_mtime(files["snapshot"], 103.0)
            _set_mtime(files["intents"], 99.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertIn("predates refreshed broker snapshot", route.reasons[0])
        self.assertTrue(any("POSITION_CLOSE_SPREAD_TOO_WIDE" in reason for reason in route.reasons))

    def test_routes_same_timestamp_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)

    def test_cli_prints_prompt_route_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-prompt-route",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--target-state",
                        str(files["target"]),
                        "--intents",
                        str(files["intents"]),
                        "--pair-charts",
                        str(files["pair_charts"]),
                        "--cross-asset",
                        str(files["cross_asset"]),
                        "--flow",
                        str(files["flow"]),
                        "--currency-strength",
                        str(files["currency_strength"]),
                        "--levels",
                        str(files["levels"]),
                        "--market-context-matrix",
                        str(files["market_context_matrix"]),
                        "--calendar",
                        str(files["calendar"]),
                        "--cot",
                        str(files["cot"]),
                        "--option-skew",
                        str(files["option_skew"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--learning-audit",
                        str(files["learning_audit"]),
                        "--campaign-plan",
                        str(files["campaign_plan"]),
                        "--memory-health",
                        str(files["memory_health"]),
                        "--self-improvement-audit",
                        str(files["self_improvement_audit"]),
                        "--strategy-profile",
                        str(files["strategy_profile"]),
                        "--trader-overrides",
                        str(files["trader_overrides"]),
                        "--decision-response",
                        str(root / "missing_decision_response.json"),
                        "--live-order",
                        str(files["live_order"]),
                        "--position-execution",
                        str(files["position_execution"]),
                        "--autotrade-report",
                        str(files["autotrade_report"]),
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["branch"], BRANCH_ENTRY)
        self.assertTrue(any(item["path"].endswith("30_entry_decision.md") for item in payload["read_order"]))


def _route_paths(files: dict[str, Path]) -> dict[str, Path]:
    return {
        "snapshot_path": files["snapshot"],
        "target_state_path": files["target"],
        "intents_path": files["intents"],
        "pair_charts_path": files["pair_charts"],
        "cross_asset_path": files["cross_asset"],
        "flow_path": files["flow"],
        "currency_strength_path": files["currency_strength"],
        "levels_path": files["levels"],
        "market_context_matrix_path": files["market_context_matrix"],
        "calendar_path": files["calendar"],
        "cot_path": files["cot"],
        "option_skew_path": files["option_skew"],
        "attack_advice_path": files["attack_advice"],
        "learning_audit_path": files["learning_audit"],
        "forecast_history_path": files["forecast_history"],
        "campaign_plan_path": files["campaign_plan"],
        "memory_health_path": files["memory_health"],
        "self_improvement_audit_path": files["self_improvement_audit"],
        "coverage_optimization_path": files["coverage_optimization"],
        "strategy_profile_path": files["strategy_profile"],
        "trader_overrides_path": files["trader_overrides"],
        "gpt_decision_path": files["gpt_decision"],
        "live_order_path": files["live_order"],
        "position_management_path": files["position_management"],
        "position_execution_path": files["position_execution"],
        "autotrade_report_path": files["autotrade_report"],
    }


def _read_paths(route) -> list[str]:
    return [str(doc.path) for doc in route.read_order]


def _fixtures(root: Path, *, positions: list[dict] | None = None) -> dict[str, Path]:
    files = {
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "pair_charts": root / "pair_charts.json",
        "cross_asset": root / "cross_asset_snapshot.json",
        "flow": root / "flow_snapshot.json",
        "currency_strength": root / "currency_strength.json",
        "levels": root / "levels_snapshot.json",
        "market_context_matrix": root / "market_context_matrix.json",
        "calendar": root / "economic_calendar.json",
        "cot": root / "cot_snapshot.json",
        "option_skew": root / "option_skew_snapshot.json",
        "attack_advice": root / "ai_attack_advice.json",
        "learning_audit": root / "learning_audit.json",
        "forecast_history": root / "forecast_history.jsonl",
        "campaign_plan": root / "daily_campaign_plan.json",
        "memory_health": root / "memory_health.json",
        "self_improvement_audit": root / "self_improvement_audit.json",
        "coverage_optimization": root / "coverage_optimization.json",
        "strategy_profile": root / "strategy_profile.json",
        "history_db": root / "legacy_history.db",
        "trader_overrides": root / "trader_overrides.json",
        "gpt_decision": root / "gpt_trader_decision.json",
        "live_order": root / "live_order_request.json",
        "position_management": root / "position_management.json",
        "position_execution": root / "position_execution.json",
        "autotrade_report": root / "autotrade_cycle_report.md",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": positions or [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701, "timestamp_utc": now}},
            }
        )
    )
    files["target"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "PURSUE_TARGET",
                "start_balance_jpy": 10000.0,
                "target_jpy": 1000.0,
                "remaining_target_jpy": 1000.0,
                "progress_jpy": 0.0,
            }
        )
    )
    files["campaign_plan"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "start_balance_jpy": 10000.0,
                "target_jpy": 1000.0,
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                    }
                ],
            }
        )
    )
    files["intents"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
    )
    for key in (
        "pair_charts",
        "cross_asset",
        "flow",
        "currency_strength",
        "levels",
        "cot",
        "option_skew",
        "attack_advice",
        "learning_audit",
    ):
        files[key].write_text(json.dumps({}))
    files["forecast_history"].write_text("")
    files["market_context_matrix"].write_text(
        json.dumps(
            {
                "pairs": {
                    "EUR_USD": {
                        "LONG": {"context_refs": ["fixture:matrix"]},
                        "SHORT": {"context_refs": ["fixture:matrix"]},
                    }
                },
            }
        )
    )
    files["calendar"].write_text(json.dumps({}))
    files["memory_health"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "MEMORY_HEALTH_PASS",
                "layers": {
                    "short_term": "PASS",
                    "medium_term": "PASS",
                    "long_term": "PASS",
                    "position_memory": "PASS",
                },
                "issues": [],
                "blockers": [],
                "warnings": [],
            }
        )
    )
    files["self_improvement_audit"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "SELF_IMPROVEMENT_OK",
                "p0_findings": 0,
                "p1_findings": 0,
                "p2_findings": 0,
                "findings": [],
            }
        )
    )
    files["coverage_optimization"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "LIVE_READY_COVERAGE_READY",
                "artifact_diagnostics": {"requires_market_evidence_refresh": False},
                "action_items": [],
            }
        )
    )
    files["history_db"].write_text("not a real sqlite db; only mtime matters for routing tests")
    files["strategy_profile"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "history_db": str(files["history_db"]),
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "status": "CANDIDATE",
                        "required_fix": "fixture",
                    }
                ],
            }
        )
    )
    files["trader_overrides"].write_text(
        json.dumps(
            {
                "expires_at_utc": (
                    datetime.fromisoformat(now) + timedelta(hours=1)
                ).isoformat(),
                "bias_overrides": {},
                "blocked_lanes": [],
            }
        )
    )
    files["position_management"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "action": "HOLD_PROTECTED",
                "positions": [
                    {
                        "trade_id": str(position.get("trade_id") or ""),
                        "pair": position.get("pair"),
                        "side": position.get("side"),
                        "action": "HOLD_PROTECTED",
                    }
                    for position in (positions or [])
                    if str(position.get("owner") or "") == "trader"
                ],
            }
        )
    )
    return files


def _set_mtime(path: Path, value: float) -> None:
    os.utime(path, (value, value))


if __name__ == "__main__":
    unittest.main()
